# -*- coding: utf-8 -*-
"""Builds the fine-tuning dataset from the hand-written matrices.

Validates first, then emits one training example per parameter pair. Validation matters more than
it looks: a verdict string of the wrong length, or a value that appears in a grid but has no
characterisation, would train the model on a silently misaligned grid. Every such case is a hard
error here rather than a quiet corruption downstream.

Outputs:
  dataset_train.jsonl   ~85% of matrices, split by DOMAIN not by example
  dataset_val.jsonl     the rest
  REVIEW.md             human-readable, with the derivation notes, for spot-checking
  STATS.md              verdict density per matrix and overall
"""
import json, os, glob, sys, random, re, argparse, itertools

HERE = os.path.dirname(os.path.abspath(__file__))

# The v2 corpus, not the v1 matrices/ directory. Every N in v2 carries a typed reason
# code and verbatim citations that build_matrices.py re-verifies, so the labels here
# cannot drift from the evidence behind them. Training against v1 would reproduce the
# ground truth that scored at chance.
MATRIX_DIR = os.path.join(HERE, 'matrices_v2')

SYSTEM = (
    "You are a defence and security analyst carrying out cross-consistency assessment for General "
    "Morphological Analysis. Two lists describe two facets of ONE single scenario. First state what "
    "each option requires and provides. Then, for every combination, decide whether one real "
    "scenario could have both facets at once."
)

# --------------------------------------------------------------------------------------
# Why there are three formats
#
# 'grid' is what the first two runs trained on, and it has a defect that probably explains
# their chance-level results. The input carries only the VALUE NAMES; the model has to
# invent every requires/provides itself and only then emit verdicts. Under teacher forcing
# it learns P(verdict | OUR characterisation), but at generation time on an unseen domain it
# conditions on ITS OWN invented characterisation. That is exposure bias, and it is severe
# here because ~250 tokens of self-generated premises precede the first verdict token. It
# fits the evidence exactly: training loss to 0.015 (our text memorised) with no transfer.
#
# 'grid-given' puts the characterisations in the INPUT. The premises are then fixed and
# shared, and the model is asked the question we actually care about. Same output shape as
# 'grid', so the two are directly comparable.
#
# 'cell' goes further: one example per cell, and the target is the typed reason from
# CRITERION.md v2 rather than a bare letter. That turns 1229 justified exclusions into
# explicit relational supervision and yields ~16x more training examples, each with the
# loss landing exactly on one decision.
# --------------------------------------------------------------------------------------

SYSTEM_CELL = (
    "You are a defence and security analyst carrying out cross-consistency assessment for General "
    "Morphological Analysis. You are given two options from different facets of ONE scenario, with "
    "what each requires and provides. Decide whether a single real scenario could contain both. "
    "Answer N only when you can name the clause that makes it impossible or means it never occurs; "
    "ineffective or unwise is still Y."
)

SWAP = [False]   # set from --swap; module-level so examples_for can see it
SIBLINGS = [False]   # set from --siblings, same reason
SHUFFLE = [0]        # set from --shuffle-clauses

CODE_NAMES = {'N1': 'definitional', 'N2': 'physical', 'N3': 'capability', 'N4': 'temporal',
              'N5': 'authority', 'N6': 'scope', 'N7': 'implausible'}


def load_matrices():
    out = []
    for f in sorted(glob.glob(os.path.join(MATRIX_DIR, '*.json'))):
        with open(f, encoding='utf-8') as fh:
            data = json.load(fh)
        for m in (data if isinstance(data, list) else [data]):
            m['_file'] = os.path.basename(f)
            out.append(m)
    return out


def validate(m):
    """Every failure here would otherwise become a silently misaligned training example."""
    errs = []
    params = m['parameters']
    chars = m['characterisations']

    for p, vals in params.items():
        if len(vals) != len(set(vals)):
            errs.append('%s: duplicate values' % p)
        for v in vals:
            if v not in chars:
                errs.append('%s: value "%s" has no characterisation' % (p, v))
            elif not chars[v].get('requires') or not chars[v].get('provides'):
                errs.append('"%s": characterisation missing requires or provides' % v)

    names = list(params)
    expected = set()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            expected.add(names[i] + '|' + names[j])

    got = set(m['verdicts'])
    for k in expected - got:
        errs.append('missing verdict grid: %s' % k)
    for k in got - expected:
        errs.append('unexpected verdict grid: %s' % k)

    for key, rows in m['verdicts'].items():
        if key not in expected:
            continue
        pa, pb = key.split('|')
        if len(rows) != len(params[pa]):
            errs.append('%s: %d rows, expected %d' % (key, len(rows), len(params[pa])))
        for idx, r in enumerate(rows):
            if len(r) != len(params[pb]):
                errs.append('%s row %d: %d verdicts, expected %d'
                            % (key, idx + 1, len(r), len(params[pb])))
            bad = set(r) - set('YN')
            if bad:
                errs.append('%s row %d: illegal characters %s' % (key, idx + 1, sorted(bad)))
    return errs


def _shuffle_clauses(text, rng):
    """Permute a comma-separated clause list.

    requires/provides are unordered lists -- "police powers, arrest authority" says
    the same thing as "arrest authority, police powers". The model currently sees
    each characterisation in exactly one order, so anything it learns about position
    is noise, and the validation domains are written by the same hand in the same
    style, which hides the problem until a new analyst phrases things differently.

    Permuting is the one augmentation that is safe here: every N target quotes its
    clauses VERBATIM and build_matrices.py re-verifies those quotes, so a paraphrase
    would break its own citation. Reordering leaves each clause byte-identical.
    """
    parts = [p.strip() for p in re.split(r',\s*', text or '') if p.strip()]
    if len(parts) < 2:
        return text
    rng.shuffle(parts)
    return ', '.join(parts)


def _char_block(m, values, rng=None):
    c = m['characterisations']
    out = []
    for v in values:
        req, prov = c[v]['requires'], c[v]['provides']
        if rng is not None:
            req, prov = _shuffle_clauses(req, rng), _shuffle_clauses(prov, rng)
        out.append('%s | REQUIRES: %s | PROVIDES: %s' % (v, req, prov))
    return out


def render_input(m, pa, pb, fmt):
    lines = ['Dimension 1 — %s:' % pa]
    for i, v in enumerate(m['parameters'][pa]):
        lines.append('  %d. %s' % (i + 1, v))
    lines.append('')
    lines.append('Dimension 2 — %s:' % pb)
    for i, v in enumerate(m['parameters'][pb]):
        lines.append('  %s. %s' % (chr(ord('A') + i), v))
    if fmt == 'grid-given':
        lines.append('')
        lines.append('CHARACTERISATIONS')
        lines += _char_block(m, m['parameters'][pa] + m['parameters'][pb])
    return '\n'.join(lines)


def render_output(m, pa, pb, fmt):
    lines = []
    if fmt == 'grid':
        # the model must supply the premises itself -- see the note above
        lines.append('CHARACTERISATIONS')
        lines += _char_block(m, m['parameters'][pa] + m['parameters'][pb])
        lines.append('')
    lines.append('VERDICTS')
    for i, row in enumerate(m['verdicts'][pa + '|' + pb]):
        lines.append('%d:%s' % (i + 1, row))
    return '\n'.join(lines)


def cell_examples(m, pa, pb, swap=False, rng=None):
    """One example per cell, with the typed reason as the target for every N.

    swap=True presents facet B first. Cross-consistency is symmetric -- consistent(A,B)
    is the same claim as consistent(B,A) -- but the model only ever sees row-then-column,
    so nothing teaches it that invariance, and any order-sensitivity it acquires is pure
    noise. Mixing both presentations removes it, at no extra training cost if the pool is
    subsampled back to its original size.
    """
    out = []
    ra, cb = m['parameters'][pa], m['parameters'][pb]
    grid = m['verdicts'][pa + '|' + pb]
    reasons = m.get('verdict_reasons', {}).get(pa + '|' + pb, {})
    for i, row in enumerate(ra):
        for j, col in enumerate(cb):
            v = grid[i][j]
            fa, fv, sa, sv = (pb, col, pa, row) if swap else (pa, row, pb, col)
            if SIBLINGS[0]:
                # A value's meaning is partly contrastive: what an option IS depends
                # on what the alternatives were. The cell format has always hidden
                # that contrast set. grid-given exposed it, but paired it with
                # whole-grid generation, so the two effects were never separated.
                fsib = [x for x in m['parameters'][fa] if x != fv]
                ssib = [x for x in m['parameters'][sa] if x != sv]
                user = '\n'.join(
                    ['Facet A — %s   (other options: %s)' % (fa, '; '.join(fsib))] +
                    _char_block(m, [fv], rng) +
                    ['', 'Facet B — %s   (other options: %s)' % (sa, '; '.join(ssib))] +
                    _char_block(m, [sv], rng) +
                    ['', 'Can one scenario contain both?'])
            else:
                user = '\n'.join(
                    ['Facet A — %s' % fa] + _char_block(m, [fv], rng) +
                    ['', 'Facet B — %s' % sa] + _char_block(m, [sv], rng) +
                    ['', 'Can one scenario contain both?'])
            r = reasons.get('%d,%d' % (i + 1, j + 1))
            if v == 'N' and r and rng is not None:
                # A cited quote can span two clauses, and permuting then separates them,
                # so the target would quote text no longer present in the prompt. Verify
                # and fall back to the corpus order for that cell -- measured at ~4% of
                # N cells, and training on an unfindable citation is worse than losing
                # the augmentation on those few.
                if r['requires'] not in user or r['blocked_by'] not in user:
                    if SIBLINGS[0]:
                        user = '\n'.join(
                            ['Facet A — %s   (other options: %s)' % (fa, '; '.join(fsib))] +
                            _char_block(m, [fv]) +
                            ['', 'Facet B — %s   (other options: %s)' % (sa, '; '.join(ssib))] +
                            _char_block(m, [sv]) +
                            ['', 'Can one scenario contain both?'])
                    else:
                        user = '\n'.join(
                            ['Facet A — %s' % fa] + _char_block(m, [fv]) +
                            ['', 'Facet B — %s' % sa] + _char_block(m, [sv]) +
                            ['', 'Can one scenario contain both?'])
            if v == 'N' and r:
                ans = ('VERDICT: N\nCODE: %s (%s)\nREQUIRES: %s\nBLOCKED BY: %s'
                       % (r['code'], CODE_NAMES.get(r['code'], '?'), r['requires'], r['blocked_by']))
            elif v == 'N':
                ans = 'VERDICT: N'
            else:
                ans = 'VERDICT: Y'
            out.append({'domain': m['domain'], 'pair': '%s x %s' % (row, col), 'label': v,
                        'order': 'swapped' if swap else 'normal',
                        'messages': [{'role': 'system', 'content': SYSTEM_CELL},
                                     {'role': 'user', 'content': user},
                                     {'role': 'assistant', 'content': ans}]})
    return out


def examples_for(m, fmt):
    names = list(m['parameters'])
    out = []
    for pa, pb in itertools.combinations(names, 2):
        if fmt == 'cell':
            # SHUFFLE[0] extra copies per cell, each with the clause lists permuted.
            # Variant 0 is always the corpus order, so the unaugmented data is a
            # subset of the augmented data and nothing is lost.
            for k in range(1 + SHUFFLE[0]):
                rng = random.Random(hash((m['domain'], pa, pb, k)) & 0xffffffff) if k else None
                out += cell_examples(m, pa, pb, rng=rng)
                if SWAP[0]:
                    out += cell_examples(m, pa, pb, swap=True, rng=rng)
        else:
            out.append({
                'domain': m['domain'],
                'pair': pa + ' x ' + pb,
                'messages': [
                    {'role': 'system', 'content': SYSTEM},
                    {'role': 'user', 'content': render_input(m, pa, pb, fmt)},
                    {'role': 'assistant', 'content': render_output(m, pa, pb, fmt)},
                ]
            })
    return out


def density(m):
    n = bad = 0
    for rows in m['verdicts'].values():
        for r in rows:
            n += len(r)
            bad += r.count('N')
    return bad, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--format', choices=('grid', 'grid-given', 'cell'), default='grid-given',
                    help="'grid' is the original (model invents its own premises -- see the note "
                         "at the top of this file); 'grid-given' fixes the premises by putting "
                         "them in the input; 'cell' is one example per cell with the typed reason")
    ap.add_argument('--balance', type=float, default=0.0,
                    help='cell format only: oversample N examples until they are this share of '
                         'the training set (e.g. 0.40). 0 disables. Y is ~82%% of cells, which '
                         'makes "always consistent" the safe bet for a token-level loss.')
    ap.add_argument('--swap', action='store_true',
                    help='cell format: also emit the mirrored presentation (facet B first). '
                         'Combine with --cap to keep the training cost unchanged.')
    ap.add_argument('--cap', type=int, default=0,
                    help='subsample the training set to this many examples after balancing')
    ap.add_argument('--suffix', default='',
                    help='appended to the output filenames, so formats do not overwrite one another')
    ap.add_argument('--shuffle-clauses', type=int, default=0, metavar='N',
                    help='emit N extra copies of every cell with the requires/provides '
                         'clause lists permuted. Order in those lists is arbitrary, so '
                         'anything the model learns from it is noise.')
    ap.add_argument('--siblings', action='store_true',
                    help="show each facet's other options beside the pair, so a value "
                         "is read against its contrast set")
    ap.add_argument('--exclude-codes', default='',
                    help="comma-separated exclusion codes to relabel as consistent, "
                         "e.g. 'N7'. Narrows the criterion; does not delete cells.")
    ap.add_argument('--val-from', default='',
                    help='reuse the validation domains of an existing dataset jsonl, so '
                         'that adding domains to the corpus does not silently change the '
                         'held-out set and invalidate comparison with earlier runs')
    a = ap.parse_args()
    SWAP[0] = a.swap
    SIBLINGS[0] = a.siblings
    SHUFFLE[0] = a.shuffle_clauses

    ms = load_matrices()
    if not ms:
        print('no matrices found in', MATRIX_DIR)
        return 1

    if a.exclude_codes:
        # Narrow the CRITERION, not the evaluation. Cells carrying one of these codes
        # become consistent, because the criterion now excludes only on a clause that
        # makes a pairing impossible. Dropping them from the evaluation instead would
        # just be deleting the cases the system gets wrong.
        drop = {c.strip().upper() for c in a.exclude_codes.split(',') if c.strip()}
        flipped = 0
        for m in ms:
            for key, grid in m['verdicts'].items():
                reasons = m.get('verdict_reasons', {}).get(key, {})
                rows = [list(r) for r in grid]
                for rc, r in list(reasons.items()):
                    if r.get('code') not in drop:
                        continue
                    ri, ci = (int(x) - 1 for x in rc.split(','))
                    if 0 <= ri < len(rows) and 0 <= ci < len(rows[ri]):
                        rows[ri][ci] = 'Y'
                        reasons.pop(rc)
                        flipped += 1
                m['verdicts'][key] = [''.join(r) for r in rows]
        print('  criterion narrowed: %d cells with code(s) %s relabelled consistent'
              % (flipped, ','.join(sorted(drop))))
    print('format: %s' % a.format)

    print('loaded %d matrices' % len(ms))
    failed = False
    for m in ms:
        errs = validate(m)
        if errs:
            failed = True
            print('\nINVALID: %s  (%s)' % (m['domain'], m['_file']))
            for e in errs:
                print('   -', e)
    if failed:
        print('\nfix the above before building')
        return 1
    print('all matrices valid')

    # split by domain, never by example -- splitting by example would leak a domain's
    # characterisations into validation and make the score meaningless
    random.seed(42)
    doms = sorted({m['domain'] for m in ms})
    if a.val_from:
        # Pin the validation domains to a previous split. random.shuffle reorders a
        # list of 92 domains completely differently from one of 70, so simply adding
        # domains to the corpus swaps the held-out set for a new one and every
        # published figure becomes incomparable. Pinning keeps the SAME held-out
        # cells, so a change in score measures the extra training data and nothing
        # else. New domains all land in train.
        prev = {json.loads(l)['domain']
                for l in open(a.val_from, encoding='utf-8') if l.strip()}
        missing = prev - set(doms)
        if missing:
            raise SystemExit('  --val-from names domains not in the corpus: %s'
                             % ', '.join(sorted(missing)))
        val_doms = prev
        print('  validation pinned to %d domains from %s'
              % (len(val_doms), os.path.basename(a.val_from)))
    else:
        random.shuffle(doms)
        n_val = max(1, round(len(doms) * 0.15))
        val_doms = set(doms[:n_val])

    train, val = [], []
    for m in ms:
        (val if m['domain'] in val_doms else train).extend(examples_for(m, a.format))

    # Oversample the minority class. Only meaningful for 'cell', where one example carries
    # exactly one verdict; in the grid formats a single example mixes both classes, so
    # duplicating it changes nothing about the balance.
    if a.balance and a.format == 'cell':
        neg = [r for r in train if r['label'] == 'N']
        pos = [r for r in train if r['label'] != 'N']
        if neg:
            want = a.balance * len(pos) / (1 - a.balance)
            reps = max(1, int(round(want / len(neg))))
            train = pos + neg * reps
            random.shuffle(train)
            share = len(neg) * reps / len(train)
            print('  balanced: N x%d  ->  %d examples, N share %.1f%%'
                  % (reps, len(train), 100 * share))
    elif a.balance:
        print('  --balance ignored: it only applies to the cell format')

    if a.cap and len(train) > a.cap:
        random.shuffle(train)
        train = train[:a.cap]
        print('  capped training set to %d examples' % len(train))

    sfx = a.suffix or ('' if a.format == 'grid-given' else '_' + a.format.replace('-', ''))
    for name, rows in [('dataset_train%s.jsonl' % sfx, train), ('dataset_val%s.jsonl' % sfx, val)]:
        with open(os.path.join(HERE, name), 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print('wrote %-22s %4d examples' % (name, len(rows)))

    # ---- stats
    tot_bad = tot_all = 0
    lines = ['# Verdict density', '',
             'Density is a diagnostic, not a target. It is what applying the rule',
             'honestly produces: constrained domains land high, loose ones land low.', '',
             '| domain | group | cells | inconsistent | density |', '|---|---|---|---|---|']
    for m in sorted(ms, key=lambda x: x['domain']):
        b, n = density(m)
        tot_bad += b
        tot_all += n
        lines.append('| %s | %s | %d | %d | %.0f%% |'
                     % (m['domain'], m.get('group', '-'), n, b, 100.0 * b / n))
    lines += ['', '**Overall: %d of %d cells inconsistent = %.1f%%**'
              % (tot_bad, tot_all, 100.0 * tot_bad / tot_all), '',
              'Matrices: %d   Examples: %d train + %d val   Validation domains: %s'
              % (len(ms), len(train), len(val), ', '.join(sorted(val_doms)))]
    with open(os.path.join(HERE, 'STATS.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    # ---- review file
    rev = ['# Derivations for spot-checking', '',
           'The `notes` never reach the model. They exist so each grid can be checked by hand.', '']
    for m in sorted(ms, key=lambda x: x['domain']):
        b, n = density(m)
        rev.append('## %s  *(%s, %.0f%% inconsistent)*' % (m['domain'], m.get('group', '-'), 100.0 * b / n))
        rev.append('')
        for key in sorted(m['verdicts']):
            note = m.get('notes', {}).get(key)
            if note:
                rev.append('**%s** — %s' % (key.replace('|', ' x '), note))
                rev.append('')
    with open(os.path.join(HERE, 'REVIEW.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(rev) + '\n')

    print('\noverall density: %.1f%% inconsistent  (diagnostic, not a target)' % (100.0 * tot_bad / tot_all))
    print('validation domains:', ', '.join(sorted(val_doms)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
