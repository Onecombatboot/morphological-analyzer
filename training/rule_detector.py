# -*- coding: utf-8 -*-
"""Deterministic detector for N3 capability conflicts.

RATIONALE
---------
A capability conflict has a mechanical signature: one value REQUIRES something, and
the other value's PROVIDES contains an explicit negation of that same thing.

    artisanal small boat  PROVIDES: no monitoring equipment
    satellite monitoring  REQUIRES: a transponder fitted and functioning

57% of the corpus's 1229 exclusions are coded N3, and 19% of all provides-clauses are
negations, so a large share of the exclusions should be reachable this way with no
model at all. What this buys over a fine-tune: it is deterministic, it cites the two
clauses it fired on, and its failure modes are inspectable rather than statistical.

It cannot see N7 (implausible), which is a judgement about the world rather than a
clash of stated clauses. That is the honest division of labour: rules for what the
text states, a model for what the text implies.

METHOD
------
Clause-level matching. Split requires/provides on commas, mark provides-clauses that
begin with a negation, strip the negation and stopwords, and fire when the remaining
content words overlap a requirement's content words above a threshold. Overlap is
measured with a Dice coefficient over stemmed content words, so "no arrest power"
matches "arrest authority and a court" on {arrest} without needing an exact match.
"""
import argparse, json, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))

NEG = re.compile(r'^(no|not|never|without|unable to|cannot|nothing)\b', re.I)
STOP = set('a an the of to and or for in on at by with from into within own its it is are '
           'be been being that this these those any all some each other more most only '
           'very such as than then so if but per via up out over under between both'.split())


def clauses(text):
    return [c.strip() for c in re.split(r'[,;]| and (?=no\b)', text or '') if c.strip()]


def stem(w):
    for suf in ('ations', 'ation', 'ities', 'ity', 'ing', 'ers', 'er', 'ies', 'es', 's'):
        if len(w) > 4 and w.endswith(suf):
            return w[:-len(suf)]
    return w


def content(text):
    ws = re.findall(r"[a-z]+", (text or '').lower())
    return {stem(w) for w in ws if w not in STOP and len(w) > 2}


def dice(a, b):
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def negated_provisions(prov):
    """Provision clauses that assert the ABSENCE of something, with the thing itself."""
    out = []
    for c in clauses(prov):
        m = NEG.match(c)
        if m:
            rest = c[m.end():].strip()
            if rest:
                out.append((c, content(rest)))
    return out


def detect(char_a, char_b, threshold, wide=False):
    """Return (asserted_clause, blocking_clause, score) for the strongest conflict, or None.

    Checked in both directions: A may assert what B denies, or the reverse.

    narrow (default) matches negations only against the other side's REQUIRES. High
    precision, low recall -- the clean "it needs X, the other cannot supply X" case.

    wide also matches against the other side's PROVIDES, catching provides-vs-provides
    clashes such as 'no ash column' against 'aircraft at risk from ash'. Roughly
    +60% recall for -25% precision, so it is the right default only when the output
    is a ranked list for a human rather than an automatic exclusion.
    """
    best = None
    fields = ('requires', 'provides') if wide else ('requires',)
    for assert_side, deny_side in ((char_a, char_b), (char_b, char_a)):
        for f in fields:
            for rc in clauses(assert_side.get(f, '')):
                if NEG.match(rc):
                    continue          # a negation cannot itself be the thing denied
                rset = content(rc)
                for pc, pset in negated_provisions(deny_side.get('provides', '')):
                    s = dice(rset, pset)
                    if s >= threshold and (best is None or s > best[2]):
                        best = (rc, pc, s)
    return best


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def evaluate(rows, threshold, wide=False):
    tp = fp = fn = tn = 0
    hits = []
    for r in rows:
        d = detect(r['char_a'], r['char_b'], threshold, wide)
        pred = 'N' if d else 'Y'
        t = r['label']
        if pred == 'N' and t == 'N':
            tp += 1; hits.append((r, d, True))
        elif pred == 'N':
            fp += 1; hits.append((r, d, False))
        elif t == 'N':
            fn += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    rY = tn / (tn + fp) if (tn + fp) else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec,
                f1=2 * prec * rec / (prec + rec) if (prec + rec) else 0.0,
                balanced_accuracy=(rec + rY) / 2,
                accuracy=(tp + tn) / len(rows)), hits


def load_cells(corpus_path, domains=None):
    """Rebuild every cell with both characterisations attached."""
    corpus = json.load(open(corpus_path, encoding='utf-8'))
    out = []
    for m in corpus:
        if domains is not None and m['domain'] not in domains:
            continue
        ch = m['characterisations']
        for key, grid in m['verdicts'].items():
            pa, pb = key.split('|')
            ra, cb = m['parameters'][pa], m['parameters'][pb]
            reasons = m.get('verdict_reasons', {}).get(key, {})
            for i, a in enumerate(ra):
                for j, b in enumerate(cb):
                    r = reasons.get('%d,%d' % (i + 1, j + 1))
                    out.append({'domain': m['domain'], 'a': a, 'b': b,
                                'char_a': ch[a], 'char_b': ch[b],
                                'label': grid[i][j],
                                'code': r['code'] if r else None})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', default=os.path.join(HERE, 'matrices_v2', 'corpus.json'))
    ap.add_argument('--val', default=os.path.join(HERE, 'dataset_val_cell.jsonl'))
    ap.add_argument('--show', type=int, default=6)
    ap.add_argument('--wide', action='store_true', help='also match against provides')
    ap.add_argument('--target-precision', type=float, default=0.0,
                    help='if set, choose the training threshold with the highest recall '
                         'whose precision clears this, instead of maximising balanced accuracy')
    a = ap.parse_args()

    val_doms = {json.loads(l)['domain'] for l in open(a.val, encoding='utf-8')}
    train_cells = load_cells(a.corpus, domains=None)
    train_cells = [c for c in train_cells if c['domain'] not in val_doms]
    val_cells = load_cells(a.corpus, domains=val_doms)
    print('  threshold tuned on %d training-domain cells, reported on %d validation cells'
          % (len(train_cells), len(val_cells)))

    best_t, best_key = 0.3, -1
    print('\n  threshold sweep on TRAINING domains  (mode: %s)'
          % ('wide' if a.wide else 'narrow'))
    print('    thr    prec    rec     bal     fires')
    for t in [x / 100 for x in range(10, 75, 5)]:
        m, _ = evaluate(train_cells, t, a.wide)
        print('    %.2f   %.3f   %.3f   %.3f   %d' % (t, m['precision'], m['recall'],
                                                      m['balanced_accuracy'], m['tp'] + m['fp']))
        # with a precision target, take the highest-recall threshold that clears it;
        # otherwise maximise balanced accuracy
        if a.target_precision:
            key = m['recall'] if (m['precision'] >= a.target_precision and m['tp'] >= 5) else -1
        else:
            key = m['balanced_accuracy']
        if key > best_key:
            best_key, best_t = key, t

    m, hits = evaluate(val_cells, best_t, a.wide)
    nN = m['tp'] + m['fn']
    base = nN / len(val_cells)
    pl, ph = wilson(m['tp'], max(m['tp'] + m['fp'], 1))
    rN, rY = m['recall'], m['tn'] / (m['tn'] + m['fp'])
    se = 0.5 * math.sqrt(rN * (1 - rN) / nN + rY * (1 - rY) / (m['tn'] + m['fp']))
    print('\n' + '=' * 70)
    print('  RULE DETECTOR on held-out domains   (threshold %.2f)' % best_t)
    print('=' * 70)
    print('  %d cells, %d true N (%.1f%%)' % (len(val_cells), nN, 100 * base))
    print('  precision          %.3f   95%% CI [%.3f, %.3f]   base rate %.3f'
          % (m['precision'], pl, ph, base))
    print('  recall             %.3f' % m['recall'])
    print('  balanced accuracy  %.3f   95%% CI [%.3f, %.3f]'
          % (m['balanced_accuracy'], m['balanced_accuracy'] - 1.96 * se,
             m['balanced_accuracy'] + 1.96 * se))
    print('  F1                 %.3f      accuracy %.3f' % (m['f1'], m['accuracy']))
    print('  confusion          TP %d  FP %d  FN %d  TN %d' % (m['tp'], m['fp'], m['fn'], m['tn']))

    by_code = {}
    for c in val_cells:
        if c['label'] == 'N':
            by_code.setdefault(c['code'] or '?', [0, 0])[1] += 1
    for r, d, ok in hits:
        if ok:
            by_code.setdefault(r['code'] or '?', [0, 0])[0] += 1
    print('\n  recall by exclusion type (what the rule can and cannot see):')
    for code in sorted(by_code):
        got, tot = by_code[code]
        print('    %-4s %3d/%-3d  %5.1f%%' % (code, got, tot, 100 * got / tot if tot else 0))

    print('\n  sample firings:')
    for r, d, ok in hits[:a.show]:
        print('    [%s] %s x %s' % ('correct' if ok else 'WRONG ', r['a'][:32], r['b'][:32]))
        print('           requires %r' % d[0][:58])
        print('           blocked  %r   (overlap %.2f)' % (d[1][:58], d[2]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
