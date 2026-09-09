# -*- coding: utf-8 -*-
"""Cross-consistency assessment for a morphological box.

This is the deliverable. Given a morphological box -- parameters, values, and a
requires/provides characterisation for each value -- it produces the cross-consistency
matrix, the citation behind every exclusion, and the resulting solution space.

    python cca.py --matrix authoring/src/001-maritime-interdiction.json
    python cca.py --matrix mybox.json --with-model adapter-best --report out.md

DESIGN
------
Two assessors, deliberately separated because they fail differently:

  rules   deterministic clause matching (rule_detector). Fires only when one value
          asserts something the other explicitly denies, and names both clauses.
          Measured on 10 held-out domains: precision 0.513, recall 0.173.

  model   the fine-tuned adapter, scored by P(N) on a single token so a threshold can
          be set. Measured on the same cells: precision 0.264, recall 0.173. Weaker,
          but it can reach implausibility (N7) that no clause match can see.

Every exclusion is reported with its source and its evidence, so an analyst can audit
the matrix rather than trust it. That is the point: at these accuracies the honest
product is an assistant that proposes and explains, not an oracle that decides.

SOLUTION SPACE
--------------
The value of CCA is the reduction it produces. With k parameters the raw space is the
product of the value counts; a configuration survives only if every one of its k(k-1)/2
pairs is consistent. Both numbers are reported, because the reduction is the result an
analyst actually uses.
"""
import argparse, itertools, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rule_detector import detect            # noqa: E402


def load_box(path):
    data = json.load(open(path, encoding='utf-8'))
    if isinstance(data, list):
        data = data[0]
    for k in ('parameters', 'characterisations'):
        if k not in data:
            raise SystemExit('box is missing "%s"' % k)
    missing = [v for vs in data['parameters'].values() for v in vs
               if v not in data['characterisations']]
    if missing:
        raise SystemExit('values without a characterisation: %s' % ', '.join(missing[:5]))
    return data


def assess(box, threshold, wide, model_ctx=None, model_thr=0.5):
    """Return {('PA|PB'): [[cell, ...], ...]} where each cell is a dict."""
    chars = box['characterisations']
    params = box['parameters']
    out = {}
    for pa, pb in itertools.combinations(list(params), 2):
        grid = []
        for a in params[pa]:
            row = []
            for b in params[pb]:
                hit = detect(chars[a], chars[b], threshold, wide)
                cell = {'a': a, 'b': b, 'verdict': 'Y', 'source': None, 'evidence': None}
                if hit:
                    cell.update(verdict='N', source='rule',
                                evidence={'asserts': hit[0], 'denied_by': hit[1],
                                          'overlap': round(hit[2], 3)})
                elif model_ctx is not None:
                    p = model_ctx['score_one'](a, chars[a], b, chars[b], pa, pb)
                    if p >= model_thr:
                        cell.update(verdict='N', source='model',
                                    evidence={'p_inconsistent': round(p, 3)})
                row.append(cell)
            grid.append(row)
        out['%s|%s' % (pa, pb)] = grid
    return out


def solution_space(box, verdicts):
    """Enumerate configurations, pruning on the first inconsistent pair.

    Depth-first with pruning rather than generate-and-filter: the raw space is
    exponential in the number of parameters, and pruning at the shallowest possible
    level is what makes a real box tractable.
    """
    names = list(box['parameters'])
    vals = [box['parameters'][n] for n in names]
    raw = 1
    for v in vals:
        raw *= len(v)

    lookup = {}
    for key, grid in verdicts.items():
        pa, pb = key.split('|')
        for i, a in enumerate(box['parameters'][pa]):
            for j, b in enumerate(box['parameters'][pb]):
                lookup[(a, b)] = grid[i][j]['verdict']

    def ok(chosen, cand):
        return all(lookup.get((c, cand), lookup.get((cand, c), 'Y')) == 'Y' for c in chosen)

    survivors = [0]

    def walk(depth, chosen):
        if depth == len(names):
            survivors[0] += 1
            return
        for v in vals[depth]:
            if ok(chosen, v):
                walk(depth + 1, chosen + [v])

    walk(0, [])
    return raw, survivors[0]


def render(box, verdicts, raw, kept):
    L = ['# Cross-consistency assessment', '', '**Domain:** %s' % box.get('domain', '(unnamed)'), '']
    L += ['## Morphological box', '']
    for p, vs in box['parameters'].items():
        L.append('- **%s** — %s' % (p, '; '.join(vs)))
    L += ['', '## Solution space', '',
          '| | configurations |', '|---|---|',
          '| raw (all combinations) | %d |' % raw,
          '| internally consistent | %d |' % kept,
          '| eliminated | %d (%.1f%%) |' % (raw - kept, 100.0 * (raw - kept) / raw if raw else 0),
          '']
    n_n = sum(1 for g in verdicts.values() for r in g for c in r if c['verdict'] == 'N')
    n_c = sum(len(r) for g in verdicts.values() for r in g)
    L += ['## Consistency matrices', '',
          '%d of %d pairs excluded (%.1f%%).' % (n_n, n_c, 100.0 * n_n / n_c if n_c else 0), '']
    for key, grid in verdicts.items():
        pa, pb = key.split('|')
        L += ['### %s × %s' % (pa, pb), '',
              '| | %s |' % ' | '.join(box['parameters'][pb]),
              '|---|%s' % ('---|' * len(box['parameters'][pb]))]
        for i, a in enumerate(box['parameters'][pa]):
            L.append('| **%s** | %s |' % (a, ' | '.join(c['verdict'] for c in grid[i])))
        L.append('')
    L += ['## Exclusions and their evidence', '']
    any_n = False
    for key, grid in verdicts.items():
        for row in grid:
            for c in row:
                if c['verdict'] != 'N':
                    continue
                any_n = True
                if c['source'] == 'rule':
                    L.append('- **%s** × **%s** — asserts *"%s"*, denied by *"%s"* '
                             '(overlap %.2f, deterministic)'
                             % (c['a'], c['b'], c['evidence']['asserts'],
                                c['evidence']['denied_by'], c['evidence']['overlap']))
                else:
                    L.append('- **%s** × **%s** — model, P(inconsistent) = %.2f '
                             '*(statistical, verify before relying on it)*'
                             % (c['a'], c['b'], c['evidence']['p_inconsistent']))
    if not any_n:
        L.append('*No exclusions found.*')
    return '\n'.join(L) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--matrix', required=True)
    ap.add_argument('--threshold', type=float, default=0.10)
    ap.add_argument('--narrow', action='store_true',
                    help='high-precision mode: only requires-vs-denial matches')
    ap.add_argument('--with-model', default=None, metavar='ADAPTER[,ADAPTER...]',
                    help='consult the fine-tuned adapter(s) for pairs the rules miss. '
                         'Comma-separate several to ensemble them: averaging P(N) over the '
                         'epoch-1 and epoch-2 checkpoints raised balanced accuracy 0.639 -> '
                         '0.667 and cut solution-space error from +15.6%% to +2.6%%.')
    ap.add_argument('--model-threshold', type=float, default=0.485,
                    help='P(inconsistent) above which a pair is excluded. 0.485 is the median '
                         'of ten leave-one-domain-out fits, which ranged only 0.482-0.500. At '
                         'this point: F1 0.560, balanced accuracy 0.736, solution-space error '
                         '-7.4%% on 1191 held-out cells. Raise it for fewer, safer exclusions.')
    ap.add_argument('--report', default=None)
    a = ap.parse_args()

    box = load_box(a.matrix)
    ctx = None
    if a.with_model:
        import calibrate
        import torch
        paths = [p.strip() for p in a.with_model.split(',') if p.strip()]
        loaded = [calibrate.load_model(p) for p in paths]
        tok = loaded[0][0]
        models = [m for _, m in loaded]
        y_id, n_id = calibrate.verdict_token_ids(tok)

        # the prompt must match build_dataset.cell_examples() EXACTLY -- the model was
        # trained on that wording, and any drift here silently degrades it
        from build_dataset import SYSTEM_CELL

        def score_one(a_name, ca, b_name, cb, pa, pb):
            user = '\n'.join([
                'Facet A — %s' % pa,
                '%s | REQUIRES: %s | PROVIDES: %s' % (a_name, ca['requires'], ca['provides']),
                '',
                'Facet B — %s' % pb,
                '%s | REQUIRES: %s | PROVIDES: %s' % (b_name, cb['requires'], cb['provides']),
                '',
                'Can one scenario contain both?'])
            msgs = [{'role': 'system', 'content': SYSTEM_CELL},
                    {'role': 'user', 'content': user}]
            prompt = tok.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True) + calibrate.PREFIX
            ps = []
            for mdl in models:
                enc = tok(prompt, return_tensors='pt').to(mdl.device)
                with torch.no_grad():
                    lg = mdl(**enc).logits[0, -1, :]
                pair = torch.stack([lg[y_id], lg[n_id]]).float()
                ps.append(torch.softmax(pair, dim=-1)[1].item())
            return sum(ps) / len(ps)

        def score_pair(a_name, ca, b_name, cb, pa, pb):
            """Average over both presentation orders.

            Cross-consistency is symmetric, so the two orderings are the same question.
            Averaging them is free variance reduction and measured as the single largest
            gain available at this stage: F1 0.523 -> 0.554, AUC 0.792 -> 0.817 on 1191
            held-out cells. It beat every multi-model ensemble tried.
            """
            return 0.5 * (score_one(a_name, ca, b_name, cb, pa, pb)
                          + score_one(b_name, cb, a_name, ca, pb, pa))

        ctx = {'score_one': score_pair}

    verdicts = assess(box, a.threshold, not a.narrow, ctx, a.model_threshold)
    raw, kept = solution_space(box, verdicts)
    print('  %s' % box.get('domain', '(unnamed)'))
    print('  raw configurations   %d' % raw)
    print('  consistent           %d  (%.1f%% eliminated)'
          % (kept, 100.0 * (raw - kept) / raw if raw else 0))
    n_n = sum(1 for g in verdicts.values() for r in g for c in r if c['verdict'] == 'N')
    n_c = sum(len(r) for g in verdicts.values() for r in g)
    print('  pairs excluded       %d of %d (%.1f%%)' % (n_n, n_c, 100.0 * n_n / n_c if n_c else 0))

    if a.report:
        open(a.report, 'w', encoding='utf-8').write(render(box, verdicts, raw, kept))
        print('  report -> %s' % a.report)
    return 0


if __name__ == '__main__':
    sys.exit(main())
