# -*- coding: utf-8 -*-
"""Break a score dump down by exclusion code.

WHY THIS EXISTS
---------------
HANDOVER §6: the aggregate F1 hides the thing that actually limits the system.
Recall is 1.000 on N1/N4 (clause-grounded) but 0.200 on N7 ("possible in
principle, no real case instantiates it"), and N7 is 30.3% of all exclusions.
Any claim that a change helps has to be checked against that breakdown, not
against the aggregate, because the aggregate is dominated by the easy codes.

Reports, per code: how many held-out cells carry it, recall at the operating
threshold, mean P(inconsistent), and the code's own AUC -- that code's N cells
ranked against ALL Y cells, which is threshold-free and so is comparable across
models whose score distributions differ.

Usage:
    python by_code.py scores_w8sym.json --threshold 0.485
    python by_code.py scores_w8sym.json scores_swap_cell_w8sym.json   # TTA average
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def code_map(corpus_path):
    """(domain, 'row x col') -> exclusion code, for every N cell in the corpus."""
    corpus = json.load(open(corpus_path, encoding='utf-8'))
    out = {}
    for d in corpus:
        params = d['parameters']
        names = list(params)                      # declaration order
        for key, grid in d['verdicts'].items():
            pa, _, pb = key.partition('|')
            if pa not in params or pb not in params:
                continue
            rows, cols = params[pa], params[pb]
            reasons = d['verdict_reasons'].get(key, {})
            for ri, rowstr in enumerate(grid):
                for ci, ch in enumerate(rowstr):
                    if ch != 'N':
                        continue
                    r = reasons.get('%d,%d' % (ri + 1, ci + 1), {})
                    out[(d['domain'], '%s x %s' % (rows[ri], cols[ci]))] = \
                        r.get('code', '?')
    return out


def auc(scores, labels):
    """Mann-Whitney AUC, N = positive."""
    pairs = sorted(zip(scores, labels))
    ranks = {}
    for i, (s, _) in enumerate(pairs):
        ranks.setdefault(s, []).append(i + 1)
    r = {k: sum(v) / len(v) for k, v in ranks.items()}
    n1 = sum(1 for x in labels if x == 'N')
    n0 = len(labels) - n1
    if not n1 or not n0:
        return float('nan')
    s1 = sum(r[s] for s, t in zip(scores, labels) if t == 'N')
    return (s1 - n1 * (n1 + 1) / 2) / (n1 * n0)


def load_scores(paths):
    """Average p_inconsistent across dumps, keyed by (domain, pair)."""
    acc, meta = {}, {}
    for p in paths:
        for r in json.load(open(p, encoding='utf-8')):
            k = (r['domain'], r['pair'])
            acc.setdefault(k, []).append(r['p_inconsistent'])
            meta[k] = r
    return {k: (sum(v) / len(v), meta[k]) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scores', nargs='+')
    ap.add_argument('--corpus', default=os.path.join(HERE, 'matrices_v2', 'corpus.json'))
    ap.add_argument('--threshold', type=float, default=0.485)
    ap.add_argument('--split', default='all', choices=['all', 'dev', 'test'])
    a = ap.parse_args()

    codes = code_map(a.corpus)
    data = load_scores(a.scores)
    if a.split != 'all':
        data = {k: v for k, v in data.items() if v[1].get('split') == a.split}

    scores = [p for p, _ in data.values()]
    labels = [m['label'] for _, m in data.values()]
    n_n = labels.count('N')
    print('  %s' % ' + '.join(os.path.basename(p) for p in a.scores))
    print('  %d cells (%d N, %.1f%%)   split=%s   threshold=%.3f'
          % (len(labels), n_n, 100 * n_n / max(1, len(labels)), a.split, a.threshold))
    print('  overall AUC %.3f' % auc(scores, labels))

    y_scores = [p for (p, m) in data.values() if m['label'] == 'Y']

    buckets = {}
    missing = 0
    for k, (p, m) in data.items():
        if m['label'] != 'N':
            continue
        c = codes.get(k)
        if c is None:
            missing += 1
            c = '?unmatched'
        buckets.setdefault(c, []).append(p)

    print('\n  code            n    recall   mean P(N)   AUC vs all Y')
    print('  ' + '-' * 56)
    for c in sorted(buckets):
        v = buckets[c]
        rec = sum(1 for p in v if p >= a.threshold) / len(v)
        sub = v + y_scores
        lab = ['N'] * len(v) + ['Y'] * len(y_scores)
        print('  %-12s %4d    %.3f      %.3f        %.3f'
              % (c, len(v), rec, sum(v) / len(v), auc(sub, lab)))
    allrec = sum(1 for p, m in data.values()
                 if m['label'] == 'N' and p >= a.threshold) / max(1, n_n)
    print('  ' + '-' * 56)
    print('  %-12s %4d    %.3f      %.3f' % ('ALL N', n_n, allrec,
                                             sum(v for v, m in data.values()
                                                 if m['label'] == 'N') / max(1, n_n)))
    print('  %-12s %4d               %.3f'
          % ('Y cells', len(y_scores),
             sum(y_scores) / max(1, len(y_scores))))
    if missing:
        print('\n  NOTE: %d N cells had no code in the corpus map' % missing)


if __name__ == '__main__':
    main()
