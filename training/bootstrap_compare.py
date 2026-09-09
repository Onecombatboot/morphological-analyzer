# -*- coding: utf-8 -*-
"""Paired bootstrap comparison of two score dumps over the same held-out cells.

WHY THIS EXISTS
---------------
Runs differ by 0.005-0.015 AUC and it is tempting to read that as an improvement.
It is not readable that way without a confidence interval, and the interval has to
respect the structure of the data: HANDOVER.md section 6 notes the effective sample
size is about 70 independent items, not 7790 cells, because one property of a
scenario drives many exclusions at once. Cells inside a domain are therefore
correlated, and resampling CELLS would give a CI several times too narrow.

So the resampling unit is the DOMAIN. With 10 validation domains that produces a
coarse interval -- which is the honest answer, not a defect: 10 clusters cannot
support fine distinctions, and any method suggesting otherwise is lying.

Both dumps must cover the same cells; the comparison is paired, so each bootstrap
sample scores both models on the identical resampled domains.

Usage:
    python bootstrap_compare.py scores_pe_v0.json scores_cell_92c_vw4_v0.json
"""
import argparse
import json
import os
import random

from by_code import auc


def load(path):
    d = {}
    for r in json.load(open(path, encoding='utf-8')):
        d[(r['domain'], r['pair'])] = (r['p_inconsistent'], r['label'])
    return d


def auc_over(keys, dump):
    s = [dump[k][0] for k in keys]
    t = [dump[k][1] for k in keys]
    return auc(s, t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--iters', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()

    A, B = load(a.a), load(a.b)
    keys = sorted(set(A) & set(B))
    if len(keys) != len(A) or len(keys) != len(B):
        print('  WARNING: dumps cover different cells (%d vs %d, %d shared)'
              % (len(A), len(B), len(keys)))

    by_dom = {}
    for k in keys:
        by_dom.setdefault(k[0], []).append(k)
    doms = sorted(by_dom)

    a_auc, b_auc = auc_over(keys, A), auc_over(keys, B)
    print('  %-34s AUC %.4f' % (os.path.basename(a.a), a_auc))
    print('  %-34s AUC %.4f' % (os.path.basename(a.b), b_auc))
    print('  observed difference (b - a)         %+.4f' % (b_auc - a_auc))
    print('  %d cells in %d domains; resampling DOMAINS, %d iterations'
          % (len(keys), len(doms), a.iters))

    rng = random.Random(a.seed)
    diffs = []
    for _ in range(a.iters):
        pick = [rng.choice(doms) for _ in doms]
        ks = [k for d in pick for k in by_dom[d]]
        try:
            diffs.append(auc_over(ks, B) - auc_over(ks, A))
        except Exception:
            continue
    diffs.sort()
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[int(0.975 * len(diffs))]
    frac = sum(1 for d in diffs if d > 0) / len(diffs)
    print('  95%% CI on the difference          [%+.4f, %+.4f]' % (lo, hi))
    print('  P(b better than a)                 %.3f' % frac)
    verdict = ('DIFFERENT -- the interval excludes zero'
               if (lo > 0 or hi < 0) else
               'INDISTINGUISHABLE -- the interval spans zero')
    print('  ->  %s' % verdict)


if __name__ == '__main__':
    main()
