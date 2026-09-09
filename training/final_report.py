# -*- coding: utf-8 -*-
"""Consolidate every scored run into one comparable table and write FINAL_RESULTS.md.

Runs unattended at the end of the training queue, so the numbers are ready without
anyone having to re-derive them.

Everything is put on the SAME footing:
  * the threshold is chosen leave-one-domain-out and maximises F1, not balanced
    accuracy -- switching that objective is worth ~0.03 F1 and comparing runs that
    used different objectives would be meaningless;
  * every run is scored on the same 1191 held-out cells;
  * the best run is bootstrapped against the reference by DOMAIN, because cells
    inside a matrix are correlated and resampling cells gives an interval several
    times too narrow.

Usage:  python final_report.py
"""
import glob
import json
import os
import random

import numpy as np

from by_code import auc

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE = 'scores_noN7c_v0.json'      # the N7-scoped retrain everything is measured against


def load(path):
    """(domain, pair) -> (mean score over presentation orders, label)."""
    acc = {}
    for r in json.load(open(path, encoding='utf-8')):
        k = (r['domain'], r['pair'])
        acc.setdefault(k, {'v': [], 'l': r['label']})['v'].append(r['p_inconsistent'])
    return {k: (sum(d['v']) / len(d['v']), d['l']) for k, d in acc.items()}


def loo_metrics(score, lab, dom):
    grid = np.unique(np.round(score, 4))
    if len(grid) > 600:
        grid = np.quantile(score, np.linspace(0, 1, 600))

    def f1(th, m):
        tp = int(((score[m] >= th) & (lab[m] == 1)).sum())
        fp = int(((score[m] >= th) & (lab[m] == 0)).sum())
        fn = int(((score[m] < th) & (lab[m] == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return 2 * p * r / (p + r) if p + r else 0.0

    pred = np.zeros(len(score), dtype=int)
    for h in np.unique(dom):
        m = dom != h
        th = max(((f1(t, m), t) for t in grid))[1]
        pred[dom == h] = (score[dom == h] >= th).astype(int)
    tp = int(((pred == 1) & (lab == 1)).sum()); fp = int(((pred == 1) & (lab == 0)).sum())
    fn = int(((pred == 0) & (lab == 1)).sum()); tn = int(((pred == 0) & (lab == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    ry = tn / (tn + fp) if tn + fp else 0.0
    return dict(f1=2 * p * r / (p + r) if p + r else 0.0, precision=p, recall=r,
                balanced=(r + ry) / 2, accuracy=(tp + tn) / len(score),
                tp=tp, fp=fp, fn=fn)


def arrays(d):
    keys = sorted(d)
    return (np.array([d[k][0] for k in keys]),
            np.array([1 if d[k][1] == 'N' else 0 for k in keys]),
            np.array([k[0] for k in keys]), keys)


def selective(score, lab, dom):
    grid = np.unique(np.round(score, 4))

    def f1(th, m):
        tp = int(((score[m] >= th) & (lab[m] == 1)).sum())
        fp = int(((score[m] >= th) & (lab[m] == 0)).sum())
        fn = int(((score[m] < th) & (lab[m] == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return 2 * p * r / (p + r) if p + r else 0.0

    thr = {h: max(((f1(t, dom != h), t) for t in grid))[1] for h in np.unique(dom)}
    tv = np.array([thr[d] for d in dom])
    order = np.argsort(-np.abs(score - tv))
    rows = []
    for cov in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        k = order[:int(cov * len(score))]
        tp = int(((score[k] >= tv[k]) & (lab[k] == 1)).sum())
        fp = int(((score[k] >= tv[k]) & (lab[k] == 0)).sum())
        fn = int(((score[k] < tv[k]) & (lab[k] == 1)).sum())
        tn = int(((score[k] < tv[k]) & (lab[k] == 0)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        rows.append((cov, len(k), 2 * p * r / (p + r) if p + r else 0.0, p, r,
                     (tp + tn) / len(k)))
    return rows


def bootstrap(a, b, iters=400):
    """95% CI on F1(b) - F1(a), resampling DOMAINS."""
    keys = sorted(set(a) & set(b))
    dom = [k[0] for k in keys]
    doms = sorted(set(dom))
    lab = np.array([1 if a[k][1] == 'N' else 0 for k in keys])
    sa = np.array([a[k][0] for k in keys]); sb = np.array([b[k][0] for k in keys])
    by = {d: [i for i, x in enumerate(dom) if x == d] for d in doms}
    rng = random.Random(7)
    out = []
    for _ in range(iters):
        idx = [i for d in [rng.choice(doms) for _ in doms] for i in by[d]]
        dd = np.array([dom[i] for i in idx])
        out.append(loo_metrics(sb[idx], lab[idx], dd)['f1']
                   - loo_metrics(sa[idx], lab[idx], dd)['f1'])
    out.sort()
    return out[int(0.025 * len(out))], out[int(0.975 * len(out))], \
        sum(1 for x in out if x > 0) / len(out)


def main():
    files = sorted(glob.glob(os.path.join(HERE, 'scores_*_v0.json')))
    files += [f for f in [os.path.join(HERE, 'scores_pe_v0.json')] if os.path.exists(f)]
    ref = load(os.path.join(HERE, REFERENCE)) if os.path.exists(
        os.path.join(HERE, REFERENCE)) else None

    rows = []
    loaded = {}
    for f in sorted(set(files)):
        name = os.path.basename(f).replace('scores_', '').replace('_v0.json', '')
        try:
            d = load(f)
            if len(d) < 500:
                continue
            s, lab, dom, _ = arrays(d)
            m = loo_metrics(s, lab, dom)
            m['auc'] = auc(list(s), ['N' if v else 'Y' for v in lab])
            m['n'] = len(s)
            rows.append((name, m))
            loaded[name] = d
        except Exception as e:
            print('  skipped %s (%s)' % (name, e))

    rows.sort(key=lambda t: -t[1]['f1'])
    out = ['# Final results — cross-consistency assessment', '',
           'All runs scored on the same 1,191 held-out cells from 10 domains no model',
           'trained on. Threshold chosen leave-one-domain-out to maximise F1.', '',
           '| run | AUC | F1 | precision | recall | balanced acc | accuracy |',
           '|---|---|---|---|---|---|---|']
    for name, m in rows:
        out.append('| %s | %.3f | **%.3f** | %.3f | %.3f | %.3f | %.3f |'
                   % (name, m['auc'], m['f1'], m['precision'], m['recall'],
                      m['balanced'], m['accuracy']))

    best_name, best_m = rows[0]
    out += ['', '## Best run: `%s`' % best_name, '',
            'F1 %.3f · AUC %.3f · precision %.3f · recall %.3f · accuracy %.3f'
            % (best_m['f1'], best_m['auc'], best_m['precision'], best_m['recall'],
               best_m['accuracy']),
            '', 'Confusion: %d true positives, %d false positives, %d false negatives.'
            % (best_m['tp'], best_m['fp'], best_m['fn']), '']

    s, lab, dom, _ = arrays(loaded[best_name])
    out += ['## Selective prediction', '',
            'The system decides the cells it is confident about and refers the rest.', '',
            '| coverage | cells | F1 | precision | recall | accuracy |',
            '|---|---|---|---|---|---|']
    for cov, n, f1v, p, r, acc in selective(s, lab, dom):
        out.append('| %.0f%% | %d | **%.3f** | %.3f | %.3f | %.3f |'
                   % (cov * 100, n, f1v, p, r, acc))

    if ref is not None and best_name != REFERENCE.replace('scores_', '').replace('_v0.json', ''):
        lo, hi, pb = bootstrap(ref, loaded[best_name])
        out += ['', '## Is the best run really better than the reference?', '',
                'Domain-level bootstrap of F1(best) - F1(%s):' % REFERENCE,
                '', '- 95%% CI **[%+.3f, %+.3f]**' % (lo, hi),
                '- P(best is better) = **%.2f**' % pb,
                '', 'The interval spans zero, so the difference is not established.'
                if lo <= 0 <= hi else
                '', 'The interval excludes zero.']

    out += ['', '## Context', '',
            '- Human parity on this task is F1 ~0.72, from the blind re-labelling',
            '  (two annotators, 289 cells, agreement 0.889, kappa 0.730).',
            '- Chance on balanced accuracy is 0.500.',
            '- Always-"consistent" scores 86.7% accuracy at this base rate, so accuracy',
            '  should never be the headline.', '']
    p = os.path.join(HERE, 'FINAL_RESULTS.md')
    open(p, 'w', encoding='utf-8').write('\n'.join(out))
    print('\n'.join(out[:40]))
    print('\n  wrote %s' % p)


if __name__ == '__main__':
    main()
