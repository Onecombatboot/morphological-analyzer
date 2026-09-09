# -*- coding: utf-8 -*-
"""Metrics helper shared by the reference check and the ONNX parity check."""
import json


def load_scores(path):
    d = json.load(open(path, encoding='utf-8'))
    ys, ps, keys = [], [], []
    for c in d['cells']:
        ys.append(1 if str(c['label']).upper().startswith('N') else 0)
        ps.append(c['score'])
        keys.append((c['domain'], c['pair']))
    return keys, ys, ps


def auc(ys, ps):
    n = len(ys)
    pos = sum(ys)
    neg = n - pos
    if not pos or not neg:
        return float('nan')
    order = sorted(range(n), key=lambda k: ps[k])
    rank = [0.0] * n
    k = 0
    while k < n:
        j = k
        while j + 1 < n and ps[order[j + 1]] == ps[order[k]]:
            j += 1
        avg = (k + j) / 2.0 + 1
        for t in range(k, j + 1):
            rank[order[t]] = avg
        k = j + 1
    s = sum(rank[i] for i in range(n) if ys[i] == 1)
    return (s - pos * (pos + 1) / 2.0) / (pos * neg)


def prf(ys, ps, thr):
    tp = sum(1 for i in range(len(ys)) if ps[i] >= thr and ys[i] == 1)
    fp = sum(1 for i in range(len(ys)) if ps[i] >= thr and ys[i] == 0)
    fn = sum(1 for i in range(len(ys)) if ps[i] < thr and ys[i] == 1)
    tn = len(ys) - tp - fp - fn
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return dict(f1=f1, precision=pr, recall=rc, tp=tp, fp=fp, fn=fn, tn=tn,
                accuracy=(tp + tn) / len(ys), balanced=(rc + spec) / 2.0)


def best_f1(ys, ps):
    best, bt = None, 0.0
    for t in sorted(set(ps)):
        m = prf(ys, ps, t)
        if best is None or m['f1'] > best['f1']:
            best, bt = m, t
    return best, bt


def coverage_curve(ys, ps, thr, fractions=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4)):
    """Selective prediction: confidence is distance from the decision threshold."""
    n = len(ys)
    conf = sorted(range(n), key=lambda i: -abs(ps[i] - thr))
    rows = []
    for f in fractions:
        k = max(1, int(round(n * f)))
        idx = conf[:k]
        sy = [ys[i] for i in idx]
        sp = [ps[i] for i in idx]
        m = prf(sy, sp, thr)
        rows.append((f, k, m))
    return rows
