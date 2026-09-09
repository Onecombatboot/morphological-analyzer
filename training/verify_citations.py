# -*- coding: utf-8 -*-
"""Reject exclusions whose cited evidence does not exist.

IDEA
----
The per-cell model is trained to emit not just a verdict but the reason and the two
clauses it rests on:

    VERDICT: N
    CODE: N3 (capability)
    REQUIRES: a transponder fitted and functioning
    BLOCKED BY: no monitoring equipment

Those quotes are checkable. `build_matrices.py` already re-verifies every citation in the
corpus verbatim; the same test applies to what the model produces. An N whose cited clause
does not appear in either characterisation is an exclusion the model cannot support -- a
hallucinated justification -- and downgrading it to Y should remove false positives while
leaving well-grounded exclusions untouched.

This costs generation rather than a single logit, so it runs only on the cells already
predicted N, which is a small fraction of the total.

    python verify_citations.py --adapter adapter-best --threshold 0.15
"""
import argparse, json, math, os, re, sys

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CODES = {'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7'}


def norm(s):
    return ' '.join((s or '').lower().split())


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def parse_answer(text):
    code = re.search(r'CODE:\s*(N[1-7])', text)
    req = re.search(r'REQUIRES:\s*(.+)', text)
    blk = re.search(r'BLOCKED BY:\s*(.+)', text)
    return (code.group(1) if code else None,
            req.group(1).strip() if req else None,
            blk.group(1).strip() if blk else None)


def supported(code, req, blk, hay):
    """A citation is supported only if the code is valid and both quotes are present."""
    if code not in CODES:
        return False, 'invalid or missing code'
    if not req or not blk:
        return False, 'missing a quote'
    if norm(req) not in hay:
        return False, 'REQUIRES quote not in the characterisations'
    if norm(blk) not in hay:
        return False, 'BLOCKED BY quote not in the characterisations'
    return True, 'supported'


def metrics(pred, truth):
    tp = sum(1 for p, t in zip(pred, truth) if p == 'N' and t == 'N')
    fp = sum(1 for p, t in zip(pred, truth) if p == 'N' and t == 'Y')
    fn = sum(1 for p, t in zip(pred, truth) if p == 'Y' and t == 'N')
    tn = sum(1 for p, t in zip(pred, truth) if p == 'Y' and t == 'Y')
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    rY = tn / (tn + fp) if tn + fp else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec,
                f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                balanced_accuracy=(rec + rY) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adapter', default=os.path.join(HERE, 'adapter-best'))
    ap.add_argument('--scores', default=os.path.join(HERE, 'scores_epoch-1.json'))
    ap.add_argument('--data', default=os.path.join(HERE, 'dataset_val_cell.jsonl'))
    ap.add_argument('--threshold', type=float, default=0.15)
    ap.add_argument('--split', default='test')
    ap.add_argument('--max-new', type=int, default=64)
    a = ap.parse_args()

    import torch
    import calibrate

    rows = [json.loads(l) for l in open(a.data, encoding='utf-8')]
    sc = {}
    for r in json.load(open(a.scores, encoding='utf-8')):
        sc[(r['domain'], r['pair'])] = r
    rows = [r for r in rows
            if sc.get((r['domain'], r['pair']), {}).get('split') == a.split]
    print('  %d cells in the %s split' % (len(rows), a.split))

    flagged = [r for r in rows
               if sc[(r['domain'], r['pair'])]['p_inconsistent'] >= a.threshold]
    print('  %d predicted inconsistent at threshold %.2f -- generating citations for those only'
          % (len(flagged), a.threshold))

    tok, model = calibrate.load_model(a.adapter)
    tok.padding_side = 'left'

    kept, dropped, reasons = set(), set(), {}
    for i, r in enumerate(flagged):
        prompt = tok.apply_chat_template(r['messages'][:-1], tokenize=False,
                                         add_generation_prompt=True)
        enc = tok(prompt, return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        gen = tok.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
        user = r['messages'][1]['content']
        hay = norm(user)
        ok, why = supported(*parse_answer(gen), hay)
        key = (r['domain'], r['pair'])
        (kept if ok else dropped).add(key)
        reasons[why] = reasons.get(why, 0) + 1
        if i % 25 == 0:
            print('    %d/%d' % (i, len(flagged)), flush=True)

    truth = [r['label'] for r in rows]
    before = ['N' if sc[(r['domain'], r['pair'])]['p_inconsistent'] >= a.threshold else 'Y'
              for r in rows]
    after = ['N' if (r['domain'], r['pair']) in kept else 'Y' for r in rows]

    mb, ma = metrics(before, truth), metrics(after, truth)
    print('\n' + '=' * 68)
    print('  CITATION VERIFICATION  (%s split)' % a.split)
    print('=' * 68)
    print('  exclusions kept %d, rejected %d' % (len(kept), len(dropped)))
    for why, n in sorted(reasons.items(), key=lambda x: -x[1]):
        print('    %-46s %d' % (why, n))
    print()
    print('  %-24s %-8s %-8s' % ('', 'before', 'after'))
    for k in ('precision', 'recall', 'f1', 'balanced_accuracy'):
        print('  %-24s %.3f    %.3f' % (k, mb[k], ma[k]))
    pl, ph = wilson(ma['tp'], max(ma['tp'] + ma['fp'], 1))
    print('  precision 95%% CI after: [%.3f, %.3f]' % (pl, ph))
    print('  confusion before  TP %d FP %d FN %d TN %d' % (mb['tp'], mb['fp'], mb['fn'], mb['tn']))
    print('  confusion after   TP %d FP %d FN %d TN %d' % (ma['tp'], ma['fp'], ma['fn'], ma['tn']))
    json.dump({'before': mb, 'after': ma, 'rejected': len(dropped)},
              open(os.path.join(HERE, 'citation_check.json'), 'w'), indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
