# -*- coding: utf-8 -*-
"""Calibrated scoring for the per-cell model.

WHY THIS EXISTS
---------------
Greedy decoding forces a hard Y/N with no control over the operating point, and the
model has consistently under-called N: recall 0.17 at precision 0.26 on the grid run.
That is a calibration failure as much as a knowledge failure -- an 82/18 class split
makes "consistent" the safe token, so the argmax sits in the wrong place.

In the per-cell format the verdict is a SINGLE token, so we can read P(N) directly from
the logits instead of generating text. Two consequences:

  * scoring is one forward pass per cell rather than autoregressive generation, so the
    whole 1191-cell validation set takes minutes;
  * the decision threshold becomes a free parameter we can choose deliberately.

METHOD
------
The 10 validation domains are split by DOMAIN into a dev half and a test half. The
threshold is chosen on dev and reported on test, so the reported number is never tuned
on itself. Splitting by domain rather than by cell matters: cells inside one domain
share characterisations, so a cell-level split would leak.

Two operating points are reported:
  balanced   -- threshold maximising balanced accuracy (the honest headline number)
  precision  -- lowest threshold whose dev precision clears --target-precision, for a
                "flag only what it is confident about" mode, which is the only mode in
                which a model this weak is genuinely useful to an analyst.
"""
import argparse, json, math, os, sys

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, 'base-model')
# No trailing space. The tokeniser merges the space into the verdict token itself
# (' Y' = 809, ' N' = 451); a prefix ending in a space would emit a standalone space
# token and leave the model scoring 'Y'/'N' without it -- different ids, meaningless
# probabilities. Verified: 'VERDICT: Y' -> [...,':',' Y'], so 'VERDICT:' is the prefix.
PREFIX = 'VERDICT:'


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def metrics(pred, truth):
    tp = sum(1 for p, t in zip(pred, truth) if p == 'N' and t == 'N')
    fp = sum(1 for p, t in zip(pred, truth) if p == 'N' and t == 'Y')
    fn = sum(1 for p, t in zip(pred, truth) if p == 'Y' and t == 'N')
    tn = sum(1 for p, t in zip(pred, truth) if p == 'Y' and t == 'Y')
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    rec_y = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    n = tp + fp + fn + tn
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1,
                accuracy=(tp + tn) / n if n else 0.0,
                balanced_accuracy=(rec + rec_y) / 2, cells=n)


def load_model(adapter):
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'          # last position must be the real final token
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
        dtype=torch.bfloat16, device_map={'': 0})
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


def verdict_token_ids(tok):
    """Token ids for the Y and N that follow 'VERDICT: '."""
    pre = tok(PREFIX, add_special_tokens=False)['input_ids']
    full_y = tok(PREFIX + ' Y', add_special_tokens=False)['input_ids']
    full_n = tok(PREFIX + ' N', add_special_tokens=False)['input_ids']
    # the answer must be exactly one token past the prefix, or we are reading the wrong
    # position and every probability below is meaningless
    for name, full in (('Y', full_y), ('N', full_n)):
        assert full[:len(pre)] == pre and len(full) == len(pre) + 1, (
            'verdict %s is not a single token after the prefix: %s' % (name, full))
    y, n = full_y[-1], full_n[-1]
    assert y != n, 'Y and N tokenise identically -- cannot score by logit'
    return y, n


def score(tok, model, rows, batch=8):
    """P(N) for every row, from one forward pass per cell."""
    y_id, n_id = verdict_token_ids(tok)
    out = []
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        prompts = [tok.apply_chat_template(r['messages'][:-1], tokenize=False,
                                           add_generation_prompt=True) + PREFIX
                   for r in chunk]
        enc = tok(prompts, return_tensors='pt', padding=True).to(model.device)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :]
        pair = torch.stack([logits[:, y_id], logits[:, n_id]], dim=-1).float()
        p_n = torch.softmax(pair, dim=-1)[:, 1]
        out += p_n.tolist()
        if (i // batch) % 20 == 0:
            print('    %d/%d' % (min(i + batch, len(rows)), len(rows)), flush=True)
    return out


def sweep(scores, truth):
    """Every threshold worth trying is just below one of the observed scores."""
    best_bal, best_bal_t = -1.0, 0.5
    grid = []
    for t in sorted(set([round(s, 4) for s in scores] + [0.5])):
        pred = ['N' if s >= t else 'Y' for s in scores]
        m = metrics(pred, truth)
        grid.append((t, m))
        if m['balanced_accuracy'] > best_bal:
            best_bal, best_bal_t = m['balanced_accuracy'], t
    return best_bal_t, grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adapter', default=os.path.join(HERE, 'adapter-best'))
    ap.add_argument('--base-only', action='store_true', help='untuned control')
    ap.add_argument('--data', default=os.path.join(HERE, 'dataset_val_cell.jsonl'))
    ap.add_argument('--target-precision', type=float, default=0.60)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--out', default=os.path.join(HERE, 'calibration.json'))
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.data, encoding='utf-8')]
    doms = sorted({r['domain'] for r in rows})
    dev_doms = set(doms[::2])                       # every other domain
    dev = [r for r in rows if r['domain'] in dev_doms]
    test = [r for r in rows if r['domain'] not in dev_doms]
    print('  %d cells: dev %d (%d domains), test %d (%d domains)'
          % (len(rows), len(dev), len(dev_doms), len(test), len(doms) - len(dev_doms)))

    adapter = None if a.base_only else a.adapter
    print('  loading %s' % ('BASE (untuned control)' if adapter is None else adapter))
    tok, model = load_model(adapter)

    print('  scoring dev...')
    dev_s = score(tok, model, dev, a.batch)
    print('  scoring test...')
    test_s = score(tok, model, test, a.batch)
    dev_t = [r['label'] for r in dev]
    test_t = [r['label'] for r in test]

    t_bal, grid = sweep(dev_s, dev_t)

    # AUC is threshold-free: it measures whether the model RANKS inconsistent pairs above
    # consistent ones, separately from where the cutoff happens to sit. That distinction
    # decides whether more training would help (poor ranking) or only recalibration would
    # (good ranking, badly placed argmax).
    def auc(s, t):
        pairs = sorted(zip(s, t))
        ranks = {}
        for i, (sc, _) in enumerate(pairs):
            ranks.setdefault(sc, []).append(i + 1)
        r = {k: sum(v) / len(v) for k, v in ranks.items()}
        n1 = sum(1 for x in t if x == 'N')
        n0 = len(t) - n1
        if not n1 or not n0:
            return 0.5
        s1 = sum(r[sc] for sc, tt in zip(s, t) if tt == 'N')
        return (s1 - n1 * (n1 + 1) / 2) / (n1 * n0)

    print('\n  AUC (ranking quality, threshold-free):  dev %.3f   test %.3f'
          % (auc(dev_s, dev_t), auc(test_s, test_t)))

    # lowest threshold whose dev precision clears the target, among those that still
    # flag something worth flagging
    t_prec, best_rec = None, -1
    for t, m in grid:
        if m['precision'] >= a.target_precision and m['tp'] >= 5 and m['recall'] > best_rec:
            t_prec, best_rec = t, m['recall']

    base_rate = sum(1 for t in test_t if t == 'N') / len(test_t)
    print('\n' + '=' * 74)
    print('  CALIBRATED RESULTS  (threshold chosen on dev, reported on test)')
    print('=' * 74)
    print('  test set: %d cells, %.1f%% N' % (len(test_t), 100 * base_rate))
    print('  baselines: always-Y bal 0.500 acc %.3f | always-N bal 0.500 F1 %.3f'
          % (1 - base_rate, 2 * base_rate / (1 + base_rate)))

    report = {'dev_cells': len(dev), 'test_cells': len(test), 'base_rate': base_rate}
    for name, t in (('argmax (threshold 0.500)', 0.5),
                    ('balanced-accuracy optimum', t_bal),
                    ('high-precision mode', t_prec)):
        if t is None:
            print('\n  %-28s no threshold reached precision %.2f on dev'
                  % (name, a.target_precision))
            continue
        m = metrics(['N' if s >= t else 'Y' for s in test_s], test_t)
        rN = m['recall']
        rY = m['tn'] / (m['tn'] + m['fp']) if (m['tn'] + m['fp']) else 0
        se = 0.5 * math.sqrt(rN * (1 - rN) / max(m['tp'] + m['fn'], 1)
                             + rY * (1 - rY) / max(m['tn'] + m['fp'], 1))
        lo, hi = m['balanced_accuracy'] - 1.96 * se, m['balanced_accuracy'] + 1.96 * se
        pl, ph = wilson(m['tp'], m['tp'] + m['fp'])
        print('\n  %s   (threshold %.4f)' % (name, t))
        print('    balanced accuracy  %.3f  95%% CI [%.3f, %.3f]   %s'
              % (m['balanced_accuracy'], lo, hi,
                 'ABOVE CHANCE' if lo > 0.5 else 'not distinguishable from chance'))
        print('    precision          %.3f  95%% CI [%.3f, %.3f]   base rate %.3f'
              % (m['precision'], pl, ph, base_rate))
        print('    recall             %.3f      F1 %.3f      accuracy %.3f'
              % (m['recall'], m['f1'], m['accuracy']))
        print('    confusion          TP %d  FP %d  FN %d  TN %d'
              % (m['tp'], m['fp'], m['fn'], m['tn']))
        report[name] = dict(threshold=t, **m)

    # per-cell scores, so downstream tooling can re-threshold and measure the
    # solution-space consequence without paying for scoring again
    dump = [{'domain': r['domain'], 'pair': r['pair'], 'label': r['label'],
             'p_inconsistent': s, 'split': sp}
            for rows_, ss, sp in ((dev, dev_s, 'dev'), (test, test_s, 'test'))
            for r, s in zip(rows_, ss)]
    json.dump(dump, open(os.path.join(HERE, 'cell_scores.json'), 'w'), indent=1)
    print('  per-cell scores -> cell_scores.json')

    json.dump(report, open(a.out, 'w'), indent=2)
    print('\n  written: %s' % a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
