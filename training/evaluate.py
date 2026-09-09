# -*- coding: utf-8 -*-
"""Scores the fine-tune on the held-out domains, against the untuned base as a control.

Loss is not interpretable for this task. What matters is whether the verdicts are right, so this
generates each held-out example and scores the parsed grid.

Design points that make the number mean something:

  * The base model is evaluated on the SAME examples. F1 for the tuned model alone is unreadable --
    there is no way to know whether 0.6 is good without knowing what the model did before.
  * "Inconsistent" (N) is the positive class. That is the judgement the tool exists to make, and it
    is the minority class, so accuracy alone would flatter a model that marks nothing.
  * The held-out domains never appear in training, so this measures generalisation, not recall of
    memorised grids.
  * Malformed output is counted, not silently dropped. A model that emits an unparseable grid has
    failed just as surely as one that emits wrong verdicts, and hiding that would overstate both.
  * Greedy decoding, matching the deterministic settings used in deployment.
"""
import json, os, sys, re, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, 'base-model')
ADAPTER = os.path.join(HERE, 'adapter-best')


def load_jsonl(p):
    with open(p, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def parse_grid(text, n_rows, n_cols):
    """Pulls the verdict grid out of generated text. Returns None if it cannot be read cleanly."""
    block = text.split('VERDICTS')[-1] if 'VERDICTS' in text else text
    rows = {}
    for line in block.split('\n'):
        m = re.match(r'\s*(\d+)\s*:\s*([YN]+)', line.strip())
        if not m:
            continue
        idx = int(m.group(1)) - 1
        verdicts = m.group(2)
        if 0 <= idx < n_rows and idx not in rows:
            rows[idx] = verdicts
    if len(rows) != n_rows:
        return None
    out = []
    for i in range(n_rows):
        r = rows[i]
        if len(r) != n_cols:
            return None
        out.append(r)
    return out


def truth_grid(example):
    block = example['messages'][-1]['content'].split('VERDICTS')[-1]
    return [m.group(1) for m in re.finditer(r'\d+:([YN]+)', block)]


def metrics(tp, fp, fn, tn):
    """Positive class is N (inconsistent)."""
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    # the same three for Y, so macro-F1 does not hide a model that only does one class well
    p_y = tn / (tn + fn) if tn + fn else 0.0
    r_y = tn / (tn + fp) if tn + fp else 0.0
    f1_y = 2 * p_y * r_y / (p_y + r_y) if p_y + r_y else 0.0
    n = tp + fp + fn + tn
    acc = (tp + tn) / n if n else 0.0
    bal = ((rec + r_y) / 2) if n else 0.0
    return dict(precision=prec, recall=rec, f1=f1,
                precision_y=p_y, recall_y=r_y, f1_y=f1_y,
                macro_f1=(f1 + f1_y) / 2, accuracy=acc, balanced=bal,
                tp=tp, fp=fp, fn=fn, tn=tn, n=n)


def evaluate(model, tok, rows, label, max_new=700):
    tp = fp = fn = tn = 0
    malformed = 0
    per_domain = {}

    for i, ex in enumerate(rows):
        truth = truth_grid(ex)
        n_rows, n_cols = len(truth), len(truth[0])

        prompt = tok.apply_chat_template(
            ex['messages'][:-1], tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors='pt').to(model.device)

        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        gen = tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)

        pred = parse_grid(gen, n_rows, n_cols)
        d = per_domain.setdefault(ex['domain'], [0, 0, 0])   # correct, total, malformed
        if pred is None:
            malformed += 1
            d[2] += 1
            # a grid that cannot be read is a failure on every cell it should have covered;
            # counting it as "skipped" would quietly reward unparseable output
            for r in truth:
                for c in r:
                    if c == 'N':
                        fn += 1
                    else:
                        fp += 1
                    d[1] += 1
            continue

        for tr, pr in zip(truth, pred):
            for t, p in zip(tr, pr):
                d[1] += 1
                if t == 'N' and p == 'N':
                    tp += 1; d[0] += 1
                elif t == 'Y' and p == 'N':
                    fp += 1
                elif t == 'N' and p == 'Y':
                    fn += 1
                else:
                    tn += 1; d[0] += 1

        if (i + 1) % 10 == 0:
            print('    %s: %d/%d' % (label, i + 1, len(rows)), flush=True)

    m = metrics(tp, fp, fn, tn)
    m['malformed'] = malformed
    m['per_domain'] = per_domain
    return m


def show(name, m):
    print()
    print('=' * 66)
    print('  %s' % name)
    print('=' * 66)
    print('  cells scored      %d      unparseable grids: %d' % (m['n'], m['malformed']))
    print()
    print('  positive class = N (inconsistent)')
    print('    precision       %.3f   of the pairs it excluded, this share should have been' % m['precision'])
    print('    recall          %.3f   of the pairs that should be excluded, this share were' % m['recall'])
    print('    F1              %.3f' % m['f1'])
    print()
    print('  class Y (consistent)')
    print('    precision       %.3f' % m['precision_y'])
    print('    recall          %.3f' % m['recall_y'])
    print('    F1              %.3f' % m['f1_y'])
    print()
    print('    macro F1        %.3f' % m['macro_f1'])
    print('    accuracy        %.3f' % m['accuracy'])
    print('    balanced acc    %.3f   (0.500 = chance)' % m['balanced'])
    print()
    print('  confusion:  TP %d   FP %d   FN %d   TN %d' % (m['tp'], m['fp'], m['fn'], m['tn']))
    print()
    print('  per held-out domain:')
    for dom, (ok, tot, mal) in sorted(m['per_domain'].items()):
        flag = '  (%d unparseable)' % mal if mal else ''
        print('    %-38s %3d/%-3d  %.0f%%%s' % (dom, ok, tot, 100.0 * ok / tot if tot else 0, flag))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-only', action='store_true')
    ap.add_argument('--limit', type=int, default=0,
                    help='score only the first N held-out examples')
    ap.add_argument('--max-new', type=int, default=700,
                    help='the tuned model stops on its own; the untuned base never '
                         'does and burns the full budget on every example')
    ap.add_argument('--tuned-only', action='store_true')
    a = ap.parse_args()

    rows = load_jsonl(os.path.join(HERE, 'dataset_val.jsonl'))
    if a.limit:
        rows = rows[:a.limit]
    doms = sorted({r['domain'] for r in rows})
    print('held-out: %d examples across %d domains' % (len(rows), len(doms)))
    for d in doms:
        print('   -', d)

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True)

    results = {}

    if not a.tuned_only:
        print('\nloading BASE (control)...')
        base = AutoModelForCausalLM.from_pretrained(
            BASE, quantization_config=qc, dtype=torch.bfloat16, device_map={'': 0})
        base.eval()
        results['BASE Qwen2.5-3B (untuned control)'] = evaluate(base, tok, rows, 'base', a.max_new)
        del base
        torch.cuda.empty_cache()

    if not a.base_only:
        from peft import PeftModel
        print('\nloading FINE-TUNED...')
        m = AutoModelForCausalLM.from_pretrained(
            BASE, quantization_config=qc, dtype=torch.bfloat16, device_map={'': 0})
        m = PeftModel.from_pretrained(m, ADAPTER)
        m.eval()
        results['FINE-TUNED (rank 16, all 7 projections, adapter-best)'] = evaluate(m, tok, rows, 'tuned', a.max_new)

    for k, v in results.items():
        show(k, v)

    if len(results) == 2:
        b = results['BASE Qwen2.5-3B (untuned control)']
        t = results['FINE-TUNED (rank 16, all 7 projections, adapter-best)']
        print()
        print('=' * 66)
        print('  DELTA')
        print('=' * 66)
        for key, lab in [('f1', 'F1 (inconsistent)'), ('precision', 'precision'),
                         ('recall', 'recall'), ('macro_f1', 'macro F1'),
                         ('balanced', 'balanced accuracy')]:
            d = t[key] - b[key]
            print('    %-22s %.3f -> %.3f   %+.3f' % (lab, b[key], t[key], d))
        print('    %-22s %d -> %d' % ('unparseable grids', b['malformed'], t['malformed']))

    with open(os.path.join(HERE, 'eval_results.json'), 'w', encoding='utf-8') as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'per_domain'}
                   for k, v in results.items()}, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
