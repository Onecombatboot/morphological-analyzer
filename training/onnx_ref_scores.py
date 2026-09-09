# -*- coding: utf-8 -*-
"""Reference scores from the PyTorch adapter path, for ONNX parity checking.

Reads the PINNED validation file for the deployed run (dataset_val_r32d.jsonl),
so the prompts are byte-identical to the ones the reported figures were measured
on. Nothing is reconstructed here -- reconstructing the prompt is exactly the
drift serve_cca.py warns about.

Scores every example the way serve_cca._one() does: one forward pass, last
position, softmax over the two verdict token ids. Both presentation orders are
present in the file as separate records and are paired up by (domain, pair).

The GPU is used only to produce the reference quickly. The artefact this whole
exercise produces is CPU-only; nothing here leaks into it.
"""
import argparse
import collections
import json

import torch

import calibrate

ap = argparse.ArgumentParser()
ap.add_argument('--base', default='base-model')
ap.add_argument('--adapter', default='results/cell_r32d')
ap.add_argument('--val', default='dataset_val_r32d.jsonl')
ap.add_argument('--limit', type=int, default=0, help='0 = all cells')
ap.add_argument('--out', default='onnx_parity_ref.json')
a = ap.parse_args()

# The canonical loader -- 4-bit NF4 with bf16 compute, exactly what serve_cca.py
# runs and what the reported figures were measured on. Loading fp16 here instead
# would make the reference a different numerical path from the deployed one.
print('loading via calibrate.load_model (4-bit NF4, deployed config) ...', flush=True)
tok, model = calibrate.load_model(a.adapter)

y_id, n_id = calibrate.verdict_token_ids(tok)
print('  verdict ids  Y=%d  N=%d' % (y_id, n_id), flush=True)


def prompt_of(rec):
    msgs = [m for m in rec['messages'] if m['role'] in ('system', 'user')]
    return tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True) + calibrate.PREFIX


@torch.no_grad()
def score(prompt):
    enc = tok(prompt, return_tensors='pt').to(model.device)
    lg = model(**enc, logits_to_keep=1).logits[0, -1, :]
    pair = torch.stack([lg[y_id], lg[n_id]]).float()
    return torch.softmax(pair, dim=-1)[1].item()


# ---- group the two presentation orders of each cell ------------------------
groups = collections.OrderedDict()
for line in open(a.val, encoding='utf-8'):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    groups.setdefault((r['domain'], r['pair']), []).append(r)

keys = list(groups)
if a.limit:
    keys = keys[:a.limit]
print('  cells: %d   forward passes: %d'
      % (len(keys), sum(len(groups[k]) for k in keys)), flush=True)

out = []
for i, k in enumerate(keys):
    recs = groups[k]
    per = []
    for r in recs:
        p = prompt_of(r)
        per.append({'order': r.get('order'), 'prompt': p, 'p': score(p)})
    lab = recs[0]['label']
    out.append({'domain': k[0], 'pair': k[1], 'label': lab,
                'orders': per,
                'score': sum(x['p'] for x in per) / len(per)})
    if (i + 1) % 100 == 0:
        print('    %d/%d' % (i + 1, len(keys)), flush=True)

json.dump({'y_id': y_id, 'n_id': n_id, 'base': a.base, 'adapter': a.adapter,
           'val': a.val, 'cells': out},
          open(a.out, 'w', encoding='utf-8'), indent=1)

pos = [c['score'] for c in out if str(c['label']).upper().startswith('N')]
neg = [c['score'] for c in out if not str(c['label']).upper().startswith('N')]
print('wrote %s' % a.out)
print('  cells %d   label N(inconsistent)=%d  Y=%d' % (len(out), len(pos), len(neg)))
print('  score range %.4f .. %.4f' % (min(c['score'] for c in out),
                                      max(c['score'] for c in out)))
