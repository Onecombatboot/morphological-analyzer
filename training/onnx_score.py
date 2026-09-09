# -*- coding: utf-8 -*-
"""Score cells with the INT8 ONNX model on CPU and compare against the reference.

The prompts are taken verbatim from onnx_parity_ref.json, which recorded them
when the PyTorch reference was produced. Rebuilding them here would reintroduce
exactly the drift serve_cca.py warns about, and would make any disagreement
impossible to attribute.

CPU ONLY, by construction: the session is created with CPUExecutionProvider and
nothing else, and onnxruntime in this environment ships no CUDA provider.
"""
import argparse
import json
import time

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='onnx_build/int8_fp32scale/model.onnx')
ap.add_argument('--tokenizer', default='onnx_build/int8_fp32scale')
ap.add_argument('--ref', default='onnx_parity_ref.json')
ap.add_argument('--out', default='onnx_parity_onnx.json')
ap.add_argument('--limit', type=int, default=0, help='0 = all cells')
ap.add_argument('--threads', type=int, default=8)
a = ap.parse_args()

ref = json.load(open(a.ref, encoding='utf-8'))
y_id, n_id = ref['y_id'], ref['n_id']

tok = AutoTokenizer.from_pretrained(a.tokenizer)

so = ort.SessionOptions()
so.log_severity_level = 3
so.intra_op_num_threads = a.threads
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(a.model, so, providers=['CPUExecutionProvider'])
print('  providers %s   threads %d   verdict ids Y=%d N=%d'
      % (sess.get_providers(), a.threads, y_id, n_id), flush=True)


def score(prompt):
    ids = tok(prompt, return_tensors='np')['input_ids'].astype(np.int64)
    n = ids.shape[1]
    out = sess.run(['logits'], {
        'input_ids': ids,
        'attention_mask': np.ones_like(ids),
        'position_ids': np.arange(n, dtype=np.int64)[None, :],
    })[0]
    lg = out[0, -1, :]
    # softmax over just the two verdict logits, exactly as serve_cca._one does
    pair = np.array([lg[y_id], lg[n_id]], dtype=np.float64)
    pair = pair - pair.max()
    e = np.exp(pair)
    return float(e[1] / e.sum())


cells = ref['cells']
if a.limit:
    cells = cells[:a.limit]
print('  cells %d   forward passes %d'
      % (len(cells), sum(len(c['orders']) for c in cells)), flush=True)

out = []
t0 = time.time()
for i, c in enumerate(cells):
    per = [{'order': o.get('order'), 'p': score(o['prompt'])} for o in c['orders']]
    out.append({'domain': c['domain'], 'pair': c['pair'], 'label': c['label'],
                'orders': per,
                'score': sum(x['p'] for x in per) / len(per)})
    if (i + 1) % 25 == 0:
        el = time.time() - t0
        print('    %d/%d   %.2f s/cell   eta %.1f min'
              % (i + 1, len(cells), el / (i + 1),
                 (len(cells) - i - 1) * el / (i + 1) / 60), flush=True)

el = time.time() - t0
json.dump({'y_id': y_id, 'n_id': n_id, 'model': a.model,
           'threads': a.threads, 'seconds': el, 'cells': out},
          open(a.out, 'w', encoding='utf-8'), indent=1)
print('wrote %s' % a.out)
print('  %d cells in %.0f s  (%.2f s/cell, %.3f s/forward pass)'
      % (len(out), el, el / len(out), el / sum(len(c['orders']) for c in out)))
