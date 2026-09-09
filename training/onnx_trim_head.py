# -*- coding: utf-8 -*-
"""Trim the output projection to the two verdict tokens and the last position.

WHY
The service reads exactly two numbers per forward pass: the logits for Y and N
at the final token. The exported graph computes the full [1, seq, 151936] logits
tensor to produce them. At a 350-token prompt that is

  * a 350 x 2048 x 151936 matmul, about 218 GFLOP, of which all but one row is
    discarded, and
  * a 212 MB output tensor that has to be allocated, filled and handed across
    the JNI boundary on every single call.

Both are pure waste, and on CPU the second hurts more than the first.

WHAT THIS DOES
Two edits, both exact -- the two logits that survive are bit-for-bit the ones the
untrimmed graph produces, so accuracy is unchanged and the parity measurements
carry over:

  1. Insert a Slice before /lm_head/MatMul that keeps only the last position of
     the hidden states, [1, seq, 2048] -> [1, 1, 2048].
  2. Slice the lm_head weight from [2048, 151936] to [2048, 2], keeping only the
     Y and N columns.

The output becomes [1, 1, 2], where index 0 is Y and index 1 is N.

This makes the graph specific to this task. That is the point: it is a verdict
scorer, not a general language model, and it was never used as one.
"""
import argparse
import os
import shutil

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ap = argparse.ArgumentParser()
ap.add_argument('--src', default='onnx_build/int4_fix/model.onnx')
ap.add_argument('--dst', default='onnx_build/int4_trim/model.onnx')
ap.add_argument('--y', type=int, default=809)
ap.add_argument('--n', type=int, default=451)
a = ap.parse_args()

os.makedirs(os.path.dirname(a.dst), exist_ok=True)
print('loading %s' % a.src, flush=True)
m = onnx.load(a.src, load_external_data=True)
g = m.graph

lm = next(n for n in g.node if n.name == '/lm_head/MatMul')
hidden_in, weight_name = lm.input[0], lm.input[1]
print('  lm_head: %s x %s -> %s' % (hidden_in, weight_name, lm.output[0]))

# ---- 1. slice the weight to the two verdict columns -------------------------
inits = {i.name: i for i in g.initializer}
W = numpy_helper.to_array(inits[weight_name])
print('  weight %s -> keeping columns Y=%d N=%d' % (str(W.shape), a.y, a.n))
W2 = np.ascontiguousarray(W[:, [a.y, a.n]])
new_w = numpy_helper.from_array(W2.astype(W.dtype), weight_name + '_yn')
g.initializer.append(new_w)
lm.input[1] = new_w.name

# ---- 2. slice the hidden states to the final position -----------------------
# Slice(starts=[-1], ends=[INT64_MAX], axes=[1]) keeps the last token only, and
# works for any sequence length because the bounds are relative.
for nm, vals in (('trim_starts', [-1]),
                 ('trim_ends', [np.iinfo(np.int64).max]),
                 ('trim_axes', [1])):
    g.initializer.append(numpy_helper.from_array(np.array(vals, dtype=np.int64), nm))

sliced = hidden_in + '_last'
slice_node = helper.make_node('Slice',
                              inputs=[hidden_in, 'trim_starts', 'trim_ends', 'trim_axes'],
                              outputs=[sliced],
                              name='/lm_head/SliceLastPosition')

idx = next(i for i, n in enumerate(g.node) if n.name == '/lm_head/MatMul')
g.node.insert(idx, slice_node)
lm.input[0] = sliced

# ---- 3. declare the new output shape ----------------------------------------
del g.output[:]
g.output.append(helper.make_tensor_value_info(
    'logits', TensorProto.FLOAT, ['batch_size', 1, 2]))

# value_info for the old full-width logits would now be wrong
keep = [vi for vi in g.value_info if vi.name != 'logits']
del g.value_info[:]
g.value_info.extend(keep)

print('saving -> %s' % a.dst, flush=True)
onnx.save(m, a.dst, save_as_external_data=True, all_tensors_to_one_file=True,
          location=os.path.basename(a.dst) + '.data', size_threshold=1024,
          convert_attribute=False)

srcdir, dstdir = os.path.dirname(a.src), os.path.dirname(a.dst)
for f in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
          'vocab.json', 'merges.txt', 'config.json', 'generation_config.json',
          'added_tokens.json', 'chat_template.jinja'):
    s = os.path.join(srcdir, f)
    if os.path.exists(s):
        shutil.copy(s, os.path.join(dstdir, f))

tot = sum(os.path.getsize(os.path.join(dstdir, f)) for f in os.listdir(dstdir))
print('  total %.2f GB' % (tot / 2 ** 30))
print('done -- output is [batch, 1, 2]: index 0 = Y, index 1 = N')
