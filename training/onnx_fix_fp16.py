# -*- coding: utf-8 -*-
"""Promote every float16 tensor in the INT8 graph to float32.

WHY THIS EXISTS
The model was exported at fp16 because a fp32 export cannot be saved on this
machine -- optimum calls onnx.load_model() on the traced graph, which
materialises 12.4 GB of weights in 15 GB of RAM and dies. Quantising the fp16
graph works, but produces fp16 quantisation scales, and ONNX Runtime's CPU
DequantizeLinear rejects those:

    Type 'tensor(float16)' of input parameter (..._scale) of operator
    (DequantizeLinear) is invalid.

The fix is cheap because the bulk of the model is already int8 and is left
completely untouched. Only the small tensors -- scales, zero points that are
stored as float, layer-norm weights, biases -- are float16, and promoting them
to float32 is a few megabytes of change against a 3.4 GB file.

The promotion is applied uniformly to initialisers, node attributes, graph
inputs and outputs, and value_info, so the graph stays type-consistent and needs
no inserted Cast nodes.
"""
import argparse
import os
import shutil

import onnx
from onnx import TensorProto, numpy_helper

ap = argparse.ArgumentParser()
ap.add_argument('--src', default='onnx_build/int8/model.onnx')
ap.add_argument('--dst', default='onnx_build/int8_fp32scale/model.onnx')
a = ap.parse_args()

os.makedirs(os.path.dirname(a.dst), exist_ok=True)

print('loading %s (external data)' % a.src, flush=True)
m = onnx.load(a.src, load_external_data=True)

F16 = TensorProto.FLOAT16
F32 = TensorProto.FLOAT


def promote_type(tp):
    """tensor / sequence / optional type protos, recursively."""
    if tp.HasField('tensor_type'):
        if tp.tensor_type.elem_type == F16:
            tp.tensor_type.elem_type = F32
            return 1
    elif tp.HasField('sequence_type'):
        return promote_type(tp.sequence_type.elem_type)
    elif tp.HasField('optional_type'):
        return promote_type(tp.optional_type.elem_type)
    return 0


n_init = n_attr = n_io = n_vi = 0

# ---- initialisers ----------------------------------------------------------
for init in m.graph.initializer:
    if init.data_type == F16:
        arr = numpy_helper.to_array(init).astype('float32')
        new = numpy_helper.from_array(arr, init.name)
        init.CopyFrom(new)
        n_init += 1

# ---- constant tensors held as node attributes ------------------------------
for node in m.graph.node:
    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == F16:
            arr = numpy_helper.to_array(attr.t).astype('float32')
            attr.t.CopyFrom(numpy_helper.from_array(arr, attr.t.name))
            n_attr += 1
        elif attr.type == onnx.AttributeProto.TENSORS:
            for t in attr.tensors:
                if t.data_type == F16:
                    arr = numpy_helper.to_array(t).astype('float32')
                    t.CopyFrom(numpy_helper.from_array(arr, t.name))
                    n_attr += 1
        # Cast(to=FLOAT16) must become Cast(to=FLOAT) or it re-narrows
        elif attr.name == 'to' and attr.type == onnx.AttributeProto.INT and attr.i == F16:
            attr.i = F32
            n_attr += 1

# ---- graph inputs / outputs / value_info -----------------------------------
for vi in list(m.graph.input) + list(m.graph.output):
    n_io += promote_type(vi.type)
for vi in m.graph.value_info:
    n_vi += promote_type(vi.type)

print('  initialisers promoted : %d' % n_init)
print('  node attributes       : %d' % n_attr)
print('  graph inputs/outputs  : %d' % n_io)
print('  value_info            : %d' % n_vi)

print('saving -> %s' % a.dst, flush=True)
onnx.save(m, a.dst, save_as_external_data=True, all_tensors_to_one_file=True,
          location=os.path.basename(a.dst) + '.data', size_threshold=1024,
          convert_attribute=False)

# the tokenizer and config travel with the model
srcdir = os.path.dirname(a.src)
dstdir = os.path.dirname(a.dst)
for f in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
          'vocab.json', 'merges.txt', 'config.json', 'generation_config.json',
          'added_tokens.json', 'chat_template.jinja'):
    s = os.path.join(srcdir, f)
    if os.path.exists(s):
        shutil.copy(s, os.path.join(dstdir, f))

tot = sum(os.path.getsize(os.path.join(dstdir, f)) for f in os.listdir(dstdir))
print('  total %.2f GB' % (tot / 2 ** 30))
print('done')
