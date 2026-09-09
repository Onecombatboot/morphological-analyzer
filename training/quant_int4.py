# -*- coding: utf-8 -*-
"""Block-wise INT4 quantisation of the exported graph.

WHY BLOCK-WISE RATHER THAN INT8
Dynamic INT8 gives each weight matrix (per-tensor) or each output channel
(per-channel) a single scale. Across a 2048-wide projection the dynamic range
varies enough that one scale is coarse, and the resulting error landed directly
on the verdict logit: AUC 0.8785 -> 0.8527, and F1 at 50% coverage 0.805 ->
0.716, which is the number the report is built on.

Block-wise INT4 gives every 32 weights their own scale. Fewer bits, far finer
scales -- for transformer weights that trade is usually a net win, and it is the
standard approach for LLM weight-only quantisation.

/lm_head/MatMul stays out of it. The verdict is read from two specific positions
of that projection's output, so error there is not absorbed by any later layer.

CPU ONLY. onnxruntime here has no CUDA provider; the output runs on
CPUExecutionProvider.
"""
import os, shutil, time
from onnxruntime.quantization.matmul_nbits_quantizer import (
    MatMulNBitsQuantizer, RTNWeightOnlyQuantConfig)
import onnx

SRC = 'onnx_build/fp16/model.onnx'
DST = 'onnx_build/int4/model.onnx'
os.makedirs(os.path.dirname(DST), exist_ok=True)

t0 = time.time()
print('loading graph ...', flush=True)
m = onnx.load(SRC, load_external_data=True)

print('quantising: 4 bits, block_size 32, lm_head excluded ...', flush=True)
q = MatMulNBitsQuantizer(
    model=m,
    bits=4,
    block_size=32,
    is_symmetric=False,          # asymmetric keeps a zero-point; better for
                                 # weight distributions that are not centred
    nodes_to_exclude=['/lm_head/MatMul'],
    algo_config=RTNWeightOnlyQuantConfig(),
)
q.process()
print('  quantised in %.0fs, saving ...' % (time.time() - t0), flush=True)

out = q.model.model if hasattr(q.model, 'model') else q.model
onnx.save(out, DST, save_as_external_data=True, all_tensors_to_one_file=True,
          location=os.path.basename(DST) + '.data', size_threshold=1024,
          convert_attribute=False)

for f in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json',
          'vocab.json','merges.txt','config.json','generation_config.json',
          'added_tokens.json','chat_template.jinja'):
    s = os.path.join('onnx_build/fp16', f)
    if os.path.exists(s):
        shutil.copy(s, os.path.join(os.path.dirname(DST), f))

d = os.path.dirname(DST)
print('  total %.2f GB  (%.0fs)' % (
    sum(os.path.getsize(os.path.join(d,f)) for f in os.listdir(d))/2**30,
    time.time()-t0))
