# -*- coding: utf-8 -*-
"""Merge the LoRA adapter into the base model and export it to ONNX.

Three stages, each skippable if its output already exists:

  1. merge     base-model + results/cell_r32d  ->  onnx_build/merged   (fp16 HF)
  2. export    merged                          ->  onnx_build/fp32     (ONNX)
  3. quantise  fp32                            ->  onnx_build/int8     (ONNX INT8)

WHY NO KV CACHE
The deployed scoring path runs ONE forward pass per prompt and reads the logits
at the final position; it never generates a second token. Exporting the
with-past variant would add cache inputs and outputs that are dead weight here
and complicate the Java side for nothing. The export therefore uses the
no-cache decoder.

CPU ONLY
The merge may run on the GPU because it is a one-off convenience. The exported
graph and the INT8 quantisation are CPU artefacts: quantisation targets
CPUExecutionProvider, and onnxruntime in this environment has no CUDA provider
at all, so a GPU dependency cannot be introduced accidentally.
"""
import argparse
import os
import shutil
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--base', default='base-model')
ap.add_argument('--adapter', default='results/cell_r32d')
ap.add_argument('--out', default='onnx_build')
ap.add_argument('--stage', default='all',
                choices=['all', 'merge', 'export', 'quantise'])
ap.add_argument('--keep-fp32', action='store_true',
                help='keep the fp32 ONNX after quantising (it is ~12 GB)')
a = ap.parse_args()

MERGED = os.path.join(a.out, 'merged')
FP32 = os.path.join(a.out, 'fp16')
INT8 = os.path.join(a.out, 'int8')
os.makedirs(a.out, exist_ok=True)


def free_gb(path='C:\\'):
    import ctypes
    fb = ctypes.c_ulonglong(0)
    ctypes.windll.kernel32.GetDiskFreeSpaceExW(
        ctypes.c_wchar_p(path), None, None, ctypes.pointer(fb))
    return fb.value / 2 ** 30


def banner(t):
    print()
    print('=' * 70)
    print('  ' + t + '     [free %.1f GB]' % free_gb())
    print('=' * 70, flush=True)


# ---------------------------------------------------------------- 1. merge
if a.stage in ('all', 'merge'):
    if os.path.exists(os.path.join(MERGED, 'config.json')):
        banner('1. merge -- already present, skipping')
    else:
        banner('1. merging LoRA adapter into base weights')
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        t0 = time.time()
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('  loading base (%s) on %s' % (a.base, dev), flush=True)
        m = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.float16,
                                                 device_map=dev)
        print('  applying adapter %s' % a.adapter, flush=True)
        m = PeftModel.from_pretrained(m, a.adapter)
        print('  merge_and_unload ...', flush=True)
        m = m.merge_and_unload()
        m = m.to('cpu').half()
        os.makedirs(MERGED, exist_ok=True)
        print('  saving merged fp16 -> %s' % MERGED, flush=True)
        m.save_pretrained(MERGED, safe_serialization=True)
        AutoTokenizer.from_pretrained(a.base).save_pretrained(MERGED)
        del m
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        print('  merged in %.0fs' % (time.time() - t0), flush=True)

# ---------------------------------------------------------------- 2. export
if a.stage in ('all', 'export'):
    if os.path.exists(os.path.join(FP32, 'model.onnx')):
        banner('2. export -- already present, skipping')
    else:
        banner('2. exporting to ONNX (no KV cache, one forward pass)')
        if free_gb() < 10:
            sys.exit('  need ~10 GB free for the fp16 export, have %.1f' % free_gb())
        from optimum.exporters.onnx import main_export
        t0 = time.time()
        main_export(
            model_name_or_path=MERGED,
            output=FP32,
            task='text-generation',          # no-cache decoder
            device='cpu',
            # fp16, not fp32. optimum's exporter calls onnx.load_model() on the
            # traced graph, which materialises every external weight in RAM:
            # 12.4 GB at fp32 against 15 GB of system RAM is a hard MemoryError,
            # observed twice. fp16 halves that to ~6.2 GB and fits. The INT8
            # quantisation below is what the deployed artefact actually uses, so
            # the intermediate precision only has to survive the export.
            dtype='fp16',
            # True is essential here. With post-processing on, onnx reads every
            # external weight file back into RAM to consolidate them -- 13 GB for
            # this model, against 15 GB of system RAM, which is a hard MemoryError.
            # There is nothing to post-process anyway: the no-cache export has no
            # decoder-with-past variant to merge.
            no_post_process=True,
            do_validation=False,
        )
        print('  exported in %.0fs' % (time.time() - t0), flush=True)
        for f in sorted(os.listdir(FP32)):
            p = os.path.join(FP32, f)
            print('    %-40s %8.1f MB' % (f, os.path.getsize(p) / 1e6))

# ---------------------------------------------------------------- 3. quantise
if a.stage in ('all', 'quantise'):
    banner('3. dynamic INT8 quantisation (CPU target)')
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from onnxruntime.quantization.shape_inference import quant_pre_process

    os.makedirs(INT8, exist_ok=True)
    src = os.path.join(FP32, 'model.onnx')
    pre = os.path.join(a.out, '_pre.onnx')
    dst = os.path.join(INT8, 'model.onnx')

    t0 = time.time()
    # quant_pre_process() writes a complete second copy of the model. At 6.7 GB
    # that does not fit alongside the source and the INT8 output, so the raw
    # graph is quantised directly. Pre-processing only improves shape inference,
    # which dynamic weight-only quantisation does not depend on.
    qsrc = src

    print('  quantising MatMul weights to INT8 ...', flush=True)
    # use_external_data_format is required, not optional: the INT8 model is over
    # protobuf's 2 GB single-message limit, and without it the quantisation runs
    # to completion and then dies on save with "Failed to serialize proto".
    quantize_dynamic(qsrc, dst, weight_type=QuantType.QInt8,
                     use_external_data_format=True,
                     extra_options={'MatMulConstBOnly': True})

    # the tokenizer and config travel with the model; the Java side needs them
    for f in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
              'vocab.json', 'merges.txt', 'config.json', 'generation_config.json',
              'added_tokens.json', 'chat_template.jinja'):
        s = os.path.join(FP32, f)
        if os.path.exists(s):
            shutil.copy(s, os.path.join(INT8, f))

    if os.path.exists(pre):
        os.remove(pre)
    print('  quantised in %.0fs' % (time.time() - t0), flush=True)

    tot = 0
    for f in sorted(os.listdir(INT8)):
        p = os.path.join(INT8, f)
        sz = os.path.getsize(p)
        tot += sz
        print('    %-40s %8.1f MB' % (f, sz / 1e6))
    print('  INT8 total: %.2f GB' % (tot / 2 ** 30))

    if not a.keep_fp32 and os.path.exists(FP32):
        print('  removing fp32 export to reclaim disk (pass --keep-fp32 to retain)')
        shutil.rmtree(FP32, ignore_errors=True)

banner('done')
