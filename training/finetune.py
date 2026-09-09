# -*- coding: utf-8 -*-
"""
Self-contained QLoRA fine-tune for GMA cross-consistency assessment.

RUN THIS IN A VS CODE TERMINAL:

    cd C:\\morphological-analyzer\\training
    python finetune.py

To resume after any interruption (it will pick up from the newest checkpoint):

    python finetune.py --resume

--------------------------------------------------------------------------------
WHY THE PREVIOUS TWO RUNS CRASHED THE MACHINE
--------------------------------------------------------------------------------
The longest training sequence in this dataset is 521 tokens. The previous script
reserved max_len=1600 -- a three-fold over-allocation. Attention and activation
memory scale with sequence length, so the card sat at 5913 / 6141 MiB (96.3%),
leaving 230 MB of headroom.

Under Windows WDDM, a GPU that full begins paging VRAM into system RAM. With
system memory also tight, that is the condition under which the display driver
wedges. Both bugchecks are consistent with it:

    0x0000009F  DRIVER_POWER_STATE_FAILURE   (a driver blocked a power IRP)
    0x0000001E  KMODE_EXCEPTION_NOT_HANDLED  (kernel access violation)

Neither was caused by the training mathematics. They were caused by running the
card at 96% occupancy for hours.

FIXES APPLIED HERE
    max_len            1600 -> 640    (521 max + margin)  ~3x less activation memory
    batch size            1 -> 2      the freed memory buys throughput back
    expandable_segments  off -> on    reduces allocator fragmentation over long runs
    VRAM guard          none -> abort if headroom < 800 MB after the first step
    sleep inhibit       none -> on    SetThreadExecutionState for the whole run
--------------------------------------------------------------------------------
"""
import os

# Must be set BEFORE torch initialises CUDA. Expandable segments let the allocator
# grow and shrink a single arena instead of fragmenting into many fixed blocks,
# which is what causes creeping VRAM growth over a many-hour run.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import json, sys, re, argparse, time, shutil, ctypes, platform

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments, Trainer,
                          TrainerCallback)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, 'base-model')
OUT = os.path.join(HERE, 'adapter')
BEST = os.path.join(HERE, 'adapter-best')
HISTORY = os.path.join(HERE, 'metrics_history.json')

TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                  'gate_proj', 'up_proj', 'down_proj']

MIN_FREE_VRAM_MB = int(__import__('os').environ.get('MIN_FREE_VRAM_MB', 800))
# 800 is a round number, not a measured threshold. The bugchecks in the header
# happened at 96% occupancy (230 MB free); the validated recipe runs at 85.7%
# (879 MB free). Rank 32 sits at 783 MB / 87.2%, between the two and much nearer
# the safe end -- so it is worth lowering the floor for, unlike the 7B at 253 MB.


class CpuEmbedding(torch.nn.Module):
    """Token embedding whose weight stays in system RAM.

    WHY: on the 7B the embedding is 152064 x 3584. In bf16 that is 1090 MB of a
    6 GB card, spent entirely on a lookup -- no matmul, no gradient (it is frozen
    under LoRA). Freeing it is the difference between the 7B fitting and not:
    weights drop from 4751 MB to ~3660 MB, which leaves room to actually train.

    Do NOT use accelerate's device_map={'model.embed_tokens': 'cpu'} for this.
    That does not keep the computation on the CPU -- it copies the weight ONTO
    the GPU before every forward (and llm_int8_enable_fp32_cpu_offload holds it
    in fp32 first, so it tries to allocate 2.03 GiB). Measured: it OOMs.

    The weight is a plain attribute, not a Parameter or a buffer, so .to(),
    .cuda() and the Trainer's device placement all leave it alone. Only the
    (batch, seq, hidden) result crosses the bus -- about 1.6 MB per step.
    """

    def __init__(self, weight, out_device):
        super().__init__()
        self.w = weight                 # cpu, bf16, deliberately not registered
        self.out_device = out_device
        self.num_embeddings, self.embedding_dim = weight.shape

    def forward(self, ids):
        return self.w[ids.to('cpu')].to(self.out_device)


def cpu_embed_device_map(base):
    """device_map that keeps the embedding off the GPU from the very start.

    Swapping the module after from_pretrained() is too late: transformers'
    caching_allocator_warmup reserves VRAM for every module the map places on the
    GPU before any weight is read, and on the 7B that reservation (4.80 GiB)
    already exceeds the cap. The embedding has to be excluded up front.
    """
    from transformers import AutoConfig
    n = AutoConfig.from_pretrained(base).num_hidden_layers
    dm = {'model.embed_tokens': 'cpu', 'model.norm': 0, 'model.rotary_emb': 0,
          'lm_head': 0}
    for i in range(n):
        dm['model.layers.%d' % i] = 0
    return dm


def load_embed_weight(base):
    """Read model.embed_tokens.weight straight from the checkpoint into RAM.

    Not taken from the loaded model: accelerate leaves CPU-mapped parameters on
    the meta device, so model.get_input_embeddings().weight has no data.
    """
    from safetensors.torch import safe_open
    key = 'model.embed_tokens.weight'
    idx_path = os.path.join(base, 'model.safetensors.index.json')
    if os.path.exists(idx_path):
        shard = json.load(open(idx_path))['weight_map'][key]
    else:
        shard = 'model.safetensors'
    with safe_open(os.path.join(base, shard), framework='pt') as f:
        return f.get_tensor(key).to(torch.bfloat16)


def prepare_kbit_lean(model):
    """prepare_model_for_kbit_training without the blanket fp32 upcast.

    peft casts EVERY fp16/bf16 parameter to fp32. On the 3B that is harmless --
    its embedding is tied, so there is only one large matrix. The 7B's lm_head is
    untied and 152064 x 3584: 1.09 GB in bf16, 2.18 GB once upcast, which OOMs on
    a 6 GB card before training starts.

    The numerical reason for the upcast is the normalisation layers, where fp16
    accumulation genuinely misbehaves. Those are the 1-D parameters and cost a few
    MB. Large 2-D weights are either already nf4 or are frozen projections that
    train fine in bf16, so they are left alone.
    """
    for p in model.parameters():
        p.requires_grad = False
    upcast = 0
    for _, p in model.named_parameters():
        if p.ndim == 1 and p.dtype in (torch.float16, torch.bfloat16):
            p.data = p.data.to(torch.float32)
            upcast += p.numel()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    print('  lean kbit prep: %d norm parameters upcast to fp32, large weights left '
          'in place' % upcast)
    return model


def move_embedding_to_cpu(model, base):
    """Replace the meta/CPU embedding module with a RAM-resident lookup."""
    if getattr(model.config, 'tie_word_embeddings', False):
        # lm_head would follow the embedding and end up on the CPU with it
        raise SystemExit('  --cpu-embed needs untied embeddings; this model ties them')
    w = load_embed_weight(base)
    model.set_input_embeddings(CpuEmbedding(w, torch.device('cuda:0')))
    print('  embedding %s kept in system RAM (%.0f MB of VRAM never claimed)'
          % (tuple(w.shape), w.numel() * 2 / 1e6))
    return model


# ----------------------------------------------------------------- sleep inhibit

def prevent_sleep():
    """Stops Windows putting the display or system to sleep for the run's duration.

    One of the two crashes was a power-state failure. Even with AC sleep disabled,
    a display power transition can attempt to move the GPU to a lower power state
    while a CUDA context is active. This removes that class of event entirely.
    """
    if platform.system() != 'Windows':
        return
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
        print('  sleep and display timeout inhibited for this run')
    except Exception as e:
        print('  could not inhibit sleep:', e)


def allow_sleep():
    if platform.system() != 'Windows':
        return
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    except Exception:
        pass


# ----------------------------------------------------------------- data

def load_jsonl(p):
    with open(p, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


class CcaDataset(torch.utils.data.Dataset):
    """Tokenises each example and masks the prompt out of the loss."""

    def __init__(self, rows, tok, max_len):
        self.rows, self.tok, self.max_len = rows, tok, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        m = self.rows[i]['messages']
        prompt = self.tok.apply_chat_template(m[:-1], tokenize=False, add_generation_prompt=True)
        full = prompt + m[-1]['content'] + self.tok.eos_token
        p_ids = self.tok(prompt, add_special_tokens=False)['input_ids']
        f_ids = self.tok(full, add_special_tokens=False)['input_ids'][:self.max_len]
        labels = list(f_ids)
        # -100 wherever the model is only reading, so gradient comes from the answer alone
        for j in range(min(len(p_ids), len(labels))):
            labels[j] = -100
        return {'input_ids': f_ids, 'labels': labels, 'attention_mask': [1] * len(f_ids)}


def collate(batch, pad_id):
    n = max(len(b['input_ids']) for b in batch)
    out = {'input_ids': [], 'labels': [], 'attention_mask': []}
    for b in batch:
        d = n - len(b['input_ids'])
        out['input_ids'].append(b['input_ids'] + [pad_id] * d)
        out['labels'].append(b['labels'] + [-100] * d)      # padding never contributes loss
        out['attention_mask'].append(b['attention_mask'] + [0] * d)
    return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


# ----------------------------------------------------------------- verdict scoring

def parse_grid(text, n_rows, n_cols):
    """None if the grid cannot be read cleanly. That is a failure, not a skip."""
    block = text.split('VERDICTS')[-1] if 'VERDICTS' in text else text
    rows = {}
    for line in block.split('\n'):
        m = re.match(r'\s*(\d+)\s*:\s*([YN]+)', line.strip())
        if m:
            i = int(m.group(1)) - 1
            if 0 <= i < n_rows and i not in rows:
                rows[i] = m.group(2)
    if len(rows) != n_rows or any(len(rows[i]) != n_cols for i in rows):
        return None
    return [rows[i] for i in range(n_rows)]


def truth_grid(ex):
    block = ex['messages'][-1]['content'].split('VERDICTS')[-1]
    return [m.group(1) for m in re.finditer(r'\d+:([YN]+)', block)]


def score_verdicts(model, tok, rows, max_new=400, limit=None):
    """Generates each held-out example and scores the parsed grid.

    N (inconsistent) is the positive class: it is the judgement the tool exists to
    make, and only ~26% of cells are N, so plain accuracy would flatter a model
    that marks nothing.
    """
    tp = fp = fn = tn = 0
    malformed = 0
    subset = rows[:limit] if limit else rows
    was_training = model.training
    model.eval()
    cache = model.config.use_cache
    model.config.use_cache = True
    # generation allocates a KV cache; hand back the training arena first so the two
    # peaks never coincide
    torch.cuda.empty_cache()

    for ex in subset:
        truth = truth_grid(ex)
        if not truth:
            continue
        n_rows, n_cols = len(truth), len(truth[0])
        prompt = tok.apply_chat_template(ex['messages'][:-1], tokenize=False,
                                         add_generation_prompt=True)
        ids = tok(prompt, return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        gen = tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)
        pred = parse_grid(gen, n_rows, n_cols)

        if pred is None:
            malformed += 1
            # an unreadable grid fails every cell it should have covered; treating it
            # as "skipped" would quietly reward unparseable output
            for r in truth:
                fn += r.count('N')
                fp += r.count('Y')
            continue
        for tr, pr in zip(truth, pred):
            for t, p in zip(tr, pr):
                if t == 'N' and p == 'N': tp += 1
                elif t == 'Y' and p == 'N': fp += 1
                elif t == 'N' and p == 'Y': fn += 1
                else: tn += 1

    model.config.use_cache = cache
    torch.cuda.empty_cache()
    if was_training:
        model.train()

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    p_y = tn / (tn + fn) if tn + fn else 0.0
    r_y = tn / (tn + fp) if tn + fp else 0.0
    f1_y = 2 * p_y * r_y / (p_y + r_y) if p_y + r_y else 0.0
    n = tp + fp + fn + tn
    return dict(precision=prec, recall=rec, f1=f1,
                precision_y=p_y, recall_y=r_y, f1_y=f1_y,
                macro_f1=(f1 + f1_y) / 2,
                accuracy=(tp + tn) / n if n else 0.0,
                balanced_accuracy=(rec + r_y) / 2,
                tp=tp, fp=fp, fn=fn, tn=tn, cells=n,
                malformed_grids=malformed, examples=len(subset))


def score_cells(model, tok, rows, max_new=64, limit=None):
    """Scoring for the per-cell dataset, where each example carries exactly one verdict.

    Much cheaper than grid scoring: the answer is 5 tokens for Y and about 40 for a typed N,
    against ~30 for a whole grid and 262 under the original format. That is what makes it
    affordable to score hundreds of cells at every checkpoint instead of a few dozen.
    """
    tp = fp = fn = tn = 0
    malformed = 0
    subset = rows[:limit] if limit else rows
    was_training = model.training
    model.eval()
    cache = model.config.use_cache
    model.config.use_cache = True
    torch.cuda.empty_cache()

    for ex in subset:
        truth = ex.get('label')
        if truth not in ('Y', 'N'):
            continue
        prompt = tok.apply_chat_template(ex['messages'][:-1], tokenize=False,
                                         add_generation_prompt=True)
        ids = tok(prompt, return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        gen = tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)
        m = re.search(r'VERDICT:\s*([YN])', gen)
        if not m:
            # unreadable output is a failure, not a skip: counted against whatever the truth was
            malformed += 1
            if truth == 'N':
                fn += 1
            else:
                fp += 1
            continue
        pred = m.group(1)
        if truth == 'N' and pred == 'N':
            tp += 1
        elif truth == 'Y' and pred == 'N':
            fp += 1
        elif truth == 'N' and pred == 'Y':
            fn += 1
        else:
            tn += 1

    model.config.use_cache = cache
    torch.cuda.empty_cache()
    if was_training:
        model.train()

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    r_y = tn / (tn + fp) if (tn + fp) else 0.0
    p_y = tn / (tn + fn) if (tn + fn) else 0.0
    f1_y = 2 * p_y * r_y / (p_y + r_y) if (p_y + r_y) else 0.0
    n = tp + fp + fn + tn
    return dict(precision=prec, recall=rec, f1=f1, precision_y=p_y, recall_y=r_y, f1_y=f1_y,
                macro_f1=(f1 + f1_y) / 2, accuracy=(tp + tn) / n if n else 0.0,
                balanced_accuracy=(rec + r_y) / 2, tp=tp, fp=fp, fn=fn, tn=tn, cells=n,
                malformed_grids=malformed, examples=len(subset))


# ----------------------------------------------------------------- loss

VERDICT_WEIGHT = 1.0        # set from --verdict-weight
VERDICT_IDS = [None, None]  # ' Y' and ' N' token ids, filled in main()

LOSS_CHUNK = int(__import__('os').environ.get('LOSS_CHUNK', 64))         # answer tokens scored per chunk; halving 128 -> 64 trims the
                        # step-1 peak, which came within 77 MB of the abort floor


def _chunk_ce(hidden, head, targets, tok_w=None):
    """Cross-entropy for one slice of answer positions.

    Kept as a module-level function so torch.utils.checkpoint can recompute it in
    the backward pass instead of holding its logits in memory.

    `head` is the output-embedding MODULE, not its .weight: under
    --quantize-lm-head the weight is a bitsandbytes Params4bit holding packed
    nibbles (shape 272498688 x 1), which F.linear cannot multiply. Calling the
    module lets bitsandbytes dequantise on the fly. Qwen2's lm_head has no bias,
    so for an ordinary nn.Linear this is exactly F.linear(hidden, weight).
    """
    # The final norm is fp32 (see prepare_kbit_lean) while lm_head stays bf16, so
    # hidden and the head can disagree. peft's version avoids this only by
    # upcasting both, which is the 2.18 GB we are refusing to spend. A quantised
    # head has a non-float weight dtype and casts its own input, so skip it there.
    w = getattr(head, 'weight', None)
    if w is not None and w.dtype.is_floating_point and hidden.dtype != w.dtype:
        hidden = hidden.to(w.dtype)
    ce = F.cross_entropy(head(hidden).float(), targets, reduction="none")
    if tok_w is not None:
        ce = ce * tok_w
    return ce.sum()


class ChunkedLossTrainer(Trainer):
    """Computes the language-modelling loss without ever materialising full logits.

    WHY THIS EXISTS
    Qwen2.5-3B has a 151,936-token vocabulary against a hidden size of 2048. A
    576-token sequence therefore produces a logits tensor of 576 x 151936. Between
    the bf16 logits, the fp32 upcast the loss requires, the softmax temporary and
    the gradient, that single tensor costs roughly 1.1 GB -- more than the model
    weights, the LoRA adapter and the optimiser state combined. Gradient
    checkpointing does not touch it, because it is the final output rather than an
    intermediate activation. It is what raised the OutOfMemoryError in cross_entropy.

    Two changes remove it:

      1. logits_to_keep=1 stops the model running its lm_head over the sequence at
         all. Only the hidden states come back, and those are 2.2 MB.

      2. The loss is then computed only at positions the collator did not mask
         (the assistant answer -- 67% of tokens, and the only place gradient comes
         from anyway), in chunks of LOSS_CHUNK, each wrapped in a checkpoint so its
         logits are freed immediately and recomputed during backward.

    Peak cost becomes one chunk: 128 x 151936, about 200 MB including the fp32
    upcast. The mathematics is unchanged -- test_chunked_loss() asserts this
    against the model's own loss.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        out = model(**inputs, output_hidden_states=True, logits_to_keep=1)

        # predict token t+1 from the hidden state at t
        hidden = out.hidden_states[-1][:, :-1, :]
        target = labels[:, 1:]
        keep = target != -100

        sel = hidden[keep]              # [N, H] -- answer positions only
        tgt = target[keep]              # [N]
        n = sel.shape[0]
        if n == 0:
            loss = (hidden.sum() * 0.0)
            return (loss, out) if return_outputs else loss

        head = model.get_output_embeddings()

        # Up-weight the verdict token. Only 8.5% of supervised tokens are the actual
        # Y/N decision (6.1% on N examples, where the typed reason dominates), so by
        # default ~91% of the gradient trains reason prose rather than the judgement we
        # are scored on -- the same misallocation that crippled the grid format.
        vw = torch.ones_like(tgt, dtype=torch.float32)
        if VERDICT_WEIGHT != 1.0 and VERDICT_IDS[0] is not None:
            for vid in VERDICT_IDS:
                vw = torch.where(tgt == vid, torch.full_like(vw, VERDICT_WEIGHT), vw)
            # renormalise to mean 1 so the OVERALL gradient scale is unchanged and only
            # the balance between verdict and prose shifts. Without this the loss grows
            # ~1.6x, grad_norm goes from ~1.5 to ~15, and the trainer's clipping at 1.0
            # fires on every step -- which silently shrinks the effective learning rate
            # for the whole model instead of emphasising the verdict.
            vw = vw / vw.mean().clamp(min=1e-6)

        total = None
        for i in range(0, n, LOSS_CHUNK):
            part = checkpoint(_chunk_ce, sel[i:i + LOSS_CHUNK], head, tgt[i:i + LOSS_CHUNK],
                              vw[i:i + LOSS_CHUNK], use_reentrant=False)
            total = part if total is None else total + part
        # Normalisation must follow the Trainer's contract exactly. When it passes
        # num_items_in_batch (the scored-token count across the WHOLE accumulation
        # window) it does not divide by gradient_accumulation_steps itself -- it
        # expects each micro-batch to contribute sum/num_items_in_batch so the parts
        # add up to the true mean. Dividing by this micro-batch's own n instead makes
        # every gradient gradient_accumulation_steps times too large, which is what
        # put the logged loss at ~17 when the true value is ~2.6.
        denom = num_items_in_batch if num_items_in_batch is not None else n
        if torch.is_tensor(denom):
            denom = denom.to(total.device)
        loss = total / denom
        return (loss, out) if return_outputs else loss


# ----------------------------------------------------------------- callbacks

class VramGuard(TrainerCallback):
    """Aborts cleanly if VRAM headroom falls below the safe floor.

    Existing to prevent a third bugcheck: a clean abort with checkpoints intact is
    always better than a wedged display driver and a forced restart.
    """

    STRIKES = 3          # consecutive low readings before aborting

    def __init__(self):
        self.checked = False
        self.peak_mb = 0
        self.low = 0

    def on_step_end(self, args, state, control, **kw):
        if not torch.cuda.is_available():
            return
        free, total = torch.cuda.mem_get_info()
        free_mb = free / 1024 / 1024
        used_mb = (total - free) / 1024 / 1024
        self.peak_mb = max(self.peak_mb, used_mb)

        # EVERY step, not just the first. The original checked once and never again,
        # so VRAM that crept up later -- allocator fragmentation, or the KV cache
        # during periodic scoring -- went unnoticed for hours. Nobody is watching
        # this run, so it has to police itself for its whole duration.
        if free_mb < MIN_FREE_VRAM_MB:
            self.low += 1
            print('  WARNING: VRAM headroom %.0f MB at step %d (strike %d of %d)'
                  % (free_mb, state.global_step, self.low, self.STRIKES), flush=True)
            if self.low >= self.STRIKES:
                msg = ('VRAM headroom stayed under %d MB for %d consecutive steps '
                       '(last reading %.0f MB free). Stopping to protect the display '
                       'driver. Checkpoints are intact -- resume with --resume, or '
                       'restart with a lower --vram-fraction.'
                       % (MIN_FREE_VRAM_MB, self.STRIKES, free_mb))
                print('\n  ABORTING: ' + msg, flush=True)
                raise RuntimeError(msg)
        else:
            self.low = 0     # transient dips do not count; only sustained pressure

        if not self.checked:
            self.checked = True
            print('\n  VRAM after first step: %.0f MB used, %.0f MB free of %.0f MB'
                  % (used_mb, free_mb, total / 1024 / 1024), flush=True)
            if free_mb < MIN_FREE_VRAM_MB:
                # should_training_stop is only honoured at the END of an epoch, so on
                # a 1280-step run it would let the machine thrash for hours first.
                # Raising is the only way to stop immediately.
                msg = ('VRAM headroom %.0f MB is below the %d MB floor. Stopping now rather '
                       'than risking another driver bugcheck.\n'
                       '  Try:  python finetune.py --rank 16\n'
                       '  or:   python finetune.py --vram-fraction 0.75 --rank 16\n'
                       '  and close Chrome or anything else using the GPU.'
                       % (free_mb, MIN_FREE_VRAM_MB))
                print('\n  ABORTING: ' + msg, flush=True)
                raise RuntimeError(msg)


class MetricsCallback(TrainerCallback):
    """Scores verdicts periodically, writes history immediately, keeps the best adapter."""

    def __init__(self, model, tok, val_rows, every, limit, max_new=330):
        self.model, self.tok, self.val = model, tok, val_rows
        self.every, self.limit = every, limit
        self.max_new = max_new
        self.history = []
        self.best_score = -1.0
        if os.path.exists(HISTORY):
            try:
                self.history = json.load(open(HISTORY, encoding='utf-8'))
                self.best_score = max([h.get('balanced_accuracy', -1) for h in self.history] + [-1])
                print('  resuming history: %d entries, best balanced accuracy so far %.3f'
                      % (len(self.history), self.best_score))
            except Exception:
                self.history = []

    def _write(self):
        tmp = HISTORY + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)
        os.replace(tmp, HISTORY)     # atomic: a crash mid-write cannot corrupt the record

    def on_epoch_end(self, args, state, control, **kw):
        ep = int(round(state.epoch))
        if ep == 0 or ep % self.every != 0:
            return
        t0 = time.time()
        scorer = score_cells if (self.val and 'label' in self.val[0]) else score_verdicts
        m = scorer(self.model, self.tok, self.val,
                   max_new=self.max_new, limit=self.limit)
        m['epoch'] = ep
        m['step'] = state.global_step
        m['seconds'] = round(time.time() - t0, 1)
        for rec in reversed(state.log_history):
            if 'loss' in rec:
                m['train_loss'] = rec['loss']
                break
        for rec in reversed(state.log_history):
            if 'eval_loss' in rec:
                m['eval_loss'] = rec['eval_loss']
                break
        self.history.append(m)
        self._write()

        print('\n  epoch %-3d  F1 %.3f  P %.3f  R %.3f  bal %.3f  acc %.3f  malformed %d  (%.0fs)'
              % (ep, m['f1'], m['precision'], m['recall'],
                 m['balanced_accuracy'], m['accuracy'], m['malformed_grids'], m['seconds']),
              flush=True)

        # Select on BALANCED ACCURACY, not F1. Under an 82/18 split, F1 on the minority
        # class rewards a model simply for calling N more often, and in the 16-epoch run it
        # did exactly that: it kept epoch 10 (F1 0.165, balanced accuracy 0.482 -- below
        # chance) over epoch 6 (F1 0.161, balanced accuracy 0.528). Balanced accuracy cannot
        # be gamed by changing the calling rate, so it is the honest selection criterion.
        score = m['balanced_accuracy']
        if score > self.best_score:
            self.best_score = score
            if os.path.exists(BEST):
                shutil.rmtree(BEST)
            self.model.save_pretrained(BEST)
            self.tok.save_pretrained(BEST)
            print('  new best balanced accuracy %.3f -> adapter-best' % score, flush=True)


# ----------------------------------------------------------------- main

def main():
    global BASE, OUT, BEST
    ap = argparse.ArgumentParser()
    ap.add_argument('--rank', type=int, default=32,
                    help='32 = 59.9M params on all 7 projections. Rank 64 needs '
                         '343 MB more optimiser state than a 6 GB card can spare.')
    ap.add_argument('--epochs', type=float, default=40)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--warmup-frac', type=float, default=0.03,
                    help='share of steps spent warming up. HANDOVER 10.3 lists longer '
                         'warmup and a lower LR as untested, and the loss was noisy early.')
    ap.add_argument('--max-len', type=int, default=576,
                    help='longest example is 521 tokens; 576 leaves margin')
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--accum', type=int, default=8)
    ap.add_argument('--vram-fraction', type=float, default=0.82,
                    help='share of VRAM PyTorch may use; the rest is reserved for '
                         'the Windows display driver')
    ap.add_argument('--eval-every', type=int, default=2)
    ap.add_argument('--eval-max-new', type=int, default=64,
                    help='generation cap during scoring. With the premises moved into the '
                         'prompt the answer is a bare verdict grid (<=30 tokens) or a single '
                         'typed cell verdict (<=46), so 64 is ample. It was 330 when the '
                         'model still had to emit 262 tokens of characterisations first.')
    ap.add_argument('--eval-limit', type=int, default=22)
    ap.add_argument('--max-steps', type=int, default=-1,
                    help='stop after N steps. Use --max-steps 20 for a 3-minute '
                         'VRAM smoke test before committing to a long run.')
    ap.add_argument('--verdict-weight', type=float, default=1.0,
                    help='multiplier on the loss at the Y/N verdict token. Only 8.5%% of '
                         'supervised tokens are the verdict (6.1%% on N examples), so the '
                         'default 1.0 spends ~91%% of the gradient on reason prose.')
    ap.add_argument('--seed', type=int, default=42,
                    help='changes LoRA init and data order. Different seeds give genuinely '
                         'independent adapters, which ensemble better than checkpoints of '
                         'one run (0.710 -> 0.718 from correlated checkpoints alone).')
    ap.add_argument('--data', default='',
                    help="suffix of the dataset to train on: '' for the grid datasets, "
                         "'_cell' for the per-cell one")
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--base', default=BASE,
                    help='base model directory (default: the 3B). Point this at '
                         'base-7b to train the converted Qwen2.5-7B-Instruct.')
    ap.add_argument('--out-dir', default='',
                    help='where to write the adapter (default: ./adapter). Set this when '
                         'queueing several runs so they do not overwrite one another.')
    ap.add_argument('--no-final-score', action='store_true',
                    help='skip the generation-based scoring after training. It takes '
                         '30-60 min over the full val set and HANDOVER.md 7.7 says to use '
                         'calibrate.py (logit, ~6 min) instead. The adapter is already '
                         'saved before this runs, so skipping loses nothing.')
    ap.add_argument('--cpu-embed', action='store_true',
                    help='keep the token embedding in system RAM (see CpuEmbedding). '
                         'On the 7B it frees ~1090 MB of VRAM for ~1.6 MB of transfer '
                         'per step, and the 7B does not fit in 6 GB without it.')
    ap.add_argument('--quantize-lm-head', action='store_true',
                    help='also quantise lm_head to nf4 (bitsandbytes skips it by '
                         'default). Frees a further 0.8 GB on the 7B. It is frozen, '
                         'so this only adds quantisation noise to the logits, but '
                         'that noise does land on the verdict token -- do not use it '
                         'on the 3B, which fits without.')
    a = ap.parse_args()

    BASE = a.base
    if a.out_dir:
        OUT = os.path.abspath(a.out_dir)
        BEST = OUT + '-best'
        print('  adapter -> %s' % OUT)

    print('=' * 70)
    print('  GMA cross-consistency fine-tune')
    print('=' * 70)

    if not os.path.isdir(BASE):
        print('  base model missing at', BASE)
        return 1
    for f in ('dataset_train%s.jsonl' % a.data, 'dataset_val%s.jsonl' % a.data):
        if not os.path.exists(os.path.join(HERE, f)):
            print('  %s missing -- run: python build_dataset.py' % f)
            return 1

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print('  GPU: %s  %.0f MB free of %.0f MB'
              % (torch.cuda.get_device_name(0), free / 1e6, total / 1e6))
        # THE important fix. Windows runs the desktop on this same card; if PyTorch
        # takes every byte, the display driver is starved and wedges -- which is what
        # produced the 0x9F and 0x1E bugchecks. Capping the process leaves the driver
        # a reserve it can always draw on, so the worst case becomes a Python OOM
        # (recoverable, checkpoints intact) instead of a machine restart.
        torch.cuda.set_per_process_memory_fraction(a.vram_fraction)
        print('  PyTorch capped at %.0f%% of VRAM (~%.0f MB), leaving ~%.0f MB for the display driver'
              % (a.vram_fraction * 100, total * a.vram_fraction / 1e6,
                 total * (1 - a.vram_fraction) / 1e6))
    else:
        print('  no CUDA device -- this will be unusably slow on CPU')
        return 1

    prevent_sleep()

    train_rows = load_jsonl(os.path.join(HERE, 'dataset_train%s.jsonl' % a.data))
    val_rows = load_jsonl(os.path.join(HERE, 'dataset_val%s.jsonl' % a.data))
    print('  train %d   val %d   epochs %g   rank %d   max_len %d   batch %d x accum %d'
          % (len(train_rows), len(val_rows), a.epochs, a.rank, a.max_len, a.batch, a.accum))

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    global VERDICT_WEIGHT
    VERDICT_WEIGHT = a.verdict_weight
    VERDICT_IDS[0] = tok('VERDICT: Y', add_special_tokens=False)['input_ids'][-1]
    VERDICT_IDS[1] = tok('VERDICT: N', add_special_tokens=False)['input_ids'][-1]
    if VERDICT_WEIGHT != 1.0:
        print('  verdict tokens %s weighted x%.1f in the loss'
              % (VERDICT_IDS, VERDICT_WEIGHT))

    print('  loading base in 4-bit...', flush=True)
    qcfg = dict(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True)
    if a.quantize_lm_head:
        # bitsandbytes keeps lm_head in fp16 by default; [] overrides that
        qcfg['llm_int8_skip_modules'] = []
    device_map = {'': 0}
    if a.cpu_embed:
        device_map = cpu_embed_device_map(BASE)
        qcfg['llm_int8_enable_fp32_cpu_offload'] = True   # required for a mixed map
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        quantization_config=BitsAndBytesConfig(**qcfg),
        dtype=torch.bfloat16, device_map=device_map)
    if a.cpu_embed:
        model = move_embedding_to_cpu(model, BASE)

    # the 7B path cannot afford peft's blanket fp32 upcast -- see prepare_kbit_lean
    if a.cpu_embed:
        model = prepare_kbit_lean(model)
    else:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=a.rank * 2, lora_dropout=0.05,
        bias='none', task_type='CAUSAL_LM', target_modules=TARGET_MODULES))
    model.print_trainable_parameters()

    metrics_cb = MetricsCallback(model, tok, val_rows, a.eval_every, a.eval_limit,
                                 a.eval_max_new)
    guard = VramGuard()

    args = TrainingArguments(
        output_dir=OUT,
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=a.batch,
        gradient_accumulation_steps=a.accum,
        learning_rate=a.lr,
        lr_scheduler_type='cosine',
        warmup_steps=max(1, int(a.warmup_frac * (len(train_rows) / (a.batch * a.accum)) * a.epochs)),
        logging_steps=10,
        save_strategy='epoch',
        save_total_limit=2,             # checkpoints are ~500 MB; keep disk bounded
        # Trainer's own eval loop holds a second batch of activations alongside the
        # training graph, and eval_loss is not interpretable for this task anyway.
        # MetricsCallback scores real verdicts under no_grad instead, which is both
        # cheaper in memory and the number actually worth having.
        eval_strategy='no',
        max_steps=a.max_steps,
        bf16=True,
        gradient_checkpointing=True,
        optim='paged_adamw_8bit',       # paged: an optimiser spike cannot OOM
        report_to=[],
        max_grad_norm=0.3,
        dataloader_num_workers=0,       # Windows: worker processes add failure modes
        seed=a.seed)

    trainer = ChunkedLossTrainer(model=model, args=args,
                                 train_dataset=CcaDataset(train_rows, tok, a.max_len),
                                 data_collator=lambda b: collate(b, tok.pad_token_id),
                                 callbacks=[guard, metrics_cb])

    ckpt = None
    if a.resume and os.path.isdir(OUT):
        cks = [d for d in os.listdir(OUT) if d.startswith('checkpoint-')]
        if cks:
            ckpt = os.path.join(OUT, sorted(cks, key=lambda x: int(x.split('-')[1]))[-1])
            print('  resuming from', ckpt)

    try:
        trainer.train(resume_from_checkpoint=ckpt)
        model.save_pretrained(OUT)
        tok.save_pretrained(OUT)

        if a.no_final_score:
            print('  adapter saved; skipping generation-based scoring '
                  '(--no-final-score) -- use calibrate.py', flush=True)
            return 0

        print('\n' + '=' * 70)
        print('  FULL held-out scoring on the final adapter')
        print('=' * 70, flush=True)
        scorer = score_cells if (val_rows and 'label' in val_rows[0]) else score_verdicts
        final = scorer(model, tok, val_rows, max_new=a.eval_max_new)
        final['epoch'] = 'final'
        metrics_cb.history.append(final)
        metrics_cb._write()

        print('  positive class = N (inconsistent)')
        for k in ('precision', 'recall', 'f1'):
            print('    %-20s %.3f' % (k, final[k]))
        print('  class Y (consistent)')
        for k in ('precision_y', 'recall_y', 'f1_y'):
            print('    %-20s %.3f' % (k, final[k]))
        print('  overall')
        for k in ('macro_f1', 'accuracy', 'balanced_accuracy'):
            print('    %-20s %.3f' % (k, final[k]))
        print('    %-20s TP %d  FP %d  FN %d  TN %d'
              % ('confusion', final['tp'], final['fp'], final['fn'], final['tn']))
        print('    %-20s %d' % ('malformed grids', final['malformed_grids']))
        print('\n  best balanced accuracy during run: %.3f   ->  adapter-best/'
              % metrics_cb.best_score)
        print('  peak VRAM: %.0f MB' % guard.peak_mb)
        print('  metrics: %s' % HISTORY)
        print('\n  next:  python merge_and_convert.py')
    finally:
        allow_sleep()
    return 0


if __name__ == '__main__':
    sys.exit(main())
