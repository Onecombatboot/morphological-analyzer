# Automated Cross-Consistency Assessment for General Morphological Analysis

A fine-tuned language model that decides whether two options from different
facets of a scenario can coexist — and a Java application that runs it on the
CPU, offline, with no Python at runtime.

Built during an internship at the **Institute for Systems Studies & Analyses
(ISSA), Defence Research & Development Organisation (DRDO)**.

---

## The problem

General Morphological Analysis explores problems that cannot be reduced to
equations. An analyst decomposes a problem into parameters, gives each a few
possible values, and treats every combination as a candidate configuration. Five
parameters of four values each is 1,024 configurations.

Most are nonsense. A maritime interdiction cannot be run by a private security
contractor under territorial-waters enforcement, because the contractor has no
sovereign authority and that authority is exactly what the legal basis requires.

Zwicky's method removes these through **Cross-Consistency Assessment**: examine
every pair of values from different parameters, mark the pairs that cannot
coexist, and discard any configuration containing a marked pair. The survivors
are the solution space, typically 1–10% of the original.

CCA is where the method becomes expensive. The number of pairs grows with the
square of the problem size — a five-parameter box needs 160 expert judgements,
and a realistic one needs several hundred. **This project automates that
judgement.**

---

## Results

Measured on **1,191 held-out cells from 10 domains** the model never trained on,
with the decision threshold fitted leave-one-domain-out.

| | AUC | F1 (full coverage) | F1 @ 50% coverage |
|---|---|---|---|
| **Fine-tuned model (PyTorch, 4-bit NF4)** | **0.878** | 0.589 | 0.805 |
| **Deployed model (ONNX, 4-bit block-wise)** | **0.876** | 0.583 | **0.811** |
| Second human analyst, same written criterion | — | — | 0.807 |
| Chance | 0.500 | — | — |

**At its operating point the system reproduces the corpus judgements at a rate
comparable to an independent human analyst** working from the same criterion
(blind re-labelling of 289 cells: 88.9% agreement, Cohen's κ = 0.730).

Compression to 4-bit cost essentially nothing: **Pearson r = 0.94** and 96.6%
verdict agreement against the PyTorch reference across all 1,191 cells.

### Why 50% coverage, and not 100%

The system decides the half of the matrix it is confident about and refers the
rest. This is **selective classification** (Geifman & El-Yaniv), and it is the
appropriate mode for decision support: an analyst doing CCA by hand must answer
every cell, an assistant need not.

| Coverage | 100% | 90% | 80% | 70% | 60% | **50%** | 40% |
|---|---|---|---|---|---|---|---|
| F1 | 0.583 | 0.582 | 0.630 | 0.693 | 0.754 | **0.811** | 0.827 |

The curve is monotone, which is the signature of a confidence estimate that
actually tracks correctness — the cells it is least sure of are the ones it is
most often wrong about.

---

## What made it work

Three findings did most of the work, and all three were counter-intuitive.

**1. Supply the premises; do not let the model invent them.**
An early formulation gave the model only the two value *names* and asked it to
generate characterisations before judging. It scored at chance. Under teacher
forcing the model learns to predict verdicts conditioned on *reference*
premises; at inference it conditions on its own generated text, and ~250 tokens
of drift separate the two. Supplying each value's REQUIRES and PROVIDES in the
prompt removed the mismatch and produced the single largest improvement in the
project.

**2. Symmetry averaging is load-bearing, not a refinement.**
Cross-consistency is symmetric, so each cell is scored in both presentation
orders and averaged. Dropping it to halve inference cost collapses the deployed
model's selective-prediction curve to **F1 0.000 at 50% coverage** — ranking
survives (AUC 0.82) but the confidence signal *inverts*, so the system becomes
least reliable exactly where it reports most certainty.

**3. Fewer bits was more accurate.**
Block-wise **INT4** beat per-channel **INT8** on fidelity — 0.876 AUC against
0.853, and 0.811 against 0.716 at the operating point. INT8 gives one scale to a
2048-wide output channel; INT4 gives one scale per 32 weights. For transformer
weights, scale granularity dominates bit depth, because the damage comes from
outliers stretching a shared scale.

---

## Method

**Corpus.** 96 morphological domains, 648 parameter pairs, **10,286 assessed
cells**, 1,419 exclusions. Every exclusion carries a typed reason code (N1–N6:
definitional, physical, capability, temporal, authority, scope) and **two
verbatim citations** — the requirement that is unmet, and the clause that denies
it. Citations are re-verified against source text on every build; a missing
reason or an absent quote is a hard build failure, so an unjustified exclusion
cannot be expressed in the format at all.

Domains span far beyond defence deliberately — organ transplant allocation,
patent litigation, coastal erosion, microfinance — so the only consistently
predictive feature is the structure of the requires/provides conflict itself,
not the vocabulary of any one field.

**Criterion.** A pair is inconsistent **if and only if no possible configuration
instantiates both values.** Ineffective is not inconsistent. An earlier
formulation also admitted "no plausible case would instantiate it", which
silently merged *can this occur* with *would this work well* — two different
questions, and the dominant source of annotator disagreement.

**Model.** Qwen2.5-3B adapted with **QLoRA**: base weights quantised to 4-bit
NF4 and frozen, rank-32 adapters on all seven attention and MLP projections.
**59.9M trainable parameters, 1.90% of the model**, trained within a **6 GB VRAM
budget**.

Two departures from the default recipe were necessary rather than optional:

- **Verdict-token weighting.** The verdict is one token in a target that also
  carries reason prose. Under uniform cross-entropy most of the gradient trains
  explanation rather than the decision being scored.
- **Chunked loss over supervised positions only.** The full logits tensor
  (batch × sequence × 151,936) exceeds the memory budget outright. Computing
  loss in chunks, never materialising logits at masked positions, is what
  permitted rank 32 at all.

**Scoring.** The verdict is a single token, so it is read directly from the
logits — one forward pass, softmax over the two verdict token ids, no
generation. This makes the operating point an explicit parameter rather than an
emergent property of decoding.

**Evaluation.** Split **by domain, never by cell** — cells within a box share
characterisations, so a cell-level split leaks premises and inflates every
figure. Thresholds fitted leave-one-domain-out. Configurations compared with a
**domain-level bootstrap**; the leading runs were separated by less than the
confidence interval, and the repository reports that rather than claiming a
distinction the data does not support.

---

## Deployment

The trained adapter was merged, exported to ONNX, quantised to 4-bit and
embedded in the Java application. **The deployed system needs a JRE 21 and
nothing else** — no Python, no pip, no network, no GPU.

Four optimisations, each measured:

| Change | Effect |
|---|---|
| Trimmed the output projection to the two verdict columns and the final position | removed a 350×2048×151,936 matmul and a 200 MB tensor **per call**; mathematically exact |
| Promoted the fp16 graph to fp32 compute, leaving INT4 weights untouched | 7.47 → **3.09 s/pass** (fp16 was running through CPU emulation) |
| ONNX Runtime 1.16 → 1.22 | 5.24 → **4.23 s/pass**, bit-identical scores |
| Read logits from the native `FloatBuffer` | avoided a `float[1][n][151936]` allocation per cell |

Verified byte-for-byte across the language boundary: the prompts Java constructs
are **identical to Python's** (0 mismatches), tokenise identically, and resolve
the same verdict token ids. The service refuses to start if the verdict is not
exactly one token past the prefix — scoring confidently from the wrong logit
position is worse than not running.

---

## Repository layout

```
training/
  matrices_v2/        the 96-domain corpus
  authoring/          corpus authoring toolchain, validates as it writes
  build_matrices.py   compiles sources, re-verifies every citation
  build_dataset.py    domain-level split, symmetric presentation, class balancing
  finetune.py         QLoRA trainer: chunked loss, verdict weighting, VRAM guard
  calibrate.py        logit scoring, leave-one-domain-out thresholds
  evaluate.py         held-out evaluation and the coverage curve
  bootstrap_compare.py  domain-level bootstrap intervals
  onnx_export.py      merge adapter, export, quantise
  quant_int4.py       block-wise INT4
  onnx_trim_head.py   output-projection surgery
  onnx_score.py       parity scoring against the PyTorch reference

morphological-analyzer/    Spring Boot application
  src/main/java/.../OnnxCcaService.java    the model, in process
  src/main/java/.../LlmCcaService.java     llama.cpp baseline engine
  src/main/java/.../ConsistencyService.java  verdict precedence

morphological-frontend/    Angular interface
```

---

## Weights

Not in this repository. **[→ Hugging Face model card](https://huggingface.co/)**
*(link to be filled in after upload)*

Base models are referenced, not mirrored: `Qwen/Qwen2.5-3B`,
`Qwen/Qwen2.5-7B-Instruct`, `sentence-transformers/all-MiniLM-L6-v2`.

---

## Reproducing

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python training/build_matrices.py       # compile + verify the corpus
python training/build_dataset.py --format cell --val-from dataset_val_r32d.jsonl
python training/finetune.py --rank 32 --out-dir results/cell_r32d
python training/calibrate.py --adapter results/cell_r32d
python training/evaluate.py --adapter results/cell_r32d
```

Training takes about two hours on a 6 GB GPU. See `SETUP.md` for running the
deployed application, and `ARCHITECTURE.md` for how the pieces fit.

---

## Honest limitations

- **~4 seconds per cell** on CPU. A 96-cell box takes 6–12 minutes: 192 forward
  passes of a 3B model. No configuration measured brings this near a minute.
- **The interface does not drive the fine-tuned model.** It collects value names
  only; this model needs REQUIRES and PROVIDES. The interface is served by a
  llama.cpp baseline, and the fine-tuned model is reached by API.
- **Full-coverage F1 is 0.583.** The headline 0.811 is at 50% coverage. Both
  numbers are reported here because one without the other is not interpretable.
- **The remaining errors are world-knowledge errors.** The system is strong
  where the conflict is visible in the supplied clauses, weaker where the answer
  depends on whether a combination occurs in practice. That is a limit of the
  base model's knowledge, and it is where the system refers to the analyst.

---

## References

Zwicky (1969) · Ritchey, *Wicked Problems – Social Messes* (2011) ·
Hu et al., *LoRA* (ICLR 2022) · Dettmers et al., *QLoRA* (NeurIPS 2023) ·
Geifman & El-Yaniv, *Selective Classification* (NeurIPS 2017)

## License

MIT — see `LICENSE`.
