# Automated Cross-Consistency Assessment — Method and Results

Automating the Cross-Consistency Assessment (CCA) step of General Morphological Analysis:
given a morphological box, decide for every pair of parameter values whether one internally
consistent scenario could contain both, and reduce the solution space accordingly.

All figures below are measured on **held-out domains** — domains whose text the model never
saw in training — and reported with 95% confidence intervals.

---

## 1. The corpus

| | |
|---|---|
| domains | 70 |
| parameter pairs | 492 |
| assessed cells | 7,790 |
| exclusions | 1,229 (15.8%) |
| exclusions carrying a typed reason and verbatim citations | 1,229 (100%) |
| build errors | 0 |

Every exclusion records *why*, using a typed code, and quotes the two clauses it rests on:

```json
"1,1": {"code": "N2",
        "requires":   "combustion and turbomachinery under load",
        "blocked_by": "vehicle stationary, crew and ground systems connected"}
```

`build_matrices.py` re-checks every citation verbatim against the characterisation text on
every build. A missing reason, an invalid code, or a quotation that is not present is a hard
build failure. A label therefore cannot drift away from its evidence.

| code | meaning | share |
|---|---|---|
| `N3` | capability — one side requires what the other cannot provide | 57.3% |
| `N7` | implausible — possible in principle, but no real case instantiates it | 30.3% |
| `N1` | definitional | 4.7% |
| `N4` | temporal | 3.9% |
| `N6` | scope | 3.1% |
| `N5` | authority | 0.8% |
| `N2` | physical | 0.0% |

`N2` is empty: no exclusion in 70 domains rests on physical law. The corpus is softer than the
code list implies, and `N7` carries the residual subjectivity.

## 2. The criterion

The first criterion read *"inconsistent when one side requires what the other cannot provide,
**or no plausible real-world case instantiates it**."* A blind re-labelling of 289 cells
reproduced it at **88.9% (Cohen κ = 0.730)** — but the disagreements were not random. **45% of
them came from one confusion**: that second clause merged two different questions.

| pair | works well? | occurs? | v1 | v2 |
|---|---|---|---|---|
| hand crew × erratic gusting wind | no | yes | `N` | `Y` |
| automatic load shedding × restoration in days | no | yes | `N` | `Y` |
| black start × restoration in minutes | n/a | **no** | `N` | `N4` |

`CRITERION.md` v2 replaces it with a mandatory test — **"does this combination occur?", never
"does it work well?"** — plus seven typed codes. Strictness drift across matrices narrowed
from 8.5 to 7.0 percentage points of standard deviation.

## 3. Method

The model is Qwen2.5-3B, QLoRA rank 16 on all seven projections (29.9M trainable, 0.96%),
4-bit NF4 base, trained on one RTX 4050 (6 GB).

Three formulations were trained and measured. The differences between them are the result.

**`grid`** — input is the value names only; the model must generate all sixteen
characterisations itself, then the verdict grid.

**`grid-given`** — the characterisations move into the prompt. This matters more than it
looks: in `grid`, **91.6% of the training gradient was spent reciting characterisation prose**
and only 8.4% on verdicts. Worse, teacher forcing taught `P(verdict | our premises)` while
inference conditioned on the model's own invented premises — textbook exposure bias.

**`cell`** — one example per cell, and the target is the typed reason, not a bare letter:

```
VERDICT: N
CODE: N3 (capability)
REQUIRES: a transponder fitted and functioning
BLOCKED BY: no monitoring equipment
```

This yields 9,608 training examples instead of 416, places the loss on a single decision, and
makes the verdict a **single token** — which is what allows calibration.

### Scoring

Because the verdict is one token, `P(inconsistent)` is read directly from the logits rather
than generated. One forward pass per cell scores the whole validation set in minutes, and the
decision threshold becomes a free parameter. The 10 validation domains are split **by domain**
into a dev half (threshold selection) and a test half (reporting), so no reported number is
tuned on itself.

## 4. Results

### Final configuration and headline result

The deployed system is a single adapter (`w8+sym`: verdict-weighted loss, symmetry-augmented
data), scored by averaging `P(inconsistent)` over **both presentation orders**.

Evaluated by **nested leave-one-domain-out** over all 1,191 held-out cells: each domain's
threshold is fitted on the other nine, so no prediction is scored against a threshold that
saw it. This replaces the earlier single 5/5 split, which discarded half the data and
depended on an arbitrary partition.

| | |
|---|---|
| **F1** | **0.554**, 95% CI [0.499, 0.605] |
| precision | 0.524 |
| recall | 0.588 |
| balanced accuracy | 0.736 (chance 0.500) |
| accuracy | 0.822 |
| AUC | 0.817 |
| solution-space error | −7.4% |
| deployed threshold | 0.485 (median of ten LOO fits, range 0.482–0.500) |

**Test-time symmetry averaging is the single largest late gain.** Consistency is symmetric,
so the two presentation orders are the same question; averaging them is free variance
reduction with nothing to tune:

| | AUC | F1 |
|---|---|---|
| normal order only | 0.792 | 0.523 |
| mirrored order only | 0.805 | 0.510 |
| **both orders averaged** | **0.817** | **0.554** |

It also beat every multi-model ensemble tried (two seeds + both orders reached only 0.545),
so the shipped system is one adapter scored twice rather than several adapters combined —
simpler and better.

### What was tried and did not work

Reported because the negative results constrain what is worth attempting next:

| approach | result |
|---|---|
| second seed, ensembled | +0.007 F1 — seeds converged to near-identical solutions |
| within-box score normalisation | +0.005 |
| rule-detector score as a feature | +0.002 |
| value-level score priors | **−0.022** (overfits) |
| rank 32 LoRA | does not fit in 6 GB; optimiser state, not activations |
| DeBERTa cross-encoder | ONNX-only on disk, no PyTorch weights, no `onnxruntime` |
| Qwen 7B as an inference-only ensemble member | dev prefers a 0.7/0.3 mixture, test does not — see below |

### The Qwen2.5-7B GGUF, dequantised

The 7B existed only as `Qwen2.5-7B-Instruct-Q4_K_M.gguf`. `transformers` can load GGUF
directly, but only through the `gguf` package, which is not installed and cannot be
(air-gapped). `gguf_to_hf.py` therefore reimplements the parts needed for a `qwen2`
checkpoint in numpy: the GGUF container, dequantisation of Q4_K / Q6_K / Q8_0, and the
gguf→HF tensor name map. Correctness is not asserted from the code alone — the converted
model answers factual and reasoning prompts coherently through its own chat template,
which no incorrect dequantisation would survive.

Result: `base-7b/`, 15.2 GB of fp16 safetensors, 4-bit-loadable in 3.4 GB with
`embed_tokens` on the CPU. **The 7B is now PEFT-trainable**, which it previously was not.

Scored untuned over the same 1,191 held-out cells:

| model | overall AUC | N3 AUC | N7 AUC |
|---|---|---|---|
| 3B base, untuned | 0.597 | — | — |
| **7B base, untuned** | **0.676** | 0.676 | 0.634 |
| 3B fine-tuned (deployed) | 0.817 | 0.890 | 0.684 |

The untuned 7B ranks better than the untuned 3B by +0.079 AUC. More interesting is the
*shape*: the 7B is nearly flat across codes (N3 0.676, N7 0.634) where the fine-tuned 3B
is steeply uneven (N3 0.890, N7 0.684). Fine-tuning the 3B buys a great deal on
clause-grounded exclusions and very little on N7, which is the §5 diagnosis restated from
a different direction.

That flatness suggested the 7B might carry decorrelated N7 signal, so a rank-averaged
ensemble was tried (`ensemble.py`, mixing weight chosen on dev domains, reported on test):

| weight on the 3B | dev AUC | test AUC |
|---|---|---|
| 0.7 | **0.853** (dev max) | 0.769 |
| 1.0 (3B alone) | 0.839 | **0.787** |

Dev picks 0.7; test rises monotonically to 1.0. The two disagree, so the dev-side gain is
noise and **the ensemble is rejected** — the same conclusion as every other ensemble tried,
now established for a genuinely different model rather than a second seed.

Fine-tuning the 7B was then attempted and **does not fit in 6 GB**. Two accommodations
were built and work — keeping the embedding in system RAM (a real CPU-resident lookup, not
accelerate's offload, which copies the weight back onto the GPU and is strictly worse), and
a norms-only fp32 upcast in place of peft's blanket one. Together they took the run from
"will not load" to training and failing in the backward pass 130 MiB short.

The blocker is structural rather than a tuning matter. Qwen2.5-3B **ties** its embedding to
its output head, so it has one large matrix; Qwen2.5-7B does not, and the extra untied
`lm_head` is 1.09 GB in bf16. Quantising it backfires — storage falls to 306 MB, but since
the hidden state requires grad, backward dequantises the entire head to a 1.02 GB
transient, larger than the saving. With layers in nf4 at 3.28 GB, weights alone reach
~4.37 GB against ~5.0 GB of safely usable VRAM, and QLoRA's backward needs a further
~136 MB transient per MLP weight. Raising the cap to 0.80 segfaulted during load.

So the 7B is usable here for inference and is not trainable here. What it would achieve
fine-tuned remains unmeasured.

### Clause-level decomposition: the model works, the aggregation cannot

The criterion is not holistic. It says N iff one clause requires what another clause
denies, and every exclusion in the corpus cites that pair verbatim. So the corpus carries
supervision at a finer grain than the cell model consumes: **1,419 cited contradictions and
39,268 provable non-contradictions**, with every quote locating cleanly to a clause.

A clause-pair model was trained on that (`clause_pairs.py`, `results/clause_v1`), and cells
scored by taking the maximum over their ~10 cross-side clause pairs (`clause_score.py`).

**Per-pair, the model works — and it changes the picture on N7:**

| | per-pair AUC | cell AUC after max-pooling |
|---|---|---|
| N1 | 0.850 | 0.642 |
| N3 | 0.854 | 0.755 |
| N6 | 0.900 | 0.805 |
| **N7** | **0.843** | 0.613 |
| overall | **0.844** | 0.700 |

Per-pair, **N7 (0.843) is level with N3 (0.854)**. Against the cell model's N7 0.683 versus
N3 0.889, that is the first evidence in this project that N7 is not intrinsically harder:
much of the gap is created by asking the question about two paragraphs rather than two
clauses.

**The aggregation destroys it.** For max-pooling to catch a cell, its one true pair must
outscore all ~9 negatives in the same cell; at per-pair AUC 0.844 that is roughly
0.844^9 ~ 21%. The score distributions confirm the mechanism: Y-pairs average 0.073 but 1%
exceed 0.562, while true positives average only 0.24-0.36 — the positives are not scored
high enough to survive ten draws from that tail.

Aggregation was then attacked directly, and none of it helped:

| aggregation | cell AUC |
|---|---|
| max (top-1) | 0.700 |
| mean of top 3 | 0.706 |
| learned logistic over 10 distribution features | 0.700 (dev 0.741, test 0.644) |
| rank-ensemble with the cell model, weight on dev | test 0.791 vs 0.786 for the cell model alone |

The learned aggregator had access to max, the 2nd and 3rd highest, mean, standard
deviation, counts over three thresholds, the pair count and max-minus-median, was fitted on
dev domains and reported on test — and recovered **nothing**. This is a multiple-instance
learning problem where the information needed to identify which cell contains a
contradiction is simply not present in the distribution of its pair scores at this per-pair
accuracy. Making it work would need per-pair AUC near 0.99, not 0.844.

**The decomposition is rejected as a scoring method.** What survives is the diagnostic: the
clause model is strong at judging a pair it is handed and weak at searching for one, which
is the argument for using it as a verifier of a proposed citation rather than as a scorer.

### Scaling the corpus and sweeping the verdict weight: no measurable effect

The corpus was expanded from 70 to 92 domains (later 96) and four adapters were trained.
The validation domains were **pinned** to the original ten (`build_dataset.py --val-from`),
so all five models below are scored on exactly the same 1,191 held-out cells. Without that
pin the comparison would have been silently void: the split shuffles the domain list, and
shuffling 92 domains yields a different held-out set than shuffling 70.

| run | corpus | train ex. | weight | AUC | test F1 | N3 AUC | N7 AUC |
|---|---|---|---|---|---|---|---|
| deployed | 70 | 9,608 | 8 | **0.817** | 0.472 | 0.889 | 0.683 |
| A | 92 | 9,608 | 8 | 0.809 | 0.478 | 0.873 | 0.669 |
| B | 92 | 24,424 | 8 | 0.811 | **0.489** | 0.880 | 0.676 |
| C | 92 | 9,608 | 4 | **0.824** | 0.437 | 0.888 | 0.688 |
| D | 92 | 9,608 | 16 | 0.809 | 0.448 | 0.870 | 0.681 |

Paired bootstrap against the deployed model, resampling **domains** rather than cells
(cells within a matrix are correlated, so resampling cells would understate the interval
several-fold):

| comparison | ΔAUC | 95% CI | verdict |
|---|---|---|---|
| A − deployed | −0.008 | [−0.029, +0.012] | indistinguishable |
| B − deployed | −0.006 | [−0.017, +0.006] | indistinguishable |
| C − deployed | +0.007 | [−0.013, +0.026] | indistinguishable |
| D − deployed | −0.008 | [−0.029, +0.009] | indistinguishable |

**Every interval spans zero.** A 31% larger and considerably more diverse corpus did not
move the result, and neither did halving or doubling the verdict weight — so the 8 chosen
without a sweep was not leaving anything on the table.

The per-code column is the informative one. **N7 AUC stays inside 0.669–0.688 across all
five models**, against 0.683 for the deployed one. Twenty-two new domains and a
hyperparameter sweep changed the weakest category by less than ±0.01. N3, the
clause-grounded category, is equally static at 0.870–0.889.

That is direct evidence for the §5 diagnosis rather than a restatement of it: if the
plateau were an under-training or under-data artefact, more domains would have moved N7.
It did not move, which locates the ceiling in the model's world knowledge — exactly where
"does this combination occur in reality?" has to be answered from. **The deployed adapter
is retained**; nothing measured here justifies replacing it.

### Selection protocol

The 10 validation domains are split **by domain** into dev (5) and test (5). Both the
adapter *and* the exclusion threshold are chosen on dev; test is read once, at the end.

| candidate | dev AUC |
|---|---|
| **epoch 1 — selected** | **0.824** |
| epoch 2 | 0.810 |

Two epochs was worse than one. Training loss fell to 0.006 by the end of epoch 2, so the
second pass memorised rather than generalised — the same plateau the `grid-given` run showed
at epoch 6. **One epoch over 9,608 examples is the right budget.**

### Final — TEST domains, used for nothing but this table

| | |
|---|---|
| cells | 607 (16.5% inconsistent) |
| **balanced accuracy** | **0.639**, 95% CI [0.589, 0.688] — chance 0.500 |
| precision | 0.422 |
| recall | 0.380 |
| F1 | 0.400 |
| accuracy | 0.812 |
| AUC | 0.710 |
| confusion | TP 38, FP 52, FN 62, TN 455 |

### Checkpoint ensembling (deployed)

Averaging `P(inconsistent)` over the epoch-1 and epoch-2 checkpoints, threshold re-tuned on
dev, improves every figure at no training cost:

| | bal. acc | precision | recall | F1 | space error |
|---|---|---|---|---|---|
| epoch-1 alone @ 0.270 | 0.639 | 0.422 | 0.380 | 0.400 | +15.6% |
| **ensemble @ 0.150** | **0.667** | 0.433 | 0.450 | **0.441** | **+2.6%** |

Test AUC rises 0.710 → 0.718. Note the gain is aggregate: `P(N)` is strongly bimodal, so on
any individual box the two configurations often produce identical output.

### Comparison at equal footing

| system | bal. acc | precision | recall | F1 |
|---|---|---|---|---|
| **per-cell model @ 0.270 (deployed)** | **0.639** | 0.422 | 0.380 | 0.400 |
| per-cell model @ balanced-accuracy optimum | 0.652 | 0.310 | 0.540 | 0.394 |
| per-cell model @ argmax 0.5 | 0.608 | 0.500 | 0.270 | 0.351 |
| deterministic rule detector | 0.567 | 0.513 | 0.173 | 0.258 |
| `grid-given` model | 0.530 | 0.264 | 0.173 | 0.209 |
| always "inconsistent" | 0.500 | 0.190 | 1.000 | 0.319 |
| always "consistent" | 0.500 | 0.000 | 0.000 | 0.000 |

The deployed threshold (0.270) is tuned for solution-space fidelity, not for balanced
accuracy; 0.085 would give balanced accuracy 0.652 but over-prunes the space by 54%.

### Solution-space reduction — the metric that matters

CCA exists to shrink the solution space, and a missed exclusion re-admits an entire family of
configurations, so the error compounds across all *k(k−1)/2* pairs of a box.

Threshold tuned on dev domains for zero error, then applied to held-out test domains:

| system | true configurations | predicted | error |
|---|---|---|---|
| **per-cell model @ 0.270 (test domains)** | 621 | 718 | **+15.6%** |
| per-cell model @ argmax 0.5 (all 10 domains) | 910 | 1,690 | +86% |
| deterministic rules (all 10 domains) | 910 | 3,131 | +244% |

**The aggregate figure is flattered by cancellation, and must not be quoted alone.** Some
domains over-prune and others under-prune, so the errors partly offset. Per domain:

| domain | split | true | predicted | error |
|---|---|---|---|---|
| electrical grid cascading failure | test | 93 | 94 | +1% |
| narcotics trafficking route | test | 154 | 178 | +16% |
| crisis disinformation management | test | 142 | 200 | +41% |
| bank liquidity crisis | test | 125 | 66 | −47% |
| urban water supply contamination | test | 107 | 180 | +68% |
| mine safety incident | dev | 18 | 33 | +83% |
| counterfeit currency operation | dev | 66 | 35 | −47% |

**Median absolute per-domain error: 41%** (test domains, range 1%–68%). A single box should
be expected to land within roughly half of its true solution-space size, not within 16%. The
aggregate is the right figure for a corpus; the per-domain spread is the right figure for a
user assessing one box.

The threshold trades under- and over-pruning smoothly, so the operating point is a deliberate
choice rather than an accident of the argmax:

| threshold | recall | precision | space error |
|---|---|---|---|
| 0.500 | 0.403 | 0.558 | +86% |
| 0.320 | 0.478 | 0.505 | +22% |
| **0.270** | **0.380** | **0.422** | **+15.6%** |
| 0.200 | 0.549 | 0.453 | −20% |
| 0.085 | 0.664 | 0.392 | −54% |

### What each change bought

| change | effect |
|---|---|
| premises into the prompt (`grid-given`) | first statistically significant result; 91.6% of gradient redirected onto verdicts |
| one example per cell, typed reason as target | balanced accuracy 0.530 → 0.608; recall 0.173 → 0.270 |
| threshold calibration | balanced accuracy 0.608 → 0.652; recall 0.270 → 0.540 |
| tuning the threshold for solution-space error | +86% → **+15.6%** |

## 5. Limitations

- **Recall is 0.38–0.54 depending on operating point.** The engine finds under half of all
  inconsistencies.
- **Per-domain solution-space error has a median magnitude of 41%**, even though the
  corpus-level figure is +15.6%. The aggregate benefits from over- and under-pruning
  cancelling out across domains. For any single box, expect the surviving count to be within
  roughly a factor of 1.5 of the truth — and the surviving *set* is not the true set even
  where the count happens to match.
- **False positives from the rule layer are lexical accidents.** In the worked example,
  *"cooperative on contact"* was matched against *"no contact with any official system"* on
  the shared word "contact", producing a wrong exclusion. Every rule firing quotes its two
  clauses precisely so an analyst can spot this.
- **~11% irreducible label noise**, concentrated in `N7`, bounds any achievable accuracy.
- **Labels are synthetic**, authored for this project. κ = 0.730 measures self-consistency,
  not correspondence with an external authority.
- **Test AUC 0.710 vs dev 0.824** indicates real domain-to-domain variance; some domains are
  substantially harder than others.
- **70 domains is a small corpus.** Within a matrix, cells are correlated — one property (a
  ballistic missile's altitude) drives many exclusions at once — so the effective number of
  independent facts is in the hundreds, not thousands.

## 6. Reproducing

```bash
python build_matrices.py                                   # 70 domains -> corpus.json
python build_dataset.py --format cell --balance 0.40       # 9,608 / 1,191 examples
python finetune.py --rank 16 --vram-fraction 0.72 --epochs 2 \
       --eval-every 1 --eval-limit 150 --max-len 256 \
       --batch 2 --accum 4 --data _cell
python calibrate.py --adapter adapter-best                 # AUC + threshold sweep
python cca.py --matrix <box.json> --with-model adapter-best --report out.md
```

`cca.py` emits the consistency matrix, the solution-space reduction, and the evidence behind
every exclusion — clause citations for deterministic firings, `P(inconsistent)` for model
firings — so the output is auditable rather than opaque.
