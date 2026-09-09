# ARCHITECTURE

Automated Cross-Consistency Assessment for General Morphological Analysis.

This document describes every component of the delivered system: what each file
does, what the model is, how it was trained, what the dataset contains, and how the
pieces call one another. It is written to be read by someone who has never seen the
project.

---

## PART 1 — WHAT THE SYSTEM DOES

### 1.1 The problem in one paragraph

A morphological box is a set of parameters, each with a few possible values. Every
combination of one value per parameter is a candidate configuration. Most are
impossible. Cross-Consistency Assessment (CCA) examines every pair of values drawn
from two different parameters and marks the pairs that cannot coexist; any
configuration containing a marked pair is eliminated. Four parameters of four
values gives 1,024 configurations and requires 96 pairwise judgements. The
judgements are the expensive part, and this system makes them.

### 1.2 The three-process runtime

```
   Browser
      |  HTTP, same origin
      v
   Spring Boot application            port 8080 (any port; see 3.4)
      |  serves the Angular interface from src/main/resources/static
      |  exposes /api/*
      |
      |  both models run INSIDE this process, on the CPU
      |
      +--> llama.cpp via JNI      Qwen2.5-7B GGUF
      |      drives the interface's full analysis; works from value names
      |
      +--> ONNX Runtime           fine-tuned Qwen2.5-3B, 4-bit
             drives /api/matrix/cca-score and /api/matrix/cca-model;
             needs each value's REQUIRES and PROVIDES
```

One process owns everything: the box, the grid, the analyst's standing rulings,
the solution-space enumeration and both models. There is no second process, no
local HTTP hop and no Python.

The earlier design put the fine-tuned model behind a Python service on port 8000.
That is gone. The adapter was merged into the base weights, exported to ONNX and
quantised to 4-bit block-wise, and ONNX Runtime loads it in the JVM. The
arithmetic is unchanged -- one forward pass, softmax over the two verdict token
logits, both presentation orders averaged -- and the held-out figures are
preserved: AUC 0.8758 against 0.8785, and F1 0.8113 against 0.8049 at 50%
coverage.

### 1.3 Why the model is served from Python and not embedded in Java

The Java application already runs a language model — `LlmCcaService` loads a GGUF
file through llama.cpp. The obvious thing would be to put the fine-tuned model
there too. It cannot go there as-is:

- What the training produced is a **PEFT LoRA adapter on Qwen2.5-3B in HuggingFace
  safetensors format**.
- llama.cpp loads **GGUF**, a different container with its own quantisation
  formats.
- There is no GGUF **writer** available on the build machine. The `gguf` Python
  package is not installed and cannot be (the machine is air-gapped). This is the
  same gap that forced `gguf_to_hf.py` to be written by hand in order to read the
  7B GGUF in the first place.

Converting would require merging the adapter into the base weights, writing a GGUF
encoder, and writing a Q4_K quantiser from scratch — with no reference
implementation on the machine to check it against. Serving the model from the
runtime it already works in produces the same answers for a fraction of the work,
and has one further benefit: the prompt exists in exactly one place, imported from
the module that generated the training data, so the deployed path cannot silently
drift from the evaluated one.

---

## PART 2 — THE MODEL

### 2.1 Base model

**Qwen2.5-3B**, 36 transformer layers, hidden size 2048, 16 attention heads with 2
key-value heads (grouped-query attention), SwiGLU feed-forward of width 11008,
RMSNorm, rotary position embeddings with θ = 1,000,000, vocabulary 151,936 with
tied input and output embeddings.

At load time the base weights are quantised to **4-bit NF4** with double
quantisation, compute dtype bfloat16. The base weights are frozen throughout.

### 2.2 The adaptation

**QLoRA**, rank 32. For a frozen weight W (d × k), LoRA adds a trainable low-rank
update:

```
    h = Wx + (alpha / r) * B A x        A: r × k,  B: d × r,  r = 32, alpha = 64
```

A is initialised from a Gaussian, B at zero, so the adapted model starts exactly
equal to the base model. Adapters are attached to **all seven projections** in
every layer:

```
    q_proj  k_proj  v_proj  o_proj        (attention)
    gate_proj  up_proj  down_proj         (feed-forward)
```

| | |
|---|---|
| Trainable parameters | 59,867,136 |
| Share of total | 1.90% |
| Dropout | 0.05 |
| Deployed adapter | `training/results/cell_r32d/` |

Attaching to all seven rather than to the attention projections alone matters here:
the judgement is semantic rather than positional, and the feed-forward blocks are
where most of that representation lives.

### 2.3 How a verdict is produced

The model is not asked to generate free text at inference. The prompt ends with the
literal token sequence `VERDICT:` and the decision is read directly from the
logits at that position:

```
    P(inconsistent) = softmax([ logit(' Y'), logit(' N') ])[1]
```

Two details are load-bearing and easy to get wrong:

- The prefix is `VERDICT:` with **no trailing space**. The tokeniser merges the
  space into the verdict token itself — `' Y'` is token 809 and `' N'` is token
  451. A prefix ending in a space produces a standalone space token and then scores
  `'Y'`/`'N'` without it, which are different token ids, and every probability
  becomes meaningless.
- Padding is **left**-side, so that the final position of every sequence in a batch
  is the real last token rather than padding.

Reading two logits instead of generating text is roughly an order of magnitude
faster and, more usefully, turns the decision threshold into an explicit parameter
rather than an artefact of the argmax.

### 2.4 Test-time symmetry averaging

Cross-consistency is symmetric: *consistent(A, B)* is the same claim as
*consistent(B, A)*. The model has no architectural guarantee of that, so every cell
is scored in both presentation orders and the two probabilities averaged. This
costs one extra forward pass and was measured as the single largest late-stage
improvement in the project.

### 2.5 The decision threshold

`P(inconsistent) >= 0.485` marks a cell inconsistent. The threshold is not tuned on
the data it is reported on: for each held-out domain it is fitted on the other
nine, so no domain influences its own decisions.

---

## PART 3 — THE DATASET

The corpus is the most valuable artefact in the project, and the part that is
genuinely original.

### 3.1 What it contains

| | |
|---|---|
| Domains | 96 |
| Parameter pairs | 648 |
| Cells assessed | 10,286 |
| Exclusions recorded | 1,419 |
| Exclusions with a typed reason and two verbatim citations | 100% |
| Build errors | 0 |

Each domain is a complete morphological box: four parameters of four values, a
characterisation for every value, and a verdict for every cell.

Domains deliberately span far beyond defence, so that the model learns the
*relation* of inconsistency rather than the vocabulary of one field. Examples:
maritime interdiction, electrical grid cascading failure, air defence engagement,
organ transplant allocation, air traffic control sector operations, criminal trial
proceedings, wildlife reintroduction, nuclear waste repository siting, professional
sports doping control, archaeological rescue excavation, patent litigation, deep
sea mineral extraction, coastal erosion management, railway timetable recast,
microfinance lending, refugee status determination, and others across 31 subject
groups.

### 3.2 The characterisation — why every value has two fields

Every value carries:

- **requires** — what a scenario must already provide for this value to be present
- **provides** — what this value contributes or entails, including what it rules out

Example:

```json
"National coast guard": {
  "requires": "jurisdiction within own maritime zone",
  "provides": "police powers, arrest authority, domestic legal standing"
}
```

These are not decoration. They are the premises the model reasons over, and
supplying them was the change that made the task learnable at all (see 4.2).

### 3.3 The criterion

The wording of the criterion determines the labels, and therefore determines
everything measured against them. It went through one substantial revision.

**Version 1** admitted a pair as inconsistent where one side required something the
other could not provide, **or** where no plausible real-world case would
instantiate it. A blind re-labelling of 289 cells reproduced version 1 at 88.9%
agreement, Cohen's κ = 0.730 — but a disproportionate share of the disagreements
traced to that second clause, which silently merged two different questions: *can
this occur?* and *would this work well?*

**Version 2**, used for the delivered system:

> A pair is inconsistent **if and only if no possible configuration instantiates
> both**. Ineffectiveness is not inconsistency.
>
> Mandatory test: *"does this combination occur?"* — never *"does it work well?"*

Every exclusion carries a typed reason code naming the kind of impossibility:

| Code | Meaning | Share |
|---|---|---|
| N1 | Definitional — one is by definition outside the other's category | 4.7% |
| N2 | Physical — prevented by physical law | 0% (no exclusion rests on it) |
| N3 | Capability — one requires what the other cannot provide | 57.3% |
| N4 | Temporal — the two cannot hold at the same time | 3.9% |
| N5 | Authority — the actor lacks the standing required | 0.8% |
| N6 | Scope — the two operate on different objects | 3.1% |
| N7 | Implausible — possible in principle, no real case instantiates it | 30.3% |

### 3.4 Evidence as a structural constraint

Every exclusion must quote the **exact clause from each side** that produces the
conflict — the requirement that is unmet, and the clause that denies it:

```json
{"row": "Private security contractor",
 "col": "Territorial waters enforcement",
 "code": "N5",
 "requires":   "unambiguous domestic authority",
 "blocked_by": "no sovereign authority"}
```

Two design decisions make unjustified labels impossible to express:

1. **Only exclusions are listed.** Every cell not listed is consistent by
   construction. There is no syntax for "inconsistent, no reason given".
2. **Citations are re-verified on every build.** `build_matrices.py` checks that
   both quotations appear verbatim in the relevant characterisations. A missing
   reason, an invalid code, or an absent quotation is a hard build failure, not a
   warning.

### 3.5 The N7 decision in the delivered model

N7 ("possible in principle, but no real case instantiates it") is a world-knowledge
judgement rather than a reading of the text in the prompt. It was also where the
inter-annotator disagreements concentrated.

The delivered model is trained on the **clause-grounded criterion**: N7 cells are
removed from training, and at evaluation they count as consistent. The system
therefore excludes a pair when a clause makes it impossible, and does not attempt
to exclude on plausibility. This is a narrower and sharper claim than the original
criterion, and the model's per-code performance is uniform across the codes it does
cover (N1 0.861, N3 0.888, N4 0.881, N6 0.868 AUC).

### 3.6 From corpus to training examples

`build_dataset.py` compiles the corpus into one example per cell.

**Input:**

```
Facet A — SURVEILLANCE
Satellite vessel monitoring | REQUIRES: a transponder fitted and functioning | PROVIDES: ...

Facet B — VESSEL
Artisanal small boat | REQUIRES: minimal capital, day trips | PROVIDES: no monitoring equipment, ...

Can one scenario contain both?
```

**Target:**

```
VERDICT: N
CODE: N3 (capability)
REQUIRES: a transponder fitted and functioning
BLOCKED BY: no monitoring equipment
```

Three transformations are applied:

- **Symmetric presentation** (`--swap`): each cell appears in both orders, so no
  order-sensitivity can be learned.
- **Class balancing** (`--balance 0.40`): exclusions are the minority class and are
  oversampled so they carry adequate weight in the gradient.
- **Domain-level split** (`--val-from`): the split is by domain, never by cell.
  Cells within one box share characterisations, so a cell-level split would leak
  and inflate every reported figure. Ten domains, 1,191 cells, are held out.

---

## PART 4 — HOW THE MODEL WAS TRAINED

### 4.1 The recipe

| | |
|---|---|
| Base | Qwen2.5-3B, NF4 4-bit, frozen |
| Adapter | LoRA rank 32, alpha 64, dropout 0.05, all 7 projections |
| Optimiser | paged AdamW, 8-bit |
| Learning rate | 1e-4, cosine schedule, 3% warmup |
| Epochs | 1 |
| Batch | 2 × gradient accumulation 4 (effective 8) |
| Sequence length | 256 |
| Gradient checkpointing | on |
| VRAM cap | 78% of 6 GB |

**One epoch, deliberately.** A second epoch was measured as worse every time it was
tried. The corpus is small enough that the model begins memorising specific
characterisation wording rather than the relation between clauses.

### 4.2 The decision that made the task learnable

An earlier formulation gave the model only the **value names** and required it to
generate the characterisations itself before emitting verdicts. It performed at
chance, for a reason worth stating precisely:

Under teacher forcing the model learns `P(verdict | reference characterisations)`.
At inference on an unseen domain it instead conditions on **its own generated**
characterisations. The two distributions differ, and the mismatch compounds because
roughly 250 tokens of self-generated premises precede the first verdict token. This
is textbook exposure bias, and it was severe.

Moving the characterisations into the **input** fixes it: the premises become fixed
and shared between training and inference, and the model is asked only the question
that is actually being scored. This produced the first statistically significant
result in the project and determines the input format still used.

### 4.3 Verdict-token loss weighting

The target contains the verdict, the code, and two citations. The verdict is one
token out of roughly forty. Without intervention, about 91% of the gradient trains
the model to reproduce explanation prose and 8.5% trains the decision being scored.

The loss at the verdict token is therefore multiplied by 8, and the weights are
renormalised to mean 1 so that only the *balance* changes and the overall gradient
scale does not. The renormalisation is necessary: without it the loss grows about
1.6×, gradient norm rises from ~1.5 to ~15, the trainer's clipping at 1.0 fires on
every step, and the effective learning rate is silently throttled for the whole
model.

### 4.4 Chunked loss — why the model fits in 6 GB

The vocabulary is 151,936 tokens. A 256-token sequence produces a logits tensor of
256 × 151,936. Between the bfloat16 logits, the float32 upcast the loss requires,
the softmax temporary and the gradient, that single tensor costs roughly 1.1 GB —
more than the model weights, adapter and optimiser state combined. Gradient
checkpointing does not help, because logits are the final output rather than an
intermediate activation.

`ChunkedLossTrainer` in `finetune.py` removes it in two steps:

1. `logits_to_keep=1` stops the model running its output head over the sequence at
   all; only hidden states come back.
2. The loss is computed only at positions the collator did not mask — the assistant
   answer, which is the only place gradient comes from anyway — in chunks of 64,
   each wrapped in a `torch.utils.checkpoint` so its logits are freed immediately
   and recomputed during the backward pass.

Peak cost becomes one chunk of 64 × 151,936, about 200 MB. This is what allows rank
32 to be trained at all on a 6 GB card.

### 4.5 Evaluation protocol

- 1,191 cells from 10 domains never trained on.
- Positive class is **inconsistent** — the judgement the system exists to make, and
  the minority class at 13.3%.
- Threshold fitted leave-one-domain-out.
- AUC reported as a threshold-free measure of ranking quality.
- Confidence intervals from a **domain-level** bootstrap. Cells inside a matrix are
  correlated — one property of a scenario drives many exclusions at once — so
  resampling cells rather than domains gives an interval several times too narrow.

### 4.6 Results

| | |
|---|---|
| AUC | 0.878 |
| F1 at the deployed operating point (50% coverage) | **0.828** |
| Precision / recall there | 0.800 / 0.857 |
| Accuracy there | 0.975 |
| F1 at full coverage | 0.583 |
| Human vs human on the same task | 0.807 |

The system operates with a confidence threshold: cells it is confident about are
decided automatically, the remainder are referred to the analyst. Accuracy declines
smoothly and monotonically as confidence falls — the worst decile is still 66.7%
correct — which is what makes the operating point a deliberate choice rather than
an arbitrary cut.

**Scope of these figures.** They were measured on characterisations written by hand
in the corpus. In the live application, characterisations for a new box are
generated by the model. That end-to-end condition has not been measured. The
figures describe the assessment model given good characterisations, not the
complete names-to-verdicts pipeline.

---

## PART 5 — FILE BY FILE

### 5.1 `training/` — corpus, training, evaluation, serving

#### Core pipeline

| File | Purpose |
|---|---|
| `build_matrices.py` | Compiles `authoring/src/*.json` into `matrices_v2/corpus.json`. Re-verifies all 2,458 citations verbatim; any failure aborts the build. |
| `build_dataset.py` | Corpus → training/validation JSONL. Owns the prompt format (`SYSTEM_CELL`, `cell_examples`), the domain split (`--val-from`), class balancing (`--balance`), symmetric presentation (`--swap`), criterion narrowing (`--exclude-codes`), and clause-order augmentation (`--shuffle-clauses`). |
| `finetune.py` | The QLoRA trainer. Contains `ChunkedLossTrainer` (4.4), verdict weighting (4.3), the VRAM guard, `CpuEmbedding` for keeping a large embedding off the GPU, and `prepare_kbit_lean`. Flags: `--rank --lr --warmup-frac --batch --accum --max-len --data --verdict-weight --out-dir --vram-fraction --no-final-score`. |
| `calibrate.py` | Logit scoring and threshold calibration. `load_model()` and `verdict_token_ids()` are imported by nearly everything else, so the token ids are defined once. |
| `serve_cca.py` | The **former** deployed service, kept as the record of how scoring worked. `OnnxCcaService.java` replaces it and reproduces its arithmetic exactly. Not used at runtime and not started by any launcher. |
| `cca.py` | Standalone command-line assessment of one box: verdict grids, solution space, evidence report. |
| `rule_detector.py` | Deterministic clause matcher. Low recall, but when it fires the exclusion carries two exact clauses rather than a probability. |

#### Evaluation and analysis

| File | Purpose |
|---|---|
| `prompt_ensemble.py` | Scores a validation set, optionally under several system prompts; `--keep-system` uses each row's own. Writes score dumps that everything downstream reads. |
| `by_code.py` | Per-exclusion-code breakdown: recall, mean probability, and AUC per code against all consistent cells. Also provides `code_map()` and `auc()` used elsewhere. |
| `n7_variants.py` | Evaluates a score dump under three criteria: as-is, N7 dropped, N7 relabelled consistent. Uses leave-one-domain-out thresholds. |
| `bootstrap_compare.py` | Domain-level paired bootstrap between two score dumps. The correct significance test for this data. |
| `final_report.py` | Consolidates every score dump into one comparable table, plus the selective-prediction curve; writes `FINAL_RESULTS.md`. |
| `make_folds.py`, `pool_folds.py` | K-fold cross-validation over domains, and pooling of out-of-fold predictions. |
| `adjudicate_make.py`, `adjudicate_score.py` | Stratified blind re-adjudication harness, and the estimator that reweights each stratum by its population. |
| `audit_make_worksheet.py`, `audit_score.py` | The original blind re-labelling exercise that produced κ = 0.730. |
| `verify_citations.py`, `validate_matrices.py` | Standalone corpus validators. |

#### Research code, retained for the record

| File | Purpose |
|---|---|
| `gguf_to_hf.py` | Hand-written GGUF reader and Q4_K/Q6_K/Q8_0 dequantiser. Converts a GGUF checkpoint to HuggingFace safetensors. Written because the `gguf` package is unavailable. |
| `score_7b.py` | Inference-only scoring with the 7B, with embedding and output head on the CPU. |
| `clause_pairs.py`, `clause_score.py`, `train_mil.py`, `aggregate_pairs.py` | Clause-level decomposition and multiple-instance learning experiments. |
| `probe.py` | Linear probes on hidden states from several layers. |
| `build_retrieval.py` | Retrieval-augmented prompting with precedents from other domains. |
| `hard_cells.py` | Hard-example mining at cell level. |
| `grid_context.py` | Second-stage model using row/column context within a grid. |

#### Authoring

| File | Purpose |
|---|---|
| `authoring/new.py` | Creates corpus sources from a compact text specification, validating citations, parameter ordering and duplicates as it writes. |
| `authoring/set.py`, `authoring/view.py` | Edit and inspect existing domains. |
| `authoring/specs/*.txt` | The compact specifications the corpus was authored from. |
| `authoring/src/*.json` | The corpus sources, one file per domain. |

#### Data and results

| Path | Contents |
|---|---|
| `matrices_v2/corpus.json` | The compiled corpus. |
| `dataset_train_*.jsonl`, `dataset_val_*.jsonl` | Compiled training and validation sets. Suffix identifies the recipe. |
| `results/cell_r32d/` | **The deployed adapter.** |
| `results/*` | Other adapters from the experiment series. |
| `scores_*.json` | Per-cell score dumps, reusable for analysis without a GPU. |
| `base-model/` | Qwen2.5-3B in HuggingFace format. Required at runtime. |

#### Documents

| File | Contents |
|---|---|
| `DEPLOYED.md` | Authoritative statement of what runs: adapter, threshold, ports, precedence. |
| `HEADLINE_RESULTS.md` | The result tables. |
| `CRITERION.md` | The labelling criterion in full. |
| `RESULTS.md`, `HANDOVER.md` | Development record, including approaches that did not work. |

### 5.2 `morphological-analyzer/` — the Spring Boot application

| File | Purpose |
|---|---|
| `MorphologicalAnalyzerApplication.java` | Spring entry point. |
| `AiController.java` | All HTTP endpoints under `/api`. `runFullAnalysis` builds the grid, selects the engine, applies precedence, and enumerates the solution space. |
| `FineTunedCcaService.java` | **The bridge to the fine-tuned model.** `assessMatrix()` returns verdicts in the same shape as the GGUF service, so it is a drop-in replacement. Also `post()`, `health()`. |
| `LlmCcaService.java` | The GGUF baseline engine via llama.cpp. Retained as a fallback. |
| `ConsistencyService.java` | Applies precedence to produce one verdict per cell: analyst override → calibration head → model → NLI fallback. |
| `OverrideStore.java` | Analyst standing rulings, persisted as plain JSON. |
| `CalibrationService.java` | Optional logistic head fitted on recorded rulings. |
| `OnnxNliService.java`, `OnnxEmbeddingService.java` | The original NLI and embedding models. Superseded for assessment; the embedding service still supports scenario matching. |
| `PairKey.java` | Canonical order-independent key for a value pair. Includes parameter names, escapes the separator, and sorts the halves so (A,B) and (B,A) collide deliberately. |
| `src/main/resources/application.properties` | All configuration. |
| `src/main/resources/static/` | The built Angular interface, served by Spring on the same origin as the API. |

#### Key endpoints

| Endpoint | Purpose |
|---|---|
| `POST /api/matrix/full-analysis` | **What the interface calls.** Names in, complete grid and solution space out. |
| `POST /api/matrix/cca-model` | Direct assessment when characterisations are already available. |
| `POST /api/matrix/cca-score` | Score individual pairs. |
| `GET /api/status` | Which engine is active, and whether each model loaded. |
| `GET/POST /api/overrides*` | Analyst rulings. |
| `POST /api/matrix/match-scenario` | Scenario matching over the solution space. |

#### Verdict precedence

```
1. Analyst override      recorded ruling for this pair, if any
2. llama.cpp (GGUF)      what the interface's full analysis uses
3. Calibration head      if trained and enabled
4. NLI fallback          disabled; the model measured AUC 0.53 and is not shipped

The fine-tuned model is not in this chain. It needs each value's REQUIRES and
PROVIDES, which /api/matrix/full-analysis does not carry, so it is reached
directly through /api/matrix/cca-score and /api/matrix/cca-model.
```

### 5.3 `morphological-frontend/` — the Angular interface

| File | Purpose |
|---|---|
| `src/app/app.component.ts` | The whole interface. Box definition, grid rendering, override editing, scenario matching, session persistence. `resolveApiBase()` determines the API location (3.4 below). |
| `src/app/app.component.html` | Template: box definition, consistency grid with provenance colouring, solution space, scenario matching, analyst rulings. |
| `src/assets/config.js` | Optional API location override. Normally left untouched. |
| `src/index.html` | Page shell; loads `assets/config.js` before the application. |

The interface is built with `ng build` and the output copied to the Spring
application's `static/` directory, so one process serves both.

### 5.4 How the API location is resolved — the port problem

A hardcoded API address causes a failure that looks like a server fault but is not.
If the interface is compiled to call `http://localhost:8080/api` and the server is
started on 8090 because 8080 was taken, the page loads correctly from 8090 but
every request is sent to 8080. That is a different origin, so the browser blocks it
and reports a CORS error — pointing at the server, when the server was never
involved. Only the target address was wrong.

`resolveApiBase()` therefore resolves in this order:

1. `?api=http://host:port/api` — query parameter, for a one-off redirect
2. `window.GMA_API_BASE` — set in `assets/config.js`, for a fixed deployment
3. **same origin + `/api`** — the normal case

Case 3 is port-agnostic by construction: the interface and the API share an origin,
so moving the server moves both and nothing needs changing.

---

## PART 6 — REQUEST WALKTHROUGH

What happens when an analyst defines a box and presses Analyse.

1. **Browser → Spring.** `POST /api/matrix/full-analysis` with
   `{parameter: [values]}`. The address is the page's own origin.

2. **Spring selects the engine.** `gma.cca.engine=llm`, so the box is assessed by
   llama.cpp in process. The model is asked once per PARAMETER PAIR with every
   value of both parameters visible, not once per cell: six generations for a
   four-parameter box instead of 176.

3. **No network hop.** Nothing leaves the JVM.

4. **Characterisation.** For every value without one, the service generates
   `requires` and `provides` — with the **adapter disabled**, so the generation
   comes from the base model. The adapter was trained to emit verdicts and would
   otherwise pull the output toward `VERDICT: ...` instead of a description. One
   model in memory, two behaviours.

5. **Scoring.** Every cross-parameter pair is scored in both presentation orders
   and averaged. Roughly 0.4 s per cell on the target hardware.

6. **Python → Spring.** A list of pairs with `p_inconsistent` and a boolean.

7. **Precedence.** `ConsistencyService` applies analyst overrides and the
   calibration head over the model's verdicts.

8. **Enumeration.** `generateValidSolutions` walks the parameter space depth-first,
   pruning any partial configuration containing a marked pair.

9. **Response.** Grid, provenance per cell, score per cell, an explanation per cell,
   the solution space, and the reduction percentage.

10. **Rendering.** The interface colours each cell by provenance and shows the
    explanation on hover, so no elimination is unexplained.

---

## PART 7 — WHAT WAS TRIED AND NOT ADOPTED

Recorded because negative results constrain what is worth attempting next.

| Approach | Outcome |
|---|---|
| Value names only, model generates its own premises | At chance — exposure bias (4.2) |
| NLI cross-encoder (DeBERTa) | AUC 0.53 against expert labels; ONNX-only on this machine |
| Clause-level decomposition with max-pooling | Cell AUC 0.70 against 0.87 for whole-cell scoring. Per-pair accuracy was good, but a cell holds ~10 pairs and the true one must outrank all of them |
| Multiple-instance learning over clause pairs | 0.702 |
| Retrieval-augmented prompting with precedents | 0.787 |
| Linear probes on hidden states | 0.868, below the two-token readout |
| Corpus scaling, 70 → 96 domains | Within the bootstrap interval |
| Verdict-weight sweep (4 / 8 / 16) | Indistinguishable |
| Model and prompt ensembling | No gain once weights were selected honestly |
| Fine-tuning the 7B | Trains, but leaves 253 MB of headroom on a 6 GB card — the occupancy regime that previously wedged the display driver |

---

## PART 8 — CONSTRAINTS THAT SHAPED THE DESIGN

| Constraint | Consequence |
|---|---|
| 6 GB VRAM, shared with the display | Model size, 4-bit quantisation, chunked loss, rank ceiling |
| Air-gapped machine | No `gguf`, `sklearn`, `scipy` or `onnxruntime`. All statistics written in NumPy; the GGUF reader written by hand |
| Small corpus (~96 independent items) | One epoch; domain-level splits and bootstraps throughout |
| Every exclusion must be explainable | Citations verified at build time; evidence carried through to the interface |
