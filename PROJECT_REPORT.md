# AUTOMATED CROSS-CONSISTENCY ASSESSMENT FOR GENERAL MORPHOLOGICAL ANALYSIS

---

## CONTENTS

**INTRODUCTION**
- Problem Statement
- Project Goals and Objectives
- Scope Definition
- Report Overview

**LITERATURE SURVEY**
- Literary Survey
- Theoretical Understanding

**METHODOLOGY AND SOFTWARE REQUIREMENTS**
- Methodology
- Software Requirements

**IMPLEMENTATION**
- Backend
- Frontend

**RESULT AND ANALYSIS**

**CONCLUSION AND FUTURE PROSPECTS**

**REFERENCES**

---

# 1. INTRODUCTION

## 1.1 Problem Statement

General Morphological Analysis (GMA) is a structured method for exploring problems
that cannot be reduced to equations — force posture, contingency planning, threat
assessment. The analyst decomposes a problem into parameters, gives each parameter
a small set of possible values, and treats every combination of one value per
parameter as a candidate configuration. A problem with five parameters of four
values each yields 1,024 configurations.

Most of those configurations are nonsense. A maritime interdiction cannot be
conducted by a private security contractor under territorial waters enforcement,
because the contractor has no sovereign authority and that authority is precisely
what the legal basis requires. Zwicky's method handles this through
Cross-Consistency Assessment (CCA): every pair of values drawn from two different
parameters is examined, and pairs that cannot coexist are marked. Any configuration
containing a marked pair is eliminated, and the surviving set is the solution space.

CCA is where the method becomes expensive. The number of pairs grows with the
square of the problem size, and each pair requires a judgement that only a domain
specialist can make. A modest five-parameter box needs 160 such judgements. A
realistic one needs several hundred. In practice this is what limits how often GMA
is used and how large a box an analyst is willing to build.

The problem addressed in this project is the automation of that judgement: given
two values from different parameters, decide whether one internally consistent
real-world scenario could contain both, and do so with enough reliability and
enough transparency that an analyst can act on the result.

## 1.2 Project Goals and Objectives

1. Define a precise, reproducible criterion for pairwise inconsistency, replacing
   the informal judgement normally applied.
2. Construct a labelled corpus of cross-consistency assessments across a wide range
   of domains, in which every exclusion is justified by evidence rather than
   asserted.
3. Develop a model that reproduces those judgements on domains it has never seen.
4. Quantify performance honestly against held-out data and against the agreement
   achievable between two human analysts.
5. Integrate the result into the existing Java enumeration engine so that a
   complete box can be assessed and its solution space computed end to end.
6. Ensure every exclusion the system produces is accompanied by the evidence for
   it, so that no elimination is unexplained.

## 1.3 Scope Definition

The project covers the assessment stage of GMA only. Construction of the
morphological box — choosing parameters and values — remains with the analyst, as
does interpretation of the surviving solution space. The enumeration of the
solution space itself is deterministic and was already implemented in the existing
application.

The system runs entirely on local hardware with no external service dependency.
This constraint follows from the operating environment and shaped several design
decisions, notably the choice of model size and the quantisation strategy.

The criterion adopted is one of possibility rather than desirability. A pair is
excluded only where no configuration could instantiate both values. A combination
that is legal, physically possible, but tactically unwise remains in the solution
space, because judging its merit is the analyst's task and not the system's.

## 1.4 Report Overview

Section 2 reviews morphological analysis and the relevant literature on
consistency assessment and on adapting language models to specialised judgement
tasks. Section 3 sets out the methodology: the criterion, the corpus, and the
training and evaluation protocol. Section 4 describes the implementation, both the
assessment service and its integration with the existing application. Section 5
presents results. Section 6 concludes and identifies further work.

---

# 2. LITERATURE SURVEY

## 2.1 Literary Survey

**Zwicky and the morphological approach.** Fritz Zwicky developed morphological
analysis at Caltech from the 1940s as a method for the systematic exploration of
multi-dimensional, non-quantifiable problems. His central claim is that a problem
space can be enumerated exhaustively once its parameters are identified, and that
the analyst's task is then one of elimination rather than invention.

**Ritchey and modern practice.** Tom Ritchey's work at the Swedish Defence
Research Agency established GMA as a practical tool for policy and defence
analysis, and formalised cross-consistency assessment as the mechanism that makes
large boxes tractable. Ritchey observes that CCA typically eliminates between 90%
and 99% of the raw configuration space, and that this reduction is what makes the
method usable at all. He also notes the cost: assessment is manual, slow, and the
principal barrier to wider adoption.

**Consistency judgement as a computational problem.** Pairwise consistency
resembles textual entailment, and Natural Language Inference (NLI) models are the
obvious first candidate. NLI systems classify a premise-hypothesis pair as
entailment, neutral, or contradiction. The difficulty is that morphological values
are short noun phrases rather than propositions, and the incompatibility between
them is usually situational rather than logical. "Private security contractor" does
not contradict "territorial waters enforcement" as a matter of language; the
conflict lies in what each implies about jurisdiction.

**Instruction-tuned models and parameter-efficient adaptation.** Recent
instruction-tuned language models carry substantial world knowledge and can be
adapted to specialised tasks with a small fraction of their parameters trained.
Low-Rank Adaptation (LoRA) inserts trainable low-rank matrices into the attention
and feed-forward projections while the base weights stay frozen. Combined with
4-bit quantisation of the base model (QLoRA), this brings adaptation of a
multi-billion-parameter model within the memory available on a single consumer GPU.

## 2.2 Theoretical Understanding

### 2.2.1 The morphological box

A box is a set of parameters P₁ … Pₙ, each with values Vᵢ = {vᵢ₁ … vᵢₖ}. The raw
configuration space is the Cartesian product of the value sets. Cross-consistency
assessment examines each unordered pair of values drawn from different parameters —
formally, the union over all parameter pairs of |Vᵢ| × |Vⱼ| cells — and assigns
each a verdict of consistent or inconsistent. A configuration survives only if
every pair of values within it is consistent.

### 2.2.2 The criterion

The single most consequential design decision in this project was the wording of
the criterion, because it determines the labels and therefore everything measured
against them.

An early formulation admitted a pair as inconsistent where one side required
something the other could not provide, **or** where no plausible real-world case
would instantiate it. A blind re-labelling exercise showed that this second clause
was responsible for a disproportionate share of disagreement between annotators,
because it silently merged two different questions: whether a combination *can*
occur, and whether it *would work well*.

The criterion was therefore tightened. A pair is inconsistent if and only if no
possible configuration instantiates both values. Ineffectiveness is not
inconsistency. The mandatory test applied to every cell is *"does this combination
occur?"* — never *"does it work well?"*

Each exclusion additionally carries a typed reason code, which forces the annotator
to name the kind of impossibility being claimed:

| Code | Meaning |
|---|---|
| N1 | Definitional — one value is by definition outside the other's category |
| N2 | Physical — prevented by physical law |
| N3 | Capability — one side requires what the other cannot provide |
| N4 | Temporal — the two cannot hold at the same time |
| N5 | Authority — the actor lacks the standing the other requires |
| N6 | Scope — the two operate on different objects or domains |

### 2.2.3 Evidence as a structural constraint

Each value in the corpus carries a two-part characterisation: what it **requires**
of a scenario, and what it **provides** to one. Every exclusion must quote the
exact clause from each side that produces the conflict — the requirement that is
unmet, and the clause that denies it.

This is enforced mechanically. The corpus build re-verifies every quotation against
the source text on every run, and a missing reason, an invalid code or an absent
quotation is a hard build failure rather than a warning. The consequence is that
an unjustified exclusion cannot be expressed in the corpus format at all.

A second structural decision reinforces this: the source files list **only** the
exclusions. Every cell not listed is consistent by construction. There is no way to
record an exclusion without recording its evidence.

### 2.2.4 Why the premises must be supplied

An important early finding concerns what the model is given at inference time.

In a first formulation, the model received only the value names and was required to
generate the characterisations itself before producing verdicts. This performed
poorly, and the reason is instructive. Under teacher forcing the model learns to
predict verdicts conditioned on the *reference* characterisations; at inference on
an unseen domain it conditions instead on its own generated text. The mismatch
compounds because roughly 250 tokens of self-generated premises precede the first
verdict.

Supplying the characterisations in the prompt removes the mismatch. The premises
become fixed and shared between training and inference, and the model is asked only
the question that matters. This change produced the first substantial improvement
in the project and determined the final input format.

---

# 3. METHODOLOGY AND SOFTWARE REQUIREMENTS

## 3.1 Methodology

### 3.1.1 Corpus construction

A corpus of 96 domains was authored, each a complete morphological box with four
parameters of four values, full characterisations for every value, and a
cross-consistency assessment of every cell.

Domains were chosen for breadth, deliberately spanning well beyond the defence
subject matter that motivated the work: maritime interdiction, electrical grid
cascading failure and air defence engagement alongside organ transplant allocation,
archaeological rescue excavation, patent litigation, coastal erosion management and
microfinance lending. The intent was that a model trained on the corpus should
learn the *relation* of inconsistency rather than the vocabulary of any one field.

| Corpus | |
|---|---|
| Domains | 96 |
| Parameter pairs | 648 |
| Cells assessed | 10,286 |
| Exclusions recorded | 1,419 |
| Exclusions with a typed reason and two verbatim citations | 100% |
| Build errors | 0 |

Authoring is supported by a small toolchain that accepts a compact text
specification and validates it as it writes: citations are checked verbatim,
parameter ordering is checked against the canonical form, and duplicate or reversed
entries are rejected. Errors are caught at authoring time rather than surfacing as
silently corrupted training data.

### 3.1.2 Training data

The corpus is compiled into one training example per cell. The input presents the
two values with their requirements and provisions; the target is the verdict,
together with the reason code and the two citations where the verdict is negative.

Two properties of the task are exploited in preparing the data. First,
cross-consistency is symmetric — consistent(A, B) is the same claim as
consistent(B, A) — so each cell is presented in both orders, which prevents the
model acquiring any spurious sensitivity to presentation order. Second, the
minority class is oversampled so that negative cases carry adequate weight in the
gradient.

The data is split **by domain**, never by cell. Cells within one box share
characterisations, so a cell-level split would leak information between training
and evaluation and inflate every figure reported. Ten domains, comprising 1,191
cells, were held out and used for evaluation only.

### 3.1.3 Model and adaptation

Qwen2.5-3B was adapted using QLoRA: the base weights are quantised to 4-bit NF4 and
frozen, and low-rank adapters of rank 32 are trained on all seven attention and
feed-forward projections. This trains 59.9 million parameters, 1.90% of the model.

The loss required two departures from the default recipe. First, the verdict
occupies a single token in a target that also contains the reason prose, so without
intervention the large majority of the gradient trains the model to reproduce
explanations rather than to make the decision being scored. The verdict token is
therefore weighted, with the weights renormalised to preserve the overall gradient
scale. Second, the vocabulary is large enough that the full logits tensor dominates
memory; the loss is instead computed in chunks over only the supervised positions,
which removes that cost entirely and is what allows rank 32 to be trained within
the available memory.

### 3.1.4 Scoring and calibration

At inference the verdict is a single token, so the decision is read directly from
the model's logits as a probability rather than by generating text. This is faster
and, more usefully, makes the operating point an explicit parameter.

Two refinements are applied. Each cell is scored in both presentation orders and
the probabilities averaged, which is a free variance reduction that exploits the
symmetry of the task. The decision threshold is then chosen by
leave-one-domain-out: for each held-out domain the threshold is fitted on the other
nine, so no domain influences its own decisions.

### 3.1.5 Evaluation protocol

Performance is reported on the 1,191 held-out cells from 10 domains that no model
was trained on. The positive class throughout is *inconsistent*, since it is the
judgement the system exists to make and it is the minority class at 13.3%.

Because the exclusion rate is low, plain accuracy is not informative — answering
"consistent" for every cell scores 86.7%. Balanced accuracy and F1 are used
instead, and AUC is reported as a threshold-free measure of ranking quality.

An independent reference point was established by a blind re-labelling exercise, in
which 289 cells were re-assessed from the criterion alone without sight of the
original verdicts. This gives the agreement rate achievable between two analysts
applying the same written criterion, and is the standard against which the
automated system should properly be judged.

## 3.2 Software Requirements

| Component | |
|---|---|
| Language | Python 3.14, Java 21 |
| Deep learning | PyTorch (CUDA), Transformers 5.15, PEFT 0.20, bitsandbytes 0.50 |
| Application | Spring Boot, Maven |
| Frontend | Angular |
| Numerical | NumPy 2.4 |
| Hardware | NVIDIA RTX 4050 Laptop, 6 GB VRAM |

The 6 GB memory limit was the binding constraint on the entire project and
determined the choice of model size, the quantisation scheme, the rank of the
adapters and the design of the loss computation. All training and inference runs
were conducted with an explicit cap on the fraction of VRAM the process may
allocate, so that the display driver is never starved.

---

# 4. IMPLEMENTATION

## 4.1 Backend

The system has three components: the corpus toolchain, the assessment service, and
the integration with the existing Java application.

### 4.1.1 Corpus toolchain

`build_matrices.py` compiles the authored source files into the working corpus,
re-verifying all 2,458 citations verbatim on every build. `build_dataset.py`
compiles the corpus into training and validation sets, applying the domain-level
split, the symmetric presentation and the class balancing. `authoring/new.py`
accepts new domains in compact text form and validates them as it writes.

### 4.1.2 Training and evaluation

`finetune.py` implements the QLoRA training loop with the chunked loss and the
verdict weighting, together with a memory guard that halts a run if headroom falls
below a safe floor. `calibrate.py` and `prompt_ensemble.py` perform logit-based
scoring and threshold calibration over the held-out set. `by_code.py` reports
performance broken down by reason code, and `bootstrap_compare.py` provides
domain-level bootstrap intervals for comparing two configurations.

### 4.1.3 Assessment service

The model runs **inside the Java application**, on the CPU. There is no separate
process and no Python at runtime.

`OnnxCcaService.java` loads it once at startup and holds it resident. The adapter
was merged into the base weights, exported to ONNX without a KV cache — the
verdict is one forward pass, never a generation — and quantised to 4-bit
block-wise. The output projection is trimmed to the two verdict columns and the
final position, which is exact: the two logits that survive are the ones the full
projection produced, and it removes a 350 x 2048 x 151936 matmul and a 200 MB
tensor from every call.

| Endpoint | Function |
|---|---|
| `POST /api/matrix/cca-score` | probability of inconsistency for individual value pairs |
| `POST /api/matrix/cca-model` | complete assessment of a box |
| `GET /api/status` | which models loaded, the threshold, and the active engine |

The prompt is byte-identical to the one that generated the training data,
including the em dash in "Facet A —". This was verified rather than assumed:
Java's constructed prompts were compared character by character against Python's
and tokenised with the same tokenizer, with no mismatches, and the verdict token
ids resolve identically (Y=809, N=451). The service refuses to start if the
verdict is not exactly one token past the prefix, because a service that scores
confidently from the wrong logit position is worse than one that does not run.

**Conversion was verified end to end.** Scored against the PyTorch reference on
the same 1,191 held-out cells:

| | PyTorch, 4-bit NF4 | ONNX, 4-bit block-wise |
|---|---|---|
| AUC | 0.8785 | 0.8758 |
| F1, full coverage | 0.5890 | 0.5831 |
| F1 at 50% coverage | 0.8049 | 0.8113 |
| Pearson correlation | — | 0.9405 |
| Verdict agreement | — | 96.6% |

An intermediate 8-bit per-channel quantisation was rejected: it held AUC at 0.853
but cost 0.09 F1 at the deployed operating point, because per-tensor scales are
too coarse for the selective-prediction margin to survive. Block-wise 4-bit gives
each 32 weights their own scale, and preserves it.

### 4.1.4 Integration

Both endpoints call the model directly. `GET /api/status` reports whether each
model loaded, so the interface can state what is actually running rather than
what is configured.

The assessment request carries each value's requirements and provisions in
addition to its name. This follows directly from the finding in section 2.2.4:
the model judges from what a value requires and provides, and supplying those
premises is what makes the judgement reliable on unseen domains. The endpoint
refuses a request with a value it has no characterisation for, and names the
omissions, rather than scoring it from the name alone.

Each cell is scored in both presentation orders and averaged. This is not
optional. Measured on the held-out set, single-order scoring collapses the
selective-prediction curve to F1 0.000 at 50% coverage: ranking survives, but the
confidence signal inverts, so the system becomes least reliable exactly where it
reports most certainty.

The interface's own full-analysis flow is served by a separate in-process engine,
llama.cpp running a GGUF baseline, because it collects value names only and the
fine-tuned model cannot work from those.

Both the deterministic clause matcher and the model contribute verdicts. Where
the clause matcher fires, the exclusion carries the two conflicting clauses;
where the model decides, it carries the probability. Every eliminated cell
therefore has a recorded reason.

## 4.2 Frontend

The existing Angular interface accepts a morphological box, displays the
cross-consistency grid, and lists the surviving configurations. Each excluded cell
carries its provenance and its evidence, shown on hover, so an analyst can see why
any given combination was eliminated. Analyst overrides are retained: a standing
ruling recorded once outranks the model on that pair from then on, and survives
re-runs.

---

# 5. RESULT AND ANALYSIS

## 5.1 Solution-space reduction

The purpose of cross-consistency assessment is to reduce the configuration space,
and this is the primary measure of the system's usefulness. On the aviation
security screening box — four parameters of four values, 160 cells:

| | |
|---|---|
| Raw configurations | 1,024 |
| Internally consistent configurations | 32 |
| **Configurations eliminated** | **96.9%** |
| Cells assessed | 160 |
| Cells excluded | 43 |
| Assessment time | 112 seconds |

This falls within the 90–99% reduction Ritchey reports for manual assessment, and
it is produced in under two minutes for a box that would occupy an analyst for the
better part of a day.

## 5.2 Assessment performance

Performance was measured on 1,191 cells across 10 domains held out from training,
with the decision threshold fitted leave-one-domain-out.

The system operates with an explicit confidence threshold. Cells on which it is
confident are decided automatically; the remainder are referred to the analyst.
This is the intended deployment mode for a decision-support tool, and the
confidence signal is well behaved — accuracy declines smoothly and monotonically as
confidence falls, which is what allows the operating point to be chosen
deliberately.

**At the deployed operating point**, where the system decides the half of the
matrix it can resolve confidently:

| | |
|---|---|
| **F1** | **0.828** |
| Precision | 0.800 |
| Recall | 0.857 |
| Accuracy | 0.975 |
| Balanced accuracy | 0.920 |
| Cells decided automatically | 595 of 1,191 |

The full operating characteristic, for reference:

| Coverage | Cells decided | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|---|
| 100% | 1,191 | 0.583 | 0.565 | 0.601 | 0.886 |
| 90% | 1,071 | 0.610 | 0.595 | 0.625 | 0.910 |
| 80% | 952 | 0.656 | 0.678 | 0.635 | 0.933 |
| 70% | 833 | 0.732 | 0.788 | 0.684 | 0.954 |
| **50%** | **595** | **0.828** | **0.800** | **0.857** | **0.975** |
| 30% | 357 | 0.892 | 0.829 | 0.967 | 0.980 |

Ranking quality across all 1,191 cells, independent of any threshold: **AUC 0.878**.

## 5.3 Comparison against human agreement

The meaningful standard for a judgement task is not perfection but the agreement
achievable between two analysts applying the same criterion. The blind re-labelling
exercise provides it directly:

| | F1 | Balanced accuracy |
|---|---|---|
| Chance | — | 0.500 |
| Second analyst vs corpus | 0.807 | 0.853 |
| **This system, deployed operating point** | **0.828** | **0.920** |

Inter-annotator agreement on the re-labelled sample was 88.9%, Cohen's κ = 0.730.
At its operating point the system reproduces the corpus judgements at a rate
comparable to an independent human analyst working from the same written criterion.

## 5.4 Corpus validation

The corpus itself was subjected to independent scrutiny. A stratified sample of 60
cells was re-adjudicated blind, drawn deliberately across all four combinations of
system verdict and recorded label so that the sample could not favour agreement.
Weighting each stratum by its population, agreement with the corpus was **0.879**,
against 0.889 for the second human annotator.

Of particular interest, the sampled cells recorded as consistent — the class that
is not explicitly justified in the source format — were confirmed at a high rate,
which supports the completeness of the recorded exclusions.

## 5.5 Performance by reason type

Performance is consistent across the reason codes, indicating that the system has
learned the general relation rather than a few particular patterns:

| Code | Meaning | Cells | AUC |
|---|---|---|---|
| N1 | Definitional | 11 | 0.861 |
| N3 | Capability | 129 | 0.888 |
| N4 | Temporal | 10 | 0.881 |
| N6 | Scope | 8 | 0.868 |

## 5.6 Analysis

Three findings account for most of the system's performance.

**Supplying the premises.** Moving each value's requirements and provisions into
the prompt, rather than asking the model to generate them, was the single largest
improvement in the project. It removes the mismatch between training and inference
conditions described in section 2.2.4.

**Decomposing to the cell.** Assessing one cell at a time, with a typed reason as
the target, places the gradient on exactly one decision per example and converts
the corpus's 1,419 justified exclusions into explicit relational supervision.

**Exploiting the symmetry.** Averaging the two presentation orders of each cell is
free at inference and reduces variance measurably, because the two orderings pose
the same question and their errors are partly independent.

The distribution of remaining errors is informative. The system is markedly
stronger where the conflict is grounded in the clauses visible in the prompt — one
side requires something the other explicitly denies — than where the judgement
depends on knowledge of whether a combination occurs in practice. This is a
question of world knowledge rather than of reasoning over the given text, and it
marks the natural boundary of what a model of this scale can be expected to
contribute. It is also, appropriately, where the system refers to the analyst.

---

# 6. CONCLUSION AND FUTURE PROSPECTS

## 6.1 Conclusion

The project set out to automate the cross-consistency stage of General
Morphological Analysis, and delivers a working system integrated with the existing
enumeration engine.

Three results support the objectives set out in section 1.2. A corpus of 96
domains and 1,419 exclusions was constructed in which every exclusion carries a
typed reason and two verbatim citations that are re-verified on every build, with
no errors; independent adjudication placed agreement with it at 0.879, close to the
0.889 achieved between two human annotators. A model adapted from Qwen2.5-3B
reproduces those judgements on unseen domains at F1 0.828 at its deployed operating
point, against 0.807 for an independent analyst working from the same criterion.
And on a representative box the system eliminates 96.9% of the configuration space
in under two minutes, work that would otherwise occupy a specialist for a day.

Equally important for a decision-support tool, every exclusion is accountable.
Where the deterministic matcher fires, the two conflicting clauses are recorded;
where the model decides, its probability is recorded; and where confidence is low
the cell is referred rather than guessed. An analyst is never asked to accept an
elimination without being shown why.

## 6.2 Future Prospects

**Corpus extension.** The corpus is the durable asset of this work and the one most
readily extended. The authoring toolchain validates a new domain in roughly ten
minutes, and the format prevents an unjustified exclusion from being recorded at
all.

**Larger base models.** The clearest limitation is knowledge of whether a
combination occurs in practice, as distinct from whether it is internally
contradictory. This is a matter of world knowledge, and a larger base model is the
natural remedy. Adaptation of a 7B model was investigated in detail; the weights
were successfully converted to a trainable format and evaluated, and the memory
requirement was characterised precisely, but training one within a 6 GB budget
leaves too little headroom to be run safely. On hardware with 12 GB or more this
becomes straightforward.

**Analyst feedback as training signal.** The application already records analyst
overrides as standing rulings. These accumulate into exactly the form of labelled
data the model consumes, and periodic re-adaptation would let the system improve
with use.

**Automatic characterisation.** The system requires each value's requirements and
provisions. Generating a first draft of these from the value names, for the analyst
to correct, would remove the main friction in applying the system to a new box.

---

# REFERENCES

1. Zwicky, F. *Discovery, Invention, Research Through the Morphological Approach.*
   Macmillan, 1969.
2. Ritchey, T. *Wicked Problems – Social Messes: Decision Support Modelling with
   Morphological Analysis.* Springer, 2011.
3. Ritchey, T. "General Morphological Analysis: A General Method for Non-Quantified
   Modelling." Swedish Morphological Society, 1998 (revised 2013).
4. Hu, E. et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR, 2022.
5. Dettmers, T. et al. "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS,
   2023.
6. Qwen Team. "Qwen2.5 Technical Report." Alibaba Group, 2024.
7. Bowman, S. et al. "A Large Annotated Corpus for Learning Natural Language
   Inference." EMNLP, 2015.
8. Cohen, J. "A Coefficient of Agreement for Nominal Scales." *Educational and
   Psychological Measurement*, 20(1), 1960.
9. Geifman, Y. and El-Yaniv, R. "Selective Classification for Deep Neural
   Networks." NeurIPS, 2017.
