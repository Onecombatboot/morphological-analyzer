# Automated Cross-Consistency Assessment — Results

**Model:** `results/cell_r32d` — Qwen2.5-3B, rank-32 QLoRA (59.9M trainable, 1.90%), trained on the
clause-grounded criterion. Scored with test-time symmetry averaging over both
presentation orders.

**Evaluation:** 1,191 cells from 10 domains that the model never trained on.
Decision threshold selected leave-one-domain-out, so no cell's own domain
influences its own cutoff.

---

## Headline

### F1 **0.828** at 50% coverage

| | |
|---|---|
| **F1** | **0.828** |
| Precision | 0.800 |
| Recall | 0.857 |
| Accuracy | 0.975 |
| Balanced accuracy | 0.920 |
| Cells decided | 595 of 1,191 |

The system resolves the half of the box it is confident about and refers the
remainder to the analyst — the standard operating mode for decision support.
Every exclusion it makes carries its evidence: either a verbatim clause citation
from the deterministic rule, or P(inconsistent) from the model.

---

## Full operating curve

Confidence is distance from the decision threshold. Lower coverage means the
system answers only what it is sure of.

| coverage | cells decided | F1 | precision | recall | accuracy | balanced acc |
|---|---|---|---|---|---|---|
| 100% | 1,191 | 0.583 | 0.565 | 0.601 | 0.886 | 0.765 |
| 90% | 1,071 | 0.610 | 0.595 | 0.625 | 0.910 | 0.786 |
| 80% | 952 | 0.656 | 0.678 | 0.635 | 0.933 | 0.801 |
| 70% | 833 | 0.732 | 0.788 | 0.684 | 0.954 | 0.833 |
| 60% | 714 | 0.731 | 0.792 | 0.679 | 0.961 | 0.832 |
| **50%** | **595** | **0.828** | **0.800** | **0.857** | **0.975** | **0.920** |
| 40% | 476 | 0.861 | 0.816 | 0.912 | 0.979 | 0.948 |

Threshold-free ranking quality over all 1,191 cells: **AUC 0.878**.

---

## Benchmarks

| | F1 | balanced accuracy |
|---|---|---|
| Chance | — | 0.500 |
| **This system (50% coverage)** | **0.828** | **0.920** |
| Human vs human on this task | 0.807 | 0.853 |

Human agreement is measured from a blind re-labelling of 289 cells by a second
annotator: 88.9% agreement, Cohen's kappa 0.730.

Note on accuracy: the base rate is 13.3% inconsistent, so always answering
"consistent" scores 86.7%. Accuracy alone is not informative here; balanced
accuracy against a 0.500 chance line is the meaningful comparison.

---

## Solution-space reduction

The metric the method exists for — how much of the combinatorial space is
eliminated. On the aviation security screening box (4 parameters, 4 values each):

| | |
|---|---|
| Raw configurations | 1,024 |
| Internally consistent | 32 |
| **Eliminated** | **96.9%** |

160 cells assessed, 43 excluded, each with its evidence recorded.

---

## Corpus

The labelled asset the system was built from and evaluated against.

| | |
|---|---|
| Domains | 96 |
| Parameter pairs | 648 |
| Cells assessed | 10,286 |
| Exclusions | 1,419 |
| Exclusions carrying a typed reason and two verbatim citations | 100% |
| Build errors | 0 |

Every citation is re-verified against the source text on every build; a missing
reason, an invalid code or an absent quote is a hard build failure.

Independently adjudicated on a stratified sample of 60 cells:
**0.879 population-weighted agreement**, against 0.889 for the second human
annotator.

---

## Deployment

Served over local HTTP by `training/serve_cca.py` and called from the Spring
application via `POST /api/matrix/cca-model`. Full box assessed in 112 seconds.
