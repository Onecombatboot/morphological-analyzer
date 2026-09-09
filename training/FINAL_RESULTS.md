# Final results — cross-consistency assessment

All runs scored on the same 1,191 held-out cells from 10 domains no model
trained on. Threshold chosen leave-one-domain-out to maximise F1.

| run | AUC | F1 | precision | recall | balanced acc | accuracy |
|---|---|---|---|---|---|---|
| drop7 | 0.874 | **0.594** | 0.586 | 0.601 | 0.768 | 0.891 |
| cell_aug | 0.869 | **0.585** | 0.615 | 0.557 | 0.752 | 0.895 |
| r32d | 0.878 | **0.583** | 0.565 | 0.601 | 0.765 | 0.886 |
| noN7c | 0.885 | **0.579** | 0.538 | 0.627 | 0.772 | 0.879 |
| hard | 0.877 | **0.576** | 0.520 | 0.646 | 0.777 | 0.874 |
| cell_92full | 0.811 | **0.556** | 0.550 | 0.562 | 0.727 | 0.830 |
| cell_92c_vw4 | 0.824 | **0.555** | 0.561 | 0.549 | 0.724 | 0.833 |
| cell_lowlr | 0.867 | **0.552** | 0.511 | 0.601 | 0.757 | 0.871 |
| pe | 0.817 | **0.552** | 0.526 | 0.580 | 0.729 | 0.821 |
| sib | 0.887 | **0.548** | 0.572 | 0.525 | 0.733 | 0.885 |
| cell_auglr | 0.853 | **0.526** | 0.580 | 0.481 | 0.714 | 0.885 |
| cell_92c | 0.809 | **0.514** | 0.452 | 0.597 | 0.714 | 0.786 |
| retr | 0.787 | **0.503** | 0.544 | 0.468 | 0.704 | 0.877 |
| cell_92c_vw16 | 0.809 | **0.502** | 0.430 | 0.602 | 0.708 | 0.773 |

## Best run: `drop7`

F1 0.594 · AUC 0.874 · precision 0.586 · recall 0.601 · accuracy 0.891

Confusion: 95 true positives, 67 false positives, 63 false negatives.

## Selective prediction

The system decides the cells it is confident about and refers the rest.

| coverage | cells | F1 | precision | recall | accuracy |
|---|---|---|---|---|---|
| 100% | 1191 | **0.594** | 0.586 | 0.601 | 0.891 |
| 90% | 1071 | **0.630** | 0.636 | 0.625 | 0.918 |
| 80% | 952 | **0.655** | 0.695 | 0.620 | 0.937 |
| 70% | 833 | **0.629** | 0.824 | 0.509 | 0.960 |
| 60% | 714 | **0.000** | 0.000 | 0.000 | 0.969 |
| 50% | 595 | **0.000** | 0.000 | 0.000 | 0.975 |

## Is the best run really better than the reference?

Domain-level bootstrap of F1(best) - F1(scores_noN7c_v0.json):

- 95% CI **[-0.006, +0.118]**
- P(best is better) = **0.96**

The interval spans zero, so the difference is not established.
The interval excludes zero.

## Context

- Human parity on this task is F1 ~0.72, from the blind re-labelling
  (two annotators, 289 cells, agreement 0.889, kappa 0.730).
- Chance on balanced accuracy is 0.500.
- Always-"consistent" scores 86.7% accuracy at this base rate, so accuracy
  should never be the headline.
