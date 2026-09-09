# Deployed configuration

The single authoritative statement of what runs in the application. If any other
document disagrees with this file, this file is correct.

## Model

| | |
|---|---|
| Adapter | **`results/cell_r32d/`** |
| Base | Qwen2.5-3B (`training/base-model`), 4-bit NF4 |
| Adaptation | QLoRA, rank 32, alpha 64, all seven attention and MLP projections |
| Trainable parameters | 59,867,136 (1.90%) |
| Trained on | clause-grounded criterion, N7 cells removed from training |
| Decision threshold | 0.485 |
| Scoring | logit readout of ' Y' / ' N', averaged over both presentation orders |

Verified on 2026-09-01: the running service reproduces the stored evaluation
scores for held-out cells to within 0.006, so the deployed model is the evaluated
model and not a different checkpoint.

## Figures this adapter produced

Measured on 1,191 cells from 10 domains held out from training, threshold fitted
leave-one-domain-out.

| | |
|---|---|
| AUC | 0.878 |
| F1 at the deployed operating point (50% coverage) | **0.828** |
| Precision / recall there | 0.800 / 0.857 |
| Accuracy there | 0.975 |
| F1 at full coverage | 0.583 |
| Human vs human on the same task | 0.807 |

## Services

Start in this order. The scoring service holds the only CUDA context.

```
1.  cd training
    python serve_cca.py                     # defaults to cell_r32d, port 8000

2.  cd morphological-analyzer
    java -jar target\morphological-analyzer-0.0.1-SNAPSHOT.jar --gma.llm.enabled=false
```

| Service | Port | |
|---|---|---|
| Scoring service | 8000 | `GET /health`, `POST /score`, `POST /cca`, `POST /characterise`, `POST /assess-box` |
| Spring application | 8080 | `POST /api/matrix/full-analysis` and the rest of the existing API |

`GET /api/status` reports `ccaEngine` and the scoring service's health.

## Which engine decides a pair

`gma.cca.engine=finetuned` in `application.properties`. The order of precedence
inside `ConsistencyService` is unchanged:

1. Analyst override, if one is recorded for that pair
2. Calibration head, if trained and enabled
3. **Fine-tuned model** (this adapter)
4. GGUF baseline — used only as a fallback when the scoring service is unreachable,
   so the grid is never lost

Set `gma.cca.engine=llm` to return to the Qwen2.5-7B GGUF baseline.

## Path from names to verdicts

The interface collects value names only. The model judges from what each value
requires and provides, so the service generates a characterisation for any value
that arrives without one, with the adapter disabled (base model), then scores every
cross-parameter pair with the adapter enabled.

**Note on scope of the figures above.** They were measured on characterisations
written by hand in the corpus. In the live path the characterisations are
generated. That end-to-end condition has not been measured; the figures describe
the assessment model given good characterisations, not the complete
names-to-verdicts pipeline.

## Other adapters kept

| Adapter | Full-coverage F1 | AUC | |
|---|---|---|---|
| `cell_r32d` | 0.583 | 0.878 | **deployed** — best at the operating point |
| `cell_drop7` | 0.594 | 0.874 | best at full coverage; its confidence curve collapses past 70% |
| `cell_noN7c` | 0.579 | 0.885 | N7 relabelled rather than removed |
| `cell_w8sym` | — | 0.817 | original, before the criterion was narrowed |
