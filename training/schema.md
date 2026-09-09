# Dataset schema and build decisions

## Decisions taken (change any of these and I'll rebuild)

**Slots: two — `REQUIRES` and `PROVIDES`.** Scale is folded into whichever slot it belongs to
("PROVIDES: sustained large-scale force") rather than given a third slot, because every slot costs
output tokens and on CPU tokens are wall-clock time.

**Reasoning lines: excluded from the training target.** They roughly triple output length. They are
written into `REVIEW.md` instead so the derivations can be spot-checked by hand without the model
ever paying for them.

**Verdict density: not targeted at all.** Chasing a global density figure would teach the model a
prior — "mark about half" — which is the same defect as teaching it a threshold, and would make it
over-mark loose matrices into an empty solution space. Density is an *output* of applying the rule
honestly: constrained domains land high (air defence, 46%), loose ones land low (cold chain, 22%).
**Per-matrix density is reported in `STATS.md` so drift is visible rather than assumed**, but it is
a diagnostic, not a target.

**Ritchey's threat matrix is excluded entirely.** It is a reference example, not a scoring
target — there is no baseline to match — but it stays out of training so it remains usable as an
independent sanity check.

---

## Storage format — `matrices/*.json`

```json
{
  "domain": "maritime interdiction",
  "group": "security",
  "parameters": {
    "INTERDICTING PARTY": ["National coast guard", "Naval task force"],
    "LEGAL BASIS": ["Territorial waters enforcement", "UN Security Council mandate"]
  },
  "characterisations": {
    "National coast guard": {
      "requires": "jurisdiction within own maritime zone",
      "provides": "police powers, arrest authority, domestic legal standing"
    }
  },
  "verdicts": {
    "INTERDICTING PARTY|LEGAL BASIS": ["YN", "YY"]
  },
  "notes": {
    "INTERDICTING PARTY|LEGAL BASIS": "row 1 col B is N because a mandate authorises action beyond national waters"
  }
}
```

`verdicts` is keyed by the two parameter names joined with `|`, in declaration order. Each string is
one row: one character per value of the second parameter, in declaration order.

`notes` is optional and feeds `REVIEW.md` only. It never reaches the model.

---

## Training example format — `dataset.jsonl`

One example per parameter pair. Self-contained, so it matches how it is called at inference: one
call per pair, no separate preparatory step.

```
### INPUT
Dimension 1 — INTERDICTING PARTY:
  1. National coast guard
  2. Naval task force

Dimension 2 — LEGAL BASIS:
  A. Territorial waters enforcement
  B. UN Security Council mandate

### OUTPUT
CHARACTERISATIONS
National coast guard | REQUIRES: … | PROVIDES: …
Naval task force | REQUIRES: … | PROVIDES: …
Territorial waters enforcement | REQUIRES: … | PROVIDES: …
UN Security Council mandate | REQUIRES: … | PROVIDES: …

VERDICTS
1:YN
2:YY
```

Wrapped in the Qwen ChatML template at build time, with loss masked to the assistant turn.

---

## Consistency rule the verdicts are derived from

> A pair is **inconsistent** when either:
>   1. one side REQUIRES something the other cannot PROVIDE, or their PROVIDES are mutually
>      exclusive — the pairing is *impossible*; or
>   2. no plausible real-world case instantiates the pairing — it is *implausible*.
>
> Mark Y only where a reasonable case exists, even if uncommon. "Technically conceivable" is not
> sufficient grounds for Y.

Every verdict in this dataset follows from that rule applied to the two characterisations. Where a
verdict looks wrong, the fix is the characterisation, not the verdict — that is what keeps the two
halves consistent by construction and stops the model learning to state grounds and then ignore them.

---

## Split

Matrices are split by **domain**, never by example. A matrix contributes all of its parameter pairs
to exactly one split, so no domain is ever seen in both training and validation. Splitting at the
example level would leak the characterisations of a domain into validation and make the score
meaningless.

- train: ~85% of matrices
- validation: ~15% of matrices
- test: Ritchey's threat matrix, held out entirely and never in this directory
