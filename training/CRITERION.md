# Cross-Consistency Criterion (v2)

The v1 rule was *"inconsistent when one side requires what the other cannot provide,
**or no plausible real-world case instantiates it**."* A blind re-labelling of 289
cells reproduced it at 88.9% (Cohen kappa 0.730). The 11% that did not reproduce is
not random: **45% of all disagreements are one confusion**, caused entirely by that
second clause.

## The confusion

"No plausible case instantiates it" silently merges two different questions:

| question | example | v1 answer |
|---|---|---|
| Can these co-occur at all? | black start x restoration in *minutes* | impossible |
| Would this combination work well? | hand crew x erratic gusting wind | possible, but unsafe |

Both got labelled `N`, inconsistently. Every one of these disagreed on blind re-labelling:

```
automatic load shedding   x Days     manual dispatch x Days
islanding into sub-grids  x Days     black start     x Hours
hand crew with tools      x erratic gusting
fixed-wing retardant      x sustained strong wind
contact tracing           x dense urban settlement
contact tracing           x refugee camp
blood product screening   x dispersed rural population
blood product screening   x closed institution
water treatment           x dispersed rural population
controlled drawdown       x overtopping in flood
downstream evacuation     x liquefaction under seismic load
onboard observer          x domestic coastal vessel
aerial patrol             x reefer transhipment vessel
```

## The v2 rule

> A pair is **inconsistent (N)** if and only if **no possible configuration
> instantiates both**. Mark `N` only when you can name the specific clause that
> makes it impossible.
>
> **Ineffectiveness is not inconsistency.** If the combination is merely unwise,
> unsafe, ineffective, suboptimal, rare, or expensive — it is `Y`.

An analyst wants the solution space of what *can* exist. Pruning what merely works
badly is a later, separate ranking step, and folding it in here destroys both.

## Typed exclusion reasons

Every `N` must carry a code and cite the clause it rests on.

| code | name | test |
|---|---|---|
| `N1` | definitional | one value's definition negates the other |
| `N2` | physical | violates physical law, material fact or causality |
| `N3` | capability | one REQUIRES what the other explicitly does not PROVIDE |
| `N4` | temporal | the required ordering or duration cannot be satisfied |
| `N5` | authority | a required legal or institutional power is absent |
| `N6` | scope | the two belong to disjoint scopes and cannot refer to one case |
| `N7` | implausible | possible in principle, but no real case instantiates it |

If no code fits, the answer is `Y`. "It feels wrong" is not a code.

## The occurrence test — how to apply N7 without reopening the ambiguity

`N7` exists because a solution space must exclude what does not happen, not only what
cannot happen. It is the clause that caused 45% of the disagreements, so it carries
one mandatory test:

> **Ask "does this combination occur?" — never "does it work well?"**

A combination that occurs but performs badly is `Y`. `N7` requires stating what would
have to be true for it to occur and why that is not the case. If you cannot state
that, it is not `N7`, it is `Y`.

| pair | works well? | occurs? | verdict |
|---|---|---|---|
| hand crew x erratic gusting | no, dangerous | yes, routinely | `Y` |
| automatic load shedding x Days | no, did not prevent it | yes | `Y` |
| contact tracing x refugee camp | poorly | yes | `Y` |
| blood screening x closed institution | fine | yes | `Y` |
| black start x Minutes | n/a | **no** - requires total collapse | `N4` |
| artisanal boat x VMS | n/a | **no** - no transponder fitted | `N3` |

## Citation requirement — what makes this auditable

An `N` must quote the **exact substring** from the relevant `requires` / `provides`
text. `validate_matrices.py` re-checks that the quoted string still appears verbatim,
so a label cannot drift away from its evidence:

```json
"verdict_reasons": {
  "FLIGHT PHASE|FAILED SUBSYSTEM": {
    "1,1": {"code": "N2",
            "requires": "combustion and turbomachinery under load",
            "blocked_by": "vehicle stationary, crew and ground systems connected"}
  }
}
```

A cell marked `N` with no reason, a bad code, or a quotation that is not present is a
**hard validation failure**. The dataset cannot be built until it is fixed.

## Worked re-decisions under v2

| pair | v1 | v2 | why |
|---|---|---|---|
| black start x Minutes | N | **N4** | black start *requires* total collapse; restoration in minutes contradicts it |
| automatic load shedding x Days | N | **Y** | shedding does not prevent a days-long outage; merely not the usual outcome |
| hand crew x erratic gusting | N | **Y** | dangerous and ineffective, but crews are present in gusting conditions |
| artisanal small boat x VMS | N | **N3** | `provides: no monitoring equipment` vs `requires: a transponder fitted` |
| no legal basis x seizure and prosecution | N | **N5** | prosecution requires a court; piracy confers none |
| pad before ignition x propulsion | Y | **N2** | `requires: combustion under load` cannot hold before ignition |
| contact tracing x refugee camp | N | **Y** | hard and often defeated, but done in practice |
