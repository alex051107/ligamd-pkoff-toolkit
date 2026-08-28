# 1EBY cohort timing and no-mixing receipt

This is a redacted, public review receipt. It records timing and rules needed
to audit a possible rescue analysis. It contains no trajectory, topology,
experimental pKoff value, assay row, local or cluster path, or endpoint trace.

## Two separate cohorts

| Cohort | Scope | Frozen observation | Time authority |
| --- | --- | --- | --- |
| A | 18 completed same-condition attempts | 1 complete-exit pass; exact-three gate blocked; no coordinate materialisation | Endpoint selector written 2026-08-28 13:30:38 -04:00 |
| B | Six same-condition production attempts | Pre-existing at the time cohort A selector was written; no endpoint, feature, prediction, or MAE output at the scheduler snapshot | Scheduler submission and readback |

The cohort-A selector explicitly reported `BLOCKED_FEWER_THAN_THREE_ENDPOINT_PASS`
and directed that coordinate tasks not be created. Its result remains fixed.

## Cohort B scheduler evidence

The scheduler recorded these six submissions before cohort A's endpoint
selection file existed. Times use the scheduler's local recorded offset.

| Scheduler job | Replica slot | Submitted | Started | State at 2026-08-28 16:23:55 -04:00 |
| ---: | --- | --- | --- | --- |
| 65589488 | 1.1_19 | 2026-08-28 08:56:26 | 2026-08-28 08:56:29 | RUNNING |
| 65589489 | 1.1_20 | 2026-08-28 08:56:27 | 2026-08-28 09:03:27 | RUNNING |
| 65589490 | 1.1_21 | 2026-08-28 08:56:27 | 2026-08-28 09:04:27 | RUNNING |
| 65589491 | 1.1_22 | 2026-08-28 08:56:27 | 2026-08-28 09:17:19 | RUNNING |
| 65589492 | 1.1_23 | 2026-08-28 08:56:27 | 2026-08-28 09:17:52 | RUNNING |
| 65589493 | 1.1_24 | 2026-08-28 08:56:27 | 2026-08-28 09:21:03 | RUNNING |

This timing rules out one narrow claim: the six jobs were not first submitted
after the recorded cohort-A selector output. It does not prove that cohort B
is statistically independent, that it is a new ligand group, or that its
future endpoint pass rate is unbiased.

## Frozen rule if cohort B becomes terminal

1. Analyse only the six cohort-B attempts using the existing label-blind
   endpoint contract.
2. Select the earliest three cohort-B endpoint passes if, and only if, there
   are at least three.
3. Do not combine cohort-A and cohort-B passes to reach three.
4. Do not change endpoint thresholds, the sampler, model, seed, architecture,
   or optimisation settings because of either cohort's result.
5. Report cohort A and cohort B separately whether cohort B passes or fails.
6. Do not create a third 1EBY production cohort under this review receipt.

Even a cohort-B pass produces at most one additional system-level feature row.
It cannot by itself create a new independent exact-ligand group or establish a
new-group generalisation result.
