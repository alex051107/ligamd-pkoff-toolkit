# Prospective Dynamic10 validation protocol

This protocol asks one question on ligand identities that were not used to
build the bundled N31 models:

> Does frozen P512 Dynamic10 reduce experimental-pKoff prediction error beyond
> the same frozen Static20 Ridge model?

It does not fit or tune a model, select a family after seeing results, estimate
physical koff, or turn reused development data into prospective evidence.
Predictions and cohort membership are frozen before numerical experimental
labels are connected.

## Phase 1: freeze without labels

Prepare a tab-separated cohort table with these columns:

| Column | Meaning |
| --- | --- |
| `system_id` | Unique system identifier used by its feature receipt. |
| `condition_id` | Simulation condition recorded by the receipt. |
| `exact_ligand_group` | Exact-identity group used as the independent evaluation unit. |
| `audited_ligand_identity` | Audited identity key used for overlap checks. |
| `protein_target` | Target text recorded by the feature receipt. |
| `target_family` | Predeclared family used for the robustness summary. |
| `features_json_path` | Label-blind output of `ligamd-pkoff featurize`; relative paths resolve from the cohort table. |

Prepare a separate authorised development-identity TSV containing
`audited_ligand_identity` for every N31 development identity. The public
repository does not redistribute that private campaign ledger.

```bash
python tools/prospective_dynamic_validation.py freeze \
  --cohort prospective_cohort.tsv \
  --development-identities authorised_n31_identities.tsv \
  --output-dir prospective_freeze
```

The command stops unless:

- at least 15 exact-ligand groups remain after label-blind feature and quality
  preparation;
- prospective identities are absent from the supplied N31 identity ledger;
- audited identity and exact-ligand group form a one-to-one mapping;
- each exact-ligand group has one predeclared target-family assignment;
- system identifiers are unique and feature-receipt system, condition, and
  target fields agree with the cohort;
- neither the cohort nor a feature receipt contains a top-level label or koff
  field; and
- all four frozen profiles produce finite, non-`OUT_OF_SCOPE` predictions.

The unique primary pair is `static20_ridge` and
`combined30_p512_ridge`. The Random Forest pair is frozen at the same time as
a sensitivity analysis; it cannot replace Ridge when its result looks better.
`CAUTION_OUTSIDE_OBSERVED_RANGE` predictions remain in the cohort, with their
reasons in the ledger. Removing a row after seeing that warning or its later
error would change the prospective question.

Outputs are `frozen_cohort.tsv`, `development_identity_snapshot.tsv`,
`prediction_ledger.tsv`, and `freeze_receipt.json`. The receipt records one
combined digest for the three tables and one model-bundle digest. These files
may contain private chemical identities and must not be committed or
redistributed without explicit permission.

After a successful freeze, do not add, remove, replace, or relabel systems,
groups, target families, profiles, or predictions. If preparation fails,
repair or redefine the cohort before labels are connected and create a new
freeze directory; the command refuses to overwrite an existing one.

## Phase 2: connect labels once

The label TSV contains exactly one finite experimental value for every frozen
system:

```text
system_id    experimental_pKoff
```

```bash
python tools/prospective_dynamic_validation.py evaluate \
  --freeze-dir prospective_freeze \
  --labels prospective_labels.tsv \
  --output-dir prospective_result
```

Before reading labels, the evaluator checks the freeze status and the combined
digest of the frozen cohort, development snapshot, and prediction ledger. An
altered freeze aborts rather than creating a scientific result. Missing,
extra, duplicate, or non-finite label rows produce `NOT_EVALUABLE`; no group is
substituted.

The only primary difference is:

```text
group MAE(combined30_p512_ridge) - group MAE(static20_ridge)
```

Negative values favour Dynamic10. The evaluator reuses the repository's
existing group-MAE and paired group-bootstrap implementation; it does not
introduce a second uncertainty method. It writes:

- `paired_group_errors.tsv` for the primary pair and frozen RF sensitivity;
- `target_family_sensitivity.tsv` for the primary Ridge pair; and
- `prospective_decision.json` with the primary gates and bounded status.

## Frozen decision rule

The gate passes only when all four concepts pass:

| Concept | Frozen rule |
| --- | --- |
| Mean improvement | `Combined30 - Static20 <= -0.10` experimental-pKoff MAE. |
| Breadth | Strictly more than half of exact-ligand groups improve. |
| Uncertainty | The fixed-seed 95% paired group-bootstrap upper endpoint is below zero. |
| Robustness | No leave-one-group removal reverses the primary direction, and no major target family is consistently harmed. |

A major target family has at least three frozen exact-ligand groups. It is
consistently harmed only when every one of its group differences is positive.
A group dominates the primary direction when removing it reverses the sign of
the full paired mean.

The statuses are:

- `PASS_PROSPECTIVE_DYNAMIC_INCREMENT_GATE`: every rule passes;
- `FAIL_PROSPECTIVE_DYNAMIC_INCREMENT_GATE`: at least 15 paired groups are
  evaluable, but one or more rules fail; and
- `NOT_EVALUABLE`: the frozen cohort has fewer than 15 groups or the labels do
  not exactly and finitely cover its systems.

`FAIL` does not authorize retuning, switching to Random Forest, deleting a
difficult group, or trying features sequentially. It sends the project to the
single predeclared representation experiment. `NOT_EVALUABLE` means the
question was not answered; it is not evidence against Dynamic10.

## Claim ceiling

Passing would support the bounded claim that frozen P512 Dynamic10 added
prospective predictive value over frozen Static20 Ridge on this audited cohort.
It would not establish universal cross-target transfer, recover physical koff,
select P512 as universally optimal, or validate an untested typed or neural
representation.
