# Start here: model-improvement plan review, 16 August 2026

This directory is a review packet, not a new model release. It gives a reviewer
enough context to judge whether the proposed next experiments are useful,
scientifically bounded, and proportionate to the current data. No raw
trajectory, topology, ligand identity, per-system assay label, or out-of-fold
prediction row is included.

```text
PACKET_MARKER = LIGAMD_MODEL_PLAN_REVIEW_20260816
REVIEW_REF    = codex/model-plan-review-context
BASE_COMMIT   = 593c3bb
```

The intended reviewer may know machine learning but may know nothing about the
internal project history. The four statements below establish the common
ground.

1. The **Trajectory Analysis Toolkit** reads three validated LiGaMD production
   replicas, identifies one geometry-defined feature episode, selects 512 real
   saved frames from each replica, and produces Static20 and Dynamic10.
2. The optional **Experimental pKoff Predictor** maps those features to an
   assay-derived label. It does not infer a physical dissociation rate from
   biased simulation time.
3. The frozen development panel has 31 systems but only 27 independent
   exact-ligand groups. The same panel has already supported several model and
   representation experiments, so a lower error on that panel remains
   development evidence.
4. The immediate plan has two separate questions. One track tries to qualify
   genuinely new ligand groups for a locked evaluation. The other asks whether
   the order of existing P512 frames contains information that Dynamic10 may
   have compressed away.

## What a GitHub-only review can decide

A reviewer can inspect the public contracts, source code, model card, aggregate
N31 score tables, privacy boundary, and the full proposed decision logic. This
is enough to review:

- whether the two tracks answer different, falsifiable questions;
- whether the proposed train/test separation prevents obvious label leakage;
- whether the temporal control actually tests frame order;
- whether the stop rules prevent an open-ended model search;
- whether the proposed scientific claims stay within the available evidence;
- whether any check or experiment can be removed without weakening the answer.

## What a GitHub-only review cannot decide

Some statements in [the evidence summary](current-evidence.md) come from
private, aggregate-only run receipts. The corresponding per-system labels,
ligand identities, raw trajectories, and out-of-fold predictions are excluded
by [the data notice](../../../DATA_NOTICE.md). A GitHub-only reviewer therefore
cannot independently reproduce those private aggregate numbers or certify that
a particular candidate system passes admission.

That limitation should produce `NOT_EVALUABLE_FROM_PUBLIC_REPO`, not a guessed
answer. It does not prevent review of the plan that will govern those private
inputs.

## Reading order

Read these files in order.

1. [Current evidence and status](current-evidence.md) distinguishes public
   replayable evidence, public aggregate evidence, private aggregate statements,
   proposals, and blocked work.
2. [Proposed two-track execution plan](proposed-plan-zh.md) states the exact
   objective, contracts, decision rules, and stop points.
3. [Frozen data-free P512 sequence contract](p512-sequence-contract-v1.json)
   fixes the 11-channel order, units, provenance, missing-value policy,
   fold-local scaling boundary, and label-independent shuffle manifest before
   implementation. It contains no trajectory or label row.
4. [Chinese reviewer prompt](reviewer-prompt-zh.md) gives the requested review
   format and forces the reviewer to mark unsupported claims as not evaluable.

Then use the repository itself to inspect the relevant authority.

- [Repository README](../../../README.md) explains the user-facing route.
- [Model card](../../../MODEL_CARD.md) defines the target, grouped validation,
  public N31 results, and applicability limits.
- [Endpoint and P512 explanation](../../endpoint-and-sampler.md) describes the
  geometry and selected-frame semantics.
- [Method evidence ledger](../../method-evidence.md) separates literature ideas
  from project-specific choices.
- [Existing model-development roadmap](../../model-development-roadmap.md)
  records the earlier decision sequence. The present packet proposes one
  bounded development-only temporal smoke before independent-group confirmation;
  this is a proposal to review, not a silent rewrite of that roadmap.
- [Code map](../../code-map.md) connects the documented workflow to source.
- [Public N31 scorecard](../../../ligamd_pkoff/resources/models/experimental_n31_registry_v1/model_scoreboard.tsv)
  and [paired Dynamic10 table](../../../ligamd_pkoff/resources/models/experimental_n31_registry_v1/paired_dynamic_increment.tsv)
  are the public aggregate model evidence.

## Current operational status

```text
PACKET_STATUS                         = PROPOSED_FOR_REVIEW
MODEL_RUN_AUTHORIZED                  = NO
SEQUENCE_SERIALIZER_IMPLEMENTED       = NO
SEQUENCE_INPUT_CONTRACT_FROZEN         = YES
TEMPORAL_GRU_EXECUTED                 = NO
LOCKED_EVALUATION_ELIGIBLE_GROUPS     = 0
NEW_GROUP_LABELS_ALLOWED_IN_TRAINING  = NO
ENDPOINT_SELECTED_AS_UNIVERSAL_RULE   = NO
SAMPLER_SELECTED_AS_SCIENTIFIC_WINNER = NO
MODEL_SELECTED_AS_FINAL               = NO
PHYSICAL_KOFF_ESTIMATED               = NO
```

The practical handoff is simple. Connect ChatGPT to this repository or give it
the pull-request URL, then ask it to read this file and follow
[reviewer-prompt-zh.md](reviewer-prompt-zh.md). The reviewer does not need the
private workbook to judge whether the proposed process is coherent. It would
need authorized private evidence to verify workbook-specific counts or to
certify individual candidates.
