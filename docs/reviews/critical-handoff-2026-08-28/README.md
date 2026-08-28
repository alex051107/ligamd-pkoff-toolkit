# pKoff workflow critical-review handoff, 28 August 2026

This packet asks a fresh reviewer to look for reasons the present workflow may
be wrong. Its purpose is not to defend a model result or to select a model.
It records the current state of the trajectory-to-feature workflow, the latest
aggregate comparison, and the questions that still need an independent
methodological judgment.

No raw trajectory, topology, private label table, ligand-identity crosswalk,
per-system prediction, local path, or cluster credential is included here.
The aggregate results below are supplied for methodological review. A reviewer
with GitHub access can inspect the public code, contracts, an aggregate N34
evidence table, a de-identified admission audit, and a timestamped 1EBY cohort
receipt. It still cannot reproduce private cohort membership or the aggregate
N34 calculation from this repository alone.

## Bottom line

The workflow is ready to process a system through a fixed label-blind route.
The evidence does not yet show that the ten trajectory summaries in Combined30
reliably improve pKoff prediction over the 20 static descriptors in Static20.

On the current historical development cohort of 34 systems in 30 exact-ligand
groups, the group-equal leave-one-group-out Ridge MAE was 0.8162 for Static20
and 0.7762 for Combined30. The paired difference, Combined30 minus Static20,
was -0.0400 pKoff. A 5,000-resample paired group bootstrap interval was
[-0.1196, +0.0146] pKoff and 14 of 30 groups improved. The point estimate is
favourable to Combined30; the interval still permits no average improvement.
This is development-exposed cohort-composition sensitivity, not prospective
validation or model selection.

The machine-readable aggregate values and the frozen-run settings are in
[N34 aggregate evidence](n34-aggregate-evidence-v1.tsv). That table is a
public summary of a private execution receipt. It supplies no label,
per-system prediction, or identity mapping.

The open question is therefore precise. Is the present limitation chiefly a
small and heterogeneous supervised cohort, a loss of useful trajectory signal
when each path is reduced to Dynamic10, a mismatch between simulations and
assay labels, an endpoint/admission selection effect, or some combination of
these? The current work has separated these possibilities enough to test them,
but has not quantified all of them.

## What the workflow actually does

The project has two products.

1. The trajectory toolkit starts with three matching LiGaMD production
   replicas. It applies one geometry-based complete-exit rule without reading
   experimental labels, selects real saved frames along the observed path, and
   produces 20 bound-structure and chemistry descriptors (Static20) plus ten
   trajectory summaries (Dynamic10).
2. The optional predictor maps those features to an assay-derived experimental
   pKoff label. It does not calculate a physical dissociation rate from the
   simulation clock.

For a supervised comparison, the current contract keeps one production
condition per system, selects the earliest three replicas that pass the same
complete-exit rule, pools their Dynamic10 vectors only after feature
calculation, and evaluates system-level predictions by exact-ligand
leave-one-group-out splits. Feature reduction, scaling, and Ridge tuning occur
inside each training fold.

The endpoint rule is a project contract, not a universal biochemical definition
of dissociation. It requires both a displacement from the bound reference and
clearance from the whole protein for a persistence window. This deliberately
rejects trajectories that leave the original pocket while remaining associated
with the protein.

## What has been checked

### Historical systems and reusable route

A label-blind census examined 37 historically completed systems. Thirty-five
have a complete Current30 feature chain under the present three-replica
contract. Two did not reach three complete-exit passes. One otherwise complete
trajectory asset is excluded from supervised counting because it duplicates
another asset and its assay-ligand identity does not close. That leaves the 34
independent historical systems used in the current N34 sensitivity comparison.

This result should not be simplified to “only 34 systems exist.” A larger
known roster contains systems at earlier stages, systems with production files
but no closed assay or identity match, and systems awaiting current production
output. The census distinguishes those states instead of silently dropping
them. [The public admission audit](admission-audit-aggregate-v1.tsv) publishes
these stage counts without publishing the row-level records.

The reusable processing sequence is:

```text
fixed roster
  -> production-payload check
  -> one frozen condition and shared structural reference
  -> label-blind complete-exit scan of the pre-registered attempts
  -> earliest three endpoint-passing replicas
  -> P512 real-frame selection
  -> Static20 + Dynamic10 = Current30
  -> independent label, identity, duplicate-asset, and development-exposure checks
  -> exact-ligand leave-one-group-out comparison
```

The first six stages must not inspect pKoff. A system changes MAE only after
the later identity and label gates also close.

### Latest historical model comparison

The current N34 comparison holds the endpoint, P512 sampling, replica pooling,
Static20, Dynamic10, Ridge family, and exact-ligand validation rule fixed. It
changes only the historical development cohort by adding one already
development-exposed system to the prior N33 composition.

| Feature set and model | Systems | Exact-ligand groups | Group-equal MAE |
| --- | ---: | ---: | ---: |
| Static20 + Ridge | 34 | 30 | 0.8162 |
| Combined30 + Ridge | 34 | 30 | 0.7762 |

The paired contrast above is the primary result to audit. It is not evidence
that the extra ten features will improve future systems. It also cannot show
that the original trajectories have no information. It only evaluates this
particular summary representation on a repeatedly used development cohort.

Earlier N31 comparisons covered Ridge, random forest, boosting and several
bounded representation/model variants. Their Dynamic10 increments were small,
unstable across methods, or unfavourable. The public review packet from 16
August records their aggregate evidence and the earlier proposed temporal
control. Read it as history, not as the current N34 result.

### New-system admission

The new 1EBY campaign is deliberately not mixed into the historical score.
Its initial, frozen 18-replica cohort completed with only one complete-exit
pass. Under the exact-three rule it cannot create P512, Current30, a pKoff
prediction row, or an MAE comparison. No threshold, sampler, seed, or model
was changed to obtain another pass.

Six additional 1EBY productions were submitted before that initial endpoint
result. [The cohort receipt](1eby-cohort-receipt-v1.md) gives the scheduler
identifiers, submission times, first-read time, and the fixed no-mixing rule.
It records a separate same-condition production cohort, not a new independent
ligand group and not a retroactive proof of prospective design. At the
receipt's last scheduler readback those routes had not entered the feature or
model path. If they finish, they must pass the existing endpoint gate and
exact-three selector as their own cohort before any materialisation. They must
not reopen the completed 18-replica result or be pooled with it post hoc. A
reviewer should explicitly judge whether the recorded time order is sufficient
to rule out a rescue analysis.

## What remains unresolved

The following are the actual possible bottlenecks. They should be treated as
competing explanations, not as settled causes.

| Candidate explanation | What current evidence says | What would test it without reopening the model search |
| --- | --- | --- |
| Too few independent ligand groups | Thirty groups give an imprecise paired contrast, and N34 is development-exposed. | Run the frozen Ridge pair once on eligible, development-unexposed exact-ligand groups. |
| Dynamic10 loses useful information | Current summaries do not show a stable increment. Ordered frame information has not been evaluated by a supervised temporal model. | One fixed ordered-versus-interior-shuffled sequence feasibility test, if separately authorised. |
| Label and simulation mismatch | Protein family, assay, construct, chemical state, and protocol differences are not separately quantified. Family holdout did not establish transfer. | Close label/identity/protocol evidence before a system enters the locked comparison. |
| Endpoint/admission selection | Only trajectories passing the fixed complete-exit contract enter Current30. The impact of that selection on pKoff associations is not quantified. | Audit pass/fail rates and candidate exclusions before interpreting the supervised cohort. |
| A better generic tabular learner is missing | Several conventional learners and bounded alternatives have already been compared on N31. | Do not reopen a model tournament on N34; reserve a predeclared challenger for a new independent cohort. |

The previous proposed temporal feasibility probe is not a result. It keeps the
same P512 frame multiset, fixes the first and endpoint frames, and shuffles
only the 510 internal frames in the matched control. It asks whether internal
order carries predictive information beyond the same frames in an incorrect
order. It does not ask whether simulation frames measure physical time or
physical kinetics. No GRU result is claimed in this packet.

## What a critical reviewer should challenge

The reviewer should try to invalidate the present workflow, especially by
asking:

- Does the exact-three complete-exit criterion introduce a scientifically
  important selection bias, and is the proposed audit enough to expose it?
- Does the new 1EBY cohort separation genuinely preserve a prospective test, or
  does running more pre-existing replicas become a data-dependent rescue?
- Are exact-ligand groups the right unit for all likely leakage paths, given
  protein-family, assay, construct, and chemical-state variation?
- Are the Static20 controls and fold-local preprocessing sufficient for the
  stated Dynamic10 increment question?
- Is the paired group bootstrap appropriate for 30 groups, and what failure
  modes would make its interval misleading?
- Does the ordered-versus-shuffled control isolate interior order without
  accidentally testing endpoint placement, P512 rank, or another shortcut?
- Which current component should be removed as unnecessary rather than adding
  another model, feature family, or admission state?
- What minimum aggregate evidence could make the public review more auditable
  without publishing private trajectories, labels, or identity mappings?

## Reading order for an independent review

1. Read this handoff.
2. Inspect the [N34 aggregate evidence](n34-aggregate-evidence-v1.tsv),
   [admission audit](admission-audit-aggregate-v1.tsv), and [1EBY cohort
   receipt](1eby-cohort-receipt-v1.md).
3. Read the earlier [model-plan review packet](../model-improvement-plan-2026-08-16/README.md), including its evidence ledger and its data boundary.
4. Read the current [P512 interior-shuffle decision](../../p512-sequence-v2-interior-shuffle.md).
5. Inspect [the endpoint and sampler explanation](../../endpoint-and-sampler.md), [the method-evidence ledger](../../method-evidence.md), [the model-development roadmap](../../model-development-roadmap.md), the [model card](../../../MODEL_CARD.md), and the [public score tables](../../../ligamd_pkoff/resources/models/experimental_n31_registry_v1/).
6. Use the paste-ready [Chinese review prompt](chatgpt-pro-critical-review-prompt-zh.md).

The public code can be reviewed directly. Any claim that requires private
records should be labelled `NOT_EVALUABLE_FROM_PUBLIC_REPO`, followed by the
smallest aggregate or redacted evidence needed to decide it.
