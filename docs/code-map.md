# Code map

Most users should run the installed <code>ligamd-pkoff</code> command, starting
with the [synthetic campaign tutorial](tutorial-synthetic-campaign.md). The
Python paths below are provided for inspection, reuse, and scientific review.
They are repository paths, not a requirement to run every module by hand.

## Ordinary public workflow

| Repository path | Role in the workflow | Who calls it |
| --- | --- | --- |
| [ligamd_pkoff/cli.py](../ligamd_pkoff/cli.py) | Parses <code>ligamd-pkoff featurize</code>, <code>predict</code>, and <code>evaluate</code>. | An ordinary user through the installed CLI. |
| [scripts/koff_ml/toolkit.py](../scripts/koff_ml/toolkit.py) | Orchestrates the public route. It freezes a reference, validates all three replicas, creates Static20 and Dynamic10, optionally requests the unselected typed challenger, pools replicas, and writes receipts. | Called by the CLI. Do not call its internal helpers one by one unless developing the toolkit. |
| [scripts/koff_ml/shared_reference.py](../scripts/koff_ml/shared_reference.py) | Builds the outcome-blind reference for ligand atom order, pocket residues, native contacts, and alignment atoms. | Called during <code>featurize</code>. |
| [scripts/koff_ml/reference_features.py](../scripts/koff_ml/reference_features.py) | Reads dense NetCDF coordinates and creates PBC-aware per-frame geometry and contact channels. | Called during <code>featurize</code>. |
| [scripts/koff_ml/endpoint_two_metric.py](../scripts/koff_ml/endpoint_two_metric.py) | Applies the two-metric, 100-saved-frame endpoint rule to a dense trace. | Called during <code>featurize</code>. |
| [scripts/koff_ml/p512_sampler.py](../scripts/koff_ml/p512_sampler.py) | Selects 512 unique real source frames from the episode using cumulative multiblock path change. | Called during <code>featurize</code>. |
| [scripts/koff_ml/coordinate_shards.py](../scripts/koff_ml/coordinate_shards.py) | Materialises the selected coordinates, periodic boxes, and source-index map into a compact coordinate union. | Called during <code>featurize</code>. |
| [scripts/koff_ml/prediction_first_coordinate_features.py](../scripts/koff_ml/prediction_first_coordinate_features.py) | Calculates selected-frame distances, residue contacts, RMSD channels, and the coordinate-level A/I/P_ONLY/B bookkeeping labels. | Called during <code>featurize</code>. |
| [scripts/koff_ml/normalized_progress_grid_v1.py](../scripts/koff_ml/normalized_progress_grid_v1.py) | Places selected observations on the frozen 512-point left-hold progress grid. | Called during Dynamic10 construction. |
| [scripts/koff_ml/prediction_first_geometric_contact_v2.py](../scripts/koff_ml/prediction_first_geometric_contact_v2.py) | Compresses each replica into the 41-field Core41-v2 dictionary. | Called during Dynamic10 construction. |
| [scripts/koff_ml/typed_interactions.py](../scripts/koff_ml/typed_interactions.py) | Optionally calculates the separate 32-field CORE8 interaction challenger from the existing P512 coordinate union. It does not read NetCDF again or change Current30. | Called only when `featurize --typed-interactions` is requested and the optional dependencies are installed. |
| [scripts/koff_ml/complete_static_geometric.py](../scripts/koff_ml/complete_static_geometric.py) | Defines Static20 field order and calculates bound-reference geometry. RDKit ligand descriptors are computed by the public bridge in toolkit.py. | Called during <code>featurize</code>. |
| [endpoint_v2.json](../ligamd_pkoff/resources/contracts/endpoint_v2.json) | Frozen endpoint settings read by the public route. | Bundled public resource. |
| [p512_sampler_v1.json](../ligamd_pkoff/resources/contracts/p512_sampler_v1.json) | Frozen P512 budget and path-sampling settings. | Bundled public resource. |
| [model_registry.json](../ligamd_pkoff/resources/models/experimental_n31_registry_v1/model_registry.json) | Describes the four bundled **experimental** baselines, their feature order, and their scope warnings. | Read by <code>predict</code>. |

The public route is intentionally one-way. <code>featurize</code> never reads
an experimental label, and <code>predict</code> never recomputes molecular
geometry. The [feature reference](feature-definitions.md) and
[endpoint explanation](endpoint-and-sampler.md) describe the scientific
meaning of each hand-off.

## Supporting modules a developer may inspect

| Repository path | Purpose | Public-use status |
| --- | --- | --- |
| [scripts/koff_ml/g0_trace_summary_v2.py](../scripts/koff_ml/g0_trace_summary_v2.py) | Defines the frozen state-criteria dictionary and the related 34-field full-trace control used by the Core41-v2 semantic contract. | Support library. It is not an ordinary CLI command. |
| [scripts/koff_ml/state_labels.py](../scripts/koff_ml/state_labels.py) | Supplies the same A/I/P_ONLY/B bookkeeping logic for a dense pandas reference table. | Support library for developers and audits. The coordinate route uses the matching function in prediction_first_coordinate_features.py. |
| [scripts/koff_ml/prediction_first_coordinate_materialization.py](../scripts/koff_ml/prediction_first_coordinate_materialization.py) | Validates coordinate atom order before coordinate-derived features are materialised. | Support library. |
| [scripts/koff_ml/io.py](../scripts/koff_ml/io.py) | Provides small JSON, CSV, and identity-file helpers used in receipts. | Support library. |
| [scripts/koff_ml/serialization_compat.py](../scripts/koff_ml/serialization_compat.py) | Supplies a narrowly scoped compatibility alias so the first four bundled joblib files can load. | Used automatically by <code>predict</code>. Do not invoke directly. |

## Developer-only supervised-model tools

These files are deliberately separate from trajectory processing. They need an
authorised, identity-matched supervised table and grouped fold map. They are
not required to turn a new trajectory into Current30 or to apply a bundled
baseline.

| Repository path | What it does | Use it when |
| --- | --- | --- |
| [scripts/koff_ml/fold_local_spearman_reducer.py](../scripts/koff_ml/fold_local_spearman_reducer.py) | Removes constant fields and clusters strongly correlated fields using **training rows only**. | Refitting or auditing a supervised model with a documented group split. |
| [scripts/koff_ml/run_dual_endpoint_combined30_matrix_v1.py](../scripts/koff_ml/run_dual_endpoint_combined30_matrix_v1.py) | Runs a matched, grouped classical-model comparison across endpoint arms and samplers. | Reproducing an authorised research comparison, not routine prediction. |
| [scripts/koff_ml/build_experimental_model_registry.py](../scripts/koff_ml/build_experimental_model_registry.py) | Rebuilds only the frozen N31 registry from its separately authorised panel, fold map, reused OOF file, endpoint contract, and bundled provenance receipt. It rejects any input digest or held-out-fold drift. | Auditing the historical N31 release. It is not a generic trainer for a new cohort. |
| [tools/build_paired_dynamic_increment.py](../tools/build_paired_dynamic_increment.py) | Summarises aggregate Static20-versus-Combined30 out-of-fold differences without publishing per-system labels. | Updating public aggregate evidence after an authorised comparison. |

The original campaign benchmark runner, the historical five-condition endpoint
sidecar, and alternative sampler experiments are intentionally not part of this
repository. They would blur the single public endpoint-v2 and P512 path with
internal research history.
