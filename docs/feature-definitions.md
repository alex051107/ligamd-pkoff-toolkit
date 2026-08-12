# Current30 feature reference

Current30 is the 30-column engineering representation accepted by the bundled
Combined30 experimental-pKoff profiles. It has two deliberately different
parts.

- **Static20** contains 20 values available before production-trajectory
  analysis. Ten describe the input ligand. Ten describe the canonical bound
  structure and its frozen pocket.
- **Dynamic10** contains ten summaries of a validated trajectory episode.
  Each value is calculated independently for each of three replicas and the
  three corresponding values are averaged.

The literal field order is the public runtime contract in
[Dynamic10 field list](../scripts/koff_ml/toolkit.py#L88-L99),
[Static20 field list](../scripts/koff_ml/complete_static_geometric.py#L69-L95),
and the bundled [model registry](../ligamd_pkoff/resources/models/experimental_n31_registry_v1/model_registry.json).
This page expands those machine-readable names into a reference a researcher
can use while inspecting one new system. It does not claim that these 30
numbers are a final mechanistic feature set or that they estimate physical
koff.

## How one system becomes 30 numbers

The route below matters because a Dynamic10 value is never simply the value in
one chosen frame.

~~~text
canonical bound PDB + topology
    -> one frozen structural reference
three independently processed production replicas
    -> one endpoint-v2 episode and 512 real P512 frames per replica
    -> per-frame geometry, contacts, RMSD, and state labels
    -> one 41-field Core41-v2 vector per replica
    -> the fixed ten-field Dynamic10 view per replica
    -> arithmetic mean of matching Dynamic10 fields across three replicas
ligand chemistry + canonical bound reference
    -> one Static20 vector per system
Static20 + pooled Dynamic10
    -> one Current30 system row
~~~

The [endpoint and sampler explanation](endpoint-and-sampler.md) defines the
episode and the 512 real-frame selection. The geometry code then calculates
each frame-level quantity from coordinates, periodic-box information, and the
frozen reference. It does not read experimental pKoff while it does so.

### A note about progress windows

Episodes can contain different numbers of saved frames. To make an early or
late summary comparable across replicas, the code maps the 512 selected real
observations to a fixed 512-position progress grid. Position j is
(j + 0.5) / 512, so the grid covers the interior of the path from 0% to 100%.
If a grid position falls between two selected frames, it reuses the most recent
real frame. This is called **left-hold**. It does not create an interpolated
coordinate.

For example, <code>progress_050_080__mean</code> means the arithmetic mean of
the left-held values at grid positions from 50% inclusive to 80% exclusive.
The real endpoint-onset frame is still retained to close the episode, but it
has no ordinary progress-grid weight because the grid contains only interior
midpoints. The implementation is
[left_hold_on_normalized_grid](../scripts/koff_ml/normalized_progress_grid_v1.py#L102-L145)
and [interval_means_on_grid](../scripts/koff_ml/normalized_progress_grid_v1.py#L148-L199).

## Static20

Static20 is intentionally a control block. It lets a model use ligand and
bound-complex properties, and it makes it possible to ask later whether
Dynamic10 adds information beyond those starting differences. A static value
does not describe the observed dissociation route.

The first ten fields are calculated from the audited ligand SMILES or SDF with
RDKit after explicit hydrogens are removed. The next ten fields are calculated
once from the outcome-blind canonical bound PDB and its frozen structural
reference. A means angstrom; Da means dalton.

| Exact field name | Input and selector | Calculation | Unit | What it reflects | Code |
| --- | --- | --- | --- | --- | --- |
| <code>ligand_static__molecular_weight</code> | Parsed ligand molecule | RDKit MolWt after RemoveHs | Da | Molecular size | [RDKit Static10](../scripts/koff_ml/toolkit.py#L172-L205) |
| <code>ligand_static__logP</code> | Parsed ligand molecule | Crippen MolLogP | dimensionless | Hydrophobicity proxy | [RDKit Static10](../scripts/koff_ml/toolkit.py#L172-L205) |
| <code>ligand_static__HBD</code> | Parsed ligand molecule | Lipinski hydrogen-bond donor count | count | Donor capacity encoded by the input structure | [RDKit Static10](../scripts/koff_ml/toolkit.py#L172-L205) |
| <code>ligand_static__HBA</code> | Parsed ligand molecule | Lipinski hydrogen-bond acceptor count | count | Acceptor capacity encoded by the input structure | [RDKit Static10](../scripts/koff_ml/toolkit.py#L172-L205) |
| <code>ligand_static__TPSA</code> | Parsed ligand molecule | RDKit topological polar surface area | A<sup>2</sup> | Polar surface proxy | [RDKit Static10](../scripts/koff_ml/toolkit.py#L172-L205) |
| <code>ligand_static__rotatable_bonds</code> | Parsed ligand molecule | RDKit rotatable-bond count | count | Flexible bond count proxy | [RDKit Static10](../scripts/koff_ml/toolkit.py#L172-L205) |
| <code>ligand_static__rings</code> | Parsed ligand molecule | Lipinski ring count | count | Ring topology and rigidity proxy | [RDKit Static10](../scripts/koff_ml/toolkit.py#L172-L205) |
| <code>ligand_static__heavy_atoms</code> | Parsed ligand molecule | RDKit heavy-atom count | count | Molecular-size proxy | [RDKit Static10](../scripts/koff_ml/toolkit.py#L172-L205) |
| <code>ligand_static__formal_charge</code> | Parsed ligand molecule | RDKit formal charge | elementary-charge units | Charge state encoded in the supplied chemical structure | [RDKit Static10](../scripts/koff_ml/toolkit.py#L172-L205) |
| <code>ligand_static__fraction_CSP3</code> | Parsed ligand molecule | RDKit fraction of carbon atoms that are sp<sup>3</sup> | fraction | Saturation and three-dimensional-shape proxy | [RDKit Static10](../scripts/koff_ml/toolkit.py#L172-L205) |
| <code>initial_geometry__minimum_protein_ligand_heavy_distance_A</code> | Canonical ligand and all frozen 8 A pocket residues | Smallest canonical ligand-heavy-atom to pocket-residue-heavy-atom distance | A | Closest initial binding geometry | [bound geometry](../scripts/koff_ml/complete_static_geometric.py#L340-L425) |
| <code>initial_geometry__native_contact_residue_count</code> | Frozen 8 A pocket residues | Count residues marked native in the shared reference, meaning canonical minimum ligand distance is at most 4.5 A | residues | Size of the original contact network | [bound geometry](../scripts/koff_ml/complete_static_geometric.py#L361-L425) |
| <code>initial_geometry__native_contact_residue_fraction_of_pocket</code> | Same frozen native and 8 A pocket sets | Native-contact residue count divided by total frozen pocket-residue count | fraction | Density of the original contact network within the reference pocket | [bound geometry](../scripts/koff_ml/complete_static_geometric.py#L361-L425) |
| <code>initial_geometry__pocket_8A_residue_count</code> | Canonical standard-amino-acid residues within 8 A of any ligand heavy atom | Count frozen pocket residues | residues | Pocket-size proxy | [shared reference](../scripts/koff_ml/shared_reference.py) and [bound geometry](../scripts/koff_ml/complete_static_geometric.py#L361-L425) |
| <code>initial_geometry__pocket_8A_heavy_atom_count</code> | Heavy atoms belonging to frozen 8 A pocket residues | Count frozen pocket heavy-atom indices | atoms | Atomic size of the starting pocket | [bound geometry](../scripts/koff_ml/complete_static_geometric.py#L387-L425) |
| <code>initial_geometry__pocket_8A_radius_of_gyration_A</code> | PBC-unwrapped frozen pocket heavy atoms | Unweighted root-mean-square distance from their geometric centroid | A | Spatial extent of the starting pocket | [bound geometry](../scripts/koff_ml/complete_static_geometric.py#L390-L425) |
| <code>initial_geometry__standard_protein_residue_count</code> | Standard-amino-acid ATOM residues in canonical PDB | Count unique residue keys | residues | Protein-size proxy | [bound geometry](../scripts/koff_ml/complete_static_geometric.py#L399-L425) |
| <code>initial_geometry__pocket_hydrophobic_residue_fraction</code> | Frozen pocket residue names | Hydrophobic-class residue count divided by pocket-residue count | fraction | Hydrophobic composition of the starting pocket | [classes and calculation](../scripts/koff_ml/complete_static_geometric.py#L109-L117) and [bound geometry](../scripts/koff_ml/complete_static_geometric.py#L375-L425) |
| <code>initial_geometry__pocket_polar_uncharged_residue_fraction</code> | Frozen pocket residue names | Polar-uncharged-class count divided by pocket-residue count | fraction | Polar uncharged composition of the starting pocket | [classes and calculation](../scripts/koff_ml/complete_static_geometric.py#L109-L117) and [bound geometry](../scripts/koff_ml/complete_static_geometric.py#L375-L425) |
| <code>initial_geometry__pocket_ionizable_residue_fraction</code> | Frozen pocket residue names | Ionizable-class count divided by pocket-residue count | fraction | Ionizable-residue composition of the starting pocket | [classes and calculation](../scripts/koff_ml/complete_static_geometric.py#L109-L117) and [bound geometry](../scripts/koff_ml/complete_static_geometric.py#L375-L425) |

The three pocket-composition fields are residue-class fractions. They do not
claim that a particular hydrogen bond, salt bridge, or pi interaction was
observed in a trajectory.

## Dynamic10

Every row below starts from the same 512 selected **real** frames in one
replica. The code first calculates the stated frame-level quantity, applies
the left-hold normalized-progress grid when the calculation names a progress
window, and then takes a mean or a state fraction. Finally, the public
workflow averages each named value across exactly three endpoint-passing
replicas. The per-frame calculations are in
[derive_coordinate_features](../scripts/koff_ml/prediction_first_coordinate_features.py#L285-L330);
the common 41-field summarisation is in
[summarize_geometric_contact_replica_v2](../scripts/koff_ml/prediction_first_geometric_contact_v2.py#L505-L598).

| Exact field name | Input and selector | Calculation | Unit | What it reflects | Code |
| --- | --- | --- | --- | --- | --- |
| <code>g50v2__pocket_com_distance_A__progress_000_050__mean</code> | Ligand heavy atoms and frozen pocket heavy atoms; 0% to 50% progress | PBC-aware geometric-centre distance per selected frame, then mean on the early grid window | A | Early movement away from the original pocket | [frame distance](../scripts/koff_ml/prediction_first_coordinate_features.py#L285-L330) and [summary](../scripts/koff_ml/prediction_first_geometric_contact_v2.py#L547-L580) |
| <code>g50v2__native_contact_fraction__progress_050_080__mean</code> | Frozen native residue set; 50% to 80% progress | At each frame, fraction of frozen native residues whose minimum ligand-heavy-atom distance is at most 4.5 A; then grid-window mean | fraction | Retention or loss of the initial contact network in the middle of the episode | [frame contacts](../scripts/koff_ml/prediction_first_coordinate_features.py#L285-L330) and [summary](../scripts/koff_ml/prediction_first_geometric_contact_v2.py#L547-L580) |
| <code>g50v2__ligand_pose_rmsd_A__progress_080_095__mean</code> | Ligand heavy atoms; frozen pocket heavy atoms; 80% to 95% progress | Proper Kabsch alignment of the frozen pocket to the canonical reference, apply that transform to the ligand, then ligand heavy-atom RMSD and window mean | A | Late ligand-pose change relative to the bound pocket | [frame pose](../scripts/koff_ml/prediction_first_coordinate_features.py#L285-L330) and [summary](../scripts/koff_ml/prediction_first_geometric_contact_v2.py#L547-L580) |
| <code>g50v2__ligand_pose_rmsd_A__progress_095_100__mean</code> | Same pose measurement; 95% to 100% interior grid support | Same pocket-aligned ligand RMSD, then late-window mean | A | Pose change close to the endpoint onset; a sensitivity field, not the endpoint itself | [frame pose](../scripts/koff_ml/prediction_first_coordinate_features.py#L285-L330) and [summary](../scripts/koff_ml/prediction_first_geometric_contact_v2.py#L547-L580) |
| <code>g50v2__global_min_heavy_distance_A__progress_095_100__mean</code> | Ligand heavy atoms and all standard-protein residue heavy atoms; 95% to 100% | Per-frame minimum over every ligand/protein heavy-atom pair, then late-window mean | A | How close the ligand remains to any protein surface near the end of the path | [frame contacts](../scripts/koff_ml/prediction_first_coordinate_features.py#L285-L330) and [summary](../scripts/koff_ml/prediction_first_geometric_contact_v2.py#L547-L580) |
| <code>g50v2__global_contact_fraction__progress_000_050__mean</code> | All standard protein residues; 0% to 50% progress | Contacted residues, each with minimum ligand distance at most 4.5 A, divided by all standard protein residues; then early-window mean | fraction | Early whole-protein association burden | [frame contacts](../scripts/koff_ml/prediction_first_coordinate_features.py#L285-L330) and [summary](../scripts/koff_ml/prediction_first_geometric_contact_v2.py#L547-L580) |
| <code>g50v2__global_contact_fraction__progress_080_095__mean</code> | Same whole-protein residue universe; 80% to 95% progress | Same global-contact fraction, then late-window mean | fraction | Residual whole-protein surface association before the endpoint | [frame contacts](../scripts/koff_ml/prediction_first_coordinate_features.py#L285-L330) and [summary](../scripts/koff_ml/prediction_first_geometric_contact_v2.py#L547-L580) |
| <code>g50v2__state_progress_occupancy__I</code> | Operational state label on all interior progress positions | Number of left-held grid positions labelled I divided by 512 | fraction | Portion of the path that is neither classified bound-like nor pocket-exited | [state rules](../scripts/koff_ml/prediction_first_coordinate_features.py#L428-L466) and [occupancy](../scripts/koff_ml/prediction_first_geometric_contact_v2.py#L565-L580) |
| <code>g50v2__state_progress_occupancy__P_ONLY</code> | Operational state label on all interior progress positions | Number of left-held grid positions labelled P_ONLY divided by 512 | fraction | Portion of the path that has left the original pocket but still has a whole-protein contact | [state rules](../scripts/koff_ml/prediction_first_coordinate_features.py#L428-L466) and [occupancy](../scripts/koff_ml/prediction_first_geometric_contact_v2.py#L565-L580) |
| <code>g50v2__ligand_internal_rmsd_A__progress_mean</code> | Ligand heavy atoms; all interior progress positions | Align the ligand to its own bound-reference coordinates, removing rigid translation and rotation; calculate ligand heavy-atom RMSD and take the full-grid mean | A | Internal ligand deformation along the episode | [frame pose](../scripts/koff_ml/prediction_first_coordinate_features.py#L285-L330) and [incremental summary](../scripts/koff_ml/prediction_first_geometric_contact_v2.py#L409-L500) |

P_ONLY means **pocket exited only**. It does not mean protein only and it does
not say that the frame is a physically validated metastable state.

## The A, I, P_ONLY, and B bookkeeping labels

The state labels are a compact way to turn several geometry channels into two
Dynamic10 occupancy fields. They are **feature bookkeeping rules**, not the
endpoint-v2 conditions. In particular, the endpoint requires an aligned
centroid displacement of at least 15 A and a whole-protein minimum heavy-atom
distance strictly greater than 10 A for 100 consecutive saved frames. State B
uses a different, lower 6 A global-distance threshold and has no persistence
rule. It must therefore never be substituted for endpoint-v2.

All state values use the frozen criteria in
[FROZEN_STATE_CRITERIA](../scripts/koff_ml/g0_trace_summary_v2.py#L63-L77).
For one selected frame, let d_pocket be the PBC-aware
ligand-to-frozen-pocket geometric-centre distance, Q_native the frozen-native
contact fraction, n_pocket the number of current frozen-pocket contacts,
n_global the number of current whole-protein contacted residues, and d_global
the nearest ligand/protein heavy-atom distance.

| Label | Numerical rule | Plain-language role |
| --- | --- | --- |
| <code>A</code> | d_pocket <= 6 A **and** Q_native >= 0.50 | Bound-like by both proximity to the original pocket and retention of at least half of its frozen native contacts. |
| <code>P_ONLY</code> candidate | d_pocket >= 15 A, Q_native <= 0.05, and n_pocket = 0 | The ligand has left the original pocket according to the frozen pocket criteria. |
| <code>B</code> | P_ONLY candidate **and** n_global = 0 **and** d_global >= 6 A | Clear of the whole-protein contact universe under the state bookkeeping thresholds. |
| <code>P_ONLY</code> final | P_ONLY candidate but not B | Outside the original pocket while still associated with another part of the protein surface. |
| <code>I</code> | Every remaining frame | Intermediate or ambiguous geometry under this deliberately coarse classification. |

The implementation initialises every frame as I, then assigns P_ONLY, then B,
and finally A. The resulting precedence is A > B > P_ONLY > I. The coordinate
implementation is
[coordinate_state_labels](../scripts/koff_ml/prediction_first_coordinate_features.py#L428-L466);
[operational_state_labels](../scripts/koff_ml/state_labels.py#L18-L52) provides
the same rule for dense-reference tables.

## Core41, Dynamic10, and fold-local preprocessing are different operations

These three stages can sound similar because all reduce information. They do
different jobs and occur at different times.

| Stage | What is fixed or fitted | Inputs used | Output | Why it is separate |
| --- | --- | --- | --- | --- |
| 1. Core41-v2 construction | Fixed 41-field coordinate dictionary. It contains 24 window means from six continuous channels, six endpoint-minus-start values, four state occupancies, and seven additional contact/conformation summaries. | One replica's selected coordinates, its real source indices, and frozen state rules. No pKoff labels. | One 41-value replica vector. | It retains a traceable, broader physical measurement dictionary. Its historical module prefix is g50v2; the current size is 41. |
| 2. Dynamic10 projection | Fixed literal subset listed in Dynamic10 field list. | That replica's Core41-v2 vector. No panel-wide variance, correlation, model coefficient, or pKoff label. | Ten named replica values, then one three-replica arithmetic mean. | It defines the public Current30 representation. It is a frozen design choice, not a data-driven winner-selection step. |
| 3. Fold-local reducer and scaler during supervised refitting | Learned separately inside each training split. Columns with variance at or below 1e-12 are removed. Remaining columns with absolute Spearman correlation at least 0.95 are complete-linkage clustered and one medoid is retained. StandardScaler is fit on that training split only. | Training feature rows only. The outer held-out exact-ligand group is excluded. | A fitted preprocessing path for that fold or a frozen bundled pipeline. | It avoids a test row influencing feature cleanup or scaling. It does not rename or redefine Current30. |

Core41 construction is implemented in
[prediction_first_geometric_contact_v2.py](../scripts/koff_ml/prediction_first_geometric_contact_v2.py#L505-L598).
The fixed Dynamic10 projection and the three-replica mean are in
[toolkit.py](../scripts/koff_ml/toolkit.py#L246-L276) and
[toolkit.py](../scripts/koff_ml/toolkit.py#L430-L470). The training-only
reducer is [fold_local_spearman_reducer.py](../scripts/koff_ml/fold_local_spearman_reducer.py),
and the developer-only grouped-refit route is
[run_dual_endpoint_combined30_matrix_v1.py](../scripts/koff_ml/run_dual_endpoint_combined30_matrix_v1.py#L319-L330).

The supplied predict command does not refit a reducer or use a new pKoff
table. It applies a pre-fitted experimental pipeline to a completed Current30
row. See the [model card](../MODEL_CARD.md) for the resulting claim boundary
and [method evidence](method-evidence.md) for the literature-supported
principles versus this project's fixed settings.
