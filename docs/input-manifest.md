# Input-manifest reference

`ligamd-pkoff featurize` reads one JSON manifest for one system and one
simulation condition. The manifest identifies the structural reference, the
three production replicas, the ligand used for static descriptors, and the
saved-coordinate cadence. It contains no experimental pKoff value and the
feature command never reads a label.

Use [the real-input how-to](prepare-real-inputs.md) when you are preparing
files. This page describes what the command accepts and the states in which it
stops.

## Complete schema

```json
{
  "schema_version": "ligamd_pkoff_toolkit_input_v1.0",
  "system_id": "your-stable-system-name",
  "condition_id": "your-simulation-condition",
  "protein_target": "a-clear-human-readable-target-name",
  "saved_frame_interval_ps": 1.0,
  "canonical_bound_pdb": "canonical_bound.pdb",
  "canonical_topology": "canonical.parm7",
  "ligand": {
    "resname": "LIG",
    "resid": "501",
    "chain": "A",
    "smiles": "CCO"
  },
  "replicas": [
    {"replica_id": "r1", "trajectory": "r1.nc", "pdb": "r1.pdb", "topology": "r1.parm7"},
    {"replica_id": "r2", "trajectory": "r2.nc", "pdb": "r2.pdb", "topology": "r2.parm7"},
    {"replica_id": "r3", "trajectory": "r3.nc", "pdb": "r3.pdb", "topology": "r3.parm7"}
  ]
}
```

## Top-level fields

| Field | Type and unit | Required | Meaning and validation |
| --- | --- | --- | --- |
| `schema_version` | string | yes | Must be exactly `ligamd_pkoff_toolkit_input_v1.0`. It prevents the command from guessing an older layout. |
| `system_id` | non-empty string | yes | Stable local name for one system. It is written to receipts and feature outputs. It is not a chemical identity key and it is never an experimental label. |
| `condition_id` | non-empty string | yes | Name for the simulation condition represented by all three replicas. It is recorded for traceability and does not alter geometry. |
| `protein_target` | non-empty string | yes | Human-readable target name. Prediction uses a literal match only as an applicability warning. It is not a protein-family generalization test. |
| `saved_frame_interval_ps` | finite positive number, ps | no, defaults to `1.0` | Physical time between adjacent *saved coordinate frames*. It is not the molecular-dynamics integration time step. One value applies to every replica. The endpoint rule counts 100 saved frames, so its physical duration is `100 × saved_frame_interval_ps`. |
| `canonical_bound_pdb` | existing PDB path | yes | Outcome-blind bound structure used once to freeze the ligand atom order, pocket residues, native contacts, and pose-alignment coordinates. |
| `canonical_topology` | existing topology path | yes | Topology paired with `canonical_bound_pdb`. Its identity is recorded in the shared reference and must remain compatible with the canonical atom order. |
| `ligand` | JSON object | yes | Selector for one structural ligand and chemistry input for the Static10 ligand descriptors. See the next table. |
| `replicas` | array of exactly three objects | yes | The three independent production replicas. Each object supplies a trajectory, a PDB atom map, and a matching topology. |

## Ligand fields

| Field | Type | Required | Meaning and validation |
| --- | --- | --- | --- |
| `ligand.resname` | non-empty string | yes | Residue name of the structural ligand. Matching is case-insensitive and the resolved selector is frozen in the shared reference. |
| `ligand.resid` | string or number | conditionally | Residue identifier used to narrow the selector. Supply it when `resname` occurs in more than one residue. |
| `ligand.chain` | string | conditionally | Chain identifier used to narrow the selector. Supply it when chain information is needed to make the ligand selector unique. |
| `ligand.smiles` | string | at least one of `smiles` or `sdf` | Audited molecular input passed to RDKit for the ten ligand descriptors. If both are supplied, the current implementation uses `smiles`. |
| `ligand.sdf` | existing SDF path | at least one of `smiles` or `sdf` | Alternative audited molecular input passed to RDKit when a SMILES string is not used. |

The structural selector must match one and only one ligand residue in the
canonical PDB. A zero-match or multi-residue match is a hard error. The public
toolkit does not infer whether a SMILES or SDF describes that selected PDB
residue, so that chemical identity must be audited before the run.

## Replica fields

Each object in `replicas` has these fields.

| Field | Type | Required | Meaning and validation |
| --- | --- | --- | --- |
| `replica_id` | non-empty string | yes | A unique identifier within this manifest. It labels receipts and output subdirectories. |
| `trajectory` | existing NetCDF coordinate path | yes | Production coordinates. The formal path expects real saved coordinates and compatible periodic-cell metadata; it does not interpolate invented frames. |
| `pdb` | existing PDB path | yes | Atom identity and order for the trajectory. It must be compatible with the topology and frozen shared reference. |
| `topology` | existing topology path | yes | Topology for this replica. It is checked against the frozen reference so that coordinate positions do not silently acquire a different atom meaning. |

`replica_id` values and resolved trajectory paths must be distinct, and there
must be exactly three entries. Reusing one trajectory under multiple replica
names fails before feature extraction.
The bundled Combined30 profiles pool one Dynamic10 vector from each of exactly
three endpoint-passing replicas. The command never converts two passing
replicas into a three-replica average.

## Path resolution and file rules

Every path may be absolute or relative. A relative path is resolved from the
directory that contains the manifest, then converted to an absolute path for
the run receipt. `~` is expanded before this resolution. Each supplied path
must exist and be a regular file before extraction starts.

The canonical PDB must contain the periodic `CRYST1` information needed for
the shared reference. The current public geometry route supports an
orthorhombic periodic box. It fails rather than approximate a triclinic box.
The 8 Å pocket definition must also be no larger than half the shortest box
length, otherwise a minimum-image pocket selection would be ambiguous.

## Identity and cadence constraints

The manifest has two different identity jobs.

1. The canonical PDB and canonical topology define the atom identities used to
   create a shared structural reference. Their ordered atom/residue mapping is
   checked once. Every replica then has to preserve the canonical PDB atom
   schema and topology identity, and its NetCDF atom count must match before
   geometry is used.
2. The ligand SMILES or SDF defines the small-molecule graph used for Static10.
   It must be audited by the user against the structural ligand selector.

The manifest also gives one cadence value to all replicas. The endpoint uses
100 consecutive saved frames. A trajectory saved every 1 ps and a trajectory
saved every 10 ps therefore do not give the same physical persistence time,
even if both pass the literal 100-frame rule. The public registry keeps this
limitation visible as an applicability caution for Dynamic10 profiles.

## Output statuses and common stopping conditions

| Where it appears | Status or condition | Meaning |
| --- | --- | --- |
| `trajectory_status.tsv` | `PASS` | This replica contains a first 100-frame run in which both endpoint-v2 geometry conditions hold. It may contribute a Dynamic10 vector. |
| `trajectory_status.tsv` | `FAIL_NO_PERSISTENCE_CONFIRMED_COMPLETE_EXIT` | The replica did not contain the required consecutive run. It cannot contribute to a combined prediction. |
| `features.json` | `PASS_FEATURES_READY_FOR_EXPERIMENTAL_PREDICTION` | All three replicas passed, 512 real frames were selected per replica, and Current30 was written. This is feature readiness, not model validation. |
| `features.json` | `OUT_OF_SCOPE_NO_PREDICTION` | At least one required replica failed or required feature preparation could not complete. The code records the reason and does not make a partial prediction. |

The successful receipt contains the parsed endpoint and P512 executable
settings, not only their file paths. A bundled Combined30 model accepts the
dynamic vector only when those settings and exact-three-replica pooling match
the frozen training route. A Static20 model does not impose this trajectory
contract because its inputs do not use the trajectory-derived block.
| command error | invalid schema, missing file, ambiguous ligand, unsupported box, or atom/topology mismatch | The command stops before it creates a scientifically ambiguous feature vector. Correct the input rather than changing the result by hand. |
