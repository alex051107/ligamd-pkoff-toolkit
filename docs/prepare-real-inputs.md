# How to prepare a real LiGaMD input set

Use this guide when you already have one bound complex and three production
replicas and want to create a valid manifest for `ligamd-pkoff featurize`. It
does not teach molecular-dynamics setup or decide whether a scientific system
is suitable for training. Those are scientific responsibilities outside the
public command.

For the exact field definitions, read the [input-manifest reference](input-manifest.md).
For the meaning of the exit rule and sampled frames, read the
[episode and sampler explanation](endpoint-and-sampler.md).

## 1. Put one system's files in one directory

Start with a directory whose manifest and trajectory files belong to one
simulation condition. A practical layout is:

```text
my-system/
  manifest.json
  canonical_bound.pdb
  canonical.parm7
  replica-1.nc
  replica-1.pdb
  replica-1.parm7
  replica-2.nc
  replica-2.pdb
  replica-2.parm7
  replica-3.nc
  replica-3.pdb
  replica-3.parm7
```

The three trajectories must be independent production replicas of the same
condition and must resolve to three different files. They must use one
saved-frame cadence because the one manifest cadence applies to all three
persistence checks.

## 2. Choose the canonical bound reference before inspecting outcomes

Choose a bound PDB that represents the intended starting complex, then use its
matching topology as `canonical_topology`. The code derives one shared ligand,
pocket, native-contact set, and alignment reference from these two inputs.

Use `ligand.resname`, `ligand.resid`, and `ligand.chain` to identify exactly
one ligand residue in the canonical PDB. Supply `resid` and `chain` whenever a
residue name alone could match more than one molecule. The command stops on an
ambiguous selector rather than picking one arbitrarily.

For a PDB whose chain column is blank, either omit `ligand.chain` when the
remaining selector is unique or set it to an empty string. The manifest reader
normalizes an explicitly blank chain to the PDB reader's `_` representation.

Before running the command, independently check that the SMILES or SDF used
for ligand descriptors represents the same chemical entity as the selected
structural ligand. The toolkit can parse that descriptor input, but a PDB atom
list alone is not a reliable chemical-identity proof.

## 3. Write the manifest with relative paths

Create `manifest.json` in the directory above. Relative paths are resolved
from the directory containing that manifest, so the layout can be moved as a
unit. Record the physical interval between saved coordinate frames in
`saved_frame_interval_ps`; do not enter the integration time step.

```json
{
  "schema_version": "ligamd_pkoff_toolkit_input_v1.0",
  "system_id": "example-system-001",
  "condition_id": "one-production-condition",
  "protein_target": "your-target-name",
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
    {"replica_id": "r1", "trajectory": "replica-1.nc", "pdb": "replica-1.pdb", "topology": "replica-1.parm7"},
    {"replica_id": "r2", "trajectory": "replica-2.nc", "pdb": "replica-2.pdb", "topology": "replica-2.parm7"},
    {"replica_id": "r3", "trajectory": "replica-3.nc", "pdb": "replica-3.pdb", "topology": "replica-3.parm7"}
  ]
}
```

## 4. Run feature extraction once

Choose a new output directory. The command refuses to overwrite an existing
one so that the receipts remain attributable to a single run.

```bash
ligamd-pkoff featurize \
  --manifest my-system/manifest.json \
  --output-dir my-system/feature-receipt
```

Read `trajectory_status.tsv` first. If all three replicas report `PASS`, then
`features.json` should report `PASS_FEATURES_READY_FOR_EXPERIMENTAL_PREDICTION`
and `system_features.tsv` contains Current30, the 20 static and 10 dynamic
numerical inputs. If one replica
does not pass, `features.json` reports `OUT_OF_SCOPE_NO_PREDICTION`; do not
replace the missing replica with a partial average.

## 5. Decide what to do with the result

Feature readiness is a trajectory-analysis result. It is not evidence that the
system has an experimental label, that a bundled model is in scope, or that a
prediction estimates a physical rate. If you intend to use a bundled model,
read the [model card](../MODEL_CARD.md) before running `predict`.
