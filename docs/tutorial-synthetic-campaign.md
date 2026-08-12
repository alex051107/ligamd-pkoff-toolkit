# Tutorial: run a complete synthetic three-replica campaign

This tutorial is for a reader who wants to see one successful end-to-end
calculation before preparing a real LiGaMD campaign. It creates a tiny,
artificial trajectory in which a ligand becomes clearly separated from a small
protein. The output proves that the files, contracts, and command-line route
are connected. It does not validate a molecular mechanism or a model score.

This is deliberately a **clone-only tutorial**. Step 1 runs a generator from
the repository's `examples/` directory, which is not installed with a wheel.
If you installed only a wheel, first run the [installed-wheel smoke check in
the README](../README.md#verify-an-installed-wheel-without-data), then use
[the real-input how-to](prepare-real-inputs.md). The wheel does not promise a
synthetic example campaign.

## Before you start

Clone the repository, install the package, and activate the Conda environment
described in the [README](../README.md). Choose a temporary directory that does
not already contain an earlier tutorial output.

## 1. Create three synthetic replicas

```bash
python examples/make_synthetic_three_replica_campaign.py \
  --output-dir /tmp/ligamd-pkoff-synthetic
```

You will see a `manifest.json`, one canonical PDB, one placeholder topology,
and three tiny NetCDF coordinate files. The manifest has exactly three entries
under `replicas`. The public workflow requires this exact number because it
averages feature values across three independently processed replicas.

## 2. Calculate the episode and Current30 features

```bash
ligamd-pkoff featurize \
  --manifest /tmp/ligamd-pkoff-synthetic/manifest.json \
  --output-dir /tmp/ligamd-pkoff-features
```

The command first freezes a reference from the canonical bound structure. It
then processes every replica separately. A replica contributes only when its
endpoint-v2 rule passes. The final `features.json` should report
`PASS_FEATURES_READY_FOR_EXPERIMENTAL_PREDICTION`.

Open these files after a successful run.

| File | What it lets you check |
| --- | --- |
| `trajectory_status.tsv` | whether each replica had a persistent geometry-defined complete exit |
| `endpoint_events.tsv` | the onset frame and the endpoint receipt for each replica |
| `replicas/<id>/p512_selected_frames.tsv` | the 512 unique real source frames selected from that replica |
| `system_features.tsv` | one Static20 plus Dynamic10 row for the system |
| `features.json` | the machine-readable hand-off to prediction |

If a replica does not pass, the command still writes a status receipt. It does
not make a partial three-replica average or a prediction. Read the reason in
`features.json` before changing any input.

## 3. Apply one bundled experimental baseline

```bash
ligamd-pkoff predict \
  --features /tmp/ligamd-pkoff-features/features.json \
  --model-id combined30_p512_ridge \
  --output /tmp/ligamd-pkoff-prediction.json
```

The synthetic protein target was never part of N31. A warning about
applicability is therefore expected. The command still shows that the bundled
registry and model file were found after installation. The returned number is
a predicted assay-derived experimental pKoff, not a rate calculated from the
synthetic trajectory duration.

## 4. What to change for real data

Replace only the inputs named in the manifest. Give the command a canonical
bound PDB, the matching topology, a ligand identity, and three production
replicas with the same saved-frame cadence. Keep the contract defaults unless
you have created and documented a new approved configuration. The
[real-input how-to](prepare-real-inputs.md) gives the working sequence, the
[input-manifest reference](input-manifest.md) describes each field, and the
[endpoint and sampler explanation](endpoint-and-sampler.md) explains why the
workflow stops at the first persistent exit onset.
