# Synthetic three-replica smoke campaign

Run the generator from the repository root.

```bash
python examples/make_synthetic_three_replica_campaign.py \
  --output-dir /tmp/ligamd-pkoff-synthetic
```

It writes a small canonical PDB, a topology identity placeholder, three Amber
NetCDF coordinate files, and a manifest. The geometry is intentionally simple.
The ligand moves away at source frame 521 and remains clear of the protein for
the final 100 saved frames, so the endpoint-v2 test is expected to pass.

Use the generated `manifest.json` with the `featurize` command in the root
README. The example is useful for checking installation and file wiring. It is
not a simulation result and it cannot validate a pKoff model.
