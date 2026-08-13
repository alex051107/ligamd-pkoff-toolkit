# Optional typed-interaction challenger

This route asks one narrow engineering question: does a compact description of
named protein–ligand interactions deserve a later, matched model comparison?
It does not assume that the answer is yes.

## Input

Run the ordinary three-replica `featurize` workflow with
`--typed-interactions`. The manifest must also provide `ligand.sdf`, pointing
to one sanitized ligand structure whose chemistry matches the topology-derived
ligand. Install the optional dependencies with `pip install ".[typed]"`.

The typed extractor receives the `p512_coordinate_union.npz` that the ordinary
workflow has already created for each endpoint-passing replica. It does not
read the NetCDF trajectory again, choose another endpoint, select another set
of frames, or read an experimental label.

## Computation

For every selected frame, the extractor makes the ligand and each standard
amino-acid residue whole under periodic boundaries, images each residue to the
ligand, and applies the frozen ProLIF 2.2.0 CORE8 contract:

1. Hydrophobic
2. HBAcceptor
3. HBDonor
4. PiStacking
5. Anionic
6. Cationic
7. CationPi
8. PiCation

For each class, the number of active standard-protein residues is divided by
the total number of standard-protein residues. Those frame-level fractions are
summarized over four normalized episode intervals: 0–0.50, 0.50–0.80,
0.80–0.95, and 0.95–1.00. Nearest-selected-frame Voronoi support supplies the
stage weights. The result is 8 × 4 = 32 values per replica. Corresponding
fields are then averaged across exactly three endpoint-passing replicas.

The frozen SMARTS patterns, geometry thresholds, order, stage definitions, and
scientific boundary are in
`ligamd_pkoff/resources/contracts/typed_interactions_core8_v1.json`.

## Output

Each replica directory receives `typed_interactions.json`. The system output
directory receives `typed_interactions.json` as its receipt and
`typed_interactions.tsv` as the three-replica machine-learning table.
`features.json` contains a pointer and the pooled 32-field dictionary, but
`combined30` remains unchanged.

## Evidence boundary

Every typed artifact is marked `ENGINEERING_CHALLENGER_NOT_SELECTED`.

- The 32 fields are not inputs to any bundled model.
- Generating them does not prove that typed interactions improve experimental
  pKoff prediction.
- A larger typed block must not be credited for improvement unless it is
  compared with a matched-capacity untyped control on the same held-out
  exact-ligand groups.
- Normalized episode progress is not physical time, and these fields do not
  estimate a physical koff.

Until that comparison passes its predeclared evidence rule, this route is a
reusable feature calculator, not a selected scientific representation.

## Delivery status

The implementation is currently a local review branch. It has not been pushed,
merged, or opened as a public pull request because a real typed-dependency
smoke run and the matched scientific representation decision are still
pending. Publishing the branch before those two boundaries are resolved would
make the engineering option look more mature than the evidence supports.

The opt-in wiring has passed a synthetic three-replica integration test: it
consumes the already selected P512 unions, produces 32 typed fields, and leaves
the existing Combined30 predictor input unchanged. A real ProLIF calculation
has not yet run on this branch. The previously materialized coordinate unions,
topologies, ligand templates, and ProLIF wheel referenced by the internal N5
receipt are Longleaf-only paths and are not present in the local workspace.
Rather than rebuild that environment or copy production data solely for a
smoke test, the real-path check is deferred to the first representative system
that already has an accessible P512 union and authoritative ligand template.
