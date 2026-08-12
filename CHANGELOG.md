# Changelog

## Unreleased

- Added an opt-in `--typed-interactions` engineering route that reuses each
  existing P512 coordinate union to calculate a separate 32-field CORE8
  interaction challenger.
- Kept the challenger out of Current30, bundled model inputs, and the model
  registry; every output is marked `ENGINEERING_CHALLENGER_NOT_SELECTED`.
- Added a bundled typed-interaction contract, optional dependency group,
  focused PBC/shape check, and first-reader decision note.
- Made raw-input reuse fail closed on duplicate trajectory paths and canonical
  PDB/topology atom-order drift, while normalizing explicitly blank PDB chains.
- Bound Combined30 inference to the bundled endpoint-v2, P512, and exact-three
  pooling semantics recorded in the feature receipt; Static20 remains usable
  independently of the trajectory contract.

## 0.1.0, 2026-08-12

First public experimental release.

- Added a clean, whitelist-only public repository boundary.
- Added endpoint-v2 and a P512-only sampler contract.
- Added a command-line route from three raw LiGaMD replicas to Current30.
- Added four frozen `EXPERIMENTAL` N31 model profiles and aggregate evidence.
- Added a synthetic campaign generator, focused regression tests, and public
  release checks.
- Excluded raw trajectories, per-system labels, OOF prediction rows, source
  papers, and internal campaign-reconciliation materials.
