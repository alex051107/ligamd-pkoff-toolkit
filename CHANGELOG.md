# Changelog

## Unreleased

- Added the default P512 sequence v2 contract. It preserves the first and
  endpoint P512 rows in the shuffled arm and deterministically permutes only
  the 510 interior rows, so a final-state sequence encoder cannot use
  endpoint-last placement as an arm-specific shortcut. The historical v1
  all-row contract remains available only by explicit path for payload replay.
- Added a data-free, deterministic order-positive fixture and focused tests for
  boundary preservation, interior-row bijection, row integrity, and v1 replay.
  These are engineering controls only; no GRU or experimental label was run.
- Added a data-free, GitHub-only review packet for the proposed model-performance
  plan. The packet records the current aggregate evidence, the bounded two-track
  proposal, its stop rules, and a Chinese prompt for an independent reviewer.
- Revised that proposal after independent review: added one trajectory-protocol
  compatibility gate, froze the data-free 11-channel P512 sequence contract,
  consolidated temporal engineering checks into one receipt, narrowed a
  negative GRU claim to the fixed protocol, and separated BiCoA from the Ridge
  primary comparison.
- Added a developer-only, label-blind P512 sequence serializer plus a synthetic
  contract test. It reuses existing dense traces and P512 identities, emits
  matched ordered/fixed-shuffle tensors, and does not run a model.
- The serializer accepts the existing three-method `selected_identities.tsv`
  artifact directly by applying one frozen `method=P512` filter and checking
  exact ranks `0..511`; no per-route copied selection file is required.
- No endpoint, sampler, Current30 feature, bundled model, prediction, or public
  data boundary changed.

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
