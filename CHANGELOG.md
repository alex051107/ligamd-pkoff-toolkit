# Changelog

## Unreleased

- Added a data-free, GitHub-only review packet for the proposed model-performance
  plan. The packet records the current aggregate evidence, the bounded two-track
  proposal, its stop rules, and a Chinese prompt for an independent reviewer.
- Revised that proposal after independent review: added one trajectory-protocol
  compatibility gate, froze the data-free 11-channel P512 sequence contract,
  consolidated temporal engineering checks into one receipt, narrowed a
  negative GRU claim to the fixed protocol, and separated BiCoA from the Ridge
  primary comparison.
- No endpoint, sampler, feature, model artifact, prediction, or public data
  boundary changed; the sequence contract is not yet a runtime serializer.

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
