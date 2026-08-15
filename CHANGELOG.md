# Changelog

## Unreleased

- Clarified for first-time users that endpoint-v2's 15 A pocket-aligned
  displacement, greater-than-10 A whole-protein clearance, and 100-saved-frame
  persistence define a feature-extraction episode boundary. They do not
  replace a campaign-specific 35/40 A long-distance dissociation certificate.

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
