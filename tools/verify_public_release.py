#!/usr/bin/env python3
"""Check the public toolkit boundary without rehashing scientific data.

This verifier protects against three release mistakes: a missing executable
component, an accidental raw-data file, and a local path or obvious credential
string entering the public tree. It is intentionally small. It does not rerun
the N31 model matrix and it does not recompute hashes for ordinary source edits.
"""

from __future__ import annotations

import json
import py_compile
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "MODEL_CARD.md",
    "DATA_NOTICE.md",
    "pyproject.toml",
    "AGENTS.md",
    "ligamd_pkoff/__init__.py",
    "ligamd_pkoff/cli.py",
    "ligamd_pkoff/resources/__init__.py",
    "ligamd_pkoff/resources/contracts/endpoint_v2.json",
    "ligamd_pkoff/resources/contracts/p512_sampler_v1.json",
    "ligamd_pkoff/resources/contracts/p512_sequence_v2.json",
    "ligamd_pkoff/resources/contracts/frozen_n31_registry_input_provenance_v1.json",
    "scripts/koff_ml/toolkit.py",
    "scripts/koff_ml/endpoint_two_metric.py",
    "scripts/koff_ml/p512_sampler.py",
    "scripts/koff_ml/p512_sequence.py",
    "scripts/koff_ml/temporal_order_fixture.py",
    "scripts/koff_ml/serialization_compat.py",
    "ligamd_pkoff/resources/models/experimental_n31_registry_v1/model_registry.json",
    "ligamd_pkoff/resources/models/experimental_n31_registry_v1/model_scoreboard.tsv",
    "ligamd_pkoff/resources/models/experimental_n31_registry_v1/paired_dynamic_increment.tsv",
    "examples/make_synthetic_three_replica_campaign.py",
    "tests/test_endpoint_two_metric.py",
    "tests/test_koff_ml_toolkit_smoke.py",
    "docs/code-map.md",
    "docs/method-evidence.md",
    "docs/tutorial-synthetic-campaign.md",
    "tests/test_p512_sampler.py",
    "tests/test_p512_sequence.py",
)
FORBIDDEN_PUBLIC_FILES = (
    "scripts/koff_ml/label_blind_sampler_sidecar.py",
    "scripts/koff_ml/successful_replica_sampler_benchmark.py",
    "scripts/koff_ml/global_sampler.py",
    "scripts/koff_ml/metrics.py",
    "scripts/koff_ml/sampling.py",
    "scripts/koff_ml/prediction_first_sampling.py",
)
BANNED_SUFFIXES = {".nc", ".netcdf", ".parm7", ".prmtop", ".rst7", ".rst", ".dcd", ".xtc", ".trr"}
TEXT_SUFFIXES = {".md", ".py", ".json", ".tsv", ".toml", ".yml", ".yaml", ".cff", ".txt"}
SUSPICIOUS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bghp_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"/Users/[^/]+/"),
    re.compile(r"/users/[A-Za-z0-9_/-]+/"),
)


def _fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(2)


def main() -> int:
    missing = [path for path in REQUIRED if not (ROOT / path).is_file()]
    if missing:
        _fail("missing required public files: " + ", ".join(missing))
    retained_history = [path for path in FORBIDDEN_PUBLIC_FILES if (ROOT / path).exists()]
    if retained_history:
        _fail("historical campaign modules remain in public runtime: " + ", ".join(retained_history))

    raw_files = [
        path.relative_to(ROOT)
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.parts and path.suffix.lower() in BANNED_SUFFIXES
    ]
    if raw_files:
        _fail("raw trajectory or topology-like files are not allowed: " + ", ".join(map(str, raw_files)))

    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        # This verifier contains the very patterns it looks for. It audits
        # release content, not its own regular-expression source code.
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in SUSPICIOUS:
            if pattern.search(text):
                _fail(f"suspicious public string in {path.relative_to(ROOT)}: {pattern.pattern}")

    resource_root = ROOT / "ligamd_pkoff/resources"
    endpoint = json.loads((resource_root / "contracts/endpoint_v2.json").read_text())
    p512 = json.loads((resource_root / "contracts/p512_sampler_v1.json").read_text())
    sequence_v2 = json.loads((resource_root / "contracts/p512_sequence_v2.json").read_text())
    registry_root = resource_root / "models/experimental_n31_registry_v1"
    registry = json.loads((registry_root / "model_registry.json").read_text())
    if endpoint.get("schema_version") != "ligamd_endpoint_contract_v2.0":
        _fail("endpoint contract schema mismatch")
    if p512.get("schema_version") != "ligamd_p512_sampler_contract_v1.0":
        _fail("P512 contract schema mismatch")
    if sequence_v2.get("schema_version") != "ligamd_p512_sequence_contract_v2.0":
        _fail("P512 sequence v2 contract schema mismatch")
    if sequence_v2.get("model_run_authorized") is not False:
        _fail("P512 sequence v2 must not authorize a model run")
    shuffled = sequence_v2.get("shuffled_control", {})
    if (
        shuffled.get("mode") != "ENDPOINT_PRESERVING_INTERIOR_PERMUTATION"
        or shuffled.get("fixed_rank0") != [0, 511]
        or shuffled.get("interior_rank0_range") != [1, 510]
    ):
        _fail("P512 sequence v2 shuffled-control boundary contract mismatch")
    if registry.get("scientific_status") != "EXPERIMENTAL":
        _fail("bundled registry must remain EXPERIMENTAL")
    for profile_id, profile in registry.get("profiles", {}).items():
        artifact = registry_root / str(profile["artifact"])
        if not artifact.is_file():
            _fail(f"missing model artifact for {profile_id}")
    scorecard = (registry_root / "model_scoreboard.tsv").read_text()
    if "experimental_pKoff" in scorecard:
        _fail("aggregate scorecard unexpectedly contains per-system labels")
    increment_header = (
        registry_root / "paired_dynamic_increment.tsv"
    ).read_text().splitlines()[0].split("\t")
    for forbidden in ("system_id", "exact_ligand_group", "experimental_pKoff", "InChIKey"):
        if forbidden in increment_header:
            _fail(f"aggregate increment table unexpectedly contains raw field {forbidden}")

    for path in (ROOT / "scripts").rglob("*.py"):
        py_compile.compile(str(path), doraise=True)
    print("PASS: public release boundary, contracts, registry, and Python syntax are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
