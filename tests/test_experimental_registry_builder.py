"""Focused safeguards for the developer-only frozen N31 registry builder.

The public package does not distribute the labelled N31 panel, its fold table,
or per-system out-of-fold predictions.  A developer who is separately
authorised to rebuild that historical registry must therefore supply exactly
the frozen inputs recorded in the bundled provenance receipt.  These tests use
small synthetic files only; they do not reconstruct, inspect, or publish N31
labels.
"""

from __future__ import annotations

import builtins
import json
from pathlib import Path

import pytest

from scripts.koff_ml import build_experimental_model_registry as registry_builder
from scripts.koff_ml.io import sha256_file


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _receipt(
    *,
    source_panel: Path,
    folds: Path,
    reused_oof: Path,
    endpoint_contract: Path,
    required_columns: list[str] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": registry_builder.INPUT_PROVENANCE_SCHEMA,
        "input_artifacts": {
            "source_panel": {
                "artifact_name": source_panel.name,
                "sha256": sha256_file(source_panel),
            },
            "folds": {
                "artifact_name": folds.name,
                "sha256": sha256_file(folds),
            },
            "reused_oof": {
                "artifact_name": reused_oof.name,
                "sha256": sha256_file(reused_oof),
            },
        },
        "endpoint_contract": {
            "contract_id": "synthetic-endpoint-v2",
            "sha256": sha256_file(endpoint_contract),
        },
        "reused_oof_cell_contract": {
            "endpoint_arm": registry_builder.ENDPOINT_ARM,
            "sampler": registry_builder.SAMPLER,
            "combined_feature_block": "Combined30",
            "dummy_feature_block": "No_features",
            "models": ["ridge", "random_forest", "dummy_median"],
            "system_count": registry_builder.EXPECTED_SYSTEMS,
            "required_columns": required_columns
            or [
                "system_id",
                "endpoint_arm",
                "sampler",
                "feature_block",
                "model",
                "experimental_pKoff",
                "exact_ligand_group",
                "held_out_exact_ligand_group",
                "outer_test_fold",
                "predicted_pKoff",
            ],
        },
    }


def test_registry_builder_does_not_import_xgboost_for_its_fixed_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ridge/Random Forest registry rebuilding works without the optional extra."""

    original_import = builtins.__import__

    def reject_xgboost(name: str, *args: object, **kwargs: object) -> object:
        if name == "xgboost" or name.startswith("xgboost."):
            raise AssertionError("the fixed registry rebuild must not import xgboost")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_xgboost)
    families = registry_builder.model_families(include_xgboost=False)

    assert set(families) == {"ridge", "random_forest"}


def test_registry_builder_rejects_drift_from_its_authorised_input_receipt(
    tmp_path: Path,
) -> None:
    """Hashes and fold-provenance fields fail closed before a registry is built."""

    source_panel = _write(tmp_path / "panel.tsv", "synthetic panel\n")
    folds = _write(tmp_path / "folds.tsv", "synthetic folds\n")
    reused_oof = _write(tmp_path / "oof.tsv", "synthetic OOF\n")
    endpoint_contract = _write(tmp_path / "endpoint.json", '{"contract_id": "synthetic-endpoint-v2"}\n')
    receipt_path = tmp_path / "frozen-inputs.json"
    receipt_path.write_text(
        json.dumps(
            _receipt(
                source_panel=source_panel,
                folds=folds,
                reused_oof=reused_oof,
                endpoint_contract=endpoint_contract,
            )
        ),
        encoding="utf-8",
    )

    accepted = registry_builder._load_frozen_input_provenance(
        source_panel=source_panel,
        folds=folds,
        reused_oof=reused_oof,
        endpoint_contract=endpoint_contract,
        contract={"contract_id": "synthetic-endpoint-v2"},
        receipt_path=receipt_path,
    )
    assert accepted["schema_version"] == registry_builder.INPUT_PROVENANCE_SCHEMA

    folds.write_text("changed folds\n", encoding="utf-8")
    with pytest.raises(registry_builder.RegistryError, match="folds hash differs"):
        registry_builder._load_frozen_input_provenance(
            source_panel=source_panel,
            folds=folds,
            reused_oof=reused_oof,
            endpoint_contract=endpoint_contract,
            contract={"contract_id": "synthetic-endpoint-v2"},
            receipt_path=receipt_path,
        )


def test_registry_builder_requires_explicit_held_out_fold_columns(tmp_path: Path) -> None:
    """A digest alone is insufficient when the OOF receipt omits split semantics."""

    source_panel = _write(tmp_path / "panel.tsv", "synthetic panel\n")
    folds = _write(tmp_path / "folds.tsv", "synthetic folds\n")
    reused_oof = _write(tmp_path / "oof.tsv", "synthetic OOF\n")
    endpoint_contract = _write(tmp_path / "endpoint.json", '{"contract_id": "synthetic-endpoint-v2"}\n')
    columns = [
        "system_id",
        "endpoint_arm",
        "sampler",
        "feature_block",
        "model",
        "experimental_pKoff",
        "exact_ligand_group",
        "predicted_pKoff",
    ]
    receipt_path = tmp_path / "missing-fold-provenance.json"
    receipt_path.write_text(
        json.dumps(
            _receipt(
                source_panel=source_panel,
                folds=folds,
                reused_oof=reused_oof,
                endpoint_contract=endpoint_contract,
                required_columns=columns,
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(registry_builder.RegistryError, match="explicit OOF fold provenance"):
        registry_builder._load_frozen_input_provenance(
            source_panel=source_panel,
            folds=folds,
            reused_oof=reused_oof,
            endpoint_contract=endpoint_contract,
            contract={"contract_id": "synthetic-endpoint-v2"},
            receipt_path=receipt_path,
        )
