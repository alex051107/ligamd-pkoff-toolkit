from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tools.prospective_dynamic_validation import (
    MODEL_ROLES,
    ProspectiveValidationError,
    SCHEMA,
    _digest_files,
    evaluate,
    freeze,
)


def _write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False, lineterminator="\n")


def _cohort_rows(count: int = 15) -> list[dict[str, object]]:
    return [
        {
            "system_id": f"system-{index:02d}",
            "condition_id": f"condition-{index:02d}",
            "exact_ligand_group": f"group-{index:02d}",
            "audited_ligand_identity": f"identity-{index:02d}",
            "protein_target": "HSP90A",
            "target_family": "HSP90",
            "features_json_path": f"features-{index:02d}.json",
        }
        for index in range(count)
    ]


def _frozen_inputs(tmp_path: Path) -> Path:
    freeze_dir = tmp_path / "freeze"
    freeze_dir.mkdir()
    cohort = _cohort_rows()
    for index, row in enumerate(cohort):
        row["feature_receipt_sha256"] = f"frozen-feature-{index:02d}"
    cohort_path = freeze_dir / "frozen_cohort.tsv"
    _write_tsv(cohort_path, cohort)
    ledger_rows: list[dict[str, object]] = []
    for index, row in enumerate(cohort):
        label = 5.0 + index / 10.0
        for model_id in MODEL_ROLES:
            if model_id == "static20_ridge":
                prediction = label + 0.20
            elif model_id == "combined30_p512_ridge":
                prediction = label
            elif model_id == "static20_random_forest":
                prediction = label + 0.15
            else:
                prediction = label + 0.05
            ledger_rows.append(
                {
                    "system_id": row["system_id"],
                    "model_id": model_id,
                    "prediction_scope": "WITHIN_OBSERVED_SCOPE",
                    "predicted_experimental_pKoff": prediction,
                }
            )
    ledger_path = freeze_dir / "prediction_ledger.tsv"
    _write_tsv(ledger_path, ledger_rows)
    development_path = freeze_dir / "development_identity_snapshot.tsv"
    _write_tsv(development_path, [{"audited_ligand_identity": "development-only"}])
    digest = _digest_files(
        (
            ("frozen_cohort.tsv", cohort_path),
            ("development_identity_snapshot.tsv", development_path),
            ("prediction_ledger.tsv", ledger_path),
        )
    )
    (freeze_dir / "freeze_receipt.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA,
                "operation": "freeze",
                "status": "FROZEN_BEFORE_LABEL_ATTACHMENT",
                "combined_frozen_table_sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    return freeze_dir


def test_freeze_rejects_development_identity_overlap(tmp_path: Path) -> None:
    cohort_rows = _cohort_rows()
    cohort_path = tmp_path / "cohort.tsv"
    _write_tsv(cohort_path, cohort_rows)
    development_path = tmp_path / "development.tsv"
    _write_tsv(
        development_path,
        [{"audited_ligand_identity": cohort_rows[0]["audited_ligand_identity"]}],
    )

    with pytest.raises(ProspectiveValidationError, match="overlap development"):
        freeze(
            cohort_path=cohort_path,
            development_identities_path=development_path,
            output_dir=tmp_path / "freeze-output",
        )


def test_freeze_rejects_case_only_identity_and_family_aliases(tmp_path: Path) -> None:
    development_path = tmp_path / "development.tsv"
    _write_tsv(development_path, [{"audited_ligand_identity": "development-only"}])

    identity_aliases = _cohort_rows()
    identity_aliases[1]["audited_ligand_identity"] = str(
        identity_aliases[0]["audited_ligand_identity"]
    ).upper()
    identity_path = tmp_path / "identity-aliases.tsv"
    _write_tsv(identity_path, identity_aliases)
    with pytest.raises(ProspectiveValidationError, match="case-only aliases"):
        freeze(
            cohort_path=identity_path,
            development_identities_path=development_path,
            output_dir=tmp_path / "identity-output",
        )

    family_aliases = _cohort_rows()
    family_aliases[1]["target_family"] = "hsp90"
    family_path = tmp_path / "family-aliases.tsv"
    _write_tsv(family_path, family_aliases)
    with pytest.raises(ProspectiveValidationError, match="case-only aliases"):
        freeze(
            cohort_path=family_path,
            development_identities_path=development_path,
            output_dir=tmp_path / "family-output",
        )


def test_evaluate_synthetic_fifteen_group_pass_and_label_mismatch_refusal(
    tmp_path: Path,
) -> None:
    freeze_dir = _frozen_inputs(tmp_path)
    labels = [
        {"system_id": f"system-{index:02d}", "experimental_pKoff": 5.0 + index / 10.0}
        for index in range(15)
    ]
    labels_path = tmp_path / "labels.tsv"
    _write_tsv(labels_path, labels)

    pass_output = tmp_path / "pass"
    decision = evaluate(
        freeze_dir=freeze_dir,
        labels_path=labels_path,
        output_dir=pass_output,
    )

    assert decision["status"] == "PASS_PROSPECTIVE_DYNAMIC_INCREMENT_GATE"
    assert all(decision["primary_gate_checks"].values())
    assert decision["primary_summary"]["total_group_count"] == 15
    assert len(pd.read_csv(pass_output / "paired_group_errors.tsv", sep="\t")) == 30
    assert len(pd.read_csv(pass_output / "target_family_sensitivity.tsv", sep="\t")) == 1

    mismatch_path = tmp_path / "mismatched-labels.tsv"
    _write_tsv(mismatch_path, labels[:-1])
    mismatch_output = tmp_path / "mismatch"
    mismatch = evaluate(
        freeze_dir=freeze_dir,
        labels_path=mismatch_path,
        output_dir=mismatch_output,
    )

    assert mismatch["status"] == "NOT_EVALUABLE"
    assert mismatch["evaluated_system_count"] == 0
    assert mismatch["replacement_groups_used"] is False
    assert "do not exactly match" in " ".join(mismatch["reasons"])
    assert pd.read_csv(mismatch_output / "paired_group_errors.tsv", sep="\t").empty
    saved = json.loads((mismatch_output / "prospective_decision.json").read_text())
    assert saved["status"] == "NOT_EVALUABLE"

    ledger_path = freeze_dir / "prediction_ledger.tsv"
    ledger_path.write_text(
        ledger_path.read_text(encoding="utf-8").replace("system-00", "tampered-system", 1),
        encoding="utf-8",
    )
    with pytest.raises(ProspectiveValidationError, match="differs from the freeze receipt"):
        evaluate(
            freeze_dir=freeze_dir,
            labels_path=labels_path,
            output_dir=tmp_path / "tampered",
        )
