#!/usr/bin/env python3
"""Freeze and evaluate one prospective Dynamic10 validation cohort.

``freeze`` runs the four already-bundled experimental models before labels are
attached. ``evaluate`` later joins a complete label table to that immutable
prediction ledger and applies the predeclared, group-equal decision rule.  The
tool does not fit or tune a model, replace missing groups, or estimate physical
``koff``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

if __package__ in {None, ""}:  # pragma: no cover - direct invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.koff_ml import toolkit
from scripts.koff_ml.io import write_json
from scripts.koff_ml.run_dual_endpoint_combined30_matrix_v1 import (
    BOOTSTRAP_REPEATS,
    paired_group_summary,
    per_group_mae,
)


SCHEMA = "ligamd_prospective_dynamic_validation_v1.0"
SEED = 20260812
MINIMUM_GROUPS = 15
PRIMARY_DELTA_THRESHOLD = -0.10
COHORT_FIELDS = (
    "system_id",
    "condition_id",
    "exact_ligand_group",
    "audited_ligand_identity",
    "protein_target",
    "target_family",
    "features_json_path",
)
LABEL_FIELDS = ("system_id", "experimental_pKoff")
MODEL_ROLES = {
    "static20_ridge": "PRIMARY_STATIC_REFERENCE",
    "combined30_p512_ridge": "PRIMARY_DYNAMIC_CANDIDATE",
    "static20_random_forest": "SENSITIVITY_STATIC_REFERENCE",
    "combined30_p512_random_forest": "SENSITIVITY_DYNAMIC_CANDIDATE",
}
MODEL_PAIRS = {
    "ridge_primary": ("static20_ridge", "combined30_p512_ridge"),
    "random_forest_sensitivity": (
        "static20_random_forest",
        "combined30_p512_random_forest",
    ),
}
ACCEPTED_SCOPES = {"WITHIN_OBSERVED_SCOPE", "CAUTION_OUTSIDE_OBSERVED_RANGE"}
PAIRED_COLUMNS = (
    "comparison_id",
    "exact_ligand_group",
    "static_group_mae",
    "combined_group_mae",
    "combined_minus_static_mae",
    "combined_improved",
)
FAMILY_COLUMNS = (
    "target_family",
    "exact_ligand_group_count",
    "combined_minus_static_mae",
    "improved_group_count",
    "harmed_group_count",
    "all_groups_harmed",
    "major_target_family",
)


class ProspectiveValidationError(RuntimeError):
    """Raised when a freeze artifact is unsafe or internally inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProspectiveValidationError(message)


def _read_tsv(path: Path) -> pd.DataFrame:
    _require(path.is_file(), f"TSV does not exist: {path}")
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    _require(len(frame) > 0, f"TSV is empty: {path}")
    return frame


def _reject_label_columns(columns: Iterable[object], source: str) -> None:
    forbidden = [
        str(column)
        for column in columns
        if "label" in str(column).casefold() or "koff" in str(column).casefold()
    ]
    _require(not forbidden, f"{source} contains forbidden label columns: {forbidden}")


def _clean_required(frame: pd.DataFrame, required: Iterable[str], source: str) -> pd.DataFrame:
    required = tuple(required)
    missing = sorted(set(required) - set(frame.columns))
    _require(not missing, f"{source} is missing required columns: {missing}")
    result = frame.loc[:, required].copy()
    for column in required:
        result[column] = result[column].astype(str).str.strip()
        _require(result[column].ne("").all(), f"{source} contains empty {column}")
    return result


def _validate_bijection(frame: pd.DataFrame, left: str, right: str) -> None:
    pairs = frame[[left, right]].drop_duplicates()
    left_to_right = pairs.groupby(left, sort=False)[right].nunique()
    right_to_left = pairs.groupby(right, sort=False)[left].nunique()
    _require(
        bool((left_to_right == 1).all() and (right_to_left == 1).all()),
        f"{left} and {right} must form a bijection",
    )


def _validate_many_to_one(frame: pd.DataFrame, left: str, right: str) -> None:
    counts = frame[[left, right]].drop_duplicates().groupby(left, sort=False)[right].nunique()
    _require(bool((counts == 1).all()), f"each {left} must map to exactly one {right}")


def _prepare_cohort(path: Path) -> pd.DataFrame:
    raw = _read_tsv(path)
    _reject_label_columns(raw.columns, "cohort")
    cohort = _clean_required(raw, COHORT_FIELDS, "cohort")
    _require(cohort["system_id"].is_unique, "cohort system_id values must be unique")
    _validate_bijection(cohort, "audited_ligand_identity", "exact_ligand_group")
    _validate_many_to_one(cohort, "exact_ligand_group", "target_family")
    _require(
        cohort["exact_ligand_group"].nunique() >= MINIMUM_GROUPS,
        f"cohort needs at least {MINIMUM_GROUPS} exact-ligand groups",
    )
    return cohort.sort_values(["exact_ligand_group", "system_id"], kind="stable").reset_index(drop=True)


def _prepare_development_identities(path: Path) -> pd.DataFrame:
    raw = _read_tsv(path)
    _reject_label_columns(raw.columns, "development identity table")
    identities = _clean_required(raw, ("audited_ligand_identity",), "development identity table")
    return identities.drop_duplicates().sort_values("audited_ligand_identity", kind="stable").reset_index(drop=True)


def _resolve_feature_path(cohort_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = cohort_path.parent / path
    path = path.resolve()
    _require(path.is_file(), f"features_json_path does not exist: {path}")
    return path


def _digest_files(paths: Iterable[tuple[str, Path]]) -> str:
    digest = hashlib.sha256()
    for logical_name, path in sorted(paths, key=lambda item: item[0]):
        _require(path.is_file(), f"digest input does not exist: {path}")
        name = logical_name.encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        with path.open("rb") as handle:
            while chunk := handle.read(1 << 20):
                digest.update(chunk)
    return digest.hexdigest()


def _model_bundle_digest(registry_path: Path) -> str:
    with registry_path.open("r", encoding="utf-8") as handle:
        registry = json.load(handle)
    profiles = registry.get("profiles")
    _require(isinstance(profiles, Mapping), "model registry lacks profiles")
    inputs: list[tuple[str, Path]] = [("model_registry.json", registry_path)]
    for model_id in MODEL_ROLES:
        profile = profiles.get(model_id)
        _require(isinstance(profile, Mapping), f"model registry lacks {model_id}")
        artifact = (registry_path.parent / str(profile.get("artifact", ""))).resolve()
        inputs.append((f"models/{model_id}", artifact))
    return _digest_files(inputs)


def _write_tsv(path: Path, frame: pd.DataFrame, columns: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.reindex(columns=list(columns)).to_csv(path, sep="\t", index=False, lineterminator="\n")


def freeze(
    *,
    cohort_path: Path,
    development_identities_path: Path,
    output_dir: Path,
    registry_path: Path = toolkit.DEFAULT_MODEL_REGISTRY,
) -> Mapping[str, Any]:
    """Freeze cohort membership, identities and four pre-label predictions."""

    _require(not output_dir.exists(), f"refusing to overwrite freeze output: {output_dir}")
    cohort_path = cohort_path.resolve()
    registry_path = registry_path.resolve()
    cohort = _prepare_cohort(cohort_path)
    development = _prepare_development_identities(development_identities_path.resolve())
    prospective_identities = set(cohort["audited_ligand_identity"].str.casefold())
    development_set = set(development["audited_ligand_identity"].str.casefold())
    overlap = sorted(prospective_identities & development_set)
    _require(not overlap, f"prospective identities overlap development identities: {overlap}")

    prediction_rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="prospective-freeze-") as temporary:
        temporary_dir = Path(temporary)
        for row in cohort.to_dict(orient="records"):
            feature_path = _resolve_feature_path(cohort_path, str(row["features_json_path"]))
            feature_payload = toolkit._read_json(feature_path)
            _reject_label_columns(feature_payload.keys(), f"feature receipt {feature_path}")
            for model_id, role in MODEL_ROLES.items():
                prediction = toolkit.predict(
                    features_path=feature_path,
                    registry_path=registry_path,
                    model_id=model_id,
                    output_path=temporary_dir / f"{row['system_id']}--{model_id}.json",
                )
                _require(
                    str(prediction.get("system_id", "")) == row["system_id"]
                    and str(prediction.get("condition_id", "")) == row["condition_id"]
                    and str(prediction.get("protein_target", "")) == row["protein_target"],
                    f"{row['system_id']}/{model_id} feature identity disagrees with cohort",
                )
                scope = str(prediction.get("prediction_scope", ""))
                _require(
                    scope != "OUT_OF_SCOPE_NO_PREDICTION",
                    f"{row['system_id']}/{model_id} is OUT_OF_SCOPE_NO_PREDICTION",
                )
                _require(scope in ACCEPTED_SCOPES, f"unexpected prediction scope: {scope}")
                value = prediction.get("predicted_experimental_pKoff")
                _require(
                    isinstance(value, (int, float)) and math.isfinite(float(value)),
                    f"{row['system_id']}/{model_id} emitted no finite prediction",
                )
                prediction_rows.append(
                    {
                        "system_id": row["system_id"],
                        "condition_id": row["condition_id"],
                        "exact_ligand_group": row["exact_ligand_group"],
                        "protein_target": row["protein_target"],
                        "target_family": row["target_family"],
                        "model_id": model_id,
                        "model_role": role,
                        "prediction_scope": scope,
                        "scope_reasons_json": json.dumps(
                            prediction.get("scope_reasons", []),
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "predicted_experimental_pKoff": float(value),
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=False)
    cohort_output = output_dir / "frozen_cohort.tsv"
    development_output = output_dir / "development_identity_snapshot.tsv"
    ledger_output = output_dir / "prediction_ledger.tsv"
    _write_tsv(cohort_output, cohort, COHORT_FIELDS)
    _write_tsv(development_output, development, ("audited_ligand_identity",))
    ledger = pd.DataFrame(prediction_rows).sort_values(["system_id", "model_id"], kind="stable")
    _write_tsv(ledger_output, ledger, ledger.columns)
    receipt = {
        "schema_version": SCHEMA,
        "operation": "freeze",
        "status": "FROZEN_BEFORE_LABEL_ATTACHMENT",
        "system_count": int(cohort["system_id"].nunique()),
        "exact_ligand_group_count": int(cohort["exact_ligand_group"].nunique()),
        "prediction_count": int(len(ledger)),
        "primary_models": ["static20_ridge", "combined30_p512_ridge"],
        "sensitivity_models": [
            "static20_random_forest",
            "combined30_p512_random_forest",
        ],
        "caution_prediction_count": int(
            (ledger["prediction_scope"] == "CAUTION_OUTSIDE_OBSERVED_RANGE").sum()
        ),
        "out_of_scope_prediction_count": 0,
        "combined_frozen_table_sha256": _digest_files(
            (
                ("frozen_cohort.tsv", cohort_output),
                ("development_identity_snapshot.tsv", development_output),
                ("prediction_ledger.tsv", ledger_output),
            )
        ),
        "model_bundle_sha256": _model_bundle_digest(registry_path),
        "labels_read": False,
        "physical_koff_estimated": False,
    }
    write_json(output_dir / "freeze_receipt.json", receipt)
    return receipt


def _not_evaluable(
    *, output_dir: Path, cohort: pd.DataFrame, reasons: list[str], label_count: int
) -> Mapping[str, Any]:
    _write_tsv(output_dir / "paired_group_errors.tsv", pd.DataFrame(columns=PAIRED_COLUMNS), PAIRED_COLUMNS)
    _write_tsv(
        output_dir / "target_family_sensitivity.tsv",
        pd.DataFrame(columns=FAMILY_COLUMNS),
        FAMILY_COLUMNS,
    )
    decision = {
        "schema_version": SCHEMA,
        "operation": "evaluate",
        "status": "NOT_EVALUABLE",
        "reasons": reasons,
        "frozen_system_count": int(cohort["system_id"].nunique()),
        "frozen_exact_ligand_group_count": int(cohort["exact_ligand_group"].nunique()),
        "provided_label_row_count": int(label_count),
        "evaluated_system_count": 0,
        "replacement_groups_used": False,
        "seed": SEED,
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "physical_koff_estimated": False,
    }
    write_json(output_dir / "prospective_decision.json", decision)
    return decision


def _load_prediction_ledger(path: Path, frozen_systems: set[str]) -> pd.DataFrame:
    ledger = _read_tsv(path)
    required = {
        "system_id",
        "model_id",
        "prediction_scope",
        "predicted_experimental_pKoff",
    }
    missing = sorted(required - set(ledger.columns))
    _require(not missing, f"prediction ledger is missing columns: {missing}")
    ledger["system_id"] = ledger["system_id"].str.strip()
    ledger["model_id"] = ledger["model_id"].str.strip()
    expected = {(system, model) for system in frozen_systems for model in MODEL_ROLES}
    observed = set(zip(ledger["system_id"], ledger["model_id"], strict=True))
    _require(len(ledger) == len(observed), "prediction ledger has duplicate system/model rows")
    _require(observed == expected, "prediction ledger does not exactly cover frozen systems and models")
    _require(
        ledger["prediction_scope"].isin(ACCEPTED_SCOPES).all(),
        "prediction ledger contains an OUT_OF_SCOPE or unknown scope",
    )
    ledger["predicted_experimental_pKoff"] = pd.to_numeric(
        ledger["predicted_experimental_pKoff"], errors="coerce"
    )
    _require(
        np.isfinite(ledger["predicted_experimental_pKoff"].to_numpy(dtype=float)).all(),
        "prediction ledger contains a non-finite prediction",
    )
    return ledger


def _label_status(labels_path: Path, frozen_systems: set[str]) -> tuple[pd.DataFrame | None, list[str], int]:
    if not labels_path.is_file():
        return None, [f"labels TSV does not exist: {labels_path}"], 0
    try:
        labels = pd.read_csv(labels_path, sep="\t", dtype=str, keep_default_na=False)
    except Exception as exc:  # malformed external handoff becomes an auditable refusal
        return None, [f"labels TSV could not be read: {exc}"], 0
    reasons: list[str] = []
    if not set(LABEL_FIELDS) <= set(labels.columns):
        reasons.append(f"labels require columns {list(LABEL_FIELDS)}")
        return None, reasons, len(labels)
    labels = labels.loc[:, LABEL_FIELDS].copy()
    labels["system_id"] = labels["system_id"].astype(str).str.strip()
    if not labels["system_id"].is_unique:
        reasons.append("labels contain duplicate system_id rows")
    observed = set(labels["system_id"])
    if observed != frozen_systems:
        reasons.append(
            "label systems do not exactly match frozen systems; "
            f"missing={sorted(frozen_systems - observed)}, extra={sorted(observed - frozen_systems)}"
        )
    labels["experimental_pKoff"] = pd.to_numeric(labels["experimental_pKoff"], errors="coerce")
    if not np.isfinite(labels["experimental_pKoff"].to_numpy(dtype=float)).all():
        reasons.append("labels contain a non-finite experimental_pKoff")
    return (None if reasons else labels), reasons, len(labels)


def _pair_summary(
    data: pd.DataFrame, static_id: str, combined_id: str
) -> tuple[dict[str, Any], dict[str, float], dict[str, float]]:
    groups = data["exact_ligand_group"].astype(str).to_numpy()
    labels = data["experimental_pKoff"].to_numpy(dtype=float)
    static = data[static_id].to_numpy(dtype=float)
    combined = data[combined_id].to_numpy(dtype=float)
    static_group = per_group_mae(labels, static, groups)
    combined_group = per_group_mae(labels, combined, groups)
    return paired_group_summary(combined_group, static_group, SEED), static_group, combined_group


def evaluate(
    *, freeze_dir: Path,
    labels_path: Path,
    output_dir: Path,
) -> Mapping[str, Any]:
    """Evaluate all and only the frozen systems without fitting a model."""

    _require(not output_dir.exists(), f"refusing to overwrite evaluation output: {output_dir}")
    freeze_dir = freeze_dir.resolve()
    frozen_cohort_path = freeze_dir / "frozen_cohort.tsv"
    development_path = freeze_dir / "development_identity_snapshot.tsv"
    prediction_ledger_path = freeze_dir / "prediction_ledger.tsv"
    receipt = toolkit._read_json(freeze_dir / "freeze_receipt.json")
    _require(
        receipt.get("schema_version") == SCHEMA
        and receipt.get("operation") == "freeze"
        and receipt.get("status") == "FROZEN_BEFORE_LABEL_ATTACHMENT",
        "freeze receipt is missing or has the wrong status",
    )
    observed_digest = _digest_files(
        (
            ("frozen_cohort.tsv", frozen_cohort_path),
            ("development_identity_snapshot.tsv", development_path),
            ("prediction_ledger.tsv", prediction_ledger_path),
        )
    )
    _require(
        observed_digest == receipt.get("combined_frozen_table_sha256"),
        "frozen cohort or prediction ledger differs from the freeze receipt",
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_cohort = _read_tsv(frozen_cohort_path)
    cohort = _clean_required(raw_cohort, COHORT_FIELDS, "frozen cohort")
    _require(cohort["system_id"].is_unique, "frozen cohort system_id values must be unique")
    frozen_systems = set(cohort["system_id"])
    group_count = cohort["exact_ligand_group"].nunique()
    if group_count < MINIMUM_GROUPS:
        return _not_evaluable(
            output_dir=output_dir,
            cohort=cohort,
            reasons=[f"frozen cohort has fewer than {MINIMUM_GROUPS} exact-ligand groups"],
            label_count=0,
        )
    labels, label_reasons, label_count = _label_status(labels_path.resolve(), frozen_systems)
    if labels is None:
        return _not_evaluable(
            output_dir=output_dir,
            cohort=cohort,
            reasons=label_reasons,
            label_count=label_count,
        )

    ledger = _load_prediction_ledger(prediction_ledger_path, frozen_systems)
    prediction_wide = ledger.pivot(index="system_id", columns="model_id", values="predicted_experimental_pKoff")
    data = cohort.merge(labels, on="system_id", how="inner", validate="one_to_one").merge(
        prediction_wide, left_on="system_id", right_index=True, how="inner", validate="one_to_one"
    )
    _require(len(data) == len(cohort), "evaluation did not retain every frozen system")

    summaries: dict[str, dict[str, Any]] = {}
    paired_rows: list[dict[str, Any]] = []
    for comparison_id, (static_id, combined_id) in MODEL_PAIRS.items():
        summary, static_group, combined_group = _pair_summary(data, static_id, combined_id)
        summaries[comparison_id] = summary
        for group in sorted(static_group):
            delta = combined_group[group] - static_group[group]
            paired_rows.append(
                {
                    "comparison_id": comparison_id,
                    "exact_ligand_group": group,
                    "static_group_mae": static_group[group],
                    "combined_group_mae": combined_group[group],
                    "combined_minus_static_mae": delta,
                    "combined_improved": bool(delta < 0.0),
                }
            )
    _write_tsv(output_dir / "paired_group_errors.tsv", pd.DataFrame(paired_rows), PAIRED_COLUMNS)

    family_rows: list[dict[str, Any]] = []
    static_id, combined_id = MODEL_PAIRS["ridge_primary"]
    for family, family_data in data.groupby("target_family", sort=True):
        family_groups = family_data["exact_ligand_group"].astype(str).to_numpy()
        family_labels = family_data["experimental_pKoff"].to_numpy(dtype=float)
        static_group = per_group_mae(
            family_labels,
            family_data[static_id].to_numpy(dtype=float),
            family_groups,
        )
        combined_group = per_group_mae(
            family_labels,
            family_data[combined_id].to_numpy(dtype=float),
            family_groups,
        )
        deltas = np.asarray(
            [combined_group[group] - static_group[group] for group in sorted(static_group)], dtype=float
        )
        is_major = len(deltas) >= 3
        family_rows.append(
            {
                "target_family": family,
                "exact_ligand_group_count": len(deltas),
                "combined_minus_static_mae": float(np.mean(deltas)),
                "improved_group_count": int(np.sum(deltas < 0.0)),
                "harmed_group_count": int(np.sum(deltas > 0.0)),
                "all_groups_harmed": bool(np.all(deltas > 0.0)),
                "major_target_family": is_major,
            }
        )
    family_frame = pd.DataFrame(family_rows)
    _write_tsv(output_dir / "target_family_sensitivity.tsv", family_frame, FAMILY_COLUMNS)

    primary = summaries["ridge_primary"]
    checks = {
        "mean_delta_at_most_minus_0_10": bool(
            primary["group_equal_mae_delta"] <= PRIMARY_DELTA_THRESHOLD
        ),
        "more_than_half_groups_improved": bool(
            primary["improved_group_count"] > primary["total_group_count"] / 2
        ),
        "bootstrap_95_ci_upper_below_zero": bool(primary["bootstrap_95_ci_upper"] < 0.0),
        "no_leave_one_group_sign_reversal": not bool(primary["leave_one_group_sign_reversal"]),
        "no_major_target_family_all_harmed": not bool(
            (
                family_frame["major_target_family"].astype(bool)
                & family_frame["all_groups_harmed"].astype(bool)
            ).any()
        ),
    }
    passed = all(checks.values())
    decision = {
        "schema_version": SCHEMA,
        "operation": "evaluate",
        "status": (
            "PASS_PROSPECTIVE_DYNAMIC_INCREMENT_GATE"
            if passed
            else "FAIL_PROSPECTIVE_DYNAMIC_INCREMENT_GATE"
        ),
        "primary_comparison": "combined30_p512_ridge_minus_static20_ridge",
        "primary_summary": primary,
        "primary_gate_checks": checks,
        "random_forest_sensitivity_summary": summaries["random_forest_sensitivity"],
        "frozen_system_count": int(len(cohort)),
        "frozen_exact_ligand_group_count": int(group_count),
        "evaluated_system_count": int(len(data)),
        "replacement_groups_used": False,
        "seed": SEED,
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "decision_scope": "candidate recommendation for experimental pKoff prediction only",
        "physical_koff_estimated": False,
    }
    write_json(output_dir / "prospective_decision.json", decision)
    return decision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze", help="freeze a label-free cohort and predictions")
    freeze_parser.add_argument("--cohort", type=Path, required=True)
    freeze_parser.add_argument("--development-identities", type=Path, required=True)
    freeze_parser.add_argument("--output-dir", type=Path, required=True)
    freeze_parser.add_argument("--model-registry", type=Path, default=toolkit.DEFAULT_MODEL_REGISTRY)
    evaluate_parser = commands.add_parser("evaluate", help="evaluate labels against frozen predictions")
    evaluate_parser.add_argument("--freeze-dir", type=Path, required=True)
    evaluate_parser.add_argument("--labels", type=Path, required=True)
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "freeze":
        freeze(
            cohort_path=args.cohort,
            development_identities_path=args.development_identities,
            output_dir=args.output_dir,
            registry_path=args.model_registry,
        )
    else:
        evaluate(
            freeze_dir=args.freeze_dir,
            labels_path=args.labels,
            output_dir=args.output_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
