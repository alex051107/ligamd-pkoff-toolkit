#!/usr/bin/env python3
"""Build the small, frozen N31 experimental model registry.

This command deliberately does *less* than a model-search project.  It keeps
one panel, one grouped split rule, one default endpoint and one dynamic frame
view fixed, then answers a narrow question: after ligand/starting-structure
controls are present, does the P512 Dynamic10 block add useful out-of-fold
signal on the frozen engineering panel?

Two Combined30 out-of-fold (OOF) results are imported from the compatible N31
matrix rather than recomputed.  Only the missing Static20 Ridge and Random
Forest cells are fitted.  This matters scientifically as well as practically:
we do not quietly change a previously evaluated combined model while adding a
static ablation.  All four final estimators are then fit once on the full N31
panel and saved with their fold-local reducer and scaler inside the joblib
pipeline.  They remain ``EXPERIMENTAL`` models, not selected scientific
winners and not physical-koff estimators.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from ligamd_pkoff.resources import bundled_path
from sklearn.dummy import DummyRegressor

if __package__ in {None, ""}:  # pragma: no cover - cluster/direct invocation
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.koff_ml.complete_static_geometric import FEATURES as STATIC20_FEATURES
from scripts.koff_ml.endpoint_two_metric import endpoint_spec_from_contract
from scripts.koff_ml.io import sha256_file, write_json
from scripts.koff_ml.run_dual_endpoint_combined30_matrix_v1 import (
    BOOTSTRAP_REPEATS,
    MatrixError,
    RANDOM_STATE,
    build_pipeline,
    choose_configuration,
    frozen_outer_folds,
    metrics,
    model_families,
    paired_group_summary,
    per_group_mae,
)


SCHEMA_VERSION = "ligamd_experimental_model_registry_v1.0"
SCIENTIFIC_STATUS = "EXPERIMENTAL"
ENDPOINT_ARM = "simple_two_metric"
SAMPLER = "P512"
EXPECTED_SYSTEMS = 31
EXPECTED_EXACT_LIGAND_GROUPS = 27
INPUT_PROVENANCE_SCHEMA = "ligamd_frozen_n31_registry_input_provenance_v1.0"
DEFAULT_INPUT_PROVENANCE = bundled_path("contracts/frozen_n31_registry_input_provenance_v1.json")
DYNAMIC10_FEATURES = (
    "g50v2__pocket_com_distance_A__progress_000_050__mean",
    "g50v2__native_contact_fraction__progress_050_080__mean",
    "g50v2__ligand_pose_rmsd_A__progress_080_095__mean",
    "g50v2__ligand_pose_rmsd_A__progress_095_100__mean",
    "g50v2__global_min_heavy_distance_A__progress_095_100__mean",
    "g50v2__global_contact_fraction__progress_000_050__mean",
    "g50v2__global_contact_fraction__progress_080_095__mean",
    "g50v2__state_progress_occupancy__I",
    "g50v2__state_progress_occupancy__P_ONLY",
    "g50v2__ligand_internal_rmsd_A__progress_mean",
)
MODEL_PROFILES = (
    ("static20_ridge", "Static20", "ridge"),
    ("static20_random_forest", "Static20", "random_forest"),
    ("combined30_p512_ridge", "Combined30", "ridge"),
    ("combined30_p512_random_forest", "Combined30", "random_forest"),
)


class RegistryError(RuntimeError):
    """Raised when a frozen panel, split, or re-used OOF cell drifts."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RegistryError(message)


def write_tsv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    table = pd.DataFrame(list(rows))
    require(not table.empty, f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False, lineterminator="\n")


def _load_frozen_input_provenance(
    *,
    source_panel: Path,
    folds: Path,
    reused_oof: Path,
    endpoint_contract: Path,
    contract: Mapping[str, Any],
    receipt_path: Path,
) -> Mapping[str, Any]:
    """Refuse a nominal N31 rebuild when one authorised input has changed.

    This builder deliberately reuses two already-frozen Combined30 OOF cells.
    Its safety therefore depends on knowing exactly which panel, fold map, OOF
    table, and endpoint rule were authorised together.  A small public receipt
    stores only their names and digests; it contains no labels or system IDs.
    """

    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(f"cannot read frozen input provenance receipt: {exc}") from exc
    require(
        receipt.get("schema_version") == INPUT_PROVENANCE_SCHEMA,
        f"input provenance receipt schema must be {INPUT_PROVENANCE_SCHEMA}",
    )
    artifacts = receipt.get("input_artifacts")
    require(isinstance(artifacts, Mapping), "input provenance receipt lacks input_artifacts")
    for name, path in (
        ("source_panel", source_panel),
        ("folds", folds),
        ("reused_oof", reused_oof),
    ):
        record = artifacts.get(name)
        require(isinstance(record, Mapping), f"input provenance receipt lacks {name}")
        expected_hash = record.get("sha256")
        require(isinstance(expected_hash, str) and len(expected_hash) == 64, f"{name} lacks a SHA-256")
        require(
            sha256_file(path) == expected_hash,
            f"{name} hash differs from the authorised frozen N31 input receipt",
        )
    endpoint = receipt.get("endpoint_contract")
    require(isinstance(endpoint, Mapping), "input provenance receipt lacks endpoint_contract")
    require(
        endpoint.get("contract_id") == contract.get("contract_id"),
        "endpoint contract ID differs from the authorised frozen N31 receipt",
    )
    expected_endpoint_hash = endpoint.get("sha256")
    require(
        isinstance(expected_endpoint_hash, str) and sha256_file(endpoint_contract) == expected_endpoint_hash,
        "endpoint contract hash differs from the authorised frozen N31 receipt",
    )
    oof_contract = receipt.get("reused_oof_cell_contract")
    require(isinstance(oof_contract, Mapping), "input provenance receipt lacks reused_oof_cell_contract")
    require(oof_contract.get("endpoint_arm") == ENDPOINT_ARM, "receipt OOF endpoint arm is incompatible")
    require(oof_contract.get("sampler") == SAMPLER, "receipt OOF sampler is incompatible")
    require(oof_contract.get("system_count") == EXPECTED_SYSTEMS, "receipt OOF system count is incompatible")
    require(
        oof_contract.get("combined_feature_block") == "Combined30"
        and oof_contract.get("dummy_feature_block") == "No_features",
        "receipt OOF feature-block contract is incompatible",
    )
    require(
        set(oof_contract.get("models", ())) == {"ridge", "random_forest", "dummy_median"},
        "receipt OOF model contract is incompatible",
    )
    require(
        {"held_out_exact_ligand_group", "outer_test_fold"} <= set(oof_contract.get("required_columns", ())),
        "receipt must require explicit OOF fold provenance columns",
    )
    return receipt


def _load_panel(path: Path, folds_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_csv(path, sep="\t")
    folds = pd.read_csv(folds_path, sep="\t")
    required = {
        "system_id", "method", "experimental_pKoff", "exact_ligand_group",
        "protein_target", "endpoint_arm", *STATIC20_FEATURES, *DYNAMIC10_FEATURES,
    }
    missing = sorted(required - set(panel.columns))
    require(not missing, f"supervised panel is missing required columns: {missing}")
    require(set(panel["endpoint_arm"].astype(str)) == {ENDPOINT_ARM}, "panel endpoint arm is not simple_two_metric")
    require(set(panel["method"].astype(str)) == {"E512_v2", "P512", "U512"}, "panel sampler methods drifted")
    require(len(panel) == EXPECTED_SYSTEMS * 3, "N31 panel must contain 31 systems x three samplers")
    systems = sorted(panel["system_id"].astype(str).unique())
    require(len(systems) == EXPECTED_SYSTEMS, "N31 system count drifted")
    groups = panel[["system_id", "exact_ligand_group"]].drop_duplicates()
    require(len(groups) == EXPECTED_SYSTEMS, "one system must have one exact ligand group")
    require(groups["exact_ligand_group"].nunique() == EXPECTED_EXACT_LIGAND_GROUPS, "exact ligand group count drifted")
    fold_required = {"system_id", "exact_ligand_group", "outer_test_fold"}
    require(fold_required <= set(folds.columns), "frozen fold table misses required columns")
    require(len(folds) == EXPECTED_SYSTEMS and folds["system_id"].nunique() == EXPECTED_SYSTEMS, "frozen folds are not N31")
    require(set(folds["system_id"].astype(str)) == set(systems), "panel/fold system sets differ")
    joined = groups.merge(folds[list(fold_required)], on="system_id", suffixes=("_panel", "_fold"), validate="one_to_one")
    require(
        np.array_equal(
            joined["exact_ligand_group_panel"].astype(str).to_numpy(),
            joined["exact_ligand_group_fold"].astype(str).to_numpy(),
        ),
        "panel exact-ligand groups differ from frozen folds",
    )
    return panel, folds


def _one_row_per_system(panel: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Collapse sampler-duplicated Static20 only after proving equality."""

    rows: list[pd.Series] = []
    for system_id, group in panel.groupby("system_id", sort=True):
        require(len(group) == 3, f"{system_id}: expected exactly three sampler rows")
        values = group[list(features)].apply(pd.to_numeric, errors="raise")
        reference = values.iloc[0].to_numpy(dtype=float)
        require(
            np.allclose(values.to_numpy(dtype=float), reference[None, :], rtol=0.0, atol=0.0),
            f"{system_id}: Static20 differs between sampler rows",
        )
        for field in ("experimental_pKoff", "exact_ligand_group", "protein_target"):
            require(group[field].astype(str).nunique() == 1, f"{system_id}: {field} differs between sampler rows")
        rows.append(group.iloc[0])
    output = pd.DataFrame(rows).reset_index(drop=True)
    require(len(output) == EXPECTED_SYSTEMS, "Static20 system collapse did not produce N31")
    return output


def _panel_for_model(frame: pd.DataFrame, folds: pd.DataFrame, features: Sequence[str]) -> Any:
    """Construct the small Panel interface expected by frozen split helpers."""

    from types import SimpleNamespace

    ordered = frame.sort_values("system_id").reset_index(drop=True)
    frozen = folds.set_index("system_id").loc[ordered["system_id"].astype(str)]
    return SimpleNamespace(
        systems=ordered["system_id"].astype(str).to_numpy(),
        y=ordered["experimental_pKoff"].to_numpy(dtype=float),
        groups=ordered["exact_ligand_group"].astype(str).to_numpy(),
        outer_fold_ids=frozen["outer_test_fold"].astype(str).to_numpy(),
        features=tuple(features),
        frame=ordered,
    )


def _run_outer_oof(
    *, profile_id: str, feature_block: str, family_name: str, panel: Any
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit only the previously missing Static20 grouped OOF cells."""

    family = model_families(include_xgboost=False)[family_name]
    x = panel.frame[list(panel.features)].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    prediction = np.full(len(panel.y), np.nan, dtype=float)
    selection_rows: list[dict[str, Any]] = []
    reduction_rows: list[dict[str, Any]] = []
    for fold_number, (fold_id, held_out, train, test) in enumerate(frozen_outer_folds(panel), start=1):
        config, inner_score = choose_configuration(family, x[train], panel.y[train], panel.groups[train])
        estimator = build_pipeline(family, config)
        estimator.fit(x[train], panel.y[train])
        values = np.asarray(estimator.predict(x[test]), dtype=float)
        require(np.all(np.isfinite(values)), f"{profile_id}/{fold_id}: non-finite outer prediction")
        require(np.all(~np.isfinite(prediction[test])), f"{profile_id}/{fold_id}: duplicate OOF prediction")
        prediction[test] = values
        reducer = estimator.named_steps["reduce"]
        retained = [panel.features[index] for index in reducer.selected_original_indices_]
        selection_rows.append({
            "profile_id": profile_id,
            "feature_block": feature_block,
            "model": family_name,
            "outer_fold_number": fold_number,
            "outer_fold_id": fold_id,
            "held_out_exact_ligand_group": held_out,
            "inner_selection_metric": "exact_ligand_group_equal_mae",
            "inner_selected_score": inner_score,
            "selected_hyperparameters_json": json.dumps(dict(config), sort_keys=True, separators=(",", ":")),
        })
        reduction_rows.append({
            "profile_id": profile_id,
            "outer_fold_id": fold_id,
            "input_feature_count": len(panel.features),
            "retained_feature_count": len(retained),
            "retained_features": ";".join(retained),
            "fit_scope": "outer_training_only",
            "variance_threshold": 1e-12,
            "absolute_spearman_threshold": 0.95,
            "correlation_rule": "complete_linkage_medoid",
            "scaling": "standard_scaler",
        })
    require(np.all(np.isfinite(prediction)), f"{profile_id}: incomplete OOF prediction")
    return prediction, selection_rows, reduction_rows


def _reuse_oof(
    *, path: Path, expected_panel: Any, model: str, feature_block: str = "Combined30"
) -> np.ndarray:
    """Read the exact compatible P512 Combined30 OOF cell without refitting it."""

    table = pd.read_csv(path, sep="\t")
    required = {
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
    }
    missing = sorted(required - set(table.columns))
    require(not missing, f"re-used OOF is missing provenance columns: {missing}")
    mask = (
        (table["endpoint_arm"].astype(str) == ENDPOINT_ARM)
        & (table["sampler"].astype(str) == SAMPLER)
        & (table["feature_block"].astype(str) == feature_block)
        & (table["model"].astype(str) == model)
    )
    cell = table.loc[mask].copy()
    require(len(cell) == EXPECTED_SYSTEMS, f"re-used {model} OOF cell is not N31")
    cell = cell.set_index("system_id").loc[list(expected_panel.systems)].reset_index()
    require(np.allclose(cell["experimental_pKoff"].to_numpy(float), expected_panel.y, rtol=0.0, atol=0.0), f"re-used {model} labels differ")
    require(np.array_equal(cell["exact_ligand_group"].astype(str).to_numpy(), expected_panel.groups), f"re-used {model} groups differ")
    require(
        np.array_equal(cell["held_out_exact_ligand_group"].astype(str).to_numpy(), expected_panel.groups),
        f"re-used {model} held-out ligand groups differ from frozen folds",
    )
    require(
        np.array_equal(cell["outer_test_fold"].astype(str).to_numpy(), expected_panel.outer_fold_ids),
        f"re-used {model} outer-fold IDs differ from frozen folds",
    )
    require(np.all(np.isfinite(cell["predicted_pKoff"].to_numpy(float))), f"re-used {model} OOF has non-finite values")
    return cell["predicted_pKoff"].to_numpy(dtype=float)


def _reuse_dummy(*, path: Path, expected_panel: Any) -> np.ndarray:
    return _reuse_oof(
        path=path,
        expected_panel=expected_panel,
        model="dummy_median",
        feature_block="No_features",
    )


def _fit_full_profile(*, profile_id: str, feature_block: str, family_name: str, panel: Any) -> tuple[Any, Mapping[str, Any], float]:
    family = model_families(include_xgboost=False)[family_name]
    x = panel.frame[list(panel.features)].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    config, inner_score = choose_configuration(family, x, panel.y, panel.groups)
    estimator = build_pipeline(family, config)
    estimator.fit(x, panel.y)
    return estimator, dict(config), float(inner_score)


def _feature_ranges(frame: pd.DataFrame, features: Sequence[str]) -> dict[str, dict[str, float]]:
    return {
        feature: {
            "minimum": float(frame[feature].min()),
            "maximum": float(frame[feature].max()),
        }
        for feature in features
    }


def build_registry(
    *,
    source_panel: Path,
    folds: Path,
    reused_oof: Path,
    endpoint_contract: Path,
    output_dir: Path,
    input_provenance: Path = DEFAULT_INPUT_PROVENANCE,
) -> dict[str, Any]:
    """Build four full-fit experimental models and their transparent OOF evidence."""

    require(not output_dir.exists(), f"refusing to overwrite registry output: {output_dir}")
    contract = json.loads(endpoint_contract.read_text(encoding="utf-8"))
    require(contract.get("schema_version") == "ligamd_endpoint_contract_v2.0", "endpoint-v2 contract schema differs")
    require(contract.get("compatibility", {}).get("sampler", "").startswith("P512"), "endpoint-v2 contract is not P512 compatible")
    try:
        endpoint_spec = endpoint_spec_from_contract(contract)
    except ValueError as exc:
        raise RegistryError(f"endpoint contract cannot be converted to an executable rule: {exc}") from exc
    input_receipt = _load_frozen_input_provenance(
        source_panel=source_panel,
        folds=folds,
        reused_oof=reused_oof,
        endpoint_contract=endpoint_contract,
        contract=contract,
        receipt_path=input_provenance,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    # A registry needs the exact rule that produced its dynamic features beside
    # its model files. Copying this small public JSON makes a newly generated
    # registry self-contained without copying a panel, a fold map, or labels.
    shutil.copyfile(endpoint_contract, output_dir / "endpoint_v2.json")
    panel, fold_table = _load_panel(source_panel, folds)
    static_frame = _one_row_per_system(panel, STATIC20_FEATURES)
    combined_frame = panel.loc[panel["method"].astype(str) == SAMPLER].copy()
    require(len(combined_frame) == EXPECTED_SYSTEMS, "P512 Combined30 rows are not N31")
    static_panel = _panel_for_model(static_frame, fold_table, STATIC20_FEATURES)
    combined_panel = _panel_for_model(combined_frame, fold_table, STATIC20_FEATURES + DYNAMIC10_FEATURES)
    require(np.array_equal(static_panel.systems, combined_panel.systems), "Static20 and Combined30 system order differs")
    require(np.allclose(static_panel.y, combined_panel.y, rtol=0.0, atol=0.0), "Static20 and Combined30 labels differ")
    require(np.array_equal(static_panel.groups, combined_panel.groups), "Static20 and Combined30 groups differ")

    static_predictions: dict[str, np.ndarray] = {}
    inner_rows: list[dict[str, Any]] = []
    reduction_rows: list[dict[str, Any]] = []
    for profile_id, _, family in MODEL_PROFILES[:2]:
        prediction, selected, reduced = _run_outer_oof(
            profile_id=profile_id, feature_block="Static20", family_name=family, panel=static_panel
        )
        static_predictions[family] = prediction
        inner_rows.extend(selected)
        reduction_rows.extend(reduced)

    combined_predictions = {
        "ridge": _reuse_oof(path=reused_oof, expected_panel=combined_panel, model="ridge"),
        "random_forest": _reuse_oof(path=reused_oof, expected_panel=combined_panel, model="random_forest"),
    }
    dummy_prediction = _reuse_dummy(path=reused_oof, expected_panel=combined_panel)

    score_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    profiles: dict[str, dict[str, Any]] = {}
    model_dir = output_dir / "models"
    model_dir.mkdir()
    profile_panels = {
        "static20_ridge": static_panel,
        "static20_random_forest": static_panel,
        "combined30_p512_ridge": combined_panel,
        "combined30_p512_random_forest": combined_panel,
    }
    predictions = {
        "static20_ridge": static_predictions["ridge"],
        "static20_random_forest": static_predictions["random_forest"],
        "combined30_p512_ridge": combined_predictions["ridge"],
        "combined30_p512_random_forest": combined_predictions["random_forest"],
    }
    oof_sources = {
        "static20_ridge": "NEW_FROZEN_STATIC20_GROUP_LOGO",
        "static20_random_forest": "NEW_FROZEN_STATIC20_GROUP_LOGO",
        "combined30_p512_ridge": "REUSED_COMPATIBLE_N31_MATRIX_OOF",
        "combined30_p512_random_forest": "REUSED_COMPATIBLE_N31_MATRIX_OOF",
    }
    dummy_metrics = metrics(combined_panel.y, dummy_prediction, combined_panel.groups)
    dummy_group_mae = per_group_mae(combined_panel.y, dummy_prediction, combined_panel.groups)
    score_rows.append({
        "profile_id": "dummy_median",
        "scientific_status": SCIENTIFIC_STATUS,
        "feature_block": "No_features",
        "sampler": "not_applicable",
        "model": "dummy_median",
        "oof_source": "REUSED_COMPATIBLE_N31_MATRIX_OOF",
        **dummy_metrics,
        "versus_dummy_group_equal_mae_delta": 0.0,
        "bootstrap_95_ci_lower": 0.0,
        "bootstrap_95_ci_upper": 0.0,
    })
    for index, (profile_id, feature_block, family_name) in enumerate(MODEL_PROFILES, start=1):
        profile_panel = profile_panels[profile_id]
        prediction = predictions[profile_id]
        result_metrics = metrics(profile_panel.y, prediction, profile_panel.groups)
        candidate_group_mae = per_group_mae(profile_panel.y, prediction, profile_panel.groups)
        paired = paired_group_summary(candidate_group_mae, dummy_group_mae, RANDOM_STATE + index)
        score_rows.append({
            "profile_id": profile_id,
            "scientific_status": SCIENTIFIC_STATUS,
            "feature_block": feature_block,
            "sampler": SAMPLER if feature_block == "Combined30" else "not_applicable",
            "model": family_name,
            "oof_source": oof_sources[profile_id],
            **result_metrics,
            **paired,
            "versus_dummy_group_equal_mae_delta": paired["group_equal_mae_delta"],
        })
        for row_index, system_id in enumerate(profile_panel.systems):
            oof_rows.append({
                "profile_id": profile_id,
                "scientific_status": SCIENTIFIC_STATUS,
                "system_id": system_id,
                "protein_target": str(profile_panel.frame.iloc[row_index]["protein_target"]),
                "exact_ligand_group": profile_panel.groups[row_index],
                "experimental_pKoff": profile_panel.y[row_index],
                "predicted_experimental_pKoff": prediction[row_index],
                "absolute_error": abs(prediction[row_index] - profile_panel.y[row_index]),
                "oof_source": oof_sources[profile_id],
            })
        full_estimator, full_config, full_inner_score = _fit_full_profile(
            profile_id=profile_id, feature_block=feature_block, family_name=family_name, panel=profile_panel
        )
        artifact_name = f"{profile_id}.joblib"
        joblib.dump(full_estimator, model_dir / artifact_name)
        profiles[profile_id] = {
            "scientific_status": SCIENTIFIC_STATUS,
            "feature_block": feature_block,
            "model_family": family_name,
            "artifact": f"models/{artifact_name}",
            "input_feature_order": list(profile_panel.features),
            "input_feature_count": len(profile_panel.features),
            "training_system_count": len(profile_panel.systems),
            "training_exact_ligand_group_count": len(set(profile_panel.groups)),
            "training_protein_targets": sorted(set(profile_panel.frame["protein_target"].astype(str))),
            "training_feature_ranges": _feature_ranges(profile_panel.frame, profile_panel.features),
            "full_panel_inner_logo_selection_metric": "exact_ligand_group_equal_mae",
            "full_panel_selected_hyperparameters": full_config,
            "full_panel_inner_logo_score": full_inner_score,
            "endpoint_contract": "endpoint_v2.json" if feature_block == "Combined30" else "canonical_bound_structure_only",
            "sampler": SAMPLER if feature_block == "Combined30" else None,
            "replica_pooling": "arithmetic_mean_of_exactly_three_endpoint_PASS_replicas" if feature_block == "Combined30" else None,
            "scope_limitations": [
                "predictions target experimental pKoff labels; they are not physical koff estimates",
                "range checks are plausibility warnings, not evidence of external generalization",
                "the N31 panel is a frozen engineering panel; no profile is a selected final scientific model",
            ],
        }

    write_tsv(output_dir / "model_scoreboard.tsv", score_rows)
    write_tsv(output_dir / "oof_predictions.tsv", oof_rows)
    write_tsv(output_dir / "static20_inner_selection.tsv", inner_rows)
    write_tsv(output_dir / "static20_fold_local_reduction.tsv", reduction_rows)
    registry = {
        "schema_version": SCHEMA_VERSION,
        "scientific_status": SCIENTIFIC_STATUS,
        "target": "experimental_pKoff",
        "physical_koff_estimated": False,
        "endpoint": {
            "contract_file": "endpoint_v2.json",
            "spec": asdict(endpoint_spec),
            "episode": "production frame 0 through first persistence-confirmed onset, inclusive",
        },
        "panel": {
            "source_panel": {
                "artifact_name": source_panel.name,
                "sha256": sha256_file(source_panel),
                "availability": "separately_authorised_supervised_panel_not_distributed",
            },
            "folds": {
                "artifact_name": folds.name,
                "sha256": sha256_file(folds),
                "availability": "separately_authorised_fold_map_not_distributed",
            },
            "system_count": EXPECTED_SYSTEMS,
            "exact_ligand_group_count": EXPECTED_EXACT_LIGAND_GROUPS,
            "outer_and_inner_split": "exact-ligand leave-one-group-out",
            "preprocessing_scope": "constant/correlation reduction and scaling fit inside each training fold",
        },
        "reused_evidence": {
            "n31_combined_oof": {
                "artifact_name": reused_oof.name,
                "sha256": sha256_file(reused_oof),
                "availability": "separately_authorised_saved_OOF_not_distributed",
            },
            "authorised_input_provenance_receipt": {
                "artifact_name": input_provenance.name,
                "sha256": sha256_file(input_provenance),
                "schema_version": input_receipt["schema_version"],
            },
            "reason": "compatible simple-two-metric/P512/Combined30 N31 cells were already frozen; not rerun",
        },
        "profiles": profiles,
        "prediction_status_rules": {
            "WITHIN_OBSERVED_SCOPE": "all required inputs finite and within the training min/max range; protein target appears in the observed training target set",
            "CAUTION_OUTSIDE_OBSERVED_RANGE": "inputs are structurally complete but at least one required feature is outside its N31 observed range or the target is unfamiliar",
            "OUT_OF_SCOPE_NO_PREDICTION": "required input is missing, non-finite, wrong-dimensional, lacks three endpoint-PASS replicas for a Combined30 profile, or endpoint-v2 validation fails",
        },
    }
    write_json(output_dir / "model_registry.json", registry)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS_EXPERIMENTAL_N31_REGISTRY_BUILT",
        "oof_combined_cells_recomputed": False,
        "oof_static_cells_recomputed": True,
        "full_fit_profiles": list(profiles),
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "claims": {
            "sampler_selected": False,
            "representation_selected": False,
            "model_selected": False,
            "physical_koff_estimated": False,
        },
    }
    write_json(output_dir / "execution_receipt.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--reused-oof", type=Path, required=True)
    parser.add_argument(
        "--endpoint-contract",
        type=Path,
        default=bundled_path("contracts/endpoint_v2.json"),
    )
    parser.add_argument(
        "--input-provenance",
        type=Path,
        default=DEFAULT_INPUT_PROVENANCE,
        help="receipt binding the authorised N31 panel, folds, OOF cells, and endpoint contract",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        receipt = build_registry(
            source_panel=args.source_panel,
            folds=args.folds,
            reused_oof=args.reused_oof,
            endpoint_contract=args.endpoint_contract,
            input_provenance=args.input_provenance,
            output_dir=args.output_dir,
        )
    except (OSError, ValueError, RegistryError, MatrixError) as exc:
        print(json.dumps({"schema_version": SCHEMA_VERSION, "status": "FAIL", "reason": str(exc)}))
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
