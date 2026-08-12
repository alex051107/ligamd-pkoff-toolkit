#!/usr/bin/env python3
"""Run the matched dual-endpoint Combined30 classical-model matrix.

This runner requires two explicit
endpoint-specific Current30 tables and refuses to run unless they have the
same systems, samplers, Static20 values, labels, exact-ligand folds and feature
schema.

The scientific comparison is intentionally narrow:

* endpoint arms: current-v2.1 and the two-metric/simple endpoint;
* frame samplers: E512_v2, P512 and U512;
* representation: Combined30 = Static20 + sampler-specific Dynamic10;
* models: one shared median Dummy, Ridge, Random Forest and XGBoost.

Every learned operation is fitted inside the relevant training split.  Outer
and inner splits are exact-ligand leave-one-group-out (LOGO).  Constant-field
removal, absolute-Spearman correlation reduction, scaling and hyperparameter
selection never read an outer-test row.

The command can be exercised with ``--synthetic-smoke-only``.  That flag is
written into every output and must never be used for a scientific claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, pearsonr, spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.koff_ml.fold_local_spearman_reducer import FoldLocalSpearmanReducer


SCHEMA = "ligamd_dual_endpoint_combined30_matrix_v1.0"
RANDOM_STATE = 20260809
BOOTSTRAP_REPEATS = 5000
EXPECTED_METHODS = ("E512_v2", "P512", "U512")
CURRENT_ENDPOINT_DEFAULT = "current_v2_1"
SIMPLE_ENDPOINT_DEFAULT = "simple_two_metric"
STATIC_PREFIXES = ("ligand_static__", "initial_geometry__")
DYNAMIC_PREFIX = "g50v2__"
VARIANCE_THRESHOLD = 1e-12
CORRELATION_THRESHOLD = 0.95
FEATURE_BLOCK = "Combined30"
FORBIDDEN_FEATURE_TOKENS = (
    "pkoff",
    "koff",
    "label",
    "target",
    "temperature",
    "sigma",
    "boost",
    "outcome",
)


class MatrixError(RuntimeError):
    """Fail-closed input, fold, model or output error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MatrixError(message)


def write_tsv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    require(bool(materialized), f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(materialized[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(materialized)


def normalize_ids(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    require(column in frame.columns, f"missing required column: {column}")
    result = frame.copy()
    result[column] = result[column].astype(str).str.upper().str.strip()
    require(result[column].ne("").all(), f"empty identifier in {column}")
    return result


def read_tsv(path: Path, label: str) -> pd.DataFrame:
    require(path.is_file(), f"missing {label}: {path}")
    frame = pd.read_csv(path, sep="\t")
    require(not frame.empty, f"empty {label}: {path}")
    return frame


def choose_column(frame: pd.DataFrame, choices: Sequence[str], label: str) -> str:
    observed = [name for name in choices if name in frame.columns]
    require(len(observed) == 1, f"{label} must contain exactly one of {list(choices)}; observed={observed}")
    return observed[0]


def group_logo(groups: np.ndarray, minimum_groups: int = 3) -> list[tuple[str, np.ndarray, np.ndarray]]:
    values = np.asarray(groups, dtype=object).astype(str)
    unique = sorted(set(values))
    require(len(unique) >= minimum_groups, f"exact-ligand LOGO requires >={minimum_groups} groups; observed={len(unique)}")
    folds: list[tuple[str, np.ndarray, np.ndarray]] = []
    for held_out in unique:
        test = np.flatnonzero(values == held_out)
        train = np.flatnonzero(values != held_out)
        require(len(train) > 0 and len(test) > 0, f"empty LOGO split: {held_out}")
        require(not (set(values[train]) & set(values[test])), f"exact-ligand leakage: {held_out}")
        folds.append((held_out, train, test))
    return folds


def same_partition(left: Sequence[str], right: Sequence[str]) -> bool:
    if len(left) != len(right):
        return False
    return all(
        (left[i] == left[j]) == (right[i] == right[j])
        for i in range(len(left))
        for j in range(len(left))
    )


@dataclass(frozen=True)
class EndpointTable:
    endpoint_arm: str
    frame: pd.DataFrame
    static_features: tuple[str, ...]
    dynamic_features: tuple[str, ...]


@dataclass(frozen=True)
class Panel:
    systems: tuple[str, ...]
    y: np.ndarray
    groups: np.ndarray
    targets: np.ndarray
    label_tiers: np.ndarray
    outer_fold_ids: np.ndarray
    manifest_groups: np.ndarray
    current: EndpointTable
    simple: EndpointTable


def load_endpoint_table(path: Path, expected_arm: str) -> EndpointTable:
    frame = normalize_ids(read_tsv(path, f"Current30 table for {expected_arm}"), "system_id")
    required = {
        "system_id",
        "method",
        "budget",
        "endpoint_arm",
        "experimental_pkoff_read_during_feature_freeze",
    }
    require(required <= set(frame.columns), f"{expected_arm}: missing columns {sorted(required - set(frame.columns))}")
    require(not frame.duplicated(["system_id", "method"]).any(), f"{expected_arm}: duplicate system/sampler row")
    require(set(frame["endpoint_arm"].astype(str)) == {expected_arm}, f"{expected_arm}: endpoint_arm value drifted")
    require(set(frame["method"].astype(str)) == set(EXPECTED_METHODS), f"{expected_arm}: sampler set drifted")
    require(set(pd.to_numeric(frame["budget"], errors="raise").astype(int)) == {512}, f"{expected_arm}: budget is not 512")
    require(frame.groupby("system_id")["method"].nunique().eq(3).all(), f"{expected_arm}: a system lacks one of three samplers")
    read_flag = frame["experimental_pkoff_read_during_feature_freeze"].astype(str).str.lower()
    require(set(read_flag) <= {"false", "0"}, f"{expected_arm}: feature freeze reports reading pKoff")

    static = tuple(name for name in frame.columns if name.startswith(STATIC_PREFIXES))
    dynamic = tuple(name for name in frame.columns if name.startswith(DYNAMIC_PREFIX))
    require(len(static) == 20, f"{expected_arm}: expected Static20; observed={len(static)}")
    require(len(dynamic) == 10, f"{expected_arm}: expected Dynamic10; observed={len(dynamic)}")
    require(len(set(static + dynamic)) == 30, f"{expected_arm}: feature names are duplicated")
    prohibited = [
        name
        for name in static + dynamic
        if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    require(not prohibited, f"{expected_arm}: prohibited feature names: {prohibited}")
    numeric = frame.loc[:, list(static + dynamic)].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    require(np.all(np.isfinite(numeric)), f"{expected_arm}: features contain NaN/inf")

    # Static20 describes the starting structure and cannot change with sampler.
    static_drift = frame.groupby("system_id")[list(static)].nunique(dropna=False).max(axis=1)
    require(static_drift.le(1).all(), f"{expected_arm}: Static20 changes across samplers")
    return EndpointTable(expected_arm, frame, static, dynamic)


def validate_endpoint_parity(current: EndpointTable, simple: EndpointTable) -> None:
    require(current.endpoint_arm != simple.endpoint_arm, "endpoint arms must have distinct identifiers")
    require(current.static_features == simple.static_features, "Static20 schema differs between endpoint arms")
    require(current.dynamic_features == simple.dynamic_features, "Dynamic10 schema differs between endpoint arms")
    key = ["system_id", "method"]
    current_sorted = current.frame.sort_values(key, kind="stable").reset_index(drop=True)
    simple_sorted = simple.frame.sort_values(key, kind="stable").reset_index(drop=True)
    require(current_sorted[key].equals(simple_sorted[key]), "endpoint arms do not contain identical system/sampler rows")
    require(
        np.array_equal(
            current_sorted.loc[:, list(current.static_features)].to_numpy(dtype=float),
            simple_sorted.loc[:, list(simple.static_features)].to_numpy(dtype=float),
        ),
        "Static20 values differ between endpoint arms",
    )


def load_panel(
    panel_path: Path,
    fold_path: Path,
    current: EndpointTable,
    simple: EndpointTable,
) -> Panel:
    membership = normalize_ids(read_tsv(panel_path, "common endpoint panel"), "system_id")
    folds = normalize_ids(read_tsv(fold_path, "common exact-ligand folds"), "system_id")
    require(membership["system_id"].is_unique, "panel has duplicate systems")
    require(folds["system_id"].is_unique, "fold table has duplicate systems")
    require("exact_ligand_group" in membership.columns, "panel lacks exact_ligand_group")
    require("exact_ligand_group" in folds.columns, "folds lack exact_ligand_group")
    require("outer_test_fold" in folds.columns, "folds lack outer_test_fold")
    require("protein_target" in membership.columns, "panel lacks protein_target")
    y_column = choose_column(membership, ("pKoff", "experimental_pKoff"), "panel label")
    tier_column = choose_column(membership, ("label_role", "label_source_tier", "label_tier"), "panel label tier")

    table_systems = set(current.frame["system_id"])
    require(table_systems == set(simple.frame["system_id"]), "endpoint system parity was lost")
    require(set(membership["system_id"]) == table_systems, "panel systems differ from endpoint feature tables")
    require(set(folds["system_id"]) == table_systems, "fold systems differ from endpoint feature tables")
    membership = membership.sort_values("system_id", kind="stable").reset_index(drop=True)
    systems = tuple(membership["system_id"].astype(str))
    fold_indexed = folds.set_index("system_id").loc[list(systems)]
    panel_groups = membership["exact_ligand_group"].astype(str).to_numpy()
    fold_groups = fold_indexed["exact_ligand_group"].astype(str).to_numpy()
    require(np.array_equal(panel_groups, fold_groups), "panel and fold exact-ligand groups disagree")
    fold_ids = fold_indexed["outer_test_fold"].astype(str).to_numpy()
    require(same_partition(panel_groups.tolist(), fold_ids.tolist()), "outer_test_fold does not reproduce exact-ligand partition")
    for fold_id in sorted(set(fold_ids)):
        test = fold_ids == fold_id
        require(len(set(panel_groups[test])) == 1, f"fold {fold_id} contains multiple exact ligands")
        require(not (set(panel_groups[test]) & set(panel_groups[~test])), f"fold {fold_id} leaks an exact ligand")

    y = pd.to_numeric(membership[y_column], errors="raise").to_numpy(dtype=float)
    require(np.all(np.isfinite(y)), "panel labels contain NaN/inf")
    require(float(np.var(y, ddof=0)) > 0.0, "panel pKoff labels have zero variance")
    require(len(set(panel_groups)) >= 3, "panel requires at least three exact-ligand groups")
    group_logo(panel_groups)
    return Panel(
        systems=systems,
        y=y,
        groups=panel_groups,
        targets=membership["protein_target"].astype(str).to_numpy(),
        label_tiers=membership[tier_column].astype(str).to_numpy(),
        outer_fold_ids=fold_ids,
        manifest_groups=fold_groups,
        current=current,
        simple=simple,
    )


@dataclass(frozen=True)
class ModelFamily:
    name: str
    configurations: tuple[Mapping[str, Any], ...]
    build: Callable[[Mapping[str, Any]], Any]


def model_families(*, include_xgboost: bool = True) -> dict[str, ModelFamily]:
    """Return the requested classical model families.

    The full historical matrix includes XGBoost, so its ordinary route still
    fails closed when that optional dependency is missing.  A small registry
    rebuild uses only Ridge and Random Forest, however.  Keeping the import
    inside the requested branch lets that narrower, documented developer task
    run in the public base environment without quietly changing its model set.
    """

    families = {
        "ridge": ModelFamily(
            "ridge",
            tuple({"alpha": alpha} for alpha in (0.01, 0.1, 1.0, 10.0, 100.0)),
            lambda config: Ridge(alpha=float(config["alpha"])),
        ),
        "random_forest": ModelFamily(
            "random_forest",
            (
                {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1, "max_features": 1.0},
                {"n_estimators": 100, "max_depth": 3, "min_samples_leaf": 2, "max_features": "sqrt"},
                {"n_estimators": 100, "max_depth": 4, "min_samples_leaf": 1, "max_features": "sqrt"},
            ),
            lambda config: RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1, **dict(config)),
        ),
    }
    if not include_xgboost:
        return families
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:  # fail closed rather than silently substituting
        raise MatrixError(f"XGBoost is required for this fixed matrix: {type(exc).__name__}: {exc}") from exc
    families["xgboost"] = ModelFamily(
            "xgboost",
            (
                {"n_estimators": 50, "max_depth": 1, "learning_rate": 0.05, "reg_lambda": 10.0},
                {"n_estimators": 100, "max_depth": 1, "learning_rate": 0.05, "reg_lambda": 1.0},
                {"n_estimators": 50, "max_depth": 2, "learning_rate": 0.05, "reg_lambda": 10.0},
                {"n_estimators": 100, "max_depth": 2, "learning_rate": 0.10, "reg_lambda": 1.0},
            ),
            lambda config: XGBRegressor(
                objective="reg:squarederror",
                min_child_weight=1.0,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.0,
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbosity=0,
                **dict(config),
            ),
        )
    return families


def build_pipeline(family: ModelFamily, config: Mapping[str, Any]) -> Pipeline:
    return Pipeline([
        (
            "reduce",
            FoldLocalSpearmanReducer(
                variance_threshold=VARIANCE_THRESHOLD,
                correlation_threshold=CORRELATION_THRESHOLD,
            ),
        ),
        ("scale", StandardScaler()),
        ("model", family.build(config)),
    ])


def group_equal_mae(y: np.ndarray, prediction: np.ndarray, groups: np.ndarray) -> float:
    error = np.abs(np.asarray(prediction, dtype=float) - np.asarray(y, dtype=float))
    return float(np.mean([np.mean(error[groups.astype(str) == group]) for group in sorted(set(groups.astype(str)))]))


def frozen_outer_folds(panel: Panel) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    rows: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    for fold_id in sorted(set(panel.outer_fold_ids.astype(str))):
        test = np.flatnonzero(panel.outer_fold_ids.astype(str) == fold_id)
        train = np.flatnonzero(panel.outer_fold_ids.astype(str) != fold_id)
        held = sorted(set(panel.groups[test].astype(str)))
        require(len(held) == 1, f"{fold_id}: test fold is not one exact ligand")
        require(not (set(panel.groups[train]) & set(panel.groups[test])), f"{fold_id}: group leakage")
        rows.append((fold_id, held[0], train, test))
    require(len(rows) == len(set(panel.groups)), "fold count does not equal exact-ligand group count")
    return rows


def choose_configuration(
    family: ModelFamily,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[Mapping[str, Any], float]:
    candidates: list[tuple[float, str, Mapping[str, Any]]] = []
    for config in family.configurations:
        prediction = np.full(len(y), np.nan, dtype=float)
        for _, train, test in group_logo(groups, minimum_groups=2):
            estimator = build_pipeline(family, config)
            estimator.fit(x[train], y[train])
            prediction[test] = estimator.predict(x[test])
        require(np.all(np.isfinite(prediction)), f"non-finite inner OOF: {family.name}/{config}")
        score = group_equal_mae(y, prediction, groups)
        canonical = json.dumps(dict(config), sort_keys=True, separators=(",", ":"))
        candidates.append((score, canonical, config))
    score, _, selected = min(candidates, key=lambda item: (item[0], item[1]))
    return selected, float(score)


def metrics(y: np.ndarray, prediction: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction, dtype=float) - np.asarray(y, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        rho = float(spearmanr(y, prediction).statistic)
        pearson = float(pearsonr(y, prediction).statistic)
    variance = float(np.var(prediction, ddof=0))
    if variance <= 1e-12:
        intercept, slope = float("nan"), float("nan")
    else:
        slope = float(np.mean((prediction - np.mean(prediction)) * (y - np.mean(y))) / variance)
        intercept = float(np.mean(y) - slope * np.mean(prediction))
    mse = float(np.mean(error**2))
    return {
        "exact_ligand_group_equal_mae": group_equal_mae(y, prediction, groups),
        "system_equal_mae": float(np.mean(np.abs(error))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "median_absolute_error": float(np.median(np.abs(error))),
        "r_squared": float(r2_score(y, prediction)),
        "spearman_rho": rho if math.isfinite(rho) else float("nan"),
        "pearson_r": pearson if math.isfinite(pearson) else float("nan"),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def per_group_mae(y: np.ndarray, prediction: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    absolute = np.abs(np.asarray(prediction, dtype=float) - np.asarray(y, dtype=float))
    return {
        group: float(np.mean(absolute[groups.astype(str) == group]))
        for group in sorted(set(groups.astype(str)))
    }


def paired_group_summary(candidate: Mapping[str, float], reference: Mapping[str, float], seed: int) -> dict[str, Any]:
    require(set(candidate) == set(reference), "paired group comparison has mismatched groups")
    groups = sorted(candidate)
    delta = np.asarray([candidate[group] - reference[group] for group in groups], dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.asarray([
        np.mean(rng.choice(delta, size=len(delta), replace=True))
        for _ in range(BOOTSTRAP_REPEATS)
    ])
    leave_one_out = np.asarray([
        np.mean(np.delete(delta, index)) for index in range(len(delta))
    ]) if len(delta) > 1 else np.asarray([float("nan")])
    full = float(np.mean(delta))
    finite_loo = leave_one_out[np.isfinite(leave_one_out)]
    sign_reversal = bool(
        len(finite_loo)
        and any((value > 0.0) != (full > 0.0) for value in finite_loo if value != 0.0)
        and full != 0.0
    )
    return {
        "group_equal_mae_delta": full,
        "bootstrap_95_ci_lower": float(np.quantile(draws, 0.025)),
        "bootstrap_95_ci_upper": float(np.quantile(draws, 0.975)),
        "bootstrap_probability_delta_below_zero": float(np.mean(draws < 0.0)),
        "improved_group_count": int(np.sum(delta < 0.0)),
        "tied_group_count": int(np.sum(delta == 0.0)),
        "total_group_count": len(groups),
        "leave_one_group_delta_min": float(np.min(finite_loo)) if len(finite_loo) else float("nan"),
        "leave_one_group_delta_max": float(np.max(finite_loo)) if len(finite_loo) else float("nan"),
        "leave_one_group_sign_reversal": sign_reversal,
    }


@dataclass(frozen=True)
class UniqueCellResult:
    endpoint_arm: str
    sampler: str
    model: str
    cell_id: str
    prediction: np.ndarray
    metrics: Mapping[str, float]
    group_mae: Mapping[str, float]


def endpoint_matrix(panel: Panel, endpoint: EndpointTable) -> tuple[list[UniqueCellResult], list[dict[str, Any]], list[dict[str, Any]]]:
    families = model_families()
    outer_folds = frozen_outer_folds(panel)
    systems = list(panel.systems)
    selection_rows: list[dict[str, Any]] = []
    reduction_rows: list[dict[str, Any]] = []
    results: list[UniqueCellResult] = []
    indexed = endpoint.frame.set_index(["system_id", "method"])
    feature_names = tuple(endpoint.static_features + endpoint.dynamic_features)

    for sampler in EXPECTED_METHODS:
        x = (
            indexed.loc[[(system, sampler) for system in systems], list(feature_names)]
            .apply(pd.to_numeric, errors="raise")
            .to_numpy(dtype=float)
        )
        require(x.shape == (len(systems), 30), f"{endpoint.endpoint_arm}/{sampler}: Combined30 shape drifted")
        for model_name in ("ridge", "random_forest", "xgboost"):
            family = families[model_name]
            prediction = np.full(len(systems), np.nan, dtype=float)
            cell_id = f"{endpoint.endpoint_arm}__{sampler}__Combined30__{model_name}"
            for fold_index, (fold_id, held_out, train, test) in enumerate(outer_folds, start=1):
                config, inner_score = choose_configuration(family, x[train], panel.y[train], panel.groups[train])
                estimator = build_pipeline(family, config)
                estimator.fit(x[train], panel.y[train])
                fold_prediction = np.asarray(estimator.predict(x[test]), dtype=float)
                require(np.all(np.isfinite(fold_prediction)), f"{cell_id}/{fold_id}: non-finite prediction")
                require(np.all(~np.isfinite(prediction[test])), f"{cell_id}/{fold_id}: duplicate OOF prediction")
                prediction[test] = fold_prediction
                reducer = estimator.named_steps["reduce"]
                retained = [feature_names[index] for index in reducer.selected_original_indices_]
                selection_rows.append({
                    "endpoint_arm": endpoint.endpoint_arm,
                    "sampler": sampler,
                    "feature_block": FEATURE_BLOCK,
                    "model": model_name,
                    "execution_cell_id": cell_id,
                    "outer_fold": fold_index,
                    "outer_fold_id": fold_id,
                    "held_out_exact_ligand_group": held_out,
                    "inner_selection_metric": "exact_ligand_group_equal_mae",
                    "inner_selection_rule": "MINIMUM_MEAN_INNER_GROUP_EQUAL_MAE_THEN_CANONICAL_CONFIG",
                    "inner_selected_score": f"{inner_score:.12g}",
                    "selected_hyperparameters_json": json.dumps(dict(config), sort_keys=True, separators=(",", ":")),
                })
                reduction_rows.append({
                    "endpoint_arm": endpoint.endpoint_arm,
                    "sampler": sampler,
                    "feature_block": FEATURE_BLOCK,
                    "model": model_name,
                    "execution_cell_id": cell_id,
                    "outer_fold": fold_index,
                    "outer_fold_id": fold_id,
                    "held_out_exact_ligand_group": held_out,
                    "input_feature_count": 30,
                    "retained_feature_count": len(retained),
                    "retained_features": ";".join(retained),
                    "variance_threshold": VARIANCE_THRESHOLD,
                    "absolute_spearman_threshold": CORRELATION_THRESHOLD,
                    "correlation_rule": "COMPLETE_LINKAGE_MEDOID",
                    "scaling": "STANDARD_SCALER",
                    "fit_scope": "OUTER_TRAIN_ONLY",
                })
            require(np.all(np.isfinite(prediction)), f"{cell_id}: incomplete OOF prediction")
            results.append(UniqueCellResult(
                endpoint.endpoint_arm,
                sampler,
                model_name,
                cell_id,
                prediction,
                metrics(panel.y, prediction, panel.groups),
                per_group_mae(panel.y, prediction, panel.groups),
            ))
    return results, selection_rows, reduction_rows


def shared_dummy(panel: Panel) -> tuple[UniqueCellResult, list[dict[str, Any]]]:
    prediction = np.full(len(panel.systems), np.nan, dtype=float)
    selection: list[dict[str, Any]] = []
    for fold_index, (fold_id, held_out, train, test) in enumerate(frozen_outer_folds(panel), start=1):
        estimator = DummyRegressor(strategy="median")
        estimator.fit(np.zeros((len(train), 1)), panel.y[train])
        prediction[test] = estimator.predict(np.zeros((len(test), 1)))
        selection.append({
            "endpoint_arm": "SHARED_ACROSS_ENDPOINTS",
            "sampler": "SHARED_ACROSS_SAMPLERS",
            "feature_block": "No_features",
            "model": "dummy_median",
            "execution_cell_id": "SharedDummyMedian",
            "outer_fold": fold_index,
            "outer_fold_id": fold_id,
            "held_out_exact_ligand_group": held_out,
            "inner_selection_metric": "NOT_APPLICABLE",
            "inner_selection_rule": "TRAINING_MEDIAN_ONLY",
            "inner_selected_score": "",
            "selected_hyperparameters_json": '{"strategy":"median"}',
        })
    require(np.all(np.isfinite(prediction)), "shared Dummy has incomplete OOF prediction")
    return (
        UniqueCellResult(
            "SHARED_ACROSS_ENDPOINTS",
            "SHARED_ACROSS_SAMPLERS",
            "dummy_median",
            "SharedDummyMedian",
            prediction,
            metrics(panel.y, prediction, panel.groups),
            per_group_mae(panel.y, prediction, panel.groups),
        ),
        selection,
    )


def display(value: Any) -> Any:
    if isinstance(value, float):
        return "" if not math.isfinite(value) else f"{value:.12g}"
    return value


def run_matrix(panel: Panel, scientific_status: str) -> dict[str, list[dict[str, Any]]]:
    dummy, dummy_selection = shared_dummy(panel)
    current_results, current_selection, current_reduction = endpoint_matrix(panel, panel.current)
    simple_results, simple_selection, simple_reduction = endpoint_matrix(panel, panel.simple)
    learned = current_results + simple_results
    result_index = {(item.endpoint_arm, item.sampler, item.model): item for item in learned}

    scoreboard: list[dict[str, Any]] = []
    oof: list[dict[str, Any]] = []
    paired_dummy: list[dict[str, Any]] = []
    seed = 0
    outer_by_system: dict[str, tuple[int, str, str, bool]] = {}
    for fold_index, (fold_id, held_out, train, test) in enumerate(frozen_outer_folds(panel), start=1):
        protein_seen = {index: bool(panel.targets[index] in set(panel.targets[train])) for index in test}
        for index in test:
            outer_by_system[panel.systems[index]] = (fold_index, fold_id, held_out, protein_seen[index])

    for endpoint in (panel.current.endpoint_arm, panel.simple.endpoint_arm):
        for sampler in EXPECTED_METHODS:
            for model_name in ("dummy_median", "ridge", "random_forest", "xgboost"):
                is_dummy = model_name == "dummy_median"
                result = dummy if is_dummy else result_index[(endpoint, sampler, model_name)]
                paired = paired_group_summary(result.group_mae, dummy.group_mae, RANDOM_STATE + seed)
                seed += 1
                row = {
                    "schema": SCHEMA,
                    "scientific_status": scientific_status,
                    "endpoint_arm": endpoint,
                    "sampler": sampler,
                    "feature_block": "No_features" if is_dummy else FEATURE_BLOCK,
                    "model": model_name,
                    "is_shared_dummy": str(is_dummy).lower(),
                    "execution_cell_id": result.cell_id,
                    "system_count": len(panel.systems),
                    "exact_ligand_group_count": len(set(panel.groups)),
                    **result.metrics,
                    "group_equal_mae_delta_vs_shared_dummy": paired["group_equal_mae_delta"],
                    "delta_vs_dummy_bootstrap_95_ci_lower": paired["bootstrap_95_ci_lower"],
                    "delta_vs_dummy_bootstrap_95_ci_upper": paired["bootstrap_95_ci_upper"],
                    "delta_vs_dummy_bootstrap_probability_below_zero": paired["bootstrap_probability_delta_below_zero"],
                    "improved_ligand_group_count_vs_dummy": paired["improved_group_count"],
                    "tied_ligand_group_count_vs_dummy": paired["tied_group_count"],
                    "total_ligand_group_count": paired["total_group_count"],
                    "leave_one_group_delta_min_vs_dummy": paired["leave_one_group_delta_min"],
                    "leave_one_group_delta_max_vs_dummy": paired["leave_one_group_delta_max"],
                    "leave_one_group_sign_reversal_vs_dummy": str(paired["leave_one_group_sign_reversal"]).lower(),
                }
                scoreboard.append({key: display(value) for key, value in row.items()})
                if not is_dummy:
                    paired_dummy.append({
                        "schema": SCHEMA,
                        "scientific_status": scientific_status,
                        "endpoint_arm": endpoint,
                        "sampler": sampler,
                        "feature_block": FEATURE_BLOCK,
                        "model": model_name,
                        "execution_cell_id": result.cell_id,
                        **{key: display(value) for key, value in paired.items()},
                    })

                for index, system in enumerate(panel.systems):
                    fold_index, fold_id, held_out, protein_seen = outer_by_system[system]
                    predicted = float(result.prediction[index])
                    error = predicted - float(panel.y[index])
                    oof.append({
                        "schema": SCHEMA,
                        "scientific_status": scientific_status,
                        "endpoint_arm": endpoint,
                        "sampler": sampler,
                        "feature_block": "No_features" if is_dummy else FEATURE_BLOCK,
                        "model": model_name,
                        "is_shared_dummy": str(is_dummy).lower(),
                        "execution_cell_id": result.cell_id,
                        "system_id": system,
                        "protein_target": panel.targets[index],
                        "label_tier": panel.label_tiers[index],
                        "exact_ligand_group": panel.groups[index],
                        "experimental_pKoff": f"{panel.y[index]:.12g}",
                        "predicted_pKoff": f"{predicted:.12g}",
                        "error_pred_minus_exp": f"{error:.12g}",
                        "absolute_error": f"{abs(error):.12g}",
                        "squared_error": f"{error**2:.12g}",
                        "outer_fold": fold_index,
                        "outer_fold_id": fold_id,
                        "held_out_exact_ligand_group": held_out,
                        "exact_ligand_seen_in_training": "false",
                        "protein_target_seen_in_training": str(protein_seen).lower(),
                    })

    paired_endpoint: list[dict[str, Any]] = []
    for sampler in EXPECTED_METHODS:
        for model_name in ("dummy_median", "ridge", "random_forest", "xgboost"):
            current = dummy if model_name == "dummy_median" else result_index[(panel.current.endpoint_arm, sampler, model_name)]
            simple = dummy if model_name == "dummy_median" else result_index[(panel.simple.endpoint_arm, sampler, model_name)]
            paired = paired_group_summary(simple.group_mae, current.group_mae, RANDOM_STATE + seed)
            seed += 1
            system_delta = np.abs(simple.prediction - panel.y) - np.abs(current.prediction - panel.y)
            paired_endpoint.append({
                "schema": SCHEMA,
                "scientific_status": scientific_status,
                "sampler": sampler,
                "feature_block": "No_features" if model_name == "dummy_median" else FEATURE_BLOCK,
                "model": model_name,
                "reference_endpoint": panel.current.endpoint_arm,
                "candidate_endpoint": panel.simple.endpoint_arm,
                "delta_definition": "candidate_simple_minus_reference_current; negative favors simple",
                "system_equal_mae_delta": display(float(np.mean(system_delta))),
                **{key: display(value) for key, value in paired.items()},
            })

    require(len(scoreboard) == 24, f"scoreboard row count drifted: {len(scoreboard)}")
    require(len(oof) == 24 * len(panel.systems), f"OOF row count drifted: {len(oof)}")
    require(len(paired_dummy) == 18, f"paired-Dummy row count drifted: {len(paired_dummy)}")
    require(len(paired_endpoint) == 12, f"paired-endpoint row count drifted: {len(paired_endpoint)}")
    return {
        "scoreboard": scoreboard,
        "oof": oof,
        "selection": dummy_selection + current_selection + simple_selection,
        "reduction": current_reduction + simple_reduction,
        "paired_dummy": paired_dummy,
        "paired_endpoint": paired_endpoint,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-v21-current30", type=Path, required=True)
    parser.add_argument("--simple-current30", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--current-endpoint-id", default=CURRENT_ENDPOINT_DEFAULT)
    parser.add_argument("--simple-endpoint-id", default=SIMPLE_ENDPOINT_DEFAULT)
    parser.add_argument(
        "--synthetic-smoke-only",
        action="store_true",
        help="Mark every artifact NON_SCIENTIFIC_SYNTHETIC_SMOKE_ONLY.",
    )
    args = parser.parse_args(argv)
    require(not args.output_dir.exists(), f"refusing to overwrite output directory: {args.output_dir}")

    current = load_endpoint_table(args.current_v21_current30, args.current_endpoint_id)
    simple = load_endpoint_table(args.simple_current30, args.simple_endpoint_id)
    validate_endpoint_parity(current, simple)
    panel = load_panel(args.panel, args.folds, current, simple)
    status = "NON_SCIENTIFIC_SYNTHETIC_SMOKE_ONLY" if args.synthetic_smoke_only else "SCIENTIFIC_MATCHED_DUAL_ENDPOINT_OOF"
    outputs = run_matrix(panel, status)

    args.output_dir.mkdir(parents=True)
    try:
        write_tsv(args.output_dir / "MODEL_COMBINATION_SCOREBOARD_v1.tsv", outputs["scoreboard"])
        write_tsv(args.output_dir / "OOF_PREDICTIONS_ALL_COMBINATIONS_v1.tsv", outputs["oof"])
        write_tsv(args.output_dir / "INNER_MODEL_SELECTION_v1.tsv", outputs["selection"])
        write_tsv(args.output_dir / "FOLD_LOCAL_FEATURE_REDUCTION_v1.tsv", outputs["reduction"])
        write_tsv(args.output_dir / "PAIRED_DELTAS_VS_DUMMY_v1.tsv", outputs["paired_dummy"])
        write_tsv(args.output_dir / "PAIRED_ENDPOINT_COMPARISON_v1.tsv", outputs["paired_endpoint"])
        receipt = {
            "schema": SCHEMA,
            "scientific_status": status,
            "inputs": {
                "current_v21_current30": str(args.current_v21_current30.resolve()),
                "simple_current30": str(args.simple_current30.resolve()),
                "panel": str(args.panel.resolve()),
                "folds": str(args.folds.resolve()),
            },
            "endpoint_arms": [current.endpoint_arm, simple.endpoint_arm],
            "samplers": list(EXPECTED_METHODS),
            "feature_block": FEATURE_BLOCK,
            "model_families": ["dummy_median", "ridge", "random_forest", "xgboost"],
            "unique_execution_cells": 19,
            "displayed_scoreboard_rows": 24,
            "system_count": len(panel.systems),
            "exact_ligand_group_count": len(set(panel.groups)),
            "outer_split": "FROZEN_EXACT_LIGAND_LOGO",
            "inner_split": "EXACT_LIGAND_LOGO_WITHIN_OUTER_TRAINING_ONLY",
            "feature_engineering": {
                "constant_variance_threshold": VARIANCE_THRESHOLD,
                "absolute_spearman_threshold": CORRELATION_THRESHOLD,
                "correlation_rule": "COMPLETE_LINKAGE_MEDOID",
                "scaling": "StandardScaler fitted inside each training split",
            },
            "hyperparameter_selection": "minimum inner exact-ligand group-equal MAE; canonical JSON breaks ties",
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "source_reuse": {
                "fold_local_reducer": "scripts/koff_ml/fold_local_spearman_reducer.py",
                "xgboost_grid": "scripts/koff_ml/run_feature_engineered_baselines_v1.py",
            },
            "claim_ceiling": (
                "Synthetic smoke artifacts are code checks only. Scientific artifacts are matched nested-OOF "
                "engineering evidence for experimental pKoff and do not estimate physical koff."
            ),
        }
        (args.output_dir / "execution_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(args.output_dir, ignore_errors=True)
        raise
    print(json.dumps({"status": "PASS", "output_dir": str(args.output_dir), "scientific_status": status}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
