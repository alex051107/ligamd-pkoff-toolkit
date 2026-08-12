#!/usr/bin/env python3
"""Summarize the direct Dynamic10 increment from a frozen OOF table.

The input table contains out-of-fold predictions from four already-fitted
profiles.  This script does not fit a model, tune a parameter, or look at raw
trajectories. It asks one narrow question: when the model family is held
fixed, did adding P512 Dynamic10 reduce group-equal absolute error relative to
Static20?

The output is aggregate-only. It intentionally omits system identifiers,
ligand identifiers, experimental labels, and individual predictions so that a
public scorecard can document the comparison without redistributing a
campaign-specific assay table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PAIRS = (
    ("static20_ridge", "combined30_p512_ridge"),
    ("static20_random_forest", "combined30_p512_random_forest"),
)


def _bootstrap_interval(deltas: np.ndarray, *, repeats: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n_groups = len(deltas)
    draw_indices = rng.integers(0, n_groups, size=(repeats, n_groups))
    draws = deltas[draw_indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, (0.025, 0.975)))


def summarize(oof: pd.DataFrame, *, repeats: int, seed: int) -> pd.DataFrame:
    required = {"profile_id", "exact_ligand_group", "absolute_error"}
    missing = sorted(required - set(oof.columns))
    if missing:
        raise ValueError(f"OOF table is missing required columns: {missing}")
    rows: list[dict[str, object]] = []
    for static_id, combined_id in PAIRS:
        subset = oof.loc[oof["profile_id"].isin((static_id, combined_id))].copy()
        grouped = (
            subset.groupby(["profile_id", "exact_ligand_group"], sort=True)["absolute_error"]
            .mean()
            .unstack("profile_id")
        )
        if set((static_id, combined_id)) - set(grouped.columns):
            raise ValueError(f"OOF table lacks a complete pair for {static_id} and {combined_id}")
        paired = grouped[[static_id, combined_id]].dropna().sort_index()
        if len(paired) < 2:
            raise ValueError(f"need at least two paired ligand groups for {combined_id}")
        deltas = (paired[combined_id] - paired[static_id]).to_numpy(dtype=float)
        lower, upper = _bootstrap_interval(deltas, repeats=repeats, seed=seed)
        rows.append(
            {
                "static_profile_id": static_id,
                "combined_profile_id": combined_id,
                "comparison": "Combined30_P512_minus_Static20; negative favors Combined30",
                "metric": "exact_ligand_group_equal_MAE",
                "independent_exact_ligand_group_count": int(len(paired)),
                "static_group_equal_mae": float(paired[static_id].mean()),
                "combined_group_equal_mae": float(paired[combined_id].mean()),
                "combined_minus_static_mae": float(deltas.mean()),
                "combined_improved_group_count": int(np.sum(deltas < 0.0)),
                "combined_worsened_group_count": int(np.sum(deltas > 0.0)),
                "tied_group_count": int(np.sum(deltas == 0.0)),
                "bootstrap_repeats": int(repeats),
                "bootstrap_seed": int(seed),
                "paired_bootstrap_95_ci_lower": lower,
                "paired_bootstrap_95_ci_upper": upper,
                "recommendation_status": "NOT_SELECTED_CI_CROSSES_ZERO",
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, required=True, help="private or otherwise authorized OOF table")
    parser.add_argument("--output", type=Path, required=True, help="aggregate-only TSV to write")
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    if args.bootstrap_repeats < 100:
        raise SystemExit("--bootstrap-repeats must be at least 100")
    output = summarize(
        pd.read_csv(args.oof, sep="\t"),
        repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, sep="\t", index=False, lineterminator="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
