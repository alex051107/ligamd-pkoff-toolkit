"""Select 512 real trajectory frames along an already-defined LiGaMD episode.

This module deliberately has one job.  Given a dense table of frame-level
geometry measurements and an endpoint onset that was established elsewhere,
it returns 512 distinct source-frame indices between production frame 0 and
that onset.  It does not decide whether a trajectory dissociated, read an
experimental label, inspect a system name, or choose a regression model.

P512 represents a path by three physical blocks.  The separation block records
motion away from the original pocket and from the protein.  The contact block
records loss or rearrangement of frozen contacts.  The pose block records
ligand and pocket geometry.  After a small past-only smoothing and a
within-episode scale normalisation, P512 accumulates the size of the change
between adjacent saved frames.  It then selects equal fractions of that
accumulated change.  Every returned index points to an existing saved frame;
no coordinate is interpolated.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


BASE_COLUMNS = (
    "frame",
    "pocket_geometric_com_distance_A",
    "native_contact_fraction",
    "contact_formations",
    "contact_breaks",
    "pose__aligned_ligand_rmsd_A",
    "pose__aligned_centroid_displacement_A",
    "pose__ligand_internal_conformation_rmsd_A",
    "pose__pocket_alignment_rmsd_A",
    "global__protein_min_heavy_distance_A",
    "global__protein_contact_residue_count",
)


class P512SamplerError(RuntimeError):
    """Raised when a dense trace cannot support the frozen P512 operation."""


@dataclass(frozen=True)
class P512Trace:
    """Validated dense measurements for one production replica.

    ``frames1`` stores the original one-based saved-frame identifiers.  The
    numeric arrays and contact-bit matrix have the same first dimension and
    never contain interpolated frames.  ``contact_columns`` documents the
    frozen-pocket contact vocabulary represented by ``contacts``.
    """

    frames1: np.ndarray
    numeric: Mapping[str, np.ndarray]
    contacts: np.ndarray
    contact_columns: tuple[str, ...]

    @property
    def n_frames(self) -> int:
        """Return the number of dense, consecutive saved frames."""

        return len(self.frames1)


def load_p512_trace(path: Path, expected_frames: int) -> P512Trace:
    """Load one dense reference table and enforce the public P512 schema.

    The two historical contact-formation columns remain required in v1 even
    though P512 itself does not use them.  Keeping the original schema avoids a
    silent change in what the first public release accepts.  A later relaxed
    schema would need a new versioned contract and its own comparison.
    """

    with path.open(newline="", encoding="utf-8") as handle:
        try:
            header = next(csv.reader(handle))
        except StopIteration as exc:
            raise P512SamplerError("reference trace is empty") from exc
    contact_columns = tuple(name for name in header if name.startswith("contact__"))
    missing = sorted(set(BASE_COLUMNS) - set(header))
    if missing or not contact_columns:
        raise P512SamplerError(
            "reference trace schema is incomplete: "
            f"missing={missing}, contacts={len(contact_columns)}"
        )
    table = pd.read_csv(path, usecols=[*BASE_COLUMNS, *contact_columns])
    if len(table) != expected_frames or table.isna().any().any():
        raise P512SamplerError(
            "reference trace rows/missing values fail: "
            f"rows={len(table)} expected={expected_frames}"
        )
    frames1 = table["frame"].to_numpy(dtype=np.int64)
    expected = np.arange(1, expected_frames + 1, dtype=np.int64)
    if not np.array_equal(frames1, expected):
        raise P512SamplerError("reference trace frame column is not consecutive 1..N")
    numeric = {
        name: table[name].to_numpy(dtype=float)
        for name in BASE_COLUMNS
        if name != "frame"
    }
    contact_matrix = table[list(contact_columns)].to_numpy(dtype=np.float32)
    if not np.all(np.isin(contact_matrix, (0.0, 1.0))):
        raise P512SamplerError("contact columns are not binary")
    return P512Trace(frames1, numeric, contact_matrix, contact_columns)


def _causal_trailing_mean(matrix: np.ndarray, window: int) -> np.ndarray:
    """Smooth with current and past observations only, never future frames."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if window < 1:
        raise ValueError("causal trailing window must be positive")
    cumulative = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)))
    starts = np.maximum(0, np.arange(len(values)) + 1 - window)
    stops = np.arange(len(values)) + 1
    counts = (stops - starts).astype(float)[:, None]
    return (cumulative[stops] - cumulative[starts]) / counts


def _prefix_difference_scale(smoothed: np.ndarray, prefix_frames: int) -> np.ndarray:
    """Scale channels by early observed change, with deterministic fallbacks."""

    prefix = smoothed[: min(prefix_frames, len(smoothed))]
    differences = np.diff(prefix, axis=0)
    if len(differences) == 0:
        return np.ones(smoothed.shape[1], dtype=float)
    median = np.median(differences, axis=0)
    mad = 1.4826 * np.median(np.abs(differences - median), axis=0)
    std = np.std(differences, axis=0)
    scale = np.where(mad > 1e-8, mad, np.where(std > 1e-8, std, 1.0))
    return np.asarray(scale, dtype=float)


def _sampler_blocks(
    trace: P512Trace, endpoint_onset_index0: int, protocol: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    """Build equally normalised separation, contact, and pose path blocks."""

    stop = int(endpoint_onset_index0) + 1
    numeric = trace.numeric
    separation = np.column_stack(
        (
            numeric["pocket_geometric_com_distance_A"][:stop],
            numeric["pose__aligned_centroid_displacement_A"][:stop],
            numeric["global__protein_min_heavy_distance_A"][:stop],
        )
    )
    contact = np.column_stack(
        (
            numeric["native_contact_fraction"][:stop],
            numeric["global__protein_contact_residue_count"][:stop],
            trace.contacts[:stop],
        )
    )
    pose = np.column_stack(
        (
            numeric["pose__aligned_ligand_rmsd_A"][:stop],
            numeric["pose__ligand_internal_conformation_rmsd_A"][:stop],
            numeric["pose__pocket_alignment_rmsd_A"][:stop],
        )
    )
    shared = protocol["samplers"]["shared"]
    window = int(shared["causal_trailing_mean_frames"])
    prefix = int(shared["scale_prefix_frames"])
    output: dict[str, np.ndarray] = {}
    for name, matrix in (("separation", separation), ("contact", contact), ("pose", pose)):
        smooth = _causal_trailing_mean(matrix, window)
        scale = _prefix_difference_scale(smooth, prefix)
        output[name] = (smooth / scale[None, :]) / math.sqrt(matrix.shape[1])
    return output


def _maximum_gap_fill(selected: set[int], endpoint_onset_index0: int, budget: int) -> set[int]:
    """Fill duplicate path quantiles by splitting the earliest largest time gap."""

    chosen = set(map(int, selected))
    chosen.update((0, int(endpoint_onset_index0)))
    while len(chosen) < budget:
        ordered = sorted(chosen)
        candidates: list[tuple[int, int]] = []
        for left, right in zip(ordered, ordered[1:]):
            if right - left > 1:
                candidates.append((right - left, left))
        if not candidates:
            raise P512SamplerError("not enough unique real frames for the requested budget")
        _, left = max(candidates, key=lambda item: (item[0], -item[1]))
        right = next(value for value in ordered if value > left)
        chosen.add((left + right) // 2)
    return chosen


def p512_path_arclength_indices(
    trace: P512Trace, endpoint_onset_index0: int, protocol: Mapping[str, Any]
) -> tuple[np.ndarray, dict[int, str], np.ndarray]:
    """Return exactly the contract budget of source-frame indices for one episode.

    The path interval includes frame 0 and the first persistence-confirmed
    endpoint onset.  It excludes the later confirmation tail and any later
    solvent diffusion.  The return values are the selected zero-based indices,
    a transparent selection-role map, and cumulative path arclength.
    """

    try:
        budget = int(protocol["samplers"]["budget"])
    except (KeyError, TypeError, ValueError) as exc:
        raise P512SamplerError("P512 contract has no valid frame budget") from exc
    if endpoint_onset_index0 < 0 or endpoint_onset_index0 >= trace.n_frames:
        raise P512SamplerError("endpoint onset must be a valid dense-trace index")
    if endpoint_onset_index0 + 1 < budget:
        raise P512SamplerError("P512 requires at least 512 distinct real frames")
    blocks = _sampler_blocks(trace, endpoint_onset_index0, protocol)
    step_squared = np.zeros(endpoint_onset_index0, dtype=float)
    for matrix in blocks.values():
        delta = np.diff(matrix, axis=0)
        step_squared += np.sum(delta * delta, axis=1)
    step = np.sqrt(step_squared)
    cumulative = np.r_[0.0, np.cumsum(step)]
    total = float(cumulative[-1])
    if not np.isfinite(total):
        raise P512SamplerError("P512 path arclength is not finite")
    selected: set[int] = {0, int(endpoint_onset_index0)}
    roles: dict[int, str] = {0: "forced_endpoint", int(endpoint_onset_index0): "forced_endpoint"}
    if total > 0:
        # Endpoints are reserved explicitly. Quantising only interior targets
        # avoids a terminal path-length plateau creating one extra frame.
        targets = np.linspace(0.0, total, budget)[1:-1]
        right = np.searchsorted(cumulative, targets, side="left")
        right = np.clip(right, 0, len(cumulative) - 1)
        left = np.maximum(0, right - 1)
        choose_left = np.abs(cumulative[left] - targets) <= np.abs(cumulative[right] - targets)
        nearest = np.where(choose_left, left, right)
        for index in map(int, nearest):
            selected.add(index)
            roles.setdefault(index, "path_arclength_quantile")
    before_fill = set(selected)
    selected = _maximum_gap_fill(selected, endpoint_onset_index0, budget)
    for index in selected - before_fill:
        roles[index] = "fallback_time_fill"
    roles[0] = roles[endpoint_onset_index0] = "forced_endpoint"
    output = np.asarray(sorted(selected), dtype=np.int64)
    if len(output) != budget or output[0] != 0 or output[-1] != endpoint_onset_index0:
        raise P512SamplerError("P512 lost an episode endpoint or the exact frame budget")
    return output, roles, cumulative
