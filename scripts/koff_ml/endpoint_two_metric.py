"""The frozen two-metric complete-exit rule used by the public toolkit.

The rule says that a saved frame is *geometrically clear of the original
complex* only when two separate observations agree:

1. after aligning the frozen pocket to the canonical bound structure, the
   ligand centroid has moved at least 15 Å from its bound position; and
2. the nearest ligand-heavy-atom / standard-protein-heavy-atom pair is more
   than 10 Å apart.

Neither observation is an interaction energy or a physical ``koff``.  The
first prevents a ligand from being called "exited" merely because it moved to
another part of the protein; the second prevents a ligand that is far from the
original pocket but still attached to the protein surface from being called
fully clear.  Both must hold for 100 *consecutive saved frames*.  The first
frame of that run is the episode endpoint; the later 99 frames only confirm it
and are excluded from the model input.

This module reads no labels and has no model or sampler preference.  It is the
single shared implementation used by the analysis toolkit and its tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


ALLOWED_DISPLACEMENT_FIELDS = frozenset(
    {
        "pose__aligned_centroid_displacement_A",
        "pose__aligned_ligand_rmsd_A",
    }
)
ENDPOINT_CONTRACT_SCHEMA = "ligamd_endpoint_contract_v2.0"


class TwoMetricEndpointError(ValueError):
    """Raised when a two-metric endpoint input is incomplete or ambiguous."""


def endpoint_spec_from_contract(contract: Mapping[str, Any]) -> "TwoMetricEndpointSpec":
    """Build the executable endpoint rule from its machine-readable contract.

    Keeping this parser next to the numerical predicate prevents a quiet split
    between a JSON file that documents one rule and Python code that evaluates
    another. The public v2 contract is intentionally narrow: two named frame
    conditions joined by ``AND`` plus one saved-frame persistence length.
    """

    if contract.get("schema_version") != ENDPOINT_CONTRACT_SCHEMA:
        raise TwoMetricEndpointError(
            f"endpoint contract schema must be {ENDPOINT_CONTRACT_SCHEMA}"
        )
    raw_conditions = contract.get("frame_conditions_all_required")
    if not isinstance(raw_conditions, list) or len(raw_conditions) != 2:
        raise TwoMetricEndpointError("endpoint contract must declare exactly two required frame conditions")
    conditions: dict[str, Mapping[str, Any]] = {}
    for item in raw_conditions:
        if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
            raise TwoMetricEndpointError("endpoint frame conditions must be named objects")
        conditions[str(item["name"])] = item
    displacement = conditions.get("pocket_aligned_ligand_centroid_displacement")
    clearance = conditions.get("whole_protein_heavy_atom_clearance")
    if displacement is None or clearance is None:
        raise TwoMetricEndpointError("endpoint contract is missing a required named geometry condition")
    if displacement.get("operator") != ">=" or clearance.get("operator") != ">":
        raise TwoMetricEndpointError("endpoint-v2 operators must be displacement >= threshold and clearance > threshold")
    field = displacement.get("dense_trace_column")
    if not isinstance(field, str):
        raise TwoMetricEndpointError("endpoint displacement condition needs dense_trace_column")
    persistence = contract.get("persistence", {}).get("consecutive_saved_frames")
    if isinstance(persistence, bool) or not isinstance(persistence, int):
        raise TwoMetricEndpointError("endpoint persistence must be an integer number of saved frames")
    try:
        spec = TwoMetricEndpointSpec(
            name=str(contract.get("contract_id", "endpoint_v2_two_metric_geometric_clearance")),
            displacement_field=field,
            displacement_min_A=float(displacement["threshold_A"]),
            whole_protein_min_distance_strictly_greater_A=float(clearance["threshold_A"]),
            persistence_frames=persistence,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TwoMetricEndpointError("endpoint contract has invalid numeric thresholds") from exc
    _validate_spec(spec)
    return spec


@dataclass(frozen=True)
class TwoMetricEndpointSpec:
    """Parameters for the endpoint-v2 geometric-clearance rule."""

    name: str = "endpoint_v2_two_metric_geometric_clearance"
    displacement_field: str = "pose__aligned_centroid_displacement_A"
    displacement_min_A: float = 15.0
    whole_protein_min_distance_strictly_greater_A: float = 10.0
    persistence_frames: int = 100


@dataclass(frozen=True)
class TwoMetricEndpoint:
    """An auditable answer for one dense trajectory trace."""

    status: str
    first_instantaneous_exit_index0: int | None
    confirmed_exit_onset_index0: int | None
    confirmation_index0: int | None
    true_frame_count: int
    run_count: int
    longest_run_frames: int
    terminal_run_frames: int
    qualifying_run_count: int


def _validate_spec(spec: TwoMetricEndpointSpec) -> None:
    if not isinstance(spec.name, str) or not spec.name.strip():
        raise TwoMetricEndpointError("endpoint name must be a non-empty string")
    if spec.displacement_field not in ALLOWED_DISPLACEMENT_FIELDS:
        allowed = ", ".join(sorted(ALLOWED_DISPLACEMENT_FIELDS))
        raise TwoMetricEndpointError(f"displacement_field must be one of: {allowed}")
    if (
        isinstance(spec.persistence_frames, (bool, np.bool_))
        or not isinstance(spec.persistence_frames, (int, np.integer))
        or int(spec.persistence_frames) <= 0
    ):
        raise TwoMetricEndpointError("persistence_frames must be a positive integer")
    for name in (
        "displacement_min_A",
        "whole_protein_min_distance_strictly_greater_A",
    ):
        value = float(getattr(spec, name))
        if not np.isfinite(value) or value <= 0:
            raise TwoMetricEndpointError(f"{name} must be finite and positive")


def _required_channel(
    numeric: Mapping[str, np.ndarray], name: str, *, n_frames: int | None = None
) -> np.ndarray:
    if name not in numeric:
        raise TwoMetricEndpointError(f"missing required numeric channel: {name}")
    values = np.asarray(numeric[name], dtype=float)
    if values.ndim != 1 or (n_frames is not None and len(values) != n_frames):
        raise TwoMetricEndpointError(f"numeric channel has invalid shape: {name}")
    if len(values) < 1 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise TwoMetricEndpointError(f"numeric channel must be finite and non-negative: {name}")
    return values


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive start/stop indices for every uninterrupted True run."""

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise TwoMetricEndpointError("exit mask must be one-dimensional")
    if len(values) == 0:
        return []
    padded = np.r_[False, values, False].astype(np.int8)
    changes = np.diff(padded)
    return [
        (int(start), int(stop))
        for start, stop in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1) - 1)
    ]


def _validate_frames1(frames1: np.ndarray, *, n_frames: int) -> np.ndarray:
    """Require dense saved-frame identity, so 100 means 100 adjacent observations."""

    raw = np.asarray(frames1)
    if raw.ndim != 1 or len(raw) != n_frames:
        raise TwoMetricEndpointError("frames1 must be a one-dimensional N-frame array")
    try:
        integer = raw.astype(np.int64)
    except (TypeError, ValueError) as exc:
        raise TwoMetricEndpointError("frames1 must contain integer frame identifiers") from exc
    expected = np.arange(1, n_frames + 1, dtype=np.int64)
    if not np.array_equal(integer, expected):
        raise TwoMetricEndpointError("frames1 must be the exact saved-frame sequence 1..N")
    return integer


def two_metric_exit_mask(
    numeric: Mapping[str, np.ndarray], spec: TwoMetricEndpointSpec
) -> np.ndarray:
    """Evaluate the two per-frame geometry conditions without persistence."""

    _validate_spec(spec)
    displacement = _required_channel(numeric, spec.displacement_field)
    clearance = _required_channel(
        numeric,
        "global__protein_min_heavy_distance_A",
        n_frames=len(displacement),
    )
    return (displacement >= float(spec.displacement_min_A)) & (
        clearance > float(spec.whole_protein_min_distance_strictly_greater_A)
    )


def detect_two_metric_endpoint(
    numeric: Mapping[str, np.ndarray], spec: TwoMetricEndpointSpec, *, frames1: np.ndarray
) -> TwoMetricEndpoint:
    """Find the first run that lasts for the frozen persistence duration.

    The endpoint is its *onset*, not the end of the run.  This preserves the
    whole bound-to-exit path and excludes subsequent solvent diffusion while
    still using the tail as evidence that the observed clearance was stable.
    """

    mask = two_metric_exit_mask(numeric, spec)
    _validate_frames1(frames1, n_frames=len(mask))
    runs = true_runs(mask)
    lengths = [stop - start + 1 for start, stop in runs]
    qualifying = [run for run in runs if run[1] - run[0] + 1 >= int(spec.persistence_frames)]
    if not runs:
        status, onset, confirmation = "TWO_METRIC_EXIT_NOT_OBSERVED", None, None
    elif not qualifying:
        status, onset, confirmation = "TWO_METRIC_EXIT_OBSERVED_BUT_NOT_PERSISTENT", None, None
    else:
        onset = int(qualifying[0][0])
        confirmation = onset + int(spec.persistence_frames) - 1
        status = "FIRST_PERSISTENCE_CONFIRMED_TWO_METRIC_EXIT_ONSET"
    return TwoMetricEndpoint(
        status=status,
        first_instantaneous_exit_index0=(int(runs[0][0]) if runs else None),
        confirmed_exit_onset_index0=onset,
        confirmation_index0=confirmation,
        true_frame_count=int(np.sum(mask)),
        run_count=len(runs),
        longest_run_frames=int(max(lengths, default=0)),
        terminal_run_frames=int(lengths[-1] if runs and runs[-1][1] == len(mask) - 1 else 0),
        qualifying_run_count=len(qualifying),
    )
