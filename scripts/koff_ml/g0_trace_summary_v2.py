#!/usr/bin/env python3
"""G0-v2 low-cost full-trace control for the LiGaMD pKoff screen.

G0-v2 contains 34 values per successful Production replica:

* 24 continuous-channel means on the shared NormalizedProgressGrid-v1;
* six real-endpoint minus start changes; and
* four state occupancies on the same shared grid.

Every normalized-support summary uses the frozen 512-point half-open
midpoint grid and previous-observation/left-hold reconstruction.  Therefore
the terminal observation at progress 1 has zero support and is used only by
the explicitly named endpoint-change features.  No absolute time, episode
length, FPT, sigma, boost, event count, outcome, or source frame identity is
returned.

This is a new semantic/schema version.  It deliberately uses ``g0v2__``
public column names rather than silently changing the meaning of G0-v1.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType

import numpy as np

from scripts.koff_ml.normalized_progress_grid_v1 import (
    GRID_SIZE,
    NORMALIZED_PROGRESS_GRID_SEMANTIC_ID,
    interval_means_on_grid,
    left_hold_on_normalized_grid,
)


SCHEMA_VERSION = "ligamd_G0_full_trace_summary_34D_v2.0"
SEMANTIC_VERSION = "LiGaMD-G0-v2:NormalizedProgressGrid-v1:shared34"
EXPECTED_REPLICAS_PER_SYSTEM = 3
NORMALIZED_PROGRESS_GRID_SIZE = GRID_SIZE
G0_GRID_SEMANTIC_ID = NORMALIZED_PROGRESS_GRID_SEMANTIC_ID

CHANNEL_SOURCE_FIELDS = MappingProxyType(
    {
        "pocket_com_distance_A": "pocket_geometric_com_distance_A",
        "native_contact_fraction": "native_contact_fraction",
        "ligand_pose_rmsd_A": "pose__aligned_ligand_rmsd_A",
        "ligand_centroid_displacement_A": "pose__aligned_centroid_displacement_A",
        "global_min_heavy_distance_A": "global__protein_min_heavy_distance_A",
        "global_contact_fraction": "global__protein_contact_residue_fraction",
    }
)
CONTINUOUS_CHANNELS = tuple(CHANNEL_SOURCE_FIELDS)

PROGRESS_INTERVALS = (
    ("000_050", 0.00, 0.50),
    ("050_080", 0.50, 0.80),
    ("080_095", 0.80, 0.95),
    ("095_100", 0.95, 1.00),
)
STATE_ORDER = ("A", "I", "P_ONLY", "B")
REQUIRED_AUXILIARY_CHANNELS = ("global__protein_contact_residue_count",)

FROZEN_STATE_CRITERIA = MappingProxyType(
    {
        "bound_max_pocket_com_A": 6.0,
        "bound_min_native_contact_fraction": 0.50,
        "pocket_exit_min_pocket_com_A": 15.0,
        "pocket_exit_max_native_contact_fraction": 0.05,
        "pocket_exit_max_initial_contacts": 0.0,
        "bulk_unbound_max_global_contacts": 0.0,
        "bulk_unbound_min_global_distance_A": 6.0,
    }
)

FORBIDDEN_PRIMARY_FEATURE_TOKENS = (
    "pkoff",
    "label",
    "outcome",
    "sigma",
    "boost",
    "fpt",
    "duration",
    "elapsed",
    "frame",
    "onset",
    "confirmation",
    "event_count",
    "replica_count",
    "readiness",
)


def _newline_digest(values: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def state_criteria_sha256(criteria: Mapping[str, float]) -> str:
    """Return the canonical SHA-256 for an exact state-criteria mapping."""

    if set(criteria) != set(FROZEN_STATE_CRITERIA):
        raise ValueError("state criteria keys differ from the frozen v2 vocabulary")
    canonical: dict[str, float] = {}
    for key in sorted(FROZEN_STATE_CRITERIA):
        value = criteria[key]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"state criterion {key} must be a finite real number")
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(f"state criterion {key} must be finite")
        canonical[key] = numeric
    payload = json.dumps(
        canonical,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256((payload + "\n").encode("utf-8")).hexdigest()


STATE_CRITERIA_SHA256 = state_criteria_sha256(FROZEN_STATE_CRITERIA)


def _validate_frozen_state_criteria(criteria: Mapping[str, float]) -> None:
    if state_criteria_sha256(criteria) != STATE_CRITERIA_SHA256:
        raise ValueError(
            "state criteria values drift from the frozen G0-v2/G50-v2 contract"
        )


def _feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for channel in CONTINUOUS_CHANNELS:
        names.extend(
            f"g0v2__{channel}__progress_{label}__mean"
            for label, _, _ in PROGRESS_INTERVALS
        )
    names.extend(
        f"g0v2__{channel}__end_minus_start" for channel in CONTINUOUS_CHANNELS
    )
    names.extend(
        f"g0v2__state_progress_occupancy__{state}" for state in STATE_ORDER
    )
    return tuple(names)


def _shared34_semantic_ids() -> tuple[str, ...]:
    semantic_ids: list[str] = []
    for channel in CONTINUOUS_CHANNELS:
        semantic_ids.extend(
            "normalized_support::continuous::"
            f"{channel}::progress_{label}::mean"
            for label, _, _ in PROGRESS_INTERVALS
        )
    semantic_ids.extend(
        f"endpoint::continuous::{channel}::end_minus_start"
        for channel in CONTINUOUS_CHANNELS
    )
    semantic_ids.extend(
        f"normalized_support::state::{state}::occupancy" for state in STATE_ORDER
    )
    return tuple(semantic_ids)


G0_FEATURE_NAMES = _feature_names()
G0_SHARED34_SEMANTIC_IDS = _shared34_semantic_ids()
G0_FEATURE_SCHEMA_SHA256 = _newline_digest(G0_FEATURE_NAMES)
G0_SHARED34_SEMANTIC_SHA256 = _newline_digest(G0_SHARED34_SEMANTIC_IDS)
G0_SEMANTIC_ID_TO_PUBLIC_NAME = MappingProxyType(
    dict(zip(G0_SHARED34_SEMANTIC_IDS, G0_FEATURE_NAMES, strict=True))
)
MATCHED_NOISE_G0_BUDGET = "MATCHED_G0V2_SHARED34"

if len(G0_FEATURE_NAMES) != 34:  # pragma: no cover - import-time invariant
    raise RuntimeError("G0-v2 schema must contain exactly 34 primary features")
if len(set(G0_SHARED34_SEMANTIC_IDS)) != 34:  # pragma: no cover
    raise RuntimeError("G0-v2 shared semantic IDs must be unique")
if (
    G0_SHARED34_SEMANTIC_SHA256
    != "96c8f8a7e39f00f30896c97acc00158860f99051ce26ff09a76a9c98968b030d"
):  # pragma: no cover
    raise RuntimeError("G0-v2 shared34 semantic-ID contract drifted")
for _feature_name in G0_FEATURE_NAMES:  # pragma: no cover - import invariant
    lowered = _feature_name.lower()
    if any(token in lowered for token in FORBIDDEN_PRIMARY_FEATURE_TOKENS):
        raise RuntimeError(
            f"G0-v2 primary feature name contains a forbidden token: {_feature_name}"
        )


@dataclass(frozen=True)
class G0V2ReplicaEmbedding:
    """One complete successful-dissociation trace represented as 34 values."""

    values: np.ndarray
    feature_names: tuple[str, ...] = G0_FEATURE_NAMES
    semantic_ids: tuple[str, ...] = G0_SHARED34_SEMANTIC_IDS

    def as_mapping(self) -> dict[str, float]:
        return {
            name: float(value)
            for name, value in zip(self.feature_names, self.values, strict=True)
        }


@dataclass(frozen=True)
class G0V2ThreeReplicaAggregation:
    """Primary and sensitivity views for exactly three frozen replicas."""

    replica_ids: tuple[str, ...]
    primary_feature_names: tuple[str, ...]
    primary_semantic_ids: tuple[str, ...]
    primary_mean: np.ndarray
    sensitivity_feature_names: tuple[str, ...]
    sensitivity_mean_and_population_sd: np.ndarray


def _validate_numeric_inputs(
    numeric: Mapping[str, Sequence[float] | np.ndarray],
    pocket_contacts: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    required = (*CHANNEL_SOURCE_FIELDS.values(), *REQUIRED_AUXILIARY_CHANNELS)
    missing = [name for name in required if name not in numeric]
    if missing:
        raise ValueError(f"G0-v2 trace lacks required channels: {missing}")

    arrays: dict[str, np.ndarray] = {}
    length: int | None = None
    for name in required:
        values = np.asarray(numeric[name], dtype=np.float64)
        if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
            raise ValueError(
                f"G0-v2 channel {name} must be a finite 1D array with >=2 rows"
            )
        if length is None:
            length = len(values)
        elif len(values) != length:
            raise ValueError("all G0-v2 trace channels must have the same row count")
        arrays[name] = values

    for name in (
        "native_contact_fraction",
        "global__protein_contact_residue_fraction",
    ):
        if np.any((arrays[name] < 0.0) | (arrays[name] > 1.0)):
            raise ValueError(f"G0-v2 channel {name} must remain in [0, 1]")
    for name in (
        "pocket_geometric_com_distance_A",
        "pose__aligned_ligand_rmsd_A",
        "pose__aligned_centroid_displacement_A",
        "global__protein_min_heavy_distance_A",
    ):
        if np.any(arrays[name] < 0.0):
            raise ValueError(f"G0-v2 channel {name} must be nonnegative")
    global_count = arrays["global__protein_contact_residue_count"]
    if np.any(global_count < 0.0) or not np.all(global_count == np.floor(global_count)):
        raise ValueError(
            "G0-v2 global contact residue count must contain nonnegative integers"
        )

    contacts = np.asarray(pocket_contacts, dtype=np.float64)
    if (
        contacts.ndim != 2
        or contacts.shape[0] != length
        or contacts.shape[1] < 1
        or not np.all(np.isfinite(contacts))
        or not np.all(np.isin(contacts, (0.0, 1.0)))
    ):
        raise ValueError(
            "pocket_contacts must be a finite binary matrix with one row per dense frame"
        )
    return arrays, contacts


def _instantaneous_state_labels(
    numeric: Mapping[str, np.ndarray],
    pocket_contacts: np.ndarray,
    *,
    criteria: Mapping[str, float],
) -> np.ndarray:
    _validate_frozen_state_criteria(criteria)
    distance = numeric["pocket_geometric_com_distance_A"]
    native = numeric["native_contact_fraction"]
    initial_contact_count = np.sum(pocket_contacts, axis=1)
    global_count = numeric["global__protein_contact_residue_count"]
    global_min = numeric["global__protein_min_heavy_distance_A"]

    a = (
        (distance <= criteria["bound_max_pocket_com_A"])
        & (native >= criteria["bound_min_native_contact_fraction"])
    )
    pocket_exit = (
        (distance >= criteria["pocket_exit_min_pocket_com_A"])
        & (native <= criteria["pocket_exit_max_native_contact_fraction"])
        & (
            initial_contact_count
            <= criteria["pocket_exit_max_initial_contacts"]
        )
    )
    b = (
        pocket_exit
        & (global_count <= criteria["bulk_unbound_max_global_contacts"])
        & (global_min >= criteria["bulk_unbound_min_global_distance_A"])
    )

    labels = np.full(len(distance), 1, dtype=np.int8)  # I
    labels[pocket_exit] = 2  # P_ONLY
    labels[b] = 3  # B
    labels[a] = 0  # A precedence is part of the frozen state contract.
    return labels


def build_g0_replica_embedding(
    numeric: Mapping[str, Sequence[float] | np.ndarray],
    pocket_contacts: Sequence[Sequence[float]] | np.ndarray,
    *,
    criteria: Mapping[str, float],
) -> G0V2ReplicaEmbedding:
    """Build one label-blind G0-v2 vector from a complete dense episode.

    Callers must crop to Production frame 0 through the first subsequently
    confirmed terminal-B onset, inclusive.  ``criteria`` is mandatory and
    must match the frozen hash shared with G50-v2.
    """

    _validate_frozen_state_criteria(criteria)
    arrays, contacts = _validate_numeric_inputs(numeric, pocket_contacts)
    n_frames = len(next(iter(arrays.values())))

    # Supply the complete observed episode so the shared grid can verify both
    # real boundaries.  Its half-open q<1 contract guarantees that the
    # terminal row is never selected and therefore has zero normalized
    # support.  Only the explicit endpoint deltas below use row -1.
    support_indices0 = np.arange(n_frames, dtype=np.int64)
    continuous_support = np.column_stack(
        [arrays[CHANNEL_SOURCE_FIELDS[channel]] for channel in CONTINUOUS_CHANNELS]
    )
    query, continuous_grid = left_hold_on_normalized_grid(
        continuous_support,
        source_indices0=support_indices0,
        episode_frames=n_frames,
    )
    interval_means = interval_means_on_grid(
        query,
        continuous_grid,
        intervals=PROGRESS_INTERVALS,
    )

    values: list[float] = []
    for channel_index in range(len(CONTINUOUS_CHANNELS)):
        values.extend(float(value) for value in interval_means[:, channel_index])
    values.extend(
        float(arrays[source_field][-1] - arrays[source_field][0])
        for source_field in CHANNEL_SOURCE_FIELDS.values()
    )

    support_labels = _instantaneous_state_labels(
        arrays,
        contacts,
        criteria=criteria,
    )
    _, state_grid = left_hold_on_normalized_grid(
        support_labels,
        source_indices0=support_indices0,
        episode_frames=n_frames,
    )
    occupancy = np.asarray(
        [
            np.mean(state_grid == state_index)
            for state_index in range(len(STATE_ORDER))
        ],
        dtype=np.float64,
    )
    if not np.isclose(np.sum(occupancy), 1.0):
        raise RuntimeError("G0-v2 fixed-grid A/I/P_ONLY/B occupancy must sum to one")
    values.extend(float(value) for value in occupancy)

    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (len(G0_FEATURE_NAMES),) or not np.all(np.isfinite(vector)):
        raise RuntimeError("G0-v2 construction did not produce one finite 34D vector")
    vector.setflags(write=False)
    return G0V2ReplicaEmbedding(values=vector)


def aggregate_exactly_three_g0_replicas(
    embeddings_by_replica: Mapping[str, G0V2ReplicaEmbedding],
) -> G0V2ThreeReplicaAggregation:
    """Aggregate exactly three preregistered replicas before pKoff fitting."""

    if len(embeddings_by_replica) != EXPECTED_REPLICAS_PER_SYSTEM:
        raise ValueError(
            "G0-v2 primary aggregation requires exactly three frozen replicas"
        )
    replica_ids = tuple(sorted(str(key) for key in embeddings_by_replica))
    if len(set(replica_ids)) != EXPECTED_REPLICAS_PER_SYSTEM:
        raise ValueError("G0-v2 replica identifiers must be unique")

    rows: list[np.ndarray] = []
    for replica_id in replica_ids:
        embedding = embeddings_by_replica[replica_id]
        if (
            embedding.feature_names != G0_FEATURE_NAMES
            or embedding.semantic_ids != G0_SHARED34_SEMANTIC_IDS
        ):
            raise ValueError("G0-v2 replica feature/semantic schema drift")
        row = np.asarray(embedding.values, dtype=np.float64)
        if row.shape != (len(G0_FEATURE_NAMES),) or not np.all(np.isfinite(row)):
            raise ValueError("G0-v2 replica vector is invalid")
        rows.append(row)

    matrix = np.vstack(rows)
    primary_mean = np.mean(matrix, axis=0)
    population_sd = np.std(matrix, axis=0, ddof=0)
    sensitivity_names = tuple(
        name
        for feature_name in G0_FEATURE_NAMES
        for name in (
            f"{feature_name}__replica_mean",
            f"{feature_name}__replica_population_sd",
        )
    )
    sensitivity = np.asarray(
        [
            value
            for mean, sd in zip(primary_mean, population_sd, strict=True)
            for value in (mean, sd)
        ],
        dtype=np.float64,
    )
    primary_mean.setflags(write=False)
    sensitivity.setflags(write=False)
    return G0V2ThreeReplicaAggregation(
        replica_ids=replica_ids,
        primary_feature_names=G0_FEATURE_NAMES,
        primary_semantic_ids=G0_SHARED34_SEMANTIC_IDS,
        primary_mean=primary_mean,
        sensitivity_feature_names=sensitivity_names,
        sensitivity_mean_and_population_sd=sensitivity,
    )
