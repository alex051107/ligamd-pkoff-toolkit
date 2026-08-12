#!/usr/bin/env python3
"""G50-v2 coordinate-derived path representation for the LiGaMD pKoff screen.

The primary total-utility view contains 41 values per successful Production
replica.  Its first 34 semantic fields are exactly the G0-v2 shared block:

* 24 continuous-channel interval means on NormalizedProgressGrid-v1;
* six real endpoint-minus-start changes; and
* four state occupancies on the same normalized-support grid.

The remaining seven fields add coordinate-only contact/conformation content.
Every support summary uses the same 512 half-open midpoint grid with
previous-observation/left-hold reconstruction.  The terminal observation at
progress 1 therefore has zero support.  It is read only by explicitly named
endpoint fields.

This module deliberately uses ``g50v2__`` public names.  Existing G50-v1
tables cannot be silently accepted under the new semantics.

In practical terms, this module performs three deliberately separate jobs.
It first places the 512 real frame observations selected by a sampler onto one
common 512-position progress scale. It then builds the traceable Core41-v2
replica vector from those shared measurements. A separate label-blind contract
may later name a ten-field G10-v2 view from Core41-v2. This module never reads
experimental pKoff and never keeps a field because a regression coefficient
looks favourable.

``left-hold`` means that a query position reuses the most recent real observed
frame. It never interpolates a coordinate that was not saved by the simulation.
The terminal complete-exit onset is retained for explicitly named endpoint
features, but has no duration weight because all support queries lie in
``(0, 1)``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType

import numpy as np

from .g0_trace_summary_v2 import (
    FROZEN_STATE_CRITERIA,
    G0_SHARED34_SEMANTIC_IDS,
    G0_SHARED34_SEMANTIC_SHA256,
    STATE_CRITERIA_SHA256,
    state_criteria_sha256,
)
from .normalized_progress_grid_v1 import (
    GRID_SIZE,
    NORMALIZED_PROGRESS_GRID_SEMANTIC_ID,
    interval_means_on_grid,
    left_hold_on_normalized_grid,
)
from .prediction_first_coordinate_features import (
    CoordinateFeatureBlock,
    coordinate_state_labels,
)


SCHEMA_VERSION = "ligamd_prediction_first_geometric_contact_G50_core41_v2.0"
SEMANTIC_VERSION = "LiGaMD-G50-core41-v2:NormalizedProgressGrid-v1"
EXPECTED_REPLICAS_PER_SYSTEM = 3

CONTINUOUS_CHANNELS = (
    "pocket_com_distance_A",
    "native_contact_fraction",
    "ligand_pose_rmsd_A",
    "ligand_centroid_displacement_A",
    "global_min_heavy_distance_A",
    "global_contact_fraction",
)
# Each channel answers a distinct geometric question: displacement from the
# starting pocket, retention of starting contacts, change in ligand pose, or
# continued proximity to any protein surface. The order is frozen because the
# feature-schema hash and every downstream table depend on it.
_BLOCK_NUMERIC_FIELDS = MappingProxyType(
    {
        "pocket_com_distance_A": "pocket_geometric_com_distance_A",
        "native_contact_fraction": "native_contact_fraction",
        "ligand_pose_rmsd_A": "pose__aligned_ligand_rmsd_A",
        "ligand_centroid_displacement_A": "pose__aligned_centroid_displacement_A",
        "global_min_heavy_distance_A": "global__protein_min_heavy_distance_A",
        "global_contact_fraction": "global__protein_contact_residue_fraction",
    }
)
PROGRESS_INTERVALS = (
    ("000_050", 0.00, 0.50),
    ("050_080", 0.50, 0.80),
    ("080_095", 0.80, 0.95),
    ("095_100", 0.95, 1.00),
)
# These four summary windows are project-specific frozen choices, not
# literature-proven universal boundaries. Half-open intervals prevent a grid
# midpoint from contributing to two windows.
STATE_ORDER = ("A", "I", "P_ONLY", "B")


def _newline_digest(values: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def _shared34_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for channel in CONTINUOUS_CHANNELS:
        names.extend(
            f"g50v2__{channel}__progress_{label}__mean"
            for label, _, _ in PROGRESS_INTERVALS
        )
    names.extend(
        f"g50v2__{channel}__end_minus_start" for channel in CONTINUOUS_CHANNELS
    )
    names.extend(
        f"g50v2__state_progress_occupancy__{state}" for state in STATE_ORDER
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


G50_SHARED34_FEATURE_NAMES = _shared34_feature_names()
G50_SHARED34_SEMANTIC_IDS = _shared34_semantic_ids()

G50_INCREMENTAL7_FEATURE_NAMES = (
    "g50v2__pocket_contacts__contact_fraction_progress_mean",
    "g50v2__pocket_contacts__initial_contacts_lost_by_end_fraction",
    "g50v2__pocket_contacts__new_contact_residues_ever_fraction",
    "g50v2__global_contacts__contact_fraction_max",
    "g50v2__global_contacts__contact_fraction_last20pct_progress_p90",
    "g50v2__ligand_internal_rmsd_A__progress_mean",
    "g50v2__ligand_internal_rmsd_A__progress_population_sd",
)
G50_INCREMENTAL7_SEMANTIC_IDS = (
    "normalized_support::pocket_contacts::contact_fraction::mean",
    "endpoint::pocket_contacts::initial_contacts_lost_by_end_fraction",
    "normalized_support::pocket_contacts::new_contact_residues_ever_fraction",
    "normalized_support::global_contacts::contact_fraction::max",
    "normalized_support::global_contacts::contact_fraction::last20pct_p90",
    "normalized_support::ligand_internal_rmsd_A::mean",
    "normalized_support::ligand_internal_rmsd_A::population_sd",
)

G50_TOTAL_CORE41_V2_FEATURE_NAMES = (
    G50_SHARED34_FEATURE_NAMES + G50_INCREMENTAL7_FEATURE_NAMES
)
G50_TOTAL_CORE41_V2_SEMANTIC_IDS = (
    G50_SHARED34_SEMANTIC_IDS + G50_INCREMENTAL7_SEMANTIC_IDS
)


@dataclass(frozen=True)
class G50V2FeatureView:
    """Frozen machine-readable feature view."""

    name: str
    feature_names: tuple[str, ...]
    semantic_ids: tuple[str, ...]
    feature_schema_sha256: str
    semantic_ids_sha256: str
    contract_sha256: str

    @property
    def dimension(self) -> int:
        return len(self.feature_names)

    def as_mapping(self) -> dict[str, object]:
        return {
            "name": self.name,
            "schema_version": SCHEMA_VERSION,
            "semantic_version": SEMANTIC_VERSION,
            "normalized_progress_grid_semantic_id": (
                NORMALIZED_PROGRESS_GRID_SEMANTIC_ID
            ),
            "normalized_progress_grid_size": GRID_SIZE,
            "state_criteria_sha256": STATE_CRITERIA_SHA256,
            "dimension": self.dimension,
            "feature_names": list(self.feature_names),
            "semantic_ids": list(self.semantic_ids),
            "feature_schema_sha256": self.feature_schema_sha256,
            "semantic_ids_sha256": self.semantic_ids_sha256,
            "contract_sha256": self.contract_sha256,
        }


def _view_contract_sha256(
    *,
    name: str,
    feature_names: tuple[str, ...],
    semantic_ids: tuple[str, ...],
) -> str:
    payload = {
        "name": name,
        "schema_version": SCHEMA_VERSION,
        "semantic_version": SEMANTIC_VERSION,
        "normalized_progress_grid_semantic_id": NORMALIZED_PROGRESS_GRID_SEMANTIC_ID,
        "normalized_progress_grid_size": GRID_SIZE,
        "state_criteria_sha256": STATE_CRITERIA_SHA256,
        "feature_names": list(feature_names),
        "semantic_ids": list(semantic_ids),
    }
    canonical = json.dumps(
        payload, allow_nan=False, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256((canonical + "\n").encode("utf-8")).hexdigest()


def _make_view(
    name: str,
    feature_names: tuple[str, ...],
    semantic_ids: tuple[str, ...],
) -> G50V2FeatureView:
    if len(feature_names) != len(semantic_ids):
        raise RuntimeError(f"{name} feature/semantic dimensions differ")
    if len(set(feature_names)) != len(feature_names):
        raise RuntimeError(f"{name} feature names are not unique")
    if len(set(semantic_ids)) != len(semantic_ids):
        raise RuntimeError(f"{name} semantic IDs are not unique")
    return G50V2FeatureView(
        name=name,
        feature_names=feature_names,
        semantic_ids=semantic_ids,
        feature_schema_sha256=_newline_digest(feature_names),
        semantic_ids_sha256=_newline_digest(semantic_ids),
        contract_sha256=_view_contract_sha256(
            name=name,
            feature_names=feature_names,
            semantic_ids=semantic_ids,
        ),
    )


G50_TOTAL_CORE41_V2 = _make_view(
    "G50_TOTAL_CORE41_V2",
    G50_TOTAL_CORE41_V2_FEATURE_NAMES,
    G50_TOTAL_CORE41_V2_SEMANTIC_IDS,
)
G50_INCREMENTAL7 = _make_view(
    "G50_INCREMENTAL7",
    G50_INCREMENTAL7_FEATURE_NAMES,
    G50_INCREMENTAL7_SEMANTIC_IDS,
)

EXPECTED_G50_TOTAL_CORE41_V2_CONTRACT_SHA256 = (
    "11653bcbb4f7cea33a72dd1379dfeaed83b60c0e517bd04d08d5d5b73b6a4576"
)
EXPECTED_G50_INCREMENTAL7_CONTRACT_SHA256 = (
    "074a39557826962bdf1b69b1bbf119817628dcca62798950004beb364fd06af5"
)


def g50_v2_machine_readable_views() -> dict[str, object]:
    """Return a JSON-serializable copy of both frozen views."""

    return {
        "schema_version": SCHEMA_VERSION,
        "semantic_version": SEMANTIC_VERSION,
        "views": {
            G50_TOTAL_CORE41_V2.name: G50_TOTAL_CORE41_V2.as_mapping(),
            G50_INCREMENTAL7.name: G50_INCREMENTAL7.as_mapping(),
        },
    }


@dataclass(frozen=True)
class G50V2ReplicaEmbedding:
    """One successful-dissociation replica represented as total41."""

    values: np.ndarray
    state_criteria_sha256: str
    feature_names: tuple[str, ...] = G50_TOTAL_CORE41_V2_FEATURE_NAMES
    semantic_ids: tuple[str, ...] = G50_TOTAL_CORE41_V2_SEMANTIC_IDS

    def as_mapping(self) -> dict[str, float]:
        return {
            name: float(value)
            for name, value in zip(self.feature_names, self.values, strict=True)
        }

    def incremental7_mapping(self) -> dict[str, float]:
        start = len(G50_SHARED34_FEATURE_NAMES)
        return {
            name: float(value)
            for name, value in zip(
                G50_INCREMENTAL7_FEATURE_NAMES,
                self.values[start:],
                strict=True,
            )
        }


def _validate_source_indices(
    source_indices0: Sequence[int] | np.ndarray,
    *,
    episode_frames: int,
    selected_frame_count: int,
) -> np.ndarray:
    """Verify that selected frames are real, ordered observations spanning one episode.

    ``episode_frames`` counts from production frame 0 through the first
    persistence-confirmed complete-exit onset, including both endpoints. The
    function rejects a missing endpoint, duplicate frames, and floating-point
    or Boolean indices. Those would otherwise make a sampler appear to have
    selected 512 frames while no longer referring to 512 real coordinates.
    """

    if (
        isinstance(episode_frames, bool)
        or not isinstance(episode_frames, Integral)
        or int(episode_frames) < 2
    ):
        raise ValueError("episode_frames must be an integer >=2")
    raw = np.asarray(source_indices0)
    if (
        raw.ndim != 1
        or len(raw) != selected_frame_count
        or np.issubdtype(raw.dtype, np.bool_)
        or not np.issubdtype(raw.dtype, np.integer)
    ):
        raise ValueError("source_indices0 must contain one exact integer per frame")
    indices = raw.astype(np.int64, copy=False)
    if int(episode_frames) < 2:
        raise ValueError("episode_frames must include frame 0 and terminal-B onset")
    if len(indices) < 2 or np.any(np.diff(indices) <= 0):
        raise ValueError("selected source indices must be unique and increasing")
    if indices[0] != 0 or indices[-1] != int(episode_frames) - 1:
        raise ValueError(
            "G50-v2 requires Production frame 0 and first persistent-B onset"
        )
    return indices


def _continuous_matrix(block: CoordinateFeatureBlock) -> np.ndarray:
    """Assemble the six frozen continuous channels in their public column order.

    The result has shape ``(n_selected_frames, 6)``. Fraction fields must lie
    between zero and one; distance and RMSD fields must be non-negative; all
    values must be finite. These simple checks stop a unit or missing-value
    error before it is compressed into a plausible-looking trajectory summary.
    """

    frame_count = len(block.pocket_contacts)
    columns: list[np.ndarray] = []
    for channel in CONTINUOUS_CHANNELS:
        field = _BLOCK_NUMERIC_FIELDS[channel]
        if field not in block.numeric:
            raise ValueError(f"coordinate block lacks required field: {field}")
        values = np.asarray(block.numeric[field], dtype=float)
        if values.shape != (frame_count,) or not np.all(np.isfinite(values)):
            raise ValueError(f"invalid selected-frame values for {channel}")
        columns.append(values)
    matrix = np.column_stack(columns)
    fraction_columns = (
        CONTINUOUS_CHANNELS.index("native_contact_fraction"),
        CONTINUOUS_CHANNELS.index("global_contact_fraction"),
    )
    if any(
        np.any((matrix[:, column] < 0) | (matrix[:, column] > 1))
        for column in fraction_columns
    ):
        raise ValueError("contact fractions must remain in [0, 1]")
    nonnegative_columns = tuple(
        column
        for column, channel in enumerate(CONTINUOUS_CHANNELS)
        if channel not in ("native_contact_fraction", "global_contact_fraction")
    )
    if any(np.any(matrix[:, column] < 0) for column in nonnegative_columns):
        raise ValueError("distance/RMSD channels must be nonnegative")
    return matrix


def _binary_contacts(
    contacts: Sequence[Sequence[bool]] | np.ndarray,
    *,
    frame_count: int,
) -> np.ndarray:
    """Require a non-empty Boolean matrix with one row per selected frame."""

    raw = np.asarray(contacts)
    if (
        raw.ndim != 2
        or raw.shape[0] != frame_count
        or raw.shape[1] < 1
        or not np.all(np.isin(raw, (False, True, 0, 1)))
    ):
        raise ValueError("pocket contacts must be a nonempty binary matrix")
    return raw.astype(bool, copy=False)


def _incremental7(
    block: CoordinateFeatureBlock,
    *,
    indices: np.ndarray,
    episode_frames: int,
    query: np.ndarray,
) -> np.ndarray:
    """Build the seven Core41-v2 additions that require real selected coordinates.

    The output order is exactly ``G50_INCREMENTAL7_FEATURE_NAMES``. The fields
    are mean frozen-pocket contact fraction; fraction of initial contacts lost
    by the terminal onset; fraction of previously absent pocket residues that
    ever contact the ligand; maximum global-contact fraction; 90th percentile
    of global-contact fraction in the last 20% of progress; mean ligand
    internal RMSD; and its population standard deviation (``ddof=0``).

    Only the explicitly named initial-contact-loss field reads the terminal
    real frame directly. Duration-like summaries are calculated on the
    left-hold support grid, so a sampler cannot change their weight merely by
    saving many frames in one short portion of the episode.
    """

    contacts = _binary_contacts(
        block.pocket_contacts, frame_count=len(indices)
    )
    _, grid_contacts = left_hold_on_normalized_grid(
        contacts,
        source_indices0=indices,
        episode_frames=episode_frames,
    )
    grid_contacts = np.asarray(grid_contacts, dtype=bool)
    grid_contact_fraction = np.mean(grid_contacts, axis=1)
    initial = contacts[0]
    noninitial = ~initial
    lost_initial = (
        float(np.mean(~contacts[-1, initial])) if np.any(initial) else 0.0
    )
    new_ever = (
        float(np.mean(np.any(grid_contacts[:, noninitial], axis=0)))
        if np.any(noninitial)
        else 0.0
    )

    global_fraction = np.asarray(
        block.numeric["global__protein_contact_residue_fraction"], dtype=float
    )
    if (
        global_fraction.shape != (len(indices),)
        or not np.all(np.isfinite(global_fraction))
        or np.any((global_fraction < 0) | (global_fraction > 1))
    ):
        raise ValueError("global contact fraction trace is invalid")
    _, grid_global_fraction = left_hold_on_normalized_grid(
        global_fraction,
        source_indices0=indices,
        episode_frames=episode_frames,
    )
    last20 = np.asarray(grid_global_fraction)[query >= 0.80]
    if len(last20) == 0:  # pragma: no cover - frozen grid invariant
        raise RuntimeError("last-20% normalized-support grid is empty")

    internal = np.asarray(
        block.numeric["pose__ligand_internal_conformation_rmsd_A"], dtype=float
    )
    if (
        internal.shape != (len(indices),)
        or not np.all(np.isfinite(internal))
        or np.any(internal < 0)
    ):
        raise ValueError("ligand internal RMSD trace is invalid")
    _, grid_internal = left_hold_on_normalized_grid(
        internal,
        source_indices0=indices,
        episode_frames=episode_frames,
    )
    grid_internal = np.asarray(grid_internal, dtype=float)

    sorted_last20 = np.sort(last20, kind="mergesort")
    p90_index = max(0, int(np.ceil(0.90 * len(sorted_last20))) - 1)
    values = np.asarray(
        (
            float(np.mean(grid_contact_fraction)),
            lost_initial,
            new_ever,
            float(np.max(grid_global_fraction)),
            float(sorted_last20[p90_index]),
            float(np.mean(grid_internal)),
            float(np.std(grid_internal, ddof=0)),
        ),
        dtype=float,
    )
    if values.shape != (7,) or not np.all(np.isfinite(values)):
        raise RuntimeError("G50-v2 incremental7 construction failed")
    return values


def summarize_geometric_contact_replica_v2(
    block: CoordinateFeatureBlock,
    *,
    source_indices0: Sequence[int] | np.ndarray,
    episode_frames: int,
    criteria: Mapping[str, float],
) -> G50V2ReplicaEmbedding:
    """Compress one selected dissociation replica into the frozen 41-value Core41-v2 vector.

    The calculation first verifies the state-rule hash and the 512 real source
    indices. It maps six continuous channels to the left-hold support grid,
    calculates four progress-window means per channel, adds six real
    endpoint-minus-start changes, converts the A/I/P_ONLY/B labels into four
    occupancies, and finally appends :func:`_incremental7`.

    This function returns one *replica-level* representation. Averaging the
    corresponding values across replicas and fitting a model to experimental
    pKoff happen later, in a separate stage.
    """

    criteria_sha = state_criteria_sha256(criteria)
    if criteria_sha != STATE_CRITERIA_SHA256:
        raise ValueError("state criteria drift from the frozen G0-v2/G50-v2 contract")
    indices = _validate_source_indices(
        source_indices0,
        episode_frames=episode_frames,
        selected_frame_count=len(block.pocket_contacts),
    )
    frames = int(episode_frames)
    continuous = _continuous_matrix(block)
    global_count = np.asarray(
        block.numeric["global__protein_contact_residue_count"], dtype=float
    )
    if (
        global_count.shape != (len(indices),)
        or not np.all(np.isfinite(global_count))
        or np.any(global_count < 0)
        or not np.all(global_count == np.floor(global_count))
    ):
        raise ValueError(
            "global protein contact count must contain finite nonnegative integers"
        )
    query, grid_continuous = left_hold_on_normalized_grid(
        continuous,
        source_indices0=indices,
        episode_frames=frames,
    )
    interval_means = interval_means_on_grid(
        query,
        grid_continuous,
        intervals=PROGRESS_INTERVALS,
    )
    # ``interval_means_on_grid`` is interval-major (4 x 6), whereas the public
    # schema is channel-major. Keeping one channel's four windows adjacent makes
    # its name and its numerical position agree in every exported table.
    shared: list[float] = []
    for column in range(len(CONTINUOUS_CHANNELS)):
        shared.extend(float(value) for value in interval_means[:, column])
    shared.extend(float(value) for value in (continuous[-1] - continuous[0]))

    labels = coordinate_state_labels(block, criteria)
    encoded = np.asarray(
        [STATE_ORDER.index(str(label)) for label in labels], dtype=np.int8
    )
    _, grid_labels = left_hold_on_normalized_grid(
        encoded,
        source_indices0=indices,
        episode_frames=frames,
    )
    occupancy = np.asarray(
        [np.mean(grid_labels == state_index) for state_index in range(4)],
        dtype=float,
    )
    if not np.isclose(np.sum(occupancy), 1.0):
        raise RuntimeError("G50-v2 state occupancy does not sum to one")
    shared.extend(float(value) for value in occupancy)

    shared_array = np.asarray(shared, dtype=float)
    if shared_array.shape != (34,) or not np.all(np.isfinite(shared_array)):
        raise RuntimeError("G50-v2 shared34 construction failed")
    incremental = _incremental7(
        block,
        indices=indices,
        episode_frames=frames,
        query=query,
    )
    vector = np.concatenate((shared_array, incremental))
    if vector.shape != (41,) or not np.all(np.isfinite(vector)):
        raise RuntimeError("G50-v2 total41 construction failed")
    vector.setflags(write=False)
    return G50V2ReplicaEmbedding(
        values=vector,
        state_criteria_sha256=criteria_sha,
    )


if G50_SHARED34_SEMANTIC_IDS != G0_SHARED34_SEMANTIC_IDS:  # pragma: no cover
    raise RuntimeError("G50-v2 shared34 semantic IDs differ from G0-v2")
if _newline_digest(G50_SHARED34_SEMANTIC_IDS) != G0_SHARED34_SEMANTIC_SHA256:
    raise RuntimeError("G50-v2/G0-v2 shared34 semantic digest differs")
if len(G50_TOTAL_CORE41_V2_FEATURE_NAMES) != 41:  # pragma: no cover
    raise RuntimeError("G50-v2 total view must contain exactly 41 fields")
if len(G50_INCREMENTAL7_FEATURE_NAMES) != 7:  # pragma: no cover
    raise RuntimeError("G50-v2 incremental view must contain exactly seven fields")
if set(G50_INCREMENTAL7_SEMANTIC_IDS) & set(G0_SHARED34_SEMANTIC_IDS):
    raise RuntimeError("G50-v2 incremental7 overlaps G0-v2 shared34 semantics")
if state_criteria_sha256(FROZEN_STATE_CRITERIA) != STATE_CRITERIA_SHA256:
    raise RuntimeError("G50-v2/G0-v2 state-criteria binding drifted")
if (
    G50_TOTAL_CORE41_V2.contract_sha256
    != EXPECTED_G50_TOTAL_CORE41_V2_CONTRACT_SHA256
):
    raise RuntimeError("G50-v2 total41 machine-readable contract drifted")
if (
    G50_INCREMENTAL7.contract_sha256
    != EXPECTED_G50_INCREMENTAL7_CONTRACT_SHA256
):
    raise RuntimeError("G50-v2 incremental7 machine-readable contract drifted")
