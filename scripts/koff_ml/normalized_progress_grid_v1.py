#!/usr/bin/env python3
"""Shared normalized-progress reconstruction for LiGaMD path summaries.

The contract is intentionally narrow and frozen:

* exactly 512 query points;
* query ``j`` is the midpoint ``(j + 0.5) / 512`` of a half-open cell;
* reconstruction uses the most recent observation at or before the query
  (previous-observation / left-hold);
* the terminal observation at normalized progress 1 is never selected.

This module does not use interpolation or Voronoi support.  It is shared by
G0-v2 and coordinate-derived G50-v2 so that their normalized-support
summaries have one auditable mathematical meaning.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


GRID_SIZE = 512
NORMALIZED_PROGRESS_GRID_SEMANTIC_ID = (
    "NormalizedProgressGrid-v1:"
    "K512:half-open-midpoint:previous-observation-left-hold:terminal-excluded"
)


def _require_frozen_grid_size(grid_size: int) -> int:
    """Accept only the frozen 512-point grid used by every path summary.

    A feature window is comparable only when every replica uses the same number
    of normalized observation positions.  Rejecting a different grid size makes
    an accidental change in summary resolution visible at the call site.
    """

    if isinstance(grid_size, bool) or not isinstance(grid_size, (int, np.integer)):
        raise TypeError("grid_size must be the frozen integer value 512")
    if int(grid_size) != GRID_SIZE:
        raise ValueError("NormalizedProgressGrid-v1 is frozen at exactly 512 points")
    return int(grid_size)


def midpoint_query(*, grid_size: int = GRID_SIZE) -> np.ndarray:
    """Return the frozen half-open midpoint query grid as a read-only array."""

    size = _require_frozen_grid_size(grid_size)
    query = (np.arange(size, dtype=np.float64) + 0.5) / float(size)
    if not np.all((query > 0.0) & (query < 1.0)):  # pragma: no cover - invariant
        raise RuntimeError("normalized-progress midpoint grid must remain inside (0, 1)")
    query.setflags(write=False)
    return query


def _validated_source_indices(
    source_indices0: Sequence[int] | np.ndarray,
    *,
    episode_frames: int,
    observation_count: int,
) -> np.ndarray:
    """Check that sampled observations still describe one complete episode.

    ``source_indices0`` contains real zero-based NetCDF frame indices selected
    from the episode.  It must have one value per observation, be strictly
    increasing without duplicates, start at production frame 0, and end at the
    persistence-confirmed complete-exit onset.  The endpoint is retained to
    prove that the sampling interval is complete even though it receives no
    normalized-grid support below.
    """

    if (
        isinstance(episode_frames, bool)
        or not isinstance(episode_frames, (int, np.integer))
        or int(episode_frames) < 2
    ):
        raise ValueError("episode_frames must be an integer >=2")
    frames = int(episode_frames)

    raw = np.asarray(source_indices0)
    if raw.ndim != 1 or len(raw) != observation_count or len(raw) < 2:
        raise ValueError(
            "source_indices0 must be a 1D array with >=2 rows matching values"
        )
    if np.issubdtype(raw.dtype, np.bool_) or not np.issubdtype(
        raw.dtype, np.integer
    ):
        raise TypeError("source_indices0 must contain integer frame indices")
    indices = raw.astype(np.int64, copy=False)
    if indices[0] != 0:
        raise ValueError("source_indices0 must start at production episode frame 0")
    if np.any(np.diff(indices) <= 0):
        raise ValueError("source_indices0 must be strictly increasing")
    if indices[-1] != frames - 1:
        raise ValueError(
            "source_indices0 must include the real terminal-B onset endpoint"
        )
    return indices


def left_hold_on_normalized_grid(
    values: Sequence[float] | np.ndarray,
    *,
    source_indices0: Sequence[int] | np.ndarray,
    episode_frames: int,
    grid_size: int = GRID_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct observations on the shared normalized-progress grid.

    ``values`` may be one- or multi-dimensional; its first axis corresponds
    one-to-one with ``source_indices0``.  The function deliberately does not
    select the required observation whose source index is
    ``episode_frames - 1``.  The real endpoint must be present for contract
    closure and may be used by separate endpoint-only features, but receives
    zero normalized-support weight.
    """

    query = midpoint_query(grid_size=grid_size)
    observed = np.asarray(values)
    if observed.ndim < 1 or observed.shape[0] < 1:
        raise ValueError("values must have a non-empty observation axis")
    indices = _validated_source_indices(
        source_indices0,
        episode_frames=episode_frames,
        observation_count=observed.shape[0],
    )

    progress = indices.astype(np.float64) / float(int(episode_frames) - 1)
    # ``side="right" - 1`` implements previous-observation, or left-hold.
    # At an exact observation position it uses that observation; between two
    # observations it uses the nearest earlier real frame.  No interpolated
    # coordinate or weighted average is created.
    positions = np.searchsorted(progress, query, side="right") - 1
    if np.any(positions < 0):  # pragma: no cover - first source index is zero
        raise RuntimeError("every midpoint query must have a previous observation")
    chosen_source_indices = indices[positions]
    if np.any(chosen_source_indices >= int(episode_frames) - 1):
        raise RuntimeError(
            "NormalizedProgressGrid-v1 must never select the terminal endpoint"
        )

    reconstructed = np.asarray(observed[positions]).copy()
    reconstructed.setflags(write=False)
    return query, reconstructed


def interval_means_on_grid(
    query: Sequence[float] | np.ndarray,
    sampled: Sequence[float] | np.ndarray,
    *,
    intervals: Sequence[tuple[str, float, float]],
) -> np.ndarray:
    """Average fixed-grid samples inside named half-open progress intervals."""

    query_array = np.asarray(query, dtype=np.float64)
    expected_query = midpoint_query()
    if (
        query_array.shape != expected_query.shape
        or not np.all(np.isfinite(query_array))
        or not np.array_equal(query_array, expected_query)
    ):
        raise ValueError("query must be the exact frozen 512-point midpoint grid")

    sampled_array = np.asarray(sampled)
    if sampled_array.ndim < 1 or sampled_array.shape[0] != GRID_SIZE:
        raise ValueError("sampled must have exactly 512 rows on the shared grid")
    if not np.issubdtype(sampled_array.dtype, np.number):
        raise TypeError("sampled interval values must be numeric")
    if not np.all(np.isfinite(sampled_array)):
        raise ValueError("sampled interval values must be finite")
    if not intervals:
        raise ValueError("at least one normalized-progress interval is required")

    means: list[np.ndarray] = []
    for interval in intervals:
        if len(interval) != 3:
            raise ValueError("each interval must be (label, left, right)")
        _, left_raw, right_raw = interval
        left = float(left_raw)
        right = float(right_raw)
        if (
            not np.isfinite(left)
            or not np.isfinite(right)
            or left < 0.0
            or right > 1.0
            or left >= right
        ):
            raise ValueError("interval bounds must satisfy 0 <= left < right <= 1")
        mask = (query_array >= left) & (query_array < right)
        if not np.any(mask):
            raise ValueError("each interval must contain at least one grid midpoint")
        means.append(np.asarray(np.mean(sampled_array[mask], axis=0)))

    result = np.stack(means, axis=0)
    if not np.all(np.isfinite(result)):  # pragma: no cover - guarded above
        raise RuntimeError("fixed-grid interval means must be finite")
    result.setflags(write=False)
    return result
