"""Small, dependency-free mathematics shared by public trajectory summaries."""

from __future__ import annotations

import numpy as np


def temporal_voronoi_weights(indices: np.ndarray, n_frames: int) -> np.ndarray:
    """Return the temporal support represented by each ordered selected frame.

    A selected frame represents the midpoint interval between itself and its
    neighbours.  The returned weights sum to ``n_frames`` and are used only by
    legacy summary helpers retained for backwards-compatible geometry audits.
    """

    selected = np.asarray(indices, dtype=np.int64)
    if len(selected) == 1:
        return np.asarray([float(n_frames)])
    boundaries = np.r_[0.0, (selected[:-1] + selected[1:] + 1) / 2.0, float(n_frames)]
    weights = np.diff(boundaries)
    return weights * (float(n_frames) / float(np.sum(weights)))
