"""Training-only constant removal and correlation reduction for tabular features.

The reducer first drops constant columns from the current training partition.
It then groups columns whose absolute Spearman correlations all exceed the
declared threshold and retains one medoid per group.  It never reads a label
or a held-out test row while fitting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FoldLocalSpearmanReducer(BaseEstimator, TransformerMixin):
    """Remove constant and redundant columns using the current training rows."""

    def __init__(self, variance_threshold: float = 1e-12, correlation_threshold: float = 0.95):
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold

    def fit(self, x: np.ndarray, y: np.ndarray | None = None):
        del y
        values = np.asarray(x, dtype=float)
        if values.ndim != 2 or values.shape[1] == 0 or not np.all(np.isfinite(values)):
            raise ValueError("FoldLocalSpearmanReducer requires a finite nonempty 2D matrix")
        if not 0.0 < float(self.correlation_threshold) < 1.0:
            raise ValueError("correlation_threshold must be in (0,1)")
        variance_indices = np.flatnonzero(np.var(values, axis=0, ddof=0) > float(self.variance_threshold))
        if len(variance_indices) == 0:
            raise ValueError("variance filter removed every feature")
        ranks = pd.DataFrame(values[:, variance_indices]).rank(axis=0, method="average").to_numpy(dtype=float)
        correlation = np.corrcoef(ranks, rowvar=False)
        if correlation.ndim == 0:
            correlation = np.asarray([[1.0]], dtype=float)
        absolute = np.abs(np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0))
        np.fill_diagonal(absolute, 1.0)

        clusters: list[tuple[int, ...]] = [(index,) for index in range(absolute.shape[0])]
        while True:
            candidates = []
            for left_index, left in enumerate(clusters):
                for right_index in range(left_index + 1, len(clusters)):
                    right = clusters[right_index]
                    minimum_cross = min(absolute[i, j] for i in left for j in right)
                    if minimum_cross >= float(self.correlation_threshold):
                        candidates.append((-float(minimum_cross), tuple(sorted(left + right)), left_index, right_index))
            if not candidates:
                break
            _, merged, left_index, right_index = min(candidates)
            clusters = [cluster for index, cluster in enumerate(clusters) if index not in {left_index, right_index}]
            clusters.append(merged)
            clusters.sort(key=lambda cluster: (cluster[0], cluster))

        representatives = []
        for cluster in clusters:
            if len(cluster) == 1:
                representatives.append(cluster[0])
                continue
            score = {index: float(np.mean([1.0 - absolute[index, other] for other in cluster])) for index in cluster}
            representatives.append(min(cluster, key=lambda index: (score[index], index)))
        self.n_features_in_ = values.shape[1]
        self.variance_indices_ = variance_indices.astype(int)
        self.selected_original_indices_ = variance_indices[np.asarray(sorted(representatives), dtype=int)].astype(int)
        self.clusters_after_variance_ = tuple(clusters)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError("feature schema changed between fit and transform")
        return values[:, self.selected_original_indices_]
