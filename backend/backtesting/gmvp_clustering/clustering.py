"""Bounded K-Means clustering for asset grouping."""

import logging
import random
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class BoundedKMeansClustering:
    """
    K-Means clustering with maximum cluster size constraint.

    Assigns points to nearest centroid while respecting size limits,
    useful for grouping assets before portfolio optimization.
    """

    def __init__(
        self,
        n_clusters: int,
        max_cluster_size: int,
        n_iter: int = 10,
        n_init: int = 10,
        random_state: int | None = None
    ):
        self.n_clusters = n_clusters
        self.max_cluster_size = max_cluster_size
        self.n_iter = n_iter
        self.n_init = n_init
        self.random_state = random_state
        self._n_points = None

    def fit(
        self,
        X: np.ndarray,
        weights: np.ndarray,
        dist_array: np.ndarray | None = None
    ) -> Tuple[float, List[List[int]]]:
        """
        Fit clustering to data.

        Args:
            X: Feature matrix (n_samples, n_features)
            weights: Sample weights for size constraints
            dist_array: Precomputed distance matrix (optional)

        Returns:
            (best_cost, cluster_indices) where cluster_indices[i] = list of point indices in cluster i
        """
        if self.random_state is not None:
            random.seed(self.random_state)

        self._n_points = X.shape[0]
        if dist_array is None:
            dist_array = self._compute_dist_array(X)

        results = [
            self._fit_one_iteration(X, weights, dist_array)
            for _ in range(self.n_init)
        ]

        costs = [r[0] for r in results]
        clusters = [r[1] for r in results]

        if all(np.isnan(costs)):
            logger.warning("All clustering iterations failed")
            self._n_points = None
            return np.nan, [[]]

        best_idx = np.nanargmin(costs)
        self._n_points = None
        return costs[best_idx], clusters[best_idx]

    def _compute_dist_array(self, X: np.ndarray) -> np.ndarray:
        """Compute pairwise Euclidean distance matrix."""
        n = X.shape[0]
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(X[i] - X[j])
                dist[i, j] = d
                dist[j, i] = d
        return dist

    def _fit_one_iteration(
        self,
        X: np.ndarray,
        weights: np.ndarray,
        dist_array: np.ndarray
    ) -> Tuple[float, List[List[int]]]:
        """Single clustering iteration with random initialization."""
        try:
            clusters = self._initialize_clusters(weights, dist_array)
            best_clusters = clusters
            best_cost = self._max_mean_dist(dist_array, clusters)

            for _ in range(self.n_iter):
                clusters, cost = self._optimize_clusters(X, weights, dist_array, clusters)
                if clusters == best_clusters:
                    break
                if cost < best_cost:
                    best_cost = cost
                    best_clusters = clusters

            return best_cost, best_clusters

        except ValueError as e:
            logger.debug("Clustering iteration failed: %s", e)
            return np.nan, [[]]

    def _initialize_clusters(
        self,
        weights: np.ndarray,
        dist_array: np.ndarray
    ) -> List[List[int]]:
        """Initialize clusters with random centroids."""
        centroid_idxs = random.sample(range(self._n_points), self.n_clusters)
        return self._assign_to_clusters(weights, dist_array, centroid_idxs)

    def _assign_to_clusters(
        self,
        weights: np.ndarray,
        dist_array: np.ndarray,
        centroid_idxs: List[int]
    ) -> List[List[int]]:
        """Assign points to nearest valid cluster."""
        clusters = [[c] for c in centroid_idxs]
        cluster_weights = np.array([weights[c] for c in centroid_idxs])

        # Sort remaining points by weight (descending)
        remaining = [i for i in np.argsort(-weights) if i not in centroid_idxs]

        for p_idx in remaining:
            assigned = False
            # Try clusters in order of distance
            sorted_clusters = np.argsort(dist_array[p_idx][centroid_idxs])

            for c_idx in sorted_clusters:
                if cluster_weights[c_idx] + weights[p_idx] <= self.max_cluster_size:
                    clusters[c_idx].append(p_idx)
                    cluster_weights[c_idx] += weights[p_idx]
                    assigned = True
                    break

            if not assigned:
                raise ValueError(f"Point {p_idx} could not be assigned")

        return clusters

    def _max_mean_dist(
        self,
        dist_array: np.ndarray,
        clusters: List[List[int]]
    ) -> float:
        """Compute max mean within-cluster distance."""
        mean_dists = []
        for cluster in clusters:
            if len(cluster) < 2:
                mean_dists.append(0.0)
                continue
            sub = dist_array[np.ix_(cluster, cluster)]
            upper = np.triu(sub, k=1)
            mean_dists.append(np.mean(upper[upper > 0]) if np.any(upper > 0) else 0.0)
        return max(mean_dists) if mean_dists else 0.0

    def _optimize_clusters(
        self,
        X: np.ndarray,
        weights: np.ndarray,
        dist_array: np.ndarray,
        clusters: List[List[int]]
    ) -> Tuple[List[List[int]], float]:
        """Update centroids and reassign points."""
        centroid_idxs = self._update_centroids(X, clusters)
        new_clusters = self._assign_to_clusters(weights, dist_array, centroid_idxs)
        cost = self._max_mean_dist(dist_array, new_clusters)
        return new_clusters, cost

    def _update_centroids(
        self,
        X: np.ndarray,
        clusters: List[List[int]]
    ) -> List[int]:
        """Find point closest to cluster center."""
        centroids = []
        for cluster in clusters:
            center = np.mean(X[cluster], axis=0)
            distances = np.linalg.norm(X[cluster] - center, axis=1)
            closest = cluster[np.argmin(distances)]
            centroids.append(closest)
        return centroids
