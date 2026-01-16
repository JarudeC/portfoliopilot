"""GMVP optimization with Ledoit-Wolf shrinkage and clustering."""

import logging
import math
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

from .clustering import BoundedKMeansClustering

logger = logging.getLogger(__name__)

EPS = 1e-5  # Ridge term for covariance stabilization


def gmvp_weights(cov: np.ndarray) -> np.ndarray:
    """
    Compute Global Minimum Variance Portfolio weights.

    Args:
        cov: Covariance matrix

    Returns:
        Weight vector (sums to 1)
    """
    n = cov.shape[0]
    cov_reg = cov + EPS * np.eye(n)
    ones = np.ones(n)

    try:
        inv = np.linalg.pinv(cov_reg)
        w = inv @ ones
        denom = ones @ inv @ ones

        if denom == 0 or not np.isfinite(denom):
            raise ValueError("Invalid denominator")

        w = w / denom
    except (np.linalg.LinAlgError, ValueError):
        logger.warning("GMVP optimization failed, using equal weights")
        w = np.full(n, 1 / n)

    return w


def compute_clustered_weights(
    returns: pd.DataFrame,
    n_clusters: int,
    max_cluster_size: int,
    use_shrinkage: bool = True,
    random_state: int = 42
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute GMVP weights using two-level clustering approach.

    1. Cluster assets using bounded K-means
    2. Compute GMVP weights within each cluster
    3. Compute GMVP weights across cluster portfolios
    4. Combine to get final asset weights

    Args:
        returns: Return DataFrame (rows=dates, cols=tickers)
        n_clusters: Number of asset clusters
        max_cluster_size: Maximum assets per cluster
        use_shrinkage: Use Ledoit-Wolf covariance shrinkage
        random_state: Random seed for clustering

    Returns:
        (weights, tickers) where weights[i] corresponds to tickers[i]
    """
    # Clean returns
    returns = (
        returns.replace([np.inf, -np.inf], np.nan)
        .ffill()
        .bfill()
        .dropna(axis=1, how="any")
    )

    if returns.empty:
        raise ValueError("No valid tickers after cleaning")

    tickers = returns.columns.tolist()
    n_assets = len(tickers)

    # Prepare features for clustering (transpose: assets as rows)
    X = returns.T.values
    X = StandardScaler().fit_transform(X)

    # Adjust cluster count if needed
    actual_clusters = max(n_clusters, math.ceil(n_assets / max_cluster_size))
    actual_clusters = min(actual_clusters, n_assets)

    # Run bounded K-means
    bkm = BoundedKMeansClustering(
        n_clusters=actual_clusters,
        max_cluster_size=max_cluster_size,
        n_iter=30,
        n_init=10,
        random_state=random_state
    )
    _, clusters = bkm.fit(X, np.ones(n_assets))

    if not clusters or all(len(c) == 0 for c in clusters):
        logger.warning("Clustering failed, using equal weights")
        return np.full(n_assets, 1 / n_assets), tickers

    # Compute inner weights for each cluster
    inner_weights = {}
    cluster_returns = []

    for label, indices in enumerate(clusters):
        if not indices:
            continue

        sub_returns = returns.iloc[:, indices]

        if use_shrinkage:
            cov = LedoitWolf().fit(sub_returns.values).covariance_
        else:
            cov = sub_returns.cov().values

        w = gmvp_weights(cov)
        inner_weights[label] = (indices, w)

        # Cluster portfolio returns
        cluster_returns.append(sub_returns.values @ w)

    if not cluster_returns:
        logger.warning("No valid clusters, using equal weights")
        return np.full(n_assets, 1 / n_assets), tickers

    # Compute outer weights across cluster portfolios
    cluster_df = pd.DataFrame(cluster_returns).T
    outer_cov = cluster_df.cov().values
    outer_w = gmvp_weights(outer_cov)

    # Combine inner and outer weights
    full_weights = np.zeros(n_assets)
    for label, ow in enumerate(outer_w):
        if label in inner_weights:
            indices, iw = inner_weights[label]
            full_weights[indices] = iw * ow

    # Ensure long-only and normalized
    full_weights = np.clip(full_weights, 0, None)
    total = full_weights.sum()
    if total > 0:
        full_weights = full_weights / total
    else:
        full_weights = np.full(n_assets, 1 / n_assets)

    return full_weights, tickers
