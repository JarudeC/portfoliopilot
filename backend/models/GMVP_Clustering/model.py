"""
Clustering-Aided Global Minimum-Variance Portfolio (GMVP)

Changes – v2 (July 2025)
------------------------
* Treat the input `prices` frame as a **return matrix** that is already in
  `% daily change` form.  The extra `.pct_change()` step has been removed
  so the covariance is estimated on the intended data. :contentReference[oaicite:0]{index=0}
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from .bounded_Kmeans_clustering import BoundedKMeansClustering

EPS = 1e-5  # ridge term for covariance stabilisation


class CAMinVar:
    # ───────────────────────── initialisation ──────────────────────────
    def __init__(
        self,
        rebalance_freq: int = 5,
        lookback: int = 252,
        n_clusters: int = 12,
        max_cluster_size: int = 80,
        scale: bool = True,
        dimred: Literal["none", "pca", "tsne"] = "none",
        pca_k: int = 3,
        tsne_k: int = 3,
        shrink: bool = True,
        random_state: int = 42,
    ):
        self.rebalance_freq = rebalance_freq
        self.lookback = lookback
        self.n_clusters = n_clusters
        self.max_cluster_size = max_cluster_size
        self.scale = scale
        self.dimred = dimred
        self.pca_k = pca_k
        self.tsne_k = tsne_k
        self.shrink = shrink
        self.random_state = random_state
        self.weights_: pd.Series | None = None

    # ───────────────────────── utilities ───────────────────────────────
    @staticmethod
    def _gmvp_w(cov: np.ndarray) -> np.ndarray:
        """Global-minimum-variance weights with ridge & pseudo-inverse fallback."""
        n = cov.shape[0]
        cov = cov + EPS * np.eye(n)
        ones = np.ones(n)

        try:
            inv = np.linalg.pinv(cov)
            w = inv @ ones
            denom = ones @ inv @ ones
            if denom == 0 or not np.isfinite(denom):
                raise ZeroDivisionError
            w /= denom
        except (np.linalg.LinAlgError, ZeroDivisionError):
            w = np.full(n, 1 / n)  # equal-weight fallback
        return w

    # ───────────────────────── clustering ──────────────────────────────
    def _bounded_kmeans(self, X: np.ndarray) -> list[list[int]]:
        n = X.shape[0]
        k = max(self.n_clusters, math.ceil(n / self.max_cluster_size))
        bkm = BoundedKMeansClustering(
            n_clusters=k,
            max_cluster_size=self.max_cluster_size,
            n_iter=30,
            n_init=10,
            plot_every_iteration=False,
        )
        _, clusters = bkm.fit(X, np.ones(n))
        return clusters

    # ───────────────────────── core API ────────────────────────────────
    def fit(self, prices: pd.DataFrame) -> "CAMinVar":
        """
        Parameters
        ----------
        prices
            **Daily return matrix** (rows = dates, cols = tickers).  The most
            recent `self.lookback` rows are used.
        """
        rets = prices.iloc[-self.lookback:]  # ← removed redundant .pct_change()
        rets = (
            rets.replace([np.inf, -np.inf], np.nan)
            .ffill()
            .bfill()
            .dropna(axis=1, how="any")
        )
        if rets.empty:
            raise RuntimeError("No usable tickers after cleaning.")

        X = rets.T.values
        if self.scale:
            X = StandardScaler().fit_transform(X)
        if self.dimred == "pca":
            X = PCA(self.pca_k, random_state=self.random_state).fit_transform(X)
        elif self.dimred == "tsne":
            X = TSNE(self.tsne_k, random_state=self.random_state).fit_transform(X)

        clusters = self._bounded_kmeans(X)
        tickers = rets.columns.to_numpy()

        inner_w = {}
        cluster_returns = []
        for lbl, idx in enumerate(clusters):
            sub = rets.iloc[:, idx]
            if sub.shape[1] == 0:
                continue
            cov = (
                LedoitWolf().fit(sub.values).covariance_
                if self.shrink
                else sub.cov().values
            )
            w = self._gmvp_w(cov)
            inner_w[lbl] = (idx, w)
            cluster_returns.append(sub @ w)

        if not cluster_returns:
            raise RuntimeError("All clusters empty after filtering.")

        outer_cov = pd.concat(cluster_returns, axis=1).cov().values
        outer_w = self._gmvp_w(outer_cov)

        full = np.zeros(len(tickers))
        for lbl, ow in enumerate(outer_w):
            if lbl in inner_w:
                idx, iw = inner_w[lbl]
                full[idx] = iw * ow

        full = np.clip(full, 0, None)
        full /= full.sum()
        self.weights_ = pd.Series(full, index=tickers, name="weight")
        return self

    def predict_weights(self) -> pd.Series:
        if self.weights_ is None:
            raise RuntimeError("Call fit() first.")
        return self.weights_
