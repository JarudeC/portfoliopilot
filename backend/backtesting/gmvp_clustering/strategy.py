"""GMVP Clustering portfolio strategy implementation."""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..base import BaseStrategy
from ..config import GMVPClusteringConfig
from ..metrics import compute_metrics_from_nav
from .optimizer import compute_clustered_weights

logger = logging.getLogger(__name__)

# Data cleaning thresholds
NA_TOLERANCE = 0.20    # Drop columns with >20% NaN
CLIP_LIMIT = 0.30      # ±30% return cap


def _good_cols(look: pd.DataFrame) -> pd.Index:
    """Filter out columns with too many gaps or extreme moves."""
    bad = (
        (look.isna().mean() > NA_TOLERANCE)
        | look.isna().any()
        | np.isinf(look).any()
        | (np.abs(look) >= 1).any()
    )
    return look.columns[~bad]


class GMVPClusteringStrategy(BaseStrategy):
    """
    Clustering-Aided Global Minimum Variance Portfolio.

    Groups assets using bounded K-means, computes GMVP within clusters,
    then combines using outer GMVP across cluster portfolios.
    Uses Ledoit-Wolf shrinkage for stable covariance estimation.
    """

    name = "GMVP Clustering"

    def __init__(self, config: GMVPClusteringConfig | None = None):
        self.config = config or GMVPClusteringConfig()

    def optimize(
        self,
        prices: pd.DataFrame,
        lookback: int,
        n_clusters: int = 12,
        max_cluster: int = 80,
        **kwargs
    ) -> np.ndarray:
        """
        Compute GMVP weights using clustering.

        Args:
            prices: Price DataFrame
            lookback: Days of history to use
            n_clusters: Number of asset clusters
            max_cluster: Maximum cluster size

        Returns:
            Portfolio weight vector
        """
        returns = prices.pct_change().dropna()
        returns = returns.iloc[-lookback:] if len(returns) > lookback else returns

        if len(returns) < 2:
            n = len(prices.columns)
            return np.ones(n) / n

        cols = _good_cols(returns)
        if cols.empty:
            n = len(prices.columns)
            return np.ones(n) / n

        returns = returns[cols]

        try:
            weights, tickers = compute_clustered_weights(
                returns,
                n_clusters=min(n_clusters, len(cols)),
                max_cluster_size=max_cluster,
                use_shrinkage=self.config.shrinkage,
                random_state=self.config.random_seed
            )

            # Map back to original columns
            full_weights = np.zeros(len(prices.columns))
            for i, col in enumerate(prices.columns):
                if col in tickers:
                    idx = tickers.index(col)
                    full_weights[i] = weights[idx]

            total = full_weights.sum()
            if total > 0:
                full_weights = full_weights / total

            return full_weights

        except (ValueError, RuntimeError):
            n = len(prices.columns)
            return np.ones(n) / n

    def backtest(
        self,
        prices: pd.DataFrame,
        lookback: int,
        eval_win: int,
        eta: float,
        tc: float,
        n_clusters: int = 12,
        max_cluster: int = 80,
        **kwargs
    ) -> Tuple[pd.Series, Dict[str, float], Dict[str, float]]:
        """
        Run rolling-window backtest.

        Args:
            prices: Historical price DataFrame
            lookback: Rolling window for estimation
            eval_win: Days between rebalancing
            eta: Unused (API compatibility)
            tc: Transaction cost rate
            n_clusters: Number of asset clusters
            max_cluster: Maximum cluster size

        Returns:
            (nav_series, final_weights, metrics)
        """
        # Convert prices to returns
        rets_all = prices.sort_index().pct_change().dropna(how="all")
        rets_all = rets_all.replace([np.inf, -np.inf], np.nan)

        nav: List[float] = [1.0]
        w_prev = None

        steps = range(lookback, len(rets_all) - eval_win, eval_win)

        for t0 in steps:
            look = rets_all.iloc[t0 - lookback:t0]
            cols = _good_cols(look)

            if cols.empty:
                continue

            look = look[cols]

            # Compute weights
            try:
                actual_clusters = min(n_clusters, len(cols))
                weights, tickers = compute_clustered_weights(
                    look,
                    n_clusters=actual_clusters,
                    max_cluster_size=max_cluster,
                    use_shrinkage=self.config.shrinkage,
                    random_state=self.config.random_seed
                )
                w = pd.Series(weights, index=tickers)
                w = w.clip(lower=0)
                if w.sum() <= 0:
                    raise RuntimeError("All weights zero")
                w = w / w.sum()

            except (ValueError, RuntimeError):
                w = pd.Series(1 / len(cols), index=cols)

            # Forward evaluation block
            block = (
                rets_all.iloc[t0:t0 + eval_win][cols]
                .fillna(0)
                .clip(-CLIP_LIMIT, CLIP_LIMIT)
            )

            common = block.columns.intersection(w.index)
            w = w[common] / w[common].sum()

            # Portfolio returns
            drets = block[common] @ w

            # Transaction costs
            if w_prev is None:
                turnover = float(w.abs().sum())
            else:
                all_idx = w.index.union(w_prev.index)
                turnover = (
                    w.reindex(all_idx, fill_value=0)
                    - w_prev.reindex(all_idx, fill_value=0)
                ).abs().sum()

            drets.iloc[0] -= tc * turnover
            w_prev = w

            # Update NAV
            nav.extend(((1 + drets).cumprod() * nav[-1]).values)

        # Build NAV series
        nav_series = pd.Series(
            nav,
            index=rets_all.index[lookback:lookback + len(nav)],
            name="NAV"
        )

        # Final weights
        final_weights = w_prev.to_dict() if w_prev is not None else {}

        # Metrics
        metrics = compute_metrics_from_nav(nav_series)

        logger.info(
            "%s backtest complete: %d periods, final NAV=%.4f",
            self.name, len(nav_series), nav_series.iloc[-1] if len(nav_series) > 0 else 1.0
        )

        return nav_series, final_weights, metrics
