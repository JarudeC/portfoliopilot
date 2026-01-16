"""API wrapper for GMVP Clustering strategy - maintains compatibility with main.py."""

from typing import Dict, Tuple
import pandas as pd

from .strategy import GMVPClusteringStrategy
from ..config import GMVPClusteringConfig


def run(
    prices: pd.DataFrame,
    lookback: int = 252,
    eval_win: int = 5,
    tc: float = 0.002,
    clusters: int = 12,
    max_cluster: int = 80,
    write_files: bool = False,
    tag: str | None = None,
    # Accept eta for API compatibility even though GMVP doesn't use it
    eta: float = 0.0,
) -> Tuple[pd.Series, Dict[str, float], Dict[str, float]]:
    """
    Run GMVP Clustering backtest.

    This function provides the standard API expected by main.py.

    Args:
        prices: Price DataFrame with assets as columns
        lookback: Rolling window for parameter estimation
        eval_win: Rebalancing frequency in days
        tc: Transaction cost rate
        clusters: Number of asset clusters
        max_cluster: Maximum assets per cluster
        write_files: Ignored (legacy parameter)
        tag: Ignored (legacy parameter)
        eta: Ignored (API compatibility)

    Returns:
        (nav_series, final_weights, metrics)
    """
    config = GMVPClusteringConfig(
        n_clusters=clusters,
        max_clusters=max_cluster,
    )
    strategy = GMVPClusteringStrategy(config)

    return strategy.backtest(
        prices=prices,
        lookback=lookback,
        eval_win=eval_win,
        eta=eta,
        tc=tc,
        n_clusters=clusters,
        max_cluster=max_cluster,
    )
