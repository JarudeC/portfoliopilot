"""API wrapper for Naive Markowitz strategy - maintains compatibility with main.py."""

from typing import Dict, Tuple
import pandas as pd

from .strategy import NaiveMarkowitzStrategy
from ..config import NaiveMarkowitzConfig


def run(
    prices: pd.DataFrame,
    lookback: int = 252,
    eval_win: int = 5,
    eta: float = 0.02,
    tc: float = 0.002,
    write_files: bool = False,  # legacy, unused
    tag: str | None = None,     # legacy, unused
) -> Tuple[pd.Series, Dict[str, float], Dict[str, float]]:
    """
    Run Naive Markowitz backtest.

    Args:
        prices: Price DataFrame with assets as columns
        lookback: Rolling window for parameter estimation
        eval_win: Rebalancing frequency in days
        eta: Noise parameter for expected returns
        tc: Transaction cost rate

    Returns:
        (nav_series, final_weights, metrics)
    """
    _ = write_files, tag  # Suppress unused warnings
    config = NaiveMarkowitzConfig()
    strategy = NaiveMarkowitzStrategy(config)

    return strategy.backtest(
        prices=prices,
        lookback=lookback,
        eval_win=eval_win,
        eta=eta,
        tc=tc,
    )
