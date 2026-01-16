"""API wrapper for Margin Trader strategy - maintains compatibility with main.py."""

from typing import Dict, Tuple

import pandas as pd

from .strategy import MarginTraderStrategy
from ..config import MarginTraderConfig


def run(
    prices: pd.DataFrame,
    lookback: int = 252,
    eval_win: int = 5,
    eta: float = 0.02,
    tc: float = 0.002,
    total_steps: int = 20_000,
    **kwargs,
) -> Tuple[pd.Series, Dict[str, float], Dict[str, float]]:
    """Run Margin Trader backtest.

    Args:
        prices: Price DataFrame with assets as columns
        lookback: Days for train/trade split
        eval_win: Unused (API compatibility)
        eta: Unused (API compatibility)
        tc: Transaction cost rate
        total_steps: Training timesteps

    Returns:
        (nav_series, final_weights, metrics)
    """
    config = MarginTraderConfig()
    strategy = MarginTraderStrategy(config)

    return strategy.backtest(
        prices=prices,
        lookback=lookback,
        eval_win=eval_win,
        eta=eta,
        tc=tc,
        total_steps=total_steps,
    )
