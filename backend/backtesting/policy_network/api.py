"""API wrapper for Policy Network strategy - maintains compatibility with main.py."""

from typing import Dict, Tuple
import pandas as pd

from .strategy import PolicyNetworkStrategy
from ..config import PolicyNetworkConfig


def run(
    prices: pd.DataFrame,
    lookback: int = 252,
    eval_win: int = 5,
    eta: float = 0.02,  # unused but kept for API compatibility
    tc: float = 0.002,
    device: str = "cpu",
    write_files: bool = False,
    tag: str | None = None,
) -> Tuple[pd.Series, Dict[str, float], Dict[str, float]]:
    """
    Run Policy Network backtest.

    This function provides the standard API expected by main.py.

    Args:
        prices: Price DataFrame with assets as columns
        lookback: Window size for network input
        eval_win: Rebalancing frequency in days
        eta: Unused (API compatibility)
        tc: Transaction cost rate
        device: 'cpu' or 'cuda'
        write_files: Ignored (legacy parameter)
        tag: Ignored (legacy parameter)

    Returns:
        (nav_series, final_weights, metrics)
    """
    config = PolicyNetworkConfig(
        window_size=lookback,
        epochs=50,
    )
    strategy = PolicyNetworkStrategy(config, device=device)

    return strategy.backtest(
        prices=prices,
        lookback=lookback,
        eval_win=eval_win,
        eta=eta,
        tc=tc,
    )
