# models/MarginTrader/api.py
from __future__ import annotations
from typing import Tuple, Dict
import pandas as pd

from .train import run as _run

def run(
    prices: pd.DataFrame,
    lookback: int    = 252,
    eval_win: int    = 5,
    eta: float       = 0.02,   # ignored
    tc: float        = 0.002,
    total_steps: int = 20_000,
    **kwargs,
) -> Tuple[pd.Series, Dict[str, float], Dict[str, str]]:
    return _run(
        prices      = prices,
        lookback    = lookback,
        eval_win    = eval_win,
        eta         = eta,
        tc          = tc,
        total_steps = total_steps,
    )