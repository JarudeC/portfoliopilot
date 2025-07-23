# models/PortfolioPolicyNetwork/api.py
from __future__ import annotations
from typing import Tuple, Dict
import pandas as pd

from .train import run as _run   # re-export

def run(
    prices: pd.DataFrame,
    lookback: int = 252,
    eval_win: int = 5,
    eta: float = 0.02,     # kept for uniform signature (ignored)
    tc: float = 0.002,
    write_files: bool = False,
    tag: str | None = None,
):
    return _run(
        prices       = prices,
        lookback     = lookback,
        eval_win     = eval_win,
        eta          = eta,
        tc           = tc,
        write_files  = write_files,
        tag          = tag,
    )
