# models/GMVP_Clustering/api.py
from __future__ import annotations
from typing import Tuple, Dict
import pandas as pd

from .train import run as _run     # your v8 train.run()

def run(
    prices: pd.DataFrame,
    lookback: int = 252,
    eval_win: int = 5,
    eta: float = 0.02,       
    clusters: int = 12,
    max_cluster: int = 80,
    tc: float = 0.002,
    write_files: bool = False,
    tag: str | None = None,
) -> Tuple[pd.Series, Dict[str, float], Dict[str, float]]:
    """Uniform façade for FastAPI – silently ignores `eta`."""
    return _run(
        prices       = prices,
        lookback     = lookback,
        eval_win     = eval_win,
        clusters     = clusters,
        max_cluster  = max_cluster,
        tc           = tc,
        write_files  = write_files,
        tag          = tag,
    )