# Naive Markowitz portfolio optimization API wrapper
# Provides uniform interface for FastAPI integration

from __future__ import annotations
from typing import Tuple, Dict
import pandas as pd

# Import main implementation
from .train import run as _run


def run(
    prices: pd.DataFrame,
    lookback: int = 252,
    eval_win: int = 5,
    eta: float = 0.02,
    tc: float = 0.002,
    write_files: bool = False,
    tag: str | None = None,
) -> Tuple[pd.Series, Dict[str, float], Dict[str, float]]:
    """
    Wrapper so every model exposes the *same* callable.

    See models.NaiveMarkowitz.train.run for full docstring.
    """
    return _run(
        prices=prices,
        lookback=lookback,
        eval_win=eval_win,
        eta=eta,
        tc=tc,
        write_files=write_files,
        tag=tag,
    )
