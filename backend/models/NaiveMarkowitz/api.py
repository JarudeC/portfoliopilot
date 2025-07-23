# models/NaiveMarkowitz/api.py
"""
Thin façade for FastAPI (and any other caller).

Why a separate file?
────────────────────
• Keeps a **uniform interface** across *all* algorithms:
      from models.<Algo>.api import run
  Every model folder (GMVP, CA-GMVP, MarginTrader…) will expose the very
  same `run()` signature, so the FastAPI dispatcher can stay generic.

• Avoids re-import loops: FastAPI only needs this tiny wrapper; the heavy
  stuff in train.py is imported lazily when `run()` is called.
"""

from __future__ import annotations
from typing import Tuple, Dict
import pandas as pd

# single-line re-export ↓  – all real work lives in train.py
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
