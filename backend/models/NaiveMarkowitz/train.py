"""
Naive-Markowitz · API + CLI  ·  v3  (22 Jul 2025)

Usage
-----
# ➊ As a library function (recommended)
>>> from models.NaiveMarkowitz.train import run
>>> nav, weights, metrics = run(prices=df, lookback=252, eval_win=5)

# ➋ As a standalone script (legacy)
$ python train.py --tickers AAPL MSFT NVDA --hist 730
  ⟹ fetches prices with yfinance, runs model, prints summary,
     and drops the usual CSVs in models/NaiveMarkowitz/results
"""

from __future__ import annotations
import numpy as np, pandas as pd, argparse, warnings, sys
from pathlib import Path
from typing import Tuple, Dict, List

from .model import Naive_Markowitz               # original class
from .record import compute_metrics_from_nav

warnings.filterwarnings("ignore")
np.random.seed(43)

# ──────────────────────────────────────────────────────────────────────────
# Core callable
# ──────────────────────────────────────────────────────────────────────────
def run(
    prices: pd.DataFrame,
    lookback: int               = 252,
    eval_win: int               = 5,
    eta: float                  = 0.02,
    tc: float                   = 0.002,
    write_files: bool           = False,
    tag: str | None             = None,
) -> Tuple[pd.Series, Dict[str,float], Dict[str,float]]:
    """
    Execute the Naive-Markowitz back-test.

    Parameters
    ----------
    prices      : (T, N) adjusted close price DataFrame (index = date)
    lookback    : rolling window length (days)
    eval_win    : holding / rebalance period (days)
    eta         : risk-aversion hyper-parameter
    tc          : round-trip transaction-cost rate
    write_files : if True, dumps NAV / returns / turnovers CSVs
    tag         : extra suffix for filenames (optional)

    Returns
    -------
    nav         : Series of cumulative NAV (starts at 1.0)
    weights     : dict {ticker: final_weight}
    metrics     : dict {Sharpe, CAGR, maxDD, …}  via utils.record
    """
    hist = prices.dropna().copy()
    dates = hist.index.to_series()

    first_sig = lookback
    num_win   = (len(hist) - first_sig) // eval_win
    look      = [0, lookback]

    nav          = [1.0]
    turnovers    = []
    daily_rets   = []
    w_prev       = None

    for step in range(1, num_win + 1):
        mdl = Naive_Markowitz(
            historical_data       = hist,
            lookback_window       = look,
            evaluation_window     = eval_win,
            eta                   = eta,
            transaction_cost_rate = tc
        )

        w_new = mdl.weights().values
        turnover = 1.0 if w_prev is None else abs(w_new - w_prev).sum()
        w_prev   = w_new.copy()
        turnovers.append(turnover)
        cost      = turnover * tc

                # ── forward returns & NAV segment ──────────────────────────────
        fr = mdl.forward_returns()
        s  = (fr["return"] if isinstance(fr, pd.DataFrame) and "return" in fr
              else fr.squeeze())
        r_fwd = s.copy()
        r_fwd.iloc[0] -= cost

        # ① clean / cap returns  .......................................
        CAP = 0.1
        r_fwd = (
            r_fwd.replace([np.inf, -np.inf], np.nan)   # drop rogue values
                 .fillna(0)
                 .clip(lower=-CAP, upper=CAP)          # ±50 % hard cap
        )
        daily_rets.append(r_fwd)

        # ② accumulate in **log space** to avoid overflow .............
        nav_last   = nav[-1]
        nav_last  *= np.exp(np.log1p(r_fwd).sum())      # single exp, no cumprod
        nav.append(nav_last)
        look = [look[0] + eval_win, look[1] + eval_win]

    nav_series = pd.Series(
        nav[1:],
        index = dates.iloc[first_sig : first_sig + len(nav) - 1],
        name  = "NAV"
    )
    weights = pd.Series(w_prev, index = hist.columns).to_dict()
    metrics = compute_metrics_from_nav(nav_series)

    # optional side-effect: drop legacy CSVs
    if write_files:
        _dump_csvs(nav_series, daily_rets, turnovers, eta, tag)

    return nav_series, weights, metrics


# ──────────────────────────────────────────────────────────────────────────
# Helper: optional CSV writers (keeps original filenames)
# ──────────────────────────────────────────────────────────────────────────
def _dump_csvs(nav: pd.Series,
               daily_rets: List[pd.Series],
               turnovers: List[float],
               eta: float,
               tag: str | None):

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix  = f"eta={eta}" + (f"_{tag}" if tag else "")

    # (1) Overall daily return series
    ret_series = pd.concat(daily_rets, ignore_index=True)
    ret_series.index = nav.index
    ret_series.name  = "Return"
    ret_series.reset_index()\
        .rename(columns={"index": "Date"})\
        .to_csv(out_dir / f"Overall_return_NaiveMarkowitz_{suffix}.csv",
                index=False)

    # (2) NAV
    pd.DataFrame({"Date": nav.index, "PortfolioValue": nav.values})\
        .to_csv(out_dir / f"pnl_NaiveMarkowitz_{suffix}.csv", index=False)

    # (3) Turnovers
    pd.DataFrame({"Turnover": turnovers})\
        .to_csv(out_dir / f"Turnovers_NaiveMarkowitz_{suffix}.csv", index=False)


# ──────────────────────────────────────────────────────────────────────────
# Optional CLI for ad-hoc runs (fetches yfinance instead of CSV)
# ──────────────────────────────────────────────────────────────────────────
def _cli():
    import yfinance as yf

    p = argparse.ArgumentParser("Naive-Markowitz CLI")
    p.add_argument("--tickers", nargs="+", required=True,
                   help="e.g. --tickers AAPL MSFT NVDA")
    p.add_argument("--hist", type=int, default=730,
                   help="history window (days) to download")
    p.add_argument("--lookback", type=int, default=252)
    p.add_argument("--eval", type=int, default=5)
    p.add_argument("--eta", type=float, default=0.02)
    p.add_argument("--tc",  type=float, default=0.002)
    p.add_argument("--no-files", action="store_true",
                   help="skip writing CSVs")
    args = p.parse_args()

    df = yf.download(
        " ".join(args.tickers),
        period=f"{args.hist}d", interval="1d", auto_adjust=True)["Close"]
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    nav, w, m = run(df,
                    lookback=args.lookback,
                    eval_win=args.eval,
                    eta=args.eta,
                    tc=args.tc,
                    write_files=not args.no_files)

    print("\nFinal NAV:", f"{nav.iloc[-1]:.4f}")
    print("Final weights:", w)
    print("Metrics:", m)


if __name__ == "__main__":
    if len(sys.argv) == 1:            # guard against accidental double-click
        print(__doc__)
        sys.exit(0)
    _cli()