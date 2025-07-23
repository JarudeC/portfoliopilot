"""
CA-GMVP (clustered GMVP) · v8  (22 Jul 2025)
────────────────────────────────────────────────────────────────────────────
• Library call:   nav, w, m = run(prices_df, lookback=252, eval_win=5, …)
• CLI utility:    python train.py --tickers AAPL MSFT --hist 730
                  python train.py --tickers AAPL MSFT --start 2015-01-01
"""

from __future__ import annotations

import sys, argparse, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

HERE     = Path(__file__).resolve()
BACKEND  = HERE.parents[2]        # backend/
RESULTS  = HERE.parent / "results"
sys.path.append(str(BACKEND))     # so `models` is importable

from models.GMVP_Clustering.model import CAMinVar                     # noqa: E402
from models.GMVP_Clustering.record import compute_metrics_from_nav    # noqa: E402

# ─── hyper-params (defaults) ──────────────────────────────────────────────
VERBOSE_EVERY  = 1
BACKLOOK       = 252           # rolling window (days)
EVAL_WINDOW    = 5             # rebalance freq  (days)
CLIP_LIMIT     = 0.30          # ±30 % cap on fwd returns
CLEAN_NA_TOL   = 0.20          # drop cols with >20 % NaNs
TC_RATE        = 0.002         # 20 bp round-trip cost
# ──────────────────────────────────────────────────────────────────────────


def _good_cols(look: pd.DataFrame) -> pd.Index:
    """Filter out columns with too many gaps / extreme moves."""
    bad = (
        (look.isna().mean() > CLEAN_NA_TOL)
        | look.isna().any()
        | np.isinf(look).any()
        | (np.abs(look) >= 1).any()
    )
    return look.columns[~bad]


# ╭─────────────────────── Public API ─────────────────────────╮ #
def run(
    prices: pd.DataFrame,
    lookback: int = BACKLOOK,
    eval_win: int = EVAL_WINDOW,
    clusters: int = 12,
    max_cluster: int = 80,
    tc: float = TC_RATE,
    write_files: bool = False,
    tag: str | None = None,
) -> Tuple[pd.Series, Dict[str, float], Dict[str, float]]:
    """
    Execute the CA-GMVP back-test on an **adjusted-close price** DataFrame.
    Returns NAV, final weights, and performance metrics.
    """
    rets_all = prices.sort_index().pct_change().dropna(how="all")
    rets_all = rets_all.replace([np.inf, -np.inf], np.nan)

    model = CAMinVar(
        rebalance_freq=eval_win,
        lookback=lookback,
        n_clusters=clusters,
        max_cluster_size=max_cluster,
    )

    nav = [1.0]
    ret_chunks: List[pd.Series] = []
    wlog: Dict[pd.Timestamp, pd.Series] = {}
    w_prev = None

    steps = range(lookback, len(rets_all) - eval_win, eval_win)
    total_steps = len(list(steps))

    for k, t0 in enumerate(steps, 1):
        look = rets_all.iloc[t0 - lookback : t0]
        cols = _good_cols(look)
        if cols.empty:
            continue
        look = look[cols]

        try:
            if model.n_clusters > look.shape[1]:
                model.n_clusters = look.shape[1]
            model.fit(look)
            w = model.predict_weights().loc[cols]
            # keep long-only, but don't drop everything to empty
            w = w.clip(lower=0)
            if w.sum() <= 0:
                raise RuntimeError
            w = w / w.sum()
            print("spread:", (w.max() - w.min()).round(6))
        except RuntimeError:
            w = pd.Series(1 / len(cols), index=cols)
            print("EW fallback:", rets_all.index[t0].date())

        # forward block
        block = (
            rets_all.iloc[t0 : t0 + eval_win][cols]
            .fillna(0)
            .clip(-CLIP_LIMIT, CLIP_LIMIT)
        )
        common = block.columns.intersection(w.index)
        w = w[common] / w[common].sum()
        wlog[rets_all.index[t0]] = w

        drets = block[common] @ w
        if w_prev is None:
            turnover = float(w.abs().sum())        # initial buy
            drets.iloc[0] -= tc * turnover
        else:
            all_idx = w.index.union(w_prev.index)
            turnover = (
                w.reindex(all_idx, fill_value=0)
                - w_prev.reindex(all_idx, fill_value=0)
            ).abs().sum()
            drets.iloc[0] -= tc * turnover
        w_prev = w

        ret_chunks.append(drets)
        nav.extend(((1 + drets).cumprod() * nav[-1]).values)

        if VERBOSE_EVERY and k % VERBOSE_EVERY == 0:
            last = drets.iloc[-1]
            print(f"[{k:>3}/{total_steps}] tickers={len(cols):3d}   r={last:+.4f}",
                  flush=True)

    nav_series = pd.Series(nav, index=rets_all.index[: len(nav)], name="NAV")
    metrics = compute_metrics_from_nav(nav_series)
    final_weights = w_prev.to_dict() if w_prev is not None else {}

    if write_files:
        _dump_csvs(nav_series, ret_chunks, wlog, clusters, max_cluster, tag)

    return nav_series, final_weights, metrics
# ╰─────────────────────────────────────────────────────────────╯


def _dump_csvs(
    nav: pd.Series,
    ret_chunks: List[pd.Series],
    wlog: Dict[pd.Timestamp, pd.Series],
    clusters: int,
    max_cluster: int,
    tag: str | None,
) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    suffix = f"cl{clusters}_max{max_cluster}" + (f"_{tag}" if tag else "")

    pd.concat(ret_chunks).rename("Return").to_frame()\
        .to_csv(RESULTS / f"Overall_return_CA_GMVP_{suffix}.csv", index=False)

    nav.to_csv(RESULTS / f"pnl_CA_GMVP_{suffix}.csv")

    pd.concat(wlog, axis=1).T.to_csv(RESULTS / f"Weights_log_{suffix}.csv")


# ╭──────────────────────── CLI helper ─────────────────────────╮ #
def _cli() -> None:
    import yfinance as yf

    p = argparse.ArgumentParser("CA-GMVP CLI  (yfinance)")
    p.add_argument("--tickers", nargs="+", required=True,
                   help="e.g. --tickers AAPL MSFT NVDA")
    p.add_argument("--hist", type=int, default=None,
                   help="history window in days (alternative to --start/--end)")
    p.add_argument("--start", type=str, default=None,
                   help="start date YYYY-MM-DD (ignored if --hist given)")
    p.add_argument("--end",   type=str, default=None,
                   help="end   date YYYY-MM-DD (defaults = today)")
    p.add_argument("--lookback", type=int, default=BACKLOOK)
    p.add_argument("--eval",     type=int, default=EVAL_WINDOW)
    p.add_argument("--clusters", type=int, default=12)
    p.add_argument("--max_cluster", type=int, default=80)
    p.add_argument("--tc", type=float, default=TC_RATE)
    p.add_argument("--no-files", action="store_true",
                   help="skip writing CSVs")
    args = p.parse_args()

    if args.hist:
        df = yf.download(
            " ".join(args.tickers),
            period=f"{args.hist}d",
            interval="1d",
            auto_adjust=True,
        )["Close"]
    else:
        end = args.end or datetime.date.today().isoformat()
        df = yf.download(
            " ".join(args.tickers),
            start=args.start,
            end=end,
            interval="1d",
            auto_adjust=True,
        )["Close"]

    if isinstance(df.columns, pd.MultiIndex):    # single-level columns
        df.columns = df.columns.get_level_values(0)

    nav, w, m = run(
        df,
        lookback=args.lookback,
        eval_win=args.eval,
        clusters=args.clusters,
        max_cluster=args.max_cluster,
        tc=args.tc,
        write_files=not args.no_files,
    )

    print("\nFinal NAV :", f"{nav.iloc[-1]:.4f}")
    print("Weights   :", {k: f"{v:.3%}" for k, v in w.items()})
    print("Metrics   :", m)


if __name__ == "__main__":
    if len(sys.argv) == 1:          # protect against accidental double-click
        _cli()