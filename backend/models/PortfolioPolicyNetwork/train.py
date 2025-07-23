"""
Portfolio Policy Network back-test  ·  v4  (22 Jul 2025)
────────────────────────────────────────────────────────────────────────────
• Library call:  nav, w, m = run(prices_df, lookback=252, eval_win=5, tc=0.002)
• CLI helper:    python train.py --tickers AAPL MSFT NVDA --hist 730
"""

from __future__ import annotations

import torch
torch.autograd.set_detect_anomaly(True)

import argparse, datetime, sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

HERE   = Path(__file__).resolve().parent
ROOT   = HERE.parents[2]                  # project root
RESULT = HERE / "results"; RESULT.mkdir(exist_ok=True)
sys.path.append(str(ROOT / "backend"))    # for utils if needed

from models.PortfolioPolicyNetwork.model import PortfolioPolicyNetwork  # noqa: E402
from models.PortfolioPolicyNetwork.record import compute_metrics        # noqa: E402

# ─── defaults / hyper-params ──────────────────────────────────────────────
BACKLOOK     = 252          # look-back window
EVAL_WINDOW  = 5            # holding period
TC_RATE      = 0.002        # one-way commission
DEVICE       = "cpu"        # "cuda" if model supports it
VERBOSE_EVERY = 20
# ──────────────────────────────────────────────────────────────────────────


# ╭────────────────────── Public API entry point ───────────────────────╮ #
def run(
    prices: pd.DataFrame,
    lookback: int = BACKLOOK,
    eval_win: int = EVAL_WINDOW,
    eta: float = 0.02,            # kept for uniform FastAPI signature (unused)
    tc: float = TC_RATE,
    device: str = DEVICE,
    write_files: bool = False,
    tag: str | None = None,
) -> Tuple[pd.Series, Dict[str, float], Dict[str, float]]:
    """
    Back-test PPN on **adjusted-close prices**.
    Returns
    -------
    nav      : pd.Series  – Net-asset-value (starts at 1.0)
    weights  : dict       – Final portfolio weights
    metrics  : dict       – Performance metrics (Sharpe, Sortino, …)
    """
    prices = prices.sort_index()                    # assure monotonic
    tickers = prices.columns.tolist()
    dates  = prices.index.to_numpy()
    prices_np = prices.to_numpy(dtype=np.float32)   # (T, N)
    prices_np = prices_np[:, :, None]               # (T, N, 1)

    print(f"DEBUG: Original prices shape: {prices.shape}")
    print(f"DEBUG: Tickers: {tickers}")
    print(f"DEBUG: Number of tickers: {len(tickers)}")
    print(f"DEBUG: prices_np shape: {prices_np.shape}")
    print(f"DEBUG: n_assets will be: {prices_np.shape[1]}")

    n_assets = prices_np.shape[1]
    model    = PortfolioPolicyNetwork(
                   lookback=lookback,
                   n_assets=n_assets,
                   device=device)

    # ── initial portfolio state ────────────────────────────────────────
    w_prev   = np.full(n_assets, 1 / n_assets, dtype=np.float32)
    log_nav  = 0.0
    nav      = [1.0]
    rlog: List[float] = []
    turns: List[float] = []

    # ── main loop ──────────────────────────────────────────────────────
    for step, t0 in enumerate(range(lookback, len(prices_np), eval_win), 1):
        hist = prices_np[t0 - lookback : t0]           # (lookback, N, 1)
        print(f"DEBUG Step {step}: hist shape: {hist.shape}")
        print(f"DEBUG Step {step}: w_prev shape: {w_prev.shape}")
        print(f"DEBUG Step {step}: About to call model.predict...")

        try:
            w_t = model.predict(hist, w_prev)
            if np.isnan(w_t).any():
                raise RuntimeError(f"NaN in w_t at step {step}")
            print(f"DEBUG Step {step}: w_t shape: {w_t.shape}")
        except Exception as e:
            print(f"ERROR in step {step}: {e}")
            print(f"hist.shape: {hist.shape}")
            print(f"w_prev.shape: {w_prev.shape}")
            print(f"w_prev: {w_prev}")
            raise
        x_batch = np.transpose(hist, (2, 1, 0))[None, ...]                # (1,1,R,T)
        y_next  = (prices_np[t0, :, 0] / prices_np[t0 - 1, :, 0]).astype(np.float32)
        y_next  = np.nan_to_num(y_next, nan=1.0, posinf=1.0, neginf=1.0)  # safe ratios
        assert w_prev.shape[0] == prices_np.shape[1], "w_prev len changed"
        model._agent.train_batch(x_batch, y_next[None, :], w_prev[None, :])

        w_t = model.predict(hist, w_prev)  # (N,) - already normalized in model.predict()
        print(f"DEBUG Step {step}: Model output w_t = {w_t}")
        # Only sanitize for safety, don't renormalize since model.predict() already does this correctly
        w_t = np.nan_to_num(w_t, nan=0.0, posinf=0.0, neginf=0.0)
        # Ensure non-negative (long-only constraint)
        if w_t.min() < 0:
            w_t = np.maximum(w_t, 0.0)
            # Only renormalize if we had to clip negative weights
            s = float(w_t.sum())
            if s > 0:
                w_t = w_t / s
            else:
                w_t = np.full(n_assets, 1 / n_assets, dtype=np.float32)
        print(f"DEBUG Step {step}: Final w_t = {w_t}, sum = {w_t.sum()}")

        turn = float(np.sum(np.abs(w_t - w_prev)))
        if step <= 3:
            print("ptp:", np.ptp(w_t), "turn:", f"{turn:.6f}")                                               
        # leverage cap 1× (kept from your original code)
        lev = np.sum(np.abs(w_t))
        if lev > 1.0:
            w_t = w_t / lev

        turn = float(np.sum(np.abs(w_t - w_prev)))
        turns.append(turn)
        w_prev = w_t.copy()

        # simulate over holding window
        for d in range(eval_win):
            if t0 + d >= len(prices_np):
                break

            step_ret = np.dot(
                w_prev,
                prices_np[t0 + d, :, 0] / prices_np[t0 + d - 1, :, 0] - 1.0,
            )
            fee   = tc * turn if d == 0 else 0.0
            r_net = step_ret - fee

            rlog.append(r_net)
            log_nav += np.log1p(r_net)
            nav.append(np.exp(log_nav))

        if VERBOSE_EVERY and step % VERBOSE_EVERY == 0:
            print(f"[{step:>4}]  nav={nav[-1]:.4f}  "
                  f"mean r={np.mean(rlog[-eval_win:]):+.5f}  "
                  f"turn={turn:.3f}", flush=True)

        w_prev = w_t.copy()

    # ── package outputs ───────────────────────────────────────────────
    nav_dates = dates[lookback - 1 : lookback - 1 + len(nav)]
    nav_series = pd.Series(nav, index=nav_dates, name="NAV")

    metrics = compute_metrics(nav_series)
    final_weights = dict(zip(tickers, map(float, w_prev)))

    if write_files:
        _dump_csvs(nav_series, rlog, tag)

    return nav_series, final_weights, metrics
# ╰─────────────────────────────────────────────────────────────────────╯


def _dump_csvs(nav: pd.Series, rets: List[float], tag: str | None):
    suffix = f"_PPN{('_'+tag) if tag else ''}"
    # NAV
    nav.to_frame("PortfolioValue").reset_index()\
        .rename(columns={"index": "Date"})\
        .to_csv(RESULT / f"pnl{suffix}.csv", index=False)
    # Returns
    pd.DataFrame({
        "Date":   nav.index[1:],     # first NAV (1.0) has no return
        "Return": rets,
    }).to_csv(RESULT / f"Overall_return{suffix}.csv", index=False)
    print("PPN CSVs written to", RESULT)


# ╭──────────────────────────── CLI helper ─────────────────────────────╮ #
def _cli():
    import yfinance as yf

    p = argparse.ArgumentParser("PPN CLI (yfinance)")
    p.add_argument("--tickers", nargs="+", required=True)
    p.add_argument("--hist", type=int, default=730,
                   help="history window in days (overrides --start/--end)")
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end",   type=str, default=None)
    p.add_argument("--lookback", type=int, default=BACKLOOK)
    p.add_argument("--eval",     type=int, default=EVAL_WINDOW)
    p.add_argument("--tc",       type=float, default=TC_RATE)
    p.add_argument("--device",   choices=["cpu", "cuda"], default=DEVICE)
    p.add_argument("--no-files", action="store_true")
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

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    nav, w, m = run(
        df, lookback=args.lookback, eval_win=args.eval,
        tc=args.tc, device=args.device, write_files=not args.no_files,
    )
    print("\nFinal NAV :", f"{nav.iloc[-1]:.4f}")
    print("Weights   :", {k: f"{v:.3%}" for k, v in w.items()})
    print("Metrics   :", m)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _cli()