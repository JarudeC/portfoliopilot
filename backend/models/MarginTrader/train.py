# models/MarginTrader/train.py
from __future__ import annotations
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import A2C

from .env.agent import DRLAgent
from .env.margin_env import MarginTradingEnv
from .record import compute_metrics

# ── Defaults ────────────────────────────────────────────────────────────────
LOOKBACK_DEF    = 252
EVAL_WIN_DEF    = 5
TC_DEF          = 0.002
TOTAL_STEPS_DEF = 20_000

def run(
    prices: pd.DataFrame,
    lookback: int    = LOOKBACK_DEF,
    eval_win: int    = EVAL_WIN_DEF,
    eta: float       = 0.02,     # ignored
    tc: float        = TC_DEF,
    total_steps: int = TOTAL_STEPS_DEF,
    seed: int        = 0,
) -> Tuple[pd.Series, Dict[str, float], Dict[str, str]]:
    """
    FastAPI entrypoint for Margin Trader.
    Returns:
      - NAV (normalized to start at 1.0),
      - weights dict keyed by ticker (sums to 1),
      - metrics dict (formatted like the others).
    """
    # reproducibility
    set_random_seed(seed)
    torch.manual_seed(seed)

    # ─── 1) Clean & reshape prices ───────────────────────────────────────
    prices = (
        prices.sort_index()
              .dropna(how="all")
              .replace([np.inf, -np.inf], np.nan)
              .dropna(how="all")
    )
    if prices.empty:
        raise ValueError("Price DataFrame is empty after cleaning.")

    df_price = prices.stack().reset_index()
    df_price.columns = ["date", "tic", "close"]

    # ─── 2) Train/trade split ────────────────────────────────────────────
    all_dates = sorted(df_price["date"].unique())
    if len(all_dates) <= lookback + eval_win:
        raise ValueError(f"Need >{lookback+eval_win} unique dates; got {len(all_dates)}.")
    split_date = all_dates[lookback]
    train_df = df_price[df_price["date"] < split_date]
    trade_df = df_price[df_price["date"] >= split_date]

    # ─── 3) Build env kwargs, plugging in UI’s tc ───────────────────────
    tickers   = prices.columns.tolist()
    stock_dim = len(tickers)
    env_kwargs = {
        "hmax":                1,
        "initial_amount":      1_000_000,
        "num_stock_shares":    [0] * stock_dim,
        "buy_cost_pct":        [tc] * stock_dim,
        "sell_cost_pct":       [tc] * stock_dim,
        "state_space":         2*3 + 2*stock_dim,
        "stock_dim":           stock_dim,
        "tech_indicator_list": [],
        "action_space":        2*stock_dim,
        "reward_scaling":      1e-4,
        "penalty_sharpe":      0.05,
        "max_leverage":        1.5,
    }

    # ─── 4) Quick A2C training ──────────────────────────────────────────
    train_env = MarginTradingEnv(df=train_df, **env_kwargs).get_sb_env()[0]
    agent     = DRLAgent(env=train_env)
    model = agent.get_model(
    "a2c",
    model_kwargs={
            "n_steps": 5,
            "gamma":   0.99,
            "learning_rate": 0.005,
            "ent_coef":      0.005,
        },
        seed=seed,
    )
    model.learn(total_timesteps=total_steps, progress_bar=False)

    # ─── 5) Back-test & collect output ─────────────────────────────────
    trade_env = MarginTradingEnv(df=trade_df, **env_kwargs)
    account_df, _, state_df = DRLAgent.DRL_prediction(model, trade_env)

    # ensure DataFrame
    if not isinstance(account_df, pd.DataFrame):
        account_df = pd.DataFrame(account_df)
    account_df.columns = [str(c).lower() for c in account_df.columns]

    # find the equity curve column
    for candidate in ("account_value", "portfoliovalue"):
        if candidate in account_df.columns:
            eq_col = candidate
            break
    else:
        # fallback to second column if it exists
        eq_col = account_df.columns[1] if account_df.shape[1] > 1 else account_df.columns[0]

    # ensure we have dates aligned
    if "date" not in account_df.columns:
        dates = sorted(trade_df["date"].unique())[: len(account_df)]
        account_df.insert(0, "date", dates)
    account_df["date"] = pd.to_datetime(account_df["date"])

    # build NAV series and normalize to 1.0
    nav = account_df.set_index("date")[eq_col].astype(float)
    nav = nav / nav.iloc[0]
    nav.name = "NAV"

    # ─── 6) Derive final weights from the last env state ──────────────
    # state_df comes from save_state_memory(), with columns like "AAPL_h" and "AAPL_c"
    if isinstance(state_df, pd.DataFrame) and any(c.endswith("_h") for c in state_df.columns):
        last = state_df.iloc[-1]
        exps = {}
        for t in tickers:
            h = float(last.get(f"{t}_h", 0.0))
            p = float(last.get(f"{t}_c", np.nan))
            exps[t] = h * p if not np.isnan(p) else 0.0
        total = sum(abs(v) for v in exps.values())
        if total > 0:
            weights = {t: abs(v) / total for t, v in exps.items()}
        else:
            weights = {t: 0.0 for t in tickers}
    else:
        # fallback to equal-weight if we can’t extract holdings
        weights = {t: 1.0 / len(tickers) for t in tickers}

    # ─── 7) Metrics & return ───────────────────────────────────────────
    metrics = compute_metrics(nav)
    return nav, weights, metrics
