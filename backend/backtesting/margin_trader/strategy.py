"""Margin Trader A2C reinforcement learning strategy."""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.utils import set_random_seed

from ..base import BaseStrategy
from ..config import MarginTraderConfig
from ..metrics import compute_metrics_from_nav
from .agent import DRLAgent
from .environment import MarginTradingEnv

logger = logging.getLogger(__name__)

# Default hyperparameters
DEFAULT_LOOKBACK = 252
DEFAULT_EVAL_WIN = 5
DEFAULT_TC = 0.002
DEFAULT_TOTAL_STEPS = 20_000


class MarginTraderStrategy(BaseStrategy):
    """A2C reinforcement learning strategy with margin trading.

    Uses Stable Baselines 3 A2C algorithm to learn trading policies
    supporting both long and short positions with leverage.
    """

    name = "Margin Trader"

    def __init__(self, config: MarginTraderConfig | None = None):
        self.config = config or MarginTraderConfig()

    def optimize(
        self,
        prices: pd.DataFrame,
        lookback: int,
        **kwargs
    ) -> np.ndarray:
        """Not implemented for RL strategies.

        RL strategies require full training before making predictions.
        Use backtest() instead.
        """
        raise NotImplementedError(
            "RL strategies don't support single-step optimization. Use backtest()."
        )

    def backtest(
        self,
        prices: pd.DataFrame,
        lookback: int,
        eval_win: int,
        eta: float,
        tc: float,
        total_steps: int = DEFAULT_TOTAL_STEPS,
        seed: int = 0,
        **kwargs
    ) -> Tuple[pd.Series, Dict[str, float], Dict[str, float]]:
        """Run A2C training and backtesting.

        Args:
            prices: Historical price DataFrame
            lookback: Days for train/trade split
            eval_win: Unused (API compatibility)
            eta: Unused (API compatibility)
            tc: Transaction cost rate
            total_steps: Training timesteps
            seed: Random seed

        Returns:
            (nav_series, final_weights, metrics)
        """
        # Reproducibility
        set_random_seed(seed)
        torch.manual_seed(seed)

        # Clean and prepare prices
        prices = (
            prices.sort_index()
            .dropna(how="all")
            .replace([np.inf, -np.inf], np.nan)
            .dropna(how="all")
        )

        if prices.empty:
            raise ValueError("Price DataFrame is empty after cleaning.")

        # Reshape to long format
        df_price = prices.stack().reset_index()
        df_price.columns = ["date", "tic", "close"]

        # Train/trade split
        all_dates = sorted(df_price["date"].unique())
        if len(all_dates) <= lookback + eval_win:
            raise ValueError(
                f"Need >{lookback + eval_win} unique dates; got {len(all_dates)}."
            )

        split_date = all_dates[lookback]
        train_df = df_price[df_price["date"] < split_date]
        trade_df = df_price[df_price["date"] >= split_date]

        # Environment configuration
        tickers = prices.columns.tolist()
        stock_dim = len(tickers)

        env_kwargs = {
            "hmax": 1,
            "initial_amount": 1_000_000,
            "num_stock_shares": [0] * stock_dim,
            "buy_cost_pct": [tc] * stock_dim,
            "sell_cost_pct": [tc] * stock_dim,
            "state_space": 2 * 3 + 2 * stock_dim,
            "stock_dim": stock_dim,
            "tech_indicator_list": [],
            "action_space": 2 * stock_dim,
            "reward_scaling": 1e-4,
            "penalty_sharpe": 0.05,
            "max_leverage": 1.5,
        }

        # A2C training
        train_env = MarginTradingEnv(df=train_df, **env_kwargs).get_sb_env()[0]
        agent = DRLAgent(env=train_env)

        model = agent.get_model(
            "a2c",
            model_kwargs={
                "n_steps": 5,
                "gamma": 0.99,
                "learning_rate": 0.005,
                "ent_coef": 0.005,
            },
            seed=seed,
        )
        model.learn(total_timesteps=total_steps, progress_bar=False)

        # Backtest
        trade_env = MarginTradingEnv(df=trade_df, **env_kwargs)
        account_df, _, state_df = DRLAgent.DRL_prediction(model, trade_env)

        # Process account DataFrame
        if not isinstance(account_df, pd.DataFrame):
            account_df = pd.DataFrame(account_df)
        account_df.columns = [str(c).lower() for c in account_df.columns]

        # Find equity curve column
        eq_col = None
        for candidate in ("account_value", "portfoliovalue"):
            if candidate in account_df.columns:
                eq_col = candidate
                break
        if eq_col is None:
            eq_col = account_df.columns[1] if account_df.shape[1] > 1 else account_df.columns[0]

        # Ensure dates
        if "date" not in account_df.columns:
            trade_dates = sorted(trade_df["date"].unique())
            dates = trade_dates[:len(account_df)]
            account_df.insert(0, "date", dates)
        account_df["date"] = pd.to_datetime(account_df["date"])

        # Build NAV series
        nav = account_df.set_index("date")[eq_col].astype(float)
        nav = nav / nav.iloc[0]
        nav.name = "NAV"

        # Extract final weights from state memory
        weights = self._extract_weights(state_df, tickers)

        # Compute metrics
        metrics = compute_metrics_from_nav(nav)

        logger.info(
            "%s backtest complete: %d periods, final NAV=%.4f",
            self.name, len(nav), nav.iloc[-1] if len(nav) > 0 else 1.0
        )

        return nav, weights, metrics

    def _extract_weights(
        self,
        state_df: pd.DataFrame,
        tickers: list
    ) -> Dict[str, float]:
        """Extract portfolio weights from final state.

        Args:
            state_df: State memory DataFrame
            tickers: List of ticker symbols

        Returns:
            Dict mapping tickers to weights
        """
        if isinstance(state_df, pd.DataFrame) and any(
            c.endswith("_h") for c in state_df.columns
        ):
            last = state_df.iloc[-1]
            exposures = {}

            for t in tickers:
                h = float(last.get(f"{t}_h", 0.0))
                p = float(last.get(f"{t}_c", np.nan))
                exposures[t] = h * p if not np.isnan(p) else 0.0

            total = sum(abs(v) for v in exposures.values())
            if total > 0:
                return {t: abs(v) / total for t, v in exposures.items()}

        # Fallback to equal weights
        return {t: 1.0 / len(tickers) for t in tickers}
