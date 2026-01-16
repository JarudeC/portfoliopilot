"""Portfolio Policy Network strategy implementation."""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..base import BaseStrategy
from ..config import PolicyNetworkConfig
from ..metrics import compute_metrics_from_nav
from .agent import PolicyAgent

logger = logging.getLogger(__name__)

# Default hyperparameters
DEFAULT_LOOKBACK = 252
DEFAULT_EVAL_WINDOW = 5
DEFAULT_TC_RATE = 0.002
VERBOSE_EVERY = 20


def _load_default_config(lookback: int, n_assets: int) -> Dict:
    """Build default config dict for PolicyAgent."""
    return {
        "input": {
            "feature_number": 1,
            "coin_number": n_assets,
            "window_size": lookback,
        },
        "training": {
            "learning_rate": 0.001,
            "gamma": 0.1,
            "alpha": 0.01,
            "dropout": 0.2,
            "decay_steps": 1000,
            "decay_rate": 0.9,
        },
        "trading": {
            "trading_consumption": 0.002,
        },
    }


class PolicyNetworkStrategy(BaseStrategy):
    """
    CNN-TCN-LSTM deep learning strategy for portfolio optimization.

    Uses online learning to adapt weights during backtest.
    Architecture: Temporal Convolutional Network + LSTM for feature extraction,
    followed by decision layers that output portfolio weights.
    """

    name = "Policy Network"

    def __init__(self, config: PolicyNetworkConfig | None = None, device: str = "cpu"):
        self.config = config or PolicyNetworkConfig()
        self.device = device
        self._agent = None

    def optimize(
        self,
        prices: pd.DataFrame,
        lookback: int,
        **kwargs
    ) -> np.ndarray:
        """
        Predict weights for single period.

        Args:
            prices: Price DataFrame
            lookback: Window size

        Returns:
            Portfolio weights
        """
        n_assets = len(prices.columns)

        if self._agent is None:
            cfg = _load_default_config(lookback, n_assets)
            self._agent = PolicyAgent(cfg, device=self.device)

        window = prices.iloc[-lookback:] if len(prices) >= lookback else prices
        history = window.to_numpy(dtype=np.float32)[:, :, np.newaxis]

        prev_w = np.full(n_assets + 1, 1 / (n_assets + 1), dtype=np.float32)

        weights_full = self._agent.predict(history, prev_w)

        # Remove cash position and renormalize
        asset_weights = weights_full[1:]
        asset_sum = asset_weights.sum()
        if asset_sum > 0:
            return asset_weights / asset_sum
        return np.full(n_assets, 1.0 / n_assets, dtype=np.float32)

    def backtest(
        self,
        prices: pd.DataFrame,
        lookback: int,
        eval_win: int,
        eta: float,
        tc: float,
        **kwargs
    ) -> Tuple[pd.Series, Dict[str, float], Dict[str, float]]:
        """
        Run rolling-window backtest with online learning.

        Args:
            prices: Historical price DataFrame
            lookback: Window size for network input
            eval_win: Rebalancing frequency in days
            eta: Unused (API compatibility)
            tc: Transaction cost rate

        Returns:
            (nav_series, final_weights, metrics)
        """
        prices = prices.sort_index()
        tickers = prices.columns.tolist()
        dates = prices.index.to_numpy()
        prices_np = prices.to_numpy(dtype=np.float32)[:, :, None]  # (T, N, 1)

        n_assets = prices_np.shape[1]

        # Initialize agent
        cfg = _load_default_config(lookback, n_assets)
        cfg["trading"]["trading_consumption"] = tc
        agent = PolicyAgent(cfg, device=self.device)

        # Initial portfolio state
        w_prev = np.full(n_assets, 1 / n_assets, dtype=np.float32)
        log_nav = 0.0
        nav: List[float] = [1.0]
        rlog: List[float] = []
        turns: List[float] = []

        # Main backtest loop
        for step, t0 in enumerate(range(lookback, len(prices_np), eval_win), 1):
            hist = prices_np[t0 - lookback:t0]  # (lookback, N, 1)

            # Prepare previous weights with cash position for agent
            w_prev_full = np.concatenate([[0.0], w_prev]).astype(np.float32)

            # Training step
            x_batch = np.transpose(hist, (2, 1, 0))[None, ...]  # (1, 1, N, T)
            y_next = (prices_np[t0, :, 0] / prices_np[t0 - 1, :, 0]).astype(np.float32)
            y_next = np.nan_to_num(y_next, nan=1.0, posinf=1.0, neginf=1.0)

            agent.train_step(x_batch, y_next[None, :], w_prev[None, :])

            # Get new weights
            w_full = agent.predict(hist, w_prev_full)
            w_t = w_full[1:]  # Remove cash position

            # Sanitize weights
            w_t = np.nan_to_num(w_t, nan=0.0, posinf=0.0, neginf=0.0)
            if w_t.min() < 0:
                w_t = np.maximum(w_t, 0.0)
            s = float(w_t.sum())
            if s > 0:
                w_t = w_t / s
            else:
                w_t = np.full(n_assets, 1 / n_assets, dtype=np.float32)

            # Leverage cap at 1x
            lev = np.sum(np.abs(w_t))
            if lev > 1.0:
                w_t = w_t / lev

            turn = float(np.sum(np.abs(w_t - w_prev)))
            turns.append(turn)
            w_prev = w_t.copy()

            # Simulate over holding window
            for d in range(eval_win):
                if t0 + d >= len(prices_np):
                    break

                step_ret = np.dot(
                    w_prev,
                    prices_np[t0 + d, :, 0] / prices_np[t0 + d - 1, :, 0] - 1.0,
                )
                fee = tc * turn if d == 0 else 0.0
                r_net = step_ret - fee

                rlog.append(r_net)
                log_nav += np.log1p(r_net)
                nav.append(np.exp(log_nav))

            if VERBOSE_EVERY and step % VERBOSE_EVERY == 0:
                logger.info(
                    "[%4d] nav=%.4f mean_r=%+.5f turn=%.3f",
                    step, nav[-1], np.mean(rlog[-eval_win:]), turn
                )

            w_prev = w_t.copy()

        # Build output
        total_days = len(rlog)
        nav_dates = dates[lookback - 1:lookback - 1 + total_days]
        nav_series = pd.Series(nav[:total_days], index=nav_dates, name="NAV")

        metrics = compute_metrics_from_nav(nav_series)
        final_weights = dict(zip(tickers, map(float, w_prev)))

        logger.info(
            "%s backtest complete: %d periods, final NAV=%.4f",
            self.name, len(nav_series), nav_series.iloc[-1] if len(nav_series) > 0 else 1.0
        )

        return nav_series, final_weights, metrics
