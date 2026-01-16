"""Naive Markowitz portfolio strategy implementation."""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..base import BaseStrategy
from ..config import NaiveMarkowitzConfig
from ..metrics import compute_metrics_from_nav

logger = logging.getLogger(__name__)

# Constants
RETURN_CAP = 0.1  # ±10% hard cap on returns
RIDGE_LAMBDA = 1e-4  # Ridge regularization for covariance


class NaiveMarkowitzModel:
    """
    Classic mean-variance portfolio with optional Gaussian-noise alpha.

    The expected-return vector is estimated on the look-back window
    (strictly past data). Noise is added in-sample.
    """

    def __init__(
        self,
        historical_data: pd.DataFrame,
        lookback_window: List[int],
        evaluation_window: int,
        eta: float,
        markowitz_type: str = "expected_returns",
        transaction_cost_rate: float = 0.0001,
    ):
        self.historical_data = historical_data
        self.lb0, self.lb1 = lookback_window
        self.evaluation_window = evaluation_window
        self.eta = eta
        self.markowitz_type = markowitz_type
        self.tc = transaction_cost_rate

        # Pre-compute
        self.cov = self._cov_matrix()
        self.mu = self._exp_returns_with_noise()
        self.w = self._markowitz_weights()

    def _cov_matrix(self) -> pd.DataFrame:
        """Compute covariance matrix with ridge regularization."""
        rets = self.historical_data.iloc[self.lb0:self.lb1].pct_change().dropna()
        cov = rets.cov().fillna(0.0)
        cov.values[np.diag_indices_from(cov)] += RIDGE_LAMBDA
        return cov

    def _exp_returns_with_noise(self) -> pd.Series:
        """Past-window mean + optional zero-mean Gaussian noise."""
        mu = (
            self.historical_data
            .iloc[self.lb0:self.lb1]
            .pct_change()
            .dropna()
            .mean()
        )

        if self.eta == 0:
            return mu

        # Noise scale proportional to std(μ) * eta
        noise_sd = mu.std() * self.eta
        noise = np.random.normal(0, noise_sd, size=len(mu))
        return mu + noise

    def _markowitz_weights(self) -> np.ndarray:
        """Compute Markowitz optimal weights."""
        e = np.ones(len(self.mu))
        try:
            cov_inv = np.linalg.inv(self.cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(self.cov)

        w_minvar = cov_inv @ e / (e @ cov_inv @ e)
        w_mk = cov_inv @ self.mu / (e @ cov_inv @ self.mu)

        if self.markowitz_type == "min_variance":
            w = w_minvar
        else:  # expected_returns
            target = 0.0008  # 0.08% daily
            alpha = (target - self.mu @ w_minvar) / (self.mu @ (w_mk - w_minvar))
            w = w_minvar if self.mu @ w_minvar >= target else w_minvar + alpha * (w_mk - w_minvar)

        # Gross-leverage = 1
        return w / np.sum(np.abs(w))

    def weights(self) -> pd.Series:
        """Return weights as Series."""
        return pd.Series(self.w, index=self.historical_data.columns, name="weight")

    def forward_returns(self) -> pd.Series:
        """Compute forward returns for evaluation window."""
        win = (
            self.historical_data
            .iloc[self.lb1 - 1:self.lb1 + self.evaluation_window]
            .pct_change()
            .dropna()
        )
        return win @ self.w


class NaiveMarkowitzStrategy(BaseStrategy):
    """
    Classic mean-variance optimization with optional noise.

    Uses historical returns to estimate expected returns and covariance,
    then computes optimal weights targeting a specified return level.
    """

    name = "Naive Markowitz"

    def __init__(self, config: NaiveMarkowitzConfig | None = None):
        self.config = config or NaiveMarkowitzConfig()
        np.random.seed(self.config.random_seed)

    def optimize(
        self,
        prices: pd.DataFrame,
        lookback: int,
        eta: float = 0.0,
        **kwargs
    ) -> np.ndarray:
        """
        Compute optimal weights for single period.

        Args:
            prices: Price data
            lookback: Number of historical days to use
            eta: Noise parameter for expected returns

        Returns:
            Portfolio weight vector
        """
        model = NaiveMarkowitzModel(
            historical_data=prices,
            lookback_window=[0, lookback],
            evaluation_window=1,
            eta=eta,
            markowitz_type=self.config.markowitz_type,
        )
        return model.weights().values

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
        Run rolling-window backtest.

        Args:
            prices: Historical price DataFrame
            lookback: Rolling window for estimation
            eval_win: Days between rebalancing
            eta: Noise parameter
            tc: Transaction cost rate

        Returns:
            (nav_series, final_weights, metrics)
        """
        hist = prices.dropna().copy()
        dates = hist.index.to_series()

        first_sig = lookback
        num_win = (len(hist) - first_sig) // eval_win
        look = [0, lookback]

        nav = [1.0]
        w_prev = None

        for step in range(1, num_win + 1):
            mdl = NaiveMarkowitzModel(
                historical_data=hist,
                lookback_window=look,
                evaluation_window=eval_win,
                eta=eta,
                transaction_cost_rate=tc,
            )

            w_new = mdl.weights().values
            turnover = 1.0 if w_prev is None else abs(w_new - w_prev).sum()
            w_prev = w_new.copy()
            cost = turnover * tc

            fr = mdl.forward_returns()
            s = fr["return"] if isinstance(fr, pd.DataFrame) and "return" in fr else fr.squeeze()
            r_fwd = s.copy()
            r_fwd.iloc[0] -= cost

            # Clip returns
            r_fwd = (
                r_fwd.replace([np.inf, -np.inf], np.nan)
                .fillna(0)
                .clip(lower=-RETURN_CAP, upper=RETURN_CAP)
            )

            nav_last = nav[-1]
            nav_last *= np.exp(np.log1p(r_fwd).sum())
            nav.append(nav_last)
            look = [look[0] + eval_win, look[1] + eval_win]

        nav_series = pd.Series(
            nav[1:],
            index=dates.iloc[first_sig:first_sig + num_win * eval_win:eval_win],
            name="NAV",
        )
        weights = pd.Series(w_prev, index=hist.columns).to_dict()
        metrics = compute_metrics_from_nav(nav_series)

        logger.info(
            "%s backtest complete: %d periods, final NAV=%.4f",
            self.name, len(nav_series), nav_series.iloc[-1] if len(nav_series) > 0 else 1.0
        )

        return nav_series, weights, metrics
