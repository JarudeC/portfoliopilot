"""Abstract base class for portfolio optimization strategies."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple
import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    Base class for all portfolio strategies.

    Subclasses must implement optimize() and backtest() methods.
    """

    name: str = "BaseStrategy"

    @abstractmethod
    def optimize(
        self,
        prices: pd.DataFrame,
        lookback: int,
        **kwargs
    ) -> np.ndarray:
        """
        Calculate optimal portfolio weights for a single period.

        Args:
            prices: Price DataFrame with assets as columns
            lookback: Days of history to use for estimation

        Returns:
            Portfolio weights array summing to 1
        """
        pass

    @abstractmethod
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
            prices: Price DataFrame with assets as columns
            lookback: Rolling window size for estimation
            eval_win: Rebalancing frequency in days
            eta: Strategy-specific noise/exploration parameter
            tc: Transaction cost rate

        Returns:
            Tuple of (nav_series, final_weights_dict, metrics_dict)
        """
        pass

    def validate_prices(self, prices: pd.DataFrame, min_rows: int) -> None:
        """Validate price data has sufficient rows and no all-NaN columns."""
        if prices.empty:
            raise ValueError("Price data is empty")

        if len(prices) < min_rows:
            raise ValueError(
                f"Insufficient data: {len(prices)} rows, need {min_rows}"
            )

        all_nan_mask = prices.isnull().all()
        if all_nan_mask.any():
            bad_cols = prices.columns[all_nan_mask].tolist()
            raise ValueError(f"All-NaN columns: {bad_cols}")

        logger.debug("Validated: %d rows, %d assets", len(prices), len(prices.columns))

    def calculate_turnover(
        self,
        old_weights: np.ndarray | None,
        new_weights: np.ndarray
    ) -> float:
        """Calculate turnover between rebalancing periods."""
        if old_weights is None:
            return 1.0  # Initial investment
        return float(np.abs(new_weights - old_weights).sum())

    def apply_transaction_cost(
        self,
        returns: pd.Series,
        turnover: float,
        tc_rate: float
    ) -> pd.Series:
        """Deduct transaction cost from first return in period."""
        adjusted = returns.copy()
        if len(adjusted) > 0:
            adjusted.iloc[0] -= turnover * tc_rate
        return adjusted
