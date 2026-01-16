"""Fallback strategies for ARIMA model fitting.

Implements a chain-of-responsibility pattern for trying different
ARIMA configurations when optimal order selection fails.
"""

import logging
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from .validation import validate_fitted_model

logger = logging.getLogger(__name__)


class FallbackStrategy:
    """
    Tries a sequence of fallback ARIMA orders when auto-selection fails.

    This implements a chain-of-responsibility pattern, trying progressively
    simpler models until one succeeds. Each order is validated before acceptance.
    """

    def __init__(self, fallback_orders: Tuple[Tuple[int, int, int], ...]):
        """
        Initialize with a sequence of ARIMA orders to try.

        Args:
            fallback_orders: Tuple of (p, d, q) orders to try in sequence
        """
        self.fallback_orders = fallback_orders

    def try_fallbacks(
        self,
        series: pd.Series
    ) -> Optional[Tuple[any, Tuple[int, int, int]]]:
        """
        Try each fallback order until one succeeds.

        Args:
            series: Time series to model

        Returns:
            Tuple of (fitted_model, order) if successful, None if all fail
        """
        logger.info(f"Trying {len(self.fallback_orders)} fallback orders")

        for order in self.fallback_orders:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    model = ARIMA(series, order=order)
                    fitted = model.fit(method="statespace")

                    if validate_fitted_model(fitted):
                        logger.info(f"Fallback successful with ARIMA{order}")
                        return fitted, order
                    else:
                        logger.debug(f"Fallback ARIMA{order} failed validation")

            except Exception as e:
                logger.debug(f"Fallback ARIMA{order} failed: {e}")
                continue

        logger.warning("All fallback orders failed")
        return None

    def try_random_walk(
        self,
        series: pd.Series
    ) -> Optional[Tuple[any, Tuple[int, int, int]]]:
        """
        Last resort: try random walk model ARIMA(0,1,0).

        The random walk model assumes the best prediction is the last value
        plus a random step. This is equivalent to ARIMA(0,1,0).

        Args:
            series: Time series to model

        Returns:
            Tuple of (fitted_model, order) if successful, None if fails
        """
        logger.info("Attempting random walk model as last resort")

        try:
            model = ARIMA(series, order=(0, 1, 0))
            fitted = model.fit(method="statespace")
            logger.info("Random walk model fitted successfully")
            return fitted, (0, 1, 0)
        except Exception as e:
            logger.error(f"Even random walk model failed: {e}")
            return None


def generate_naive_forecast(
    series: pd.Series,
    horizon: int,
    method: str = 'trend'
) -> List[float]:
    """
    Generate naive forecast when all ARIMA models fail.

    This is the absolute last resort when even the random walk model fails.
    Provides a simple but reasonable forecast based on recent data trends.

    Args:
        series: Historical time series
        horizon: Number of steps to forecast
        method: 'trend' (linear extrapolation) or 'last' (repeat last value)

    Returns:
        List of forecasted values

    Notes:
        - 'trend' method fits a line to recent data and extrapolates
        - 'last' method simply repeats the last observed value
        - Falls back to 'last' if 'trend' calculation fails
    """
    logger.warning(f"Generating naive {method} forecast")

    if method == 'trend' and len(series) >= 2:
        # Use recent trend (last 10 values or 25% of data, whichever is smaller)
        recent_window = min(10, len(series) // 4)
        recent_values = series.tail(recent_window)

        if len(recent_values) >= 2:
            x_vals = np.arange(len(recent_values))
            y_vals = recent_values.values
            slope = np.polyfit(x_vals, y_vals, 1)[0]
            last_val = float(series.iloc[-1])

            forecast = [last_val + slope * (i + 1) for i in range(horizon)]
            logger.info(f"Trend-based naive forecast: slope={slope:.4f}")
            return forecast

    # Fallback: repeat last value
    last_val = float(series.iloc[-1])
    logger.info("Using last-value naive forecast")
    return [last_val] * horizon
