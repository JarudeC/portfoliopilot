"""ARIMA order selection using statistical criteria.

This module contains logic for determining optimal ARIMA(p,d,q) orders
using ADF tests for differencing and AIC criterion for AR/MA orders.
"""

import logging
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


def determine_differencing(
    series: pd.Series,
    max_d: int = 2,
    adf_threshold: float = 0.05
) -> int:
    """
    Determine optimal differencing order using Augmented Dickey-Fuller test.

    The ADF test checks for stationarity in the time series. A stationary series
    has constant mean and variance over time, which is required for ARIMA modeling.

    Args:
        series: Time series data to test
        max_d: Maximum differencing order to test (default: 2)
        adf_threshold: P-value threshold for stationarity (default: 0.05)

    Returns:
        Optimal differencing order (0 to max_d)

    Notes:
        - Tests series at each differencing level
        - Returns first d where p-value <= threshold (series is stationary)
        - Returns 1 as safe fallback if all tests fail
    """
    try:
        # Test original series
        adf_stat, p_value, *_ = adfuller(series.dropna(), autolag='AIC')
        logger.debug(
            f"ADF test on original series: statistic={adf_stat:.4f}, "
            f"p-value={p_value:.4f}"
        )

        if p_value <= adf_threshold:
            logger.info("Series is already stationary (d=0)")
            return 0

        # Test with increasing differencing orders
        for d in range(1, max_d + 1):
            diff_series = series.diff(d).dropna()

            # Need minimum observations for valid test
            if len(diff_series) < 10:
                logger.warning(f"Insufficient data for differencing d={d}")
                break

            adf_stat, p_value, *_ = adfuller(diff_series, autolag='AIC')
            logger.debug(
                f"ADF test with d={d}: statistic={adf_stat:.4f}, "
                f"p-value={p_value:.4f}"
            )

            if p_value <= adf_threshold:
                logger.info(f"Series becomes stationary at d={d}")
                return d

        # Default to first difference if no clear signal
        logger.warning("No clear differencing order found, defaulting to d=1")
        return 1

    except Exception as e:
        logger.warning(f"Error in differencing selection: {e}, defaulting to d=1")
        return 1


def select_best_order(
    series: pd.Series,
    max_p: int = 3,
    max_q: int = 3,
    max_d: int = 2,
    adf_threshold: float = 0.05
) -> Tuple[int, int, int]:
    """
    Select optimal ARIMA(p,d,q) order using AIC criterion.

    This function performs a two-stage selection process:
    1. Determine differencing order (d) using ADF test
    2. Grid search over AR (p) and MA (q) orders using AIC

    Args:
        series: Time series data to model
        max_p: Maximum autoregressive order to test (default: 3)
        max_q: Maximum moving average order to test (default: 3)
        max_d: Maximum differencing order (default: 2)
        adf_threshold: P-value threshold for stationarity (default: 0.05)

    Returns:
        Tuple of (p, d, q) representing the best ARIMA order

    Notes:
        - Lower AIC indicates better model fit
        - Returns (1, 1, 1) as safe default if all models fail
        - Suppresses convergence warnings during grid search
    """
    # Step 1: Determine differencing order
    d = determine_differencing(series, max_d=max_d, adf_threshold=adf_threshold)
    logger.info(f"Selected differencing order d={d}")

    # Step 2: Grid search for best (p, q)
    best_aic = float('inf')
    best_order = (1, d, 1)
    tested_count = 0

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            # Skip (0,0,0) model - meaningless
            if p == 0 and q == 0 and d == 0:
                continue

            try:
                order = (p, d, q)

                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    model = ARIMA(series, order=order)
                    fitted = model.fit(method="statespace")

                    # Validate AIC is finite
                    if hasattr(fitted, 'aic') and np.isfinite(fitted.aic):
                        tested_count += 1

                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = order
                            logger.debug(
                                f"New best order: {order} with AIC={best_aic:.2f}"
                            )

            except Exception as e:
                logger.debug(f"Failed to fit ARIMA{order}: {e}")
                continue

    logger.info(
        f"Tested {tested_count} ARIMA orders, best: {best_order} "
        f"(AIC={best_aic:.2f})"
    )
    return best_order
