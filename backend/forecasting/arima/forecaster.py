"""ARIMA forecasting implementation.

Provides ARIMA-based time series forecasting with automatic order selection,
robust fallback strategies, and comprehensive error handling.
"""

import logging
import warnings
from datetime import timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from ..config import ARIMAConfig
from core.data_loader import load_series
from ..schemas import ForecastRequest
from .fallback_strategies import FallbackStrategy, generate_naive_forecast
from .order_selection import select_best_order
from .validation import validate_fitted_model

logger = logging.getLogger(__name__)

# Type alias for forecast return value (matches existing API contract)
ForecastResult = Tuple[List[str], List[float], List[str], List[float]]


class ARIMAForecaster:
    """
    ARIMA forecaster with automatic order selection and robust fallbacks.

    This forecaster implements a comprehensive strategy for fitting ARIMA models:
    1. Automatic order selection using ADF test and AIC criterion
    2. Multiple predefined fallback orders if auto-selection fails
    3. Random walk model as penultimate fallback
    4. Trend-based naive forecast as absolute last resort

    Features:
        - Statistical order selection (ADF for d, AIC for p/q)
        - Robust validation of fitted models
        - Multi-level fallback strategies
        - Business day frequency handling
        - Comprehensive logging
    """

    def __init__(self, config: ARIMAConfig = None):
        """
        Initialize ARIMA forecaster with configuration.

        Args:
            config: ARIMAConfig instance (uses defaults if None)
        """
        self.config = config or ARIMAConfig()
        self.fallback_strategy = FallbackStrategy(self.config.fallback_orders)

    def _ensure_frequency(self, series: pd.Series) -> pd.Series:
        """
        Ensure series has business day frequency.

        Financial data should have business day frequency (excludes weekends/holidays).
        This method infers or forces the correct frequency.

        Args:
            series: Time series with potential missing frequency

        Returns:
            Series with business day frequency set
        """
        if series.index.freq is None:
            try:
                # Try to infer frequency
                series.index.freq = pd.infer_freq(series.index)

                if series.index.freq is None:
                    # Force business day frequency
                    logger.debug("Forcing business day frequency")
                    series = series.asfreq('B', method='ffill')
            except Exception as e:
                # Resample to business days
                logger.debug(f"Resampling to business days: {e}")
                series = series.resample('B').last().dropna()

        logger.debug(f"Series frequency: {series.index.freq}")
        return series

    def _fit_optimal_model(
        self,
        series: pd.Series
    ) -> Tuple[any, Tuple[int, int, int]]:
        """
        Fit ARIMA model using automatic order selection with fallbacks.

        Tries strategies in this order:
        1. Automatic statistical order selection
        2. Predefined fallback orders
        3. Random walk model

        Args:
            series: Time series to model

        Returns:
            Tuple of (fitted_model, order)

        Raises:
            RuntimeError: If all fitting strategies fail
        """
        # Strategy 1: Automatic order selection
        logger.info("Attempting automatic order selection")
        try:
            best_order = select_best_order(
                series,
                max_p=self.config.max_p,
                max_q=self.config.max_q,
                max_d=self.config.max_d,
                adf_threshold=self.config.adf_threshold
            )

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                model = ARIMA(series, order=best_order)
                fitted = model.fit(method="statespace")

                if validate_fitted_model(fitted):
                    logger.info(f"Auto-selection succeeded with ARIMA{best_order}")
                    return fitted, best_order
                else:
                    logger.warning(
                        f"Auto-selected model ARIMA{best_order} failed validation"
                    )

        except Exception as e:
            logger.warning(f"Automatic order selection failed: {e}")

        # Strategy 2: Fallback orders
        logger.info("Attempting fallback orders")
        result = self.fallback_strategy.try_fallbacks(series)
        if result is not None:
            return result

        # Strategy 3: Random walk
        logger.info("Attempting random walk model")
        result = self.fallback_strategy.try_random_walk(series)
        if result is not None:
            return result

        # All strategies failed
        raise RuntimeError("All ARIMA fitting strategies failed")

    def _generate_forecast(
        self,
        fitted_model: any,
        series: pd.Series,
        order: Tuple[int, int, int],
        horizon: int
    ) -> List[float]:
        """
        Generate forecast from fitted model with fallback strategies.

        Tries forecasting approaches in this order:
        1. Standard get_forecast() method
        2. Step-by-step recursive forecasting
        3. Naive trend-based forecast

        Args:
            fitted_model: Fitted ARIMA model
            series: Original time series
            order: ARIMA order used for fitting
            horizon: Number of steps to forecast

        Returns:
            List of forecasted values
        """
        # Primary strategy: get_forecast
        try:
            fc_res = fitted_model.get_forecast(steps=horizon)
            fc_values = fc_res.predicted_mean.tolist()

            # Validate forecast
            if any(pd.isna(fc_values)) or not all(np.isfinite(fc_values)):
                raise ValueError("Forecast contains invalid values")

            logger.info("Forecast generated successfully using get_forecast")
            return fc_values

        except Exception as e:
            logger.warning(f"get_forecast failed: {e}")

        # Fallback: step-by-step forecasting
        try:
            logger.info("Attempting step-by-step forecasting")
            fc_values = []
            current_series = series.copy()

            for step in range(horizon):
                # Refit and forecast one step
                temp_model = ARIMA(current_series, order=order)
                temp_fitted = temp_model.fit(method="statespace")
                next_val = temp_fitted.forecast(steps=1).iloc[0]
                fc_values.append(float(next_val))

                # Add to series for next iteration
                next_date = current_series.index[-1] + pd.Timedelta(days=1)
                # Skip weekends
                while next_date.weekday() >= 5:
                    next_date += pd.Timedelta(days=1)
                current_series = pd.concat([
                    current_series,
                    pd.Series([next_val], index=[next_date])
                ])

            logger.info("Step-by-step forecasting succeeded")
            return fc_values

        except Exception as e:
            logger.warning(f"Step-by-step forecasting failed: {e}")

        # Last resort: naive forecast
        return generate_naive_forecast(series, horizon, method='trend')

    def forecast(self, req: ForecastRequest) -> ForecastResult:
        """
        Generate ARIMA forecast for the given request.

        This is the main entry point for ARIMA forecasting. It orchestrates
        data loading, model fitting, forecasting, and result formatting.

        Args:
            req: ForecastRequest with ticker, dates, and horizon

        Returns:
            Tuple of (hist_dates, hist_values, forecast_dates, forecast_values)
            where:
                - hist_dates: List of historical dates (YYYY-MM-DD strings)
                - hist_values: List of historical prices (floats)
                - forecast_dates: List of forecast dates (YYYY-MM-DD strings)
                - forecast_values: List of predicted prices (floats)

        Raises:
            ValueError: If data is invalid or insufficient
            RuntimeError: If forecasting fails completely
        """
        try:
            # Load data
            series = load_series(req.ticker, req.start, req.end)

            # Validate sufficient data
            if len(series) < self.config.min_observations:
                raise ValueError(
                    f"Insufficient data for {req.ticker}: {len(series)} observations, "
                    f"need at least {self.config.min_observations}"
                )

            # Ensure proper frequency
            series = self._ensure_frequency(series)
            logger.info(
                f"Data frequency: {series.index.freq}, length: {len(series)}"
            )

            # Fit model
            fitted_model, order = self._fit_optimal_model(series)
            logger.info(f"Using ARIMA{order} for {req.ticker}")

            # Generate forecast
            fc_values = self._generate_forecast(
                fitted_model, series, order, req.horizon
            )

            # Build response
            hist_dates = series.index.strftime("%Y-%m-%d").tolist()
            hist_values = series.tolist()

            last_date = series.index[-1].date()
            fc_dates = pd.bdate_range(
                last_date + timedelta(days=1),
                periods=req.horizon
            ).strftime("%Y-%m-%d").tolist()

            # Ensure forecast values are clean floats
            fc_values = [float(v) for v in fc_values]

            # Final validation
            if len(fc_dates) != len(fc_values):
                raise ValueError(
                    f"Forecast length mismatch: {len(fc_dates)} dates vs "
                    f"{len(fc_values)} values"
                )

            logger.info(f"ARIMA forecast complete for {req.ticker}")
            return hist_dates, hist_values, fc_dates, fc_values

        except Exception as e:
            error_msg = f"ARIMA forecasting failed for {req.ticker}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e


# Public API function for backward compatibility with main.py
def forecast(req: ForecastRequest) -> ForecastResult:
    """
    Public API function for ARIMA forecasting.

    Maintains backward compatibility with existing main.py imports.
    Creates a new ARIMAForecaster instance and calls its forecast method.

    Args:
        req: ForecastRequest with ticker, dates, and horizon

    Returns:
        Tuple of (hist_dates, hist_values, forecast_dates, forecast_values)
    """
    forecaster = ARIMAForecaster()
    return forecaster.forecast(req)
