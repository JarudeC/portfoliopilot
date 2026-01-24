"""Time series forecasting module for portfolio analysis.

This module provides ARIMA, LSTM, and Autoformer-based forecasting with automatic
model selection, robust fallback strategies, and a consistent API.

Public API:
    - arima: ARIMA forecasting with automatic order selection
    - lstm: LSTM neural network forecasting
    - autoformer: Autoformer with series decomposition and auto-correlation

All forecasters provide a `forecast(req)` function that returns:
    (hist_dates, hist_values, forecast_dates, forecast_values)

Example:
    >>> from forecasting import arima
    >>> from forecasting.schemas import ForecastRequest
    >>> from datetime import date
    >>>
    >>> req = ForecastRequest(
    ...     ticker="AAPL",
    ...     start=date(2024, 1, 1),
    ...     end=date(2024, 12, 31),
    ...     horizon=14
    ... )
    >>> hist_dates, hist_vals, fc_dates, fc_vals = arima.forecast(req)
"""

from . import arima, lstm, autoformer
from .base import BaseForecaster, ForecastResult
from .schemas import ForecastRequest

__all__ = [
    # Forecasting modules
    'arima',
    'lstm',
    'autoformer',
    # Base classes and types
    'BaseForecaster',
    'ForecastRequest',
    'ForecastResult',
]
