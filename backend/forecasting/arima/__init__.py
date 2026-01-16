"""ARIMA forecasting module with automatic order selection and robust fallbacks.

This module provides ARIMA (AutoRegressive Integrated Moving Average) forecasting
with statistical order selection, comprehensive validation, and multiple fallback
strategies for robustness.
"""

from .forecaster import ARIMAForecaster, forecast

__all__ = ['ARIMAForecaster', 'forecast']
