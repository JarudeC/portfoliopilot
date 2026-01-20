"""Abstract base classes and shared utilities for forecasting models.

This module provides the foundation for all forecasting implementations,
ensuring a consistent API and enforcing the Liskov Substitution Principle.
"""

from abc import ABC, abstractmethod
from datetime import date
from typing import List, Tuple

from core.data_loader import load_series as _load_series
from .schemas import ForecastRequest

# Type alias for forecast return value
ForecastResult = Tuple[List[str], List[float], List[str], List[float]]


class BaseForecaster(ABC):
    """
    Abstract base class for all forecasting models.

    All forecasters must implement the `forecast` method which takes
    a ForecastRequest and returns historical + forecast data in a
    consistent format.

    This ensures all forecasters are interchangeable (Liskov Substitution Principle)
    and makes it easy to add new forecasting algorithms.
    """

    @abstractmethod
    def forecast(self, req: ForecastRequest) -> ForecastResult:
        """
        Generate forecast for the given request.

        Args:
            req: ForecastRequest with ticker, dates, and horizon

        Returns:
            Tuple of (hist_dates, hist_values, forecast_dates, forecast_values)
            where:
                - hist_dates: List of date strings (YYYY-MM-DD)
                - hist_values: List of historical prices
                - forecast_dates: List of forecast date strings
                - forecast_values: List of predicted prices

        Raises:
            ValueError: If request data is invalid or insufficient
            RuntimeError: If model fitting/forecasting fails
        """
        pass

    def validate_data_length(
        self,
        series_length: int,
        min_required: int,
        ticker: str
    ) -> None:
        """
        Validate that we have sufficient data for modeling.

        Args:
            series_length: Actual data length
            min_required: Minimum required observations
            ticker: Ticker symbol for error message

        Raises:
            ValueError: If insufficient data
        """
        if series_length < min_required:
            raise ValueError(
                f"Insufficient data for {ticker}: {series_length} observations, "
                f"need at least {min_required}"
            )


# Backward compatibility: maintain load_series function for existing code
def load_series(ticker: str, start: date, end: date):
    """
    Download adjusted close prices from Yahoo Finance.

    This is a backward-compatible wrapper around the new data_loader module.
    Existing code can continue using this import path.

    Args:
        ticker: Yahoo Finance ticker symbol
        start: Inclusive start date
        end: Exclusive end date

    Returns:
        pandas Series with price data

    See Also:
        forecasting.data_loader.load_series: New consolidated implementation
    """
    return _load_series(ticker, start, end, validate=True)
