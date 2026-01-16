"""Request and response schemas for forecasting API.

This module defines Pydantic models for validating forecast requests and responses,
ensuring data integrity at the API boundary.
"""

from datetime import date, datetime
from typing import List

from pydantic import BaseModel, Field, validator


class ForecastRequest(BaseModel):
    """Request schema for all forecasting endpoints.

    Validates ticker symbols, date ranges, and forecast horizons to ensure
    requests are well-formed before processing.

    Attributes:
        ticker: Stock ticker symbol (e.g., "AAPL", "MSFT")
        start: Inclusive start date for historical data window
        end: Exclusive end date for historical data window
        horizon: Number of trading days to forecast forward (1-365)
    """

    ticker: str = Field(
        ...,
        examples=["AAPL", "NVDA"],
        description="Single equity ticker symbol (Yahoo Finance format)",
    )
    start: date = Field(
        ...,
        description="Inclusive start date for historical window (YYYY-MM-DD)",
        examples=["2024-01-01"],
    )
    end: date = Field(
        ...,
        description="Exclusive end date for historical window (YYYY-MM-DD)",
        examples=["2025-12-31"],
    )
    horizon: int = Field(
        14,
        gt=0,
        le=365,
        description="Number of trading days to predict forward",
    )

    @validator("ticker")
    def _normalize_ticker(cls, v: str) -> str:
        """Normalize ticker to uppercase and strip whitespace."""
        return v.strip().upper()

    @validator("end")
    def _validate_end_date(cls, v: date) -> date:
        """Ensure end date is not in the future."""
        today = datetime.utcnow().date()
        if v > today:
            raise ValueError(
                f"end date {v} cannot be in the future (today: {today})"
            )
        return v

    @validator("start")
    def _validate_date_range(cls, v: date, values) -> date:
        """Ensure start date is before end date."""
        end = values.get("end")
        if end and v >= end:
            raise ValueError("start must be earlier than end")
        return v


class ForecastResponse(BaseModel):
    """Response schema for forecasting endpoints.

    Attributes:
        history_dates: List of ISO date strings for historical data
        history_values: List of historical price values
        forecast_dates: List of ISO date strings for forecasted dates
        forecast_values: List of predicted price values
    """

    history_dates: List[str] = Field(
        ..., description="Historical dates in YYYY-MM-DD format"
    )
    history_values: List[float] = Field(
        ..., description="Historical price values"
    )
    forecast_dates: List[str] = Field(
        ..., description="Forecast dates in YYYY-MM-DD format"
    )
    forecast_values: List[float] = Field(
        ..., description="Predicted price values"
    )
