"""Request schema for forecasting API.

This module defines Pydantic models for validating forecast requests,
ensuring data integrity at the API boundary.
"""

from datetime import date, datetime
from pydantic import BaseModel, Field, field_validator, model_validator


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

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, v: str) -> str:
        """Normalize ticker to uppercase and strip whitespace."""
        return v.strip().upper()

    @field_validator("end")
    @classmethod
    def _validate_end_date(cls, v: date) -> date:
        """Ensure end date is not in the future."""
        today = datetime.utcnow().date()
        if v > today:
            raise ValueError(
                f"end date {v} cannot be in the future (today: {today})"
            )
        return v

    @model_validator(mode="after")
    def _validate_date_range(self) -> "ForecastRequest":
        """Ensure start date is before end date."""
        if self.start >= self.end:
            raise ValueError("start must be earlier than end")
        return self
