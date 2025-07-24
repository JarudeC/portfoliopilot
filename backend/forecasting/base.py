"""
backend/forecasting/base.py
Shared dataclasses and helpers for all forecasting models.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import List

import pandas as pd
import yfinance as yf
from pydantic import BaseModel, Field, validator


# ──────────────────────────────────────────────────────────────────────
#  Request / response schema
# ──────────────────────────────────────────────────────────────────────
class ForecastRequest(BaseModel):
    """
    JSON schema expected by the /forecast/<algo> endpoints and used
    by every model implementation (arima.py, lstm.py, autoformer.py,…).
    """

    ticker: str = Field(
        ...,
        examples=["AAPL", "NVDA"],
        description="Single equity ticker symbol (Yahoo style).",
    )
    start: date = Field(
        ...,
        description="Inclusive start date for historical window (YYYY-MM-DD).",
        examples=["2024-01-01"],
    )
    end: date = Field(
        ...,
        description="Exclusive end date for historical window (YYYY-MM-DD).",
        examples=["2025-07-23"],
    )
    horizon: int = Field(
        14,
        gt=0,
        le=365,
        description="Number of trading days to predict forward.",
    )

    # ----- validators -------------------------------------------------
    @validator("ticker")
    def _upper(cls, v: str) -> str:  # noqa: N805
        return v.strip().upper()

    @validator("end")
    def _end_not_future(cls, v: date) -> date:  # noqa: N805
        today = datetime.utcnow().date()
        if v > today:
            raise ValueError(f"end date {v} cannot be in the future (today: {today})")
        # Also check if it's too far in the future for stock data availability
        # Most stock APIs have data delays, so use a conservative cutoff
        max_reasonable_date = date(2024, 12, 31)  # Adjust as needed
        if v > max_reasonable_date:
            raise ValueError(f"end date {v} is beyond reliable stock data availability. Use {max_reasonable_date} or earlier.")
        return v

    @validator("start")
    def _logical_window(cls, v: date, values) -> date:  # noqa: N805
        end = values.get("end")
        if end and v >= end:
            raise ValueError("start must be earlier than end")
        return v


# ──────────────────────────────────────────────────────────────────────
#  Data acquisition helper
# ──────────────────────────────────────────────────────────────────────
def load_series(ticker: str, start: date, end: date) -> pd.Series:
    """
    Download Adjusted Close prices from Yahoo Finance and return a
    pandas Series indexed by `DatetimeIndex`.

    Raises if no data returned (e.g. invalid ticker).
    """
    # Add some debugging info
    print(f"Loading data for {ticker} from {start} to {end}")
    
    try:
        # Use the exact same pattern as the working utils/data_loader.py
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)["Close"]
        print(f"Downloaded data shape: {df.shape}")
        
        # Handle MultiIndex columns if they exist (same as data_loader.py)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if df.empty:
            raise ValueError(f"No price data returned for {ticker!r} between {start} and {end}. Check if ticker exists and dates are valid.")
        
        # For single ticker, df["Close"] returns a Series, but we need to handle both cases
        if isinstance(df, pd.Series):
            series = df.rename(ticker).astype("float32")
        else:
            # Multiple tickers case (shouldn't happen with single ticker, but handle it)
            if ticker in df.columns:
                series = df[ticker].rename(ticker).astype("float32")
            else:
                # Take the first column if ticker name doesn't match exactly
                series = df.iloc[:, 0].rename(ticker).astype("float32")
        
        # Remove any NaN values that might cause issues
        series = series.dropna()
        
        if len(series) == 0:
            raise ValueError(f"All price data is NaN for {ticker!r} between {start} and {end}")
        
        print(f"Final series length: {len(series)}, date range: {series.index[0]} to {series.index[-1]}")
        return series
        
    except Exception as e:
        if "No price data" in str(e) or "All price data is NaN" in str(e):
            raise  # Re-raise our custom errors
        else:
            # Wrap other errors (network, parsing, etc.)
            raise ValueError(f"Failed to load data for {ticker!r}: {str(e)}") from e
