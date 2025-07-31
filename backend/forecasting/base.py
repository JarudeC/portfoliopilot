# Base classes and utilities for forecasting models
# Contains request/response schemas and data loading functions

from __future__ import annotations

from datetime import date, datetime
from typing import List

import pandas as pd
import yfinance as yf
from pydantic import BaseModel, Field, validator


# Request schema for forecasting endpoints
class ForecastRequest(BaseModel):
    """Request schema for all forecasting endpoints"""

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

    # Field validators
    @validator("ticker")
    def _upper(cls, v: str) -> str:  # noqa: N805
        return v.strip().upper()

    @validator("end")
    def _end_not_future(cls, v: date) -> date:  # noqa: N805
        today = datetime.utcnow().date()
        if v > today:
            raise ValueError(f"end date {v} cannot be in the future (today: {today})")
        # Check data availability limits
        max_reasonable_date = date(2024, 12, 31)
        if v > max_reasonable_date:
            raise ValueError(f"end date {v} is beyond reliable stock data availability. Use {max_reasonable_date} or earlier.")
        return v

    @validator("start")
    def _logical_window(cls, v: date, values) -> date:  # noqa: N805
        end = values.get("end")
        if end and v >= end:
            raise ValueError("start must be earlier than end")
        return v


# Data loading utility
def load_series(ticker: str, start: date, end: date) -> pd.Series:
    """Download adjusted close prices from Yahoo Finance"""
    # Debug info
    print(f"Loading data for {ticker} from {start} to {end}")
    
    try:
        # Match data_loader.py pattern
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)["Close"]
        print(f"Downloaded data shape: {df.shape}")
        
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if df.empty:
            raise ValueError(f"No price data returned for {ticker!r} between {start} and {end}. Check if ticker exists and dates are valid.")
        
        # Handle single/multiple ticker responses
        if isinstance(df, pd.Series):
            series = df.rename(ticker).astype("float32")
        else:
            # Multiple tickers fallback
            if ticker in df.columns:
                series = df[ticker].rename(ticker).astype("float32")
            else:
                # Use first column if ticker name doesn't match
                series = df.iloc[:, 0].rename(ticker).astype("float32")
        
        # Clean NaN values
        series = series.dropna()
        
        if len(series) == 0:
            raise ValueError(f"All price data is NaN for {ticker!r} between {start} and {end}")
        
        print(f"Final series length: {len(series)}, date range: {series.index[0]} to {series.index[-1]}")
        return series
        
    except Exception as e:
        if "No price data" in str(e) or "All price data is NaN" in str(e):
            raise
        else:
            # Wrap network/parsing errors
            raise ValueError(f"Failed to load data for {ticker!r}: {str(e)}") from e
