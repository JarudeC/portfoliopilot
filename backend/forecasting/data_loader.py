"""Unified data loading utilities for financial time series.

This module consolidates all data loading logic from the forecasting module,
providing a single source of truth for fetching stock price data from Yahoo Finance.
Eliminates duplication between forecasting.base and utils.data_loader.
"""

import logging
from datetime import date
from typing import List

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def load_series(
    ticker: str,
    start: date,
    end: date,
    validate: bool = True
) -> pd.Series:
    """
    Download adjusted close prices from Yahoo Finance for a single ticker.

    This function is used by forecasting algorithms to fetch historical price data
    for model training and prediction.

    Args:
        ticker: Yahoo Finance ticker symbol (e.g., "AAPL", "MSFT")
        start: Inclusive start date for data window
        end: Exclusive end date for data window
        validate: If True, raise errors on empty/invalid data; if False, log warnings

    Returns:
        pandas Series with DatetimeIndex and float32 price values

    Raises:
        ValueError: If no data returned, all data is NaN, or download fails (when validate=True)

    Notes:
        - Automatically handles MultiIndex columns when yfinance returns multiple tickers
        - Returns business day frequency for financial data
        - Drops NaN values to ensure clean data for modeling
    """
    logger.info(f"Loading data for {ticker} from {start} to {end}")

    try:
        # Download data using yfinance
        df = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True
        )

        logger.debug(f"Downloaded data shape: {df.shape}")

        # Check for empty DataFrame
        if df.empty:
            msg = (
                f"No price data returned for {ticker!r} between {start} and {end}. "
                f"Check if ticker exists and dates are valid."
            )
            if validate:
                raise ValueError(msg)
            logger.warning(msg)
            return pd.Series(dtype='float32')

        # Extract Close prices from the DataFrame
        # Handle both single-level and MultiIndex columns
        if "Close" in df.columns:
            df = df["Close"]
        elif isinstance(df.columns, pd.MultiIndex) and "Close" in df.columns.get_level_values(0):
            df = df["Close"]
        else:
            # If no Close column, assume the data is already price data
            logger.debug(f"No 'Close' column found, using raw data")

        # Handle MultiIndex columns after selecting Close (can happen with multiple tickers)
        if isinstance(df, pd.DataFrame) and isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Handle Series vs DataFrame response
        if isinstance(df, pd.Series):
            series = df.rename(ticker).astype('float32')
        else:
            # DataFrame - select the requested ticker
            if ticker in df.columns:
                series = df[ticker].rename(ticker).astype('float32')
            else:
                # Use first column if ticker name doesn't match
                logger.debug(
                    f"Ticker {ticker} not in columns {list(df.columns)}, "
                    f"using first column"
                )
                series = df.iloc[:, 0].rename(ticker).astype('float32')

        # Clean NaN values
        series = series.dropna()

        # Validate we have actual data
        if len(series) == 0:
            msg = f"All price data is NaN for {ticker!r} between {start} and {end}"
            if validate:
                raise ValueError(msg)
            logger.warning(msg)

        logger.info(
            f"Loaded {len(series)} observations from "
            f"{series.index[0].date()} to {series.index[-1].date()}"
        )
        return series

    except Exception as e:
        # Re-raise validation errors as-is
        if "No price data" in str(e) or "All price data is NaN" in str(e):
            raise
        else:
            # Wrap network/parsing errors with context
            raise ValueError(
                f"Failed to load data for {ticker!r}: {str(e)}"
            ) from e


def load_prices(tickers: List[str], days: int) -> pd.DataFrame:
    """
    Load historical price data for multiple tickers.

    This function is used by portfolio backtesting algorithms to fetch
    multi-asset price histories.

    Args:
        tickers: List of ticker symbols (e.g., ["AAPL", "MSFT", "GOOGL"])
        days: Number of days of historical data to fetch

    Returns:
        DataFrame with tickers as columns and dates as index

    Notes:
        - Used by portfolio backtesting module (not forecasting)
        - Automatically drops NaN values
        - Flattens MultiIndex columns from yfinance
    """
    logger.info(f"Loading {days} days of data for {len(tickers)} tickers")

    df = yf.download(
        " ".join(tickers),
        period=f"{days}d",
        interval="1d",
        auto_adjust=True,
        progress=False
    )["Close"]

    # Flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    result = df.dropna()
    logger.info(
        f"Loaded {len(result)} observations for {len(result.columns)} tickers"
    )
    return result
