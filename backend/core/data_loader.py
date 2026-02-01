"""Unified data loading utilities for financial time series.

This module provides the single source of truth for fetching stock price data
from Yahoo Finance. Used by forecasting, backtesting, and price endpoints.

Functions:
    load_series: Load price data for a single ticker (base function)
    load_series_batch: Load price data for multiple tickers in parallel using ThreadPoolExecutor
                       (used by /prices/batch, /forecast/{algo}/batch, and backtest)
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import Dict, List, Union

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

    # Normalize ticker to uppercase for consistency
    ticker = ticker.upper().strip()

    try:
        # Create a fresh Ticker object to avoid caching issues
        yf_ticker = yf.Ticker(ticker)
        df = yf_ticker.history(
            start=start,
            end=end,
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
        # Ticker.history() returns DataFrame with standard columns: Open, High, Low, Close, Volume
        if "Close" in df.columns:
            series = df["Close"].rename(ticker).astype('float32')
        else:
            # Fallback: use first column if no Close column found
            logger.warning(f"No 'Close' column found for {ticker}, columns: {list(df.columns)}")
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


def load_series_batch(
    tickers: List[str],
    start: date,
    end: date,
    return_errors: bool = False
) -> Union[pd.DataFrame, Dict[str, Union[pd.Series, Dict[str, str]]]]:
    """
    Load price data for multiple tickers in parallel using ThreadPoolExecutor.

    This is the primary batch loading function, used by:
        - /prices/batch endpoint (returns dict with errors)
        - /forecast/{algo}/batch endpoint (returns dict with errors)
        - _backtest_worker (returns DataFrame)

    Args:
        tickers: List of ticker symbols (e.g., ["AAPL", "MSFT", "GOOGL"])
        start: Inclusive start date for data window
        end: Exclusive end date for data window
        return_errors: If True, return dict with {ticker: Series or {error: msg}}
                      If False, return DataFrame (drops failed tickers)

    Returns:
        If return_errors=True: Dict mapping ticker -> Series or {"error": str}
        If return_errors=False: DataFrame with tickers as columns, dates as index

    Notes:
        - Uses ThreadPoolExecutor for parallel I/O (network calls release GIL)
        - Max 10 workers to avoid rate limiting
        - Reuses load_series() for consistent single-ticker logic
    """
    logger.info(f"Loading batch data for {len(tickers)} tickers from {start} to {end}")

    results: Dict[str, Union[pd.Series, Dict[str, str]]] = {}
    max_workers = min(len(tickers), 10)

    def fetch_single(ticker: str) -> tuple:
        try:
            series = load_series(ticker, start, end)
            return ticker, series
        except Exception as e:
            logger.warning(f"Failed to load {ticker}: {e}")
            return ticker, {"error": str(e)}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_single, t): t for t in tickers}
        for future in as_completed(futures):
            ticker, data = future.result()
            results[ticker] = data

    if return_errors:
        # Return dict with errors included (for API endpoints)
        logger.info(f"Batch load complete: {len(results)} tickers processed")
        return results
    else:
        # Return DataFrame, dropping failed tickers (for backtest)
        valid_series = {k: v for k, v in results.items() if isinstance(v, pd.Series)}
        if not valid_series:
            raise ValueError("No valid price data for any ticker")
        df = pd.DataFrame(valid_series).dropna()
        logger.info(f"Batch load complete: {len(df)} observations for {len(df.columns)} tickers")
        return df
