# Data loading utility for historical price data
import yfinance as yf
import pandas as pd

def load_prices(tickers: list[str], days: int) -> pd.DataFrame:
    """Load historical price data for given tickers and time period"""
    df = yf.download(" ".join(tickers),
                     period=f"{days}d",
                     interval="1d",
                     auto_adjust=True)["Close"]
    if isinstance(df.columns, pd.MultiIndex):  # Flatten MultiIndex columns
        df.columns = df.columns.get_level_values(0)
    return df.dropna()