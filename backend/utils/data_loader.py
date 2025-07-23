import yfinance as yf
import pandas as pd

def load_prices(tickers: list[str], days: int) -> pd.DataFrame:
    """
    Returns an (T, N) price frame indexed by date, columns = tickers.
    `days` is total look-back history you want to feed the model.
    """
    df = yf.download(" ".join(tickers),
                     period=f"{days}d",
                     interval="1d",
                     auto_adjust=True)["Close"]
    if isinstance(df.columns, pd.MultiIndex):          # flatten if needed
        df.columns = df.columns.get_level_values(0)
    return df.dropna()