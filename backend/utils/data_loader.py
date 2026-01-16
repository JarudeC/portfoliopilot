"""Data loading utility for historical price data.

This module re-exports functions from forecasting.data_loader to maintain
backward compatibility with existing code while consolidating implementation.
"""

from forecasting.data_loader import load_prices, load_series

__all__ = ['load_prices', 'load_series']
