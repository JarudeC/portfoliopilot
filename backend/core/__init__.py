"""Core shared utilities for portfolio backtesting and forecasting.

This module contains shared components used across multiple modules,
including data loading, common configurations, and utility functions.
"""

from .data_loader import load_series, load_series_batch

__all__ = ['load_series', 'load_series_batch']
