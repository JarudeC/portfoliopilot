"""
Backtesting module for portfolio optimization strategies.

Strategies: Naive Markowitz, GMVP Clustering, Policy Network, Margin Trader.
All strategies return (NAV, weights, metrics).
"""

from .config import (
    NaiveMarkowitzConfig,
    GMVPClusteringConfig,
    PolicyNetworkConfig,
    MarginTraderConfig,
)
from .metrics import calculate_metrics, format_metrics
from .base import BaseStrategy

__all__ = [
    "NaiveMarkowitzConfig",
    "GMVPClusteringConfig",
    "PolicyNetworkConfig",
    "MarginTraderConfig",
    "BaseStrategy",
    "calculate_metrics",
    "format_metrics",
]
