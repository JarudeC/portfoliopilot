"""
Shared metrics calculation for all backtesting strategies.

Consolidates duplicated record.py logic from individual models.
"""

import numpy as np
import pandas as pd
from typing import Dict


def calculate_metrics(
    nav: pd.Series,
    trading_days: int = 252,
    risk_free_rate: float = 0.0
) -> Dict[str, float]:
    """
    Calculate performance metrics from NAV series.

    Args:
        nav: Net Asset Value time series (starting at 1.0)
        trading_days: Trading days per year for annualization
        risk_free_rate: Annual risk-free rate

    Returns:
        Dict with raw metric values (not formatted)
    """
    nav = nav.dropna().astype(float)

    if len(nav) < 2:
        return _empty_metrics()

    returns = nav.pct_change().dropna()

    if len(returns) == 0:
        return _empty_metrics()

    # Cumulative and annualized returns
    cum_return = nav.iloc[-1] / nav.iloc[0] - 1
    ann_return = (nav.iloc[-1] / nav.iloc[0]) ** (trading_days / len(nav)) - 1

    # Volatility
    daily_vol = returns.std()
    ann_vol = daily_vol * np.sqrt(trading_days)

    # Risk-adjusted metrics
    excess_return = ann_return - risk_free_rate
    sharpe = excess_return / ann_vol if ann_vol > 0 else np.nan

    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(trading_days) if len(downside_returns) > 0 else 0
    sortino = excess_return / downside_vol if downside_vol > 0 else np.nan

    # Maximum drawdown
    rolling_max = nav.cummax()
    drawdown = (nav - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    return {
        "cum_return": cum_return,
        "ann_return": ann_return,
        "daily_vol": daily_vol,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
    }


def format_metrics(metrics: Dict[str, float]) -> Dict[str, str]:
    """
    Format raw metrics as display strings.

    Args:
        metrics: Raw metric values from calculate_metrics()

    Returns:
        Dict with formatted strings (percentages, ratios)
    """
    def fmt_pct(v: float) -> str:
        if not np.isfinite(v) or abs(v) > 100:
            return "N/A"
        return f"{v * 100:.2f}%"

    def fmt_ratio(v: float) -> str:
        if not np.isfinite(v):
            return "N/A"
        return f"{v:.2f}"

    return {
        "Return": fmt_pct(metrics.get("cum_return", np.nan)),
        "AnnualReturn": fmt_pct(metrics.get("ann_return", np.nan)),
        "DailyVol": fmt_pct(metrics.get("daily_vol", np.nan)),
        "AnnualVol": fmt_pct(metrics.get("ann_vol", np.nan)),
        "Sharpe": fmt_ratio(metrics.get("sharpe", np.nan)),
        "Sortino": fmt_ratio(metrics.get("sortino", np.nan)),
    }


def compute_metrics_from_nav(
    nav: pd.Series,
    td: int = 252,
    rf: float = 0.0
) -> Dict[str, str]:
    """
    Convenience function matching old record.py interface.

    Args:
        nav: NAV series
        td: Trading days per year
        rf: Risk-free rate

    Returns:
        Formatted metrics dict
    """
    raw = calculate_metrics(nav, trading_days=td, risk_free_rate=rf)
    return format_metrics(raw)


def _empty_metrics() -> Dict[str, float]:
    """Return empty metrics dict when calculation not possible."""
    return {
        "cum_return": np.nan,
        "ann_return": np.nan,
        "daily_vol": np.nan,
        "ann_vol": np.nan,
        "sharpe": np.nan,
        "sortino": np.nan,
        "max_drawdown": np.nan,
    }
