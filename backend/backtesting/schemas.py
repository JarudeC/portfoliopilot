"""Pydantic schemas for backtesting API requests and responses."""

from typing import Dict, List, Literal
from pydantic import BaseModel, Field, field_validator


class BacktestRequest(BaseModel):
    """Request schema for portfolio backtest."""
    algo: Literal["Naive Markowitz", "GVMP", "PPN", "Margin Trader"]
    tickers: List[str] = Field(..., min_length=1, max_length=8)
    hist_days: int = Field(default=730, ge=100, le=3650)  # Historical days to fetch
    lookback: int = Field(default=252, ge=20, le=504)     # Estimation window size
    eval_win: int = Field(default=5, ge=1, le=21)         # Rebalancing frequency (days)
    eta: float = Field(default=0.02, ge=0.0, le=1.0)      # Exploration/noise parameter
    tc: float = Field(default=0.001, ge=0.0, le=0.05)     # Transaction cost rate

    @field_validator("tickers")
    @classmethod
    def validate_tickers(cls, v: List[str]) -> List[str]:
        """Uppercase and strip tickers."""
        return [t.strip().upper() for t in v if t.strip()]


class BacktestResult(BaseModel):
    """Response schema for completed backtest."""
    status: Literal["done", "error", "running"]
    algo: str
    nav: Dict[str, float] | None = None      # NAV series: {timestamp: value}
    weights: Dict[str, float] | None = None  # Final weights: {ticker: weight}
    metrics: Dict[str, str] | None = None    # Performance metrics
    detail: str | None = None                # Error message if failed


class MetricsResponse(BaseModel):
    """Performance metrics with formatted string values."""
    Return: str        # Cumulative return (e.g., '15.23%')
    AnnualReturn: str  # Annualized return
    DailyVol: str      # Daily volatility
    AnnualVol: str     # Annualized volatility
    Sharpe: str        # Sharpe ratio
    Sortino: str       # Sortino ratio
