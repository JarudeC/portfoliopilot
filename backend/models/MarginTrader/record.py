"""
Create daily PnL series for Margin Trader
and update root-level utils/Metrics.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────
HERE          = Path(__file__).resolve().parent        # …/MarginTrader
ROOT          = Path(__file__).resolve().parents[2]    # repo root
RESULTS_DIR   = HERE / "results"
CSV_FILE      = RESULTS_DIR / "pnl_margin_trader.csv"  # must exist
METRICS_PATH  = ROOT / "utils" / "Metrics.json"
MODEL_NAME    = "margin_trader"

# ── Metric helpers ─────────────────────────────────────────────────
def compute_metrics(
    portfolio_values: pd.Series,
    trading_days: int = 252,
    rf: float = 0.0,
) -> dict[str, str]:
    """
    Compute standardized metrics matching other models in the toolkit.
    """
    # Clean the data
    portfolio_vals = portfolio_values.dropna().astype(float)
    
    if len(portfolio_vals) < 2:
        # Return zero metrics for insufficient data
        return {
            "Return": "0.00%",
            "AnnualReturn": "0.00%",
            "DailyVol": "0.00%", 
            "AnnualVol": "0.00%",
            "Sharpe": "0.00",
            "Sortino": "0.00"
        }

    # Calculate returns
    daily_returns = portfolio_vals.pct_change().dropna()
    
    # Basic metrics
    cumulative_return = (portfolio_vals.iloc[-1] / portfolio_vals.iloc[0]) - 1
    n_days = len(daily_returns)
    
    # Annualized return - geometric method like other models
    if n_days > 0 and portfolio_vals.iloc[0] > 0:
        ann_ret = (portfolio_vals.iloc[-1] / portfolio_vals.iloc[0]) ** (trading_days / n_days) - 1
    else:
        ann_ret = 0.0

    # Volatility
    daily_vol = daily_returns.std() if len(daily_returns) > 1 else 0.0
    annual_vol = daily_vol * np.sqrt(trading_days)

    # Sharpe ratio
    if annual_vol > 0:
        sharpe = (ann_ret - rf) / annual_vol
    else:
        sharpe = np.nan

    # Sortino ratio
    downside_returns = daily_returns[daily_returns < 0]
    if len(downside_returns) > 1:
        downside_vol = downside_returns.std() * np.sqrt(trading_days)
        sortino = (ann_ret - rf) / downside_vol if downside_vol > 0 else np.nan
    else:
        sortino = np.nan

    return {
        "Return":       f"{cumulative_return * 100:.2f}%",
        "AnnualReturn": f"{ann_ret * 100:.2f}%",
        "DailyVol":     f"{daily_vol * 100:.2f}%",
        "AnnualVol":    f"{annual_vol * 100:.2f}%",
        "Sharpe":       f"{sharpe:.2f}" if not np.isnan(sharpe) else "0.00",
        "Sortino":      f"{sortino:.2f}" if not np.isnan(sortino) else "0.00"
    }


def update_metrics(model: str, metrics: dict[str, str], path: Path) -> None:
    """Merge/overwrite `metrics` for this model into utils/Metrics.json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            existing = {}
    existing[model] = metrics
    path.write_text(json.dumps(existing, indent=4), encoding="utf-8")


# ── Main ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"{CSV_FILE} not found. Run train.py first.")

    try:
        df = pd.read_csv(CSV_FILE)
        
        # Handle different possible column names
        pv_col = None
        for col in ["PortfolioValue", "NAV", "portfolio_value", "account_value"]:
            if col in df.columns:
                pv_col = col
                break
        
        if pv_col is None:
            raise KeyError(f"Could not find portfolio value column. Available columns: {df.columns.tolist()}")
        
        portfolio_vals = df[pv_col]
        
        # Convert to Series with proper index if needed
        if "Date" in df.columns:
            dates = pd.to_datetime(df["Date"], errors='coerce')
            portfolio_vals = pd.Series(portfolio_vals.values, index=dates, name=pv_col)
        
        metrics = compute_metrics(portfolio_vals)
        update_metrics(MODEL_NAME, metrics, METRICS_PATH)

        print(f"Metrics for {MODEL_NAME} saved to {METRICS_PATH}")
        print("\nMetrics Summary:")
        for k, v in metrics.items():
            print(f"{k:12} {v}")
            
    except Exception as e:
        print(f"Error processing metrics: {e}")
        # Write fallback metrics
        fallback_metrics = {
            "Return": "0.00%",
            "AnnualReturn": "0.00%",
            "DailyVol": "0.00%",
            "AnnualVol": "0.00%", 
            "Sharpe": "0.00",
            "Sortino": "0.00"
        }
        update_metrics(MODEL_NAME, fallback_metrics, METRICS_PATH)
        print(f"Fallback metrics saved for {MODEL_NAME}")