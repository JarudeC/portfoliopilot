"""
Create daily PnL series for PortfolioPolicyNetwork and update utils/Metrics.json
"""
import json, os
from pathlib import Path
import pandas as pd, numpy as np

# ------------------------------------------------------------------
# »> paths that work from **any** working-directory
# ------------------------------------------------------------------
HERE  = Path(__file__).resolve().parent              # …/models/PortfolioPolicyNetwork
ROOT  = HERE.parent.parent                           # project root

results_dir  = HERE / "results"                      # …/results/
csv_file     = results_dir / "pnl_PPN.csv"
metrics_path = ROOT / "utils" / "Metrics.json"
model_name   = "PPN"

# ------------------------------------------------------------------
def compute_metrics(portfolio_values: pd.Series,
                    trading_days: int = 252,
                    rf: float = 0.0) -> dict:
    daily_returns   = portfolio_values.pct_change().dropna()
    cumulative_ret  = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
    n_days          = len(daily_returns)
    annualised_ret  = (1 + cumulative_ret) ** (trading_days / n_days) - 1 if n_days else np.nan
    daily_vol       = daily_returns.std()
    annual_vol      = daily_vol * np.sqrt(trading_days)
    sharpe          = np.nan if annual_vol == 0 else (annualised_ret - rf) / annual_vol
    downside_std    = daily_returns[daily_returns < 0].std()
    sortino         = np.nan if downside_std == 0 else (annualised_ret - rf) / (downside_std * np.sqrt(trading_days))

    return {
        "Return":     f"{cumulative_ret  * 100:.2f}%",
        "AnnualReturn":         f"{annualised_ret * 100:.2f}%",
        "DailyVol":  f"{daily_vol       * 100:.2f}%",
        "AnnualVol": f"{annual_vol      * 100:.2f}%",
        "Sharpe":     f"{sharpe:.2f}",
        "Sortino":    f"{sortino:.2f}",
    }

def update_metrics(model_name: str, metrics: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            all_metrics = json.load(f)
    all_metrics[model_name] = metrics
    with path.open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4)

# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
if not csv_file.exists():
    raise FileNotFoundError(f"{csv_file} not found. Run train.py first.")

df = pd.read_csv(csv_file)          # Date column position is fine
portfolio_values = df["PortfolioValue"]

metrics = compute_metrics(portfolio_values)
update_metrics(model_name, metrics, metrics_path)

print(f"Metrics for {model_name} saved to {metrics_path}")