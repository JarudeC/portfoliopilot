"""
record.py for CA-GMVP — uses NAV_CA_GMVP.csv (column: NAV).
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

def compute_metrics_from_nav(nav: pd.Series, td: int = 252, rf: float = 0.0):
    nav = nav.dropna().astype(float)
    if (nav <= 0).any():
        raise ValueError("NAV must remain strictly positive.")

    r_hat = nav.pct_change().dropna()
    cum_ret   = nav.iloc[-1] - 1
    ann_ret   = nav.iloc[-1] ** (td / len(nav)) - 1
    daily_vol = r_hat.std()
    ann_vol   = daily_vol * np.sqrt(td)
    sharpe    = np.nan if ann_vol == 0 else (ann_ret - rf) / ann_vol
    downside  = r_hat[r_hat < 0].std()
    sortino   = np.nan if downside == 0 else (ann_ret - rf) / (downside * np.sqrt(td))

    return {
        "Return":        f"{cum_ret  * 100:.2f}%",
        "AnnualReturn":  f"{ann_ret * 100:.2f}%",   
        "DailyVol":      f"{daily_vol * 100:.2f}%",
        "AnnualVol":     f"{ann_vol  * 100:.2f}%",
        "Sharpe":        f"{sharpe:.2f}",
        "Sortino":       f"{sortino:.2f}",
    }

def update_metrics(model_name: str, metrics: dict, path: Path):
    os.makedirs(path.parent, exist_ok=True)
    all_metrics = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            all_metrics = json.load(f)
    all_metrics[model_name] = metrics
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    HERE         = Path(__file__).resolve().parent
    results_dir  = HERE / "results"
    ROOT         = Path(__file__).resolve().parents[2]  # project root
    metrics_path = ROOT / "utils" / "Metrics.json"      # correct file
    model_name   = "CA-GMVP"

    nav_file = results_dir / "pnl_CA_GMVP.csv"
    if not nav_file.exists():
        raise FileNotFoundError(f"NAV file not found: {nav_file}")
    df = pd.read_csv(nav_file)

    # Accept either 'NAV' or 'PortfolioValue'
    nav_col = "NAV" if "NAV" in df.columns else "PortfolioValue"
    nav = df[nav_col]

    metrics = compute_metrics_from_nav(nav)
    update_metrics(model_name, metrics, metrics_path)

    print(f"Metrics for {model_name} saved to {metrics_path}")
