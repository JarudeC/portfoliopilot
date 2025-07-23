import json, pandas as pd, numpy as np, math
from pathlib import Path

def calc_metrics(daily: pd.Series, trading_days=252):
    total_return = (1 + daily).prod() - 1
    ar = (1 + total_return) ** (trading_days / len(daily)) - 1
    daily_vol = daily.std()
    annual_vol = daily_vol * math.sqrt(trading_days)
    sharpe = daily.mean() / daily_vol * math.sqrt(trading_days)

    downside = daily[daily < 0]
    sortino = float('inf') if downside.empty else daily.mean() / downside.std() * math.sqrt(trading_days)

    return {
        "Return": f"{total_return * 100:.2f}%",
        "AR": f"{ar * 100:.2f}%",
        "Daily Vol": f"{daily_vol * 100:.2f}%",
        "Annual Vol": f"{annual_vol * 100:.2f}%",
        "Sharpe": f"{sharpe:.2f}",
        "Sortino": f"{sortino:.2f}" if sortino != float("inf") else "∞"
    }

# Centralised metrics store
all_metrics = {}
metrics_path = Path("utils/Metrics.json")

# Loop through model folders
for model_dir in Path("models").iterdir():
    results_dir = model_dir / "results"
    if not results_dir.exists():
        print(f"Skipped: {model_dir.name} (no results folder)")
        continue

    returns_files = list(results_dir.glob("Overall_return*.csv"))
    if not returns_files:
        print(f"No Overall_return*.csv found in {results_dir}")
        continue

    daily = pd.read_csv(returns_files[0])
    if "return" in daily.columns:
        daily = daily["return"]
    else:
        daily = daily.iloc[:, 0]

    model_name = model_dir.name.replace("_", " ")
    all_metrics[model_name] = calc_metrics(daily)

# Save all to utils
metrics_path.parent.mkdir(parents=True, exist_ok=True)
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(all_metrics, f, indent=4)

print(f"\nCentralised Metrics written to {metrics_path}")

