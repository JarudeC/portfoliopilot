# utils/plot_equity.py
"""
Plot cumulative PnL curves for every model that has a results/pnl_*.csv,
aligned on a shared integer-based trading-day axis so comparisons are fair.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ───────────────────────── fixed paths ────────────────────────────────
THIS_DIR   = Path(__file__).resolve().parent
ROOT_DIR   = THIS_DIR.parent
MODELS_DIR = ROOT_DIR / "models"
OUT_PNG    = THIS_DIR / "equity_curves.png"

# ───────────────────────── helpers ────────────────────────────────────
def load_one_pnl(csv_path: Path) -> pd.Series:
    """
    Return a cumulative-PnL Series whose *index is dates* (if present)
    or integers (trading days), zero-based.
    """
    df = pd.read_csv(csv_path)

    # pick the numeric PnL column
    if "PortfolioValue" in df.columns:
        y = pd.to_numeric(df["PortfolioValue"], errors="coerce")
    else:
        y = None
        for col in df.columns:
            if "date" in col.lower():
                continue
            cand = pd.to_numeric(df[col], errors="coerce")
            if cand.notna().any():
                y = cand
                break
        if y is None:
            return pd.Series(dtype=float)

    # pick or fabricate the date index
    if "Date" in df.columns:
        x = pd.to_datetime(df["Date"], errors="coerce")
    elif "date" in df.columns:
        x = pd.to_datetime(df["date"], errors="coerce")
    else:
        x = pd.RangeIndex(len(y))

    s = pd.Series(y.values, index=x).dropna()
    # zero-base so curves start at 0
    return s - s.iloc[0]


def collect_curves() -> dict[str, pd.Series]:
    """Load and align every model’s curve on a shared integer axis."""
    raw: dict[str, pd.Series] = {}

    for res_dir in MODELS_DIR.glob("*/results"):
        pnl_files = [f for f in res_dir.glob("pnl*.csv")]
        if not pnl_files:
            continue
        s = load_one_pnl(pnl_files[0])
        if s.empty:
            print(f"{pnl_files[0].name}: no numeric data – skipped")
            continue
        raw[res_dir.parent.name] = s

    if not raw:
        return {}

    # align on integer-based trading-day index to avoid mixed-type sorting
    max_len = max(len(s) for s in raw.values())
    full_index = pd.RangeIndex(start=0, stop=max_len)
    aligned: dict[str, pd.Series] = {}
    for label, s in raw.items():
        s_int = s.reset_index(drop=True)
        s_aligned = s_int.reindex(full_index).ffill().bfill()
        aligned[label] = s_aligned
    return aligned


def plot_curves(curves: dict[str, pd.Series]) -> None:
    """Plot every aligned curve with the styling of the original script."""
    if not curves:
        print("No PnL files found under models/*/results – nothing to plot.")
        return

    plt.figure(figsize=(10, 5))
    for label, s in curves.items():
        plt.plot(range(len(s)), s.values, label=label)

    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Cumulative PnL Comparison")
    plt.xlabel("Trading days")
    plt.ylabel("PnL (base = 0)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150)
    try:
        plt.show()          # only if a GUI backend is available
    except Exception:
        pass

    print(f"Equity curves written to {OUT_PNG.resolve()}")


if __name__ == "__main__":
    plot_curves(collect_curves())
