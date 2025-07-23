# utils/compare_all.py
"""
Run every models/*/record.py, then build a unified performance table
from utils/Metrics.json.

Usage
-----
python utils/compare_all.py
"""

from __future__ import annotations
import subprocess, sys, json, textwrap
from pathlib import Path

import pandas as pd

# ─── Paths ─────────────────────────────────────────────────────────
try:
    ROOT = Path(__file__).resolve().parents[1]  # normal run
except NameError:
    # __file__ doesn't exist if run via VS Code's Run menu
    ROOT = Path.cwd().parent

MODELS       = ROOT / "models"
METRIC_JSON  = ROOT / "utils" / "Metrics.json"
OUTTXT       = ROOT / "utils" / "performance_summary.txt"

# ─── 1. Run each record.py ─────────────────────────────────────────
print("\nExecuting every models/*/record.py …\n")
for rec in MODELS.rglob("record.py"):
    rel = rec.relative_to(ROOT)
    print(f"{rel}")
    res = subprocess.run([sys.executable, str(rec)],
                         capture_output=True, text=True)
    if res.returncode:
        print("ERROR:\n", textwrap.indent(res.stderr, "      "))
    else:
        print("finished")

# ─── 2. Load consolidated JSON ─────────────────────────────────────
if not METRIC_JSON.exists():
    sys.exit(f"\n{METRIC_JSON} not found. Did the record scripts write it?")

with open(METRIC_JSON, encoding="utf-8") as fh:
    metrics_dict: dict[str, dict[str, str]] = json.load(fh)

# ─── 3. Build DataFrame of all metrics ────────────────────────────
rows, all_keys = [], set()
for model, met in metrics_dict.items():
    all_keys.update(met.keys())
    rows.append({"Model": model, **met})

cols = ["Model"] + sorted(all_keys)
df   = pd.DataFrame(rows)[cols].fillna("-")

# ─── 4. Pretty fixed-width table ──────────────────────────────────
width = {c: max(len(c), df[c].astype(str).str.len().max()) + 2 for c in df.columns}

def fmt(val, w):
    sval = str(val)
    # Right-align if it's a pure number (not %, not 'nan', not empty)
    is_number = False
    try:
        float(sval)
        is_number = True
    except ValueError:
        pass
    return sval.rjust(w) if is_number else sval.ljust(w)

header = " | ".join(c.ljust(width[c]) for c in df.columns)
line   = "-+-".join("-" * width[c] for c in df.columns)
body   = "\n".join(
    " | ".join(fmt(row[c], width[c]) for c in df.columns)
    for _, row in df.iterrows()
)

summary = f"=== Performance Summary ===\n{header}\n{line}\n{body}"
print("\n" + summary)

OUTTXT.write_text(summary, encoding="utf-8")
print(f"\nSummary written to {OUTTXT}")
