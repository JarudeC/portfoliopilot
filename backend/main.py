# backend/main.py   ──  uvicorn main:app --reload --port 8000
from __future__ import annotations

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from importlib import import_module
from typing import Dict, List, Literal, Any
from uuid import uuid4
import numpy as np

from utils.data_loader import load_prices   # unchanged

app: FastAPI = FastAPI()
jobs: Dict[str, Dict[str, Any]] = {}        # in-mem job store

# ─── route strings to their api modules ───────────────────────────────────
ALGO_MAP: Dict[str, str] = {
    "Naive Markowitz": "models.NaiveMarkowitz.api",
    "GVMP":            "models.GMVP_Clustering.api",
    "PPN":             "models.PortfolioPolicyNetwork.api",
    "Margin Trader":   "models.MarginTrader.api",
}

# ─── request schema ───────────────────────────────────────────────────────
class TrainReq(BaseModel):
    algo: Literal["Naive Markowitz", "GVMP", "PPN", "Margin Trader"]
    tickers: List[str] = Field(..., min_items=1, max_items=8)
    hist_days: int     = 365
    lookback:  int     = 252
    eval_win:  int     = 5
    eta:       float   = 0.02      # ignored by some algos but harmless
    tc:        float   = 0.002

# ─── endpoints ────────────────────────────────────────────────────────────
@app.post("/train")
def launch(req: TrainReq, bt: BackgroundTasks):
    jid = str(uuid4())
    jobs[jid] = {"status": "queued"}
    bt.add_task(worker, jid, req)
    return {"job_id": jid}

def worker(jid: str, req: TrainReq):
    try:
        # 1) download prices once
        prices = load_prices(req.tickers, req.hist_days)

        # 2) dynamic import → api.run()
        api_mod = import_module(ALGO_MAP[req.algo])
        nav, weights, metrics = api_mod.run(
            prices,
            lookback = req.lookback,
            eval_win = req.eval_win,
            eta      = req.eta,
            tc       = req.tc,
        )

        nav_json = {
            str(ts): float(v)
            for ts, v in nav.replace([np.inf, -np.inf], np.nan).dropna().items()
        }
        jobs[jid] = {
            "status":  "done",
            "nav":     nav_json,
            "weights": weights,
            "metrics": metrics,
        }

    except Exception as e:
        jobs[jid] = {"status": "error", "detail": str(e)}

@app.get("/train/{jid}")
def status(jid: str):
    if jid not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[jid]
