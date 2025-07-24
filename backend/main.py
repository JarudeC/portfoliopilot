# backend/main.py   ──  uvicorn main:app --reload --port 8000
"""FastAPI entry‑point for
  • Portfolio back‑tests (/train …)
  • Price forecasting (/forecast …)

We deliberately keep the two job lanes isolated so that long‑running
Autoformer forecasts never block RL back‑tests.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, List, Literal, Callable, Tuple
from uuid import uuid4

import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from utils.data_loader import load_prices        # unchanged

# ─── forecasting package imports (local) ────────────────────────────────
from forecasting import arima, autoformer, base, lstm

# ────────────────────────────────────────────────────────────────────────
app = FastAPI()

# ======================================================================
# 1) PORTFOLIO‑TRAINING  (unchanged logic – just namespaced)
# ======================================================================
ALGO_MAP: Dict[str, str] = {
    "Naive Markowitz": "models.NaiveMarkowitz.api",
    "GVMP": "models.GMVP_Clustering.api",
    "PPN": "models.PortfolioPolicyNetwork.api",
    "Margin Trader": "models.MarginTrader.api",
}


class TrainReq(BaseModel):
    algo: Literal[tuple(ALGO_MAP.keys())]  # type: ignore[arg-type]
    tickers: List[str] = Field(..., min_items=1, max_items=8)
    hist_days: int = 365
    lookback: int = 252
    eval_win: int = 5
    eta: float = 0.02
    tc: float = 0.002


_train_jobs: Dict[str, Dict[str, Any]] = {}


@app.post("/train")
def launch_backtest(req: TrainReq, bt: BackgroundTasks):
    jid = uuid4().hex
    _train_jobs[jid] = {"status": "queued"}
    bt.add_task(_train_worker, jid, req)
    return {"job_id": jid}


def _train_worker(jid: str, req: TrainReq):
    try:
        prices = load_prices(req.tickers, req.hist_days)
        api_mod = import_module(ALGO_MAP[req.algo])
        nav, weights, metrics = api_mod.run(
            prices,
            lookback=req.lookback,
            eval_win=req.eval_win,
            eta=req.eta,
            tc=req.tc,
        )

        nav_json = {
            str(ts): float(v)
            for ts, v in nav.replace([np.inf, -np.inf], np.nan).dropna().items()
        }
        _train_jobs[jid] = {
            "status": "done",
            "nav": nav_json,
            "weights": weights,
            "metrics": metrics,
        }
    except Exception as exc:  # noqa: BLE001
        _train_jobs[jid] = {"status": "error", "detail": str(exc)}


@app.get("/train/{jid}")
def train_status(jid: str):
    if jid not in _train_jobs:
        raise HTTPException(404, "Job not found")
    return _train_jobs[jid]


# ======================================================================
# 2) FORECASTING – cleaner, dictionary‑driven implementation
# ======================================================================

# map algo → callable[ForecastRequest, Tuple[dates, hist_values, fc_dates, fc_values]]
_SYNC_FORECASTERS: Dict[str, Callable[[base.ForecastRequest], Tuple[List[str], List[float], List[str], List[float]]]] = {
    "arima": arima.forecast,
    "lstm": lstm.forecast,
    "autoformer": autoformer.forecast,
}

_ASYNC_FORECASTERS: Dict[str, Callable[[base.ForecastRequest], Tuple[List[str], List[float], List[str], List[float]]]] = {
}


_forecast_jobs: Dict[str, Dict[str, Any]] = {}


def _payload(hd: List[str], hv: List[float], fd: List[str], fv: List[float]) -> Dict[str, Any]:
    return {
        "history_dates": hd,
        "history_values": hv,
        "forecast_dates": fd,
        "forecast_values": fv,
    }


@app.post("/forecast/{algo}")
def forecast(algo: Literal["arima", "lstm", "autoformer"], req: base.ForecastRequest, bg: BackgroundTasks):
    """Route dispatcher – behaves synchronously for cheap models, async for heavy."""

    if algo in _SYNC_FORECASTERS:
        try:
            hd, hv, fd, fv = _SYNC_FORECASTERS[algo](req)
            return _payload(hd, hv, fd, fv)
        except Exception as e:
            raise HTTPException(500, f"Forecasting error: {str(e)}")

    # async branch
    if algo in _ASYNC_FORECASTERS:
        task_id = uuid4().hex
        _forecast_jobs[task_id] = {"status": "running"}
        bg.add_task(_async_wrapper, algo, req, task_id)
        return {"task_id": task_id, "status": "running"}

    raise HTTPException(400, "Unknown forecasting algorithm")


def _async_wrapper(algo_key: str, req: base.ForecastRequest, tid: str) -> None:
    try:
        hd, hv, fd, fv = _ASYNC_FORECASTERS[algo_key](req)
        _forecast_jobs[tid] = {"status": "done", **_payload(hd, hv, fd, fv)}
    except Exception as exc:  # noqa: BLE001
        _forecast_jobs[tid] = {"status": "error", "detail": str(exc)}


@app.get("/forecast/result/{task_id}")
def forecast_result(task_id: str):
    job = _forecast_jobs.get(task_id)
    if job is None:
        raise HTTPException(404, "task_id not found")
    return job
