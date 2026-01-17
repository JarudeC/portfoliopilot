# FastAPI backend for portfolio backtesting and price forecasting
# Provides REST endpoints for running portfolio algorithms and time series models

from __future__ import annotations

import logging
import os
from importlib import import_module
from typing import Any, Dict, List, Literal, Callable, Tuple
from uuid import uuid4

import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from utils.data_loader import load_prices, load_series

# Configure logging to show INFO level messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Forecasting model imports
from forecasting import arima, autoformer, lstm
from forecasting.schemas import ForecastRequest


app = FastAPI()

# CORS middleware for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://portfoliopilot-research.com",
        "https://www.portfoliopilot-research.com",
        "http://localhost:3000",  # Local development
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint for Railway
@app.get("/health")
def health():
    """Health check endpoint for deployment monitoring."""
    return {"status": "ok"}

# Historical Price Data Endpoints

class PriceRequest(BaseModel):
    """Request schema for historical price data."""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")
    start: str = Field(..., description="Start date (YYYY-MM-DD)")
    end: str = Field(..., description="End date (YYYY-MM-DD)")


@app.post("/prices")
def get_prices(req: PriceRequest):
    """
    Fetch historical stock prices from Yahoo Finance.

    This is a dedicated endpoint for fetching raw price data without
    running any forecasting models. Use this when you only need historical
    data (e.g., for Custom AI Strategy, data visualization, etc.)

    Returns:
        ticker: The requested ticker symbol
        dates: List of date strings (YYYY-MM-DD)
        prices: List of adjusted close prices
    """
    from datetime import datetime

    try:
        start_date = datetime.strptime(req.start, "%Y-%m-%d").date()
        end_date = datetime.strptime(req.end, "%Y-%m-%d").date()
    except ValueError as e:
        raise HTTPException(400, f"Invalid date format. Use YYYY-MM-DD. Error: {str(e)}")

    try:
        series = load_series(req.ticker, start_date, end_date)

        return {
            "ticker": req.ticker,
            "dates": series.index.strftime("%Y-%m-%d").tolist(),
            "prices": series.tolist()
        }
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch prices: {str(e)}")

# Portfolio algorithm mapping
ALGO_MAP: Dict[str, str] = {
    "Naive Markowitz": "backtesting.naive_markowitz.api",
    "GVMP": "backtesting.gmvp_clustering.api",
    "PPN": "backtesting.policy_network.api",
    "Margin Trader": "backtesting.margin_trader.api",
}


class TrainReq(BaseModel):
    algo: Literal[tuple(ALGO_MAP.keys())]  # type: ignore[arg-type]
    tickers: List[str] = Field(..., min_items=1, max_items=8)
    hist_days: int
    lookback: int
    eval_win: int
    eta: float
    tc: float


_train_jobs: Dict[str, Dict[str, Any]] = {}


@app.post("/train")
def launch_backtest(req: TrainReq, bt: BackgroundTasks):
    jid = uuid4().hex
    _train_jobs[jid] = {"status": "queued", "algo": req.algo}
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
            "algo": req.algo,
            "nav": nav_json,
            "weights": weights,
            "metrics": metrics,
        }
    except Exception as exc:  # noqa: BLE001
        _train_jobs[jid] = {"status": "error", "algo": req.algo, "detail": str(exc)}


@app.get("/train/{jid}")
def train_status(jid: str):
    if jid not in _train_jobs:
        raise HTTPException(404, "Job not found")
    return _train_jobs[jid]


# Price forecasting endpoints

# Forecasting algorithm mappings
_SYNC_FORECASTERS: Dict[str, Callable[[ForecastRequest], Tuple[List[str], List[float], List[str], List[float]]]] = {
    "arima": arima.forecast,
    "lstm": lstm.forecast,
    "autoformer": autoformer.forecast,
}

_ASYNC_FORECASTERS: Dict[str, Callable[[ForecastRequest], Tuple[List[str], List[float], List[str], List[float]]]] = {
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
def forecast(algo: Literal["arima", "lstm", "autoformer"], req: ForecastRequest, bg: BackgroundTasks):
    """Route dispatcher - runs fast models synchronously, heavy models asynchronously"""

    if algo in _SYNC_FORECASTERS:
        try:
            hd, hv, fd, fv = _SYNC_FORECASTERS[algo](req)
            return _payload(hd, hv, fd, fv)
        except Exception as e:
            raise HTTPException(500, f"Forecasting error: {str(e)}")

    # Handle async models
    if algo in _ASYNC_FORECASTERS:
        task_id = uuid4().hex
        _forecast_jobs[task_id] = {"status": "running", "algo": algo}
        bg.add_task(_async_wrapper, algo, req, task_id)
        return {"task_id": task_id, "status": "running"}

    raise HTTPException(400, "Unknown forecasting algorithm")


def _async_wrapper(algo_key: str, req: ForecastRequest, tid: str) -> None:
    try:
        hd, hv, fd, fv = _ASYNC_FORECASTERS[algo_key](req)
        _forecast_jobs[tid] = {"status": "done", "algo": algo_key, **_payload(hd, hv, fd, fv)}
    except Exception as exc:  # noqa: BLE001
        _forecast_jobs[tid] = {"status": "error", "algo": algo_key, "detail": str(exc)}


@app.get("/forecast/result/{task_id}")
def forecast_result(task_id: str):
    job = _forecast_jobs.get(task_id)
    if job is None:
        raise HTTPException(404, "task_id not found")
    return job
