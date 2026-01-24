# FastAPI backend for portfolio backtesting and price forecasting
# Provides REST endpoints for running portfolio algorithms and time series models

from __future__ import annotations

import logging
from importlib import import_module
from typing import Any, Dict, List, Literal, Callable, Tuple
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.data_loader import load_prices, load_series

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

# NOTE: Single-ticker endpoint replaced by /prices/batch - kept for potential future use
# class PriceRequest(BaseModel):
#     """Request schema for historical price data."""
#     ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")
#     start: str = Field(..., description="Start date (YYYY-MM-DD)")
#     end: str = Field(..., description="End date (YYYY-MM-DD)")
#
#
# @app.post("/prices")
# def get_prices(req: PriceRequest):
#     """
#     Fetch historical stock prices from Yahoo Finance.
#
#     This is a dedicated endpoint for fetching raw price data without
#     running any forecasting models. Use this when you only need historical
#     data (e.g., for Custom AI Strategy, data visualization, etc.)
#
#     Returns:
#         ticker: The requested ticker symbol
#         dates: List of date strings (YYYY-MM-DD)
#         prices: List of adjusted close prices
#     """
#     from datetime import datetime
#
#     try:
#         start_date = datetime.strptime(req.start, "%Y-%m-%d").date()
#         end_date = datetime.strptime(req.end, "%Y-%m-%d").date()
#     except ValueError as e:
#         raise HTTPException(400, f"Invalid date format. Use YYYY-MM-DD. Error: {str(e)}")
#
#     try:
#         series = load_series(req.ticker, start_date, end_date)
#
#         return {
#             "ticker": req.ticker,
#             "dates": series.index.strftime("%Y-%m-%d").tolist(),
#             "prices": series.tolist()
#         }
#     except ValueError as e:
#         raise HTTPException(400, str(e))
#     except Exception as e:
#         raise HTTPException(500, f"Failed to fetch prices: {str(e)}")


class BatchPriceRequest(BaseModel):
    """Request schema for batch historical price data."""
    tickers: List[str] = Field(..., min_length=1, max_length=20)
    start: str = Field(..., description="Start date (YYYY-MM-DD)")
    end: str = Field(..., description="End date (YYYY-MM-DD)")


@app.post("/prices/batch")
def get_prices_batch(req: BatchPriceRequest):
    """
    Fetch historical stock prices for multiple tickers in parallel.
    Uses ThreadPoolExecutor for concurrent yfinance calls.

    Returns:
        Dict mapping ticker -> {dates, prices} or {error} if failed
    """
    from datetime import datetime

    try:
        start_date = datetime.strptime(req.start, "%Y-%m-%d").date()
        end_date = datetime.strptime(req.end, "%Y-%m-%d").date()
    except ValueError as e:
        raise HTTPException(400, f"Invalid date format. Use YYYY-MM-DD. Error: {str(e)}")

    def fetch_single(ticker: str):
        try:
            series = load_series(ticker, start_date, end_date)
            return ticker, {
                "dates": series.index.strftime("%Y-%m-%d").tolist(),
                "prices": series.tolist()
            }
        except Exception as e:
            return ticker, {"error": str(e)}

    results = {}
    max_workers = min(len(req.tickers), 10)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_single, t): t for t in req.tickers}
        for future in as_completed(futures):
            ticker, data = future.result()
            results[ticker] = data

    return results


# Portfolio algorithm mapping
ALGO_MAP: Dict[str, str] = {
    "Naive Markowitz": "backtesting.naive_markowitz.api",
    "GVMP": "backtesting.gmvp_clustering.api",
    "PPN": "backtesting.policy_network.api",
    "Margin Trader": "backtesting.margin_trader.api",
}


class BacktestRequest(BaseModel):
    """Request schema for portfolio backtesting."""
    algo: Literal[tuple(ALGO_MAP.keys())]  # type: ignore[arg-type]
    tickers: List[str] = Field(..., min_items=1, max_items=8)
    hist_days: int
    lookback: int
    eval_win: int
    eta: float
    tc: float


_backtest_jobs: Dict[str, Dict[str, Any]] = {}


@app.post("/backtest")
def launch_backtest(req: BacktestRequest, bt: BackgroundTasks):
    """
    Launch a portfolio backtesting job asynchronously.

    Accepts algorithm type and parameters, queues the backtest as a background
    task, and returns a job ID for polling status via GET /backtest/{jid}.
    """
    jid = uuid4().hex
    _backtest_jobs[jid] = {"status": "queued", "algo": req.algo}
    bt.add_task(_backtest_worker, jid, req)
    return {"job_id": jid}


def _backtest_worker(jid: str, req: BacktestRequest):
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
        _backtest_jobs[jid] = {
            "status": "done",
            "algo": req.algo,
            "nav": nav_json,
            "weights": weights,
            "metrics": metrics,
        }
    except Exception as exc:  # noqa: BLE001
        _backtest_jobs[jid] = {"status": "error", "algo": req.algo, "detail": str(exc)}


@app.get("/backtest/{jid}")
def backtest_status(jid: str):
    """
    Poll the status of a backtesting job.

    Returns job status ('queued', 'done', 'error') along with results
    (nav, weights, metrics) when complete, or error details if failed.
    """
    if jid not in _backtest_jobs:
        raise HTTPException(404, "Job not found")
    return _backtest_jobs[jid]


# Price forecasting endpoints

_FORECASTERS: Dict[str, Callable[[ForecastRequest], Tuple[List[str], List[float], List[str], List[float]]]] = {
    "arima": arima.forecast,
    "lstm": lstm.forecast,
    "autoformer": autoformer.forecast,
}


@app.post("/forecast/{algo}")
def forecast(algo: Literal["arima", "lstm", "autoformer"], req: ForecastRequest):
    """
    Generate price forecast using the specified algorithm.

    Runs the forecasting model synchronously and returns historical data
    along with predicted values for the requested horizon.
    """
    if algo not in _FORECASTERS:
        raise HTTPException(400, "Unknown forecasting algorithm")

    try:
        hd, hv, fd, fv = _FORECASTERS[algo](req)
        return {
            "history_dates": hd,
            "history_values": hv,
            "forecast_dates": fd,
            "forecast_values": fv,
        }
    except Exception as e:
        raise HTTPException(500, f"Forecasting error: {str(e)}")
