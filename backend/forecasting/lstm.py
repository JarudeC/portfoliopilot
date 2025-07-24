"""LSTM forecaster (quick CPU‑only implementation).

* Normalises the price series with `MinMaxScaler`.
* Trains a 2‑layer LSTM (hidden=64) for `epochs=20` by default.
* Recursive prediction for `horizon` future steps.

The signature and payload mirror `arima.forecast()` so the FastAPI
adapter can treat all synchronous forecasters identically.
"""

from __future__ import annotations

from datetime import timedelta
import pandas as pd
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from . import base


# ────────────────────────────────────────────
#  tiny PyTorch model
# ────────────────────────────────────────────
class _LSTM(nn.Module):
    def __init__(self, n_features: int = 1, hidden: int = 64, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ────────────────────────────────────────────
#  main entry‑point
# ────────────────────────────────────────────

def _prepare(series: np.ndarray, window: int):
    """Scale → windowed (X, y) tensors."""
    scaler = MinMaxScaler()
    data = scaler.fit_transform(series.reshape(-1, 1))

    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window : i])
        y.append(data[i])

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    return X, y, scaler


def forecast(
    req: base.ForecastRequest,
    *,
    window: int = 60,
    epochs: int = 20,
    lr: float = 1e-3,
) -> Tuple[List[str], List[float], List[str], List[float]]:
    """Run an LSTM forecast and return four parallel lists."""
    # 1. load historical prices
    series = base.load_series(req.ticker, req.start, req.end)
    hist_vals = series.values.astype("float32")

    # 2. validate data length and adjust window if needed
    if len(hist_vals) < window:
        window = max(10, len(hist_vals) // 2)  # Use smaller window for short series
        print(f"Warning: Adjusting window size to {window} due to insufficient data")
    
    if len(hist_vals) < window:
        raise ValueError(f"Insufficient data: {len(hist_vals)} points, need at least {window}")

    # 3. prepare data
    X, y, scaler = _prepare(hist_vals, window)

    # 3. train
    model = _LSTM()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(X).squeeze(), y.squeeze())
        loss.backward()
        opt.step()

    # 4. recursive forecast
    model.eval()
    preds = []
    seq = scaler.transform(hist_vals[-window:].reshape(-1, 1))
    seq = torch.tensor(seq.reshape(1, window, 1), dtype=torch.float32)

    with torch.no_grad():
        for _ in range(req.horizon):
            nxt = model(seq).item()
            preds.append(nxt)
            nxt_t = torch.tensor([[[nxt]]], dtype=torch.float32)
            seq = torch.cat([seq[:, 1:, :], nxt_t], dim=1)

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten().tolist()

    # 5. build date lists with proper business days
    last_date = series.index[-1].date()
    hist_dates = [d.strftime("%Y-%m-%d") for d in series.index.date]
    
    # Generate business day dates for forecast
    fc_dates = pd.bdate_range(
        start=last_date + timedelta(days=1), 
        periods=req.horizon
    ).strftime("%Y-%m-%d").tolist()

    return hist_dates, hist_vals.tolist(), fc_dates, preds
