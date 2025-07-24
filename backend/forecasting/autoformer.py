"""Simple Transformer-based forecaster (fallback implementation).

This is a simplified transformer implementation that doesn't require external packages.
Uses PyTorch's built-in transformer layers to create a basic autoformer-like model.
"""

from __future__ import annotations

from datetime import timedelta
from typing import List, Tuple

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from . import base

__all__ = ["forecast"]

# ───────────────────────────────────────────────────────────────────
#  Simple Transformer Model
# ───────────────────────────────────────────────────────────────────
class SimpleTransformer(nn.Module):
    def __init__(self, seq_len: int = 60, pred_len: int = 14, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(seq_len, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, pred_len)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, 1)
        batch_size, seq_len, _ = x.shape
        
        # Project input to d_model dimensions
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Apply transformer
        x = self.transformer(x)  # (batch_size, seq_len, d_model)
        
        # Use the last timestep to predict future
        x = x[:, -1, :]  # (batch_size, d_model)
        
        # Project to prediction length
        x = self.output_projection(x)  # (batch_size, pred_len)
        
        return x

# ───────────────────────────────────────────────────────────────────
#  hyper‑params – tweakable via kwargs if you extend the signature
# ───────────────────────────────────────────────────────────────────
SEQ_LEN = 60   # look‑back window (reduced for simpler model)


# ───────────────────────────────────────────────────────────────────
#  Public API
# ───────────────────────────────────────────────────────────────────

def forecast(
    req: base.ForecastRequest,
    epochs: int = 20,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
) -> Tuple[List[str], List[float], List[str], List[float]]:
    """Train a simple Transformer on the past `SEQ_LEN` steps and predict forward.

    Returns four lists: history dates, history values, forecast dates, forecast values.
    Uses PyTorch's built-in transformer layers for a simplified implementation.
    """
    
    try:
        # 1) load price series
        series = base.load_series(req.ticker, req.start, req.end)
        if len(series) < SEQ_LEN + 10:
            raise ValueError(f"Insufficient data: need at least {SEQ_LEN + 10} observations, got {len(series)}")
        
        if len(series) < 30:
            raise ValueError(f"Insufficient data: only {len(series)} observations. Need at least 30 for Transformer.")

        # 2) normalise to zero‑mean/std for training stability
        mu, sigma = series.mean(), series.std()
        if sigma == 0:
            sigma = 1.0  # prevent division by zero
        norm = (series - mu) / sigma
        data = torch.tensor(norm.values, dtype=torch.float32).unsqueeze(-1)  # (T, 1)

        # 3) model instantiation
        model = SimpleTransformer(
            seq_len=SEQ_LEN,
            pred_len=req.horizon,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        )

        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()

        # 4) simple sliding‑window training loop
        model.train()
        for epoch in range(epochs):
            total_loss: float = 0.0
            steps = 0
            
            # Create training windows
            for i in range(SEQ_LEN, len(data) - req.horizon + 1, max(1, (len(data) - SEQ_LEN) // 20)):
                x = data[i - SEQ_LEN : i].unsqueeze(0)  # (1, seq_len, 1)
                y_true = data[i : i + req.horizon].squeeze(-1).unsqueeze(0)  # (1, pred_len)
                
                optim.zero_grad()
                y_pred = model(x)  # (1, pred_len)
                loss = criterion(y_pred, y_true)
                loss.backward()
                optim.step()
                
                total_loss += loss.item()
                steps += 1
                
            if steps == 0:
                break

        # 5) forecast on the latest window
        model.eval()
        with torch.no_grad():
            src = data[-SEQ_LEN:].unsqueeze(0)  # (1, seq_len, 1)
            pred_norm = model(src).squeeze().cpu().numpy()  # (pred_len,)
            
        # Ensure pred_norm is 1D array
        if pred_norm.ndim == 0:
            pred_norm = np.array([pred_norm])
        elif pred_norm.ndim > 1:
            pred_norm = pred_norm.flatten()
            
        # Take only the requested horizon length and ensure it's the right size
        if len(pred_norm) > req.horizon:
            pred_norm = pred_norm[:req.horizon]
        elif len(pred_norm) < req.horizon:
            # Pad with the last value if we got fewer predictions
            pad_length = req.horizon - len(pred_norm)
            last_val = pred_norm[-1] if len(pred_norm) > 0 else 0.0
            pred_norm = np.concatenate([pred_norm, np.full(pad_length, last_val)])
            
        pred = pred_norm * sigma + mu  # denormalise

        # 6) build payload lists
        hist_dates = series.index.strftime("%Y-%m-%d").tolist()
        hist_vals = series.tolist()

        last_date = series.index[-1].date()
        # Use business days like ARIMA to avoid weekends
        fc_dates = pd.bdate_range(last_date + timedelta(days=1), periods=req.horizon).strftime("%Y-%m-%d").tolist()
        fc_vals = pred.astype(np.float32).tolist()

        # Final validation - ensure all return values are proper lists
        if not isinstance(hist_dates, list) or not isinstance(hist_vals, list):
            raise ValueError("History data is not in list format")
        if not isinstance(fc_dates, list) or not isinstance(fc_vals, list):
            raise ValueError("Forecast data is not in list format")
        if len(hist_dates) != len(hist_vals):
            raise ValueError(f"History data length mismatch: {len(hist_dates)} dates vs {len(hist_vals)} values")
        if len(fc_dates) != len(fc_vals):
            raise ValueError(f"Forecast data length mismatch: {len(fc_dates)} dates vs {len(fc_vals)} values")
        if len(hist_dates) == 0:
            raise ValueError("No historical data returned")
        if len(fc_dates) == 0:
            raise ValueError("No forecast data returned")

        return hist_dates, hist_vals, fc_dates, fc_vals
        
    except Exception as e:
        # Convert any exception to a structured error that main.py can handle
        error_msg = f"Transformer forecasting failed for {req.ticker}: {str(e)}"
        raise RuntimeError(error_msg) from e
