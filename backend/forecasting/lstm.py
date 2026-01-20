"""LSTM forecasting with recursive prediction.

This module provides LSTM-based time series forecasting using a simple
2-layer LSTM architecture with MinMaxScaler normalization.
"""

import logging
from datetime import timedelta
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from .base import BaseForecaster, ForecastResult
from .config import LSTMConfig
from core.data_loader import load_series
from .schemas import ForecastRequest

logger = logging.getLogger(__name__)


class _LSTMModel(nn.Module):
    """PyTorch LSTM model for time series forecasting.

    Simple 2-layer LSTM architecture with a linear output layer.
    """

    def __init__(self, n_features: int = 1, hidden: int = 64, layers: int = 2):
        """
        Initialize LSTM model.

        Args:
            n_features: Number of input features (1 for univariate time series)
            hidden: Number of units in LSTM hidden state
            layers: Number of stacked LSTM layers
        """
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM and linear layer.

        Args:
            x: Input tensor of shape (batch, sequence, features)

        Returns:
            Output tensor of shape (batch, 1)
        """
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class LSTMForecaster(BaseForecaster):
    """
    LSTM-based time series forecaster.

    Uses a 2-layer LSTM neural network with MinMaxScaler normalization
    for recursive multi-step forecasting.

    Features:
        - MinMaxScaler normalization (scales to [0, 1] range)
        - Sliding window approach for training data preparation
        - Recursive forecasting (predicts one step, feeds back as input)
        - Configurable hyperparameters (window, hidden size, epochs, etc.)
    """

    def __init__(self, config: LSTMConfig = None):
        """
        Initialize LSTM forecaster with configuration.

        Args:
            config: LSTMConfig instance (uses defaults if None)
        """
        self.config = config or LSTMConfig()

    def _prepare_data(
        self,
        series: np.ndarray,
        window: int
    ) -> Tuple[torch.Tensor, torch.Tensor, MinMaxScaler]:
        """
        Scale data and create windowed sequences for training.

        Args:
            series: Time series values as numpy array
            window: Look-back window size

        Returns:
            Tuple of (X, y, scaler) where:
                - X: Input sequences of shape (n_samples, window, 1)
                - y: Target values of shape (n_samples, 1)
                - scaler: Fitted MinMaxScaler for inverse transform
        """
        scaler = MinMaxScaler()
        data = scaler.fit_transform(series.reshape(-1, 1))

        X, y = [], []
        for i in range(window, len(data)):
            X.append(data[i - window : i])
            y.append(data[i])

        X = torch.tensor(np.array(X), dtype=torch.float32)
        y = torch.tensor(np.array(y), dtype=torch.float32)

        logger.debug(f"Prepared {len(X)} training samples with window={window}")
        return X, y, scaler

    def forecast(self, req: ForecastRequest) -> ForecastResult:
        """
        Generate LSTM forecast for the given request.

        Args:
            req: ForecastRequest with ticker, dates, and horizon

        Returns:
            Tuple of (hist_dates, hist_values, forecast_dates, forecast_values)

        Raises:
            ValueError: If data is invalid or insufficient
        """
        logger.info(f"Starting LSTM forecast for {req.ticker}")

        # Load data
        series = load_series(req.ticker, req.start, req.end)
        hist_vals = series.values.astype('float32')

        # Adjust window if needed for limited data
        window = self.config.window
        if len(hist_vals) < window:
            window = max(self.config.min_observations, len(hist_vals) // 2)
            logger.warning(
                f"Adjusting window to {window} due to limited data "
                f"({len(hist_vals)} observations)"
            )

        # Validate sufficient data
        self.validate_data_length(len(hist_vals), window, req.ticker)

        # Prepare training data
        X, y, scaler = self._prepare_data(hist_vals, window)

        # Build and train model
        model = _LSTMModel(
            hidden=self.config.hidden_size,
            layers=self.config.num_layers
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate
        )
        loss_fn = nn.MSELoss()

        model.train()
        for epoch in range(self.config.epochs):
            optimizer.zero_grad(set_to_none=True)
            predictions = model(X).squeeze()
            loss = loss_fn(predictions, y.squeeze())
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                logger.debug(
                    f"Epoch {epoch + 1}/{self.config.epochs}, "
                    f"Loss: {loss.item():.6f}"
                )

        logger.info("LSTM training complete")

        # Generate recursive forecasts
        model.eval()
        fc_values = []
        seq = scaler.transform(hist_vals[-window:].reshape(-1, 1))
        seq = torch.tensor(seq.reshape(1, window, 1), dtype=torch.float32)

        with torch.no_grad():
            for step in range(req.horizon):
                next_val = model(seq).item()
                fc_values.append(next_val)

                # Update sequence for next prediction
                next_tensor = torch.tensor([[[next_val]]], dtype=torch.float32)
                seq = torch.cat([seq[:, 1:, :], next_tensor], dim=1)

        # Inverse transform to original scale
        fc_values = scaler.inverse_transform(
            np.array(fc_values).reshape(-1, 1)
        ).flatten().tolist()

        # Build response
        hist_dates = [d.strftime("%Y-%m-%d") for d in series.index.date]
        last_date = series.index[-1].date()
        fc_dates = pd.bdate_range(
            start=last_date + timedelta(days=1),
            periods=req.horizon
        ).strftime("%Y-%m-%d").tolist()

        logger.info(f"LSTM forecast complete for {req.ticker}")
        return hist_dates, hist_vals.tolist(), fc_dates, fc_values


# Public API function for backward compatibility with main.py
def forecast(
    req: ForecastRequest,
    *,
    window: int = None,
    epochs: int = None,
    lr: float = None
) -> ForecastResult:
    """
    Public API function for LSTM forecasting.

    Maintains backward compatibility with existing kwargs-based API while
    allowing configuration overrides.

    Args:
        req: ForecastRequest with ticker, dates, and horizon
        window: Optional window size override
        epochs: Optional epochs override
        lr: Optional learning rate override

    Returns:
        Tuple of (hist_dates, hist_values, forecast_dates, forecast_values)
    """
    # Build config with overrides
    config = LSTMConfig()
    if window is not None:
        config = LSTMConfig(
            window=window,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            epochs=config.epochs if epochs is None else epochs,
            learning_rate=config.learning_rate if lr is None else lr,
            min_observations=config.min_observations
        )
    elif epochs is not None or lr is not None:
        config = LSTMConfig(
            window=config.window,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            epochs=config.epochs if epochs is None else epochs,
            learning_rate=config.learning_rate if lr is None else lr,
            min_observations=config.min_observations
        )

    forecaster = LSTMForecaster(config)
    return forecaster.forecast(req)
