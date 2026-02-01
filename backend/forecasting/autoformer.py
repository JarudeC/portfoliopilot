"""Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting.

This module implements the Autoformer architecture which uses series decomposition
and auto-correlation mechanisms for improved time series forecasting.

Reference: Wu et al. (2021) "Autoformer: Decomposition Transformers with
Auto-Correlation for Long-Term Series Forecasting"
"""

import logging
from datetime import timedelta
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseForecaster, ForecastResult
from .config import TransformerConfig
from core.data_loader import load_series
from .schemas import ForecastRequest

logger = logging.getLogger(__name__)


class SeriesDecomposition(nn.Module):
    """
    Series decomposition layer that separates trend and seasonal components.

    Uses moving average to extract the trend component, with the residual
    representing the seasonal/cyclic component.
    """

    def __init__(self, kernel_size: int = 25):
        """
        Initialize series decomposition layer.

        Args:
            kernel_size: Size of moving average window for trend extraction
        """
        super().__init__()
        self.kernel_size = kernel_size
        # Moving average for trend extraction
        self.avg_pool = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            count_include_pad=False
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompose time series into seasonal and trend components.

        Args:
            x: Input tensor of shape (batch, seq_len, features)

        Returns:
            Tuple of (seasonal, trend) components
        """
        # Apply moving average to extract trend
        # Need to transpose for AvgPool1d: (batch, features, seq_len)
        trend = self.avg_pool(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Seasonal component is the residual
        seasonal = x - trend

        return seasonal, trend


class AutoCorrelation(nn.Module):
    """
    Auto-Correlation mechanism for time series.

    Replaces traditional self-attention with auto-correlation to better
    capture period-based dependencies in time series data. Uses FFT for
    efficient computation.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        factor: int = 3
    ):
        """
        Initialize auto-correlation layer.

        Args:
            d_model: Dimension of the model
            n_heads: Number of attention heads
            factor: Factor for selecting top-k auto-correlations
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.factor = factor

        # Linear projections for queries, keys, values
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply auto-correlation mechanism.

        Args:
            queries: Query tensor (batch, seq_len, d_model)
            keys: Key tensor (batch, seq_len, d_model)
            values: Value tensor (batch, seq_len, d_model)

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = queries.shape

        # Project queries, keys, values
        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)

        # Reshape for multi-head: (batch, n_heads, seq_len, d_k)
        d_k = self.d_model // self.n_heads
        queries = queries.view(batch_size, seq_len, self.n_heads, d_k).transpose(1, 2)
        keys = keys.view(batch_size, seq_len, self.n_heads, d_k).transpose(1, 2)
        values = values.view(batch_size, seq_len, self.n_heads, d_k).transpose(1, 2)

        # Compute auto-correlation using FFT
        # This is more efficient than direct convolution for long sequences
        q_fft = torch.fft.rfft(queries, dim=2)
        k_fft = torch.fft.rfft(keys, dim=2)

        # Auto-correlation in frequency domain (element-wise product)
        autocorr = torch.fft.irfft(q_fft * torch.conj(k_fft), n=seq_len, dim=2)

        # Normalize
        autocorr = autocorr / seq_len

        # Find top-k delays with highest auto-correlation
        top_k = int(min(self.factor * np.log(seq_len + 1), seq_len))
        if top_k < 1:
            top_k = 1

        # Mean over feature dimension to get correlation strengths: (batch, n_heads, seq_len)
        autocorr_mean = autocorr.mean(dim=3)
        corr_weights, delays = torch.topk(autocorr_mean, top_k, dim=2)
        corr_weights = F.softmax(corr_weights, dim=-1)  # (batch, n_heads, top_k)

        # Time-delay aggregation: aggregate values at different time lags
        # Use vectorized roll for each of the top-k delays
        # delays: (batch, n_heads, top_k) - contains the time shifts

        out = torch.zeros_like(values)

        # Process each k in the top-k delays
        for k in range(top_k):
            # Get weights for this k: (batch, n_heads, 1, 1)
            w_k = corr_weights[:, :, k].unsqueeze(-1).unsqueeze(-1)

            # For time-delay aggregation, we use the mean delay across batch/heads
            # This approximates the per-element delay with a global one for efficiency
            mean_d = int(delays[:, :, k].float().mean().item())

            # Roll (circular shift) all values by this delay
            rolled_v = torch.roll(values, shifts=-mean_d, dims=2)

            # Accumulate weighted contribution
            out = out + w_k * rolled_v

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_projection(out)


class AutoformerLayer(nn.Module):
    """
    Single Autoformer encoder layer with auto-correlation and decomposition.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        decomp_kernel: int = 25
    ):
        """
        Initialize Autoformer layer.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Dimension of feedforward network
            dropout: Dropout rate
            decomp_kernel: Kernel size for series decomposition
        """
        super().__init__()

        # Auto-correlation mechanism
        self.autocorrelation = AutoCorrelation(d_model, n_heads)

        # Series decomposition
        self.decomp1 = SeriesDecomposition(decomp_kernel)
        self.decomp2 = SeriesDecomposition(decomp_kernel)

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Autoformer layer.

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Tuple of (seasonal, trend) components
        """
        # Auto-correlation with residual connection
        residual = x
        x = self.norm1(x)
        x = self.autocorrelation(x, x, x)
        x = residual + self.dropout(x)

        # Decompose after auto-correlation
        x, trend1 = self.decomp1(x)

        # Feedforward with residual connection
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)

        # Decompose after FFN
        x, trend2 = self.decomp2(x)

        # Accumulate trend components
        trend = trend1 + trend2

        return x, trend


class _AutoformerModel(nn.Module):
    """
    Autoformer model for time series forecasting.

    Implements progressive series decomposition and auto-correlation
    for improved long-term forecasting.
    """

    def __init__(
        self,
        seq_len: int = 60,
        pred_len: int = 14,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        decomp_kernel: int = 25
    ):
        """
        Initialize Autoformer model.

        Args:
            seq_len: Input sequence length
            pred_len: Output prediction length
            d_model: Model dimension
            n_heads: Number of attention heads
            num_layers: Number of encoder layers
            d_ff: Feedforward network dimension
            dropout: Dropout rate
            decomp_kernel: Kernel size for decomposition
        """
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Input embedding
        self.value_embedding = nn.Linear(1, d_model)

        # Positional encoding (learned)
        self.position_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))

        # Initial decomposition
        self.decomp = SeriesDecomposition(decomp_kernel)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            AutoformerLayer(d_model, n_heads, d_ff, dropout, decomp_kernel)
            for _ in range(num_layers)
        ])

        # Output projections for seasonal and trend
        self.seasonal_projection = nn.Linear(d_model, pred_len)
        self.trend_projection = nn.Linear(seq_len, pred_len)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Autoformer.

        Args:
            x: Input tensor (batch, seq_len, 1)

        Returns:
            Forecast tensor (batch, pred_len)
        """
        # Embed input
        x = self.value_embedding(x)
        x = x + self.position_embedding[:, :x.size(1), :]
        x = self.dropout(x)

        # Initial decomposition
        seasonal_init, trend = self.decomp(x)

        # Progressive decomposition through encoder layers
        seasonal = seasonal_init
        for layer in self.encoder_layers:
            seasonal, trend_layer = layer(seasonal)
            trend = trend + trend_layer  # Accumulate trend

        # Project seasonal component
        seasonal_out = self.seasonal_projection(seasonal[:, -1, :])

        # Project trend component
        trend_out = self.trend_projection(trend.permute(0, 2, 1))[:, -1, :]

        # Combine seasonal and trend for final forecast
        return seasonal_out + trend_out


class AutoformerForecaster(BaseForecaster):
    """
    Autoformer-based time series forecaster.

    Uses series decomposition and auto-correlation mechanisms for
    improved long-term forecasting of financial time series.

    Features:
        - Progressive series decomposition (separates trend from seasonality)
        - Auto-correlation mechanism (captures period-based dependencies)
        - Multi-head attention with FFT-based efficiency
        - Z-score normalization for training stability
        - Direct multi-step forecasting
    """

    def __init__(self, config: TransformerConfig = None):
        """
        Initialize Autoformer forecaster with configuration.

        Args:
            config: TransformerConfig instance (uses defaults if None)
        """
        self.config = config or TransformerConfig()

    def forecast(self, req: ForecastRequest) -> ForecastResult:
        """
        Generate Autoformer forecast for the given request.

        Args:
            req: ForecastRequest with ticker, dates, and horizon

        Returns:
            Tuple of (hist_dates, hist_values, forecast_dates, forecast_values)

        Raises:
            ValueError: If data is invalid or insufficient
            RuntimeError: If forecasting fails
        """
        try:
            logger.info(f"Starting Autoformer forecast for {req.ticker}")

            # Use pre-fetched series if available, otherwise load
            if req._series is not None:
                series = req._series
            else:
                series = load_series(req.ticker, req.start, req.end)

            # Validate sufficient data
            min_required = self.config.seq_len + 10
            self.validate_data_length(len(series), min_required, req.ticker)

            # Z-score normalization for training stability
            mu, sigma = series.mean(), series.std()
            if sigma == 0:
                sigma = 1.0  # Prevent division by zero
            norm = (series - mu) / sigma
            data = torch.tensor(norm.values, dtype=torch.float32).unsqueeze(-1)

            logger.debug(
                f"Normalized data: mean={mu:.2f}, std={sigma:.2f}, "
                f"shape={data.shape}"
            )

            # Build Autoformer model
            model = _AutoformerModel(
                seq_len=self.config.seq_len,
                pred_len=req.horizon,
                d_model=self.config.d_model,
                n_heads=self.config.nhead,
                num_layers=self.config.num_layers,
                d_ff=self.config.d_model * 4,
                dropout=0.1,
                decomp_kernel=25  # Default decomposition kernel
            )

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate
            )
            criterion = nn.MSELoss()

            # Train with sliding windows
            model.train()
            for epoch in range(self.config.epochs):
                total_loss = 0.0
                steps = 0

                # Sliding window stride - sample ~20 windows across the data
                stride = max(1, (len(data) - self.config.seq_len) // 20)

                for i in range(
                    self.config.seq_len,
                    len(data) - req.horizon + 1,
                    stride
                ):
                    x = data[i - self.config.seq_len : i].unsqueeze(0)
                    y_true = data[i : i + req.horizon].squeeze(-1).unsqueeze(0)

                    optimizer.zero_grad()
                    y_pred = model(x)
                    loss = criterion(y_pred, y_true)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    steps += 1

                if steps == 0:
                    logger.warning("No training steps - data may be too short")
                    break

                if (epoch + 1) % 5 == 0:
                    avg_loss = total_loss / steps
                    logger.debug(
                        f"Epoch {epoch + 1}/{self.config.epochs}, "
                        f"Loss: {avg_loss:.6f}"
                    )

            logger.info("Autoformer training complete")

            # Generate forecast
            model.eval()
            with torch.no_grad():
                src = data[-self.config.seq_len:].unsqueeze(0)
                pred_norm = model(src).squeeze().cpu().numpy()

            # Ensure correct shape
            if pred_norm.ndim == 0:
                pred_norm = np.array([pred_norm])
            elif pred_norm.ndim > 1:
                pred_norm = pred_norm.flatten()

            # Ensure correct length
            if len(pred_norm) > req.horizon:
                pred_norm = pred_norm[:req.horizon]
            elif len(pred_norm) < req.horizon:
                # Pad with last value if needed
                pad_length = req.horizon - len(pred_norm)
                last_val = pred_norm[-1] if len(pred_norm) > 0 else 0.0
                pred_norm = np.concatenate([pred_norm, np.full(pad_length, last_val)])
                logger.warning(
                    f"Padded forecast from {len(pred_norm)} to {req.horizon} steps"
                )

            # Denormalize
            pred = pred_norm * sigma + mu

            # Build response
            hist_dates = series.index.strftime("%Y-%m-%d").tolist()
            hist_vals = series.tolist()

            last_date = series.index[-1].date()
            fc_dates = pd.bdate_range(
                last_date + timedelta(days=1),
                periods=req.horizon
            ).strftime("%Y-%m-%d").tolist()
            fc_vals = pred.astype(np.float32).tolist()

            # Final validation
            if len(fc_dates) != len(fc_vals):
                raise ValueError(
                    f"Forecast length mismatch: {len(fc_dates)} dates vs "
                    f"{len(fc_vals)} values"
                )

            logger.info(f"Autoformer forecast complete for {req.ticker}")
            return hist_dates, hist_vals, fc_dates, fc_vals

        except Exception as e:
            error_msg = f"Autoformer forecasting failed for {req.ticker}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e


# Public API function for backward compatibility with main.py
def forecast(
    req: ForecastRequest,
    epochs: int = None,
    d_model: int = None,
    nhead: int = None,
    num_layers: int = None
) -> ForecastResult:
    """
    Public API function for Autoformer forecasting.

    Maintains backward compatibility with existing kwargs-based API while
    allowing configuration overrides.

    Args:
        req: ForecastRequest with ticker, dates, and horizon
        epochs: Optional epochs override
        d_model: Optional model dimension override
        nhead: Optional attention heads override
        num_layers: Optional number of layers override

    Returns:
        Tuple of (hist_dates, hist_values, forecast_dates, forecast_values)
    """
    # Build config with overrides
    config = TransformerConfig()
    if any(x is not None for x in [epochs, d_model, nhead, num_layers]):
        config = TransformerConfig(
            seq_len=config.seq_len,
            d_model=config.d_model if d_model is None else d_model,
            nhead=config.nhead if nhead is None else nhead,
            num_layers=config.num_layers if num_layers is None else num_layers,
            epochs=config.epochs if epochs is None else epochs,
            learning_rate=config.learning_rate,
            min_observations=config.min_observations
        )

    forecaster = AutoformerForecaster(config)
    return forecaster.forecast(req)
