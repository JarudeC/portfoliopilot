"""Centralized configuration for forecasting models.

This module contains hyperparameter configurations for all forecasting algorithms.
Using frozen dataclasses ensures configurations are immutable and type-safe.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ARIMAConfig:
    """ARIMA model hyperparameters and configuration.

    Attributes:
        max_p: Maximum autoregressive (AR) order to test during grid search
        max_q: Maximum moving average (MA) order to test during grid search
        max_d: Maximum differencing order to test
        adf_threshold: P-value threshold for ADF stationarity test (default: 0.05)
        min_observations: Minimum data points required for reliable ARIMA modeling
        fallback_orders: Sequence of (p,d,q) orders to try if auto-selection fails
    """
    max_p: int = 3
    max_q: int = 3
    max_d: int = 2
    adf_threshold: float = 0.05
    min_observations: int = 20
    fallback_orders: Tuple[Tuple[int, int, int], ...] = (
        (1, 1, 1),  # Classic ARIMA baseline
        (2, 1, 2),  # More complex AR and MA
        (1, 1, 0),  # AR with differencing only
        (0, 1, 1),  # MA with differencing only
        (2, 1, 0),  # Higher-order AR
        (0, 1, 2),  # Higher-order MA
    )


@dataclass(frozen=True)
class LSTMConfig:
    """LSTM neural network hyperparameters and configuration.

    Attributes:
        window: Look-back window size (number of historical points to use)
        hidden_size: Number of units in LSTM hidden state
        num_layers: Number of stacked LSTM layers
        epochs: Number of training epochs
        learning_rate: Adam optimizer learning rate
        min_observations: Minimum data points required (must exceed window size)
    """
    window: int = 60
    hidden_size: int = 64
    num_layers: int = 2
    epochs: int = 20
    learning_rate: float = 1e-3
    min_observations: int = 10


@dataclass(frozen=True)
class TransformerConfig:
    """Transformer model hyperparameters and configuration.

    Attributes:
        seq_len: Input sequence length (historical window)
        d_model: Dimension of transformer model embeddings
        nhead: Number of attention heads (must divide d_model evenly)
        num_layers: Number of transformer encoder layers
        epochs: Number of training epochs
        learning_rate: Adam optimizer learning rate
        min_observations: Minimum data points required (must exceed seq_len + horizon)
    """
    seq_len: int = 60
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    epochs: int = 20
    learning_rate: float = 1e-3
    min_observations: int = 30
