"""Centralized hyperparameter configuration for all backtesting strategies."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class NaiveMarkowitzConfig:
    """Configuration for Naive Markowitz mean-variance optimization."""
    target_return: float = 0.0008  # 0.08% daily ≈ 20% annual
    ridge_lambda: float = 1e-4
    markowitz_type: Literal["expected_returns", "min_variance"] = "expected_returns"
    return_cap: float = 0.10
    random_seed: int = 43


@dataclass(frozen=True)
class GMVPClusteringConfig:
    """Configuration for Clustering-Aided GMVP with Ledoit-Wolf shrinkage."""
    n_clusters: int | None = None
    min_clusters: int = 2
    max_clusters: int = 10
    shrinkage: bool = True
    cluster_method: Literal["kmeans", "bounded_kmeans"] = "bounded_kmeans"
    random_seed: int = 42


@dataclass(frozen=True)
class PolicyNetworkConfig:
    """Configuration for CNN-TCN-LSTM policy network."""
    window_size: int = 50
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    dropout: float = 0.2
    hidden_size: int = 64
    num_layers: int = 2
    kernel_size: int = 3
    early_stopping_patience: int = 10
    validation_split: float = 0.2


@dataclass(frozen=True)
class MarginTraderConfig:
    """Configuration for A2C reinforcement learning margin trader."""
    initial_cash: float = 100000.0
    max_leverage: float = 2.0
    margin_rate: float = 0.0002  # ~5% annual
    maintenance_margin: float = 0.25
    gamma: float = 0.99
    actor_lr: float = 0.0003
    critic_lr: float = 0.001
    hidden_sizes: tuple[int, ...] = (256, 128)
    n_episodes: int = 100
    max_steps: int = 252
    entropy_coef: float = 0.01


@dataclass(frozen=True)
class BacktestConfig:
    """Common configuration for all backtests."""
    trading_days_per_year: int = 252
    risk_free_rate: float = 0.0
    transaction_cost: float = 0.001
