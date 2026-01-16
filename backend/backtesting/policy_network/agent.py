"""Neural network agent for portfolio policy learning."""

import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.optim as optim

from .network import PolicyNetwork

logger = logging.getLogger(__name__)


class PolicyAgent:
    """
    Portfolio Policy Network agent with online learning.

    Wraps the CNN-TCN-LSTM network and provides training/inference methods.
    """

    def __init__(self, config: Dict[str, Any], device: str = "cpu"):
        """
        Initialize agent.

        Args:
            config: Configuration dict with keys: input, training, trading
            device: 'cpu' or 'cuda'
        """
        self.config = config
        self.device = torch.device(device)

        # Network parameters
        n_features = config["input"]["feature_number"]
        n_assets = config["input"]["coin_number"]
        window_size = config["input"]["window_size"]
        dropout = config["training"].get("dropout", 0.2)

        self.network = PolicyNetwork(
            n_features=n_features,
            n_assets=n_assets,
            window_size=window_size,
            dropout=dropout
        ).to(self.device)

        # Training parameters
        self.gamma = config["training"]["gamma"]  # Turnover penalty
        self.alpha = config["training"]["alpha"]  # Variance penalty
        self.commission = config["trading"]["trading_consumption"]

        # Optimizer
        lr = config["training"]["learning_rate"]
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        self.decay_steps = config["training"]["decay_steps"]
        self.decay_rate = config["training"]["decay_rate"]
        self.global_step = 0

    def predict(self, history: np.ndarray, prev_weights: np.ndarray) -> np.ndarray:
        """
        Predict portfolio weights given price history.

        Args:
            history: Price history (window, n_assets, features)
            prev_weights: Previous weights including cash (n_assets+1,)

        Returns:
            New weights including cash (n_assets+1,)
        """
        self.network.eval()

        with torch.no_grad():
            # Reshape: (window, n_assets, feat) -> (1, feat, n_assets, window)
            hist_t = (
                torch.from_numpy(history.astype(np.float32))
                .permute(2, 1, 0)
                .unsqueeze(0)
                .to(self.device)
            )

            # Previous weights without cash position
            prev_t = (
                torch.from_numpy(prev_weights[1:].astype(np.float32))
                .unsqueeze(0)
                .to(self.device)
            )

            weights = self.network(hist_t, prev_t)
            return weights.squeeze(0).cpu().numpy()

    def train_step(
        self,
        x: np.ndarray,
        y_next: np.ndarray,
        prev_w: np.ndarray
    ) -> float:
        """
        Single training step.

        Args:
            x: Price tensor (batch, features, n_assets, window)
            y_next: Next period price ratios (batch, n_assets)
            prev_w: Previous weights (batch, n_assets)

        Returns:
            Loss value
        """
        self.network.train()

        batch_size = x.shape[0]
        n_assets_x = x.shape[2]
        n_assets_w = prev_w.shape[1]

        # Handle cash position mismatch
        if n_assets_w == n_assets_x + 1:
            prev_w = prev_w[:, 1:]  # Remove cash

        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y_next, dtype=torch.float32, device=self.device).clamp_min(1e-8)
        w_prev = torch.tensor(prev_w, dtype=torch.float32, device=self.device)

        # Forward pass
        w = self.network(x_t, w_prev)

        # Portfolio value calculation
        w_assets = w[:, 1:]  # Exclude cash
        price_vec = torch.cat([torch.ones(batch_size, 1, device=self.device), y_t], dim=1)

        # Turnover
        turnover = torch.abs(w_assets - w_prev).sum(dim=1)

        # Portfolio growth
        pv = (w * price_vec).sum(dim=1)
        pv = pv * (1 - self.commission * turnover).clamp_min(1e-8)
        pv = pv.clamp_min(1e-8)

        # Loss components
        neg_log_growth = -torch.log(pv).mean()
        var_penalty = torch.var(torch.log(pv), unbiased=False)
        cost_penalty = turnover.mean()

        loss = neg_log_growth + self.gamma * cost_penalty + self.alpha * var_penalty

        # Check for NaN
        if torch.isnan(loss):
            logger.error("NaN detected in training loss")
            raise RuntimeError("NaN in training")

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Learning rate decay
        self.global_step += 1
        if self.global_step % self.decay_steps == 0:
            for g in self.optimizer.param_groups:
                g["lr"] *= self.decay_rate

        return float(loss.detach().cpu())

    def save(self, path: str | Path):
        """Save network weights."""
        torch.save(self.network.state_dict(), path)

    def load(self, path: str | Path):
        """Load network weights."""
        self.network.load_state_dict(torch.load(path, map_location=self.device))
