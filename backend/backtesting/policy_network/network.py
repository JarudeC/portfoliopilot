"""CNN-TCN-LSTM network architecture for portfolio weight prediction."""

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TemporalBlock(nn.Module):
    """Dilated TCN block with residual connection."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: Tuple[int, int],
        dilation: Tuple[int, int],
        dropout: float,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(n_inputs, n_outputs, kernel_size,
                               dilation=dilation, padding="same")
        self.conv2 = nn.Conv2d(n_outputs, n_outputs, kernel_size,
                               dilation=dilation, padding="same")
        self.dropout = nn.Dropout(dropout)
        self.proj = (nn.Conv2d(n_inputs, n_outputs, 1)
                     if n_inputs != n_outputs else nn.Identity())
        self._asset_conv = None

        # Initialize weights
        for m in (self.conv1, self.conv2, self.proj):
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=1e-2)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.conv1(x))
        y = self.dropout(y)
        y = F.relu(self.conv2(y))
        y = self.dropout(y)

        # Dynamic asset convolution
        k = y.size(2)
        if self._asset_conv is None or self._asset_conv.kernel_size[0] != k:
            self._asset_conv = nn.Conv2d(
                y.size(1), y.size(1), (k, 1), padding="same"
            ).to(y.device)
            nn.init.normal_(self._asset_conv.weight, std=1e-2)
            nn.init.zeros_(self._asset_conv.bias)

        y = F.relu(self._asset_conv(y))
        return F.relu(self.proj(x) + y)


class PolicyNetwork(nn.Module):
    """
    CNN-TCN-LSTM network for portfolio weight prediction.

    Input: (batch, features=1, n_assets, window_size)
    Output: (batch, n_assets+1) - weights including cash position
    """

    def __init__(self, n_features: int, n_assets: int, window_size: int, dropout: float = 0.2):
        super().__init__()
        self.n_assets = n_assets
        self.window_size = window_size

        # TCN blocks with increasing dilation
        self.tcn0 = TemporalBlock(n_features, 8, (1, 3), (1, 1), dropout)
        self.tcn1 = TemporalBlock(8, 16, (1, 3), (1, 2), dropout)
        self.tcn2 = TemporalBlock(16, 16, (1, 3), (1, 4), dropout)

        # Squeeze convolution (collapse time dimension)
        self.squeeze_conv = nn.Conv2d(16, 16, (1, window_size))
        nn.init.normal_(self.squeeze_conv.weight, std=1e-2)
        nn.init.zeros_(self.squeeze_conv.bias)

        # LSTM branch
        self.lstm = nn.LSTM(input_size=1, hidden_size=16, batch_first=True, num_layers=1)

        # Decision heads (conv features + lstm features + prev_weights)
        in_channels = 16 + 16 + 1
        self.decision_i = nn.Conv2d(in_channels, 1, 1)
        self.decision_s = nn.Conv2d(in_channels, 1, 1)

        # Initialize decision layers with variance to break symmetry
        nn.init.normal_(self.decision_i.weight, mean=0.0, std=0.5)
        nn.init.uniform_(self.decision_i.bias, -1.0, 1.0)
        nn.init.normal_(self.decision_s.weight, mean=0.0, std=0.3)
        nn.init.uniform_(self.decision_s.bias, -0.5, 1.5)

        # Learnable cash bias
        self.cash_bias = nn.Parameter(torch.randn(1, 1, 1, 1) * 0.01)

    def forward(self, x: torch.Tensor, prev_w: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Price tensor (batch, features, n_assets, window)
            prev_w: Previous weights (batch, n_assets)

        Returns:
            Portfolio weights (batch, n_assets+1) including cash
        """
        batch_size = x.size(0)
        n_assets = x.size(2)

        if prev_w.size(1) != n_assets:
            raise RuntimeError(f"prev_w shape {prev_w.shape} doesn't match n_assets {n_assets}")

        # Normalize by last price
        denom = x[:, :, :, -1:].clamp_min(1e-8).expand_as(x)
        x = x / denom

        # TCN path
        y = self.tcn2(self.tcn1(self.tcn0(x)))
        y = F.relu(self.squeeze_conv(y))  # (B, 16, n_assets, 1)
        y = y.permute(0, 2, 3, 1)  # (B, n_assets, 1, 16)

        # LSTM path
        lstm_input = x.permute(0, 2, 3, 1).reshape(batch_size * n_assets, self.window_size, 1)
        lstm_out, _ = self.lstm(lstm_input)
        lstm_features = lstm_out[:, -1, :].view(batch_size, n_assets, 1, 16)

        # Concatenate features
        pw = prev_w.view(batch_size, n_assets, 1, 1)
        cat = torch.cat([y, lstm_features, pw], dim=3)

        # Add cash position
        cash_features = self.cash_bias.expand(batch_size, 1, 1, cat.size(3))
        cat = torch.cat([cash_features, cat], dim=1)

        # Decision layers (channel-first for conv2d)
        cat = cat.permute(0, 3, 1, 2)
        wi = self.decision_i(cat).view(batch_size, -1)
        ws = self.decision_s(cat).view(batch_size, -1)

        # Combine and normalize
        combined = wi + ws

        # Add exploration noise during training
        if self.training:
            noise = torch.randn_like(combined) * 0.1
            combined = combined + noise

        return torch.softmax(combined, dim=1)
