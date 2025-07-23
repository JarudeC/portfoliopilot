# pgportfolio_pytorch/learn/network.py
# ------------------------------------
# PyTorch re-implementation of the PPN CNN-TCN-LSTM backbone
# (Shreenivas & Velu, 2020)

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    """2-layer dilated TCN block with symmetric padding & residual add."""
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
        self.asset_conv = None
        for m in (self.conv1, self.conv2, self.proj):
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=1e-2)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.conv1(x)); y = self.dropout(y)
        y = F.relu(self.conv2(y)); y = self.dropout(y)

        k = y.size(2)
        if (self.asset_conv is None) or (self.asset_conv.kernel_size[0] != k):
            self.asset_conv = nn.Conv2d(
                y.size(1), y.size(1), (k, 1), padding="same"
            ).to(y.device)
            nn.init.normal_(self.asset_conv.weight, std=1e-2)
            nn.init.zeros_(self.asset_conv.bias)

        y = F.relu(self.asset_conv(y))
        return F.relu(self.proj(x) + y)


class CNN(nn.Module):
    """
    Back-test network.
    Input  : (B, F=1, rows, cols)
    Output : (B, rows+1) – weights (cash + assets)
    """
    
    def __init__(self, feat: int, rows: int, cols: int, dropout: float = 0.2):
        super().__init__()
        self.rows, self.cols = rows, cols

        # ─── TCN blocks ─────────────────────────────────────────
        self.tcn0 = TemporalBlock(feat,  8, (1, 3), (1, 1), dropout)
        self.tcn1 = TemporalBlock(8,   16, (1, 3), (1, 2), dropout)
        self.tcn2 = TemporalBlock(16,  16, (1, 3), (1, 4), dropout)

        # ─── squeeze conv (collapse time dim) ───────────────────
        self.squeeze_conv = nn.Conv2d(16, 16, (1, cols))
        nn.init.normal_(self.squeeze_conv.weight, std=1e-2)
        nn.init.zeros_(self.squeeze_conv.bias)

        # ─── LSTM branch ────────────────────────────────────────
        self.lstm = nn.LSTM(input_size=1, hidden_size=16,
                            batch_first=True, num_layers=1)

        # ─── decision heads ─────────────────────────────────────
        in_ch = 16 + 16 + 1  # conv_feat + lstm_feat + prev_w
        self.decision_i = nn.Conv2d(in_ch, 1, 1)
        self.decision_s = nn.Conv2d(in_ch, 1, 1)
        
        # Better initialization to break symmetry and encourage diversity
        for m in (self.decision_i, self.decision_s):
            nn.init.xavier_uniform_(m.weight)
            nn.init.uniform_(m.bias, -0.1, 0.1)  # Small random bias to break symmetry

        # ─── learnable cash bias ─────────────────────────────────
        self.btc_bias = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def forward(self, x: torch.Tensor, prev_w: torch.Tensor) -> torch.Tensor:
        B    = x.size(0)
        rows = x.size(2)          # actual asset count from input
        if prev_w.size(1) != rows:
            print("[CNN.forward] SHAPE MISMATCH:",
                "x", tuple(x.shape), "prev_w", tuple(prev_w.shape),
                "rows", rows)
            # raise immediately with clear info
            raise RuntimeError(f"prev_w {prev_w.shape} != rows {rows}")

        assert prev_w.shape[1] == rows, f"prev_w {prev_w.shape} != rows {rows}"
        self.rows = rows

        # normalize by last price
        denom = x[:, :, :, -1:].clamp_min(1e-8).expand_as(x)
        x = x / denom

        # ─── TCN path ───────────────────────────────────────────
        y = self.tcn2(self.tcn1(self.tcn0(x)))
        y = F.relu(self.squeeze_conv(y))      # (B,16,rows,1)
        y = y.permute(0, 2, 3, 1)             # → (B, rows,1,16)

        # ─── LSTM path ──────────────────────────────────────────
        lst = x.permute(0, 2, 3, 1).reshape(B*rows, self.cols, 1)
        out,_ = self.lstm(lst)
        lf = out[:, -1, :].view(B, rows, 1, 16)

        # ─── concat conv + lstm + prev_w ───────────────────────
        pw  = prev_w.view(B, rows, 1, 1)               # (B,rows,1,1)
        # Ensure all tensors have consistent shapes for concatenation
        assert y.shape[:3] == lf.shape[:3] == pw.shape[:3], f"Shape mismatch: y={y.shape}, lf={lf.shape}, pw={pw.shape}"
        cat = torch.cat([y, lf, pw], dim=3)                   # (B,rows,1,33)
        cb  = self.btc_bias.expand(B,1,1,cat.size(3))
        cat = torch.cat([cb, cat], dim=1)                     # (B,rows+1,1,33)

        # to channel-first for conv2d
        cat = cat.permute(0, 3, 1, 2)                         # (B,33,rows+1,1)

        print(f"DEBUG network: cat.shape before decisions = {cat.shape}")
        wi_raw = self.decision_i(cat)
        ws_raw = self.decision_s(cat)
        print(f"DEBUG network: wi_raw.shape = {wi_raw.shape}, ws_raw.shape = {ws_raw.shape}")
        
        # More careful squeezing to ensure correct output shape
        wi = wi_raw.view(B, -1)     # (B,rows+1) 
        ws = ws_raw.view(B, -1)     # (B,rows+1)
        print(f"DEBUG network: wi.shape after squeeze = {wi.shape}, ws.shape = {ws.shape}")
        
        # Apply temperature scaling to sharpen distributions
        temperature = 2.0
        wi = torch.softmax(wi / temperature, 1)
        ws = torch.softmax(ws / temperature, 1)
        print(f"DEBUG network: wi after softmax = {wi}")
        print(f"DEBUG network: ws after softmax = {ws}")
        
        # Different combination to encourage diversity
        result = wi * (2.0 - ws)  # This maintains positivity and sum=1
        # Renormalize to ensure sum=1
        result = result / result.sum(dim=1, keepdim=True)
        
        print(f"DEBUG network: result = {result}")
        print(f"DEBUG network: result sum = {result.sum(dim=1)}")
        print(f"DEBUG network: final result.shape = {result.shape}")
        return result
