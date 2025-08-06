# CNN-TCN + LSTM agent rewritten for PyTorch (eager, TF-free).
# The public API -- especially `decide_by_history(...)` --
# stays identical so models/PortfolioPolicyNetwork/model.py

from __future__ import annotations
import json, pathlib, math
from typing import Tuple, Callable, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .network import CNN


class NNAgent:
    """
    Cost-sensitive Portfolio Policy Network implemented in PyTorch.

    Parameters
    ----------
    config : dict (parsed config.json with same keys as original)
    device : "cpu" | "cuda"
    """

    def __init__(self, config: dict, device: str = "cpu"):
        self.cfg = config
        self.dev = torch.device(device)

        feat  = config["input"]["feature_number"]
        rows  = config["input"]["coin_number"]
        cols  = config["input"]["window_size"]
        drop  = config["training"]["dropout"]
        self.net = CNN(feat, rows, cols, drop).to(self.dev)

        self.gamma = config["training"]["gamma"]
        self.alpha = config["training"]["alpha"]
        self.commission = config["trading"]["trading_consumption"]

        lr   = config["training"]["learning_rate"]
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.decay_steps = config["training"]["decay_steps"]
        self.decay_rate  = config["training"]["decay_rate"]
        self.global_step = 0

    # public API
    def decide_by_history(
        self, history: np.ndarray, prev_w_full: np.ndarray
    ) -> np.ndarray:
        """
        history : (window, rows, feat)
        prev_w_full : (rows+1,)  – cash+assets
        returns     : (rows+1,)
        """
        self.net.eval()
        with torch.no_grad():
            hist_t = (
                torch.from_numpy(history.astype(np.float32))
                     .permute(2, 1, 0)   
                     .unsqueeze(0)       
                     .to(self.dev)
            )
            prev = (
                torch.from_numpy(prev_w_full[1:].astype(np.float32))
                     .unsqueeze(0)       
                     .to(self.dev)
            )
            print(f"DEBUG decide_by_history: hist_t.shape = {hist_t.shape}")
            print(f"DEBUG decide_by_history: prev.shape = {prev.shape}")
            print(f"DEBUG decide_by_history: prev_w_full.shape = {prev_w_full.shape}")
            w = self.net(hist_t, prev)      
            return w.squeeze(0).cpu().numpy()

    def train_batch(
        self,
        x: np.ndarray,               
        y_next: np.ndarray,          
        prev_w: np.ndarray,        
    ) -> float:
        """
        Performs one forward/backward step and updates network weights.
        Returns the scalar loss value.
        """
        self.net.train()
        print(f"DEBUG train_batch: x.shape = {x.shape}")
        print(f"DEBUG train_batch: y_next.shape = {y_next.shape}")
        print(f"DEBUG train_batch: prev_w.shape = {prev_w.shape}")
        B, F, R, T = x.shape
        x_t  = torch.tensor(x, dtype=torch.float32, device=self.dev)
        y_t  = torch.tensor(y_next, dtype=torch.float32, device=self.dev).clamp_min(1e-8)
        wprev= torch.tensor(prev_w, dtype=torch.float32, device=self.dev)
        print(f"DEBUG train_batch: x_t.shape = {x_t.shape}")
        print(f"DEBUG train_batch: wprev.shape = {wprev.shape}")
        R_x  = x_t.size(2)
        R_pw = wprev.size(1)
        if R_pw != R_x:
            print("[NNAgent.train_batch] MISMATCH",
                "x_t", tuple(x_t.shape), "wprev", tuple(wprev.shape))
            # emergency slice if cash sneaked in:
            if R_pw == R_x + 1:
                wprev = wprev[:, 1:]
            else:
                raise RuntimeError(f"train_batch mismatch: x_t {x_t.shape}, wprev {wprev.shape}")


        print(f"DEBUG train_batch: About to call net forward with x_t={x_t.shape}, wprev={wprev.shape}")
        w = self.net(x_t, wprev)           
        print(f"DEBUG train_batch: net forward completed, w.shape = {w.shape}")

        # cash index 0, asset cols 1:                            (B,R)
        w_assets = w[:, 1:]
        price_vec = torch.cat([torch.ones(B,1, device=self.dev), y_t], dim=1)

        # turn-over cost
        turnover = torch.abs(w_assets - wprev).sum(dim=1)        

        # portfolio growth vector
        pv = (w * price_vec).sum(dim=1)
        pv = pv * (1 - self.commission * turnover).clamp_min(1e-8)
        pv = pv.clamp_min(1e-8)                    # <- never allow <=0

        neg_log_growth = -torch.log(pv).mean()
        var_penalty    = torch.var(torch.log(pv), unbiased=False)
        cost_penalty   = turnover.mean()

        loss = (
            neg_log_growth +
            self.gamma * cost_penalty +
            self.alpha * var_penalty
        )

        # NaN check after loss is calculated
        if torch.isnan(w).any() or torch.isnan(pv).any() or torch.isnan(loss).any():
            print("[NaN DEBUG] w:", w.detach().cpu().numpy())
            print("[NaN DEBUG] pv:", pv.detach().cpu().numpy())
            print("[NaN DEBUG] y_t:", y_t.detach().cpu().numpy())
            print("[NaN DEBUG] loss:", loss.detach().cpu().numpy())
            raise RuntimeError("NaN detected in train_batch")

        # optimization step
        # Store some weights before optimization to check if they're changing
        before_weights = self.net.decision_i.weight.data.clone()
        
        self.opt.zero_grad()
        loss.backward()
        
        # Check if gradients are being computed
        total_grad_norm = 0.0
        param_count = 0
        for param in self.net.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.data.norm(2).item() ** 2
                param_count += 1
        total_grad_norm = total_grad_norm ** 0.5
        
        # Clip gradients to prevent vanishing/exploding gradients that cause uniform convergence
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        
        self.opt.step()
        
        # Check if weights actually changed
        after_weights = self.net.decision_i.weight.data.clone()
        weight_change = (after_weights - before_weights).norm().item()
        
        print(f"DEBUG train_batch: loss = {loss.item():.6f}, grad_norm = {total_grad_norm:.6f}, weight_change = {weight_change:.8f}")

        # lr decay
        self.global_step += 1
        if self.global_step % self.decay_steps == 0:
            for g in self.opt.param_groups:
                g["lr"] *= self.decay_rate

        return float(loss.detach().cpu())

    def save(self, path: str | pathlib.Path):
        torch.save(self.net.state_dict(), path)

    def load(self, path: str | pathlib.Path):
        self.net.load_state_dict(torch.load(path, map_location=self.dev))
