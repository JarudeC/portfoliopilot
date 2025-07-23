# models/PortfolioPolicyNetwork/model.py
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pgportfolio_pytorch.learn.nnagent import NNAgent

class PortfolioPolicyNetwork:
    def __init__(
        self,
        *,
        lookback: int,              # ← window size from UI
        n_assets: int,              # ← len(tickers) from UI
        config_path: str | Path | None = None,
        device: str = "cpu",
    ):
        if config_path is None:
            config_path = Path(__file__).parent / "config.json"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")

        cfg = json.loads(config_path.read_text())
        cfg["input"]["window_size"] = lookback
        cfg["input"]["coin_number"] = n_assets
        cfg.setdefault("training", {}).setdefault("dropout", 0.2)
        self._agent = NNAgent(cfg, device=device)

    # In your model.py, add better error handling to the predict method:

    def predict(self, h: np.ndarray, prev: np.ndarray) -> np.ndarray:
        print(f"DEBUG predict: h.shape = {h.shape}")
        print(f"DEBUG predict: prev.shape = {prev.shape}")
        
        # Check if prev has the right number of assets
        expected_assets = self._agent.cfg['input']['coin_number']
        print(f"DEBUG predict: expected n_assets = {expected_assets}")
        if prev.shape[0] != expected_assets:
            raise ValueError(f"prev weights shape {prev.shape[0]} doesn't match expected assets {expected_assets}")
        
        h32 = h.astype(np.float32)
        prev32 = prev.astype(np.float32)
        
        # The issue might be here - make sure concatenation is correct
        print(f"DEBUG predict: prev32.shape before concat = {prev32.shape}")
        full_prev = np.concatenate(([0.0], prev32)).astype(np.float32)
        print(f"DEBUG predict: full_prev.shape after concat = {full_prev.shape}")
        
        try:
            w_full = self._agent.decide_by_history(h32, full_prev)
            print(f"DEBUG predict: w_full.shape = {w_full.shape if hasattr(w_full, 'shape') else type(w_full)}")
        except Exception as e:
            print(f"ERROR in decide_by_history: {e}")
            print(f"h32.shape: {h32.shape}")
            print(f"full_prev.shape: {full_prev.shape}")
            raise
        
        w = np.asarray(w_full, dtype=np.float32)
        if w.ndim == 2:
            w = w.squeeze(0)
        
        print(f"DEBUG predict: final w.shape before return = {w.shape}")
        # Remove cash position but renormalize the asset weights to sum to 1.0
        asset_weights = w[1:]  # Remove cash position
        asset_sum = asset_weights.sum()
        if asset_sum > 0:
            result = asset_weights / asset_sum  # Renormalize to sum=1.0
        else:
            result = np.full_like(asset_weights, 1.0 / len(asset_weights))
        print(f"DEBUG predict: returning shape = {result.shape}")
        print(f"DEBUG predict: asset weights sum = {result.sum()}")
        return result
