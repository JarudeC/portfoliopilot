"""Shared utilities for backtesting strategies."""

import logging
import numpy as np
import pandas as pd
from typing import List

logger = logging.getLogger(__name__)


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Normalize weights to sum to 1 (by absolute value for leverage)."""
    total = np.sum(np.abs(weights))
    if total == 0:
        return np.ones(len(weights)) / len(weights)
    return weights / total


def clip_returns(
    returns: pd.Series,
    cap: float = 0.10
) -> pd.Series:
    """Clip returns to [-cap, cap] and handle inf/nan values."""
    return (
        returns
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .clip(lower=-cap, upper=cap)
    )


def rolling_window_indices(
    total_length: int,
    lookback: int,
    eval_win: int
) -> List[tuple]:
    """
    Generate (start, end) indices for rolling window backtest.

    Args:
        total_length: Total number of data points
        lookback: Lookback window size
        eval_win: Evaluation window size

    Returns:
        List of (window_start, window_end) tuples
    """
    indices = []
    start = 0
    end = lookback

    while end + eval_win <= total_length:
        indices.append((start, end))
        start += eval_win
        end += eval_win

    return indices


def safe_inverse(matrix: np.ndarray) -> np.ndarray:
    """Invert matrix, falling back to pseudo-inverse if singular."""
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        logger.warning("Matrix singular, using pseudo-inverse")
        return np.linalg.pinv(matrix)


def add_ridge_regularization(
    cov_matrix: np.ndarray,
    ridge_lambda: float = 1e-4
) -> np.ndarray:
    """Add ridge regularization to covariance matrix diagonal."""
    cov = cov_matrix.copy()
    np.fill_diagonal(cov, cov.diagonal() + ridge_lambda)
    return cov
