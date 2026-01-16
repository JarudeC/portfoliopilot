"""Mean-variance optimization logic for Naive Markowitz strategy."""

import logging
import numpy as np
import pandas as pd

from ..utils import safe_inverse, add_ridge_regularization

logger = logging.getLogger(__name__)


def compute_covariance_matrix(
    returns: pd.DataFrame,
    ridge_lambda: float = 1e-4
) -> np.ndarray:
    """
    Compute covariance matrix with ridge regularization.

    Args:
        returns: Daily returns DataFrame
        ridge_lambda: Regularization added to diagonal

    Returns:
        Regularized covariance matrix
    """
    cov = returns.cov().fillna(0).values
    return add_ridge_regularization(cov, ridge_lambda)


def compute_expected_returns(
    returns: pd.DataFrame,
    eta: float = 0.0,
    seed: int | None = None
) -> np.ndarray:
    """
    Compute expected returns with optional noise.

    Args:
        returns: Daily returns DataFrame
        eta: Noise scale (0 = no noise)
        seed: Random seed for reproducibility

    Returns:
        Expected return vector
    """
    mu = returns.mean().values

    if eta > 0:
        if seed is not None:
            np.random.seed(seed)
        noise_scale = mu.std() * eta
        mu = mu + np.random.normal(0, noise_scale, size=len(mu))

    return mu


def markowitz_weights(
    mu: np.ndarray,
    cov: np.ndarray,
    target_return: float | None = None,
    min_variance: bool = False
) -> np.ndarray:
    """
    Compute Markowitz portfolio weights.

    Args:
        mu: Expected return vector
        cov: Covariance matrix
        target_return: Target portfolio return (ignored if min_variance=True)
        min_variance: If True, compute minimum variance portfolio

    Returns:
        Portfolio weights (normalized to sum of abs = 1)
    """
    n_assets = len(mu)
    ones = np.ones(n_assets)
    cov_inv = safe_inverse(cov)

    # Minimum variance portfolio
    w_minvar = cov_inv @ ones / (ones @ cov_inv @ ones)

    if min_variance or target_return is None:
        weights = w_minvar
    else:
        # Markowitz tangent portfolio
        w_mk = cov_inv @ mu
        denom = ones @ cov_inv @ mu
        if abs(denom) > 1e-10:
            w_mk = w_mk / denom
        else:
            w_mk = w_minvar

        # Target return interpolation
        minvar_return = mu @ w_minvar
        if minvar_return >= target_return:
            weights = w_minvar
        else:
            mk_return = mu @ w_mk
            diff = mk_return - minvar_return
            if abs(diff) > 1e-10:
                alpha = (target_return - minvar_return) / diff
                weights = w_minvar + alpha * (w_mk - w_minvar)
            else:
                weights = w_minvar

    # Normalize by sum of absolute weights (gross leverage = 1)
    total = np.sum(np.abs(weights))
    if total > 0:
        weights = weights / total

    return weights
