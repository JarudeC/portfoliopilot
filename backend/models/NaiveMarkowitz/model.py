import numpy as np
import pandas as pd


class Naive_Markowitz:
    """
    Classic mean-variance portfolio with an optional Gaussian-noise alpha.

    The expected-return vector is *always* estimated on the look-back window
    (strictly past data).  Noise is added in-sample; no evaluation-window
    peeking.
    """

    def __init__(
        self,
        historical_data: pd.DataFrame,
        lookback_window: list[int],          # [start_row, end_row]
        evaluation_window: int,              # e.g. 5 days
        eta: float,                          # noise scale (0 ⇒ no noise)
        markowitz_type: str = "expected_returns",
        transaction_cost_rate: float = 0.0001,
    ):
        self.historical_data = historical_data
        self.lb0, self.lb1 = lookback_window
        self.evaluation_window = evaluation_window
        self.eta = eta
        self.markowitz_type = markowitz_type
        self.tc = transaction_cost_rate

        # pre-compute once
        self.cov = self._cov_matrix()                  # (N,N)
        self.mu  = self._exp_returns_with_noise()      # (N,)
        self.w   = self._markowitz_weights()           # (N,)

    # ────────────────────────────────────────────────────────────────
    # helpers
    # ────────────────────────────────────────────────────────────────
    def _cov_matrix(self) -> pd.DataFrame:
        rets = self.historical_data.iloc[self.lb0:self.lb1].pct_change().dropna()
        cov = rets.cov().fillna(0.)
        cov.values[np.diag_indices_from(cov)] += 1e-4   # ridge
        return cov

    def _exp_returns_with_noise(self) -> pd.Series:
        """Past-window mean + optional zero-mean Gaussian noise."""
        mu = (
            self.historical_data
            .iloc[self.lb0:self.lb1]
            .pct_change()
            .dropna()
            .mean()
        )

        if self.eta == 0:
            return mu

        # noise scale proportional to std(μ) * eta
        noise_sd = mu.std() * self.eta
        noise = np.random.normal(0, noise_sd, size=len(mu))
        return mu + noise

    def _markowitz_weights(self) -> np.ndarray:
        e = np.ones(len(self.mu))
        try:
            cov_inv = np.linalg.inv(self.cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(self.cov)

        w_minvar = cov_inv @ e / (e @ cov_inv @ e)
        w_mk     = cov_inv @ self.mu / (e @ cov_inv @ self.mu)

        if self.markowitz_type == "min_variance":
            w = w_minvar
        else:                                         # expected_returns
            target = 0.0008                           # 0.08 % daily
            alpha = (target - self.mu @ w_minvar) / (self.mu @ (w_mk - w_minvar))
            w = w_minvar if self.mu @ w_minvar >= target else w_minvar + alpha * (w_mk - w_minvar)

        # gross-leverage = 1
        return w / np.sum(np.abs(w))

    # ────────────────────────────────────────────────────────────────
    # public API
    # ────────────────────────────────────────────────────────────────
    def weights(self) -> pd.Series:
        return pd.Series(self.w, index=self.historical_data.columns, name="weight")

    def forward_returns(self) -> pd.Series:
        win = (
            self.historical_data
            .iloc[self.lb1 - 1 : self.lb1 + self.evaluation_window]  # include t-1
            .pct_change()
            .dropna()
        )                                            # shape (eval_win , N)
        return win @ self.w  