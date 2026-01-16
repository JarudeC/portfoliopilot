"""State index constants for margin trading environment.

Replaces magic numbers with named constants for maintainability.
State vector layout:
  [0-2]  Long account: cash, loan, equity
  [3-5]  Short account: limit, credit, equity
  [6:6+N]  Asset prices
  [6+N:6+2N]  Holdings (positive=long, negative=short)
  [6+2N:]  Technical indicators (optional)
"""


class StateIndex:
    """Named indices for the state vector."""

    # Long account (indices 0-2)
    LONG_CASH = 0
    LONG_LOAN = 1
    LONG_EQUITY = 2

    # Short account (indices 3-5)
    SHORT_LIMIT = 3
    SHORT_CREDIT = 4
    SHORT_EQUITY = 5

    # Offset where prices begin
    PRICES_START = 6

    @staticmethod
    def prices_slice(n_assets: int) -> slice:
        """Slice for asset prices."""
        return slice(StateIndex.PRICES_START, StateIndex.PRICES_START + n_assets)

    @staticmethod
    def holdings_slice(n_assets: int) -> slice:
        """Slice for asset holdings."""
        start = StateIndex.PRICES_START + n_assets
        return slice(start, start + n_assets)

    @staticmethod
    def price_index(asset_idx: int) -> int:
        """Index for a specific asset's price."""
        return StateIndex.PRICES_START + asset_idx

    @staticmethod
    def holding_index(asset_idx: int, n_assets: int) -> int:
        """Index for a specific asset's holding."""
        return StateIndex.PRICES_START + n_assets + asset_idx
