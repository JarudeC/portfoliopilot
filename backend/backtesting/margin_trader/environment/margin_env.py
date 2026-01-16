"""Margin trading environment for A2C reinforcement learning.

A Gymnasium environment supporting long/short positions with margin.
"""

from __future__ import annotations

import logging
from typing import List

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

from .state import StateIndex

matplotlib.use("Agg")
logger = logging.getLogger(__name__)

# Constants
ACCOUNT_FIELDS = 6  # 3 long + 3 short account fields


class MarginTradingEnv(gym.Env):
    """Margin trading environment for OpenAI Gymnasium.

    Supports long and short positions with margin requirements,
    maintenance margin checks, and leverage limits.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int,
        initial_amount: int,
        num_stock_shares: List[int],
        buy_cost_pct: List[float],
        sell_cost_pct: List[float],
        reward_scaling: float,
        state_space: int,
        action_space: int,
        tech_indicator_list: List[str],
        turbulence_threshold: float | None = None,
        risk_indicator_col: str = "turbulence",
        make_plots: bool = False,
        print_verbosity: int = 10,
        margin: float = 2,
        long_short_ratio: float = 1,
        maintenance: float = 0.4,
        penalty_sharpe: float = 0.001,
        max_leverage: float = 1.5,
        day: int = 0,
        initial: bool = True,
        previous_state: List = None,
        model_name: str = "",
        mode: str = "",
        iteration: str = "",
        partialtrade: bool = False,
        period: str = "Day",
        num_periods: int = 30,
    ):
        """Initialize margin trading environment.

        Args:
            df: Price data with columns [date, tic, close]
            stock_dim: Number of assets
            hmax: Maximum shares per trade
            initial_amount: Starting capital
            num_stock_shares: Initial holdings per asset
            buy_cost_pct: Transaction cost for buys per asset
            sell_cost_pct: Transaction cost for sells per asset
            reward_scaling: Reward multiplier
            state_space: State vector dimension
            action_space: Action vector dimension
            tech_indicator_list: Technical indicator column names
            turbulence_threshold: Risk threshold for position clearing
            risk_indicator_col: Column name for risk indicator
            margin: Margin multiplier (2 = 100% borrowing)
            long_short_ratio: Ratio of long to short capital
            maintenance: Maintenance margin requirement
            penalty_sharpe: Sharpe ratio penalty weight
            max_leverage: Maximum allowed leverage
        """
        previous_state = previous_state or []

        self.day = day
        self.df = df.copy()
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.dates = sorted(self.df["date"].unique())
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.num_stock_shares = num_stock_shares
        self.initial_amount = initial_amount
        self.max_leverage = max_leverage
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.tech_indicator_list = tech_indicator_list

        # Gymnasium spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )

        self.data = self.df[self.df["date"] == self.dates[self.day]]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        self.margin = margin
        self.long_short_ratio = long_short_ratio
        self.maintenance = maintenance
        self.penalty_sharpe = penalty_sharpe
        self.partialtrade = partialtrade
        self.period = period
        self.num_periods = num_periods

        # Initialize state
        self.state = self._initiate_margin_state()

        # Initialize tracking variables
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0

        # Memory for tracking
        initial_asset = self.initial_amount + np.sum(
            np.array(self.num_stock_shares)
            * np.array(self.state[ACCOUNT_FIELDS:ACCOUNT_FIELDS + self.stock_dim])
        )
        self.asset_memory = [initial_asset]
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = []
        self.date_memory = [self._get_date()]

        self._seed()

        # Date tracking for period-based updates
        date_info = self.dates[self.day]
        self.weekday = date_info.isocalendar().weekday
        self.week = date_info.isocalendar().week
        self.month = date_info.month
        self.year = date_info.isocalendar().year
        self.period_counter = 0

    def step(self, actions):
        """Execute one environment step.

        Args:
            actions: Action vector from agent

        Returns:
            Tuple of (state, reward, terminated, truncated, info)
        """
        self.terminal = self.day >= len(self.dates) - 1

        if self.terminal:
            return self._handle_terminal_step()

        # Scale actions by hmax and convert to integer shares
        actions = (actions * self.hmax).astype(int)
        actions = self._check_one_position_only(actions)

        long_actions = actions[:self.stock_dim]
        short_actions = actions[self.stock_dim:]

        begin_total_asset = (
            self.state[StateIndex.LONG_EQUITY]
            + self.state[StateIndex.SHORT_EQUITY]
        )

        # Execute long trades
        self._execute_long_trades(long_actions)

        # Execute short trades
        self._execute_short_trades(short_actions)

        self.actions_memory.append(np.concatenate([long_actions, short_actions]))

        # Apply leverage constraint
        self._apply_leverage_constraint()

        # Update period counter and check margin requirements
        self._update_period_counter()

        # Advance to next day
        self.day += 1
        self.data = self.df[self.df["date"] == self.dates[self.day]]

        if self.turbulence_threshold is not None:
            self._update_turbulence()

        self.state = self._update_state()

        end_total_asset = (
            self.state[StateIndex.LONG_EQUITY]
            + self.state[StateIndex.SHORT_EQUITY]
        )

        self.asset_memory.append(end_total_asset)
        self.date_memory.append(self._get_date())

        # Calculate reward
        self.reward = self._calculate_reward(begin_total_asset, end_total_asset)
        self.rewards_memory.append(self.reward)
        self.state_memory.append(self.state)

        return self.state, self.reward, self.terminal, False, {}

    def reset(self, *, seed=None, options=None):
        """Reset environment to initial state.

        Returns:
            Tuple of (initial_state, info_dict)
        """
        self.day = 0
        self.data = self.df[self.df["date"] == self.dates[self.day]]
        self.state = self._initiate_margin_state()

        if self.initial:
            self.asset_memory = [self.initial_amount]
        else:
            previous_total = (
                self.previous_state[StateIndex.LONG_EQUITY]
                + self.previous_state[StateIndex.SHORT_EQUITY]
            )
            self.asset_memory = [previous_total]

        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = []
        self.date_memory = [self._get_date()]
        self.episode += 1

        return self.state, {}

    def render(self, mode="human", close=False):
        """Render current state."""
        return self.state

    def get_sb_env(self):
        """Get Stable Baselines compatible environment.

        Returns:
            Tuple of (DummyVecEnv, initial_observation)
        """
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

    # ---------- State initialization and update ----------

    def _initiate_margin_state(self):
        """Create initial state vector."""
        if self.initial:
            equity_long = (
                self.long_short_ratio / (self.long_short_ratio + 1)
                * self.initial_amount
            )
            cash_long = equity_long * self.margin
            loan = equity_long * (self.margin - 1)

            equity_short = self.initial_amount - equity_long
            limit_short = equity_short * self.margin
            credit = limit_short + equity_short

            state = (
                [cash_long, loan, equity_long]
                + [limit_short, credit, equity_short]
                + self.data.close.values.tolist()
                + self.num_stock_shares
                + sum(
                    (self.data[tech].values.tolist() for tech in self.tech_indicator_list),
                    [],
                )
            )
        else:
            state = (
                self.previous_state[0:ACCOUNT_FIELDS]
                + self.data.close.values.tolist()
                + self.previous_state[
                    ACCOUNT_FIELDS + self.stock_dim:ACCOUNT_FIELDS + 2 * self.stock_dim
                ]
                + sum(
                    (self.data[tech].values.tolist() for tech in self.tech_indicator_list),
                    [],
                )
            )

        return state

    def _update_state(self):
        """Update state vector after price changes."""
        long_cash = self.state[StateIndex.LONG_CASH]
        loan = self.state[StateIndex.LONG_LOAN]

        prices = np.array(self.data.close.values.tolist())
        holdings = np.array(
            self.state[ACCOUNT_FIELDS + self.stock_dim:ACCOUNT_FIELDS + 2 * self.stock_dim]
        )
        market_values = prices * holdings

        # Long position equity
        long_market = np.sum(market_values[market_values > 0])
        long_equity = long_cash + long_market - loan

        # Short position equity
        limit = self.state[StateIndex.SHORT_LIMIT]
        credit = self.state[StateIndex.SHORT_CREDIT]
        short_market = np.abs(np.sum(market_values[market_values < 0]))
        short_equity = credit - limit - short_market

        # Floor negative equity
        if short_equity < 0:
            short_equity = 0

        state = (
            [long_cash, loan, long_equity]
            + [limit, credit, short_equity]
            + self.data.close.values.tolist()
            + list(self.state[ACCOUNT_FIELDS + self.stock_dim:ACCOUNT_FIELDS + 2 * self.stock_dim])
            + sum(
                (self.data[tech].values.tolist() for tech in self.tech_indicator_list),
                [],
            )
        )

        # Numerical stabilization
        clip_limit = self.initial_amount * 10
        state = np.nan_to_num(state, nan=0.0)
        state = np.clip(state, -clip_limit, clip_limit).tolist()

        return state

    # ---------- Trading operations ----------

    def _execute_long_trades(self, actions):
        """Execute long position trades."""
        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                actions = np.array([-self.hmax] * self.stock_dim)

        argsort = np.argsort(actions)
        sell_indices = argsort[:np.where(actions < 0)[0].shape[0]]
        buy_indices = argsort[::-1][:np.where(actions > 0)[0].shape[0]]

        for idx in sell_indices:
            actions[idx] = -self._sell_long_stock(idx, actions[idx])

        for idx in buy_indices:
            actions[idx] = self._buy_long_stock(idx, actions[idx])

    def _execute_short_trades(self, actions):
        """Execute short position trades."""
        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                actions = np.array([-self.hmax] * self.stock_dim)

        argsort = np.argsort(actions)
        sell_indices = argsort[:np.where(actions < 0)[0].shape[0]]
        buy_indices = argsort[::-1][:np.where(actions > 0)[0].shape[0]]

        for idx in buy_indices:
            actions[idx] = self._buy_short_stock(idx, actions[idx])

        for idx in sell_indices:
            actions[idx] = -self._sell_short_stock(idx, actions[idx])

    def _sell_long_stock(self, index, action):
        """Sell long position."""
        if self._check_long_maintenance() <= self.maintenance:
            return 0

        price_idx = ACCOUNT_FIELDS + index
        holding_idx = ACCOUNT_FIELDS + self.stock_dim + index
        price = self.state[price_idx]
        holding = self.state[holding_idx]

        if holding <= 0:
            return 0

        sell_shares = min(abs(action), holding)
        sell_amount = price * sell_shares * (1 - self.sell_cost_pct[index])
        cost = price * sell_shares * self.sell_cost_pct[index]

        self.state[StateIndex.LONG_CASH] += sell_amount
        self.state[StateIndex.LONG_EQUITY] -= cost
        self.state[holding_idx] -= sell_shares
        self.cost += cost
        self.trades += 1

        return sell_shares

    def _buy_long_stock(self, index, action):
        """Buy long position."""
        if self._check_long_maintenance() <= self.maintenance:
            return 0

        price_idx = ACCOUNT_FIELDS + index
        holding_idx = ACCOUNT_FIELDS + self.stock_dim + index
        price = self.state[price_idx]
        cash = self.state[StateIndex.LONG_CASH]

        max_shares = cash / (price * (1 + self.buy_cost_pct[index]))
        buy_shares = min(max_shares, abs(action))

        if buy_shares <= 0:
            return 0

        buy_amount = price * buy_shares
        cost = buy_amount * self.buy_cost_pct[index]

        self.state[StateIndex.LONG_CASH] -= (buy_amount + cost)
        self.state[StateIndex.LONG_EQUITY] -= cost
        self.state[holding_idx] += buy_shares
        self.cost += cost
        self.trades += 1

        return buy_shares

    def _sell_short_stock(self, index, action):
        """Sell short (open short position)."""
        if self._check_short_maintenance() <= self.maintenance:
            return 0

        price_idx = ACCOUNT_FIELDS + index
        holding_idx = ACCOUNT_FIELDS + self.stock_dim + index
        price = self.state[price_idx]
        limit = self.state[StateIndex.SHORT_LIMIT]

        available = int(limit // price)
        sell_shares = min(available, abs(action))

        if sell_shares <= 0:
            return 0

        sell_amount = price * sell_shares
        cost = sell_amount * self.sell_cost_pct[index]

        self.state[StateIndex.SHORT_LIMIT] -= sell_amount
        self.state[StateIndex.SHORT_CREDIT] -= cost
        self.state[StateIndex.SHORT_EQUITY] -= cost
        self.state[holding_idx] -= sell_shares
        self.cost += cost
        self.trades += 1

        return sell_shares

    def _buy_short_stock(self, index, action):
        """Buy to cover short position."""
        if self._check_short_maintenance() <= self.maintenance:
            return 0

        price_idx = ACCOUNT_FIELDS + index
        holding_idx = ACCOUNT_FIELDS + self.stock_dim + index
        price = self.state[price_idx]
        holding = self.state[holding_idx]

        if holding >= 0:
            return 0

        buy_shares = min(action, abs(holding))
        buy_amount = price * buy_shares
        cost = buy_amount * self.buy_cost_pct[index]

        self.state[StateIndex.SHORT_LIMIT] += buy_amount
        self.state[StateIndex.SHORT_CREDIT] -= cost
        self.state[StateIndex.SHORT_EQUITY] -= cost
        self.state[holding_idx] += buy_shares
        self.cost += cost
        self.trades += 1

        return buy_shares

    # ---------- Maintenance and margin checks ----------

    def _check_long_maintenance(self):
        """Calculate long position maintenance ratio."""
        prices = np.array(self.state[ACCOUNT_FIELDS:ACCOUNT_FIELDS + self.stock_dim])
        holdings = np.array(
            self.state[ACCOUNT_FIELDS + self.stock_dim:ACCOUNT_FIELDS + 2 * self.stock_dim]
        )
        market_values = prices * holdings
        long_market = np.sum(market_values[market_values > 0])
        equity = self.state[StateIndex.LONG_EQUITY]

        return equity / long_market if long_market > 0 else 1.0

    def _check_short_maintenance(self):
        """Calculate short position maintenance ratio."""
        prices = np.array(self.state[ACCOUNT_FIELDS:ACCOUNT_FIELDS + self.stock_dim])
        holdings = np.array(
            self.state[ACCOUNT_FIELDS + self.stock_dim:ACCOUNT_FIELDS + 2 * self.stock_dim]
        )
        market_values = prices * holdings
        short_market = np.abs(np.sum(market_values[market_values < 0]))
        equity = self.state[StateIndex.SHORT_EQUITY]

        return equity / short_market if short_market > 0 else 1.0

    def _check_one_position_only(self, actions):
        """Ensure each asset has only long OR short position, not both."""
        for i in range(self.stock_dim):
            combined = actions[i] + actions[i + self.stock_dim]
            holding_idx = ACCOUNT_FIELDS + self.stock_dim + i
            holding = self.state[holding_idx]

            if holding > 0:  # Currently long
                actions[i] = combined
                actions[i + self.stock_dim] = 0
            elif holding < 0:  # Currently short
                actions[i + self.stock_dim] = combined
                actions[i] = 0
            else:  # No position
                if combined > 0:
                    actions[i] = combined
                    actions[i + self.stock_dim] = 0
                elif combined < 0:
                    actions[i + self.stock_dim] = combined
                    actions[i] = 0
                else:
                    actions[i] = 0
                    actions[i + self.stock_dim] = 0

        return actions

    def _apply_leverage_constraint(self):
        """Scale positions if leverage exceeds maximum."""
        prices = np.array(self.data.close.tolist())
        hold_slice = slice(ACCOUNT_FIELDS + self.stock_dim, ACCOUNT_FIELDS + 2 * self.stock_dim)
        positions = np.array(self.state[hold_slice])

        gross_long = np.sum(np.maximum(positions, 0) * prices)
        gross_short = np.sum(np.abs(np.minimum(positions, 0) * prices))
        gross = gross_long + gross_short

        equity = max(
            self.state[StateIndex.LONG_EQUITY] + self.state[StateIndex.SHORT_EQUITY],
            1
        )
        allowable = self.max_leverage * equity

        if gross > allowable:
            scale = allowable / gross
            new_positions = positions * scale
            self.state[hold_slice] = new_positions.tolist()

            # Credit cash/limit for scaled down positions
            diff = positions - new_positions
            delta_long = np.sum(np.maximum(diff, 0) * prices)
            delta_short = np.sum(np.abs(np.minimum(diff, 0)) * prices)
            self.state[StateIndex.LONG_CASH] += delta_long
            self.state[StateIndex.SHORT_LIMIT] += delta_short

    # ---------- Loan and credit updates ----------

    def _update_loan(self):
        """Update long account loan based on equity changes."""
        cash = self.state[StateIndex.LONG_CASH]
        loan = self.state[StateIndex.LONG_LOAN]
        equity = self.state[StateIndex.LONG_EQUITY]
        loan_diff = equity - loan

        if loan_diff > 0:
            self.state[StateIndex.LONG_LOAN] = equity
            self.state[StateIndex.LONG_CASH] += loan_diff
        else:
            if cash >= abs(loan_diff):
                self.state[StateIndex.LONG_CASH] -= abs(loan_diff)
                self.state[StateIndex.LONG_LOAN] -= abs(loan_diff)
            else:
                self._liquidate_long_positions(abs(loan_diff) - cash)
                self.state[StateIndex.LONG_LOAN] -= abs(loan_diff)

    def _update_credit(self):
        """Update short account credit based on equity changes."""
        limit = self.state[StateIndex.SHORT_LIMIT]
        credit = self.state[StateIndex.SHORT_CREDIT]
        equity = self.state[StateIndex.SHORT_EQUITY]

        prices = np.array(self.state[ACCOUNT_FIELDS:ACCOUNT_FIELDS + self.stock_dim])
        holdings = np.array(
            self.state[ACCOUNT_FIELDS + self.stock_dim:ACCOUNT_FIELDS + 2 * self.stock_dim]
        )
        market_values = prices * holdings
        short_market = np.abs(np.sum(market_values[market_values < 0]))

        borrow_limit = limit + short_market
        borrow_diff = self.margin * equity - borrow_limit

        if borrow_diff > 0:
            self.state[StateIndex.SHORT_LIMIT] += borrow_diff
            self.state[StateIndex.SHORT_CREDIT] += borrow_diff
        else:
            if limit >= abs(borrow_diff):
                self.state[StateIndex.SHORT_LIMIT] += borrow_diff
                self.state[StateIndex.SHORT_CREDIT] += borrow_diff
            else:
                self._cover_short_positions(borrow_diff + limit)
                self.state[StateIndex.SHORT_CREDIT] -= abs(borrow_diff)

    def _liquidate_long_positions(self, required_cash: float):
        """Force sell long positions to meet margin requirements."""
        prices = np.array(self.state[ACCOUNT_FIELDS:ACCOUNT_FIELDS + self.stock_dim])
        holdings = np.array(
            self.state[ACCOUNT_FIELDS + self.stock_dim:ACCOUNT_FIELDS + 2 * self.stock_dim]
        )
        market_values = prices * holdings

        self.state[StateIndex.LONG_CASH] = 0
        remaining = required_cash

        long_indices = np.where(market_values > 0)[0]
        sorted_indices = long_indices[np.argsort(market_values[long_indices])]

        for idx in sorted_indices:
            holding = holdings[idx]
            self._sell_long_stock(idx, holding)

            if market_values[idx] < remaining:
                remaining -= market_values[idx]
                self.state[StateIndex.LONG_CASH] = 0
            else:
                self.state[StateIndex.LONG_CASH] -= remaining
                break

    def _cover_short_positions(self, required_limit: float):
        """Force cover short positions to meet margin requirements."""
        prices = np.array(self.state[ACCOUNT_FIELDS:ACCOUNT_FIELDS + self.stock_dim])
        holdings = np.array(
            self.state[ACCOUNT_FIELDS + self.stock_dim:ACCOUNT_FIELDS + 2 * self.stock_dim]
        )
        market_values = prices * holdings

        self.state[StateIndex.SHORT_LIMIT] = 0
        remaining = required_limit

        short_indices = np.where(market_values < 0)[0]
        sorted_indices = short_indices[np.argsort(market_values[short_indices])[::-1]]

        for idx in sorted_indices:
            holding = abs(holdings[idx])
            self._buy_short_stock(idx, holding)

            if market_values[idx] > remaining:
                remaining += abs(market_values[idx])
                self.state[StateIndex.SHORT_LIMIT] = 0
            else:
                self.state[StateIndex.SHORT_LIMIT] -= abs(remaining)
                break

    # ---------- Period and turbulence updates ----------

    def _update_period_counter(self):
        """Update period counter and check margin requirements."""
        cur_date = self.dates[self.day]

        if self.period == "Day":
            if self.day != 0 and self.day % self.num_periods == 0:
                self.period_counter = self.num_periods
        elif self.period == "Week":
            if self.week != cur_date.isocalendar().week:
                self.period_counter += 1
        elif self.period == "Month":
            if self.month != cur_date.month:
                self.period_counter += 1
        elif self.period == "Year":
            if self.year != cur_date.isocalendar().year:
                self.period_counter += 1

        self.weekday = cur_date.isocalendar().weekday
        self.week = cur_date.isocalendar().week
        self.month = cur_date.month
        self.year = cur_date.isocalendar().year

        if self.period_counter >= self.num_periods:
            self._update_loan()
            self._update_credit()
            self.period_counter = 0
        else:
            if self._check_long_maintenance() < 0.3:
                self._update_loan()
            if self._check_short_maintenance() < 0.3:
                self._update_credit()

    def _update_turbulence(self):
        """Update turbulence indicator from current data."""
        if len(self.df.tic.unique()) == 1:
            self.turbulence = self.data[self.risk_indicator_col].values[0]
        else:
            self.turbulence = self.data[self.risk_indicator_col].values[0]

    # ---------- Reward calculation ----------

    def _calculate_reward(self, begin_asset: float, end_asset: float) -> float:
        """Calculate step reward with Sharpe penalty."""
        equity_base = max(begin_asset, self.initial_amount * 0.25)

        if np.isnan(begin_asset) or np.isnan(end_asset) or equity_base <= 0:
            raw_ret = 0.0
        else:
            raw_ret = (end_asset - begin_asset) / equity_base

        if np.isnan(raw_ret) or np.isinf(raw_ret):
            raw_ret = 0.0

        reward = float(np.clip(raw_ret, -1.0, 1.0)) * self.reward_scaling

        # Sharpe penalty
        if len(self.asset_memory) >= 5:
            daily_return = pd.Series(self.asset_memory[-5:]).pct_change(1)
            std_return = daily_return.std()

            if std_return > 1e-8 and not np.isnan(std_return):
                sharpe = (252**0.5) * daily_return.mean() / std_return
                sharpe = np.clip(sharpe, -10.0, 10.0)
            else:
                sharpe = 0.0

            if np.isnan(sharpe):
                sharpe = 0.0
        else:
            sharpe = 0.0

        reward = reward * self.reward_scaling + sharpe * self.penalty_sharpe

        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0

        return reward

    # ---------- Terminal step handling ----------

    def _handle_terminal_step(self):
        """Handle episode termination."""
        if self.make_plots:
            self._make_plot()

        end_total_asset = (
            self.state[StateIndex.LONG_EQUITY]
            + self.state[StateIndex.SHORT_EQUITY]
        )

        if self.episode % self.print_verbosity == 0:
            logger.info(
                "Episode %d: day=%d, begin=%.2f, end=%.2f, cost=%.2f, trades=%d",
                self.episode, self.day,
                self.asset_memory[0], end_total_asset,
                self.cost, self.trades
            )

        return self.state, self.reward, self.terminal, False, {}

    def _make_plot(self):
        """Save portfolio value plot."""
        plt.plot(self.asset_memory, "r")
        plt.savefig(f"results/account_value_trade_{self.episode}.png")
        plt.close()

    # ---------- Memory saving methods ----------

    def save_state_memory(self):
        """Save state memory as DataFrame."""
        if len(self.df.tic.unique()) > 1:
            date_list = self.date_memory[:-1]
            state_list = [
                x[:ACCOUNT_FIELDS + 2 * self.stock_dim]
                for x in self.state_memory
            ]

            columns = (
                ["cash", "loan", "long_equity"]
                + ["limit", "credit", "short_equity"]
                + [f"{t}_c" for t in self.data.tic.values]
                + [f"{t}_h" for t in self.data.tic.values]
            )

            df_states = pd.DataFrame(state_list, columns=columns)
            df_states.index = pd.to_datetime(date_list)
            return df_states
        else:
            return pd.DataFrame({
                "date": self.date_memory[:-1],
                "states": self.state_memory
            })

    def save_asset_memory(self):
        """Save asset memory as DataFrame."""
        return pd.DataFrame({
            "date": self.date_memory,
            "account_value": self.asset_memory
        })

    def save_action_memory(self):
        """Save action memory as DataFrame."""
        date_list = self.date_memory[:-1]
        df_actions = pd.DataFrame(self.actions_memory)
        df_actions.columns = (
            [f"{t}_l" for t in self.data.tic.values]
            + [f"{t}_s" for t in self.data.tic.values]
        )
        df_actions.index = pd.to_datetime(date_list)
        return df_actions

    # ---------- Utility methods ----------

    def _get_date(self):
        """Get current date."""
        return self.dates[self.day]

    def _seed(self, seed=None):
        """Set random seed."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
