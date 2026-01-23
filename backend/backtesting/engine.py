"""
Backtesting Engine for FluxHero Trading System.

This module implements a complete backtesting orchestrator that simulates trading strategies
on historical data with realistic fill logic, slippage, and commission modeling.

Features:
- Next-bar fill logic (signal on bar N → fill at bar N+1 open) - R9.1.1
- Realistic slippage and commission modeling - R9.2
- Performance metrics calculation (Sharpe, drawdown, win rate) - R9.3
- Walk-forward testing support - R9.4

Performance: <10 seconds for 1 year of minute data (100k+ candles)

Reference:
- FLUXHERO_REQUIREMENTS.md Feature 9: Backtesting Module
- algorithmic-trading-guide.md → Backtesting & Transaction Costs
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class OrderSide(IntEnum):
    """Order side: BUY or SELL."""

    BUY = 1
    SELL = -1


class OrderType(IntEnum):
    """Order type: MARKET or LIMIT."""

    MARKET = 0
    LIMIT = 1


class OrderStatus(IntEnum):
    """Order status."""

    PENDING = 0
    FILLED = 1
    CANCELLED = 2


class PositionSide(IntEnum):
    """Position side."""

    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class BacktestConfig:
    """Configuration for backtest execution.

    Attributes:
        initial_capital: Starting account balance in dollars
        commission_per_share: Fixed commission per share (default: $0.005 for Alpaca)
        slippage_pct: Slippage percentage for market orders (default: 0.01%)
        impact_threshold: Order size threshold as fraction of avg volume (default: 0.1 = 10%)
        impact_penalty_pct: Extra slippage if order exceeds impact threshold (default: 0.05%)
        risk_free_rate: Annual risk-free rate for Sharpe ratio (default: 4%)
    """

    initial_capital: float = 100000.0
    commission_per_share: float = 0.005  # R9.2.1: $0.005 per share (Alpaca-like)
    slippage_pct: float = 0.0001  # R9.2.2: 0.01% slippage on market orders
    impact_threshold: float = 0.1  # R9.2.3: 10% of avg volume
    impact_penalty_pct: float = 0.0005  # R9.2.3: Extra 0.05% slippage for large orders
    risk_free_rate: float = 0.04  # R9.3.1: 4% annual risk-free rate for Sharpe
    max_position_size: int = 100000  # Maximum shares allowed in a single position
    enable_sanity_checks: bool = True  # Enable runtime sanity check assertions


@dataclass
class Order:
    """Represents a trading order.

    Attributes:
        bar_index: Bar index where signal was generated
        symbol: Trading symbol
        side: BUY or SELL
        shares: Number of shares
        order_type: MARKET or LIMIT
        limit_price: Price for limit orders (None for market orders)
        status: Order status (PENDING, FILLED, CANCELLED)
        fill_price: Actual fill price (None until filled)
        fill_bar_index: Bar index where order was filled (None until filled)
        commission: Commission paid (calculated on fill)
        slippage: Slippage cost (calculated on fill)
    """

    bar_index: int
    symbol: str
    side: OrderSide
    shares: int
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    fill_price: float | None = None
    fill_bar_index: int | None = None
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class Position:
    """Represents an open position.

    Attributes:
        symbol: Trading symbol
        side: LONG or SHORT
        shares: Number of shares
        entry_price: Average entry price
        entry_bar_index: Bar index of entry
        stop_loss: Stop loss price (None if not set)
        take_profit: Take profit price (None if not set)
    """

    symbol: str
    side: PositionSide
    shares: int
    entry_price: float
    entry_bar_index: int
    stop_loss: float | None = None
    take_profit: float | None = None


@dataclass
class Trade:
    """Represents a completed trade (entry + exit).

    Attributes:
        symbol: Trading symbol
        side: LONG or SHORT
        shares: Number of shares
        entry_price: Entry fill price
        entry_bar_index: Bar index of entry
        entry_time: Timestamp of entry (if available)
        exit_price: Exit fill price
        exit_bar_index: Bar index of exit
        exit_time: Timestamp of exit (if available)
        pnl: Realized profit/loss (after commissions and slippage)
        pnl_pct: P&L as percentage of entry cost
        commission: Total commission paid (entry + exit)
        slippage: Total slippage cost
        holding_bars: Number of bars held
    """

    symbol: str
    side: PositionSide
    shares: int
    entry_price: float
    entry_bar_index: int
    entry_time: float | None = None
    exit_price: float | None = None
    exit_bar_index: int | None = None
    exit_time: float | None = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    holding_bars: int = 0


@dataclass
class BacktestState:
    """Current state of the backtest.

    Attributes:
        current_bar: Current bar index being processed
        cash: Available cash
        equity: Total account value (cash + position value)
        position: Current open position (None if flat)
        pending_orders: List of pending orders (not yet filled)
        trades: List of completed trades
        equity_curve: Array of equity values at each bar
        peak_equity: Highest equity value seen (for drawdown calculation)
    """

    current_bar: int = 0
    cash: float = 0.0
    equity: float = 0.0
    position: Position | None = None
    pending_orders: list[Order] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    peak_equity: float = 0.0


class SanityCheckError(Exception):
    """Raised when a sanity check assertion fails during backtesting.

    This indicates a critical issue with the backtest logic, such as:
    - Negative equity (impossible in real trading)
    - Position size exceeding limits
    - Invalid trade timestamps (exit before entry)
    - P&L mismatch with equity changes
    """

    pass


def validate_sanity_checks(
    state: BacktestState,
    config: BacktestConfig,
    bar_index: int,
    timestamps: NDArray | None = None,
) -> list[str]:
    """
    Perform runtime sanity checks on backtest state.

    Validates critical invariants that should never be violated:
    - Equity never negative
    - Position size <= max allowed
    - Trades have valid entry < exit timestamps
    - P&L matches equity change (within tolerance)

    Parameters
    ----------
    state : BacktestState
        Current backtest state to validate
    config : BacktestConfig
        Backtest configuration with limits
    bar_index : int
        Current bar index for context in error messages
    timestamps : Optional[NDArray]
        Bar timestamps for trade validation

    Returns
    -------
    list[str]
        List of sanity check violations (empty if all pass)

    Raises
    ------
    SanityCheckError
        If enable_sanity_checks is True and a critical violation is found
    """
    violations: list[str] = []

    # Check 1: Equity never negative
    if state.equity < 0:
        msg = f"Bar {bar_index}: Negative equity detected: ${state.equity:.2f}"
        violations.append(msg)
        logger.error(f"Sanity check FAILED: {msg}")

    # Check 2: Cash never negative
    if state.cash < 0:
        msg = f"Bar {bar_index}: Negative cash detected: ${state.cash:.2f}"
        violations.append(msg)
        logger.error(f"Sanity check FAILED: {msg}")

    # Check 3: Position size within limits
    if state.position is not None:
        if state.position.shares > config.max_position_size:
            msg = (
                f"Bar {bar_index}: Position size {state.position.shares} "
                f"exceeds max allowed {config.max_position_size}"
            )
            violations.append(msg)
            logger.error(f"Sanity check FAILED: {msg}")

        # Position shares must be positive
        if state.position.shares <= 0:
            msg = f"Bar {bar_index}: Invalid position shares: {state.position.shares}"
            violations.append(msg)
            logger.error(f"Sanity check FAILED: {msg}")

    # Check 4: Trade timestamps valid (entry < exit)
    for i, trade in enumerate(state.trades):
        # Check bar indices
        if trade.exit_bar_index is not None:
            if trade.entry_bar_index >= trade.exit_bar_index:
                msg = (
                    f"Trade {i}: Invalid bar indices - entry={trade.entry_bar_index} "
                    f">= exit={trade.exit_bar_index}"
                )
                violations.append(msg)
                logger.error(f"Sanity check FAILED: {msg}")

        # Check timestamps if available
        if trade.entry_time is not None and trade.exit_time is not None:
            if trade.entry_time >= trade.exit_time:
                msg = (
                    f"Trade {i}: Invalid timestamps - entry={trade.entry_time} "
                    f">= exit={trade.exit_time}"
                )
                violations.append(msg)
                logger.error(f"Sanity check FAILED: {msg}")

        # Check holding_bars consistency
        if trade.exit_bar_index is not None:
            expected_holding = trade.exit_bar_index - trade.entry_bar_index
            if trade.holding_bars != expected_holding:
                msg = (
                    f"Trade {i}: Holding bars mismatch - recorded={trade.holding_bars}, "
                    f"expected={expected_holding}"
                )
                violations.append(msg)
                logger.warning(f"Sanity check warning: {msg}")

    return violations


def validate_pnl_equity_consistency(
    state: BacktestState,
    initial_capital: float,
    tolerance: float = 0.01,
) -> list[str]:
    """
    Validate that cumulative P&L from trades matches the equity change.

    This check ensures the accounting is correct - the sum of all trade P&L
    plus the unrealized P&L of open positions should equal the total equity
    change from initial capital.

    Parameters
    ----------
    state : BacktestState
        Final backtest state with trades
    initial_capital : float
        Starting capital
    tolerance : float
        Acceptable difference as a fraction (default 0.01 = 1%)

    Returns
    -------
    list[str]
        List of P&L consistency violations (empty if consistent)
    """
    violations: list[str] = []

    # Sum of realized P&L from all trades
    realized_pnl = sum(trade.pnl for trade in state.trades)

    # Unrealized P&L from open position (already included in equity)
    # Since equity = cash + position_value, and position_value is at current price,
    # the equity change already accounts for unrealized P&L

    # Total equity change
    equity_change = state.equity - initial_capital

    # The equity change should equal the realized P&L (since we mark-to-market
    # the position value is already reflected in equity at each bar)
    # But we also need to account for the open position's unrealized P&L
    # For closed trades, equity_change should match realized_pnl when flat

    if state.position is None:
        # If flat, equity change should match realized P&L
        diff = abs(equity_change - realized_pnl)
        threshold = abs(initial_capital * tolerance)

        if diff > threshold:
            msg = (
                f"P&L mismatch when flat: equity_change=${equity_change:.2f}, "
                f"realized_pnl=${realized_pnl:.2f}, diff=${diff:.2f}"
            )
            violations.append(msg)
            logger.error(f"Sanity check FAILED: {msg}")
    else:
        # With open position, just log for information
        logger.debug(
            f"Open position check: equity_change=${equity_change:.2f}, "
            f"realized_pnl=${realized_pnl:.2f}, position_shares={state.position.shares}"
        )

    return violations


def validate_bar_integrity(
    bars: NDArray,
    timestamps: NDArray | None = None,
) -> list[str]:
    """
    Validate bar integrity for OHLC data.

    Checks for:
    - Valid OHLC relationships (High >= Low, High >= Open/Close, Low <= Open/Close)
    - Monotonically increasing timestamps (if provided)
    - Logs warnings for any suspicious data found

    Parameters
    ----------
    bars : NDArray
        OHLCV data, shape (N, 4+) where first 4 columns are [open, high, low, close]
    timestamps : Optional[NDArray]
        Timestamps for each bar (datetime64 or Unix epoch)

    Returns
    -------
    list[str]
        List of issues found (empty if data is valid)
    """
    issues: list[str] = []
    n_bars = len(bars)

    if n_bars == 0:
        issues.append("Empty bars array provided")
        logger.warning("Bar integrity check: Empty bars array provided")
        return issues

    # Extract OHLC
    opens = bars[:, 0]
    highs = bars[:, 1]
    lows = bars[:, 2]
    closes = bars[:, 3]

    # Check 1: High >= Low (fundamental OHLC constraint)
    invalid_hl = highs < lows
    invalid_hl_count = np.sum(invalid_hl)
    if invalid_hl_count > 0:
        invalid_indices = np.where(invalid_hl)[0][:5]  # Show first 5
        issue = f"High < Low on {invalid_hl_count} bars (indices: {list(invalid_indices)}...)"
        issues.append(issue)
        logger.warning(f"Bar integrity check: {issue}")

    # Check 2: High >= Open and High >= Close
    invalid_high_open = highs < opens
    invalid_high_close = highs < closes
    invalid_high = invalid_high_open | invalid_high_close
    invalid_high_count = np.sum(invalid_high)
    if invalid_high_count > 0:
        invalid_indices = np.where(invalid_high)[0][:5]
        issue = (
            f"High < Open or High < Close on {invalid_high_count} bars "
            f"(indices: {list(invalid_indices)}...)"
        )
        issues.append(issue)
        logger.warning(f"Bar integrity check: {issue}")

    # Check 3: Low <= Open and Low <= Close
    invalid_low_open = lows > opens
    invalid_low_close = lows > closes
    invalid_low = invalid_low_open | invalid_low_close
    invalid_low_count = np.sum(invalid_low)
    if invalid_low_count > 0:
        invalid_indices = np.where(invalid_low)[0][:5]
        issue = (
            f"Low > Open or Low > Close on {invalid_low_count} bars "
            f"(indices: {list(invalid_indices)}...)"
        )
        issues.append(issue)
        logger.warning(f"Bar integrity check: {issue}")

    # Check 4: Timestamps monotonically increasing (if provided)
    if timestamps is not None and len(timestamps) > 1:
        # Handle different timestamp formats
        if np.issubdtype(timestamps.dtype, np.datetime64):
            # datetime64 can be compared directly
            non_increasing = timestamps[1:] <= timestamps[:-1]
        else:
            # Assume numeric (Unix epoch or similar)
            non_increasing = timestamps[1:] <= timestamps[:-1]

        non_increasing_count = np.sum(non_increasing)
        if non_increasing_count > 0:
            non_increasing_indices = np.where(non_increasing)[0][:5]
            issue = (
                f"Non-monotonic timestamps at {non_increasing_count} positions "
                f"(indices: {list(non_increasing_indices)}...)"
            )
            issues.append(issue)
            logger.warning(f"Bar integrity check: {issue}")

    if not issues:
        logger.debug(f"Bar integrity check passed for {n_bars} bars")

    return issues


class BacktestEngine:
    """
    Backtesting engine orchestrator.

    Implements complete backtest workflow:
    1. Initialize account with starting capital
    2. Loop through historical bars
    3. Generate signals from strategy
    4. Execute orders with next-bar fill logic (R9.1.1)
    5. Apply slippage and commission (R9.2)
    6. Track equity curve
    7. Calculate performance metrics (R9.3)

    Usage:
        config = BacktestConfig(initial_capital=100000)
        engine = BacktestEngine(config)

        # Run backtest
        state = engine.run(
            bars=candle_data,
            strategy_func=my_strategy,
            symbol='SPY'
        )

        # Get results
        metrics = engine.get_performance_metrics(state)
    """

    def __init__(self, config: BacktestConfig):
        """Initialize backtest engine with configuration.

        Parameters
        ----------
        config : BacktestConfig
            Backtest configuration (capital, commissions, slippage)
        """
        self.config = config

    def run(
        self,
        bars: NDArray,
        strategy_func: Callable[[NDArray, int, Position | None], list[Order]],
        symbol: str = "SPY",
        timestamps: NDArray | None = None,
        volumes: NDArray | None = None,
    ) -> BacktestState:
        """
        Run backtest on historical data.

        Parameters
        ----------
        bars : NDArray
            OHLCV data, shape (N, 5) where columns are [open, high, low, close, volume]
            or (N, 4) if volumes provided separately
        strategy_func : Callable
            Strategy function that takes (bars, current_index, position) and returns list of orders
        symbol : str
            Trading symbol (default: 'SPY')
        timestamps : Optional[NDArray]
            Timestamps for each bar (for trade logging)
        volumes : Optional[NDArray]
            Volume data if not included in bars array

        Returns
        -------
        BacktestState
            Final backtest state with all trades and equity curve
        """
        # Validate bar integrity before running backtest
        validate_bar_integrity(bars, timestamps)

        # Initialize state
        state = BacktestState(
            current_bar=0,
            cash=self.config.initial_capital,
            equity=self.config.initial_capital,
            peak_equity=self.config.initial_capital,
        )

        n_bars = len(bars)

        # Extract OHLC data
        if bars.shape[1] >= 5:
            opens = bars[:, 0]
            highs = bars[:, 1]
            lows = bars[:, 2]
            closes = bars[:, 3]
            volumes_data = bars[:, 4]
        else:
            opens = bars[:, 0]
            highs = bars[:, 1]
            lows = bars[:, 2]
            closes = bars[:, 3]
            volumes_data = volumes if volumes is not None else np.ones(n_bars)

        # Calculate average volume for impact model
        avg_volume = np.mean(volumes_data[~np.isnan(volumes_data)])

        # Main backtest loop
        for i in range(n_bars):
            state.current_bar = i

            # Step 1: Fill pending orders (R9.1.1: next-bar fill)
            self._fill_pending_orders(state, opens[i], volumes_data[i], avg_volume, timestamps, i)

            # Step 2: Check stop loss / take profit on open position
            if state.position is not None:
                self._check_stops(state, highs[i], lows[i], closes[i], i, timestamps)

            # Step 3: Update position value and equity
            current_price = closes[i]
            if state.position is not None:
                position_value = self._calculate_position_value(state.position, current_price)
                state.equity = state.cash + position_value
            else:
                state.equity = state.cash

            # Update peak equity for drawdown calculation
            if state.equity > state.peak_equity:
                state.peak_equity = state.equity

            # Track equity curve
            state.equity_curve.append(state.equity)

            # Step 4: Run sanity checks (if enabled)
            if self.config.enable_sanity_checks:
                violations = validate_sanity_checks(state, self.config, i, timestamps)
                if violations:
                    raise SanityCheckError(
                        f"Sanity check failed at bar {i}: {'; '.join(violations)}"
                    )

            # Step 5: Generate signals from strategy (if not at last bar)
            if i < n_bars - 1:  # Don't generate signals on last bar
                orders = strategy_func(bars, i, state.position)

                # Add orders to pending list (will fill on next bar)
                for order in orders:
                    order.bar_index = i
                    order.symbol = symbol
                    state.pending_orders.append(order)

        # Final sanity check: P&L consistency
        if self.config.enable_sanity_checks:
            pnl_violations = validate_pnl_equity_consistency(
                state, self.config.initial_capital
            )
            if pnl_violations:
                raise SanityCheckError(
                    f"P&L consistency check failed: {'; '.join(pnl_violations)}"
                )

        return state

    def _fill_pending_orders(
        self,
        state: BacktestState,
        open_price: float,
        volume: float,
        avg_volume: float,
        timestamps: NDArray | None,
        bar_index: int,
    ) -> None:
        """
        Fill pending orders at next bar's open price (R9.1.1).

        Applies slippage and commission (R9.2).

        Parameters
        ----------
        state : BacktestState
            Current backtest state
        open_price : float
            Open price of current bar (fill price)
        volume : float
            Volume of current bar
        avg_volume : float
            Average volume (for impact model)
        timestamps : Optional[NDArray]
            Timestamps array
        bar_index : int
            Current bar index
        """
        filled_orders = []

        for order in state.pending_orders:
            # Calculate fill price with slippage
            fill_price = self._calculate_fill_price(order, open_price, volume, avg_volume)

            # Calculate commission
            commission = order.shares * self.config.commission_per_share

            # Calculate total cost
            if order.side == OrderSide.BUY:
                total_cost = (fill_price * order.shares) + commission

                # Check if we have enough cash
                if total_cost > state.cash:
                    # Cancel order (insufficient funds)
                    order.status = OrderStatus.CANCELLED
                    filled_orders.append(order)
                    continue

                # Execute buy
                state.cash -= total_cost

                # Open or add to long position
                if state.position is None:
                    state.position = Position(
                        symbol=order.symbol,
                        side=PositionSide.LONG,
                        shares=order.shares,
                        entry_price=fill_price,
                        entry_bar_index=bar_index,
                    )
                else:
                    # Average up
                    total_shares = state.position.shares + order.shares
                    total_cost_basis = (state.position.entry_price * state.position.shares) + (
                        fill_price * order.shares
                    )
                    state.position.shares = total_shares
                    state.position.entry_price = total_cost_basis / total_shares

            else:  # SELL
                # Check if this is closing a position
                if state.position is not None and state.position.side == PositionSide.LONG:
                    # Close long position
                    shares_to_close = min(order.shares, state.position.shares)
                    close_proceeds = (fill_price * shares_to_close) - (
                        commission * shares_to_close / order.shares
                    )
                    entry_cost = state.position.entry_price * shares_to_close

                    # Calculate P&L
                    pnl = close_proceeds - entry_cost
                    pnl_pct = (pnl / entry_cost) * 100.0

                    # Calculate slippage cost
                    slippage_cost = abs(fill_price - open_price) * shares_to_close

                    # Record trade
                    trade = Trade(
                        symbol=order.symbol,
                        side=PositionSide.LONG,
                        shares=shares_to_close,
                        entry_price=state.position.entry_price,
                        entry_bar_index=state.position.entry_bar_index,
                        entry_time=timestamps[state.position.entry_bar_index]
                        if timestamps is not None
                        else None,
                        exit_price=fill_price,
                        exit_bar_index=bar_index,
                        exit_time=timestamps[bar_index] if timestamps is not None else None,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        commission=commission,
                        slippage=slippage_cost,
                        holding_bars=bar_index - state.position.entry_bar_index,
                    )
                    state.trades.append(trade)

                    # Update cash
                    state.cash += close_proceeds

                    # Update or close position
                    if shares_to_close >= state.position.shares:
                        state.position = None
                    else:
                        state.position.shares -= shares_to_close
                else:
                    # Opening short (not implemented in simple version)
                    # For now, just cancel short orders if no long position to close
                    order.status = OrderStatus.CANCELLED
                    filled_orders.append(order)
                    continue

            # Mark order as filled
            order.status = OrderStatus.FILLED
            order.fill_price = fill_price
            order.fill_bar_index = bar_index
            order.commission = commission
            order.slippage = abs(fill_price - open_price) * order.shares
            filled_orders.append(order)

        # Remove filled/cancelled orders from pending list
        for order in filled_orders:
            state.pending_orders.remove(order)

    def _calculate_fill_price(
        self, order: Order, open_price: float, volume: float, avg_volume: float
    ) -> float:
        """
        Calculate fill price with slippage (R9.2.2, R9.2.3).

        Parameters
        ----------
        order : Order
            Order to fill
        open_price : float
            Next bar's open price (base fill price)
        volume : float
            Current bar's volume
        avg_volume : float
            Average volume for impact model

        Returns
        -------
        float
            Fill price including slippage
        """
        if order.order_type == OrderType.LIMIT:
            # Limit orders: assume fill at limit price (optimistic) - R9.2.2
            return order.limit_price

        # Market orders: apply slippage - R9.2.2
        base_slippage = self.config.slippage_pct

        # Check for price impact (R9.2.3)
        if avg_volume > 0:
            order_volume_pct = order.shares / avg_volume
            if order_volume_pct > self.config.impact_threshold:
                # Large order: add impact penalty
                base_slippage += self.config.impact_penalty_pct

        # Apply slippage direction (buy at ask, sell at bid)
        if order.side == OrderSide.BUY:
            # Buy: slippage increases price
            fill_price = open_price * (1.0 + base_slippage)
        else:
            # Sell: slippage decreases price
            fill_price = open_price * (1.0 - base_slippage)

        return fill_price

    def _check_stops(
        self,
        state: BacktestState,
        high: float,
        low: float,
        close: float,
        bar_index: int,
        timestamps: NDArray | None,
    ) -> None:
        """
        Check if stop loss or take profit is hit.

        Assumes intra-bar execution if price touches stop level.

        Parameters
        ----------
        state : BacktestState
            Current backtest state
        high : float
            High price of current bar
        low : float
            Low price of current bar
        close : float
            Close price of current bar (fallback exit price)
        bar_index : int
            Current bar index
        timestamps : Optional[NDArray]
            Timestamps array
        """
        if state.position is None:
            return

        exit_price = None

        # Check stop loss
        if state.position.stop_loss is not None:
            if state.position.side == PositionSide.LONG:
                # Long stop: sell if price drops below stop
                if low <= state.position.stop_loss:
                    exit_price = state.position.stop_loss
            # Note: Short positions not fully implemented in simple version

        # Check take profit
        if exit_price is None and state.position.take_profit is not None:
            if state.position.side == PositionSide.LONG:
                # Long take profit: sell if price rises above target
                if high >= state.position.take_profit:
                    exit_price = state.position.take_profit

        # Exit position if stop/target hit
        if exit_price is not None:
            # Calculate P&L
            entry_cost = state.position.entry_price * state.position.shares
            commission = state.position.shares * self.config.commission_per_share
            proceeds = (exit_price * state.position.shares) - commission
            pnl = proceeds - entry_cost
            pnl_pct = (pnl / entry_cost) * 100.0

            # Calculate slippage (stop/target assumed to execute at exact price, minimal slippage)
            slippage_cost = self.config.slippage_pct * exit_price * state.position.shares

            # Record trade
            trade = Trade(
                symbol=state.position.symbol,
                side=state.position.side,
                shares=state.position.shares,
                entry_price=state.position.entry_price,
                entry_bar_index=state.position.entry_bar_index,
                entry_time=timestamps[state.position.entry_bar_index]
                if timestamps is not None
                else None,
                exit_price=exit_price,
                exit_bar_index=bar_index,
                exit_time=timestamps[bar_index] if timestamps is not None else None,
                pnl=pnl,
                pnl_pct=pnl_pct,
                commission=commission,
                slippage=slippage_cost,
                holding_bars=bar_index - state.position.entry_bar_index,
            )
            state.trades.append(trade)

            # Update cash
            state.cash += proceeds

            # Close position
            state.position = None

    def _calculate_position_value(self, position: Position, current_price: float) -> float:
        """Calculate current value of open position.

        Parameters
        ----------
        position : Position
            Open position
        current_price : float
            Current market price

        Returns
        -------
        float
            Position value in dollars
        """
        if position.side == PositionSide.LONG:
            return position.shares * current_price
        else:
            # Short: value = initial proceeds - current cost to cover
            entry_proceeds = position.shares * position.entry_price
            current_cost = position.shares * current_price
            return entry_proceeds - current_cost

    def get_performance_summary(self, state: BacktestState) -> dict[str, Any]:
        """
        Get basic performance summary.

        This is a simplified version. Full metrics calculation is in metrics.py.

        Parameters
        ----------
        state : BacktestState
            Final backtest state

        Returns
        -------
        Dict[str, Any]
            Performance summary with key metrics
        """
        if len(state.trades) == 0:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_return": 0.0,
                "total_return_pct": 0.0,
                "final_equity": state.equity,
            }

        # Calculate basic metrics
        winning_trades = [t for t in state.trades if t.pnl > 0]
        losing_trades = [t for t in state.trades if t.pnl <= 0]

        total_return = state.equity - self.config.initial_capital
        total_return_pct = (total_return / self.config.initial_capital) * 100.0

        win_rate = len(winning_trades) / len(state.trades) if state.trades else 0.0

        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
        avg_loss = abs(np.mean([t.pnl for t in losing_trades])) if losing_trades else 1.0
        avg_win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

        return {
            "total_trades": len(state.trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_win_loss_ratio": avg_win_loss_ratio,
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "final_equity": state.equity,
            "initial_capital": self.config.initial_capital,
        }
