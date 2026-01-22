"""
Broker Interface and Paper Trading Implementation

This module provides an abstract broker interface for order execution and
a PaperBroker implementation for testing and paper trading.

Feature 10: Order Execution Engine (R10.1.1 - R10.1.2)
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum


class OrderSide(IntEnum):
    """Order side enumeration."""

    BUY = 1
    SELL = -1


class OrderType(IntEnum):
    """Order type enumeration."""

    MARKET = 0
    LIMIT = 1
    STOP = 2
    STOP_LIMIT = 3


class OrderStatus(IntEnum):
    """Order status enumeration."""

    PENDING = 0
    FILLED = 1
    PARTIALLY_FILLED = 2
    CANCELLED = 3
    REJECTED = 4


@dataclass
class Order:
    """
    Represents an order in the system.

    Attributes:
        order_id: Unique order identifier
        symbol: Trading symbol
        qty: Quantity of shares
        side: BUY or SELL
        order_type: MARKET, LIMIT, STOP, or STOP_LIMIT
        limit_price: Limit price (required for LIMIT and STOP_LIMIT orders)
        stop_price: Stop price (required for STOP and STOP_LIMIT orders)
        status: Current order status
        filled_qty: Number of shares filled
        filled_price: Average fill price
        created_at: Order creation timestamp
        updated_at: Last update timestamp
    """

    order_id: str
    symbol: str
    qty: int
    side: OrderSide
    order_type: OrderType
    limit_price: float | None = None
    stop_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: int = 0
    filled_price: float | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class Position:
    """
    Represents an open position.

    Attributes:
        symbol: Trading symbol
        qty: Number of shares (positive for long, negative for short)
        entry_price: Average entry price
        current_price: Current market price
        unrealized_pnl: Unrealized profit/loss
        market_value: Current market value of position
    """

    symbol: str
    qty: int
    entry_price: float
    current_price: float
    unrealized_pnl: float = 0.0
    market_value: float = 0.0

    def __post_init__(self):
        """Calculate derived fields after initialization."""
        self.market_value = self.qty * self.current_price
        self.unrealized_pnl = (self.current_price - self.entry_price) * self.qty


@dataclass
class Account:
    """
    Represents account information.

    Attributes:
        account_id: Unique account identifier
        balance: Total account balance
        buying_power: Available buying power
        equity: Total equity (balance + unrealized P&L)
        cash: Available cash
        positions_value: Total value of open positions
    """

    account_id: str
    balance: float
    buying_power: float
    equity: float
    cash: float
    positions_value: float = 0.0


class BrokerInterface(ABC):
    """
    Abstract broker interface defining methods all brokers must implement.

    This interface provides a broker-agnostic API for order execution,
    position management, and account queries.

    Requirements:
        - R10.1.1: Abstract methods for order and account management
        - R10.1.2: Concrete implementations (PaperBroker, AlpacaBroker, etc.)
    """

    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        qty: int,
        side: OrderSide,
        order_type: OrderType,
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> Order:
        """
        Place an order with the broker.

        Args:
            symbol: Trading symbol (e.g., 'SPY', 'AAPL')
            qty: Number of shares
            side: BUY or SELL
            order_type: MARKET, LIMIT, STOP, or STOP_LIMIT
            limit_price: Limit price (required for LIMIT and STOP_LIMIT)
            stop_price: Stop price (required for STOP and STOP_LIMIT)

        Returns:
            Order object with order details and status

        Raises:
            ValueError: If required prices not provided for order type
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.

        Args:
            order_id: Unique order identifier

        Returns:
            True if cancellation successful, False otherwise
        """
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str) -> Order | None:
        """
        Get the current status of an order.

        Args:
            order_id: Unique order identifier

        Returns:
            Order object with current status, or None if order not found
        """
        pass

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """
        Get all open positions.

        Returns:
            List of Position objects representing open positions
        """
        pass

    @abstractmethod
    async def get_account(self) -> Account:
        """
        Get account information.

        Returns:
            Account object with balance, buying power, equity, etc.
        """
        pass


class PaperBroker(BrokerInterface):
    """
    Paper trading broker implementation for testing.

    Simulates order fills with realistic behavior:
    - Market orders fill immediately at current price
    - Limit orders fill when market price crosses limit
    - Stop orders trigger when market price crosses stop
    - Tracks positions and account balance

    Requirements:
        - R10.1.2: PaperBroker for testing (simulated fills)
    """

    def __init__(self, initial_capital: float = 100000.0):
        """
        Initialize paper broker.

        Args:
            initial_capital: Starting account balance (default: $100,000)
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.equity = initial_capital
        self.orders: dict[str, Order] = {}
        self.positions: dict[str, Position] = {}
        self.order_counter = 0
        self.market_prices: dict[str, float] = {}

    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self.order_counter += 1
        return f"PAPER_{self.order_counter:06d}"

    def set_market_price(self, symbol: str, price: float):
        """
        Set the current market price for a symbol (for simulation).

        Args:
            symbol: Trading symbol
            price: Current market price
        """
        self.market_prices[symbol] = price

        # Update existing position prices
        if symbol in self.positions:
            position = self.positions[symbol]
            position.current_price = price
            position.market_value = position.qty * price
            position.unrealized_pnl = (price - position.entry_price) * position.qty

        # Check pending limit/stop orders
        self._check_pending_orders(symbol, price)

    def _check_pending_orders(self, symbol: str, price: float):
        """
        Check if any pending limit/stop orders should be filled.

        Args:
            symbol: Trading symbol
            price: Current market price
        """
        for order in list(self.orders.values()):
            if order.symbol != symbol or order.status != OrderStatus.PENDING:
                continue

            should_fill = False

            # Check limit orders
            if order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and price <= order.limit_price:
                    should_fill = True
                elif order.side == OrderSide.SELL and price >= order.limit_price:
                    should_fill = True

            # Check stop orders
            elif order.order_type == OrderType.STOP:
                if order.side == OrderSide.BUY and price >= order.stop_price:
                    should_fill = True
                elif order.side == OrderSide.SELL and price <= order.stop_price:
                    should_fill = True

            # Check stop-limit orders (trigger becomes limit)
            elif order.order_type == OrderType.STOP_LIMIT:
                if order.side == OrderSide.BUY and price >= order.stop_price:
                    if price <= order.limit_price:
                        should_fill = True
                elif order.side == OrderSide.SELL and price <= order.stop_price:
                    if price >= order.limit_price:
                        should_fill = True

            if should_fill:
                self._fill_order(order, price)

    def _fill_order(self, order: Order, fill_price: float):
        """
        Fill an order and update positions/cash.

        Args:
            order: Order to fill
            fill_price: Fill price
        """
        order.status = OrderStatus.FILLED
        order.filled_qty = order.qty
        order.filled_price = fill_price
        order.updated_at = time.time()

        # Update cash
        cost = order.qty * fill_price
        if order.side == OrderSide.BUY:
            self.cash -= cost
        else:
            self.cash += cost

        # Update positions
        symbol = order.symbol
        if symbol in self.positions:
            position = self.positions[symbol]

            if order.side == OrderSide.BUY:
                # Add to existing position
                total_cost = (position.qty * position.entry_price) + (order.qty * fill_price)
                total_qty = position.qty + order.qty
                position.entry_price = total_cost / total_qty if total_qty != 0 else 0.0
                position.qty = total_qty
            else:
                # Reduce or close position
                position.qty -= order.qty

                if position.qty == 0:
                    # Position closed
                    del self.positions[symbol]
                elif position.qty < 0:
                    # Reversed to short (or increased short)
                    position.entry_price = fill_price
        else:
            # New position
            qty = order.qty if order.side == OrderSide.BUY else -order.qty
            self.positions[symbol] = Position(
                symbol=symbol,
                qty=qty,
                entry_price=fill_price,
                current_price=fill_price,
            )

        # Update equity
        self._update_equity()

    def _update_equity(self):
        """Update total equity based on cash and positions."""
        positions_value = sum(abs(p.qty) * p.current_price for p in self.positions.values())
        unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        self.equity = self.cash + positions_value + unrealized_pnl

    async def place_order(
        self,
        symbol: str,
        qty: int,
        side: OrderSide,
        order_type: OrderType,
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> Order:
        """
        Place an order with the paper broker.

        Market orders fill immediately, limit/stop orders remain pending.
        """
        # Validate prices for order types
        if order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and limit_price is None:
            raise ValueError(f"limit_price required for {order_type.name} orders")

        if order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and stop_price is None:
            raise ValueError(f"stop_price required for {order_type.name} orders")

        # Create order
        order = Order(
            order_id=self._generate_order_id(),
            symbol=symbol,
            qty=qty,
            side=side,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
        )

        self.orders[order.order_id] = order

        # Fill market orders immediately
        if order_type == OrderType.MARKET:
            if symbol not in self.market_prices:
                order.status = OrderStatus.REJECTED
                raise ValueError(f"Market price not set for {symbol}")

            fill_price = self.market_prices[symbol]

            # Check sufficient capital
            cost = qty * fill_price
            if side == OrderSide.BUY and cost > self.cash:
                order.status = OrderStatus.REJECTED
                raise ValueError(f"Insufficient capital: need ${cost:.2f}, have ${self.cash:.2f}")

            self._fill_order(order, fill_price)

        return order

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]
        if order.status != OrderStatus.PENDING:
            return False

        order.status = OrderStatus.CANCELLED
        order.updated_at = time.time()
        return True

    async def get_order_status(self, order_id: str) -> Order | None:
        """Get order status."""
        return self.orders.get(order_id)

    async def get_positions(self) -> list[Position]:
        """Get all open positions."""
        return list(self.positions.values())

    async def get_account(self) -> Account:
        """Get account information."""
        self._update_equity()

        positions_value = sum(abs(p.qty) * p.current_price for p in self.positions.values())

        return Account(
            account_id="PAPER_001",
            balance=self.initial_capital,
            buying_power=self.cash,
            equity=self.equity,
            cash=self.cash,
            positions_value=positions_value,
        )
