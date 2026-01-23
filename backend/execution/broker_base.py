"""
Broker Base Class - Abstract Interface for All Broker Implementations

This module defines the abstract BrokerInterface that all broker adapters must implement.
It provides a broker-agnostic API for order execution, position management, account queries,
and connection lifecycle management.

Feature: Multi-Broker Architecture (Phase A)
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


@dataclass
class BrokerHealth:
    """
    Represents broker connection health status.

    Attributes:
        is_connected: Whether the broker connection is active
        is_authenticated: Whether authentication is valid
        latency_ms: Connection latency in milliseconds (None if not connected)
        last_heartbeat: Timestamp of last successful communication
        error_message: Error message if unhealthy (None if healthy)
    """

    is_connected: bool
    is_authenticated: bool
    latency_ms: float | None = None
    last_heartbeat: float | None = None
    error_message: str | None = None

    @property
    def is_healthy(self) -> bool:
        """Return True if broker is connected and authenticated."""
        return self.is_connected and self.is_authenticated


class BrokerInterface(ABC):
    """
    Abstract broker interface defining methods all brokers must implement.

    This interface provides a broker-agnostic API for:
    - Connection lifecycle management (connect, disconnect)
    - Order execution (place, cancel, status)
    - Position and account queries
    - Health monitoring

    All methods are async to support FastAPI's async patterns.

    Concrete implementations:
    - PaperBroker: Paper trading simulation
    - AlpacaBroker: Alpaca trading API
    """

    # -------------------------------------------------------------------------
    # Connection Lifecycle Methods
    # -------------------------------------------------------------------------

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the broker.

        This method should:
        - Validate credentials
        - Establish API connection
        - Set up any required sessions or websockets

        Returns:
            True if connection successful, False otherwise

        Raises:
            ConnectionError: If connection fails due to network issues
            AuthenticationError: If credentials are invalid
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Close connection to the broker.

        This method should:
        - Close any open sessions
        - Clean up resources
        - Cancel any pending heartbeats or background tasks

        Should be safe to call even if not connected.
        """
        pass

    @abstractmethod
    async def health_check(self) -> BrokerHealth:
        """
        Check the health of the broker connection.

        This method should:
        - Verify the connection is still active
        - Check authentication status
        - Measure connection latency if possible

        Returns:
            BrokerHealth object with connection status details
        """
        pass

    # -------------------------------------------------------------------------
    # Account and Position Methods
    # -------------------------------------------------------------------------

    @abstractmethod
    async def get_account(self) -> Account:
        """
        Get account information.

        Returns:
            Account object with balance, buying power, equity, etc.

        Raises:
            ConnectionError: If not connected to broker
        """
        pass

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """
        Get all open positions.

        Returns:
            List of Position objects representing open positions

        Raises:
            ConnectionError: If not connected to broker
        """
        pass

    # -------------------------------------------------------------------------
    # Order Methods
    # -------------------------------------------------------------------------

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
            ConnectionError: If not connected to broker
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

        Raises:
            ConnectionError: If not connected to broker
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

        Raises:
            ConnectionError: If not connected to broker
        """
        pass
