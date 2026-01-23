"""
Paper Broker Adapter - Simulated Trading for Testing and Development

This module implements a paper trading broker that simulates order execution
without real money. It uses SQLite for state persistence and provides realistic
order fills with configurable slippage.

Feature: Paper Trading System (Phase B)
"""

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger

from backend.execution.broker_base import (
    Account,
    BrokerHealth,
    BrokerInterface,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from backend.storage.sqlite_store import SQLiteStore

# Default paper account settings
DEFAULT_INITIAL_BALANCE = 100_000.0
PAPER_ACCOUNT_ID = "PAPER-001"


@dataclass
class PaperPosition:
    """Internal position tracking for paper broker."""

    symbol: str
    qty: int
    entry_price: float
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        """Calculate current market value."""
        return self.qty * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        return (self.current_price - self.entry_price) * self.qty

    @property
    def cost_basis(self) -> float:
        """Calculate total cost basis."""
        return self.qty * self.entry_price


@dataclass
class PaperTrade:
    """Record of a paper trade execution."""

    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    qty: int
    price: float
    slippage: float
    timestamp: float = field(default_factory=time.time)
    realized_pnl: float = 0.0


class PaperBroker(BrokerInterface):
    """
    Paper trading broker implementing BrokerInterface.

    Simulates order execution with realistic fills, slippage simulation,
    and persistent state storage via SQLite. Provides a safe environment
    for testing trading strategies without real money.

    Features:
    - Auto-create $100,000 paper account on first connection
    - Persistent state via SQLite (balance, positions, trades)
    - Configurable slippage simulation
    - Reset account to initial state
    - Realized and unrealized P&L tracking
    - Price fetching for realistic fills (optional)

    Attributes:
        initial_balance: Starting account balance (default: $100,000)
        db_path: Path to SQLite database for state persistence
        slippage_bps: Slippage in basis points (default: 5)
    """

    # Settings table keys for paper broker state
    SETTING_BALANCE = "paper_broker_balance"
    SETTING_REALIZED_PNL = "paper_broker_realized_pnl"
    SETTING_INITIALIZED = "paper_broker_initialized"

    def __init__(
        self,
        initial_balance: float = DEFAULT_INITIAL_BALANCE,
        db_path: str = "data/system.db",
        slippage_bps: float = 5.0,
        price_provider: Any = None,
    ):
        """
        Initialize Paper broker adapter.

        Args:
            initial_balance: Starting account balance (default: $100,000)
            db_path: Path to SQLite database for state persistence
            slippage_bps: Slippage in basis points (default: 5 bps = 0.05%)
            price_provider: Optional price provider for realistic fills
        """
        self.initial_balance = initial_balance
        self.db_path = db_path
        self.slippage_bps = slippage_bps
        self.price_provider = price_provider

        # State
        self._connected = False
        self._last_heartbeat: float | None = None
        self._store: SQLiteStore | None = None

        # In-memory state (loaded from SQLite on connect)
        self._balance: float = initial_balance
        self._realized_pnl: float = 0.0
        self._positions: dict[str, PaperPosition] = {}
        self._orders: dict[str, Order] = {}
        self._trades: list[PaperTrade] = []

        # Price cache with TTL (symbol -> (price, timestamp))
        self._price_cache: dict[str, tuple[float, float]] = {}
        self._price_cache_ttl: float = 60.0  # 1 minute TTL

    # -------------------------------------------------------------------------
    # Connection Lifecycle Methods
    # -------------------------------------------------------------------------

    async def connect(self) -> bool:
        """
        Initialize paper broker and load state from SQLite.

        Creates account with initial balance if not already initialized.

        Returns:
            True if connection successful
        """
        try:
            self._store = SQLiteStore(self.db_path)
            await self._store.initialize()

            # Check if paper broker has been initialized before
            initialized = await self._store.get_setting(self.SETTING_INITIALIZED)

            if initialized is None:
                # First time setup - initialize account
                await self._initialize_account()
                logger.info(f"Paper broker initialized with ${self.initial_balance:,.2f}")
            else:
                # Load existing state
                await self._load_state()
                logger.info(
                    f"Paper broker connected - balance: ${self._balance:,.2f}, "
                    f"positions: {len(self._positions)}"
                )

            self._connected = True
            self._last_heartbeat = time.time()
            return True

        except Exception as e:
            logger.error(f"Failed to connect paper broker: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """
        Save state and close connection.

        Persists current state to SQLite before disconnecting.
        """
        if self._connected and self._store:
            try:
                await self._save_state()
                await self._store.close()
            except Exception as e:
                logger.error(f"Error during paper broker disconnect: {e}")

        self._store = None
        self._connected = False
        self._last_heartbeat = None
        logger.info("Paper broker disconnected")

    async def health_check(self) -> BrokerHealth:
        """
        Check paper broker health.

        Returns:
            BrokerHealth object with connection status
        """
        if not self._connected or not self._store:
            return BrokerHealth(
                is_connected=False,
                is_authenticated=False,
                latency_ms=None,
                last_heartbeat=self._last_heartbeat,
                error_message="Not connected",
            )

        start_time = time.time()

        try:
            # Verify database is accessible
            await self._store.get_setting(self.SETTING_INITIALIZED)
            latency_ms = (time.time() - start_time) * 1000

            self._last_heartbeat = time.time()

            return BrokerHealth(
                is_connected=True,
                is_authenticated=True,
                latency_ms=latency_ms,
                last_heartbeat=self._last_heartbeat,
                error_message=None,
            )

        except Exception as e:
            return BrokerHealth(
                is_connected=False,
                is_authenticated=False,
                latency_ms=None,
                last_heartbeat=self._last_heartbeat,
                error_message=str(e),
            )

    # -------------------------------------------------------------------------
    # Account and Position Methods
    # -------------------------------------------------------------------------

    async def get_account(self) -> Account:
        """
        Get paper account information.

        Returns:
            Account object with balance, buying power, equity, etc.
        """
        self._ensure_connected()

        # Calculate positions value
        positions_value = sum(p.market_value for p in self._positions.values())

        # Equity = cash + positions value
        equity = self._balance + positions_value

        # Buying power = cash (simplified, no margin)
        buying_power = self._balance

        return Account(
            account_id=PAPER_ACCOUNT_ID,
            balance=self._balance,
            buying_power=buying_power,
            equity=equity,
            cash=self._balance,
            positions_value=positions_value,
        )

    async def get_positions(self) -> list[Position]:
        """
        Get all open paper positions.

        Returns:
            List of Position objects
        """
        self._ensure_connected()

        positions = []
        for symbol, paper_pos in self._positions.items():
            position = Position(
                symbol=symbol,
                qty=paper_pos.qty,
                entry_price=paper_pos.entry_price,
                current_price=paper_pos.current_price,
                unrealized_pnl=paper_pos.unrealized_pnl,
                market_value=paper_pos.market_value,
            )
            positions.append(position)

        return positions

    # -------------------------------------------------------------------------
    # Order Methods
    # -------------------------------------------------------------------------

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
        Place a paper order.

        For market orders, fills immediately at current price with slippage.
        For limit orders, validates price is reasonable then fills.

        Args:
            symbol: Trading symbol
            qty: Number of shares
            side: BUY or SELL
            order_type: MARKET, LIMIT, STOP, or STOP_LIMIT
            limit_price: Limit price (required for LIMIT and STOP_LIMIT)
            stop_price: Stop price (required for STOP and STOP_LIMIT)

        Returns:
            Order object with order details
        """
        self._ensure_connected()

        # Validate required prices
        if order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and limit_price is None:
            raise ValueError(f"limit_price required for {order_type.name} orders")

        if order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and stop_price is None:
            raise ValueError(f"stop_price required for {order_type.name} orders")

        # Generate order ID
        order_id = f"PAPER-{uuid.uuid4().hex[:12].upper()}"

        # Create pending order
        order = Order(
            order_id=order_id,
            symbol=symbol.upper(),
            qty=qty,
            side=side,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            status=OrderStatus.PENDING,
            filled_qty=0,
            filled_price=None,
            created_at=time.time(),
            updated_at=time.time(),
        )

        # Store order
        self._orders[order_id] = order

        # Execute order immediately (paper trading - instant fills)
        try:
            await self._execute_order(order)
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.updated_at = time.time()
            logger.error(f"Paper order rejected: {e}")

        # Save state after order execution
        await self._save_state()

        logger.info(
            f"Paper order {order.status.name}: {order_id} - "
            f"{symbol} {side.name} {qty} @ {order.filled_price or 'N/A'}"
        )

        return order

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a paper order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancellation successful, False otherwise
        """
        self._ensure_connected()

        if order_id not in self._orders:
            logger.warning(f"Paper order not found: {order_id}")
            return False

        order = self._orders[order_id]

        # Can only cancel pending orders
        if order.status != OrderStatus.PENDING:
            logger.warning(f"Cannot cancel paper order {order_id}: status is {order.status.name}")
            return False

        order.status = OrderStatus.CANCELLED
        order.updated_at = time.time()

        logger.info(f"Paper order cancelled: {order_id}")
        return True

    async def get_order_status(self, order_id: str) -> Order | None:
        """
        Get the current status of a paper order.

        Args:
            order_id: Order ID

        Returns:
            Order object or None if not found
        """
        self._ensure_connected()
        return self._orders.get(order_id)

    # -------------------------------------------------------------------------
    # Paper-Specific Methods
    # -------------------------------------------------------------------------

    async def reset_account(self) -> None:
        """
        Reset paper account to initial state.

        Clears all positions, orders, and trades, then restores
        the initial balance.
        """
        self._ensure_connected()

        logger.info(
            f"Resetting paper account - clearing {len(self._positions)} positions, "
            f"{len(self._trades)} trades"
        )

        # Clear all state
        self._balance = self.initial_balance
        self._realized_pnl = 0.0
        self._positions.clear()
        self._orders.clear()
        self._trades.clear()
        self._price_cache.clear()

        # Save cleared state
        await self._save_state()

        logger.info(f"Paper account reset complete - balance: ${self.initial_balance:,.2f}")

    async def get_trades(self) -> list[PaperTrade]:
        """
        Get paper trade history.

        Returns:
            List of PaperTrade objects
        """
        self._ensure_connected()
        return list(self._trades)

    async def get_realized_pnl(self) -> float:
        """
        Get total realized P&L.

        Returns:
            Realized P&L amount
        """
        self._ensure_connected()
        return self._realized_pnl

    async def get_unrealized_pnl(self) -> float:
        """
        Get total unrealized P&L.

        Returns:
            Unrealized P&L amount
        """
        self._ensure_connected()
        return sum(p.unrealized_pnl for p in self._positions.values())

    async def update_prices(self, prices: dict[str, float]) -> None:
        """
        Update current prices for positions.

        Args:
            prices: Dictionary of symbol -> price
        """
        self._ensure_connected()

        for symbol, price in prices.items():
            if symbol in self._positions:
                self._positions[symbol].current_price = price
                self._price_cache[symbol] = (price, time.time())

    async def set_price(self, symbol: str, price: float) -> None:
        """
        Set price for a specific symbol (useful for testing).

        Args:
            symbol: Trading symbol
            price: Price to set
        """
        self._price_cache[symbol.upper()] = (price, time.time())

        if symbol.upper() in self._positions:
            self._positions[symbol.upper()].current_price = price

    # -------------------------------------------------------------------------
    # Private Methods - State Management
    # -------------------------------------------------------------------------

    def _ensure_connected(self) -> None:
        """Raise error if not connected."""
        if not self._connected or not self._store:
            raise RuntimeError("Paper broker not connected. Call connect() first.")

    async def _initialize_account(self) -> None:
        """Initialize paper account with default values."""
        assert self._store is not None  # Caller ensures this

        self._balance = self.initial_balance
        self._realized_pnl = 0.0
        self._positions.clear()
        self._orders.clear()
        self._trades.clear()

        # Mark as initialized
        await self._store.set_setting(self.SETTING_INITIALIZED, "true")
        await self._save_state()

    async def _load_state(self) -> None:
        """Load paper broker state from SQLite."""
        assert self._store is not None  # Caller ensures this

        # Load balance
        balance_str = await self._store.get_setting(self.SETTING_BALANCE, str(self.initial_balance))
        self._balance = float(balance_str) if balance_str else self.initial_balance

        # Load realized P&L
        pnl_str = await self._store.get_setting(self.SETTING_REALIZED_PNL, "0.0")
        self._realized_pnl = float(pnl_str) if pnl_str else 0.0

        # Load positions from SQLite positions table
        stored_positions = await self._store.get_open_positions()
        self._positions.clear()

        for pos in stored_positions:
            # Check if this is a paper position (we can identify by entry_time format)
            if pos.symbol:
                self._positions[pos.symbol] = PaperPosition(
                    symbol=pos.symbol,
                    qty=pos.shares,
                    entry_price=pos.entry_price,
                    current_price=pos.current_price,
                )

    async def _save_state(self) -> None:
        """Save paper broker state to SQLite."""
        if not self._store:
            return

        # Save balance and P&L
        await self._store.set_setting(self.SETTING_BALANCE, str(self._balance))
        await self._store.set_setting(self.SETTING_REALIZED_PNL, str(self._realized_pnl))

        # Save positions to SQLite positions table
        # First, clear existing paper positions (we'll rewrite them)
        existing_positions = await self._store.get_open_positions()
        for pos in existing_positions:
            await self._store.delete_position(pos.symbol)

        # Save current positions
        from backend.storage.sqlite_store import Position as StorePosition

        for symbol, paper_pos in self._positions.items():
            store_pos = StorePosition(
                symbol=symbol,
                side=1,  # Long
                shares=paper_pos.qty,
                entry_price=paper_pos.entry_price,
                current_price=paper_pos.current_price,
                unrealized_pnl=paper_pos.unrealized_pnl,
                stop_loss=0.0,
                take_profit=None,
                entry_time=datetime.utcnow().isoformat(),
                updated_at=datetime.utcnow().isoformat(),
            )
            await self._store.upsert_position(store_pos)

    # -------------------------------------------------------------------------
    # Private Methods - Order Execution
    # -------------------------------------------------------------------------

    async def _execute_order(self, order: Order) -> None:
        """
        Execute a paper order.

        Args:
            order: Order to execute
        """
        # Get execution price
        base_price = await self._get_price(order.symbol, order.limit_price)

        if base_price is None:
            raise ValueError(f"Cannot determine price for {order.symbol}")

        # Apply slippage
        fill_price = self._apply_slippage(base_price, order.side, order.symbol, order.qty)

        # Calculate order value
        order_value = fill_price * order.qty

        if order.side == OrderSide.BUY:
            # Check if we have enough cash
            if order_value > self._balance:
                raise ValueError(
                    f"Insufficient funds: need ${order_value:,.2f}, have ${self._balance:,.2f}"
                )

            # Deduct cash
            self._balance -= order_value

            # Add or update position
            if order.symbol in self._positions:
                # Average into existing position
                existing = self._positions[order.symbol]
                total_qty = existing.qty + order.qty
                total_cost = existing.cost_basis + order_value
                avg_price = total_cost / total_qty

                existing.qty = total_qty
                existing.entry_price = avg_price
                existing.current_price = fill_price
            else:
                # New position
                self._positions[order.symbol] = PaperPosition(
                    symbol=order.symbol,
                    qty=order.qty,
                    entry_price=fill_price,
                    current_price=fill_price,
                )

        else:  # SELL
            # Check if we have the position
            if order.symbol not in self._positions:
                raise ValueError(f"No position to sell for {order.symbol}")

            position = self._positions[order.symbol]

            if order.qty > position.qty:
                raise ValueError(
                    f"Insufficient shares: have {position.qty}, trying to sell {order.qty}"
                )

            # Calculate realized P&L
            cost_basis = position.entry_price * order.qty
            proceeds = order_value
            realized_pnl = proceeds - cost_basis

            # Update state
            self._balance += order_value
            self._realized_pnl += realized_pnl

            # Update or remove position
            remaining_qty = position.qty - order.qty
            if remaining_qty == 0:
                del self._positions[order.symbol]
            else:
                position.qty = remaining_qty
                position.current_price = fill_price

            # Record the realized P&L
            order.filled_price = fill_price
            logger.info(f"Paper sell realized P&L: ${realized_pnl:,.2f} on {order.symbol}")

        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_qty = order.qty
        order.filled_price = fill_price
        order.updated_at = time.time()

        # Record trade
        slippage_amount = abs(fill_price - base_price)
        trade = PaperTrade(
            trade_id=f"TRADE-{uuid.uuid4().hex[:12].upper()}",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            price=fill_price,
            slippage=slippage_amount,
            realized_pnl=realized_pnl if order.side == OrderSide.SELL else 0.0,
        )
        self._trades.append(trade)

    def _apply_slippage(self, price: float, side: OrderSide, symbol: str, qty: int) -> float:
        """
        Apply slippage to execution price.

        For buys, price increases (worse fill).
        For sells, price decreases (worse fill).

        Slippage formula:
        - Buy: fill_price = price * (1 + slippage_bps / 10000)
        - Sell: fill_price = price * (1 - slippage_bps / 10000)

        Args:
            price: Base price
            side: Order side
            symbol: Trading symbol (for logging)
            qty: Order quantity (for logging)

        Returns:
            Price with slippage applied
        """
        slippage_factor = self.slippage_bps / 10000.0

        if side == OrderSide.BUY:
            fill_price = price * (1 + slippage_factor)
        else:
            fill_price = price * (1 - slippage_factor)

        # Calculate slippage impact
        slippage_per_share = abs(fill_price - price)
        slippage_total = slippage_per_share * qty
        slippage_pct = (slippage_per_share / price) * 100

        # Log slippage impact
        logger.debug(
            f"Slippage applied to {symbol} {side.name} {qty} shares: "
            f"base_price=${price:.4f}, fill_price=${fill_price:.4f}, "
            f"slippage={self.slippage_bps:.1f}bps (${slippage_per_share:.4f}/share, "
            f"${slippage_total:.2f} total, {slippage_pct:.3f}%)"
        )

        return fill_price

    async def _get_price(self, symbol: str, fallback_price: float | None = None) -> float | None:
        """
        Get current price for a symbol.

        Tries (in order):
        1. Cache (if within TTL)
        2. Price provider (if available)
        3. Fallback price

        Args:
            symbol: Trading symbol
            fallback_price: Fallback price if no other source available

        Returns:
            Current price or None
        """
        symbol = symbol.upper()

        # Check cache
        if symbol in self._price_cache:
            cached_price, cached_time = self._price_cache[symbol]
            if time.time() - cached_time < self._price_cache_ttl:
                return cached_price

        # Try price provider
        if self.price_provider is not None:
            try:
                price = await self._fetch_price_from_provider(symbol)
                if price is not None:
                    self._price_cache[symbol] = (price, time.time())
                    return price
            except Exception as e:
                logger.warning(f"Failed to fetch price for {symbol}: {e}")

        # Use fallback
        if fallback_price is not None:
            return fallback_price

        # Check if we have a position with a price
        if symbol in self._positions:
            return self._positions[symbol].current_price

        return None

    async def _fetch_price_from_provider(self, symbol: str) -> float | None:
        """
        Fetch price from the configured price provider.

        Args:
            symbol: Trading symbol

        Returns:
            Current price or None
        """
        if self.price_provider is None:
            return None

        try:
            # Try to use yahoo_provider's fetch_historical_data for last price
            from datetime import datetime, timedelta

            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

            data = await self.price_provider.fetch_historical_data(symbol, start_date, end_date)

            if data and len(data.bars) > 0:
                # Return the last close price (column index 3 is Close)
                return float(data.bars[-1][3])

        except Exception as e:
            logger.debug(f"Price provider fetch failed for {symbol}: {e}")

        return None
