"""
Order Manager with Heartbeat Monitor and Chase Logic

This module manages order lifecycle from placement to fill confirmation,
implementing heartbeat monitoring and order chasing logic.

Feature 10: Order Execution Engine (R10.2.1 - R10.2.3)
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import logging

from .broker_interface import (
    BrokerInterface,
    Order,
    OrderStatus,
    OrderSide,
    OrderType,
)


logger = logging.getLogger(__name__)


@dataclass
class ManagedOrder:
    """
    Represents an order being managed by the OrderManager.

    Attributes:
        order: The broker order object
        placed_at: Timestamp when order was placed
        last_checked_at: Timestamp of last status check
        chase_count: Number of times order has been chased
        is_abandoned: Whether order has been abandoned after max chases
        target_symbol: Symbol being traded
        original_qty: Original order quantity
        original_side: Original order side
    """
    order: Order
    placed_at: float = field(default_factory=time.time)
    last_checked_at: float = field(default_factory=time.time)
    chase_count: int = 0
    is_abandoned: bool = False
    target_symbol: str = ""
    original_qty: int = 0
    original_side: OrderSide = OrderSide.BUY

    def __post_init__(self):
        """Initialize derived fields."""
        if not self.target_symbol:
            self.target_symbol = self.order.symbol
        if not self.original_qty:
            self.original_qty = self.order.qty
        if self.original_side == OrderSide.BUY and self.order.side != OrderSide.BUY:
            self.original_side = self.order.side


@dataclass
class ChaseConfig:
    """
    Configuration for order chasing behavior.

    Attributes:
        max_chase_attempts: Maximum number of chase attempts (R10.2.3)
        chase_after_seconds: Time to wait before chasing (R10.2.2: 60s)
        poll_interval_seconds: Status polling interval (R10.2.1: 5s)
    """
    max_chase_attempts: int = 3
    chase_after_seconds: float = 60.0
    poll_interval_seconds: float = 5.0


class OrderManager:
    """
    Manages order lifecycle with heartbeat monitoring and chase logic.

    Features:
    - Polls order status every 5 seconds (R10.2.1)
    - Automatically cancels and rechases unfilled orders after 60s (R10.2.2)
    - Max 3 chase attempts before abandoning (R10.2.3)
    - Recalculates mid-price on each chase attempt
    - Tracks all active orders

    Requirements:
        - R10.2.1: Poll order status every 5 seconds
        - R10.2.2: Cancel and rechase after 60 seconds if unfilled
        - R10.2.3: Maximum 3 chase attempts, then abandon

    Example:
        >>> broker = PaperBroker()
        >>> manager = OrderManager(broker)
        >>> await manager.start()
        >>> order = await manager.place_order_with_monitoring("SPY", 100, OrderSide.BUY, OrderType.LIMIT, limit_price=450.0)
        >>> # Manager will monitor and chase if needed
        >>> await manager.stop()
    """

    def __init__(
        self,
        broker: BrokerInterface,
        config: Optional[ChaseConfig] = None,
        get_mid_price_func=None,
    ):
        """
        Initialize OrderManager.

        Args:
            broker: Broker interface for order execution
            config: Chase configuration (default: ChaseConfig())
            get_mid_price_func: Function to get current mid-price for symbol
                                Signature: async def(symbol: str) -> float
        """
        self.broker = broker
        self.config = config or ChaseConfig()
        self.get_mid_price_func = get_mid_price_func
        self.managed_orders: Dict[str, ManagedOrder] = {}
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_running = False

    async def start(self):
        """
        Start the order monitoring loop.

        This starts a background task that polls order status every 5 seconds.
        """
        if self.is_running:
            logger.warning("OrderManager already running")
            return

        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("OrderManager started")

    async def stop(self):
        """
        Stop the order monitoring loop and clean up.
        """
        if not self.is_running:
            return

        self.is_running = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("OrderManager stopped")

    async def place_order_with_monitoring(
        self,
        symbol: str,
        qty: int,
        side: OrderSide,
        order_type: OrderType,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Order:
        """
        Place an order and add it to monitoring.

        Args:
            symbol: Trading symbol
            qty: Number of shares
            side: BUY or SELL
            order_type: MARKET, LIMIT, STOP, or STOP_LIMIT
            limit_price: Limit price (required for LIMIT orders)
            stop_price: Stop price (required for STOP orders)

        Returns:
            Order object with order details

        Raises:
            ValueError: If required prices not provided
        """
        order = await self.broker.place_order(
            symbol=symbol,
            qty=qty,
            side=side,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
        )

        # Add to monitoring if not filled immediately
        if order.status not in (OrderStatus.FILLED, OrderStatus.REJECTED, OrderStatus.CANCELLED):
            managed_order = ManagedOrder(
                order=order,
                target_symbol=symbol,
                original_qty=qty,
                original_side=side,
            )
            self.managed_orders[order.order_id] = managed_order
            logger.info(f"Added order {order.order_id} to monitoring")

        return order

    async def _monitoring_loop(self):
        """
        Background loop that monitors orders every 5 seconds.

        Requirements:
            - R10.2.1: Poll status every 5 seconds
        """
        try:
            while self.is_running:
                await asyncio.sleep(self.config.poll_interval_seconds)
                await self._check_all_orders()
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
            raise

    async def _check_all_orders(self):
        """
        Check status of all managed orders and handle chasing.

        Requirements:
            - R10.2.2: Chase after 60 seconds if unfilled
            - R10.2.3: Max 3 chase attempts
        """
        current_time = time.time()
        orders_to_remove = []

        for order_id, managed_order in list(self.managed_orders.items()):
            # Skip abandoned orders
            if managed_order.is_abandoned:
                orders_to_remove.append(order_id)
                continue

            # Update order status
            updated_order = await self.broker.get_order_status(order_id)
            if updated_order is None:
                logger.warning(f"Order {order_id} not found in broker")
                orders_to_remove.append(order_id)
                continue

            managed_order.order = updated_order
            managed_order.last_checked_at = current_time

            # Remove if filled, cancelled, or rejected
            if updated_order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED):
                logger.info(f"Order {order_id} terminal status: {updated_order.status.name}")
                orders_to_remove.append(order_id)
                continue

            # Check if order should be chased (R10.2.2)
            time_since_placed = current_time - managed_order.placed_at
            if time_since_placed >= self.config.chase_after_seconds:
                if updated_order.status == OrderStatus.PENDING:
                    await self._chase_order(managed_order)

        # Clean up completed orders
        for order_id in orders_to_remove:
            del self.managed_orders[order_id]

    async def _chase_order(self, managed_order: ManagedOrder):
        """
        Cancel and rechase an order with updated mid-price.

        Requirements:
            - R10.2.2: Cancel existing order, recalculate mid-price, resubmit
            - R10.2.3: Max 3 chase attempts, then abandon

        Args:
            managed_order: The managed order to chase
        """
        order = managed_order.order

        # Check chase limit (R10.2.3)
        if managed_order.chase_count >= self.config.max_chase_attempts:
            logger.warning(
                f"Order {order.order_id} exceeded max chase attempts ({self.config.max_chase_attempts}). Abandoning."
            )
            managed_order.is_abandoned = True
            # Cancel the order
            await self.broker.cancel_order(order.order_id)
            return

        # Cancel existing order
        logger.info(f"Chasing order {order.order_id} (attempt {managed_order.chase_count + 1})")
        cancel_success = await self.broker.cancel_order(order.order_id)

        if not cancel_success:
            logger.warning(f"Failed to cancel order {order.order_id}")
            # Order might have filled during cancel attempt
            return

        # Recalculate mid-price (R10.2.2)
        new_limit_price = order.limit_price
        if self.get_mid_price_func and order.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
            try:
                new_mid_price = await self.get_mid_price_func(order.symbol)
                if new_mid_price:
                    new_limit_price = new_mid_price
                    logger.info(f"Updated limit price from {order.limit_price} to {new_limit_price}")
            except Exception as e:
                logger.error(f"Failed to get mid-price for {order.symbol}: {e}")

        # Resubmit order
        try:
            new_order = await self.broker.place_order(
                symbol=order.symbol,
                qty=order.qty,
                side=order.side,
                order_type=order.order_type,
                limit_price=new_limit_price,
                stop_price=order.stop_price,
            )

            # Update managed order
            managed_order.order = new_order
            managed_order.placed_at = time.time()
            managed_order.last_checked_at = time.time()
            managed_order.chase_count += 1

            # Update dictionary with new order ID
            if new_order.order_id != order.order_id:
                del self.managed_orders[order.order_id]
                self.managed_orders[new_order.order_id] = managed_order

            logger.info(f"Successfully rechased order as {new_order.order_id}")

        except Exception as e:
            logger.error(f"Failed to rechase order {order.order_id}: {e}")
            managed_order.is_abandoned = True

    def get_managed_order(self, order_id: str) -> Optional[ManagedOrder]:
        """
        Get a managed order by ID.

        Args:
            order_id: Order identifier

        Returns:
            ManagedOrder if found, None otherwise
        """
        return self.managed_orders.get(order_id)

    def get_all_managed_orders(self) -> List[ManagedOrder]:
        """
        Get all currently managed orders.

        Returns:
            List of ManagedOrder objects
        """
        return list(self.managed_orders.values())

    def get_active_order_count(self) -> int:
        """
        Get count of active (non-abandoned) managed orders.

        Returns:
            Number of active orders being monitored
        """
        return sum(1 for mo in self.managed_orders.values() if not mo.is_abandoned)
