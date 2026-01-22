"""
Hourly Close Optimization for Retail Trading

This module implements the "hourly close" optimization strategy to gain
a timing advantage over retail traders who wait for candle close confirmation.

**Strategy**:
- Check signals at minute 59 (e.g., 10:59:00)
- Prepare orders for submission at minute 60 (e.g., 11:00:00)
- Gives 60-second headstart vs traders waiting for 11:00:01 close

**Requirements Reference**: FLUXHERO_REQUIREMENTS.md - "The Hourly Close Optimization"

**Usage Example**:
    ```python
    from backend.execution.hourly_close_optimizer import HourlyCloseOptimizer

    optimizer = HourlyCloseOptimizer()

    # Check if current time is at minute 59 (signal check time)
    if optimizer.is_signal_check_time(current_time):
        # Evaluate signals and prepare orders
        order = prepare_order_based_on_signals()
        optimizer.schedule_order(order, current_time)

    # Check if it's time to submit scheduled orders (minute 60)
    if optimizer.is_order_submit_time(current_time):
        pending_orders = optimizer.get_pending_orders(current_time)
        for order in pending_orders:
            submit_order(order)
    ```

Author: FluxHero Team
Date: 2026-01-21
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
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


@dataclass
class ScheduledOrder:
    """
    Represents an order scheduled for submission at the next hour boundary.

    Attributes:
        symbol: Stock symbol (e.g., 'SPY', 'AAPL')
        side: Order side (BUY or SELL)
        quantity: Number of shares
        order_type: Order type (MARKET, LIMIT, etc.)
        limit_price: Limit price for LIMIT/STOP_LIMIT orders
        stop_price: Stop price for STOP/STOP_LIMIT orders
        scheduled_time: Target submission time (minute 60)
        signal_check_time: Time when signal was checked (minute 59)
        signal_reason: Explanation for the trade signal
        metadata: Additional order metadata
    """
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    scheduled_time: Optional[datetime] = None
    signal_check_time: Optional[datetime] = None
    signal_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class HourlyCloseOptimizer:
    """
    Implements the hourly close optimization strategy.

    This class manages the timing of signal checks and order submissions
    to gain a timing advantage over retail traders.

    **Key Features**:
    - Detects minute 59 for signal evaluation
    - Detects minute 60 for order submission
    - Manages scheduled orders queue
    - Validates order timing
    - Tracks submission history

    **Timing Logic**:
    - Minute 59: Check signals, prepare orders
    - Minute 60: Submit prepared orders
    - Headstart: 60 seconds before typical retail trader (11:00:01)

    Attributes:
        signal_check_minute: Minute to check signals (default: 59)
        order_submit_minute: Minute to submit orders (default: 0, i.e., top of hour)
        scheduled_orders: Queue of pending orders
        submitted_orders_history: History of submitted orders
        max_history_size: Maximum number of historical orders to keep
    """

    def __init__(
        self,
        signal_check_minute: int = 59,
        order_submit_minute: int = 0,
        max_history_size: int = 100
    ):
        """
        Initialize the hourly close optimizer.

        Args:
            signal_check_minute: Minute to check signals (0-59, default: 59)
            order_submit_minute: Minute to submit orders (0-59, default: 0)
            max_history_size: Max historical orders to keep

        Raises:
            ValueError: If minutes are not in valid range [0, 59]
        """
        if not (0 <= signal_check_minute <= 59):
            raise ValueError("signal_check_minute must be between 0 and 59")
        if not (0 <= order_submit_minute <= 59):
            raise ValueError("order_submit_minute must be between 0 and 59")

        self.signal_check_minute = signal_check_minute
        self.order_submit_minute = order_submit_minute
        self.max_history_size = max_history_size

        self.scheduled_orders: List[ScheduledOrder] = []
        self.submitted_orders_history: List[ScheduledOrder] = []

    def is_signal_check_time(self, current_time: datetime) -> bool:
        """
        Check if current time is at the signal check minute (minute 59).

        This method should be called continuously to detect when to evaluate
        trading signals.

        Args:
            current_time: Current timestamp

        Returns:
            True if current minute matches signal_check_minute, False otherwise

        Example:
            >>> optimizer = HourlyCloseOptimizer()
            >>> time_10_59 = datetime(2024, 1, 15, 10, 59, 0)
            >>> optimizer.is_signal_check_time(time_10_59)
            True
            >>> time_10_58 = datetime(2024, 1, 15, 10, 58, 0)
            >>> optimizer.is_signal_check_time(time_10_58)
            False
        """
        return current_time.minute == self.signal_check_minute

    def is_order_submit_time(self, current_time: datetime) -> bool:
        """
        Check if current time is at the order submission minute (minute 0).

        This method should be called continuously to detect when to submit
        scheduled orders.

        Args:
            current_time: Current timestamp

        Returns:
            True if current minute matches order_submit_minute, False otherwise

        Example:
            >>> optimizer = HourlyCloseOptimizer()
            >>> time_11_00 = datetime(2024, 1, 15, 11, 0, 0)
            >>> optimizer.is_order_submit_time(time_11_00)
            True
            >>> time_11_01 = datetime(2024, 1, 15, 11, 1, 0)
            >>> optimizer.is_order_submit_time(time_11_01)
            False
        """
        return current_time.minute == self.order_submit_minute

    def schedule_order(
        self,
        order: ScheduledOrder,
        current_time: datetime
    ) -> ScheduledOrder:
        """
        Schedule an order for submission at the next hour boundary.

        This method should be called at minute 59 after signal evaluation.
        It calculates the target submission time and adds the order to the queue.

        Args:
            order: Order to schedule (without scheduled_time set)
            current_time: Current timestamp (should be at minute 59)

        Returns:
            The scheduled order with submission time set

        Example:
            >>> optimizer = HourlyCloseOptimizer()
            >>> time_10_59 = datetime(2024, 1, 15, 10, 59, 30)
            >>> order = ScheduledOrder(
            ...     symbol='SPY',
            ...     side=OrderSide.BUY,
            ...     quantity=100,
            ...     order_type=OrderType.MARKET
            ... )
            >>> scheduled = optimizer.schedule_order(order, time_10_59)
            >>> scheduled.scheduled_time
            datetime.datetime(2024, 1, 15, 11, 0, 0)
        """
        # Calculate next hour boundary (minute 0)
        if current_time.minute == 59:
            # If at minute 59, submit at next hour
            submit_time = current_time.replace(
                minute=0, second=0, microsecond=0
            ) + timedelta(hours=1)
        else:
            # If not at minute 59, calculate next appropriate submit time
            if current_time.minute < self.order_submit_minute:
                # Submit this hour
                submit_time = current_time.replace(
                    minute=self.order_submit_minute, second=0, microsecond=0
                )
            else:
                # Submit next hour
                submit_time = current_time.replace(
                    minute=self.order_submit_minute, second=0, microsecond=0
                ) + timedelta(hours=1)

        # Set timing metadata
        order.scheduled_time = submit_time
        order.signal_check_time = current_time

        # Add to scheduled orders queue
        self.scheduled_orders.append(order)

        return order

    def get_pending_orders(
        self,
        current_time: datetime,
        window_seconds: int = 60
    ) -> List[ScheduledOrder]:
        """
        Get orders scheduled for submission at the current time.

        Returns orders whose scheduled_time is within the window around
        current_time. This allows for some timing flexibility.

        Args:
            current_time: Current timestamp
            window_seconds: Time window in seconds (default: 60)

        Returns:
            List of orders ready for submission

        Example:
            >>> optimizer = HourlyCloseOptimizer()
            >>> # ... schedule some orders at 10:59 ...
            >>> time_11_00 = datetime(2024, 1, 15, 11, 0, 15)
            >>> pending = optimizer.get_pending_orders(time_11_00)
            >>> len(pending)
            3  # Returns orders scheduled for 11:00
        """
        pending = []
        window_delta = timedelta(seconds=window_seconds)

        for order in self.scheduled_orders:
            if order.scheduled_time is None:
                continue

            # Check if order is within submission window
            time_diff = abs(current_time - order.scheduled_time)
            if time_diff <= window_delta:
                pending.append(order)

        return pending

    def mark_order_submitted(self, order: ScheduledOrder) -> None:
        """
        Mark an order as submitted and move it to history.

        Removes the order from scheduled_orders and adds it to
        submitted_orders_history.

        Args:
            order: Order that was successfully submitted

        Example:
            >>> optimizer = HourlyCloseOptimizer()
            >>> # ... schedule and retrieve order ...
            >>> optimizer.mark_order_submitted(order)
            >>> len(optimizer.scheduled_orders)
            0  # Order removed from queue
            >>> len(optimizer.submitted_orders_history)
            1  # Order added to history
        """
        # Remove from scheduled orders
        if order in self.scheduled_orders:
            self.scheduled_orders.remove(order)

        # Add to history
        self.submitted_orders_history.append(order)

        # Trim history if needed
        if len(self.submitted_orders_history) > self.max_history_size:
            self.submitted_orders_history = self.submitted_orders_history[-self.max_history_size:]

    def cancel_order(self, order: ScheduledOrder) -> bool:
        """
        Cancel a scheduled order before submission.

        Args:
            order: Order to cancel

        Returns:
            True if order was found and cancelled, False otherwise

        Example:
            >>> optimizer = HourlyCloseOptimizer()
            >>> # ... schedule order ...
            >>> success = optimizer.cancel_order(order)
            >>> success
            True
        """
        if order in self.scheduled_orders:
            self.scheduled_orders.remove(order)
            return True
        return False

    def cancel_all_orders(self) -> int:
        """
        Cancel all scheduled orders.

        Returns:
            Number of orders cancelled

        Example:
            >>> optimizer = HourlyCloseOptimizer()
            >>> # ... schedule 3 orders ...
            >>> cancelled_count = optimizer.cancel_all_orders()
            >>> cancelled_count
            3
            >>> len(optimizer.scheduled_orders)
            0
        """
        count = len(self.scheduled_orders)
        self.scheduled_orders.clear()
        return count

    def get_scheduled_order_count(self) -> int:
        """
        Get the number of currently scheduled orders.

        Returns:
            Number of orders in the queue
        """
        return len(self.scheduled_orders)

    def get_submission_history_count(self) -> int:
        """
        Get the number of orders in submission history.

        Returns:
            Number of historical submissions
        """
        return len(self.submitted_orders_history)

    def clear_history(self) -> None:
        """
        Clear the submission history.

        This can be called periodically to free memory.
        """
        self.submitted_orders_history.clear()

    def get_time_until_next_check(self, current_time: datetime) -> timedelta:
        """
        Calculate time remaining until next signal check window.

        Args:
            current_time: Current timestamp

        Returns:
            Time delta until next signal check minute

        Example:
            >>> optimizer = HourlyCloseOptimizer()
            >>> time_10_45 = datetime(2024, 1, 15, 10, 45, 0)
            >>> delta = optimizer.get_time_until_next_check(time_10_45)
            >>> delta.total_seconds()
            840.0  # 14 minutes until 10:59
        """
        current_minute = current_time.minute

        if current_minute < self.signal_check_minute:
            # Next check is this hour
            next_check = current_time.replace(
                minute=self.signal_check_minute, second=0, microsecond=0
            )
        else:
            # Next check is next hour
            next_check = current_time.replace(
                minute=self.signal_check_minute, second=0, microsecond=0
            ) + timedelta(hours=1)

        return next_check - current_time

    def get_time_until_next_submit(self, current_time: datetime) -> timedelta:
        """
        Calculate time remaining until next order submission window.

        Args:
            current_time: Current timestamp

        Returns:
            Time delta until next order submit minute

        Example:
            >>> optimizer = HourlyCloseOptimizer()
            >>> time_10_59 = datetime(2024, 1, 15, 10, 59, 30)
            >>> delta = optimizer.get_time_until_next_submit(time_10_59)
            >>> delta.total_seconds()
            30.0  # 30 seconds until 11:00
        """
        current_minute = current_time.minute

        if current_minute < self.order_submit_minute:
            # Next submit is this hour
            next_submit = current_time.replace(
                minute=self.order_submit_minute, second=0, microsecond=0
            )
        elif current_minute == self.order_submit_minute:
            # We're in the submit window
            next_submit = current_time.replace(second=0, microsecond=0)
        else:
            # Next submit is next hour
            next_submit = current_time.replace(
                minute=self.order_submit_minute, second=0, microsecond=0
            ) + timedelta(hours=1)

        time_diff = next_submit - current_time
        if time_diff.total_seconds() < 0:
            time_diff = timedelta(0)

        return time_diff
