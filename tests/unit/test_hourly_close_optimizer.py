"""
Unit tests for hourly close optimizer module.

Tests the timing logic for signal checks and order submissions
to gain an execution advantage over retail traders.

Author: FluxHero Team
Date: 2026-01-21
"""

from datetime import datetime

import pytest

from backend.execution.hourly_close_optimizer import (
    HourlyCloseOptimizer,
    OrderSide,
    OrderType,
    ScheduledOrder,
)


class TestHourlyCloseOptimizerInitialization:
    """Test initialization and configuration."""

    def test_default_initialization(self):
        """Test default initialization values."""
        optimizer = HourlyCloseOptimizer()

        assert optimizer.signal_check_minute == 59
        assert optimizer.order_submit_minute == 0
        assert optimizer.max_history_size == 100
        assert len(optimizer.scheduled_orders) == 0
        assert len(optimizer.submitted_orders_history) == 0

    def test_custom_initialization(self):
        """Test initialization with custom values."""
        optimizer = HourlyCloseOptimizer(
            signal_check_minute=55,
            order_submit_minute=56,
            max_history_size=50
        )

        assert optimizer.signal_check_minute == 55
        assert optimizer.order_submit_minute == 56
        assert optimizer.max_history_size == 50

    def test_invalid_signal_check_minute(self):
        """Test initialization with invalid signal_check_minute."""
        with pytest.raises(ValueError, match="signal_check_minute must be between 0 and 59"):
            HourlyCloseOptimizer(signal_check_minute=60)

        with pytest.raises(ValueError, match="signal_check_minute must be between 0 and 59"):
            HourlyCloseOptimizer(signal_check_minute=-1)

    def test_invalid_order_submit_minute(self):
        """Test initialization with invalid order_submit_minute."""
        with pytest.raises(ValueError, match="order_submit_minute must be between 0 and 59"):
            HourlyCloseOptimizer(order_submit_minute=60)

        with pytest.raises(ValueError, match="order_submit_minute must be between 0 and 59"):
            HourlyCloseOptimizer(order_submit_minute=-1)


class TestSignalCheckTiming:
    """Test signal check time detection."""

    def test_is_signal_check_time_true(self):
        """Test signal check time detection at minute 59."""
        optimizer = HourlyCloseOptimizer()
        time_10_59 = datetime(2024, 1, 15, 10, 59, 0)

        assert optimizer.is_signal_check_time(time_10_59) is True

    def test_is_signal_check_time_false(self):
        """Test signal check time detection at other minutes."""
        optimizer = HourlyCloseOptimizer()

        time_10_58 = datetime(2024, 1, 15, 10, 58, 0)
        assert optimizer.is_signal_check_time(time_10_58) is False

        time_11_00 = datetime(2024, 1, 15, 11, 0, 0)
        assert optimizer.is_signal_check_time(time_11_00) is False

        time_10_00 = datetime(2024, 1, 15, 10, 0, 0)
        assert optimizer.is_signal_check_time(time_10_00) is False

    def test_is_signal_check_time_with_seconds(self):
        """Test signal check time works with any seconds value."""
        optimizer = HourlyCloseOptimizer()

        time_10_59_30 = datetime(2024, 1, 15, 10, 59, 30)
        assert optimizer.is_signal_check_time(time_10_59_30) is True

        time_10_59_59 = datetime(2024, 1, 15, 10, 59, 59)
        assert optimizer.is_signal_check_time(time_10_59_59) is True

    def test_custom_signal_check_minute(self):
        """Test signal check time with custom minute."""
        optimizer = HourlyCloseOptimizer(signal_check_minute=45)

        time_10_45 = datetime(2024, 1, 15, 10, 45, 0)
        assert optimizer.is_signal_check_time(time_10_45) is True

        time_10_59 = datetime(2024, 1, 15, 10, 59, 0)
        assert optimizer.is_signal_check_time(time_10_59) is False


class TestOrderSubmitTiming:
    """Test order submission time detection."""

    def test_is_order_submit_time_true(self):
        """Test order submit time detection at minute 0."""
        optimizer = HourlyCloseOptimizer()
        time_11_00 = datetime(2024, 1, 15, 11, 0, 0)

        assert optimizer.is_order_submit_time(time_11_00) is True

    def test_is_order_submit_time_false(self):
        """Test order submit time detection at other minutes."""
        optimizer = HourlyCloseOptimizer()

        time_10_59 = datetime(2024, 1, 15, 10, 59, 0)
        assert optimizer.is_order_submit_time(time_10_59) is False

        time_11_01 = datetime(2024, 1, 15, 11, 1, 0)
        assert optimizer.is_order_submit_time(time_11_01) is False

        time_11_30 = datetime(2024, 1, 15, 11, 30, 0)
        assert optimizer.is_order_submit_time(time_11_30) is False

    def test_is_order_submit_time_with_seconds(self):
        """Test order submit time works with any seconds value."""
        optimizer = HourlyCloseOptimizer()

        time_11_00_15 = datetime(2024, 1, 15, 11, 0, 15)
        assert optimizer.is_order_submit_time(time_11_00_15) is True

        time_11_00_59 = datetime(2024, 1, 15, 11, 0, 59)
        assert optimizer.is_order_submit_time(time_11_00_59) is True

    def test_custom_order_submit_minute(self):
        """Test order submit time with custom minute."""
        optimizer = HourlyCloseOptimizer(order_submit_minute=5)

        time_11_05 = datetime(2024, 1, 15, 11, 5, 0)
        assert optimizer.is_order_submit_time(time_11_05) is True

        time_11_00 = datetime(2024, 1, 15, 11, 0, 0)
        assert optimizer.is_order_submit_time(time_11_00) is False


class TestOrderScheduling:
    """Test order scheduling logic."""

    def test_schedule_order_at_minute_59(self):
        """Test scheduling an order at minute 59."""
        optimizer = HourlyCloseOptimizer()
        time_10_59_30 = datetime(2024, 1, 15, 10, 59, 30)

        order = ScheduledOrder(
            symbol='SPY',
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET
        )

        scheduled = optimizer.schedule_order(order, time_10_59_30)

        assert scheduled.scheduled_time == datetime(2024, 1, 15, 11, 0, 0)
        assert scheduled.signal_check_time == time_10_59_30
        assert len(optimizer.scheduled_orders) == 1
        assert optimizer.scheduled_orders[0] == scheduled

    def test_schedule_multiple_orders(self):
        """Test scheduling multiple orders."""
        optimizer = HourlyCloseOptimizer()
        time_10_59 = datetime(2024, 1, 15, 10, 59, 0)

        orders = [
            ScheduledOrder(symbol='SPY', side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET),
            ScheduledOrder(symbol='AAPL', side=OrderSide.BUY, quantity=50, order_type=OrderType.LIMIT, limit_price=150.0),
            ScheduledOrder(symbol='MSFT', side=OrderSide.SELL, quantity=25, order_type=OrderType.MARKET),
        ]

        for order in orders:
            optimizer.schedule_order(order, time_10_59)

        assert len(optimizer.scheduled_orders) == 3
        assert all(order.scheduled_time == datetime(2024, 1, 15, 11, 0, 0) for order in optimizer.scheduled_orders)

    def test_schedule_order_with_metadata(self):
        """Test scheduling order with additional metadata."""
        optimizer = HourlyCloseOptimizer()
        time_10_59 = datetime(2024, 1, 15, 10, 59, 0)

        order = ScheduledOrder(
            symbol='SPY',
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            signal_reason="KAMA crossover + high ER",
            metadata={'regime': 'STRONG_TREND', 'atr': 2.5}
        )

        scheduled = optimizer.schedule_order(order, time_10_59)

        assert scheduled.signal_reason == "KAMA crossover + high ER"
        assert scheduled.metadata['regime'] == 'STRONG_TREND'
        assert scheduled.metadata['atr'] == 2.5

    def test_schedule_order_not_at_minute_59(self):
        """Test scheduling order at times other than minute 59."""
        optimizer = HourlyCloseOptimizer()

        # Before submit minute (minute 0)
        time_10_45 = datetime(2024, 1, 15, 10, 45, 0)
        order1 = ScheduledOrder(symbol='SPY', side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)
        scheduled1 = optimizer.schedule_order(order1, time_10_45)
        # Should schedule for next hour since 45 > 0
        assert scheduled1.scheduled_time == datetime(2024, 1, 15, 11, 0, 0)

        # After submit minute
        time_10_30 = datetime(2024, 1, 15, 10, 30, 0)
        order2 = ScheduledOrder(symbol='AAPL', side=OrderSide.BUY, quantity=50, order_type=OrderType.MARKET)
        scheduled2 = optimizer.schedule_order(order2, time_10_30)
        # Should schedule for next hour
        assert scheduled2.scheduled_time == datetime(2024, 1, 15, 11, 0, 0)


class TestPendingOrders:
    """Test retrieval of pending orders."""

    def test_get_pending_orders_exact_time(self):
        """Test getting pending orders at exact submission time."""
        optimizer = HourlyCloseOptimizer()
        time_10_59 = datetime(2024, 1, 15, 10, 59, 0)

        # Schedule 3 orders
        for i, symbol in enumerate(['SPY', 'AAPL', 'MSFT']):
            order = ScheduledOrder(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=100 * (i + 1),
                order_type=OrderType.MARKET
            )
            optimizer.schedule_order(order, time_10_59)

        # Get pending at 11:00
        time_11_00 = datetime(2024, 1, 15, 11, 0, 0)
        pending = optimizer.get_pending_orders(time_11_00)

        assert len(pending) == 3
        assert all(order.scheduled_time == datetime(2024, 1, 15, 11, 0, 0) for order in pending)

    def test_get_pending_orders_within_window(self):
        """Test getting pending orders within time window."""
        optimizer = HourlyCloseOptimizer()
        time_10_59 = datetime(2024, 1, 15, 10, 59, 0)

        order = ScheduledOrder(symbol='SPY', side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)
        optimizer.schedule_order(order, time_10_59)

        # 15 seconds after scheduled time (within default 60s window)
        time_11_00_15 = datetime(2024, 1, 15, 11, 0, 15)
        pending = optimizer.get_pending_orders(time_11_00_15)

        assert len(pending) == 1

    def test_get_pending_orders_outside_window(self):
        """Test no pending orders returned outside time window."""
        optimizer = HourlyCloseOptimizer()
        time_10_59 = datetime(2024, 1, 15, 10, 59, 0)

        order = ScheduledOrder(symbol='SPY', side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)
        optimizer.schedule_order(order, time_10_59)

        # 2 minutes after scheduled time (outside 60s window)
        time_11_02 = datetime(2024, 1, 15, 11, 2, 0)
        pending = optimizer.get_pending_orders(time_11_02)

        assert len(pending) == 0

    def test_get_pending_orders_custom_window(self):
        """Test getting pending orders with custom time window."""
        optimizer = HourlyCloseOptimizer()
        time_10_59 = datetime(2024, 1, 15, 10, 59, 0)

        order = ScheduledOrder(symbol='SPY', side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)
        optimizer.schedule_order(order, time_10_59)

        # 90 seconds after scheduled time
        time_11_01_30 = datetime(2024, 1, 15, 11, 1, 30)

        # With 60s window, should return empty
        pending_60s = optimizer.get_pending_orders(time_11_01_30, window_seconds=60)
        assert len(pending_60s) == 0

        # With 120s window, should return the order
        pending_120s = optimizer.get_pending_orders(time_11_01_30, window_seconds=120)
        assert len(pending_120s) == 1

    def test_get_pending_orders_empty_queue(self):
        """Test getting pending orders when queue is empty."""
        optimizer = HourlyCloseOptimizer()
        time_11_00 = datetime(2024, 1, 15, 11, 0, 0)

        pending = optimizer.get_pending_orders(time_11_00)
        assert len(pending) == 0


class TestOrderSubmission:
    """Test order submission and history tracking."""

    def test_mark_order_submitted(self):
        """Test marking order as submitted."""
        optimizer = HourlyCloseOptimizer()
        time_10_59 = datetime(2024, 1, 15, 10, 59, 0)

        order = ScheduledOrder(symbol='SPY', side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)
        optimizer.schedule_order(order, time_10_59)

        assert len(optimizer.scheduled_orders) == 1
        assert len(optimizer.submitted_orders_history) == 0

        optimizer.mark_order_submitted(order)

        assert len(optimizer.scheduled_orders) == 0
        assert len(optimizer.submitted_orders_history) == 1
        assert optimizer.submitted_orders_history[0] == order

    def test_mark_multiple_orders_submitted(self):
        """Test marking multiple orders as submitted."""
        optimizer = HourlyCloseOptimizer()
        time_10_59 = datetime(2024, 1, 15, 10, 59, 0)

        orders = []
        for symbol in ['SPY', 'AAPL', 'MSFT']:
            order = ScheduledOrder(symbol=symbol, side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)
            optimizer.schedule_order(order, time_10_59)
            orders.append(order)

        for order in orders:
            optimizer.mark_order_submitted(order)

        assert len(optimizer.scheduled_orders) == 0
        assert len(optimizer.submitted_orders_history) == 3

    def test_history_max_size_limit(self):
        """Test history respects max_history_size limit."""
        optimizer = HourlyCloseOptimizer(max_history_size=5)
        time_10_59 = datetime(2024, 1, 15, 10, 59, 0)

        # Submit 10 orders (exceeds max_history_size of 5)
        for i in range(10):
            order = ScheduledOrder(
                symbol=f'SYM{i}',
                side=OrderSide.BUY,
                quantity=100,
                order_type=OrderType.MARKET
            )
            optimizer.schedule_order(order, time_10_59)
            optimizer.mark_order_submitted(order)

        # Only last 5 should be kept
        assert len(optimizer.submitted_orders_history) == 5
        assert optimizer.submitted_orders_history[0].symbol == 'SYM5'
        assert optimizer.submitted_orders_history[4].symbol == 'SYM9'


class TestOrderCancellation:
    """Test order cancellation logic."""

    def test_cancel_order(self):
        """Test cancelling a scheduled order."""
        optimizer = HourlyCloseOptimizer()
        time_10_59 = datetime(2024, 1, 15, 10, 59, 0)

        order = ScheduledOrder(symbol='SPY', side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)
        optimizer.schedule_order(order, time_10_59)

        assert len(optimizer.scheduled_orders) == 1

        success = optimizer.cancel_order(order)

        assert success is True
        assert len(optimizer.scheduled_orders) == 0

    def test_cancel_nonexistent_order(self):
        """Test cancelling an order that doesn't exist."""
        optimizer = HourlyCloseOptimizer()

        order = ScheduledOrder(symbol='SPY', side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)
        success = optimizer.cancel_order(order)

        assert success is False

    def test_cancel_all_orders(self):
        """Test cancelling all scheduled orders."""
        optimizer = HourlyCloseOptimizer()
        time_10_59 = datetime(2024, 1, 15, 10, 59, 0)

        # Schedule 5 orders
        for i in range(5):
            order = ScheduledOrder(
                symbol=f'SYM{i}',
                side=OrderSide.BUY,
                quantity=100,
                order_type=OrderType.MARKET
            )
            optimizer.schedule_order(order, time_10_59)

        assert len(optimizer.scheduled_orders) == 5

        cancelled_count = optimizer.cancel_all_orders()

        assert cancelled_count == 5
        assert len(optimizer.scheduled_orders) == 0

    def test_cancel_all_orders_empty_queue(self):
        """Test cancelling all orders when queue is empty."""
        optimizer = HourlyCloseOptimizer()

        cancelled_count = optimizer.cancel_all_orders()
        assert cancelled_count == 0


class TestUtilityMethods:
    """Test utility and helper methods."""

    def test_get_scheduled_order_count(self):
        """Test getting scheduled order count."""
        optimizer = HourlyCloseOptimizer()
        time_10_59 = datetime(2024, 1, 15, 10, 59, 0)

        assert optimizer.get_scheduled_order_count() == 0

        for i in range(3):
            order = ScheduledOrder(symbol=f'SYM{i}', side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)
            optimizer.schedule_order(order, time_10_59)

        assert optimizer.get_scheduled_order_count() == 3

    def test_get_submission_history_count(self):
        """Test getting submission history count."""
        optimizer = HourlyCloseOptimizer()
        time_10_59 = datetime(2024, 1, 15, 10, 59, 0)

        assert optimizer.get_submission_history_count() == 0

        for i in range(3):
            order = ScheduledOrder(symbol=f'SYM{i}', side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)
            optimizer.schedule_order(order, time_10_59)
            optimizer.mark_order_submitted(order)

        assert optimizer.get_submission_history_count() == 3

    def test_clear_history(self):
        """Test clearing submission history."""
        optimizer = HourlyCloseOptimizer()
        time_10_59 = datetime(2024, 1, 15, 10, 59, 0)

        # Submit some orders
        for i in range(3):
            order = ScheduledOrder(symbol=f'SYM{i}', side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)
            optimizer.schedule_order(order, time_10_59)
            optimizer.mark_order_submitted(order)

        assert len(optimizer.submitted_orders_history) == 3

        optimizer.clear_history()

        assert len(optimizer.submitted_orders_history) == 0


class TestTimingCalculations:
    """Test time calculation methods."""

    def test_get_time_until_next_check_before_check_minute(self):
        """Test time until next check when before check minute."""
        optimizer = HourlyCloseOptimizer()
        time_10_45 = datetime(2024, 1, 15, 10, 45, 0)

        time_until = optimizer.get_time_until_next_check(time_10_45)

        # 14 minutes until 10:59
        assert time_until.total_seconds() == 14 * 60

    def test_get_time_until_next_check_after_check_minute(self):
        """Test time until next check when after check minute."""
        optimizer = HourlyCloseOptimizer()
        time_11_00 = datetime(2024, 1, 15, 11, 0, 0)

        time_until = optimizer.get_time_until_next_check(time_11_00)

        # 59 minutes until 11:59
        assert time_until.total_seconds() == 59 * 60

    def test_get_time_until_next_check_at_check_minute(self):
        """Test time until next check when at check minute."""
        optimizer = HourlyCloseOptimizer()
        time_10_59_30 = datetime(2024, 1, 15, 10, 59, 30)

        time_until = optimizer.get_time_until_next_check(time_10_59_30)

        # At minute 59, next check is next hour (60 minutes - 30 seconds)
        assert time_until.total_seconds() == 60 * 60 - 30

    def test_get_time_until_next_submit_before_submit_minute(self):
        """Test time until next submit when before submit minute."""
        optimizer = HourlyCloseOptimizer()
        time_10_59_30 = datetime(2024, 1, 15, 10, 59, 30)

        time_until = optimizer.get_time_until_next_submit(time_10_59_30)

        # 30 seconds until 11:00
        assert time_until.total_seconds() == 30

    def test_get_time_until_next_submit_after_submit_minute(self):
        """Test time until next submit when after submit minute."""
        optimizer = HourlyCloseOptimizer()
        time_11_01 = datetime(2024, 1, 15, 11, 1, 0)

        time_until = optimizer.get_time_until_next_submit(time_11_01)

        # 59 minutes until 12:00
        assert time_until.total_seconds() == 59 * 60

    def test_get_time_until_next_submit_at_submit_minute(self):
        """Test time until next submit when at submit minute."""
        optimizer = HourlyCloseOptimizer()
        time_11_00_15 = datetime(2024, 1, 15, 11, 0, 15)

        time_until = optimizer.get_time_until_next_submit(time_11_00_15)

        # At minute 0, time until should be 0 (or very small)
        assert time_until.total_seconds() == 0


class TestIntegrationScenarios:
    """Test complete workflow scenarios."""

    def test_complete_hourly_workflow(self):
        """Test complete workflow: check signal → schedule → submit."""
        optimizer = HourlyCloseOptimizer()

        # 1. At 10:59, check if it's signal time
        time_10_59 = datetime(2024, 1, 15, 10, 59, 0)
        assert optimizer.is_signal_check_time(time_10_59) is True

        # 2. Schedule orders based on signals
        orders = [
            ScheduledOrder(symbol='SPY', side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET),
            ScheduledOrder(symbol='AAPL', side=OrderSide.SELL, quantity=50, order_type=OrderType.MARKET),
        ]

        for order in orders:
            optimizer.schedule_order(order, time_10_59)

        assert optimizer.get_scheduled_order_count() == 2

        # 3. At 11:00, check if it's submit time
        time_11_00 = datetime(2024, 1, 15, 11, 0, 0)
        assert optimizer.is_order_submit_time(time_11_00) is True

        # 4. Get pending orders
        pending = optimizer.get_pending_orders(time_11_00)
        assert len(pending) == 2

        # 5. Submit orders and mark as submitted
        for order in pending:
            # In real system, would submit to broker here
            optimizer.mark_order_submitted(order)

        assert optimizer.get_scheduled_order_count() == 0
        assert optimizer.get_submission_history_count() == 2

    def test_multiple_hour_workflow(self):
        """Test workflow across multiple hours."""
        optimizer = HourlyCloseOptimizer()

        # Hour 1: 10:59 → 11:00
        time_10_59 = datetime(2024, 1, 15, 10, 59, 0)
        order1 = ScheduledOrder(symbol='SPY', side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)
        optimizer.schedule_order(order1, time_10_59)

        time_11_00 = datetime(2024, 1, 15, 11, 0, 0)
        pending1 = optimizer.get_pending_orders(time_11_00)
        assert len(pending1) == 1
        optimizer.mark_order_submitted(pending1[0])

        # Hour 2: 11:59 → 12:00
        time_11_59 = datetime(2024, 1, 15, 11, 59, 0)
        order2 = ScheduledOrder(symbol='AAPL', side=OrderSide.BUY, quantity=50, order_type=OrderType.MARKET)
        optimizer.schedule_order(order2, time_11_59)

        time_12_00 = datetime(2024, 1, 15, 12, 0, 0)
        pending2 = optimizer.get_pending_orders(time_12_00)
        assert len(pending2) == 1
        optimizer.mark_order_submitted(pending2[0])

        # Verify history
        assert optimizer.get_submission_history_count() == 2

    def test_missed_submission_window(self):
        """Test handling of missed submission window."""
        optimizer = HourlyCloseOptimizer()

        # Schedule order at 10:59
        time_10_59 = datetime(2024, 1, 15, 10, 59, 0)
        order = ScheduledOrder(symbol='SPY', side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)
        optimizer.schedule_order(order, time_10_59)

        # Check much later (11:05, outside 60s window)
        time_11_05 = datetime(2024, 1, 15, 11, 5, 0)
        pending = optimizer.get_pending_orders(time_11_05)

        # Should return empty (missed window)
        assert len(pending) == 0

        # Order still in queue though
        assert optimizer.get_scheduled_order_count() == 1

    def test_cancel_before_submission(self):
        """Test cancelling orders before submission."""
        optimizer = HourlyCloseOptimizer()

        # Schedule order at 10:59
        time_10_59 = datetime(2024, 1, 15, 10, 59, 0)
        order = ScheduledOrder(symbol='SPY', side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)
        optimizer.schedule_order(order, time_10_59)

        # Cancel before 11:00
        success = optimizer.cancel_order(order)
        assert success is True

        # At 11:00, no pending orders
        time_11_00 = datetime(2024, 1, 15, 11, 0, 0)
        pending = optimizer.get_pending_orders(time_11_00)
        assert len(pending) == 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_schedule_at_exactly_minute_0(self):
        """Test scheduling at exactly minute 0."""
        optimizer = HourlyCloseOptimizer()
        time_11_00 = datetime(2024, 1, 15, 11, 0, 0)

        order = ScheduledOrder(symbol='SPY', side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)
        scheduled = optimizer.schedule_order(order, time_11_00)

        # Should schedule for next hour
        assert scheduled.scheduled_time == datetime(2024, 1, 15, 12, 0, 0)

    def test_order_with_all_fields(self):
        """Test order with all fields populated."""
        optimizer = HourlyCloseOptimizer()
        time_10_59 = datetime(2024, 1, 15, 10, 59, 0)

        order = ScheduledOrder(
            symbol='AAPL',
            side=OrderSide.BUY,
            quantity=50,
            order_type=OrderType.STOP_LIMIT,
            limit_price=150.0,
            stop_price=148.0,
            signal_reason="Strong breakout with high volume",
            metadata={'atr': 2.5, 'regime': 'STRONG_TREND', 'volatility': 'NORMAL'}
        )

        scheduled = optimizer.schedule_order(order, time_10_59)

        assert scheduled.symbol == 'AAPL'
        assert scheduled.side == OrderSide.BUY
        assert scheduled.quantity == 50
        assert scheduled.order_type == OrderType.STOP_LIMIT
        assert scheduled.limit_price == 150.0
        assert scheduled.stop_price == 148.0
        assert scheduled.signal_reason == "Strong breakout with high volume"
        assert scheduled.metadata['atr'] == 2.5

    def test_short_order(self):
        """Test scheduling short (sell) order."""
        optimizer = HourlyCloseOptimizer()
        time_10_59 = datetime(2024, 1, 15, 10, 59, 0)

        order = ScheduledOrder(
            symbol='SPY',
            side=OrderSide.SELL,
            quantity=100,
            order_type=OrderType.MARKET
        )

        scheduled = optimizer.schedule_order(order, time_10_59)

        assert scheduled.side == OrderSide.SELL
        assert scheduled.scheduled_time == datetime(2024, 1, 15, 11, 0, 0)

    def test_across_day_boundary(self):
        """Test scheduling across day boundary."""
        optimizer = HourlyCloseOptimizer()
        time_23_59 = datetime(2024, 1, 15, 23, 59, 0)

        order = ScheduledOrder(symbol='SPY', side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)
        scheduled = optimizer.schedule_order(order, time_23_59)

        # Should schedule for next day at 00:00
        assert scheduled.scheduled_time == datetime(2024, 1, 16, 0, 0, 0)
