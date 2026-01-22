"""
Unit tests for broker interface and PaperBroker implementation.

Tests cover:
- Abstract broker interface definition
- PaperBroker initialization
- Order placement (market, limit, stop, stop-limit)
- Order cancellation
- Order status queries
- Position tracking
- Account management
- Fill simulation logic
- Edge cases and error handling
"""

import pytest
from backend.execution.broker_interface import (
    BrokerInterface,
    PaperBroker,
    Order,
    Position,
    OrderSide,
    OrderType,
    OrderStatus,
)


# Test BrokerInterface abstract class
def test_broker_interface_is_abstract():
    """Test that BrokerInterface cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BrokerInterface()


# Test PaperBroker initialization
def test_paper_broker_initialization():
    """Test PaperBroker initializes with correct defaults."""
    broker = PaperBroker(initial_capital=50000.0)

    assert broker.initial_capital == 50000.0
    assert broker.cash == 50000.0
    assert broker.equity == 50000.0
    assert len(broker.orders) == 0
    assert len(broker.positions) == 0
    assert broker.order_counter == 0


def test_paper_broker_default_capital():
    """Test PaperBroker default capital is $100k."""
    broker = PaperBroker()

    assert broker.initial_capital == 100000.0
    assert broker.cash == 100000.0


# Test market price setting
def test_set_market_price():
    """Test setting market prices for symbols."""
    broker = PaperBroker()
    broker.set_market_price("SPY", 450.0)
    broker.set_market_price("AAPL", 180.0)

    assert broker.market_prices["SPY"] == 450.0
    assert broker.market_prices["AAPL"] == 180.0


# Test market order placement
@pytest.mark.asyncio
async def test_place_market_order_buy():
    """Test placing a market buy order."""
    broker = PaperBroker(initial_capital=100000.0)
    broker.set_market_price("SPY", 450.0)

    order = await broker.place_order(
        symbol="SPY",
        qty=100,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
    )

    assert order.symbol == "SPY"
    assert order.qty == 100
    assert order.side == OrderSide.BUY
    assert order.status == OrderStatus.FILLED
    assert order.filled_qty == 100
    assert order.filled_price == 450.0
    assert broker.cash == 100000.0 - (100 * 450.0)  # $55,000


@pytest.mark.asyncio
async def test_place_market_order_sell():
    """Test placing a market sell order."""
    broker = PaperBroker(initial_capital=100000.0)
    broker.set_market_price("SPY", 450.0)

    # First buy
    await broker.place_order(
        symbol="SPY",
        qty=100,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
    )

    # Then sell
    broker.set_market_price("SPY", 455.0)
    order = await broker.place_order(
        symbol="SPY",
        qty=100,
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
    )

    assert order.status == OrderStatus.FILLED
    assert order.filled_price == 455.0
    # Cash: 100000 - 45000 + 45500 = 100500
    assert broker.cash == 100500.0


# Test limit order placement
@pytest.mark.asyncio
async def test_place_limit_order_buy():
    """Test placing a limit buy order that remains pending."""
    broker = PaperBroker()
    broker.set_market_price("SPY", 450.0)

    order = await broker.place_order(
        symbol="SPY",
        qty=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        limit_price=445.0,
    )

    assert order.status == OrderStatus.PENDING
    assert order.limit_price == 445.0
    assert order.filled_qty == 0


@pytest.mark.asyncio
async def test_limit_order_fill_when_price_crosses():
    """Test limit order fills when market price crosses limit."""
    broker = PaperBroker()
    broker.set_market_price("SPY", 450.0)

    order = await broker.place_order(
        symbol="SPY",
        qty=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        limit_price=445.0,
    )

    assert order.status == OrderStatus.PENDING

    # Price drops to limit
    broker.set_market_price("SPY", 445.0)

    # Check order was filled
    updated_order = await broker.get_order_status(order.order_id)
    assert updated_order.status == OrderStatus.FILLED
    assert updated_order.filled_price == 445.0


@pytest.mark.asyncio
async def test_limit_order_sell_fill():
    """Test limit sell order fills when price reaches limit."""
    broker = PaperBroker()
    broker.set_market_price("SPY", 450.0)

    # Buy first
    await broker.place_order(
        symbol="SPY",
        qty=100,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
    )

    # Place limit sell
    order = await broker.place_order(
        symbol="SPY",
        qty=100,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        limit_price=455.0,
    )

    assert order.status == OrderStatus.PENDING

    # Price rises to limit
    broker.set_market_price("SPY", 455.0)

    updated_order = await broker.get_order_status(order.order_id)
    assert updated_order.status == OrderStatus.FILLED


# Test stop order placement
@pytest.mark.asyncio
async def test_place_stop_order_buy():
    """Test placing a stop buy order."""
    broker = PaperBroker()
    broker.set_market_price("SPY", 450.0)

    order = await broker.place_order(
        symbol="SPY",
        qty=100,
        side=OrderSide.BUY,
        order_type=OrderType.STOP,
        stop_price=455.0,
    )

    assert order.status == OrderStatus.PENDING
    assert order.stop_price == 455.0


@pytest.mark.asyncio
async def test_stop_order_buy_triggers():
    """Test stop buy order triggers when price crosses stop."""
    broker = PaperBroker()
    broker.set_market_price("SPY", 450.0)

    order = await broker.place_order(
        symbol="SPY",
        qty=100,
        side=OrderSide.BUY,
        order_type=OrderType.STOP,
        stop_price=455.0,
    )

    # Price rises to stop
    broker.set_market_price("SPY", 455.0)

    updated_order = await broker.get_order_status(order.order_id)
    assert updated_order.status == OrderStatus.FILLED
    assert updated_order.filled_price == 455.0


@pytest.mark.asyncio
async def test_stop_order_sell_triggers():
    """Test stop sell order triggers when price crosses stop."""
    broker = PaperBroker()
    broker.set_market_price("SPY", 450.0)

    # Buy first
    await broker.place_order(
        symbol="SPY",
        qty=100,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
    )

    # Place stop sell
    order = await broker.place_order(
        symbol="SPY",
        qty=100,
        side=OrderSide.SELL,
        order_type=OrderType.STOP,
        stop_price=445.0,
    )

    # Price drops to stop
    broker.set_market_price("SPY", 445.0)

    updated_order = await broker.get_order_status(order.order_id)
    assert updated_order.status == OrderStatus.FILLED


# Test stop-limit orders
@pytest.mark.asyncio
async def test_stop_limit_order_triggers_and_fills():
    """Test stop-limit order triggers at stop and fills at limit."""
    broker = PaperBroker()
    broker.set_market_price("SPY", 450.0)

    order = await broker.place_order(
        symbol="SPY",
        qty=100,
        side=OrderSide.BUY,
        order_type=OrderType.STOP_LIMIT,
        stop_price=455.0,
        limit_price=456.0,
    )

    # Price rises to stop and within limit
    broker.set_market_price("SPY", 455.0)

    updated_order = await broker.get_order_status(order.order_id)
    assert updated_order.status == OrderStatus.FILLED


# Test order cancellation
@pytest.mark.asyncio
async def test_cancel_pending_order():
    """Test canceling a pending order."""
    broker = PaperBroker()
    broker.set_market_price("SPY", 450.0)

    order = await broker.place_order(
        symbol="SPY",
        qty=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        limit_price=445.0,
    )

    result = await broker.cancel_order(order.order_id)

    assert result is True
    updated_order = await broker.get_order_status(order.order_id)
    assert updated_order.status == OrderStatus.CANCELLED


@pytest.mark.asyncio
async def test_cannot_cancel_filled_order():
    """Test cannot cancel an already filled order."""
    broker = PaperBroker()
    broker.set_market_price("SPY", 450.0)

    order = await broker.place_order(
        symbol="SPY",
        qty=100,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
    )

    result = await broker.cancel_order(order.order_id)

    assert result is False


@pytest.mark.asyncio
async def test_cancel_nonexistent_order():
    """Test canceling a non-existent order returns False."""
    broker = PaperBroker()

    result = await broker.cancel_order("INVALID_ID")

    assert result is False


# Test position tracking
@pytest.mark.asyncio
async def test_position_created_after_buy():
    """Test position is created after a buy order."""
    broker = PaperBroker()
    broker.set_market_price("SPY", 450.0)

    await broker.place_order(
        symbol="SPY",
        qty=100,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
    )

    positions = await broker.get_positions()

    assert len(positions) == 1
    assert positions[0].symbol == "SPY"
    assert positions[0].qty == 100
    assert positions[0].entry_price == 450.0


@pytest.mark.asyncio
async def test_position_updated_on_additional_buy():
    """Test position averages entry price on additional buy."""
    broker = PaperBroker()
    broker.set_market_price("SPY", 450.0)

    # First buy
    await broker.place_order(
        symbol="SPY",
        qty=100,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
    )

    # Second buy at different price
    broker.set_market_price("SPY", 460.0)
    await broker.place_order(
        symbol="SPY",
        qty=100,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
    )

    positions = await broker.get_positions()

    assert len(positions) == 1
    assert positions[0].qty == 200
    # Average: (100*450 + 100*460) / 200 = 455
    assert positions[0].entry_price == 455.0


@pytest.mark.asyncio
async def test_position_closed_after_full_sell():
    """Test position is closed after selling all shares."""
    broker = PaperBroker()
    broker.set_market_price("SPY", 450.0)

    # Buy
    await broker.place_order(
        symbol="SPY",
        qty=100,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
    )

    # Sell all
    await broker.place_order(
        symbol="SPY",
        qty=100,
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
    )

    positions = await broker.get_positions()

    assert len(positions) == 0


@pytest.mark.asyncio
async def test_position_unrealized_pnl():
    """Test position unrealized P&L calculation."""
    broker = PaperBroker()
    broker.set_market_price("SPY", 450.0)

    await broker.place_order(
        symbol="SPY",
        qty=100,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
    )

    # Update price
    broker.set_market_price("SPY", 460.0)

    positions = await broker.get_positions()

    assert positions[0].current_price == 460.0
    assert positions[0].unrealized_pnl == (460.0 - 450.0) * 100  # $1,000


# Test account queries
@pytest.mark.asyncio
async def test_get_account_info():
    """Test getting account information."""
    broker = PaperBroker(initial_capital=100000.0)
    broker.set_market_price("SPY", 450.0)

    account = await broker.get_account()

    assert account.account_id == "PAPER_001"
    assert account.balance == 100000.0
    assert account.cash == 100000.0
    assert account.equity == 100000.0


@pytest.mark.asyncio
async def test_account_equity_with_positions():
    """Test account equity includes unrealized P&L."""
    broker = PaperBroker(initial_capital=100000.0)
    broker.set_market_price("SPY", 450.0)

    # Buy
    await broker.place_order(
        symbol="SPY",
        qty=100,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
    )

    # Price increases
    broker.set_market_price("SPY", 460.0)

    account = await broker.get_account()

    # Cash: 100000 - 45000 = 55000
    # Position value: 100 * 460 = 46000
    # Unrealized P&L: (460 - 450) * 100 = 1000
    # Equity: 55000 + 46000 + 1000 = 102000
    assert account.cash == 55000.0
    assert account.equity == 102000.0


# Test error handling
@pytest.mark.asyncio
async def test_market_order_without_price_raises_error():
    """Test market order without set price raises error."""
    broker = PaperBroker()

    with pytest.raises(ValueError, match="Market price not set"):
        await broker.place_order(
            symbol="SPY",
            qty=100,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
        )


@pytest.mark.asyncio
async def test_limit_order_without_limit_price_raises_error():
    """Test limit order without limit_price raises error."""
    broker = PaperBroker()
    broker.set_market_price("SPY", 450.0)

    with pytest.raises(ValueError, match="limit_price required"):
        await broker.place_order(
            symbol="SPY",
            qty=100,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
        )


@pytest.mark.asyncio
async def test_stop_order_without_stop_price_raises_error():
    """Test stop order without stop_price raises error."""
    broker = PaperBroker()
    broker.set_market_price("SPY", 450.0)

    with pytest.raises(ValueError, match="stop_price required"):
        await broker.place_order(
            symbol="SPY",
            qty=100,
            side=OrderSide.BUY,
            order_type=OrderType.STOP,
        )


@pytest.mark.asyncio
async def test_insufficient_capital_raises_error():
    """Test order with insufficient capital raises error."""
    broker = PaperBroker(initial_capital=1000.0)
    broker.set_market_price("SPY", 450.0)

    with pytest.raises(ValueError, match="Insufficient capital"):
        await broker.place_order(
            symbol="SPY",
            qty=100,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
        )


# Test order ID generation
@pytest.mark.asyncio
async def test_order_id_generation():
    """Test unique order ID generation."""
    broker = PaperBroker()
    broker.set_market_price("SPY", 450.0)

    order1 = await broker.place_order(
        symbol="SPY",
        qty=100,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
    )

    order2 = await broker.place_order(
        symbol="SPY",
        qty=100,
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
    )

    assert order1.order_id != order2.order_id
    assert "PAPER_" in order1.order_id
    assert "PAPER_" in order2.order_id


# Test get order status
@pytest.mark.asyncio
async def test_get_order_status_returns_none_for_invalid_id():
    """Test getting status of non-existent order returns None."""
    broker = PaperBroker()

    order = await broker.get_order_status("INVALID_ID")

    assert order is None


# Test multiple symbols
@pytest.mark.asyncio
async def test_multiple_symbol_positions():
    """Test tracking positions for multiple symbols."""
    broker = PaperBroker()
    broker.set_market_price("SPY", 450.0)
    broker.set_market_price("AAPL", 180.0)

    await broker.place_order(
        symbol="SPY",
        qty=100,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
    )

    await broker.place_order(
        symbol="AAPL",
        qty=50,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
    )

    positions = await broker.get_positions()

    assert len(positions) == 2
    symbols = {p.symbol for p in positions}
    assert symbols == {"SPY", "AAPL"}


# Test Position dataclass
def test_position_post_init_calculations():
    """Test Position calculates derived fields correctly."""
    position = Position(
        symbol="SPY",
        qty=100,
        entry_price=450.0,
        current_price=460.0,
    )

    assert position.market_value == 100 * 460.0  # 46000
    assert position.unrealized_pnl == (460.0 - 450.0) * 100  # 1000


# Test Order dataclass defaults
def test_order_default_values():
    """Test Order dataclass has correct defaults."""
    order = Order(
        order_id="TEST_001",
        symbol="SPY",
        qty=100,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
    )

    assert order.status == OrderStatus.PENDING
    assert order.filled_qty == 0
    assert order.filled_price is None
    assert order.limit_price is None
    assert order.stop_price is None
    assert order.created_at > 0
    assert order.updated_at > 0


# Success criteria tests (R10.1.1, R10.1.2)
@pytest.mark.asyncio
async def test_success_criteria_order_placement():
    """
    Success criteria: Submit 10 orders in 1 minute, all accepted.

    Requirements:
        - R10.1.1: place_order method works
        - R10.1.2: PaperBroker accepts orders
    """
    broker = PaperBroker()
    broker.set_market_price("SPY", 450.0)

    orders = []
    for i in range(10):
        order = await broker.place_order(
            symbol="SPY",
            qty=10,
            side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
            order_type=OrderType.LIMIT,
            limit_price=450.0 + i,
        )
        orders.append(order)

    assert len(orders) == 10
    assert all(order.status in (OrderStatus.PENDING, OrderStatus.FILLED) for order in orders)


@pytest.mark.asyncio
async def test_success_criteria_all_broker_methods():
    """
    Test all required broker interface methods are implemented.

    Requirements:
        - R10.1.1: All abstract methods implemented
    """
    broker = PaperBroker()
    broker.set_market_price("SPY", 450.0)

    # place_order
    order = await broker.place_order("SPY", 100, OrderSide.BUY, OrderType.MARKET)
    assert order is not None

    # get_order_status
    status = await broker.get_order_status(order.order_id)
    assert status is not None

    # cancel_order
    order2 = await broker.place_order("SPY", 100, OrderSide.BUY, OrderType.LIMIT, limit_price=440.0)
    cancel_result = await broker.cancel_order(order2.order_id)
    assert cancel_result is True

    # get_positions
    positions = await broker.get_positions()
    assert isinstance(positions, list)

    # get_account
    account = await broker.get_account()
    assert account is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
