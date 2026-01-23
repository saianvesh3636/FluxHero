"""
Unit tests for Alpaca broker adapter.

Tests cover:
- Connection lifecycle (connect, disconnect, health_check)
- Account and position retrieval
- Order placement, cancellation, and status
- Error handling and retry logic
- Response mapping from Alpaca format to unified models
"""

import json
import time
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from backend.execution.broker_base import (
    Account,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from backend.execution.brokers.alpaca_broker import AlpacaBroker


def create_mock_response(
    status_code: int = 200,
    json_data: dict | list | None = None,
) -> httpx.Response:
    """Create a mock httpx Response with a request attached."""
    content = json.dumps(json_data or {}).encode()
    # Create a mock request to attach to the response
    request = httpx.Request("GET", "https://paper-api.alpaca.markets/v2/account")
    response = httpx.Response(
        status_code=status_code,
        content=content,
        headers={"content-type": "application/json"},
        request=request,
    )
    return response


# Sample Alpaca API responses
SAMPLE_ACCOUNT = {
    "id": "account-id-123",
    "account_number": "PA123456",
    "status": "ACTIVE",
    "cash": "100000.00",
    "buying_power": "200000.00",
    "equity": "105000.00",
    "long_market_value": "5000.00",
    "short_market_value": "0.00",
}

SAMPLE_POSITIONS = [
    {
        "symbol": "SPY",
        "qty": "100",
        "side": "long",
        "avg_entry_price": "450.00",
        "current_price": "455.00",
        "unrealized_pl": "500.00",
        "market_value": "45500.00",
    },
    {
        "symbol": "AAPL",
        "qty": "50",
        "side": "long",
        "avg_entry_price": "175.00",
        "current_price": "180.00",
        "unrealized_pl": "250.00",
        "market_value": "9000.00",
    },
]

SAMPLE_ORDER_NEW = {
    "id": "order-123",
    "symbol": "SPY",
    "qty": "100",
    "side": "buy",
    "type": "market",
    "status": "new",
    "filled_qty": "0",
    "filled_avg_price": None,
    "limit_price": None,
    "stop_price": None,
}

SAMPLE_ORDER_FILLED = {
    "id": "order-456",
    "symbol": "SPY",
    "qty": "100",
    "side": "buy",
    "type": "market",
    "status": "filled",
    "filled_qty": "100",
    "filled_avg_price": "450.50",
    "limit_price": None,
    "stop_price": None,
}

SAMPLE_LIMIT_ORDER = {
    "id": "order-789",
    "symbol": "AAPL",
    "qty": "50",
    "side": "sell",
    "type": "limit",
    "status": "new",
    "filled_qty": "0",
    "filled_avg_price": None,
    "limit_price": "185.00",
    "stop_price": None,
}


class TestAlpacaBrokerInit:
    """Tests for AlpacaBroker initialization."""

    def test_init_with_credentials(self):
        """Test initialization with explicit credentials."""
        broker = AlpacaBroker(
            api_key="test-key",
            api_secret="test-secret",
            base_url="https://paper-api.alpaca.markets",
        )

        assert broker.api_key == "test-key"
        assert broker.api_secret == "test-secret"
        assert broker.base_url == "https://paper-api.alpaca.markets"

    def test_init_with_custom_timeout(self):
        """Test initialization with custom timeout."""
        broker = AlpacaBroker(
            api_key="test-key",
            api_secret="test-secret",
            timeout=60.0,
        )

        assert broker.timeout == 60.0

    def test_init_default_state(self):
        """Test initialization sets correct default state."""
        broker = AlpacaBroker(api_key="key", api_secret="secret")

        assert broker._client is None
        assert broker._connected is False
        assert broker._last_heartbeat is None


class TestAlpacaBrokerConnect:
    """Tests for AlpacaBroker.connect()."""

    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful connection with mocked HTTP client."""
        broker = AlpacaBroker(api_key="key", api_secret="secret")

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return create_mock_response(200, SAMPLE_ACCOUNT)

        # Mock httpx.AsyncClient to return our mocked client
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=create_mock_response(200, SAMPLE_ACCOUNT))
            mock_client.aclose = AsyncMock()
            mock_client_class.return_value = mock_client

            result = await broker.connect()

            assert result is True
            assert broker._connected is True
            assert broker._last_heartbeat is not None

        await broker.disconnect()

    @pytest.mark.asyncio
    async def test_connect_without_credentials(self):
        """Test connect fails without credentials."""
        broker = AlpacaBroker(api_key="", api_secret="")

        result = await broker.connect()

        assert result is False
        assert broker._connected is False

    @pytest.mark.asyncio
    async def test_connect_auth_failure(self):
        """Test connect fails on authentication error."""
        broker = AlpacaBroker(api_key="bad-key", api_secret="bad-secret")

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return create_mock_response(401, {"message": "Unauthorized"})

        # Mock the AsyncClient constructor
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(
                return_value=create_mock_response(401, {"message": "Unauthorized"})
            )
            mock_client.aclose = AsyncMock()
            mock_client_class.return_value = mock_client

            result = await broker.connect()

            assert result is False
            assert broker._connected is False


class TestAlpacaBrokerDisconnect:
    """Tests for AlpacaBroker.disconnect()."""

    @pytest.mark.asyncio
    async def test_disconnect_clears_state(self):
        """Test disconnect clears connection state."""
        broker = AlpacaBroker(api_key="key", api_secret="secret")
        broker._connected = True
        broker._last_heartbeat = time.time()
        broker._client = AsyncMock()
        broker._client.aclose = AsyncMock()

        await broker.disconnect()

        assert broker._connected is False
        assert broker._last_heartbeat is None
        assert broker._client is None

    @pytest.mark.asyncio
    async def test_disconnect_safe_when_not_connected(self):
        """Test disconnect is safe when not connected."""
        broker = AlpacaBroker(api_key="key", api_secret="secret")

        # Should not raise
        await broker.disconnect()

        assert broker._connected is False


class TestAlpacaBrokerHealthCheck:
    """Tests for AlpacaBroker.health_check()."""

    @pytest.mark.asyncio
    async def test_health_check_when_not_connected(self):
        """Test health check returns unhealthy when not connected."""
        broker = AlpacaBroker(api_key="key", api_secret="secret")

        health = await broker.health_check()

        assert health.is_connected is False
        assert health.is_authenticated is False
        assert health.is_healthy is False
        assert health.error_message == "Not connected"

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test health check returns healthy when connected."""
        broker = AlpacaBroker(api_key="key", api_secret="secret")

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return create_mock_response(200, SAMPLE_ACCOUNT)

        broker._client = httpx.AsyncClient(
            base_url="https://paper-api.alpaca.markets",
            transport=httpx.MockTransport(mock_handler),
        )
        broker._connected = True

        health = await broker.health_check()

        assert health.is_connected is True
        assert health.is_authenticated is True
        assert health.is_healthy is True
        assert health.latency_ms is not None
        assert health.latency_ms >= 0

        await broker.disconnect()

    @pytest.mark.asyncio
    async def test_health_check_account_not_active(self):
        """Test health check reports not authenticated for inactive account."""
        broker = AlpacaBroker(api_key="key", api_secret="secret")

        inactive_account = {**SAMPLE_ACCOUNT, "status": "INACTIVE"}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return create_mock_response(200, inactive_account)

        broker._client = httpx.AsyncClient(
            base_url="https://paper-api.alpaca.markets",
            transport=httpx.MockTransport(mock_handler),
        )
        broker._connected = True

        health = await broker.health_check()

        assert health.is_connected is True
        assert health.is_authenticated is False
        assert health.error_message == "Account not active"

        await broker.disconnect()


class TestAlpacaBrokerGetAccount:
    """Tests for AlpacaBroker.get_account()."""

    @pytest.mark.asyncio
    async def test_get_account_success(self):
        """Test get_account returns correct Account object."""
        broker = AlpacaBroker(api_key="key", api_secret="secret")

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return create_mock_response(200, SAMPLE_ACCOUNT)

        broker._client = httpx.AsyncClient(
            base_url="https://paper-api.alpaca.markets",
            transport=httpx.MockTransport(mock_handler),
        )
        broker._connected = True

        account = await broker.get_account()

        assert isinstance(account, Account)
        assert account.account_id == "PA123456"
        assert account.balance == 100000.0
        assert account.buying_power == 200000.0
        assert account.equity == 105000.0
        assert account.cash == 100000.0
        assert account.positions_value == 5000.0

        await broker.disconnect()

    @pytest.mark.asyncio
    async def test_get_account_not_connected(self):
        """Test get_account raises when not connected."""
        broker = AlpacaBroker(api_key="key", api_secret="secret")

        with pytest.raises(RuntimeError, match="Broker not connected"):
            await broker.get_account()


class TestAlpacaBrokerGetPositions:
    """Tests for AlpacaBroker.get_positions()."""

    @pytest.mark.asyncio
    async def test_get_positions_success(self):
        """Test get_positions returns correct Position objects."""
        broker = AlpacaBroker(api_key="key", api_secret="secret")

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return create_mock_response(200, SAMPLE_POSITIONS)

        broker._client = httpx.AsyncClient(
            base_url="https://paper-api.alpaca.markets",
            transport=httpx.MockTransport(mock_handler),
        )
        broker._connected = True

        positions = await broker.get_positions()

        assert len(positions) == 2

        spy_position = positions[0]
        assert isinstance(spy_position, Position)
        assert spy_position.symbol == "SPY"
        assert spy_position.qty == 100
        assert spy_position.entry_price == 450.0
        assert spy_position.current_price == 455.0
        assert spy_position.unrealized_pnl == 500.0
        assert spy_position.market_value == 45500.0

        aapl_position = positions[1]
        assert aapl_position.symbol == "AAPL"
        assert aapl_position.qty == 50

        await broker.disconnect()

    @pytest.mark.asyncio
    async def test_get_positions_empty(self):
        """Test get_positions returns empty list when no positions."""
        broker = AlpacaBroker(api_key="key", api_secret="secret")

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return create_mock_response(200, [])

        broker._client = httpx.AsyncClient(
            base_url="https://paper-api.alpaca.markets",
            transport=httpx.MockTransport(mock_handler),
        )
        broker._connected = True

        positions = await broker.get_positions()

        assert positions == []

        await broker.disconnect()

    @pytest.mark.asyncio
    async def test_get_positions_short_position(self):
        """Test get_positions handles short positions correctly."""
        broker = AlpacaBroker(api_key="key", api_secret="secret")

        short_position = [
            {
                "symbol": "TSLA",
                "qty": "50",
                "side": "short",
                "avg_entry_price": "200.00",
                "current_price": "190.00",
                "unrealized_pl": "500.00",
                "market_value": "-9500.00",
            }
        ]

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return create_mock_response(200, short_position)

        broker._client = httpx.AsyncClient(
            base_url="https://paper-api.alpaca.markets",
            transport=httpx.MockTransport(mock_handler),
        )
        broker._connected = True

        positions = await broker.get_positions()

        assert len(positions) == 1
        assert positions[0].qty == -50  # Negative for short

        await broker.disconnect()


class TestAlpacaBrokerPlaceOrder:
    """Tests for AlpacaBroker.place_order()."""

    @pytest.mark.asyncio
    async def test_place_market_order(self):
        """Test placing a market order."""
        broker = AlpacaBroker(api_key="key", api_secret="secret")

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return create_mock_response(200, SAMPLE_ORDER_NEW)

        broker._client = httpx.AsyncClient(
            base_url="https://paper-api.alpaca.markets",
            transport=httpx.MockTransport(mock_handler),
        )
        broker._connected = True

        order = await broker.place_order(
            symbol="SPY",
            qty=100,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
        )

        assert isinstance(order, Order)
        assert order.order_id == "order-123"
        assert order.symbol == "SPY"
        assert order.qty == 100
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.status == OrderStatus.PENDING

        await broker.disconnect()

    @pytest.mark.asyncio
    async def test_place_limit_order(self):
        """Test placing a limit order."""
        broker = AlpacaBroker(api_key="key", api_secret="secret")

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return create_mock_response(200, SAMPLE_LIMIT_ORDER)

        broker._client = httpx.AsyncClient(
            base_url="https://paper-api.alpaca.markets",
            transport=httpx.MockTransport(mock_handler),
        )
        broker._connected = True

        order = await broker.place_order(
            symbol="AAPL",
            qty=50,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            limit_price=185.0,
        )

        assert order.order_type == OrderType.LIMIT
        assert order.limit_price == 185.0
        assert order.side == OrderSide.SELL

        await broker.disconnect()

    @pytest.mark.asyncio
    async def test_place_limit_order_without_price_raises(self):
        """Test placing limit order without price raises ValueError."""
        broker = AlpacaBroker(api_key="key", api_secret="secret")
        broker._client = AsyncMock()
        broker._connected = True

        with pytest.raises(ValueError, match="limit_price required"):
            await broker.place_order(
                symbol="SPY",
                qty=100,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
            )

        await broker.disconnect()

    @pytest.mark.asyncio
    async def test_place_stop_order_without_price_raises(self):
        """Test placing stop order without price raises ValueError."""
        broker = AlpacaBroker(api_key="key", api_secret="secret")
        broker._client = AsyncMock()
        broker._connected = True

        with pytest.raises(ValueError, match="stop_price required"):
            await broker.place_order(
                symbol="SPY",
                qty=100,
                side=OrderSide.BUY,
                order_type=OrderType.STOP,
            )

        await broker.disconnect()


class TestAlpacaBrokerCancelOrder:
    """Tests for AlpacaBroker.cancel_order()."""

    @pytest.mark.asyncio
    async def test_cancel_order_success(self):
        """Test successful order cancellation."""
        broker = AlpacaBroker(api_key="key", api_secret="secret")

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            if request.method == "DELETE":
                return create_mock_response(204, None)
            return create_mock_response(404, {})

        broker._client = httpx.AsyncClient(
            base_url="https://paper-api.alpaca.markets",
            transport=httpx.MockTransport(mock_handler),
        )
        broker._connected = True

        result = await broker.cancel_order("order-123")

        assert result is True

        await broker.disconnect()

    @pytest.mark.asyncio
    async def test_cancel_order_not_found(self):
        """Test cancelling non-existent order returns False."""
        broker = AlpacaBroker(api_key="key", api_secret="secret")

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return create_mock_response(404, {"message": "Order not found"})

        broker._client = httpx.AsyncClient(
            base_url="https://paper-api.alpaca.markets",
            transport=httpx.MockTransport(mock_handler),
        )
        broker._connected = True

        result = await broker.cancel_order("non-existent")

        assert result is False

        await broker.disconnect()

    @pytest.mark.asyncio
    async def test_cancel_order_already_filled(self):
        """Test cancelling already filled order returns False."""
        broker = AlpacaBroker(api_key="key", api_secret="secret")

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return create_mock_response(422, {"message": "Order is not cancelable"})

        broker._client = httpx.AsyncClient(
            base_url="https://paper-api.alpaca.markets",
            transport=httpx.MockTransport(mock_handler),
        )
        broker._connected = True

        result = await broker.cancel_order("filled-order")

        assert result is False

        await broker.disconnect()


class TestAlpacaBrokerGetOrderStatus:
    """Tests for AlpacaBroker.get_order_status()."""

    @pytest.mark.asyncio
    async def test_get_order_status_success(self):
        """Test getting order status successfully."""
        broker = AlpacaBroker(api_key="key", api_secret="secret")

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return create_mock_response(200, SAMPLE_ORDER_FILLED)

        broker._client = httpx.AsyncClient(
            base_url="https://paper-api.alpaca.markets",
            transport=httpx.MockTransport(mock_handler),
        )
        broker._connected = True

        order = await broker.get_order_status("order-456")

        assert isinstance(order, Order)
        assert order.order_id == "order-456"
        assert order.status == OrderStatus.FILLED
        assert order.filled_qty == 100
        assert order.filled_price == 450.50

        await broker.disconnect()

    @pytest.mark.asyncio
    async def test_get_order_status_not_found(self):
        """Test getting status of non-existent order returns None."""
        broker = AlpacaBroker(api_key="key", api_secret="secret")

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return create_mock_response(404, {"message": "Order not found"})

        broker._client = httpx.AsyncClient(
            base_url="https://paper-api.alpaca.markets",
            transport=httpx.MockTransport(mock_handler),
        )
        broker._connected = True

        order = await broker.get_order_status("non-existent")

        assert order is None

        await broker.disconnect()


class TestAlpacaBrokerStatusMapping:
    """Tests for Alpaca status mapping."""

    def test_map_alpaca_status_new(self):
        """Test mapping 'new' status."""
        broker = AlpacaBroker(api_key="key", api_secret="secret")

        assert broker._map_alpaca_status("new") == OrderStatus.PENDING
        assert broker._map_alpaca_status("accepted") == OrderStatus.PENDING
        assert broker._map_alpaca_status("pending_new") == OrderStatus.PENDING

    def test_map_alpaca_status_filled(self):
        """Test mapping 'filled' status."""
        broker = AlpacaBroker(api_key="key", api_secret="secret")

        assert broker._map_alpaca_status("filled") == OrderStatus.FILLED

    def test_map_alpaca_status_partially_filled(self):
        """Test mapping 'partially_filled' status."""
        broker = AlpacaBroker(api_key="key", api_secret="secret")

        assert broker._map_alpaca_status("partially_filled") == OrderStatus.PARTIALLY_FILLED

    def test_map_alpaca_status_cancelled(self):
        """Test mapping cancelled statuses."""
        broker = AlpacaBroker(api_key="key", api_secret="secret")

        assert broker._map_alpaca_status("canceled") == OrderStatus.CANCELLED
        assert broker._map_alpaca_status("cancelled") == OrderStatus.CANCELLED
        assert broker._map_alpaca_status("expired") == OrderStatus.CANCELLED

    def test_map_alpaca_status_rejected(self):
        """Test mapping 'rejected' status."""
        broker = AlpacaBroker(api_key="key", api_secret="secret")

        assert broker._map_alpaca_status("rejected") == OrderStatus.REJECTED

    def test_map_alpaca_status_unknown(self):
        """Test mapping unknown status defaults to PENDING."""
        broker = AlpacaBroker(api_key="key", api_secret="secret")

        assert broker._map_alpaca_status("unknown_status") == OrderStatus.PENDING


class TestAlpacaBrokerOrderTypeMapping:
    """Tests for order type mapping."""

    def test_map_order_side(self):
        """Test mapping OrderSide to Alpaca string."""
        broker = AlpacaBroker(api_key="key", api_secret="secret")

        assert broker._map_order_side(OrderSide.BUY) == "buy"
        assert broker._map_order_side(OrderSide.SELL) == "sell"

    def test_map_order_type(self):
        """Test mapping OrderType to Alpaca string."""
        broker = AlpacaBroker(api_key="key", api_secret="secret")

        assert broker._map_order_type(OrderType.MARKET) == "market"
        assert broker._map_order_type(OrderType.LIMIT) == "limit"
        assert broker._map_order_type(OrderType.STOP) == "stop"
        assert broker._map_order_type(OrderType.STOP_LIMIT) == "stop_limit"

    def test_parse_order_type(self):
        """Test parsing Alpaca type string to OrderType."""
        broker = AlpacaBroker(api_key="key", api_secret="secret")

        assert broker._parse_order_type("market") == OrderType.MARKET
        assert broker._parse_order_type("limit") == OrderType.LIMIT
        assert broker._parse_order_type("stop") == OrderType.STOP
        assert broker._parse_order_type("stop_limit") == OrderType.STOP_LIMIT
        assert broker._parse_order_type("MARKET") == OrderType.MARKET  # Case insensitive


class TestAlpacaBrokerHeaders:
    """Tests for request headers."""

    def test_get_headers(self):
        """Test authentication headers are correct."""
        broker = AlpacaBroker(
            api_key="my-api-key",
            api_secret="my-api-secret",
        )

        headers = broker._get_headers()

        assert headers["APCA-API-KEY-ID"] == "my-api-key"
        assert headers["APCA-API-SECRET-KEY"] == "my-api-secret"
        assert headers["Content-Type"] == "application/json"


class TestAlpacaBrokerImports:
    """Tests for module imports and exports."""

    def test_import_from_brokers_package(self):
        """Test AlpacaBroker can be imported from brokers package."""
        from backend.execution.brokers import AlpacaBroker as ImportedBroker

        assert ImportedBroker is AlpacaBroker

    def test_alpaca_broker_implements_interface(self):
        """Test AlpacaBroker is a BrokerInterface."""
        from backend.execution.broker_base import BrokerInterface

        broker = AlpacaBroker(api_key="key", api_secret="secret")

        assert isinstance(broker, BrokerInterface)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
