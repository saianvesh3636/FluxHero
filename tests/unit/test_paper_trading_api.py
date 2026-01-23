"""
Unit tests for Paper Trading API Endpoints

Tests the paper trading account management and trade history:
- GET /api/paper/account - Get paper account info
- POST /api/paper/reset - Reset paper account
- GET /api/paper/trades - Get paper trade history

Feature: Paper Trading System (Phase B)
"""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.api.server import app, app_state
from backend.storage.sqlite_store import SQLiteStore


@pytest.fixture
def test_db_path(tmp_path):
    """Create a temporary test database path"""
    db_path = tmp_path / "test_paper.db"
    return str(db_path)


@pytest.fixture
def client(test_db_path):
    """Create a test client with properly initialized database"""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def test_lifespan(app):
        # Create and initialize store within the lifespan's event loop
        store = SQLiteStore(db_path=test_db_path)
        await store.initialize()
        app_state.sqlite_store = store
        app_state.data_feed_active = False
        app_state.start_time = datetime.now()
        app_state.last_update = datetime.now()

        # Reset the global paper broker for each test
        import backend.api.server as server_module

        server_module._paper_broker = None

        yield
        app_state.websocket_clients.clear()
        await store.close()

        # Clean up paper broker
        if server_module._paper_broker is not None:
            try:
                await server_module._paper_broker.disconnect()
            except Exception:
                pass
            server_module._paper_broker = None

    app.router.lifespan_context = test_lifespan

    with TestClient(app) as test_client:
        yield test_client


# ============================================================================
# GET /api/paper/account Tests
# ============================================================================


def test_get_paper_account_initial(client):
    """Test getting paper account returns initial state"""
    response = client.get("/api/paper/account")
    assert response.status_code == 200

    data = response.json()
    assert data["account_id"] == "PAPER-001"
    assert data["balance"] == 100_000.0
    assert data["buying_power"] == 100_000.0
    assert data["equity"] == 100_000.0
    assert data["cash"] == 100_000.0
    assert data["positions_value"] == 0.0
    assert data["realized_pnl"] == 0.0
    assert data["unrealized_pnl"] == 0.0
    assert data["positions"] == []


def test_get_paper_account_response_structure(client):
    """Test paper account response has all required fields"""
    response = client.get("/api/paper/account")
    assert response.status_code == 200

    data = response.json()
    required_fields = [
        "account_id",
        "balance",
        "buying_power",
        "equity",
        "cash",
        "positions_value",
        "realized_pnl",
        "unrealized_pnl",
        "positions",
    ]

    for field in required_fields:
        assert field in data, f"Missing required field: {field}"


def test_get_paper_account_multiple_calls(client):
    """Test multiple calls return consistent data"""
    response1 = client.get("/api/paper/account")
    response2 = client.get("/api/paper/account")

    assert response1.status_code == 200
    assert response2.status_code == 200
    assert response1.json() == response2.json()


# ============================================================================
# POST /api/paper/reset Tests
# ============================================================================


def test_reset_paper_account(client):
    """Test resetting paper account"""
    response = client.post("/api/paper/reset")
    assert response.status_code == 200

    data = response.json()
    assert data["message"] == "Paper account reset successfully"
    assert data["account_id"] == "PAPER-001"
    assert data["initial_balance"] == 100_000.0
    assert "timestamp" in data


def test_reset_paper_account_response_structure(client):
    """Test reset response has all required fields"""
    response = client.post("/api/paper/reset")
    assert response.status_code == 200

    data = response.json()
    required_fields = ["message", "account_id", "initial_balance", "timestamp"]

    for field in required_fields:
        assert field in data, f"Missing required field: {field}"


def test_reset_paper_account_timestamp_format(client):
    """Test reset timestamp is valid ISO format"""
    response = client.post("/api/paper/reset")
    assert response.status_code == 200

    data = response.json()
    # Should be parseable as ISO datetime
    try:
        datetime.fromisoformat(data["timestamp"])
    except ValueError:
        pytest.fail("Timestamp is not valid ISO format")


def test_reset_restores_initial_balance(client):
    """Test reset actually restores initial balance"""
    # Get initial state
    response1 = client.get("/api/paper/account")
    initial_balance = response1.json()["balance"]

    # Reset
    client.post("/api/paper/reset")

    # Verify balance restored
    response2 = client.get("/api/paper/account")
    assert response2.json()["balance"] == initial_balance


# ============================================================================
# GET /api/paper/trades Tests
# ============================================================================


def test_get_paper_trades_empty(client):
    """Test getting trades when none exist"""
    response = client.get("/api/paper/trades")
    assert response.status_code == 200

    data = response.json()
    assert data["trades"] == []
    assert data["total_count"] == 0


def test_get_paper_trades_response_structure(client):
    """Test trades response has all required fields"""
    response = client.get("/api/paper/trades")
    assert response.status_code == 200

    data = response.json()
    required_fields = ["trades", "total_count"]

    for field in required_fields:
        assert field in data, f"Missing required field: {field}"


# ============================================================================
# Integration Tests - Account with Trades
# ============================================================================


@patch("backend.api.server._get_paper_broker")
def test_get_paper_account_with_positions(mock_get_broker, client):
    """Test paper account with open positions"""
    from backend.execution.broker_base import Account, Position

    # Mock paper broker with positions
    mock_broker = MagicMock()

    mock_account = Account(
        account_id="PAPER-001",
        balance=50_000.0,
        buying_power=50_000.0,
        equity=95_000.0,
        cash=50_000.0,
        positions_value=45_000.0,
    )

    mock_positions = [
        Position(
            symbol="SPY",
            qty=100,
            entry_price=450.0,
            current_price=450.0,
            unrealized_pnl=0.0,
            market_value=45_000.0,
        )
    ]

    mock_broker.get_account = AsyncMock(return_value=mock_account)
    mock_broker.get_positions = AsyncMock(return_value=mock_positions)
    mock_broker.get_realized_pnl = AsyncMock(return_value=500.0)
    mock_broker.get_unrealized_pnl = AsyncMock(return_value=0.0)

    mock_get_broker.return_value = mock_broker

    response = client.get("/api/paper/account")
    assert response.status_code == 200

    data = response.json()
    assert data["balance"] == 50_000.0
    assert data["positions_value"] == 45_000.0
    assert data["realized_pnl"] == 500.0
    assert len(data["positions"]) == 1
    assert data["positions"][0]["symbol"] == "SPY"
    assert data["positions"][0]["qty"] == 100


@patch("backend.api.server._get_paper_broker")
def test_get_paper_trades_with_history(mock_get_broker, client):
    """Test paper trades with trade history"""
    from backend.execution.broker_base import OrderSide
    from backend.execution.brokers.paper_broker import PaperTrade

    mock_broker = MagicMock()

    mock_trades = [
        PaperTrade(
            trade_id="TRADE-001",
            order_id="ORDER-001",
            symbol="SPY",
            side=OrderSide.BUY,
            qty=100,
            price=450.0,
            slippage=0.0225,
            timestamp=datetime.now().timestamp(),
            realized_pnl=0.0,
        ),
        PaperTrade(
            trade_id="TRADE-002",
            order_id="ORDER-002",
            symbol="SPY",
            side=OrderSide.SELL,
            qty=100,
            price=460.0,
            slippage=0.023,
            timestamp=datetime.now().timestamp(),
            realized_pnl=1000.0,
        ),
    ]

    mock_broker.get_trades = AsyncMock(return_value=mock_trades)
    mock_get_broker.return_value = mock_broker

    response = client.get("/api/paper/trades")
    assert response.status_code == 200

    data = response.json()
    assert data["total_count"] == 2
    assert len(data["trades"]) == 2

    # Verify first trade
    trade1 = data["trades"][0]
    assert trade1["trade_id"] == "TRADE-001"
    assert trade1["symbol"] == "SPY"
    assert trade1["side"] == "BUY"
    assert trade1["qty"] == 100
    assert trade1["price"] == 450.0

    # Verify second trade
    trade2 = data["trades"][1]
    assert trade2["side"] == "SELL"
    assert trade2["realized_pnl"] == 1000.0


@patch("backend.api.server._get_paper_broker")
def test_reset_clears_positions_and_trades(mock_get_broker, client):
    """Test reset clears positions and trades"""
    mock_broker = MagicMock()
    mock_broker.reset_account = AsyncMock()
    mock_broker.initial_balance = 100_000.0

    mock_get_broker.return_value = mock_broker

    response = client.post("/api/paper/reset")
    assert response.status_code == 200

    # Verify reset_account was called
    mock_broker.reset_account.assert_called_once()


# ============================================================================
# Error Handling Tests
# ============================================================================


@patch("backend.api.server._get_paper_broker")
def test_get_paper_account_broker_error(mock_get_broker, client):
    """Test paper account handles broker errors"""
    mock_get_broker.side_effect = Exception("Broker connection failed")

    response = client.get("/api/paper/account")
    assert response.status_code == 500
    assert "Failed to get paper account" in response.json()["detail"]


@patch("backend.api.server._get_paper_broker")
def test_get_paper_trades_broker_error(mock_get_broker, client):
    """Test paper trades handles broker errors"""
    mock_get_broker.side_effect = Exception("Broker connection failed")

    response = client.get("/api/paper/trades")
    assert response.status_code == 500
    assert "Failed to get paper trades" in response.json()["detail"]


@patch("backend.api.server._get_paper_broker")
def test_reset_paper_account_broker_error(mock_get_broker, client):
    """Test paper reset handles broker errors"""
    mock_get_broker.side_effect = Exception("Broker connection failed")

    response = client.post("/api/paper/reset")
    assert response.status_code == 500
    assert "Failed to reset paper account" in response.json()["detail"]


# ============================================================================
# Response Model Tests
# ============================================================================


def test_paper_account_response_model():
    """Test PaperAccountResponse model structure"""
    from backend.api.server import PaperAccountResponse, PaperPositionResponse

    position = PaperPositionResponse(
        symbol="SPY",
        qty=100,
        entry_price=450.0,
        current_price=455.0,
        market_value=45500.0,
        unrealized_pnl=500.0,
        cost_basis=45000.0,
    )

    response = PaperAccountResponse(
        account_id="PAPER-001",
        balance=55_000.0,
        buying_power=55_000.0,
        equity=100_500.0,
        cash=55_000.0,
        positions_value=45_500.0,
        realized_pnl=0.0,
        unrealized_pnl=500.0,
        positions=[position],
    )

    assert response.account_id == "PAPER-001"
    assert len(response.positions) == 1
    assert response.positions[0].symbol == "SPY"


def test_paper_trade_response_model():
    """Test PaperTradeResponse model structure"""
    from backend.api.server import PaperTradeResponse

    response = PaperTradeResponse(
        trade_id="TRADE-001",
        order_id="ORDER-001",
        symbol="SPY",
        side="BUY",
        qty=100,
        price=450.0,
        slippage=0.0225,
        timestamp="2024-01-01T12:00:00",
        realized_pnl=0.0,
    )

    assert response.trade_id == "TRADE-001"
    assert response.side == "BUY"
    assert response.price == 450.0


def test_paper_trade_history_response_model():
    """Test PaperTradeHistoryResponse model structure"""
    from backend.api.server import PaperTradeHistoryResponse, PaperTradeResponse

    trade = PaperTradeResponse(
        trade_id="TRADE-001",
        order_id="ORDER-001",
        symbol="SPY",
        side="BUY",
        qty=100,
        price=450.0,
        slippage=0.0225,
        timestamp="2024-01-01T12:00:00",
        realized_pnl=0.0,
    )

    response = PaperTradeHistoryResponse(trades=[trade], total_count=1)

    assert response.total_count == 1
    assert len(response.trades) == 1


def test_paper_reset_response_model():
    """Test PaperResetResponse model structure"""
    from backend.api.server import PaperResetResponse

    response = PaperResetResponse(
        message="Paper account reset successfully",
        account_id="PAPER-001",
        initial_balance=100_000.0,
        timestamp="2024-01-01T12:00:00",
    )

    assert response.message == "Paper account reset successfully"
    assert response.initial_balance == 100_000.0


def test_paper_position_response_model():
    """Test PaperPositionResponse model structure"""
    from backend.api.server import PaperPositionResponse

    response = PaperPositionResponse(
        symbol="AAPL",
        qty=50,
        entry_price=175.0,
        current_price=180.0,
        market_value=9000.0,
        unrealized_pnl=250.0,
        cost_basis=8750.0,
    )

    assert response.symbol == "AAPL"
    assert response.qty == 50
    assert response.unrealized_pnl == 250.0


# ============================================================================
# Consistency Tests
# ============================================================================


def test_paper_endpoints_use_same_broker_instance(client):
    """Test all paper endpoints use the same broker instance"""
    # Make multiple calls to different endpoints
    client.get("/api/paper/account")
    client.get("/api/paper/trades")
    client.get("/api/paper/account")

    # All should succeed without errors
    response = client.get("/api/paper/account")
    assert response.status_code == 200


def test_reset_affects_subsequent_account_calls(client):
    """Test reset affects subsequent account calls"""
    # Get account
    response1 = client.get("/api/paper/account")
    assert response1.status_code == 200

    # Reset
    reset_response = client.post("/api/paper/reset")
    assert reset_response.status_code == 200

    # Get account again - should reflect reset
    response2 = client.get("/api/paper/account")
    assert response2.status_code == 200

    # Both should show initial balance
    assert response2.json()["balance"] == 100_000.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
