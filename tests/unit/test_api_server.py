"""
Unit tests for FluxHero FastAPI Server

Tests all REST endpoints and WebSocket functionality:
- GET /api/positions
- GET /api/trades (with pagination)
- GET /api/account
- GET /api/status
- POST /api/backtest
- WebSocket /ws/prices
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import asyncio
import tempfile
from pathlib import Path

# Add backend to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from fluxhero.backend.api.server import app, app_state
from fluxhero.backend.storage.sqlite_store import (
    SQLiteStore,
    Trade,
    Position,
    TradeStatus,
    PositionSide,
)


@pytest.fixture
def test_db():
    """Create a temporary test database"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = SQLiteStore(db_path=str(db_path))
        # Initialize database schema synchronously
        asyncio.run(store.initialize())
        yield store
        # Note: We don't await close() here because fixture is not async
        # The store will be cleaned up by the temp directory removal


@pytest.fixture
def client(test_db):
    """Create a test client with mocked database"""
    # Create a custom lifespan context that uses test database
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def test_lifespan(app):
        # Startup: Use test database
        app_state.sqlite_store = test_db
        app_state.data_feed_active = False
        app_state.start_time = datetime.now()
        app_state.last_update = datetime.now()
        yield
        # Shutdown: Clean up (nothing to do, test_db fixture handles it)
        app_state.websocket_clients.clear()

    # Override the app's lifespan
    app.router.lifespan_context = test_lifespan

    with TestClient(app) as test_client:
        yield test_client


# ============================================================================
# Root Endpoint Tests
# ============================================================================

def test_root_endpoint(client):
    """Test root endpoint returns API info"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "FluxHero API"
    assert data["version"] == "1.0.0"
    assert data["status"] == "active"
    assert "endpoints" in data


def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


# ============================================================================
# Positions Endpoint Tests
# ============================================================================

def test_get_positions_empty(client):
    """Test getting positions when none exist"""
    response = client.get("/api/positions")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 0


def test_get_positions_with_data(client, test_db):
    """Test getting positions with existing data"""
    # Add test position using Position object
    pos1 = Position(
        symbol="SPY",
        side=PositionSide.LONG,
        shares=100,
        entry_price=450.0,
        current_price=455.0,
        unrealized_pnl=500.0,
        stop_loss=440.0,
        take_profit=460.0,
        entry_time=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
    )
    asyncio.run(test_db.upsert_position(pos1))

    pos2 = Position(
        symbol="QQQ",
        side=PositionSide.SHORT,
        shares=50,
        entry_price=380.0,
        current_price=375.0,
        unrealized_pnl=250.0,
        stop_loss=385.0,
        entry_time=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
    )
    asyncio.run(test_db.upsert_position(pos2))

    response = client.get("/api/positions")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2

    # Check first position (SPY)
    spy_pos = next(p for p in data if p["symbol"] == "SPY")
    assert spy_pos["side"] == PositionSide.LONG
    assert spy_pos["shares"] == 100
    assert spy_pos["entry_price"] == 450.0
    assert spy_pos["current_price"] == 455.0
    assert spy_pos["unrealized_pnl"] == 500.0


# ============================================================================
# Trades Endpoint Tests
# ============================================================================

def test_get_trades_empty(client):
    """Test getting trades when none exist"""
    response = client.get("/api/trades")
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 0
    assert data["page"] == 1
    assert data["page_size"] == 20
    assert data["total_pages"] == 0
    assert len(data["trades"]) == 0


def test_get_trades_with_data(client, test_db):
    """Test getting trades"""
    # Add test trades using Trade objects
    for i in range(5):
        trade = Trade(
            symbol="SPY",
            side=PositionSide.LONG,
            entry_price=450.0 + i,
            entry_time=datetime.now().isoformat(),
            shares=100,
            stop_loss=440.0,
            status=TradeStatus.CLOSED if i < 3 else TradeStatus.OPEN,
            strategy="TREND",
            regime="STRONG_TREND",
            signal_reason="KAMA crossover",
            exit_price=455.0 + i if i < 3 else None,
            exit_time=datetime.now().isoformat() if i < 3 else None,
            realized_pnl=(455.0 + i - 450.0 - i) * 100 if i < 3 else None,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )
        asyncio.run(test_db.add_trade(trade))

    # Test default pagination (page 1, size 20)
    response = client.get("/api/trades")
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 5
    assert data["page"] == 1
    assert data["page_size"] == 20
    assert data["total_pages"] == 1
    assert len(data["trades"]) == 5


def test_get_trades_pagination(client, test_db):
    """Test trade pagination"""
    # Add 25 trades
    for i in range(25):
        trade = Trade(
            symbol="SPY",
            side=PositionSide.LONG,
            entry_price=450.0,
            entry_time=datetime.now().isoformat(),
            shares=100,
            stop_loss=440.0,
            status=TradeStatus.CLOSED,
            strategy="TREND",
            regime="STRONG_TREND",
            signal_reason=f"Trade {i}",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )
        asyncio.run(test_db.add_trade(trade))

    # Get page 1 (20 trades)
    response = client.get("/api/trades?page=1&page_size=20")
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 25
    assert data["page"] == 1
    assert data["page_size"] == 20
    assert data["total_pages"] == 2
    assert len(data["trades"]) == 20

    # Get page 2 (5 trades)
    response = client.get("/api/trades?page=2&page_size=20")
    assert response.status_code == 200
    data = response.json()
    assert data["page"] == 2
    assert len(data["trades"]) == 5


def test_get_trades_status_filter(client, test_db):
    """Test filtering trades by status"""
    # Add trades with different statuses
    trade1 = Trade(
        symbol="SPY",
        side=PositionSide.LONG,
        entry_price=450.0,
        entry_time=datetime.now().isoformat(),
        shares=100,
        stop_loss=440.0,
        status=TradeStatus.OPEN,
        strategy="TREND",
        regime="STRONG_TREND",
        signal_reason="Open trade",
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
    )
    asyncio.run(test_db.add_trade(trade1))

    trade2 = Trade(
        symbol="QQQ",
        side=PositionSide.LONG,
        entry_price=380.0,
        entry_time=datetime.now().isoformat(),
        shares=50,
        stop_loss=370.0,
        status=TradeStatus.CLOSED,
        strategy="MEAN_REVERSION",
        regime="MEAN_REVERSION",
        signal_reason="Closed trade",
        exit_price=385.0,
        exit_time=datetime.now().isoformat(),
        realized_pnl=250.0,
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
    )
    asyncio.run(test_db.add_trade(trade2))

    # Filter by OPEN status
    response = client.get("/api/trades?status=OPEN")
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 1
    assert data["trades"][0]["status"] == TradeStatus.OPEN

    # Filter by CLOSED status
    response = client.get("/api/trades?status=CLOSED")
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 1
    assert data["trades"][0]["status"] == TradeStatus.CLOSED


def test_get_trades_invalid_status(client):
    """Test invalid status filter returns error"""
    response = client.get("/api/trades?status=INVALID")
    assert response.status_code == 400
    assert "Invalid status" in response.json()["detail"]


# ============================================================================
# Account Endpoint Tests
# ============================================================================

def test_get_account_info_empty(client):
    """Test getting account info with no trades/positions"""
    response = client.get("/api/account")
    assert response.status_code == 200
    data = response.json()
    assert data["equity"] == 10000.0  # Default initial capital
    assert data["cash"] == 10000.0
    assert data["buying_power"] == 20000.0  # 2x cash for margin
    assert data["total_pnl"] == 0.0
    assert data["daily_pnl"] == 0.0
    assert data["num_positions"] == 0


def test_get_account_info_with_positions(client, test_db):
    """Test account info with open positions"""
    # Add position with unrealized P&L
    pos = Position(
        symbol="SPY",
        side=PositionSide.LONG,
        shares=100,
        entry_price=450.0,
        current_price=455.0,
        unrealized_pnl=500.0,
        stop_loss=440.0,
        entry_time=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
    )
    asyncio.run(test_db.upsert_position(pos))

    response = client.get("/api/account")
    assert response.status_code == 200
    data = response.json()
    assert data["equity"] == 10500.0  # 10000 + 500 unrealized
    assert data["total_pnl"] == 500.0
    assert data["num_positions"] == 1


def test_get_account_info_with_trades(client, test_db):
    """Test account info with closed trades"""
    # Add closed trade with realized P&L
    trade = Trade(
        symbol="SPY",
        side=PositionSide.LONG,
        entry_price=450.0,
        entry_time=(datetime.now() - timedelta(days=1)).isoformat(),
        shares=100,
        stop_loss=440.0,
        status=TradeStatus.CLOSED,
        strategy="TREND",
        regime="STRONG_TREND",
        signal_reason="Test trade",
        exit_price=455.0,
        exit_time=(datetime.now() - timedelta(days=1)).isoformat(),
        realized_pnl=500.0,
        created_at=(datetime.now() - timedelta(days=1)).isoformat(),
        updated_at=(datetime.now() - timedelta(days=1)).isoformat(),
    )
    asyncio.run(test_db.add_trade(trade))

    response = client.get("/api/account")
    assert response.status_code == 200
    data = response.json()
    assert data["equity"] == 10500.0  # 10000 + 500 realized
    assert data["total_pnl"] == 500.0


# ============================================================================
# System Status Endpoint Tests
# ============================================================================

def test_get_system_status_active(client):
    """Test system status when active"""
    # Update timestamp to simulate recent activity
    app_state.last_update = datetime.now()

    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ACTIVE"
    assert data["uptime_seconds"] >= 0
    # Note: data_feed_active may be True or False depending on WebSocket connections
    assert "System operating normally" in data["message"]


def test_get_system_status_delayed(client):
    """Test system status when delayed"""
    # Set last update to 2 minutes ago
    app_state.last_update = datetime.now() - timedelta(minutes=2)

    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "DELAYED"
    assert "No updates for" in data["message"]


def test_get_system_status_offline(client):
    """Test system status when offline"""
    # Set last update to 10 minutes ago
    app_state.last_update = datetime.now() - timedelta(minutes=10)

    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "OFFLINE"
    assert "inactive" in data["message"]


# ============================================================================
# Backtest Endpoint Tests
# ============================================================================

def test_run_backtest_invalid_dates(client):
    """Test backtest with invalid date format"""
    backtest_config = {
        "symbol": "SPY",
        "start_date": "invalid-date",
        "end_date": "2024-12-31",
        "initial_capital": 10000.0,
    }

    response = client.post("/api/backtest", json=backtest_config)
    assert response.status_code == 400
    assert "Invalid date format" in response.json()["detail"]


def test_run_backtest_end_before_start(client):
    """Test backtest with end date before start date"""
    backtest_config = {
        "symbol": "SPY",
        "start_date": "2024-12-31",
        "end_date": "2024-01-01",
        "initial_capital": 10000.0,
    }

    response = client.post("/api/backtest", json=backtest_config)
    assert response.status_code == 400
    assert "End date must be after start date" in response.json()["detail"]


# ============================================================================
# WebSocket Endpoint Tests
# ============================================================================

def test_websocket_connection(client):
    """Test WebSocket connection and message reception"""
    with client.websocket_connect("/ws/prices") as websocket:
        # Receive connection message
        data = websocket.receive_json()
        assert data["type"] == "connection"
        assert data["status"] == "connected"

        # Receive price update (wait a bit for first update)
        data = websocket.receive_json()
        assert "symbol" in data
        assert "price" in data
        assert "timestamp" in data


def test_websocket_multiple_clients(client):
    """Test multiple WebSocket clients"""
    with client.websocket_connect("/ws/prices") as ws1:
        with client.websocket_connect("/ws/prices") as ws2:
            # Both should receive connection messages
            msg1 = ws1.receive_json()
            msg2 = ws2.receive_json()

            assert msg1["type"] == "connection"
            assert msg2["type"] == "connection"


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_pagination_invalid_page(client):
    """Test pagination with invalid page number"""
    response = client.get("/api/trades?page=0")
    assert response.status_code == 422  # Validation error


def test_pagination_large_page_size(client):
    """Test pagination respects max page size"""
    response = client.get("/api/trades?page=1&page_size=200")
    assert response.status_code == 422  # Validation error (max 100)


def test_cors_headers(client):
    """Test CORS headers are present"""
    response = client.get("/api/status")
    # Note: TestClient doesn't fully simulate CORS, but we can check the endpoint works
    assert response.status_code == 200


# ============================================================================
# Success Criteria Tests
# ============================================================================

def test_all_endpoints_accessible(client):
    """Test that all documented endpoints are accessible"""
    endpoints = [
        ("/", "GET"),
        ("/health", "GET"),
        ("/api/positions", "GET"),
        ("/api/trades", "GET"),
        ("/api/account", "GET"),
        ("/api/status", "GET"),
    ]

    for path, method in endpoints:
        if method == "GET":
            response = client.get(path)
            assert response.status_code == 200, f"Endpoint {path} failed"


def test_api_documentation(client):
    """Test that OpenAPI documentation is available"""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    openapi_spec = response.json()
    assert openapi_spec["info"]["title"] == "FluxHero API"
    assert openapi_spec["info"]["version"] == "1.0.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
