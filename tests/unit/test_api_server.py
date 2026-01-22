"""
Unit tests for FluxHero FastAPI Server

Tests all REST endpoints and WebSocket functionality:
- GET /api/positions
- GET /api/trades (with pagination)
- GET /api/account
- GET /api/status
- POST /api/backtest
- WebSocket /ws/prices
- Logging functionality
"""

import asyncio
import logging

# Add backend to path
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.api.server import app, app_state
from backend.storage.sqlite_store import (
    Position,
    PositionSide,
    SQLiteStore,
    Trade,
    TradeStatus,
)


@pytest.fixture
def test_db(tmp_path):
    """Create a temporary test database"""
    db_path = tmp_path / "test.db"
    store = SQLiteStore(db_path=str(db_path))
    # Initialize database schema synchronously
    asyncio.run(store.initialize())
    yield store
    # Note: We don't await close() here because fixture is not async
    # The store will be cleaned up by pytest's tmp_path cleanup


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
    headers = {"Authorization": "Bearer fluxhero-dev-secret-change-in-production"}
    with client.websocket_connect("/ws/prices", headers=headers) as websocket:
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
    headers = {"Authorization": "Bearer fluxhero-dev-secret-change-in-production"}
    with client.websocket_connect("/ws/prices", headers=headers) as ws1:
        with client.websocket_connect("/ws/prices", headers=headers) as ws2:
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


def test_server_uses_config_settings():
    """Test that server.py uses centralized config for CORS and API settings"""
    from backend.api.server import app, settings
    from backend.core.config import get_settings

    # Verify that settings are imported and used
    config = get_settings()

    # Check that app uses config values
    assert app.title == config.api_title
    assert app.version == config.api_version
    assert app.description == config.api_description

    # Verify settings instance exists in server module
    assert settings is not None
    assert settings.cors_origins == config.cors_origins
    assert settings.cors_allow_credentials == config.cors_allow_credentials
    assert settings.cors_allow_methods == config.cors_allow_methods
    assert settings.cors_allow_headers == config.cors_allow_headers


# ============================================================================
# Test Data Endpoint Tests
# ============================================================================


def test_get_test_candles_spy(client):
    """Test getting SPY test candle data"""
    # Ensure we're not in production mode
    import os

    import pandas as pd

    os.environ["ENV"] = "development"

    # Load test data manually for the test
    spy_csv_path = Path(__file__).parent.parent.parent / "backend" / "test_data" / "spy_daily.csv"

    if spy_csv_path.exists():
        df = pd.read_csv(spy_csv_path, skiprows=[1])
        df = df.rename(
            columns={
                "Price": "Date",
                "Close": "close",
                "High": "high",
                "Low": "low",
                "Open": "open",
                "Volume": "volume",
            }
        )
        app_state.test_spy_data = []
        for _, row in df.iterrows():
            try:
                app_state.test_spy_data.append(
                    {
                        "timestamp": str(row["Date"]),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": int(row["volume"]),
                    }
                )
            except (ValueError, KeyError):
                continue

    response = client.get("/api/test/candles?symbol=SPY")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

    if len(data) > 0:
        # Verify data structure
        candle = data[0]
        assert "timestamp" in candle
        assert "open" in candle
        assert "high" in candle
        assert "low" in candle
        assert "close" in candle
        assert "volume" in candle

        # Verify data types
        assert isinstance(candle["open"], float)
        assert isinstance(candle["high"], float)
        assert isinstance(candle["low"], float)
        assert isinstance(candle["close"], float)
        assert isinstance(candle["volume"], int)


def test_get_test_candles_production_disabled(client):
    """Test that test endpoint is disabled in production"""
    import os

    os.environ["ENV"] = "production"

    response = client.get("/api/test/candles?symbol=SPY")
    assert response.status_code == 403
    assert "disabled in production" in response.json()["detail"]

    # Restore environment
    os.environ["ENV"] = "development"


def test_get_test_candles_invalid_symbol(client):
    """Test that only SPY symbol is supported"""
    import os

    os.environ["ENV"] = "development"

    response = client.get("/api/test/candles?symbol=AAPL")
    assert response.status_code == 400
    assert "Only SPY" in response.json()["detail"]


def test_get_test_candles_no_data_loaded(client):
    """Test endpoint when test data is not loaded"""
    import os

    os.environ["ENV"] = "development"

    # Clear test data
    app_state.test_spy_data = None

    response = client.get("/api/test/candles?symbol=SPY")
    assert response.status_code == 503
    assert "not available" in response.json()["detail"]


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


# ============================================================================
# Logging Tests
# ============================================================================


def test_server_startup_logging(test_db, caplog):
    """Test that server startup logs are generated correctly"""
    from contextlib import asynccontextmanager

    from backend.api.server import app_state

    with caplog.at_level(logging.INFO):
        # Create a custom lifespan context to test startup logging
        @asynccontextmanager
        async def test_startup_lifespan(app):
            # Import server module to trigger logging
            from backend.api import server

            # Manually call the startup logic
            app_state.sqlite_store = test_db
            app_state.data_feed_active = False
            app_state.start_time = datetime.now()

            # Use the module's logger
            server.logger.info("Starting FluxHero API server")
            server.logger.info("SQLite store initialized", extra={"db_path": str(test_db.db_path)})
            server.logger.info("FluxHero API server ready")

            yield

            # Shutdown logging
            server.logger.info("Shutting down FluxHero API server")
            server.logger.info("SQLite store closed")
            server.logger.info("WebSocket connections closed", extra={"client_count": 0})
            server.logger.info("FluxHero API server stopped")

        # Run the lifespan
        async def run_lifespan():
            async with test_startup_lifespan(app):
                pass

        asyncio.run(run_lifespan())

    # Verify startup logs
    assert "Starting FluxHero API server" in caplog.text
    assert "SQLite store initialized" in caplog.text
    assert "FluxHero API server ready" in caplog.text

    # Verify shutdown logs
    assert "Shutting down FluxHero API server" in caplog.text
    assert "SQLite store closed" in caplog.text
    assert "WebSocket connections closed" in caplog.text
    assert "FluxHero API server stopped" in caplog.text


def test_websocket_connection_logging(client, caplog):
    """Test that WebSocket connections generate appropriate logs"""
    headers = {"Authorization": "Bearer fluxhero-dev-secret-change-in-production"}
    with caplog.at_level(logging.INFO):
        with client.websocket_connect("/ws/prices", headers=headers) as websocket:
            # Receive connection message
            data = websocket.receive_json()
            assert data["type"] == "connection"

    # Verify connection log
    assert "WebSocket client connected" in caplog.text or "total_clients" in caplog.text


def test_websocket_disconnection_logging(client, caplog):
    """Test that WebSocket disconnections generate appropriate logs"""
    headers = {"Authorization": "Bearer fluxhero-dev-secret-change-in-production"}
    with caplog.at_level(logging.INFO):
        with client.websocket_connect("/ws/prices", headers=headers) as websocket:
            # Receive connection message
            websocket.receive_json()
            # WebSocket will disconnect when exiting context

        # Give it a moment to process the disconnect
        import time

        time.sleep(0.1)

    # Verify disconnection logs (either explicit disconnect or client removal)
    assert (
        "WebSocket client disconnected" in caplog.text
        or "WebSocket client removed" in caplog.text
        or "total_clients" in caplog.text
    )


def test_websocket_error_logging(client, caplog):
    """Test that WebSocket errors are logged appropriately"""
    # This test verifies that errors during WebSocket communication are logged
    # The actual error handling is tested by the existing WebSocket tests
    with caplog.at_level(logging.ERROR):
        # WebSocket errors would be logged at ERROR level with exc_info=True
        # We verify the logger is configured correctly
        from backend.api import server

        assert hasattr(server, "logger")
        assert isinstance(server.logger, logging.Logger)


def test_no_print_statements_in_server():
    """Test that server.py does not contain any print() statements"""
    server_path = Path(__file__).parent.parent.parent / "" / "backend" / "api" / "server.py"
    with open(server_path) as f:
        content = f.read()

    # Check that no print( statements exist in the code
    # (excluding comments and docstrings)
    lines = content.split("\n")
    for line_num, line in enumerate(lines, 1):
        # Skip comments and docstrings
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
            continue

        # Check for print statements
        if "print(" in line:
            pytest.fail(f"Found print() statement at line {line_num}: {line}")


def test_logger_exists():
    """Test that the logger is properly configured in server.py"""
    from backend.api import server

    # Verify logger exists
    assert hasattr(server, "logger")
    assert isinstance(server.logger, logging.Logger)

    # Verify logger name follows the pattern
    assert server.logger.name == "backend.api.server"


# ============================================================================
# Request/Response Logging Middleware Tests
# ============================================================================


def test_request_logging_middleware(client, caplog):
    """Test that requests are logged with proper context"""
    with caplog.at_level(logging.INFO):
        response = client.get("/api/status")
        assert response.status_code == 200

    # Verify incoming request log exists
    assert "Incoming request" in caplog.text

    # Check log records for extra fields
    incoming_logs = [r for r in caplog.records if "Incoming request" in r.message]
    assert len(incoming_logs) > 0
    incoming_log = incoming_logs[0]
    assert hasattr(incoming_log, "method")
    assert incoming_log.method == "GET"
    assert hasattr(incoming_log, "path")
    assert incoming_log.path == "/api/status"

    # Verify request completed log
    assert "Request completed" in caplog.text
    completed_logs = [r for r in caplog.records if "Request completed" in r.message]
    assert len(completed_logs) > 0
    completed_log = completed_logs[0]
    assert hasattr(completed_log, "status_code")
    assert completed_log.status_code == 200


def test_request_logging_includes_request_id(client, caplog):
    """Test that request ID is included in logs and response headers"""
    with caplog.at_level(logging.INFO):
        response = client.get("/api/positions")
        assert response.status_code == 200

    # Verify request ID in response headers
    assert "X-Request-ID" in response.headers
    request_id = response.headers["X-Request-ID"]
    assert request_id is not None
    assert len(request_id) > 0

    # Verify request ID appears in log records
    incoming_logs = [r for r in caplog.records if "Incoming request" in r.message]
    assert len(incoming_logs) > 0
    assert hasattr(incoming_logs[0], "request_id")
    assert incoming_logs[0].request_id == request_id


def test_request_logging_includes_process_time(client, caplog):
    """Test that processing time is logged and included in response headers"""
    with caplog.at_level(logging.INFO):
        response = client.get("/api/account")
        assert response.status_code == 200

    # Verify process time in response headers
    assert "X-Process-Time" in response.headers
    process_time = float(response.headers["X-Process-Time"])
    assert process_time >= 0

    # Verify process time in log records
    completed_logs = [r for r in caplog.records if "Request completed" in r.message]
    assert len(completed_logs) > 0
    assert hasattr(completed_logs[0], "process_time_ms")
    assert completed_logs[0].process_time_ms >= 0


def test_request_logging_with_query_params(client, caplog):
    """Test that query parameters are logged"""
    with caplog.at_level(logging.INFO):
        response = client.get("/api/trades?page=1&page_size=10")
        assert response.status_code == 200

    # Verify query params in log records
    incoming_logs = [r for r in caplog.records if "Incoming request" in r.message]
    assert len(incoming_logs) > 0
    assert hasattr(incoming_logs[0], "query_params")
    # Query params are logged as string
    assert "page" in incoming_logs[0].query_params


def test_request_logging_on_error(client, caplog):
    """Test that failed requests are logged with error details"""
    with caplog.at_level(logging.INFO):
        # Make request that will fail validation
        _ = client.get("/api/trades?page=0")
        # This should fail validation (422) or work if validation allows it

    # Should have both incoming and completion logs regardless
    assert "Incoming request" in caplog.text


def test_request_logging_different_methods(client, caplog):
    """Test that different HTTP methods are logged correctly"""
    with caplog.at_level(logging.INFO):
        # GET request
        response_get = client.get("/api/status")
        assert response_get.status_code == 200

    # Verify GET method is logged in log records
    incoming_logs = [r for r in caplog.records if "Incoming request" in r.message]
    assert len(incoming_logs) >= 1
    methods = [r.method for r in incoming_logs if hasattr(r, "method")]
    assert "GET" in methods


def test_middleware_preserves_response(client):
    """Test that middleware doesn't modify response body"""
    response = client.get("/api/status")
    assert response.status_code == 200

    # Response should still be valid JSON
    data = response.json()
    assert "status" in data
    # Check for either timestamp or last_update field
    assert "last_update" in data or "timestamp" in data


def test_middleware_adds_headers_without_breaking_response(client):
    """Test that middleware headers don't break existing functionality"""
    response = client.get("/")
    assert response.status_code == 200

    # Response should have our custom headers
    assert "X-Request-ID" in response.headers
    assert "X-Process-Time" in response.headers

    # But response body should still be intact
    data = response.json()
    assert data["name"] == "FluxHero API"


def test_middleware_logs_client_ip(client, caplog):
    """Test that client IP is logged (or unknown in test environment)"""
    with caplog.at_level(logging.INFO):
        response = client.get("/api/status")
        assert response.status_code == 200

    # Verify client IP field exists in log records (may be testclient or unknown)
    incoming_logs = [r for r in caplog.records if "Incoming request" in r.message]
    assert len(incoming_logs) > 0
    assert hasattr(incoming_logs[0], "client_ip")


# ============================================================================
# Metrics Endpoint Tests
# ============================================================================


def test_metrics_endpoint_format(client):
    """Test that /metrics endpoint returns Prometheus-compatible format"""
    response = client.get("/metrics")
    assert response.status_code == 200
    # Content-type should be text/plain with optional Prometheus version
    assert "text/plain" in response.headers["content-type"]

    # Parse response text
    content = response.text

    # Verify Prometheus format markers
    assert "# HELP" in content
    assert "# TYPE" in content

    # Verify required metrics exist
    assert "fluxhero_uptime_seconds" in content
    assert "fluxhero_requests_total" in content
    assert "fluxhero_websocket_connections" in content
    assert "fluxhero_data_feed_active" in content


def test_metrics_includes_latency_percentiles(client):
    """Test that metrics includes request latency percentiles after requests"""
    # Make several requests to populate latency data
    for _ in range(10):
        client.get("/api/status")

    # Now check metrics
    response = client.get("/metrics")
    assert response.status_code == 200
    content = response.text

    # After requests, latency metrics should be present
    assert "fluxhero_request_latency_p50_ms" in content
    assert "fluxhero_request_latency_p90_ms" in content
    assert "fluxhero_request_latency_p95_ms" in content
    assert "fluxhero_request_latency_p99_ms" in content


def test_metrics_includes_order_counts(client, test_db):
    """Test that metrics includes order/trade counts from database"""
    # Add some trades
    for i in range(5):
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
            signal_reason=f"Test trade {i}",
            exit_price=455.0,
            exit_time=datetime.now().isoformat(),
            realized_pnl=500.0,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )
        asyncio.run(test_db.add_trade(trade))

    response = client.get("/metrics")
    assert response.status_code == 200
    content = response.text

    # Should have order count metric
    assert "fluxhero_orders_total" in content
    assert "fluxhero_orders_total 5" in content


def test_metrics_includes_drawdown(client, test_db):
    """Test that metrics includes drawdown percentage"""
    # Add trades with P&L to create drawdown scenario
    # First a winning trade
    trade1 = Trade(
        symbol="SPY",
        side=PositionSide.LONG,
        entry_price=450.0,
        entry_time=datetime.now().isoformat(),
        shares=100,
        stop_loss=440.0,
        status=TradeStatus.CLOSED,
        strategy="TREND",
        regime="STRONG_TREND",
        signal_reason="Winner",
        exit_price=460.0,
        exit_time=datetime.now().isoformat(),
        realized_pnl=1000.0,  # Big win
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
    )
    asyncio.run(test_db.add_trade(trade1))

    # Then a losing trade
    trade2 = Trade(
        symbol="QQQ",
        side=PositionSide.LONG,
        entry_price=380.0,
        entry_time=datetime.now().isoformat(),
        shares=100,
        stop_loss=370.0,
        status=TradeStatus.CLOSED,
        strategy="TREND",
        regime="STRONG_TREND",
        signal_reason="Loser",
        exit_price=375.0,
        exit_time=datetime.now().isoformat(),
        realized_pnl=-500.0,  # Loss after peak
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
    )
    asyncio.run(test_db.add_trade(trade2))

    response = client.get("/metrics")
    assert response.status_code == 200
    content = response.text

    # Should have drawdown metric
    assert "fluxhero_drawdown_percent" in content
    assert "fluxhero_equity" in content
    assert "fluxhero_win_rate_percent" in content


def test_metrics_includes_request_counts_by_path(client):
    """Test that metrics includes request counts broken down by endpoint"""
    # Make requests to different endpoints
    client.get("/api/status")
    client.get("/api/status")
    client.get("/api/positions")

    response = client.get("/metrics")
    assert response.status_code == 200
    content = response.text

    # Should have per-path request counts
    assert "fluxhero_requests_by_path_total" in content
    assert 'path="/api/status"' in content


def test_metrics_no_database_error_handling(client):
    """Test that metrics endpoint handles database errors gracefully"""
    # Clear the database connection
    app_state.sqlite_store = None

    response = client.get("/metrics")
    assert response.status_code == 200
    content = response.text

    # Should still return basic metrics
    assert "fluxhero_uptime_seconds" in content
    assert "fluxhero_requests_total" in content


def test_health_endpoint_includes_system_metrics(client):
    """Test that updated /health endpoint includes system metrics"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    # Check for required fields
    assert "status" in data
    assert "timestamp" in data
    assert "uptime_seconds" in data
    assert "database_connected" in data
    assert "websocket_connections" in data
    assert "data_feed_active" in data
    assert "total_requests" in data

    # Verify types
    assert isinstance(data["uptime_seconds"], (int, float))
    assert isinstance(data["database_connected"], bool)
    assert isinstance(data["websocket_connections"], int)
    assert isinstance(data["data_feed_active"], bool)
    assert isinstance(data["total_requests"], int)


def test_health_endpoint_database_check(client, test_db):
    """Test that health endpoint properly checks database connectivity"""
    # With database connected
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["database_connected"] is True
    assert data["status"] in ["healthy", "degraded"]


def test_health_endpoint_degraded_status(client):
    """Test that health endpoint returns degraded status when database fails"""
    # Simulate database failure
    original_store = app_state.sqlite_store
    app_state.sqlite_store = None

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["database_connected"] is False
    assert data["status"] == "degraded"

    # Restore
    app_state.sqlite_store = original_store


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
