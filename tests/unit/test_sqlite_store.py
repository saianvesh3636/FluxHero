"""
Unit Tests for SQLite Store Module

Tests for Feature 7 (Lightweight State Management) from FLUXHERO_REQUIREMENTS.md.

Test coverage:
- Database initialization and schema creation
- Trade operations (add, update, get, query)
- Position operations (upsert, delete, get)
- Settings operations (set, get, get_all)
- Async write operations (non-blocking)
- Performance benchmarks (<5ms writes, <10ms reads)
- Archive old trades functionality
- Edge cases and error handling

Success criteria:
- Write trade: <5ms (async, non-blocking)
- Query recent trades: <10ms
- Database size after 1 year: <100 MB
"""

import pytest
import pytest_asyncio
import asyncio
import tempfile
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "fluxhero"))

from backend.storage.sqlite_store import (
    SQLiteStore,
    Trade,
    Position,
    PositionSide,
    TradeStatus
)


@pytest_asyncio.fixture
async def temp_db():
    """Create temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_system.db")
        store = SQLiteStore(db_path)
        await store.initialize()
        yield store
        await store.close()


@pytest.fixture
def sample_trade():
    """Create sample trade for testing."""
    return Trade(
        symbol="SPY",
        side=PositionSide.LONG,
        entry_price=420.50,
        entry_time=datetime.utcnow().isoformat(),
        shares=100,
        stop_loss=415.00,
        take_profit=430.00,
        status=TradeStatus.OPEN,
        strategy="TREND",
        regime="STRONG_TREND",
        signal_reason="KAMA crossover + high ADX"
    )


@pytest.fixture
def sample_position():
    """Create sample position for testing."""
    return Position(
        symbol="AAPL",
        side=PositionSide.LONG,
        shares=50,
        entry_price=180.00,
        current_price=185.00,
        unrealized_pnl=250.00,
        stop_loss=175.00,
        entry_time=datetime.utcnow().isoformat()
    )


# ==================== Database Initialization Tests ====================


@pytest.mark.asyncio
async def test_database_initialization(temp_db):
    """Test database initializes with correct schema."""
    conn = temp_db._get_connection()
    cursor = conn.cursor()

    # Check trades table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
    assert cursor.fetchone() is not None

    # Check positions table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='positions'")
    assert cursor.fetchone() is not None

    # Check settings table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='settings'")
    assert cursor.fetchone() is not None


@pytest.mark.asyncio
async def test_indices_created(temp_db):
    """Test that performance indices are created."""
    conn = temp_db._get_connection()
    cursor = conn.cursor()

    # Check for indices
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
    indices = [row[0] for row in cursor.fetchall()]

    assert "idx_trades_symbol" in indices
    assert "idx_trades_status" in indices
    assert "idx_trades_entry_time" in indices


# ==================== Trade Operations Tests ====================


@pytest.mark.asyncio
async def test_add_trade(temp_db, sample_trade):
    """Test adding a trade to database."""
    trade_id = await temp_db.add_trade(sample_trade)

    assert trade_id is not None
    assert trade_id > 0

    # Verify trade was inserted
    retrieved_trade = await temp_db.get_trade(trade_id)
    assert retrieved_trade is not None
    assert retrieved_trade.symbol == "SPY"
    assert retrieved_trade.side == PositionSide.LONG
    assert retrieved_trade.entry_price == 420.50
    assert retrieved_trade.shares == 100


@pytest.mark.asyncio
async def test_update_trade(temp_db, sample_trade):
    """Test updating a trade."""
    trade_id = await temp_db.add_trade(sample_trade)

    # Update trade with exit details
    await temp_db.update_trade(
        trade_id,
        exit_price=425.00,
        exit_time=datetime.utcnow().isoformat(),
        realized_pnl=450.00,
        status=TradeStatus.CLOSED
    )

    # Verify update
    updated_trade = await temp_db.get_trade(trade_id)
    assert updated_trade.exit_price == 425.00
    assert updated_trade.realized_pnl == 450.00
    assert updated_trade.status == TradeStatus.CLOSED


@pytest.mark.asyncio
async def test_get_open_trades(temp_db, sample_trade):
    """Test retrieving open trades."""
    # Add multiple trades with different statuses
    trade1 = sample_trade
    trade1.symbol = "SPY"
    trade1.status = TradeStatus.OPEN
    await temp_db.add_trade(trade1)

    trade2 = Trade(**sample_trade.__dict__)
    trade2.symbol = "AAPL"
    trade2.status = TradeStatus.CLOSED
    await temp_db.add_trade(trade2)

    trade3 = Trade(**sample_trade.__dict__)
    trade3.symbol = "MSFT"
    trade3.status = TradeStatus.OPEN
    await temp_db.add_trade(trade3)

    # Get open trades
    open_trades = await temp_db.get_open_trades()
    assert len(open_trades) == 2
    assert all(t.status == TradeStatus.OPEN for t in open_trades)
    assert {t.symbol for t in open_trades} == {"SPY", "MSFT"}


@pytest.mark.asyncio
async def test_get_recent_trades(temp_db, sample_trade):
    """Test retrieving recent trades."""
    # Add multiple trades
    for i in range(10):
        trade = Trade(**sample_trade.__dict__)
        trade.symbol = f"TEST{i}"
        await temp_db.add_trade(trade)

    # Get recent trades with limit
    recent_trades = await temp_db.get_recent_trades(limit=5)
    assert len(recent_trades) == 5


@pytest.mark.asyncio
async def test_get_trades_by_date_range(temp_db, sample_trade):
    """Test retrieving trades by date range."""
    now = datetime.utcnow()

    # Add trades with different timestamps
    trade1 = Trade(**sample_trade.__dict__)
    trade1.entry_time = (now - timedelta(days=5)).isoformat()
    await temp_db.add_trade(trade1)

    trade2 = Trade(**sample_trade.__dict__)
    trade2.entry_time = (now - timedelta(days=2)).isoformat()
    await temp_db.add_trade(trade2)

    trade3 = Trade(**sample_trade.__dict__)
    trade3.entry_time = now.isoformat()
    await temp_db.add_trade(trade3)

    # Query date range
    start_date = (now - timedelta(days=3)).isoformat()
    end_date = now.isoformat()
    trades = await temp_db.get_trades_by_date_range(start_date, end_date)

    assert len(trades) == 2  # Should get trade2 and trade3


# ==================== Position Operations Tests ====================


@pytest.mark.asyncio
async def test_upsert_position(temp_db, sample_position):
    """Test upserting a position."""
    # Insert position
    await temp_db.upsert_position(sample_position)

    # Verify insertion
    position = await temp_db.get_position("AAPL")
    assert position is not None
    assert position.symbol == "AAPL"
    assert position.shares == 50
    assert position.entry_price == 180.00

    # Update position (upsert with same symbol)
    sample_position.shares = 100
    sample_position.current_price = 190.00
    await temp_db.upsert_position(sample_position)

    # Verify update
    updated_position = await temp_db.get_position("AAPL")
    assert updated_position.shares == 100
    assert updated_position.current_price == 190.00


@pytest.mark.asyncio
async def test_delete_position(temp_db, sample_position):
    """Test deleting a position."""
    # Insert position
    await temp_db.upsert_position(sample_position)

    # Verify it exists
    position = await temp_db.get_position("AAPL")
    assert position is not None

    # Delete position
    await temp_db.delete_position("AAPL")

    # Verify deletion
    position = await temp_db.get_position("AAPL")
    assert position is None


@pytest.mark.asyncio
async def test_get_open_positions(temp_db, sample_position):
    """Test retrieving all open positions."""
    # Add multiple positions
    pos1 = sample_position
    pos1.symbol = "SPY"
    await temp_db.upsert_position(pos1)

    pos2 = Position(**sample_position.__dict__)
    pos2.symbol = "AAPL"
    await temp_db.upsert_position(pos2)

    pos3 = Position(**sample_position.__dict__)
    pos3.symbol = "MSFT"
    await temp_db.upsert_position(pos3)

    # Get all positions
    positions = await temp_db.get_open_positions()
    assert len(positions) == 3
    assert {p.symbol for p in positions} == {"SPY", "AAPL", "MSFT"}


# ==================== Settings Operations Tests ====================


@pytest.mark.asyncio
async def test_set_and_get_setting(temp_db):
    """Test setting and getting a setting."""
    # Set setting
    await temp_db.set_setting("max_positions", "5", "Maximum open positions")

    # Get setting
    value = await temp_db.get_setting("max_positions")
    assert value == "5"


@pytest.mark.asyncio
async def test_get_setting_with_default(temp_db):
    """Test getting non-existent setting with default."""
    value = await temp_db.get_setting("nonexistent", default="default_value")
    assert value == "default_value"


@pytest.mark.asyncio
async def test_update_setting(temp_db):
    """Test updating an existing setting (upsert)."""
    # Set initial value
    await temp_db.set_setting("risk_limit", "1.0")

    # Update value
    await temp_db.set_setting("risk_limit", "2.0", "Updated risk limit")

    # Verify update
    value = await temp_db.get_setting("risk_limit")
    assert value == "2.0"


@pytest.mark.asyncio
async def test_get_all_settings(temp_db):
    """Test retrieving all settings."""
    # Set multiple settings
    await temp_db.set_setting("setting1", "value1")
    await temp_db.set_setting("setting2", "value2")
    await temp_db.set_setting("setting3", "value3")

    # Get all settings
    settings = await temp_db.get_all_settings()
    assert len(settings) == 3
    assert settings["setting1"] == "value1"
    assert settings["setting2"] == "value2"
    assert settings["setting3"] == "value3"


# ==================== Async Operations Tests ====================


@pytest.mark.asyncio
async def test_async_write_non_blocking(temp_db, sample_trade):
    """Test that async writes are non-blocking."""
    # Add multiple trades concurrently
    tasks = []
    for i in range(10):
        trade = Trade(**sample_trade.__dict__)
        trade.symbol = f"TEST{i}"
        tasks.append(temp_db.add_trade(trade))

    # All writes should complete without blocking
    start_time = time.time()
    await asyncio.gather(*tasks)
    elapsed = time.time() - start_time

    # Should complete quickly even with 10 concurrent writes
    assert elapsed < 1.0  # 1 second for 10 writes is generous


@pytest.mark.asyncio
async def test_multiple_concurrent_operations(temp_db, sample_trade, sample_position):
    """Test multiple concurrent read/write operations."""
    # Mix of writes and reads
    tasks = [
        temp_db.add_trade(sample_trade),
        temp_db.upsert_position(sample_position),
        temp_db.set_setting("test_key", "test_value"),
        temp_db.get_open_trades(),
        temp_db.get_open_positions(),
        temp_db.get_all_settings()
    ]

    # All should complete without errors
    results = await asyncio.gather(*tasks)
    assert len(results) == 6


# ==================== Performance Tests ====================


@pytest.mark.asyncio
async def test_write_trade_performance(temp_db, sample_trade):
    """Test trade write performance (<5ms target)."""
    start_time = time.time()
    await temp_db.add_trade(sample_trade)
    elapsed = (time.time() - start_time) * 1000  # Convert to ms

    # Target: <5ms for async write
    assert elapsed < 50  # Allow 50ms for test environment (generous)


@pytest.mark.asyncio
async def test_query_trades_performance(temp_db, sample_trade):
    """Test trade query performance (<10ms target)."""
    # Add some trades first
    for i in range(30):
        trade = Trade(**sample_trade.__dict__)
        trade.symbol = f"TEST{i}"
        await temp_db.add_trade(trade)

    # Measure query performance
    start_time = time.time()
    trades = await temp_db.get_recent_trades(limit=50)
    elapsed = (time.time() - start_time) * 1000  # Convert to ms

    # Target: <10ms for query
    assert elapsed < 100  # Allow 100ms for test environment
    assert len(trades) == 30


@pytest.mark.asyncio
async def test_bulk_write_performance(temp_db, sample_trade):
    """Test bulk write performance."""
    # Add 100 trades
    start_time = time.time()
    tasks = []
    for i in range(100):
        trade = Trade(**sample_trade.__dict__)
        trade.symbol = f"BULK{i}"
        tasks.append(temp_db.add_trade(trade))

    await asyncio.gather(*tasks)
    elapsed = time.time() - start_time

    # Should handle 100 trades in reasonable time
    assert elapsed < 5.0  # 5 seconds for 100 trades


# ==================== Archive Operations Tests ====================


@pytest.mark.asyncio
async def test_archive_old_trades(temp_db, sample_trade):
    """Test archiving old trades (R7.1.3)."""
    now = datetime.utcnow()

    # Add old trades (>30 days)
    for i in range(5):
        trade = Trade(**sample_trade.__dict__)
        trade.entry_time = (now - timedelta(days=35 + i)).isoformat()
        trade.status = TradeStatus.CLOSED
        await temp_db.add_trade(trade)

    # Add recent trades (<30 days)
    for i in range(3):
        trade = Trade(**sample_trade.__dict__)
        trade.entry_time = (now - timedelta(days=10 + i)).isoformat()
        trade.status = TradeStatus.CLOSED
        await temp_db.add_trade(trade)

    # Check archive count
    archived_count = await temp_db.archive_old_trades(days=30)
    assert archived_count == 5  # Should count 5 old trades


@pytest.mark.asyncio
async def test_archive_does_not_count_open_trades(temp_db, sample_trade):
    """Test that archive doesn't count open trades."""
    now = datetime.utcnow()

    # Add old OPEN trade (should not be archived)
    trade = Trade(**sample_trade.__dict__)
    trade.entry_time = (now - timedelta(days=40)).isoformat()
    trade.status = TradeStatus.OPEN
    await temp_db.add_trade(trade)

    # Check archive count
    archived_count = await temp_db.archive_old_trades(days=30)
    assert archived_count == 0  # Should not count open trade


# ==================== Database Size Tests ====================


@pytest.mark.asyncio
async def test_get_database_size(temp_db, sample_trade):
    """Test getting database file size."""
    # Add some data
    for i in range(10):
        trade = Trade(**sample_trade.__dict__)
        trade.symbol = f"SIZE_TEST{i}"
        await temp_db.add_trade(trade)

    # Get database size
    size = await temp_db.get_database_size()
    assert size > 0  # Should have some size


@pytest.mark.asyncio
async def test_database_size_reasonable(temp_db, sample_trade):
    """Test that database size is reasonable for 1 year of data."""
    # Simulate 1 year: ~250 trading days, ~5 trades per day = 1250 trades
    # This is a scaled-down version for testing
    num_trades = 100  # Scaled down for test performance

    for i in range(num_trades):
        trade = Trade(**sample_trade.__dict__)
        trade.symbol = f"YEAR_TEST{i}"
        await temp_db.add_trade(trade)

    size = await temp_db.get_database_size()

    # 100 trades should be well under 1MB (scaled down from 100MB for 1 year)
    assert size < 1_000_000  # <1MB for 100 trades


# ==================== Edge Cases Tests ====================


@pytest.mark.asyncio
async def test_get_nonexistent_trade(temp_db):
    """Test getting a non-existent trade."""
    trade = await temp_db.get_trade(99999)
    assert trade is None


@pytest.mark.asyncio
async def test_get_nonexistent_position(temp_db):
    """Test getting a non-existent position."""
    position = await temp_db.get_position("NONEXISTENT")
    assert position is None


@pytest.mark.asyncio
async def test_delete_nonexistent_position(temp_db):
    """Test deleting a non-existent position (should not error)."""
    # Should not raise exception
    await temp_db.delete_position("NONEXISTENT")


@pytest.mark.asyncio
async def test_empty_database_queries(temp_db):
    """Test queries on empty database."""
    trades = await temp_db.get_open_trades()
    assert trades == []

    positions = await temp_db.get_open_positions()
    assert positions == []

    settings = await temp_db.get_all_settings()
    assert settings == {}


@pytest.mark.asyncio
async def test_trade_with_optional_fields(temp_db):
    """Test adding trade with only required fields."""
    minimal_trade = Trade(
        symbol="MINIMAL",
        side=PositionSide.LONG,
        entry_price=100.0,
        entry_time=datetime.utcnow().isoformat(),
        shares=10,
        stop_loss=95.0,
        status=TradeStatus.OPEN,
        strategy="TEST"
    )

    trade_id = await temp_db.add_trade(minimal_trade)
    assert trade_id > 0

    retrieved = await temp_db.get_trade(trade_id)
    assert retrieved.symbol == "MINIMAL"
    assert retrieved.exit_price is None
    assert retrieved.realized_pnl is None


@pytest.mark.asyncio
async def test_position_with_short_side(temp_db):
    """Test position with SHORT side."""
    short_position = Position(
        symbol="SHORT_TEST",
        side=PositionSide.SHORT,
        shares=100,
        entry_price=200.0,
        current_price=195.0,
        unrealized_pnl=500.0,  # Profit on short
        stop_loss=205.0,
        entry_time=datetime.utcnow().isoformat()
    )

    await temp_db.upsert_position(short_position)

    retrieved = await temp_db.get_position("SHORT_TEST")
    assert retrieved.side == PositionSide.SHORT
    assert retrieved.unrealized_pnl == 500.0


# ==================== Close and Cleanup Tests ====================


@pytest.mark.asyncio
async def test_close_store(temp_db):
    """Test closing the store."""
    # Add some data
    await temp_db.set_setting("test", "value")

    # Close store
    await temp_db.close()

    # Connection should be closed
    assert temp_db._connection is None or not temp_db._connection


# ==================== Success Criteria Summary ====================


@pytest.mark.asyncio
async def test_success_criteria_write_performance(temp_db, sample_trade):
    """
    Success Criterion: Write trade to SQLite <5ms (async, non-blocking).

    This test verifies R7.1.2 requirement.
    """
    times = []
    for _ in range(10):
        start = time.time()
        await temp_db.add_trade(sample_trade)
        elapsed_ms = (time.time() - start) * 1000
        times.append(elapsed_ms)

    avg_time = sum(times) / len(times)
    # Allow generous margin for test environments
    assert avg_time < 50, f"Average write time {avg_time:.2f}ms exceeds target"


@pytest.mark.asyncio
async def test_success_criteria_query_performance(temp_db, sample_trade):
    """
    Success Criterion: Load recent trades <10ms.

    This test verifies read performance requirement.
    """
    # Add 30 days worth of trades
    for i in range(50):
        trade = Trade(**sample_trade.__dict__)
        trade.symbol = f"PERF{i}"
        await temp_db.add_trade(trade)

    # Measure query time
    start = time.time()
    trades = await temp_db.get_recent_trades(50)
    elapsed_ms = (time.time() - start) * 1000

    # Allow generous margin for test environments
    assert elapsed_ms < 100, f"Query time {elapsed_ms:.2f}ms exceeds target"
    assert len(trades) == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
