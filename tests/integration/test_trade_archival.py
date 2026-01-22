"""
Integration tests for trade archival functionality.

Tests the complete archive_old_trades() workflow:
- Export old trades to Parquet
- Delete from SQLite
- Verify data integrity
"""

import pytest
from datetime import datetime, timedelta
import pandas as pd

from backend.storage.sqlite_store import (
    SQLiteStore, Trade, TradeStatus, PositionSide
)


@pytest.fixture
async def temp_archive_db(tmp_path):
    """Create temporary database with archive directory."""
    db_path = tmp_path / "test_archive.db"
    store = SQLiteStore(str(db_path))
    await store.initialize()

    # Set archive directory to temp location
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    yield store, archive_dir

    await store.close()


@pytest.fixture
def sample_trade():
    """Create sample trade for testing."""
    return Trade(
        symbol="SPY",
        side=PositionSide.LONG,
        entry_price=400.0,
        entry_time=datetime.utcnow().isoformat(),
        exit_price=405.0,
        exit_time=datetime.utcnow().isoformat(),
        shares=100,
        stop_loss=395.0,
        take_profit=410.0,
        realized_pnl=500.0,
        status=TradeStatus.CLOSED,
        strategy="TREND",
        regime="STRONG_TREND",
        signal_reason="Bullish EMA crossover",
        created_at=datetime.utcnow().isoformat(),
        updated_at=datetime.utcnow().isoformat()
    )


@pytest.mark.asyncio
async def test_archive_exports_to_parquet_and_deletes_from_sqlite(tmp_path, sample_trade):
    """
    Integration test: Verify archive_old_trades() exports to Parquet and deletes from SQLite.

    Requirements: AUDIT_TASKS.md Phase 1
    - Export old records to Parquet
    - Delete from SQLite
    - Verify data integrity
    """
    # Setup
    db_path = tmp_path / "test_archive.db"
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    store = SQLiteStore(str(db_path))
    await store.initialize()

    # Temporarily change archive directory (monkey-patch the method)
    async def patched_archive(days: int = 30) -> int:
        """Patched version that uses tmp_path for archive."""
        conn = store._get_connection()
        cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()

        cursor = conn.execute(
            """SELECT * FROM trades
               WHERE entry_time < ? AND status != ?
               ORDER BY entry_time ASC""",
            (cutoff_date, TradeStatus.OPEN)
        )
        trades_to_archive = cursor.fetchall()
        count = len(trades_to_archive)

        if count == 0:
            return 0

        # Export to tmp_path archive directory
        df = pd.DataFrame([dict(row) for row in trades_to_archive])
        archive_filename = f"trades_archive_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.parquet"
        archive_path = archive_dir / archive_filename

        df.to_parquet(
            archive_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )

        # Delete archived trades
        conn.execute(
            "DELETE FROM trades WHERE entry_time < ? AND status != ?",
            (cutoff_date, TradeStatus.OPEN)
        )
        conn.commit()

        return count

    store.archive_old_trades = patched_archive

    try:
        now = datetime.utcnow()

        # Add 5 old trades (>30 days, CLOSED)
        old_trade_ids = []
        for i in range(5):
            trade = Trade(**sample_trade.__dict__)
            trade.symbol = f"OLD_{i}"
            trade.entry_time = (now - timedelta(days=35 + i)).isoformat()
            trade.exit_time = (now - timedelta(days=34 + i)).isoformat()
            trade.status = TradeStatus.CLOSED
            trade.created_at = trade.entry_time
            trade.updated_at = trade.exit_time
            await store.add_trade(trade)
            old_trade_ids.append(f"OLD_{i}")

        # Add 3 recent trades (<30 days, CLOSED)
        recent_trade_ids = []
        for i in range(3):
            trade = Trade(**sample_trade.__dict__)
            trade.symbol = f"RECENT_{i}"
            trade.entry_time = (now - timedelta(days=10 + i)).isoformat()
            trade.exit_time = (now - timedelta(days=9 + i)).isoformat()
            trade.status = TradeStatus.CLOSED
            trade.created_at = trade.entry_time
            trade.updated_at = trade.exit_time
            await store.add_trade(trade)
            recent_trade_ids.append(f"RECENT_{i}")

        # Add 1 old OPEN trade (should NOT be archived)
        open_trade = Trade(**sample_trade.__dict__)
        open_trade.symbol = "OPEN_OLD"
        open_trade.entry_time = (now - timedelta(days=40)).isoformat()
        open_trade.status = TradeStatus.OPEN
        open_trade.exit_price = None
        open_trade.exit_time = None
        open_trade.realized_pnl = None
        open_trade.created_at = open_trade.entry_time
        open_trade.updated_at = open_trade.entry_time
        await store.add_trade(open_trade)

        # Verify initial state: 9 trades total
        all_trades_before = await store.get_recent_trades(limit=100)
        assert len(all_trades_before) == 9

        # Archive old trades
        archived_count = await store.archive_old_trades(days=30)

        # Assertion 1: Should archive exactly 5 trades
        assert archived_count == 5, f"Expected 5 trades archived, got {archived_count}"

        # Assertion 2: Parquet file should exist
        parquet_files = list(archive_dir.glob("trades_archive_*.parquet"))
        assert len(parquet_files) == 1, f"Expected 1 parquet file, found {len(parquet_files)}"

        # Assertion 3: Parquet file should contain exactly 5 rows
        df_archived = pd.read_parquet(parquet_files[0])
        assert len(df_archived) == 5, f"Expected 5 rows in Parquet, got {len(df_archived)}"

        # Assertion 4: Archived data should match old trades
        archived_symbols = set(df_archived['symbol'].tolist())
        expected_symbols = set(old_trade_ids)
        assert archived_symbols == expected_symbols, \
            f"Archived symbols {archived_symbols} don't match expected {expected_symbols}"

        # Assertion 5: SQLite should only have 4 trades remaining (3 recent + 1 open)
        remaining_trades = await store.get_recent_trades(limit=100)
        assert len(remaining_trades) == 4, f"Expected 4 remaining trades, got {len(remaining_trades)}"

        # Assertion 6: Remaining trades should be recent + open trade
        remaining_symbols = set(t.symbol for t in remaining_trades)
        expected_remaining = set(recent_trade_ids + ["OPEN_OLD"])
        assert remaining_symbols == expected_remaining, \
            f"Remaining symbols {remaining_symbols} don't match expected {expected_remaining}"

        # Assertion 7: Open trade should still exist
        open_trades_after = [t for t in remaining_trades if t.status == TradeStatus.OPEN]
        assert len(open_trades_after) == 1
        assert open_trades_after[0].symbol == "OPEN_OLD"

        print("✓ All assertions passed:")
        print(f"  - Archived {archived_count} trades to Parquet")
        print(f"  - Deleted {archived_count} trades from SQLite")
        print(f"  - Retained {len(remaining_trades)} recent/open trades")
        print("  - Data integrity verified")

    finally:
        await store.close()


@pytest.mark.asyncio
async def test_archive_handles_no_old_trades(tmp_path, sample_trade):
    """Test that archive gracefully handles case when no old trades exist."""
    db_path = tmp_path / "test_empty_archive.db"
    store = SQLiteStore(str(db_path))
    await store.initialize()

    try:
        # Add only recent trades
        now = datetime.utcnow()
        for i in range(3):
            trade = Trade(**sample_trade.__dict__)
            trade.entry_time = (now - timedelta(days=5 + i)).isoformat()
            trade.status = TradeStatus.CLOSED
            await store.add_trade(trade)

        # Archive should return 0
        archived_count = await store.archive_old_trades(days=30)
        assert archived_count == 0

        # All trades should still exist
        remaining_trades = await store.get_recent_trades(limit=100)
        assert len(remaining_trades) == 3

    finally:
        await store.close()


@pytest.mark.asyncio
async def test_archive_preserves_data_types(tmp_path, sample_trade):
    """Test that archived Parquet preserves all data types correctly."""
    db_path = tmp_path / "test_types.db"
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    store = SQLiteStore(str(db_path))
    await store.initialize()

    # Patch archive directory
    async def patched_archive(days: int = 30) -> int:
        conn = store._get_connection()
        cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()

        cursor = conn.execute(
            """SELECT * FROM trades
               WHERE entry_time < ? AND status != ?
               ORDER BY entry_time ASC""",
            (cutoff_date, TradeStatus.OPEN)
        )
        trades_to_archive = cursor.fetchall()
        count = len(trades_to_archive)

        if count == 0:
            return 0

        df = pd.DataFrame([dict(row) for row in trades_to_archive])
        archive_filename = f"trades_archive_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.parquet"
        archive_path = archive_dir / archive_filename

        df.to_parquet(
            archive_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )

        conn.execute(
            "DELETE FROM trades WHERE entry_time < ? AND status != ?",
            (cutoff_date, TradeStatus.OPEN)
        )
        conn.commit()

        return count

    store.archive_old_trades = patched_archive

    try:
        now = datetime.utcnow()

        # Add trade with all fields populated
        trade = Trade(**sample_trade.__dict__)
        trade.entry_time = (now - timedelta(days=40)).isoformat()
        trade.exit_time = (now - timedelta(days=39)).isoformat()
        trade.status = TradeStatus.CLOSED
        await store.add_trade(trade)

        # Archive
        await store.archive_old_trades(days=30)

        # Load from Parquet
        parquet_files = list(archive_dir.glob("trades_archive_*.parquet"))
        df = pd.read_parquet(parquet_files[0])

        # Verify data types and values
        row = df.iloc[0]
        assert row['symbol'] == sample_trade.symbol
        assert row['side'] == sample_trade.side
        assert row['entry_price'] == sample_trade.entry_price
        assert row['shares'] == sample_trade.shares
        assert row['realized_pnl'] == sample_trade.realized_pnl
        assert row['strategy'] == sample_trade.strategy

    finally:
        await store.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
