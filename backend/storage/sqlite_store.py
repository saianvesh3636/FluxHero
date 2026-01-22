"""
SQLite Store for FluxHero - Lightweight State Management

This module implements SQLite-based storage for operational data:
- Trades: Entry/exit prices, timestamps, P&L
- Positions: Current open positions, unrealized P&L
- Settings: System parameters, risk limits

Requirements implemented:
- R7.1.1: Store trades, positions, settings in SQLite
- R7.1.2: Async write operations (non-blocking)
- R7.1.3: Daily rollover for old trades (keep last 30 days)

Performance targets:
- Write trade: <5ms (async, non-blocking)
- Query recent trades: <10ms
- Database size after 1 year: <100 MB
"""

import asyncio
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import IntEnum
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class PositionSide(IntEnum):
    """Position side enum"""

    LONG = 1
    SHORT = -1


class TradeStatus(IntEnum):
    """Trade status enum"""

    OPEN = 0
    CLOSED = 1
    CANCELLED = 2


@dataclass
class Trade:
    """Trade record data class"""

    id: int | None = None
    symbol: str = ""
    side: int = PositionSide.LONG  # 1 = LONG, -1 = SHORT
    entry_price: float = 0.0
    entry_time: str = ""  # ISO 8601 format
    exit_price: float | None = None
    exit_time: str | None = None
    shares: int = 0
    stop_loss: float = 0.0
    take_profit: float | None = None
    realized_pnl: float | None = None
    status: int = TradeStatus.OPEN
    strategy: str = ""  # "TREND" or "MEAN_REVERSION"
    regime: str = ""  # "STRONG_TREND", "MEAN_REVERSION", "NEUTRAL"
    signal_reason: str = ""  # Explanation for signal
    signal_explanation: str | None = (
        None  # JSON signal explanation from SignalExplanation.to_dict()
    )
    created_at: str = ""
    updated_at: str = ""


@dataclass
class Position:
    """Current position data class"""

    id: int | None = None
    symbol: str = ""
    side: int = PositionSide.LONG
    shares: int = 0
    entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    stop_loss: float = 0.0
    take_profit: float | None = None
    entry_time: str = ""
    updated_at: str = ""


@dataclass
class Setting:
    """System setting data class"""

    key: str = ""
    value: str = ""
    description: str = ""
    updated_at: str = ""


class SQLiteStore:
    """
    SQLite-based storage manager for FluxHero operational data.

    Features:
    - Async write operations (non-blocking)
    - Trade logging with complete metadata
    - Position tracking with unrealized P&L
    - System settings management
    - Daily rollover for old trades (30-day retention)

    Example:
        >>> store = SQLiteStore("data/system.db")
        >>> await store.initialize()
        >>> trade = Trade(symbol="SPY", side=PositionSide.LONG, ...)
        >>> await store.add_trade(trade)
        >>> positions = await store.get_open_positions()
    """

    def __init__(self, db_path: str = "data/system.db"):
        """
        Initialize SQLite store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: sqlite3.Connection | None = None
        self._write_queue: asyncio.Queue = asyncio.Queue()
        self._write_task: asyncio.Task | None = None

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._connection is None:
            self._connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                isolation_level=None,  # Autocommit mode
            )
            self._connection.row_factory = sqlite3.Row
        return self._connection

    async def initialize(self) -> None:
        """
        Initialize database schema and start async write worker.

        Creates tables if they don't exist:
        - trades: Trade history
        - positions: Current open positions
        - settings: System configuration
        """
        logger.info("Initializing SQLite database", extra={"db_path": str(self.db_path)})
        conn = self._get_connection()

        # Create trades table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                entry_time TEXT NOT NULL,
                exit_price REAL,
                exit_time TEXT,
                shares INTEGER NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL,
                realized_pnl REAL,
                status INTEGER NOT NULL DEFAULT 0,
                strategy TEXT NOT NULL,
                regime TEXT,
                signal_reason TEXT,
                signal_explanation TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        # Migration: Add signal_explanation column to existing databases
        # Check if column exists, if not add it
        try:
            cursor = conn.execute("PRAGMA table_info(trades)")
            columns = [row[1] for row in cursor.fetchall()]
            if "signal_explanation" not in columns:
                conn.execute("ALTER TABLE trades ADD COLUMN signal_explanation TEXT")
        except sqlite3.OperationalError:
            # Table doesn't exist yet, will be created above
            pass

        # Create positions table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL UNIQUE,
                side INTEGER NOT NULL,
                shares INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                current_price REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL,
                entry_time TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        # Create settings table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                description TEXT,
                updated_at TEXT NOT NULL
            )
        """)

        # Create indexes for performance
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_symbol
            ON trades(symbol)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_status
            ON trades(status)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_entry_time
            ON trades(entry_time)
        """)

        # Start async write worker
        if self._write_task is None or self._write_task.done():
            self._write_task = asyncio.create_task(self._write_worker())
            logger.info("Started async write worker")

    async def _write_worker(self) -> None:
        """
        Background worker for async write operations.

        Processes write operations from queue without blocking main thread.
        Implements R7.1.2 (async write to avoid blocking).
        """
        while True:
            try:
                operation, args, future = await self._write_queue.get()
                try:
                    result = operation(*args)
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
                finally:
                    self._write_queue.task_done()
            except asyncio.CancelledError:
                break
            except sqlite3.Error as e:
                # Log database errors but continue processing other operations
                logger.error(
                    f"SQLite error in write worker: {e}",
                    extra={"error_type": type(e).__name__},
                    exc_info=True,
                )
                self._write_queue.task_done()

    async def _async_write(self, operation, *args) -> Any:
        """
        Execute write operation asynchronously.

        Args:
            operation: Function to execute
            *args: Arguments for operation

        Returns:
            Result from operation
        """
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        await self._write_queue.put((operation, args, future))
        return await future

    # ==================== Trade Operations ====================

    def _insert_trade_sync(self, trade: Trade) -> int:
        """Synchronous trade insert (called by async wrapper)."""
        conn = self._get_connection()
        now = datetime.utcnow().isoformat()

        logger.debug(
            "Inserting trade",
            extra={
                "symbol": trade.symbol,
                "side": trade.side,
                "entry_price": trade.entry_price,
                "shares": trade.shares,
                "strategy": trade.strategy,
            },
        )

        cursor = conn.execute(
            """
            INSERT INTO trades (
                symbol, side, entry_price, entry_time, exit_price, exit_time,
                shares, stop_loss, take_profit, realized_pnl, status,
                strategy, regime, signal_reason, signal_explanation, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                trade.symbol,
                trade.side,
                trade.entry_price,
                trade.entry_time,
                trade.exit_price,
                trade.exit_time,
                trade.shares,
                trade.stop_loss,
                trade.take_profit,
                trade.realized_pnl,
                trade.status,
                trade.strategy,
                trade.regime,
                trade.signal_reason,
                trade.signal_explanation,
                now,
                now,
            ),
        )
        trade_id = cursor.lastrowid

        logger.info(
            "Trade inserted successfully",
            extra={
                "trade_id": trade_id,
                "symbol": trade.symbol,
                "side": trade.side,
                "shares": trade.shares,
            },
        )

        return trade_id

    async def add_trade(self, trade: Trade) -> int:
        """
        Add new trade to database (async, non-blocking).

        Args:
            trade: Trade object to insert

        Returns:
            Trade ID (auto-incremented)

        Example:
            >>> trade = Trade(symbol="SPY", side=PositionSide.LONG, entry_price=420.0, ...)
            >>> trade_id = await store.add_trade(trade)
        """
        return await self._async_write(self._insert_trade_sync, trade)

    def _update_trade_sync(self, trade_id: int, updates: dict) -> None:
        """Synchronous trade update (called by async wrapper)."""
        conn = self._get_connection()
        now = datetime.utcnow().isoformat()

        logger.debug("Updating trade", extra={"trade_id": trade_id, "fields": list(updates.keys())})

        # Build update query dynamically
        fields = []
        values = []
        for key, value in updates.items():
            fields.append(f"{key} = ?")
            values.append(value)

        fields.append("updated_at = ?")
        values.append(now)
        values.append(trade_id)

        query = f"UPDATE trades SET {', '.join(fields)} WHERE id = ?"
        conn.execute(query, values)

        logger.info(
            "Trade updated successfully",
            extra={"trade_id": trade_id, "fields_updated": list(updates.keys())},
        )

    async def update_trade(self, trade_id: int, **kwargs) -> None:
        """
        Update existing trade (async, non-blocking).

        Args:
            trade_id: Trade ID to update
            **kwargs: Fields to update (e.g., exit_price=420.0, status=TradeStatus.CLOSED)

        Example:
            >>> await store.update_trade(
            ...     1, exit_price=425.0, realized_pnl=500.0, status=TradeStatus.CLOSED
            ... )
        """
        await self._async_write(self._update_trade_sync, trade_id, kwargs)

    async def get_trade(self, trade_id: int) -> Trade | None:
        """
        Get trade by ID.

        Args:
            trade_id: Trade ID to retrieve

        Returns:
            Trade object or None if not found
        """
        logger.debug("Fetching trade by ID", extra={"trade_id": trade_id})
        conn = self._get_connection()
        cursor = conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
        row = cursor.fetchone()
        result = Trade(**dict(row)) if row else None

        if result:
            logger.debug("Trade found", extra={"trade_id": trade_id, "symbol": result.symbol})
        else:
            logger.debug("Trade not found", extra={"trade_id": trade_id})

        return result

    async def get_open_trades(self) -> list[Trade]:
        """
        Get all open trades.

        Returns:
            List of Trade objects with status = OPEN
        """
        logger.debug("Fetching all open trades")
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM trades WHERE status = ? ORDER BY entry_time DESC", (TradeStatus.OPEN,)
        )
        trades = [Trade(**dict(row)) for row in cursor.fetchall()]
        logger.debug("Open trades fetched", extra={"count": len(trades)})
        return trades

    async def get_recent_trades(self, limit: int = 50) -> list[Trade]:
        """
        Get recent trades (any status).

        Args:
            limit: Maximum number of trades to return

        Returns:
            List of Trade objects, ordered by entry_time descending
        """
        logger.debug("Fetching recent trades", extra={"limit": limit})
        conn = self._get_connection()
        cursor = conn.execute("SELECT * FROM trades ORDER BY entry_time DESC LIMIT ?", (limit,))
        trades = [Trade(**dict(row)) for row in cursor.fetchall()]
        logger.debug("Recent trades fetched", extra={"count": len(trades), "limit": limit})
        return trades

    async def get_trades_by_date_range(self, start_date: str, end_date: str) -> list[Trade]:
        """
        Get trades within date range.

        Args:
            start_date: Start date (ISO 8601 format)
            end_date: End date (ISO 8601 format)

        Returns:
            List of Trade objects
        """
        logger.debug(
            "Fetching trades by date range", extra={"start_date": start_date, "end_date": end_date}
        )
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM trades WHERE entry_time >= ? AND entry_time <= ? ORDER BY entry_time",
            (start_date, end_date),
        )
        trades = [Trade(**dict(row)) for row in cursor.fetchall()]
        logger.debug(
            "Trades fetched by date range",
            extra={"count": len(trades), "start_date": start_date, "end_date": end_date},
        )
        return trades

    # ==================== Position Operations ====================

    def _upsert_position_sync(self, position: Position) -> None:
        """Synchronous position upsert (called by async wrapper)."""
        conn = self._get_connection()
        now = datetime.utcnow().isoformat()

        logger.debug(
            "Upserting position",
            extra={
                "symbol": position.symbol,
                "side": position.side,
                "shares": position.shares,
                "unrealized_pnl": position.unrealized_pnl,
            },
        )

        conn.execute(
            """
            INSERT INTO positions (
                symbol, side, shares, entry_price, current_price, unrealized_pnl,
                stop_loss, take_profit, entry_time, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                side = excluded.side,
                shares = excluded.shares,
                entry_price = excluded.entry_price,
                current_price = excluded.current_price,
                unrealized_pnl = excluded.unrealized_pnl,
                stop_loss = excluded.stop_loss,
                take_profit = excluded.take_profit,
                updated_at = excluded.updated_at
        """,
            (
                position.symbol,
                position.side,
                position.shares,
                position.entry_price,
                position.current_price,
                position.unrealized_pnl,
                position.stop_loss,
                position.take_profit,
                position.entry_time,
                now,
            ),
        )

        logger.info(
            "Position upserted successfully",
            extra={"symbol": position.symbol, "side": position.side, "shares": position.shares},
        )

    async def upsert_position(self, position: Position) -> None:
        """
        Insert or update position (async, non-blocking).

        If position with same symbol exists, updates it.
        Otherwise, inserts new position.

        Args:
            position: Position object to upsert

        Example:
            >>> position = Position(symbol="SPY", side=PositionSide.LONG, shares=100, ...)
            >>> await store.upsert_position(position)
        """
        await self._async_write(self._upsert_position_sync, position)

    def _delete_position_sync(self, symbol: str) -> None:
        """Synchronous position delete (called by async wrapper)."""
        logger.debug("Deleting position", extra={"symbol": symbol})
        conn = self._get_connection()
        conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
        logger.info("Position deleted successfully", extra={"symbol": symbol})

    async def delete_position(self, symbol: str) -> None:
        """
        Delete position by symbol (async, non-blocking).

        Args:
            symbol: Symbol to delete

        Example:
            >>> await store.delete_position("SPY")
        """
        await self._async_write(self._delete_position_sync, symbol)

    async def get_position(self, symbol: str) -> Position | None:
        """
        Get position by symbol.

        Args:
            symbol: Symbol to retrieve

        Returns:
            Position object or None if not found
        """
        logger.debug("Fetching position by symbol", extra={"symbol": symbol})
        conn = self._get_connection()
        cursor = conn.execute("SELECT * FROM positions WHERE symbol = ?", (symbol,))
        row = cursor.fetchone()
        result = Position(**dict(row)) if row else None

        if result:
            logger.debug("Position found", extra={"symbol": symbol, "shares": result.shares})
        else:
            logger.debug("Position not found", extra={"symbol": symbol})

        return result

    async def get_open_positions(self) -> list[Position]:
        """
        Get all open positions.

        Returns:
            List of Position objects
        """
        logger.debug("Fetching all open positions")
        conn = self._get_connection()
        cursor = conn.execute("SELECT * FROM positions ORDER BY entry_time DESC")
        positions = [Position(**dict(row)) for row in cursor.fetchall()]
        logger.debug("Open positions fetched", extra={"count": len(positions)})
        return positions

    # ==================== Settings Operations ====================

    def _set_setting_sync(self, key: str, value: str, description: str = "") -> None:
        """Synchronous setting upsert (called by async wrapper)."""
        logger.debug("Setting configuration value", extra={"key": key})
        conn = self._get_connection()
        now = datetime.utcnow().isoformat()

        conn.execute(
            """
            INSERT INTO settings (key, value, description, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                description = excluded.description,
                updated_at = excluded.updated_at
        """,
            (key, value, description, now),
        )

        logger.info("Setting updated successfully", extra={"key": key})

    async def set_setting(self, key: str, value: str, description: str = "") -> None:
        """
        Set or update system setting (async, non-blocking).

        Args:
            key: Setting key
            value: Setting value
            description: Optional description

        Example:
            >>> await store.set_setting("max_positions", "5", "Maximum open positions")
        """
        await self._async_write(self._set_setting_sync, key, value, description)

    async def get_setting(self, key: str, default: str | None = None) -> str | None:
        """
        Get system setting by key.

        Args:
            key: Setting key
            default: Default value if key not found

        Returns:
            Setting value or default
        """
        logger.debug("Fetching setting", extra={"key": key})
        conn = self._get_connection()
        cursor = conn.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = cursor.fetchone()
        result = row["value"] if row else default

        if row:
            logger.debug("Setting found", extra={"key": key})
        else:
            logger.debug("Setting not found, using default", extra={"key": key, "default": default})

        return result

    async def get_all_settings(self) -> dict[str, str]:
        """
        Get all system settings.

        Returns:
            Dictionary of key-value pairs
        """
        logger.debug("Fetching all settings")
        conn = self._get_connection()
        cursor = conn.execute("SELECT key, value FROM settings")
        settings = {row["key"]: row["value"] for row in cursor.fetchall()}
        logger.debug("All settings fetched", extra={"count": len(settings)})
        return settings

    # ==================== Maintenance Operations ====================

    async def archive_old_trades(self, days: int = 30) -> int:
        """
        Archive trades older than specified days (R7.1.3).

        Implementation:
        1. Export old trades to Parquet files
        2. Delete from SQLite to save space
        3. Keep only last 30 days in SQLite

        Args:
            days: Number of days to retain in SQLite

        Returns:
            Number of trades archived

        Example:
            >>> archived_count = await store.archive_old_trades(30)
            >>> print(f"Archived {archived_count} old trades")
        """
        conn = self._get_connection()
        cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()

        # Query trades to archive (closed trades older than cutoff)
        cursor = conn.execute(
            """SELECT * FROM trades
               WHERE entry_time < ? AND status != ?
               ORDER BY entry_time ASC""",
            (cutoff_date, TradeStatus.OPEN),
        )
        trades_to_archive = cursor.fetchall()
        count = len(trades_to_archive)

        if count == 0:
            logger.info("No trades to archive", extra={"cutoff_date": cutoff_date})
            return 0

        # Export to Parquet if there are trades to archive
        try:
            # Create archive directory
            archive_dir = Path("data/archive")
            archive_dir.mkdir(parents=True, exist_ok=True)

            # Convert to DataFrame
            df = pd.DataFrame([dict(row) for row in trades_to_archive])

            # Generate archive filename with timestamp
            archive_filename = (
                f"trades_archive_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.parquet"
            )
            archive_path = archive_dir / archive_filename

            # Write to Parquet with Snappy compression
            df.to_parquet(archive_path, engine="pyarrow", compression="snappy", index=False)

            logger.info(
                f"Exported {count} trades to archive",
                extra={
                    "count": count,
                    "archive_path": str(archive_path),
                    "cutoff_date": cutoff_date,
                },
            )

            # Delete archived trades from SQLite
            conn.execute(
                "DELETE FROM trades WHERE entry_time < ? AND status != ?",
                (cutoff_date, TradeStatus.OPEN),
            )
            conn.commit()

            logger.info(
                f"Deleted {count} archived trades from SQLite",
                extra={"count": count, "cutoff_date": cutoff_date},
            )

        except Exception as e:
            logger.error(
                f"Failed to archive trades: {e}",
                extra={"cutoff_date": cutoff_date, "trades_count": count},
                exc_info=True,
            )
            # Don't delete if export failed
            raise

        return count

    async def get_database_size(self) -> int:
        """
        Get database file size in bytes.

        Returns:
            Size in bytes
        """
        size = self.db_path.stat().st_size if self.db_path.exists() else 0
        logger.debug(
            "Database size retrieved", extra={"size_bytes": size, "db_path": str(self.db_path)}
        )
        return size

    async def close(self) -> None:
        """
        Close database connection and stop async worker.
        """
        logger.info("Closing SQLite database", extra={"db_path": str(self.db_path)})

        # Cancel write worker
        if self._write_task and not self._write_task.done():
            logger.debug("Cancelling async write worker")
            self._write_task.cancel()
            try:
                await self._write_task
            except asyncio.CancelledError:
                pass

        # Close connection
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("Database connection closed successfully")
