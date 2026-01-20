"""
Rolling 500-Candle Buffer for In-Memory Data Management

This module implements a memory-efficient rolling buffer that maintains the most recent
500 candles in memory, automatically discarding older data to prevent memory bloat.

Requirements implemented:
- R7.3.1: Fetch last 500 candles on startup via API
- R7.3.2: Maintain rolling 500-candle buffer in memory
- R7.3.3: Discard candles older than 500 bars (no long-term storage)

Author: FluxHero
Date: 2026-01-20
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class Candle:
    """Represents a single OHLCV candle with optional indicators.

    Attributes:
        timestamp: Unix timestamp or datetime object
        open: Opening price
        high: Highest price
        low: Lowest price
        close: Closing price
        volume: Trading volume
        ema: Optional exponential moving average
        atr: Optional average true range
        rsi: Optional relative strength index
    """
    timestamp: float  # Unix timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    ema: Optional[float] = None
    atr: Optional[float] = None
    rsi: Optional[float] = None


class CandleBuffer:
    """Rolling buffer that maintains the most recent N candles in memory.

    This buffer automatically discards old candles when the maximum size is exceeded,
    implementing a FIFO (First In, First Out) strategy. Optimized for fast append
    and efficient memory usage.

    Requirements:
    - R7.3.2: Maintain rolling 500-candle buffer in memory
    - R7.3.3: Discard candles older than 500 bars

    Example:
        >>> buffer = CandleBuffer(max_size=500)
        >>> buffer.add_candle(timestamp=1234567890, open=100, high=101, low=99, close=100.5, volume=1000)
        >>> print(buffer.size())
        1
        >>> closes = buffer.get_close_array()
        >>> print(closes)
        [100.5]
    """

    def __init__(self, max_size: int = 500):
        """Initialize the rolling candle buffer.

        Args:
            max_size: Maximum number of candles to keep in memory (default: 500)

        Raises:
            ValueError: If max_size is less than 1
        """
        if max_size < 1:
            raise ValueError("max_size must be at least 1")

        self.max_size = max_size
        self._buffer: Deque[Candle] = deque(maxlen=max_size)

    def add_candle(
        self,
        timestamp: float,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        ema: Optional[float] = None,
        atr: Optional[float] = None,
        rsi: Optional[float] = None,
    ) -> None:
        """Add a new candle to the buffer.

        If the buffer is at max capacity, the oldest candle is automatically
        discarded (FIFO). This implements R7.3.3 (discard older data).

        Args:
            timestamp: Unix timestamp of the candle
            open: Opening price
            high: Highest price
            low: Lowest price
            close: Closing price
            volume: Trading volume
            ema: Optional EMA indicator value
            atr: Optional ATR indicator value
            rsi: Optional RSI indicator value
        """
        candle = Candle(
            timestamp=timestamp,
            open=open,
            high=high,
            low=low,
            close=close,
            volume=volume,
            ema=ema,
            atr=atr,
            rsi=rsi,
        )
        self._buffer.append(candle)

    def add_candles_bulk(self, candles: list[Candle]) -> None:
        """Add multiple candles to the buffer at once.

        This is more efficient than adding candles one by one when initializing
        the buffer with historical data.

        Args:
            candles: List of Candle objects to add
        """
        for candle in candles:
            self._buffer.append(candle)

    def size(self) -> int:
        """Get the current number of candles in the buffer.

        Returns:
            Number of candles currently stored
        """
        return len(self._buffer)

    def is_full(self) -> bool:
        """Check if the buffer is at maximum capacity.

        Returns:
            True if buffer contains max_size candles, False otherwise
        """
        return len(self._buffer) == self.max_size

    def is_empty(self) -> bool:
        """Check if the buffer is empty.

        Returns:
            True if buffer contains no candles, False otherwise
        """
        return len(self._buffer) == 0

    def clear(self) -> None:
        """Remove all candles from the buffer."""
        self._buffer.clear()

    def get_latest_candle(self) -> Optional[Candle]:
        """Get the most recent candle without removing it.

        Returns:
            Most recent Candle object, or None if buffer is empty
        """
        if self.is_empty():
            return None
        return self._buffer[-1]

    def get_oldest_candle(self) -> Optional[Candle]:
        """Get the oldest candle without removing it.

        Returns:
            Oldest Candle object, or None if buffer is empty
        """
        if self.is_empty():
            return None
        return self._buffer[0]

    def get_candle_at_index(self, index: int) -> Optional[Candle]:
        """Get candle at specific index.

        Args:
            index: Index in the buffer (0 = oldest, -1 = newest)

        Returns:
            Candle at the specified index, or None if index is out of bounds
        """
        try:
            return self._buffer[index]
        except IndexError:
            return None

    def get_close_array(self) -> NDArray[np.float64]:
        """Get closing prices as NumPy array for indicator calculations.

        Returns:
            NumPy array of closing prices (oldest to newest)
        """
        if self.is_empty():
            return np.array([], dtype=np.float64)
        return np.array([c.close for c in self._buffer], dtype=np.float64)

    def get_high_array(self) -> NDArray[np.float64]:
        """Get high prices as NumPy array.

        Returns:
            NumPy array of high prices (oldest to newest)
        """
        if self.is_empty():
            return np.array([], dtype=np.float64)
        return np.array([c.high for c in self._buffer], dtype=np.float64)

    def get_low_array(self) -> NDArray[np.float64]:
        """Get low prices as NumPy array.

        Returns:
            NumPy array of low prices (oldest to newest)
        """
        if self.is_empty():
            return np.array([], dtype=np.float64)
        return np.array([c.low for c in self._buffer], dtype=np.float64)

    def get_open_array(self) -> NDArray[np.float64]:
        """Get opening prices as NumPy array.

        Returns:
            NumPy array of opening prices (oldest to newest)
        """
        if self.is_empty():
            return np.array([], dtype=np.float64)
        return np.array([c.open for c in self._buffer], dtype=np.float64)

    def get_volume_array(self) -> NDArray[np.float64]:
        """Get volumes as NumPy array.

        Returns:
            NumPy array of volumes (oldest to newest)
        """
        if self.is_empty():
            return np.array([], dtype=np.float64)
        return np.array([c.volume for c in self._buffer], dtype=np.float64)

    def get_ohlcv_arrays(self) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Get all OHLCV data as NumPy arrays.

        Returns:
            Tuple of (open, high, low, close, volume) NumPy arrays
        """
        if self.is_empty():
            empty = np.array([], dtype=np.float64)
            return empty, empty, empty, empty, empty

        opens = np.array([c.open for c in self._buffer], dtype=np.float64)
        highs = np.array([c.high for c in self._buffer], dtype=np.float64)
        lows = np.array([c.low for c in self._buffer], dtype=np.float64)
        closes = np.array([c.close for c in self._buffer], dtype=np.float64)
        volumes = np.array([c.volume for c in self._buffer], dtype=np.float64)

        return opens, highs, lows, closes, volumes

    def get_ema_array(self) -> NDArray[np.float64]:
        """Get EMA indicator values as NumPy array.

        Returns:
            NumPy array of EMA values (NaN for missing values)
        """
        if self.is_empty():
            return np.array([], dtype=np.float64)
        return np.array(
            [c.ema if c.ema is not None else np.nan for c in self._buffer],
            dtype=np.float64,
        )

    def get_atr_array(self) -> NDArray[np.float64]:
        """Get ATR indicator values as NumPy array.

        Returns:
            NumPy array of ATR values (NaN for missing values)
        """
        if self.is_empty():
            return np.array([], dtype=np.float64)
        return np.array(
            [c.atr if c.atr is not None else np.nan for c in self._buffer],
            dtype=np.float64,
        )

    def get_rsi_array(self) -> NDArray[np.float64]:
        """Get RSI indicator values as NumPy array.

        Returns:
            NumPy array of RSI values (NaN for missing values)
        """
        if self.is_empty():
            return np.array([], dtype=np.float64)
        return np.array(
            [c.rsi if c.rsi is not None else np.nan for c in self._buffer],
            dtype=np.float64,
        )

    def get_timestamp_array(self) -> NDArray[np.float64]:
        """Get timestamps as NumPy array.

        Returns:
            NumPy array of Unix timestamps
        """
        if self.is_empty():
            return np.array([], dtype=np.float64)
        return np.array([c.timestamp for c in self._buffer], dtype=np.float64)

    def get_last_n_candles(self, n: int) -> list[Candle]:
        """Get the last N candles from the buffer.

        Args:
            n: Number of candles to retrieve

        Returns:
            List of the last N candles (or all if N > buffer size)
        """
        if n <= 0:
            return []
        if n >= len(self._buffer):
            return list(self._buffer)
        return list(self._buffer)[-n:]

    def get_memory_usage_bytes(self) -> int:
        """Estimate memory usage of the buffer.

        This is an approximation based on the size of candle objects.
        Each candle contains:
        - 1 timestamp (8 bytes float)
        - 5 OHLCV values (5 × 8 = 40 bytes)
        - 3 optional indicators (3 × 8 = 24 bytes, assuming None uses minimal space)
        - Python object overhead (~56 bytes per object)

        Returns:
            Estimated memory usage in bytes
        """
        bytes_per_candle = 8 + 40 + 24 + 56  # ~128 bytes per candle
        return len(self._buffer) * bytes_per_candle

    def __len__(self) -> int:
        """Get the number of candles in the buffer (enables len(buffer))."""
        return len(self._buffer)

    def __repr__(self) -> str:
        """String representation of the buffer."""
        return f"CandleBuffer(size={len(self._buffer)}, max_size={self.max_size})"
