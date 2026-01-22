"""
Unit tests for Rolling 500-Candle Buffer (Feature 7.3)

Tests the in-memory candle buffer that maintains the most recent 500 candles,
automatically discarding older data.

Requirements tested:
- R7.3.2: Maintain rolling 500-candle buffer in memory
- R7.3.3: Discard candles older than 500 bars

Author: FluxHero
Date: 2026-01-20
"""

import time

import numpy as np
import pytest

from backend.storage.candle_buffer import Candle, CandleBuffer


class TestCandleBufferInitialization:
    """Test buffer initialization and basic properties."""

    def test_buffer_creation_default_size(self):
        """Test buffer creation with default size (500)."""
        buffer = CandleBuffer()
        assert buffer.max_size == 500
        assert buffer.size() == 0
        assert buffer.is_empty()
        assert not buffer.is_full()

    def test_buffer_creation_custom_size(self):
        """Test buffer creation with custom size."""
        buffer = CandleBuffer(max_size=100)
        assert buffer.max_size == 100
        assert buffer.size() == 0

    def test_buffer_creation_invalid_size(self):
        """Test buffer creation with invalid size raises ValueError."""
        with pytest.raises(ValueError, match="max_size must be at least 1"):
            CandleBuffer(max_size=0)

        with pytest.raises(ValueError, match="max_size must be at least 1"):
            CandleBuffer(max_size=-10)


class TestCandleAddition:
    """Test adding candles to the buffer."""

    def test_add_single_candle(self):
        """Test adding a single candle to the buffer."""
        buffer = CandleBuffer(max_size=10)
        buffer.add_candle(
            timestamp=1234567890.0,
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000.0,
        )
        assert buffer.size() == 1
        assert not buffer.is_empty()

    def test_add_candle_with_indicators(self):
        """Test adding a candle with indicator values."""
        buffer = CandleBuffer(max_size=10)
        buffer.add_candle(
            timestamp=1234567890.0,
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000.0,
            ema=100.2,
            atr=1.5,
            rsi=55.0,
        )
        assert buffer.size() == 1
        latest = buffer.get_latest_candle()
        assert latest is not None
        assert latest.ema == 100.2
        assert latest.atr == 1.5
        assert latest.rsi == 55.0

    def test_add_multiple_candles(self):
        """Test adding multiple candles sequentially."""
        buffer = CandleBuffer(max_size=10)
        for i in range(5):
            buffer.add_candle(
                timestamp=1234567890.0 + i,
                open=100.0 + i,
                high=101.0 + i,
                low=99.0 + i,
                close=100.5 + i,
                volume=1000.0,
            )
        assert buffer.size() == 5

    def test_buffer_reaches_max_size(self):
        """Test buffer correctly identifies when it reaches max capacity."""
        buffer = CandleBuffer(max_size=3)
        assert not buffer.is_full()

        for i in range(3):
            buffer.add_candle(
                timestamp=1234567890.0 + i,
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1000.0,
            )

        assert buffer.is_full()
        assert buffer.size() == 3


class TestRollingBehavior:
    """Test the rolling/FIFO behavior (R7.3.3: discard older data)."""

    def test_buffer_discards_oldest_when_full(self):
        """Test that oldest candle is discarded when buffer exceeds max_size."""
        buffer = CandleBuffer(max_size=3)

        # Add 3 candles
        for i in range(3):
            buffer.add_candle(
                timestamp=float(i),
                open=100.0 + i,
                high=101.0 + i,
                low=99.0 + i,
                close=100.5 + i,
                volume=1000.0,
            )

        # Verify oldest is timestamp=0
        oldest = buffer.get_oldest_candle()
        assert oldest is not None
        assert oldest.timestamp == 0.0

        # Add one more (should discard timestamp=0)
        buffer.add_candle(
            timestamp=3.0,
            open=103.0,
            high=104.0,
            low=102.0,
            close=103.5,
            volume=1000.0,
        )

        # Buffer should still be size 3
        assert buffer.size() == 3

        # Oldest should now be timestamp=1
        oldest = buffer.get_oldest_candle()
        assert oldest is not None
        assert oldest.timestamp == 1.0

        # Latest should be timestamp=3
        latest = buffer.get_latest_candle()
        assert latest is not None
        assert latest.timestamp == 3.0

    def test_rolling_500_candles(self):
        """Test rolling behavior with 500-candle buffer (R7.3.2)."""
        buffer = CandleBuffer(max_size=500)

        # Add 600 candles
        for i in range(600):
            buffer.add_candle(
                timestamp=float(i),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1000.0,
            )

        # Buffer should maintain exactly 500 candles
        assert buffer.size() == 500
        assert buffer.is_full()

        # Oldest should be candle 100 (600 - 500)
        oldest = buffer.get_oldest_candle()
        assert oldest is not None
        assert oldest.timestamp == 100.0

        # Latest should be candle 599
        latest = buffer.get_latest_candle()
        assert latest is not None
        assert latest.timestamp == 599.0


class TestBulkOperations:
    """Test bulk candle operations."""

    def test_add_candles_bulk(self):
        """Test adding multiple candles at once."""
        buffer = CandleBuffer(max_size=10)
        candles = [
            Candle(
                timestamp=float(i),
                open=100.0 + i,
                high=101.0 + i,
                low=99.0 + i,
                close=100.5 + i,
                volume=1000.0,
            )
            for i in range(5)
        ]
        buffer.add_candles_bulk(candles)
        assert buffer.size() == 5

    def test_bulk_add_exceeds_max_size(self):
        """Test bulk add correctly handles buffer overflow."""
        buffer = CandleBuffer(max_size=3)
        candles = [
            Candle(
                timestamp=float(i),
                open=100.0 + i,
                high=101.0 + i,
                low=99.0 + i,
                close=100.5 + i,
                volume=1000.0,
            )
            for i in range(5)
        ]
        buffer.add_candles_bulk(candles)

        # Should only keep last 3
        assert buffer.size() == 3
        assert buffer.get_oldest_candle().timestamp == 2.0
        assert buffer.get_latest_candle().timestamp == 4.0


class TestDataRetrieval:
    """Test retrieving data from the buffer."""

    def test_get_latest_candle(self):
        """Test retrieving the most recent candle."""
        buffer = CandleBuffer(max_size=10)
        buffer.add_candle(
            timestamp=1.0, open=100.0, high=101.0, low=99.0, close=100.5, volume=1000.0
        )
        buffer.add_candle(
            timestamp=2.0, open=101.0, high=102.0, low=100.0, close=101.5, volume=1100.0
        )

        latest = buffer.get_latest_candle()
        assert latest is not None
        assert latest.timestamp == 2.0
        assert latest.close == 101.5

    def test_get_oldest_candle(self):
        """Test retrieving the oldest candle."""
        buffer = CandleBuffer(max_size=10)
        buffer.add_candle(
            timestamp=1.0, open=100.0, high=101.0, low=99.0, close=100.5, volume=1000.0
        )
        buffer.add_candle(
            timestamp=2.0, open=101.0, high=102.0, low=100.0, close=101.5, volume=1100.0
        )

        oldest = buffer.get_oldest_candle()
        assert oldest is not None
        assert oldest.timestamp == 1.0
        assert oldest.close == 100.5

    def test_get_candle_at_index(self):
        """Test retrieving candle at specific index."""
        buffer = CandleBuffer(max_size=10)
        for i in range(5):
            buffer.add_candle(
                timestamp=float(i),
                open=100.0 + i,
                high=101.0 + i,
                low=99.0 + i,
                close=100.5 + i,
                volume=1000.0,
            )

        candle = buffer.get_candle_at_index(2)
        assert candle is not None
        assert candle.timestamp == 2.0

        # Test negative indexing
        candle = buffer.get_candle_at_index(-1)
        assert candle is not None
        assert candle.timestamp == 4.0

    def test_get_candle_at_invalid_index(self):
        """Test retrieving candle at invalid index returns None."""
        buffer = CandleBuffer(max_size=10)
        buffer.add_candle(
            timestamp=1.0, open=100.0, high=101.0, low=99.0, close=100.5, volume=1000.0
        )

        assert buffer.get_candle_at_index(10) is None
        assert buffer.get_candle_at_index(-10) is None

    def test_get_last_n_candles(self):
        """Test retrieving the last N candles."""
        buffer = CandleBuffer(max_size=10)
        for i in range(5):
            buffer.add_candle(
                timestamp=float(i),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1000.0,
            )

        last_3 = buffer.get_last_n_candles(3)
        assert len(last_3) == 3
        assert last_3[0].timestamp == 2.0
        assert last_3[-1].timestamp == 4.0

    def test_get_last_n_exceeds_buffer_size(self):
        """Test requesting more candles than available returns all."""
        buffer = CandleBuffer(max_size=10)
        for i in range(5):
            buffer.add_candle(
                timestamp=float(i),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1000.0,
            )

        last_10 = buffer.get_last_n_candles(10)
        assert len(last_10) == 5


class TestArrayRetrieval:
    """Test retrieving data as NumPy arrays for indicator calculations."""

    def test_get_close_array(self):
        """Test retrieving closing prices as NumPy array."""
        buffer = CandleBuffer(max_size=10)
        closes = [100.5, 101.5, 102.5]
        for i, close in enumerate(closes):
            buffer.add_candle(
                timestamp=float(i),
                open=100.0,
                high=101.0,
                low=99.0,
                close=close,
                volume=1000.0,
            )

        close_array = buffer.get_close_array()
        assert isinstance(close_array, np.ndarray)
        assert close_array.dtype == np.float64
        np.testing.assert_array_equal(close_array, np.array(closes, dtype=np.float64))

    def test_get_ohlcv_arrays(self):
        """Test retrieving all OHLCV data as NumPy arrays."""
        buffer = CandleBuffer(max_size=10)
        for i in range(3):
            buffer.add_candle(
                timestamp=float(i),
                open=100.0 + i,
                high=101.0 + i,
                low=99.0 + i,
                close=100.5 + i,
                volume=1000.0 + i * 100,
            )

        opens, highs, lows, closes, volumes = buffer.get_ohlcv_arrays()

        assert isinstance(opens, np.ndarray)
        np.testing.assert_array_equal(opens, np.array([100.0, 101.0, 102.0], dtype=np.float64))
        np.testing.assert_array_equal(highs, np.array([101.0, 102.0, 103.0], dtype=np.float64))
        np.testing.assert_array_equal(lows, np.array([99.0, 100.0, 101.0], dtype=np.float64))
        np.testing.assert_array_equal(closes, np.array([100.5, 101.5, 102.5], dtype=np.float64))
        np.testing.assert_array_equal(volumes, np.array([1000.0, 1100.0, 1200.0], dtype=np.float64))

    def test_get_indicator_arrays(self):
        """Test retrieving indicator values as NumPy arrays."""
        buffer = CandleBuffer(max_size=10)
        buffer.add_candle(
            timestamp=1.0,
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000.0,
            ema=100.2,
            atr=1.5,
            rsi=55.0,
        )
        buffer.add_candle(
            timestamp=2.0,
            open=101.0,
            high=102.0,
            low=100.0,
            close=101.5,
            volume=1100.0,
            ema=101.0,
            atr=1.6,
            rsi=60.0,
        )
        buffer.add_candle(
            timestamp=3.0,
            open=102.0,
            high=103.0,
            low=101.0,
            close=102.5,
            volume=1200.0,
            # No indicators for this candle
        )

        ema_array = buffer.get_ema_array()
        atr_array = buffer.get_atr_array()
        rsi_array = buffer.get_rsi_array()

        # First two should have values, third should be NaN
        assert ema_array[0] == 100.2
        assert ema_array[1] == 101.0
        assert np.isnan(ema_array[2])

        assert atr_array[0] == 1.5
        assert atr_array[1] == 1.6
        assert np.isnan(atr_array[2])

        assert rsi_array[0] == 55.0
        assert rsi_array[1] == 60.0
        assert np.isnan(rsi_array[2])

    def test_get_timestamp_array(self):
        """Test retrieving timestamps as NumPy array."""
        buffer = CandleBuffer(max_size=10)
        timestamps = [1234567890.0, 1234567950.0, 1234568010.0]
        for ts in timestamps:
            buffer.add_candle(
                timestamp=ts, open=100.0, high=101.0, low=99.0, close=100.5, volume=1000.0
            )

        ts_array = buffer.get_timestamp_array()
        np.testing.assert_array_equal(ts_array, np.array(timestamps, dtype=np.float64))


class TestEmptyBuffer:
    """Test behavior with empty buffer."""

    def test_empty_buffer_properties(self):
        """Test empty buffer returns None/empty arrays correctly."""
        buffer = CandleBuffer(max_size=10)

        assert buffer.is_empty()
        assert buffer.size() == 0
        assert buffer.get_latest_candle() is None
        assert buffer.get_oldest_candle() is None

    def test_empty_buffer_arrays(self):
        """Test empty buffer returns empty NumPy arrays."""
        buffer = CandleBuffer(max_size=10)

        assert len(buffer.get_close_array()) == 0
        assert len(buffer.get_high_array()) == 0
        assert len(buffer.get_low_array()) == 0
        assert len(buffer.get_open_array()) == 0
        assert len(buffer.get_volume_array()) == 0
        assert len(buffer.get_timestamp_array()) == 0

        opens, highs, lows, closes, volumes = buffer.get_ohlcv_arrays()
        assert len(opens) == 0
        assert len(highs) == 0
        assert len(lows) == 0
        assert len(closes) == 0
        assert len(volumes) == 0


class TestBufferClear:
    """Test clearing the buffer."""

    def test_clear_buffer(self):
        """Test clearing all candles from the buffer."""
        buffer = CandleBuffer(max_size=10)
        for i in range(5):
            buffer.add_candle(
                timestamp=float(i),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1000.0,
            )

        assert buffer.size() == 5
        buffer.clear()
        assert buffer.size() == 0
        assert buffer.is_empty()


class TestMemoryUsage:
    """Test memory usage estimation."""

    def test_memory_usage_empty(self):
        """Test memory usage of empty buffer is 0."""
        buffer = CandleBuffer(max_size=10)
        assert buffer.get_memory_usage_bytes() == 0

    def test_memory_usage_with_candles(self):
        """Test memory usage estimation with candles."""
        buffer = CandleBuffer(max_size=10)
        for i in range(5):
            buffer.add_candle(
                timestamp=float(i),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1000.0,
            )

        # Each candle ~128 bytes, 5 candles ~640 bytes
        usage = buffer.get_memory_usage_bytes()
        assert usage > 0
        assert usage == 5 * 128  # 640 bytes

    def test_memory_usage_500_candles(self):
        """Test memory usage with 500 candles (R7.3.2)."""
        buffer = CandleBuffer(max_size=500)
        for i in range(500):
            buffer.add_candle(
                timestamp=float(i),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1000.0,
            )

        # 500 candles × 128 bytes = 64,000 bytes = ~64 KB
        usage = buffer.get_memory_usage_bytes()
        assert usage == 500 * 128  # 64,000 bytes
        assert usage < 100_000  # Less than 100 KB (very lightweight)


class TestSpecialMethods:
    """Test special methods (__len__, __repr__)."""

    def test_len_method(self):
        """Test __len__ method."""
        buffer = CandleBuffer(max_size=10)
        assert len(buffer) == 0

        buffer.add_candle(
            timestamp=1.0, open=100.0, high=101.0, low=99.0, close=100.5, volume=1000.0
        )
        assert len(buffer) == 1

    def test_repr_method(self):
        """Test __repr__ method."""
        buffer = CandleBuffer(max_size=10)
        assert repr(buffer) == "CandleBuffer(size=0, max_size=10)"

        buffer.add_candle(
            timestamp=1.0, open=100.0, high=101.0, low=99.0, close=100.5, volume=1000.0
        )
        assert repr(buffer) == "CandleBuffer(size=1, max_size=10)"


class TestPerformance:
    """Test performance of buffer operations."""

    def test_add_performance(self):
        """Test adding candles is fast (<1ms per candle)."""
        buffer = CandleBuffer(max_size=1000)

        start = time.time()
        for i in range(1000):
            buffer.add_candle(
                timestamp=float(i),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1000.0,
            )
        elapsed = time.time() - start

        # Should complete in well under 1 second (target: <100ms)
        assert elapsed < 1.0
        # Each candle should take <1ms
        assert elapsed / 1000 < 0.001

    def test_array_retrieval_performance(self):
        """Test retrieving arrays is fast."""
        buffer = CandleBuffer(max_size=1000)
        for i in range(1000):
            buffer.add_candle(
                timestamp=float(i),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1000.0,
            )

        start = time.time()
        for _ in range(100):
            _ = buffer.get_close_array()
            _ = buffer.get_ohlcv_arrays()
        elapsed = time.time() - start

        # 100 array retrievals should be very fast (<100ms)
        assert elapsed < 0.1


class TestSuccessCriteria:
    """Test success criteria from FLUXHERO_REQUIREMENTS.md Feature 7.3."""

    def test_r7_3_2_maintain_rolling_500_buffer(self):
        """Test R7.3.2: Maintain rolling 500-candle buffer in memory."""
        buffer = CandleBuffer(max_size=500)

        # Add 1000 candles
        for i in range(1000):
            buffer.add_candle(
                timestamp=float(i),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1000.0,
            )

        # Buffer should maintain exactly 500 candles
        assert buffer.size() == 500
        assert buffer.is_full()

        # Should have candles 500-999
        assert buffer.get_oldest_candle().timestamp == 500.0
        assert buffer.get_latest_candle().timestamp == 999.0

    def test_r7_3_3_discard_candles_older_than_500(self):
        """Test R7.3.3: Discard candles older than 500 bars."""
        buffer = CandleBuffer(max_size=500)

        # Add 600 candles
        for i in range(600):
            buffer.add_candle(
                timestamp=float(i),
                open=100.0 + i * 0.01,
                high=101.0 + i * 0.01,
                low=99.0 + i * 0.01,
                close=100.5 + i * 0.01,
                volume=1000.0,
            )

        # First 100 candles (0-99) should be discarded
        # Buffer should contain candles 100-599
        assert buffer.size() == 500

        # Verify oldest is candle 100, not candle 0
        oldest = buffer.get_oldest_candle()
        assert oldest.timestamp == 100.0
        assert oldest.close == 100.5 + 100 * 0.01  # Verify it's the right candle

        # Verify latest is candle 599
        latest = buffer.get_latest_candle()
        assert latest.timestamp == 599.0
        assert latest.close == 100.5 + 599 * 0.01

    def test_memory_efficiency(self):
        """Test buffer memory usage is reasonable for 500 candles."""
        buffer = CandleBuffer(max_size=500)
        for i in range(500):
            buffer.add_candle(
                timestamp=float(i),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1000.0,
                ema=100.2,
                atr=1.5,
                rsi=55.0,
            )

        # 500 candles should use ~64 KB (very lightweight)
        usage = buffer.get_memory_usage_bytes()
        assert usage == 500 * 128  # 64,000 bytes
        assert usage < 100_000  # Less than 100 KB


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_get_last_n_zero(self):
        """Test requesting 0 candles returns empty list."""
        buffer = CandleBuffer(max_size=10)
        buffer.add_candle(
            timestamp=1.0, open=100.0, high=101.0, low=99.0, close=100.5, volume=1000.0
        )
        assert buffer.get_last_n_candles(0) == []

    def test_get_last_n_negative(self):
        """Test requesting negative candles returns empty list."""
        buffer = CandleBuffer(max_size=10)
        buffer.add_candle(
            timestamp=1.0, open=100.0, high=101.0, low=99.0, close=100.5, volume=1000.0
        )
        assert buffer.get_last_n_candles(-5) == []

    def test_single_candle_buffer(self):
        """Test buffer with max_size=1."""
        buffer = CandleBuffer(max_size=1)
        buffer.add_candle(
            timestamp=1.0, open=100.0, high=101.0, low=99.0, close=100.5, volume=1000.0
        )
        assert buffer.size() == 1
        assert buffer.is_full()

        # Add second candle (should replace first)
        buffer.add_candle(
            timestamp=2.0, open=101.0, high=102.0, low=100.0, close=101.5, volume=1100.0
        )
        assert buffer.size() == 1
        assert buffer.get_oldest_candle().timestamp == 2.0
        assert buffer.get_latest_candle().timestamp == 2.0
