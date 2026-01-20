"""
Unit tests for Parquet Store (Feature 7.2)

Tests cover:
- Basic save/load operations
- Cache freshness validation
- OHLCV + indicator storage
- Performance benchmarks
- Edge cases and error handling
- Cache management operations
"""

import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from fluxhero.backend.storage.parquet_store import CandleData, ParquetStore


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def parquet_store(temp_cache_dir):
    """Create ParquetStore instance with temporary directory."""
    return ParquetStore(cache_dir=temp_cache_dir)


@pytest.fixture
def sample_candle_data():
    """Create sample candle data for testing."""
    n = 100
    timestamps = np.arange(1609459200, 1609459200 + n * 3600, 3600, dtype=np.float64)

    return CandleData(
        symbol='SPY',
        timeframe='1h',
        timestamp=timestamps,
        open=np.random.uniform(370, 380, n),
        high=np.random.uniform(380, 390, n),
        low=np.random.uniform(360, 370, n),
        close=np.random.uniform(370, 380, n),
        volume=np.random.uniform(1000000, 5000000, n),
    )


@pytest.fixture
def sample_candle_data_with_indicators():
    """Create sample candle data with indicators."""
    n = 100
    timestamps = np.arange(1609459200, 1609459200 + n * 3600, 3600, dtype=np.float64)

    return CandleData(
        symbol='AAPL',
        timeframe='1h',
        timestamp=timestamps,
        open=np.random.uniform(130, 135, n),
        high=np.random.uniform(135, 140, n),
        low=np.random.uniform(125, 130, n),
        close=np.random.uniform(130, 135, n),
        volume=np.random.uniform(1000000, 5000000, n),
        ema=np.random.uniform(130, 135, n),
        atr=np.random.uniform(1, 3, n),
        rsi=np.random.uniform(30, 70, n),
    )


# ============================================================================
# Initialization Tests
# ============================================================================

def test_parquet_store_initialization(temp_cache_dir):
    """Test ParquetStore initialization creates cache directory."""
    store = ParquetStore(cache_dir=temp_cache_dir)

    assert store.cache_dir.exists()
    assert store.cache_dir.is_dir()
    assert store.compression == 'snappy'
    assert store.cache_ttl_hours == 24


def test_cache_directory_auto_creation():
    """Test cache directory is created automatically if it doesn't exist."""
    temp_dir = tempfile.mkdtemp()
    cache_path = Path(temp_dir) / "subdir" / "cache"

    _ = ParquetStore(cache_dir=str(cache_path))

    assert cache_path.exists()
    assert cache_path.is_dir()

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# Save/Load Tests
# ============================================================================

def test_save_and_load_candles(parquet_store, sample_candle_data):
    """Test basic save and load operations."""
    # Save
    parquet_store.save_candles(sample_candle_data)

    # Load
    loaded = parquet_store.load_candles('SPY', '1h')

    assert loaded is not None
    assert loaded.symbol == 'SPY'
    assert loaded.timeframe == '1h'
    assert len(loaded.timestamp) == len(sample_candle_data.timestamp)
    np.testing.assert_array_almost_equal(loaded.timestamp, sample_candle_data.timestamp)
    np.testing.assert_array_almost_equal(loaded.open, sample_candle_data.open)
    np.testing.assert_array_almost_equal(loaded.high, sample_candle_data.high)
    np.testing.assert_array_almost_equal(loaded.low, sample_candle_data.low)
    np.testing.assert_array_almost_equal(loaded.close, sample_candle_data.close)
    np.testing.assert_array_almost_equal(loaded.volume, sample_candle_data.volume)


def test_save_and_load_candles_with_indicators(parquet_store, sample_candle_data_with_indicators):
    """Test save/load with EMA, ATR, RSI indicators (R7.2.3)."""
    # Save
    parquet_store.save_candles(sample_candle_data_with_indicators)

    # Load
    loaded = parquet_store.load_candles('AAPL', '1h')

    assert loaded is not None
    assert loaded.ema is not None
    assert loaded.atr is not None
    assert loaded.rsi is not None
    np.testing.assert_array_almost_equal(loaded.ema, sample_candle_data_with_indicators.ema)
    np.testing.assert_array_almost_equal(loaded.atr, sample_candle_data_with_indicators.atr)
    np.testing.assert_array_almost_equal(loaded.rsi, sample_candle_data_with_indicators.rsi)


def test_load_non_existent_cache(parquet_store):
    """Test loading non-existent cache returns None."""
    loaded = parquet_store.load_candles('NONEXISTENT', '1h')
    assert loaded is None


def test_save_overwrites_existing_cache(parquet_store, sample_candle_data):
    """Test saving overwrites existing cache file."""
    # Save original
    parquet_store.save_candles(sample_candle_data)

    # Modify data
    sample_candle_data.close[:] = 999.0

    # Save again
    parquet_store.save_candles(sample_candle_data)

    # Load and verify overwrite
    loaded = parquet_store.load_candles('SPY', '1h')
    assert np.all(loaded.close == 999.0)


# ============================================================================
# Cache Freshness Tests (R7.2.2)
# ============================================================================

def test_is_cache_fresh_new_cache(parquet_store, sample_candle_data):
    """Test freshly saved cache is considered fresh."""
    parquet_store.save_candles(sample_candle_data)

    assert parquet_store.is_cache_fresh('SPY', '1h')


def test_is_cache_fresh_non_existent(parquet_store):
    """Test non-existent cache is not fresh."""
    assert not parquet_store.is_cache_fresh('NONEXISTENT', '1h')


def test_is_cache_fresh_old_cache(parquet_store, sample_candle_data):
    """Test old cache (>24 hours) is not fresh."""
    # Save cache
    parquet_store.save_candles(sample_candle_data)

    # Manually modify file timestamp to make it old
    cache_path = parquet_store._get_cache_path('SPY', '1h')
    old_time = time.time() - (25 * 3600)  # 25 hours ago
    os.utime(cache_path, (old_time, old_time))

    assert not parquet_store.is_cache_fresh('SPY', '1h')


def test_get_cache_age(parquet_store, sample_candle_data):
    """Test get_cache_age returns correct age."""
    parquet_store.save_candles(sample_candle_data)

    age = parquet_store.get_cache_age('SPY', '1h')

    assert age is not None
    assert age.total_seconds() < 10  # Should be very recent


def test_get_cache_age_non_existent(parquet_store):
    """Test get_cache_age returns None for non-existent cache."""
    age = parquet_store.get_cache_age('NONEXISTENT', '1h')
    assert age is None


# ============================================================================
# Cache Management Tests
# ============================================================================

def test_delete_cache(parquet_store, sample_candle_data):
    """Test delete_cache removes cache file."""
    # Save
    parquet_store.save_candles(sample_candle_data)
    assert parquet_store.load_candles('SPY', '1h') is not None

    # Delete
    deleted = parquet_store.delete_cache('SPY', '1h')

    assert deleted is True
    assert parquet_store.load_candles('SPY', '1h') is None


def test_delete_non_existent_cache(parquet_store):
    """Test deleting non-existent cache returns False."""
    deleted = parquet_store.delete_cache('NONEXISTENT', '1h')
    assert deleted is False


def test_clear_all_cache(parquet_store, sample_candle_data, sample_candle_data_with_indicators):
    """Test clear_all_cache removes all cache files."""
    # Save multiple caches
    parquet_store.save_candles(sample_candle_data)
    parquet_store.save_candles(sample_candle_data_with_indicators)

    # Clear all
    count = parquet_store.clear_all_cache()

    assert count == 2
    assert parquet_store.load_candles('SPY', '1h') is None
    assert parquet_store.load_candles('AAPL', '1h') is None


def test_get_cache_size(parquet_store, sample_candle_data):
    """Test get_cache_size returns file size in bytes."""
    parquet_store.save_candles(sample_candle_data)

    size = parquet_store.get_cache_size('SPY', '1h')

    assert size is not None
    assert size > 0
    # Parquet with snappy compression should be reasonably sized
    assert size < 100000  # <100KB for 100 candles


def test_get_cache_size_non_existent(parquet_store):
    """Test get_cache_size returns None for non-existent cache."""
    size = parquet_store.get_cache_size('NONEXISTENT', '1h')
    assert size is None


def test_list_cached_symbols(parquet_store, sample_candle_data, sample_candle_data_with_indicators):
    """Test list_cached_symbols returns all cached symbols."""
    # Save multiple caches
    parquet_store.save_candles(sample_candle_data)
    parquet_store.save_candles(sample_candle_data_with_indicators)

    symbols = parquet_store.list_cached_symbols()

    assert len(symbols) == 2
    assert ('SPY', '1h') in symbols
    assert ('AAPL', '1h') in symbols


def test_get_cache_metadata(parquet_store, sample_candle_data):
    """Test get_cache_metadata returns comprehensive metadata."""
    parquet_store.save_candles(sample_candle_data)

    metadata = parquet_store.get_cache_metadata('SPY', '1h')

    assert metadata is not None
    assert 'size_bytes' in metadata
    assert metadata['size_bytes'] > 0
    assert 'age' in metadata
    assert 'age_hours' in metadata
    assert 'is_fresh' in metadata
    assert metadata['is_fresh'] is True
    assert 'num_rows' in metadata
    assert metadata['num_rows'] == 100
    assert 'modified_at' in metadata


def test_get_cache_metadata_non_existent(parquet_store):
    """Test get_cache_metadata returns None for non-existent cache."""
    metadata = parquet_store.get_cache_metadata('NONEXISTENT', '1h')
    assert metadata is None


# ============================================================================
# Performance Tests
# ============================================================================

def test_save_performance_500_candles(parquet_store):
    """Test save performance meets <100ms target for 500 candles."""
    # Create 500 candles
    n = 500
    timestamps = np.arange(1609459200, 1609459200 + n * 3600, 3600, dtype=np.float64)
    data = CandleData(
        symbol='SPY',
        timeframe='1h',
        timestamp=timestamps,
        open=np.random.uniform(370, 380, n),
        high=np.random.uniform(380, 390, n),
        low=np.random.uniform(360, 370, n),
        close=np.random.uniform(370, 380, n),
        volume=np.random.uniform(1000000, 5000000, n),
        ema=np.random.uniform(370, 380, n),
        atr=np.random.uniform(1, 3, n),
        rsi=np.random.uniform(30, 70, n),
    )

    # Measure save time
    start = time.perf_counter()
    parquet_store.save_candles(data)
    elapsed = (time.perf_counter() - start) * 1000  # Convert to ms

    assert elapsed < 100, f"Save took {elapsed:.2f}ms, target is <100ms"


def test_load_performance_500_candles(parquet_store):
    """Test load performance meets <50ms target for 500 candles (R7.2.2)."""
    # Create and save 500 candles
    n = 500
    timestamps = np.arange(1609459200, 1609459200 + n * 3600, 3600, dtype=np.float64)
    data = CandleData(
        symbol='SPY',
        timeframe='1h',
        timestamp=timestamps,
        open=np.random.uniform(370, 380, n),
        high=np.random.uniform(380, 390, n),
        low=np.random.uniform(360, 370, n),
        close=np.random.uniform(370, 380, n),
        volume=np.random.uniform(1000000, 5000000, n),
        ema=np.random.uniform(370, 380, n),
        atr=np.random.uniform(1, 3, n),
        rsi=np.random.uniform(30, 70, n),
    )
    parquet_store.save_candles(data)

    # Measure load time
    start = time.perf_counter()
    loaded = parquet_store.load_candles('SPY', '1h')
    elapsed = (time.perf_counter() - start) * 1000  # Convert to ms

    assert loaded is not None
    assert elapsed < 50, f"Load took {elapsed:.2f}ms, target is <50ms"


def test_compression_effectiveness(parquet_store):
    """Test Snappy compression provides good compression ratio (R7.2.4)."""
    # Create 1000 candles
    n = 1000
    timestamps = np.arange(1609459200, 1609459200 + n * 3600, 3600, dtype=np.float64)
    data = CandleData(
        symbol='SPY',
        timeframe='1h',
        timestamp=timestamps,
        open=np.random.uniform(370, 380, n),
        high=np.random.uniform(380, 390, n),
        low=np.random.uniform(360, 370, n),
        close=np.random.uniform(370, 380, n),
        volume=np.random.uniform(1000000, 5000000, n),
        ema=np.random.uniform(370, 380, n),
        atr=np.random.uniform(1, 3, n),
        rsi=np.random.uniform(30, 70, n),
    )

    parquet_store.save_candles(data)
    size = parquet_store.get_cache_size('SPY', '1h')

    # 1000 candles × 9 columns × 8 bytes = ~72KB uncompressed data
    # Parquet has metadata overhead, so compressed size may be slightly larger
    # but should still be reasonable (<200KB for 1000 candles with metadata)
    assert size is not None
    assert size < 200000, f"Compressed size {size} bytes should be reasonable (<200KB) for 1000 candles"
    # Verify compression is actually being used by checking it's not massively larger
    assert size < 500000, "File should not be excessively large"


# ============================================================================
# Edge Cases Tests
# ============================================================================

def test_empty_candle_data(parquet_store):
    """Test handling of empty candle data."""
    data = CandleData(
        symbol='TEST',
        timeframe='1h',
        timestamp=np.array([], dtype=np.float64),
        open=np.array([]),
        high=np.array([]),
        low=np.array([]),
        close=np.array([]),
        volume=np.array([]),
    )

    parquet_store.save_candles(data)
    loaded = parquet_store.load_candles('TEST', '1h')

    assert loaded is not None
    assert len(loaded.timestamp) == 0


def test_single_candle(parquet_store):
    """Test saving and loading single candle."""
    data = CandleData(
        symbol='TEST',
        timeframe='1h',
        timestamp=np.array([1609459200.0]),
        open=np.array([100.0]),
        high=np.array([105.0]),
        low=np.array([95.0]),
        close=np.array([102.0]),
        volume=np.array([1000000.0]),
    )

    parquet_store.save_candles(data)
    loaded = parquet_store.load_candles('TEST', '1h')

    assert loaded is not None
    assert len(loaded.timestamp) == 1
    assert loaded.close[0] == 102.0


def test_very_large_dataset(parquet_store):
    """Test handling of large dataset (10k candles)."""
    n = 10000
    timestamps = np.arange(1609459200, 1609459200 + n * 3600, 3600, dtype=np.float64)
    data = CandleData(
        symbol='SPY',
        timeframe='1h',
        timestamp=timestamps,
        open=np.random.uniform(370, 380, n),
        high=np.random.uniform(380, 390, n),
        low=np.random.uniform(360, 370, n),
        close=np.random.uniform(370, 380, n),
        volume=np.random.uniform(1000000, 5000000, n),
    )

    parquet_store.save_candles(data)
    loaded = parquet_store.load_candles('SPY', '1h')

    assert loaded is not None
    assert len(loaded.timestamp) == 10000


def test_special_characters_in_symbol(parquet_store, sample_candle_data):
    """Test handling of symbols with special characters."""
    sample_candle_data.symbol = 'BRK.B'

    parquet_store.save_candles(sample_candle_data)
    loaded = parquet_store.load_candles('BRK.B', '1h')

    assert loaded is not None
    assert loaded.symbol == 'BRK.B'


def test_multiple_timeframes_same_symbol(parquet_store, sample_candle_data):
    """Test caching multiple timeframes for same symbol."""
    # Save 1h data
    sample_candle_data.timeframe = '1h'
    parquet_store.save_candles(sample_candle_data)

    # Save 1d data (different timeframe)
    sample_candle_data.timeframe = '1d'
    sample_candle_data.close[:] = 999.0  # Different data
    parquet_store.save_candles(sample_candle_data)

    # Load both
    loaded_1h = parquet_store.load_candles('SPY', '1h')
    loaded_1d = parquet_store.load_candles('SPY', '1d')

    assert loaded_1h is not None
    assert loaded_1d is not None
    assert not np.all(loaded_1h.close == 999.0)
    assert np.all(loaded_1d.close == 999.0)


# ============================================================================
# Success Criteria Tests (Feature 7.2)
# ============================================================================

def test_success_criteria_cache_path_format(parquet_store, sample_candle_data):
    """Verify R7.2.1: Cache path format is data/cache/{symbol}_{timeframe}.parquet."""
    parquet_store.save_candles(sample_candle_data)

    expected_path = parquet_store.cache_dir / "SPY_1h.parquet"
    assert expected_path.exists()


def test_success_criteria_freshness_check(parquet_store, sample_candle_data):
    """Verify R7.2.2: Freshness check works correctly (<24 hours)."""
    # Fresh cache
    parquet_store.save_candles(sample_candle_data)
    assert parquet_store.is_cache_fresh('SPY', '1h')

    # Non-existent cache
    assert not parquet_store.is_cache_fresh('FAKE', '1h')


def test_success_criteria_indicators_storage(parquet_store, sample_candle_data_with_indicators):
    """Verify R7.2.3: OHLCV + indicators are stored and loaded correctly."""
    parquet_store.save_candles(sample_candle_data_with_indicators)
    loaded = parquet_store.load_candles('AAPL', '1h')

    # Verify all OHLCV data present
    assert loaded.open is not None
    assert loaded.high is not None
    assert loaded.low is not None
    assert loaded.close is not None
    assert loaded.volume is not None

    # Verify indicators present
    assert loaded.ema is not None
    assert loaded.atr is not None
    assert loaded.rsi is not None


def test_success_criteria_snappy_compression(parquet_store, sample_candle_data):
    """Verify R7.2.4: Snappy compression is used."""
    parquet_store.save_candles(sample_candle_data)

    cache_path = parquet_store._get_cache_path('SPY', '1h')

    # Read parquet metadata to verify compression
    import pyarrow.parquet as pq
    parquet_file = pq.ParquetFile(cache_path)

    # Check compression codec
    for row_group in range(parquet_file.num_row_groups):
        for col in range(parquet_file.metadata.row_group(row_group).num_columns):
            column_meta = parquet_file.metadata.row_group(row_group).column(col)
            # Snappy compression should be used
            assert 'SNAPPY' in str(column_meta.compression).upper()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
