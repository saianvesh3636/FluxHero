"""
Full Integration Test for FluxHero System

Tests the complete workflow:
1. System startup and data loading
2. Indicator calculations
3. Signal generation
4. Regime detection
5. Storage operations

This validates that all components work together correctly.
"""

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.computation.adaptive_ema import calculate_kama_with_regime_adjustment  # noqa: E402
from backend.computation.indicators import (  # noqa: E402
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_rsi,
)
from backend.computation.volatility import calculate_atr_ma  # noqa: E402
from backend.storage.candle_buffer import CandleBuffer  # noqa: E402
from backend.storage.parquet_store import CandleData, ParquetStore  # noqa: E402
from backend.strategy.dual_mode import (  # noqa: E402
    DualModeStrategy,
    generate_mean_reversion_signals,
    generate_trend_following_signals,
)
from backend.strategy.regime_detector import detect_regime  # noqa: E402


def generate_synthetic_market_data(num_candles: int = 500, regime: str = "trending"):
    """
    Generate synthetic market data for testing.

    Args:
        num_candles: Number of candles to generate
        regime: "trending" (upward bias) or "choppy" (sideways)

    Returns:
        Tuple of (timestamps, opens, highs, lows, closes, volumes)
    """
    np.random.seed(42)

    # Start timestamp
    start_time = datetime.now() - timedelta(days=num_candles // 390)
    timestamps = np.array([
        int((start_time + timedelta(minutes=i)).timestamp())
        for i in range(num_candles)
    ])

    # Generate prices based on regime
    if regime == "trending":
        # Trending market: upward drift with noise
        trend = np.linspace(100, 120, num_candles)
        noise = np.random.normal(0, 0.5, num_candles)
        closes = trend + noise
    else:  # choppy
        # Choppy market: mean-reverting around 100
        closes = 100 + np.random.normal(0, 2, num_candles)
        # Add some autocorrelation
        for i in range(1, num_candles):
            closes[i] = 0.7 * closes[i-1] + 0.3 * closes[i]

    # Generate OHLC from closes
    opens = np.roll(closes, 1)
    opens[0] = closes[0]

    highs = closes + np.abs(np.random.normal(0, 0.3, num_candles))
    lows = closes - np.abs(np.random.normal(0, 0.3, num_candles))

    # Ensure high >= close and low <= close
    highs = np.maximum(highs, closes)
    lows = np.minimum(lows, closes)

    # Ensure high >= open and low <= open
    highs = np.maximum(highs, opens)
    lows = np.minimum(lows, opens)

    # Generate volumes
    volumes = np.random.randint(1000, 5000, num_candles).astype(np.float64)

    return timestamps, opens, highs, lows, closes, volumes


def test_full_integration_workflow(tmp_path):
    """
    Test complete workflow: startup → data fetch → indicators → signals → regime → storage.

    This is the main integration test that validates all components work together.
    """
    print("\n=== Starting Full Integration Test ===\n")

    # 1. SYSTEM STARTUP: Initialize storage components
    print("1. Initializing storage components...")
    parquet_store = ParquetStore(str(tmp_path))
    candle_buffer = CandleBuffer(max_size=500)
    print("   ✓ Storage initialized")

    # 2. DATA FETCH: Generate synthetic market data
    print("\n2. Fetching market data...")
    num_candles = 500
    timestamps, opens, highs, lows, closes, volumes = generate_synthetic_market_data(
        num_candles, regime="trending"
    )

    # Add candles to buffer
    for i in range(num_candles):
        candle_buffer.add_candle(
            timestamp=timestamps[i],
            open=opens[i],
            high=highs[i],
            low=lows[i],
            close=closes[i],
            volume=volumes[i],
        )

    assert candle_buffer.size() == num_candles
    print(f"   ✓ Loaded {num_candles} candles into buffer")

    # Cache to Parquet
    candle_data = CandleData(
        symbol="SPY",
        timeframe="1min",
        timestamp=timestamps,
        open=opens,
        high=highs,
        low=lows,
        close=closes,
        volume=volumes,
    )
    parquet_store.save_candles(candle_data)
    assert parquet_store.is_cache_fresh("SPY", "1min")
    print("   ✓ Data cached to Parquet")

    # 3. INDICATOR CALCULATIONS
    print("\n3. Calculating indicators...")
    open_array, high_array, low_array, close_array, volume_array = (
        candle_buffer.get_ohlcv_arrays()
    )

    start_time = time.perf_counter()

    ema_20 = calculate_ema(close_array, period=20)
    atr = calculate_atr(high_array, low_array, close_array, period=14)
    rsi = calculate_rsi(close_array, period=14)
    kama, efficiency_ratio, regime_kama = calculate_kama_with_regime_adjustment(
        close_array, er_period=10, fast_period=2, slow_period=30
    )
    upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(
        close_array, period=20, num_std=2.0
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Verify indicators computed successfully
    assert not np.all(np.isnan(ema_20))
    assert not np.all(np.isnan(atr))
    assert not np.all(np.isnan(rsi))
    assert not np.all(np.isnan(kama))
    print(f"   ✓ Indicators calculated in {elapsed_ms:.1f}ms")

    # 4. REGIME DETECTION
    print("\n4. Detecting market regime...")
    atr_ma = calculate_atr_ma(atr, period=50)
    regime_result = detect_regime(
        high_array, low_array, close_array, atr, atr_ma,
        adx_period=14, regression_period=50
    )
    regime_trend = regime_result["trend_regime_confirmed"]
    regime_volatility = regime_result["volatility_regime"]

    # Verify regime detection worked
    assert not np.any(np.isnan(regime_trend)), "Regime trend contains NaN values"
    assert not np.any(np.isnan(regime_volatility)), "Regime volatility contains NaN values"

    # Count regime distribution
    trend_bars = np.sum(regime_trend == 2)
    neutral_bars = np.sum(regime_trend == 1)
    mr_bars = np.sum(regime_trend == 0)
    print(f"   ✓ Regime detected: TREND={trend_bars}, NEUTRAL={neutral_bars}, MR={mr_bars}")

    # 5. SIGNAL GENERATION
    print("\n5. Generating trading signals...")

    # Trend-following signals
    tf_signals = generate_trend_following_signals(
        close_array, kama, atr,
        entry_multiplier=0.5, exit_multiplier=0.3
    )

    # Mean-reversion signals
    mr_signals = generate_mean_reversion_signals(
        close_array, rsi, lower_bb, middle_bb,
        rsi_oversold=30, rsi_overbought=70
    )

    # Verify signals were generated
    assert tf_signals.shape == close_array.shape
    assert mr_signals.shape == close_array.shape

    tf_signal_count = np.sum(tf_signals != 0)
    mr_signal_count = np.sum(mr_signals != 0)
    print(f"   ✓ Generated {tf_signal_count} trend-following and {mr_signal_count} mean-reversion signals")

    # 6. DUAL-MODE STRATEGY
    print("\n6. Testing dual-mode strategy...")
    strategy = DualModeStrategy()

    # Test strategy mode selection based on regime
    test_regime_values = [0, 1, 2]  # MR, NEUTRAL, TREND
    for regime_val in test_regime_values:
        active_mode = strategy.get_active_mode(regime_val)
        assert active_mode in [1, 2, 3], f"Invalid mode {active_mode} for regime {regime_val}"

    print("   ✓ Dual-mode strategy adapts to regime")

    # 7. PARQUET CACHE OPERATIONS
    print("\n7. Testing cache operations...")

    # Load cached data
    loaded_data = parquet_store.load_candles("SPY", "1min")
    assert loaded_data is not None
    assert len(loaded_data.close) == num_candles
    assert np.allclose(loaded_data.close, closes)

    # Check cache metadata
    metadata = parquet_store.get_cache_metadata("SPY", "1min")
    assert metadata is not None
    assert metadata["num_rows"] == num_candles
    print(f"   ✓ Cache validated: {metadata['num_rows']} rows, {metadata['size_bytes']} bytes")

    print("\n=== Integration Test Completed Successfully ===\n")


def test_integration_performance_benchmarks():
    """
    Test that indicator calculations meet performance targets.

    Target: 10k candles in <500ms for full indicator suite.
    """
    print("\n=== Performance Benchmark Test ===\n")

    # Generate 10k candles
    num_candles = 10000
    timestamps, opens, highs, lows, closes, volumes = generate_synthetic_market_data(
        num_candles, regime="trending"
    )

    # Benchmark full indicator suite
    start_time = time.perf_counter()

    # Calculate all indicators
    calculate_ema(closes, period=20)
    calculate_ema(closes, period=50)
    calculate_atr(highs, lows, closes, period=14)
    calculate_rsi(closes, period=14)
    kama, er, regime = calculate_kama_with_regime_adjustment(
        closes, er_period=10, fast_period=2, slow_period=30
    )
    upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(
        closes, period=20, num_std=2.0
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Target: <500ms for 10k candles
    assert elapsed_ms < 500, (
        f"Indicator suite took {elapsed_ms:.1f}ms, expected <500ms"
    )

    print(f"✓ Full indicator suite on 10k candles: {elapsed_ms:.1f}ms (target: <500ms)\n")


def test_integration_system_startup_sequence(tmp_path):
    """
    Test the system startup sequence completes successfully.

    Steps:
    1. Initialize storage (Parquet, Buffer)
    2. Load cached data or fetch from API
    3. Calculate indicators
    4. Ready for signal generation
    """
    print("\n=== System Startup Sequence Test ===\n")

    # 1. Initialize storage
    print("1. Initializing storage...")
    parquet_store = ParquetStore(str(tmp_path))
    candle_buffer = CandleBuffer(max_size=500)
    print("   ✓ Storage initialized")

    # 2. Simulate data load (cache miss → fetch from API)
    print("\n2. Loading data...")
    cached = parquet_store.load_candles("SPY", "1min")
    assert cached is None  # Cache miss (first run)

    # Fetch data (simulated)
    timestamps, opens, highs, lows, closes, volumes = generate_synthetic_market_data(
        500, regime="trending"
    )

    # Add to buffer
    for i in range(len(timestamps)):
        candle_buffer.add_candle(
            timestamp=timestamps[i],
            open=opens[i],
            high=highs[i],
            low=lows[i],
            close=closes[i],
            volume=volumes[i],
        )

    print(f"   ✓ Loaded {len(timestamps)} candles")

    # Cache for next startup
    candle_data = CandleData(
        symbol="SPY",
        timeframe="1min",
        timestamp=timestamps,
        open=opens,
        high=highs,
        low=lows,
        close=closes,
        volume=volumes,
    )
    parquet_store.save_candles(candle_data)
    print("   ✓ Data cached for next startup")

    # 3. Calculate indicators
    print("\n3. Calculating indicators...")
    close_array = candle_buffer.get_close_array()
    ema = calculate_ema(close_array, period=20)
    print("   ✓ Indicators calculated")

    # 4. Verify system is ready
    print("\n4. Verifying system readiness...")
    assert candle_buffer.size() == 500
    assert parquet_store.is_cache_fresh("SPY", "1min")
    assert not np.all(np.isnan(ema))
    print("   ✓ System ready for trading")

    print("\n=== Startup Sequence Completed Successfully ===\n")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
