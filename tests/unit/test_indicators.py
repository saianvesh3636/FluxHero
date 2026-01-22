"""
Unit tests for JIT-compiled technical indicators.

Tests cover:
- Correctness: Verify calculations match expected values
- Edge cases: Empty arrays, insufficient data, NaN handling
- Performance: Benchmark against target metrics (<100ms for 10k candles)
"""

import time
import numpy as np
import pytest
from backend.computation.indicators import (
    calculate_ema,
    calculate_rsi,
    calculate_true_range,
    calculate_atr,
    calculate_sma,
    calculate_bollinger_bands,
)


class TestEMA:
    """Test Exponential Moving Average calculation."""

    def test_ema_basic_calculation(self):
        """Test EMA calculation with known values."""
        prices = np.array([10.0, 11.0, 12.0, 11.0, 13.0, 14.0], dtype=np.float64)
        ema = calculate_ema(prices, period=5)

        # First 4 values should be NaN
        assert np.isnan(ema[0])
        assert np.isnan(ema[1])
        assert np.isnan(ema[2])
        assert np.isnan(ema[3])

        # 5th value should be SMA of first 5 values
        expected_sma = (10.0 + 11.0 + 12.0 + 11.0 + 13.0) / 5.0
        assert np.isclose(ema[4], expected_sma)

        # Last value should use EMA formula
        alpha = 2.0 / (5 + 1)
        expected_ema = (14.0 * alpha) + (ema[4] * (1.0 - alpha))
        assert np.isclose(ema[5], expected_ema)

    def test_ema_insufficient_data(self):
        """Test EMA with insufficient data points."""
        prices = np.array([10.0, 11.0], dtype=np.float64)
        ema = calculate_ema(prices, period=5)

        # All values should be NaN
        assert all(np.isnan(ema))

    def test_ema_empty_array(self):
        """Test EMA with empty array."""
        prices = np.array([], dtype=np.float64)
        ema = calculate_ema(prices, period=5)

        assert len(ema) == 0

    def test_ema_performance_10k_candles(self):
        """Benchmark EMA calculation on 10,000 candles (target: <100ms)."""
        np.random.seed(42)
        prices = np.random.uniform(100, 200, 10000).astype(np.float64)

        # Warm up JIT compilation
        _ = calculate_ema(prices[:100], period=20)

        # Measure performance
        start = time.perf_counter()
        ema = calculate_ema(prices, period=20)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Verify result is valid
        assert len(ema) == 10000
        assert not np.all(np.isnan(ema))

        # Check performance target
        print(f"\nEMA (10k candles): {elapsed_ms:.2f}ms")
        assert elapsed_ms < 100, f"EMA too slow: {elapsed_ms:.2f}ms (target: <100ms)"


class TestRSI:
    """Test Relative Strength Index calculation."""

    def test_rsi_basic_calculation(self):
        """Test RSI calculation with known sequence."""
        # Create simple uptrend
        prices = np.array([
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0,
            108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0
        ], dtype=np.float64)
        rsi = calculate_rsi(prices, period=14)

        # First 14 values should be NaN
        assert all(np.isnan(rsi[:14]))

        # RSI in strong uptrend should be high (>70)
        assert rsi[14] > 70, f"Expected RSI > 70 in uptrend, got {rsi[14]:.2f}"
        assert rsi[15] > 70, f"Expected RSI > 70 in uptrend, got {rsi[15]:.2f}"

    def test_rsi_overbought_oversold(self):
        """Test RSI in overbought and oversold conditions."""
        # Strong downtrend (oversold)
        prices_down = np.array([
            100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0,
            92.0, 91.0, 90.0, 89.0, 88.0, 87.0, 86.0
        ], dtype=np.float64)
        rsi_down = calculate_rsi(prices_down, period=14)

        # RSI in downtrend should be low (<30)
        assert rsi_down[14] < 30, f"Expected RSI < 30 in downtrend, got {rsi_down[14]:.2f}"

    def test_rsi_range_bounds(self):
        """Test that RSI stays within 0-100 range."""
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(100)).astype(np.float64) + 100.0
        rsi = calculate_rsi(prices, period=14)

        valid_rsi = rsi[~np.isnan(rsi)]
        assert all((valid_rsi >= 0) & (valid_rsi <= 100)), "RSI out of range [0, 100]"

    def test_rsi_performance_10k_candles(self):
        """Benchmark RSI calculation on 10,000 candles."""
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(10000)).astype(np.float64) + 100.0

        # Warm up JIT
        _ = calculate_rsi(prices[:100], period=14)

        # Measure performance
        start = time.perf_counter()
        rsi = calculate_rsi(prices, period=14)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Verify result
        assert len(rsi) == 10000
        assert not np.all(np.isnan(rsi))

        print(f"RSI (10k candles): {elapsed_ms:.2f}ms")
        assert elapsed_ms < 100, f"RSI too slow: {elapsed_ms:.2f}ms (target: <100ms)"


class TestTrueRange:
    """Test True Range calculation."""

    def test_true_range_no_gap(self):
        """Test TR when there's no gap (High - Low)."""
        high = np.array([105.0, 106.0], dtype=np.float64)
        low = np.array([100.0, 101.0], dtype=np.float64)
        close = np.array([102.0, 103.0], dtype=np.float64)

        tr = calculate_true_range(high, low, close)

        # First TR is high - low
        assert tr[0] == 5.0

        # Second TR (no gap): high - low = 106 - 101 = 5
        assert tr[1] == 5.0

    def test_true_range_gap_up(self):
        """Test TR when price gaps up."""
        high = np.array([105.0, 115.0], dtype=np.float64)
        low = np.array([100.0, 110.0], dtype=np.float64)
        close = np.array([102.0, 112.0], dtype=np.float64)

        tr = calculate_true_range(high, low, close)

        # Gap up: High - Previous Close should be largest
        # Method 1: 115 - 110 = 5
        # Method 2: |115 - 102| = 13  <- Winner
        # Method 3: |110 - 102| = 8
        assert tr[1] == 13.0

    def test_true_range_gap_down(self):
        """Test TR when price gaps down."""
        high = np.array([105.0, 95.0], dtype=np.float64)
        low = np.array([100.0, 90.0], dtype=np.float64)
        close = np.array([103.0, 92.0], dtype=np.float64)

        tr = calculate_true_range(high, low, close)

        # Gap down: Previous Close - Low should be largest
        # Method 1: 95 - 90 = 5
        # Method 2: |95 - 103| = 8
        # Method 3: |90 - 103| = 13  <- Winner
        assert tr[1] == 13.0


class TestATR:
    """Test Average True Range calculation."""

    def test_atr_basic_calculation(self):
        """Test ATR calculation with known values."""
        # Create constant volatility scenario
        high = np.full(20, 105.0, dtype=np.float64)
        low = np.full(20, 100.0, dtype=np.float64)
        close = np.full(20, 102.5, dtype=np.float64)

        atr = calculate_atr(high, low, close, period=14)

        # First 14 values should be NaN
        assert all(np.isnan(atr[:14]))

        # ATR should be close to 5.0 (constant TR of 5)
        assert np.isclose(atr[14], 5.0, atol=0.1)

    def test_atr_insufficient_data(self):
        """Test ATR with insufficient data."""
        high = np.array([105.0, 106.0], dtype=np.float64)
        low = np.array([100.0, 101.0], dtype=np.float64)
        close = np.array([102.0, 103.0], dtype=np.float64)

        atr = calculate_atr(high, low, close, period=14)

        # All values should be NaN
        assert all(np.isnan(atr))

    def test_atr_performance_10k_candles(self):
        """Benchmark ATR calculation on 10,000 candles."""
        np.random.seed(42)
        base_price = 100.0
        prices = np.cumsum(np.random.randn(10000) * 0.5).astype(np.float64) + base_price

        # Create realistic OHLC data
        high = prices + np.random.uniform(0.5, 2.0, 10000)
        low = prices - np.random.uniform(0.5, 2.0, 10000)
        close = prices.copy()

        # Warm up JIT
        _ = calculate_atr(high[:100], low[:100], close[:100], period=14)

        # Measure performance
        start = time.perf_counter()
        atr = calculate_atr(high, low, close, period=14)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Verify result
        assert len(atr) == 10000
        assert not np.all(np.isnan(atr))

        print(f"ATR (10k candles): {elapsed_ms:.2f}ms")
        assert elapsed_ms < 100, f"ATR too slow: {elapsed_ms:.2f}ms (target: <100ms)"


class TestSMA:
    """Test Simple Moving Average calculation."""

    def test_sma_basic_calculation(self):
        """Test SMA calculation with known values."""
        prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0], dtype=np.float64)
        sma = calculate_sma(prices, period=3)

        # First 2 values should be NaN
        assert np.isnan(sma[0])
        assert np.isnan(sma[1])

        # Check calculated values
        assert sma[2] == 11.0  # (10 + 11 + 12) / 3
        assert sma[3] == 12.0  # (11 + 12 + 13) / 3
        assert sma[4] == 13.0  # (12 + 13 + 14) / 3


class TestBollingerBands:
    """Test Bollinger Bands calculation."""

    def test_bollinger_bands_basic(self):
        """Test Bollinger Bands calculation."""
        prices = np.array([
            100.0, 101.0, 102.0, 101.0, 100.0,
            99.0, 100.0, 101.0, 102.0, 103.0,
            104.0, 103.0, 102.0, 101.0, 100.0,
            101.0, 102.0, 103.0, 104.0, 105.0
        ], dtype=np.float64)

        upper, middle, lower = calculate_bollinger_bands(prices, period=20, num_std=2.0)

        # First 19 values should be NaN
        assert all(np.isnan(upper[:19]))
        assert all(np.isnan(middle[:19]))
        assert all(np.isnan(lower[:19]))

        # At position 19, bands should be calculated
        assert not np.isnan(upper[19])
        assert not np.isnan(middle[19])
        assert not np.isnan(lower[19])

        # Upper band should be above middle, middle above lower
        assert upper[19] > middle[19]
        assert middle[19] > lower[19]

        # Bands should be symmetric around middle
        assert np.isclose(upper[19] - middle[19], middle[19] - lower[19])


class TestFullIndicatorSuite:
    """Test performance of all indicators together."""

    def test_full_suite_performance_10k_candles(self):
        """
        Benchmark full indicator suite on 10,000 candles.
        Target: <500ms for EMA + RSI + ATR + SMA + Bollinger Bands
        """
        np.random.seed(42)
        base_price = 100.0
        prices = np.cumsum(np.random.randn(10000) * 0.5).astype(np.float64) + base_price

        # Create OHLC data
        high = prices + np.random.uniform(0.5, 2.0, 10000)
        low = prices - np.random.uniform(0.5, 2.0, 10000)
        close = prices.copy()

        # Warm up JIT for all functions
        _ = calculate_ema(close[:100], period=20)
        _ = calculate_rsi(close[:100], period=14)
        _ = calculate_atr(high[:100], low[:100], close[:100], period=14)
        _ = calculate_sma(close[:100], period=20)
        _ = calculate_bollinger_bands(close[:100], period=20, num_std=2.0)

        # Measure full suite performance
        start = time.perf_counter()

        ema_12 = calculate_ema(close, period=12)
        ema_26 = calculate_ema(close, period=26)
        rsi = calculate_rsi(close, period=14)
        atr = calculate_atr(high, low, close, period=14)
        sma_50 = calculate_sma(close, period=50)
        upper, middle, lower = calculate_bollinger_bands(close, period=20, num_std=2.0)

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Verify all results are valid
        assert len(ema_12) == 10000
        assert len(ema_26) == 10000
        assert len(rsi) == 10000
        assert len(atr) == 10000
        assert len(sma_50) == 10000
        assert len(upper) == 10000

        # Check performance target
        print(f"\nFull indicator suite (10k candles): {elapsed_ms:.2f}ms")
        assert elapsed_ms < 500, f"Suite too slow: {elapsed_ms:.2f}ms (target: <500ms)"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_constant_prices_rsi(self):
        """Test RSI with constant prices (no movement)."""
        prices = np.full(20, 100.0, dtype=np.float64)
        rsi = calculate_rsi(prices, period=14)

        # RSI should be NaN or 100 (no losses)
        valid_rsi = rsi[~np.isnan(rsi)]
        if len(valid_rsi) > 0:
            assert all(valid_rsi == 100.0), "RSI should be 100 with no price movement"

    def test_negative_prices_handled(self):
        """Test that indicators handle negative prices gracefully."""
        # Technical indicators should work with any numeric values
        prices = np.array([-10.0, -11.0, -12.0, -11.0, -10.0, -9.0], dtype=np.float64)
        ema = calculate_ema(prices, period=5)

        # Should complete without error
        assert len(ema) == 6
        assert not np.isnan(ema[4])  # First valid EMA

    def test_single_value_array(self):
        """Test indicators with single value array."""
        prices = np.array([100.0], dtype=np.float64)

        ema = calculate_ema(prices, period=5)
        rsi = calculate_rsi(prices, period=14)
        sma = calculate_sma(prices, period=3)

        # All should return arrays with NaN (insufficient data)
        assert len(ema) == 1 and np.isnan(ema[0])
        assert len(rsi) == 1 and np.isnan(rsi[0])
        assert len(sma) == 1 and np.isnan(sma[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
