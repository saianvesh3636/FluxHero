"""
Performance Benchmark Tests

Tests that indicator calculations meet performance targets:
- 10k candle calculations must complete in <500ms per indicator
- All core indicators must pass this threshold
- Validates both individual and composite calculations

Requirements Coverage:
- Phase 17 Task 7: Performance benchmark for 10k candles
"""

import time
import numpy as np
import pytest
from backend.computation.indicators import calculate_ema, calculate_rsi, calculate_atr
from backend.computation.adaptive_ema import calculate_kama
from backend.computation.volatility import classify_volatility_state
from backend.strategy.regime_detector import detect_regime, calculate_adx


class TestPerformanceBenchmark:
    """Performance benchmark tests for 10k candle calculations."""

    @pytest.fixture
    def large_dataset(self):
        """Generate realistic 10k candle dataset for benchmarking."""
        np.random.seed(42)
        n = 10000

        # Simulate realistic price movement with trend + noise
        trend = np.linspace(100, 120, n)
        noise = np.random.randn(n) * 2
        close_prices = trend + noise

        # Generate OHLC data
        high_prices = close_prices + np.abs(np.random.randn(n) * 0.5)
        low_prices = close_prices - np.abs(np.random.randn(n) * 0.5)
        open_prices = close_prices + np.random.randn(n) * 0.3

        # Ensure OHLC consistency (High >= Close >= Low, etc.)
        high_prices = np.maximum(high_prices, close_prices)
        low_prices = np.minimum(low_prices, close_prices)

        volume = np.random.randint(100000, 1000000, size=n).astype(np.float64)

        # Warmup JIT compilation with small dataset
        warmup_close = close_prices[:100]
        warmup_high = high_prices[:100]
        warmup_low = low_prices[:100]

        # Warmup all JIT functions
        calculate_ema(warmup_close, 20)
        calculate_rsi(warmup_close, 14)
        calculate_atr(warmup_high, warmup_low, warmup_close, 14)
        calculate_kama(warmup_close, 10, 2, 30)

        from backend.computation.volatility import calculate_atr_ma
        warmup_atr = calculate_atr(warmup_high, warmup_low, warmup_close, 14)
        warmup_atr_ma = calculate_atr_ma(warmup_atr, 50)
        calculate_adx(warmup_high, warmup_low, warmup_close, warmup_atr, 14)
        classify_volatility_state(warmup_atr, warmup_atr_ma)
        detect_regime(warmup_high, warmup_low, warmup_close, warmup_atr, warmup_atr_ma)

        return {
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume,
        }

    def benchmark_function(self, func, *args, target_ms=500):
        """
        Execute function and measure execution time.

        Args:
            func: Function to benchmark
            *args: Arguments to pass to function
            target_ms: Target execution time in milliseconds

        Returns:
            Tuple of (result, execution_time_ms, passed)
        """
        start = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()

        execution_ms = (end - start) * 1000
        passed = execution_ms < target_ms

        return result, execution_ms, passed

    def test_ema_performance(self, large_dataset):
        """Test EMA calculation performance on 10k candles."""
        close_prices = large_dataset["close"]
        period = 20

        result, exec_time, passed = self.benchmark_function(
            calculate_ema, close_prices, period, target_ms=500
        )

        assert passed, f"EMA calculation took {exec_time:.2f}ms (target: <500ms)"
        assert len(result) == len(close_prices), "Result length mismatch"
        print(f"✅ EMA (10k candles): {exec_time:.2f}ms")

    def test_rsi_performance(self, large_dataset):
        """Test RSI calculation performance on 10k candles."""
        close_prices = large_dataset["close"]
        period = 14

        result, exec_time, passed = self.benchmark_function(
            calculate_rsi, close_prices, period, target_ms=500
        )

        assert passed, f"RSI calculation took {exec_time:.2f}ms (target: <500ms)"
        assert len(result) == len(close_prices), "Result length mismatch"
        print(f"✅ RSI (10k candles): {exec_time:.2f}ms")

    def test_atr_performance(self, large_dataset):
        """Test ATR calculation performance on 10k candles."""
        high_prices = large_dataset["high"]
        low_prices = large_dataset["low"]
        close_prices = large_dataset["close"]
        period = 14

        result, exec_time, passed = self.benchmark_function(
            calculate_atr, high_prices, low_prices, close_prices, period, target_ms=500
        )

        assert passed, f"ATR calculation took {exec_time:.2f}ms (target: <500ms)"
        assert len(result) == len(close_prices), "Result length mismatch"
        print(f"✅ ATR (10k candles): {exec_time:.2f}ms")

    def test_kama_performance(self, large_dataset):
        """Test KAMA calculation performance on 10k candles."""
        close_prices = large_dataset["close"]
        period = 10
        fast_period = 2
        slow_period = 30

        result, exec_time, passed = self.benchmark_function(
            calculate_kama,
            close_prices,
            period,
            fast_period,
            slow_period,
            target_ms=500,
        )

        assert passed, f"KAMA calculation took {exec_time:.2f}ms (target: <500ms)"
        assert len(result) == len(close_prices), "Result length mismatch"
        print(f"✅ KAMA (10k candles): {exec_time:.2f}ms")

    def test_adx_performance(self, large_dataset):
        """Test ADX calculation performance on 10k candles."""
        high_prices = large_dataset["high"]
        low_prices = large_dataset["low"]
        close_prices = large_dataset["close"]
        period = 14

        # Pre-calculate ATR as required by calculate_adx
        atr = calculate_atr(high_prices, low_prices, close_prices, period)

        result, exec_time, passed = self.benchmark_function(
            calculate_adx, high_prices, low_prices, close_prices, atr, period, target_ms=500
        )

        assert passed, f"ADX calculation took {exec_time:.2f}ms (target: <500ms)"
        assert len(result) == len(close_prices), "Result length mismatch"
        print(f"✅ ADX (10k candles): {exec_time:.2f}ms")

    def test_volatility_state_performance(self, large_dataset):
        """Test volatility state classification performance on 10k candles."""
        high_prices = large_dataset["high"]
        low_prices = large_dataset["low"]
        close_prices = large_dataset["close"]

        # Pre-calculate ATR and ATR_MA as required by classify_volatility_state
        atr = calculate_atr(high_prices, low_prices, close_prices, 14)
        from backend.computation.volatility import calculate_atr_ma
        atr_ma = calculate_atr_ma(atr, 50)

        result, exec_time, passed = self.benchmark_function(
            classify_volatility_state,
            atr,
            atr_ma,
            target_ms=500,
        )

        assert (
            passed
        ), f"Volatility state calculation took {exec_time:.2f}ms (target: <500ms)"
        assert len(result) == len(close_prices), "Result length mismatch"
        print(f"✅ Volatility State (10k candles): {exec_time:.2f}ms")

    def test_regime_detection_performance(self, large_dataset):
        """Test regime detection performance on 10k candles."""
        high_prices = large_dataset["high"]
        low_prices = large_dataset["low"]
        close_prices = large_dataset["close"]

        # Pre-calculate ATR and ATR_MA as required by detect_regime
        atr = calculate_atr(high_prices, low_prices, close_prices, 14)
        from backend.computation.volatility import calculate_atr_ma
        atr_ma = calculate_atr_ma(atr, 50)

        result, exec_time, passed = self.benchmark_function(
            detect_regime, high_prices, low_prices, close_prices, atr, atr_ma, target_ms=500
        )

        assert (
            passed
        ), f"Regime detection took {exec_time:.2f}ms (target: <500ms)"
        # detect_regime returns a dict, not an array
        assert isinstance(result, dict), "Result should be a dictionary"
        print(f"✅ Regime Detection (10k candles): {exec_time:.2f}ms")

    def test_composite_calculation_performance(self, large_dataset):
        """
        Test composite calculation performance (all indicators at once).

        This simulates real-world usage where multiple indicators are calculated
        together on the same dataset.
        """
        high_prices = large_dataset["high"]
        low_prices = large_dataset["low"]
        close_prices = large_dataset["close"]

        def composite_calculation():
            """Calculate all core indicators."""
            from backend.computation.volatility import calculate_atr_ma

            ema = calculate_ema(close_prices, 20)
            rsi = calculate_rsi(close_prices, 14)
            atr = calculate_atr(high_prices, low_prices, close_prices, 14)
            kama = calculate_kama(close_prices, 10, 2, 30)
            atr_ma = calculate_atr_ma(atr, 50)
            adx = calculate_adx(high_prices, low_prices, close_prices, atr, 14)
            vol_state = classify_volatility_state(atr, atr_ma)
            regime = detect_regime(high_prices, low_prices, close_prices, atr, atr_ma)

            return ema, rsi, atr, kama, adx, vol_state, regime

        # Target: all indicators combined should complete in <500ms
        result, exec_time, passed = self.benchmark_function(
            composite_calculation, target_ms=500
        )

        assert (
            passed
        ), f"Composite calculation took {exec_time:.2f}ms (target: <500ms)"
        print(f"✅ Composite (all indicators, 10k candles): {exec_time:.2f}ms")

    def test_warmup_jit_compilation(self, large_dataset):
        """
        Test JIT compilation warmup effectiveness.

        Since warmup happens in the fixture, both calls should be fast and cached.
        This test validates that the JIT cache is working properly.
        """
        close_prices = large_dataset["close"][:1000]

        # Both calls should be fast since warmup happened in fixture
        start = time.perf_counter()
        calculate_ema(close_prices, 20)
        first_call = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        calculate_ema(close_prices, 20)
        second_call = (time.perf_counter() - start) * 1000

        print(f"First call (JIT cached): {first_call:.2f}ms")
        print(f"Second call (JIT cached): {second_call:.2f}ms")

        # Both calls should be fast (<10ms for 1000 candles)
        assert first_call < 10.0, f"JIT cache not working (first call: {first_call:.2f}ms)"
        assert second_call < 10.0, f"JIT cache not working (second call: {second_call:.2f}ms)"

    def test_memory_efficiency(self, large_dataset):
        """
        Test that calculations don't create excessive memory overhead.

        Validates that results are returned as views or efficient arrays.
        """
        close_prices = large_dataset["close"]

        # Calculate indicator
        result = calculate_ema(close_prices, 20)

        # Check result is numpy array with expected properties
        assert isinstance(result, np.ndarray), "Result should be numpy array"
        assert result.dtype == np.float64, "Result should be float64"
        assert len(result) == len(close_prices), "Result length should match input"

        # Calculate memory footprint (in MB)
        input_memory = close_prices.nbytes / (1024 * 1024)
        output_memory = result.nbytes / (1024 * 1024)

        print(f"Input memory: {input_memory:.2f}MB")
        print(f"Output memory: {output_memory:.2f}MB")

        # Output should not be significantly larger than input
        assert output_memory <= input_memory * 1.1, "Excessive memory overhead"


class TestBenchmarkSummary:
    """Summary test that runs all benchmarks and reports aggregate statistics."""

    def test_benchmark_summary(self, capsys):
        """Run all benchmarks and display summary statistics."""
        print("\n" + "=" * 60)
        print("PERFORMANCE BENCHMARK SUMMARY (10,000 candles)")
        print("=" * 60)

        # This test intentionally runs through pytest to collect all benchmark results
        # The actual benchmark results are printed by individual tests
        # This test exists to provide a summary section in test output

        print("\nTarget: All indicators < 500ms")
        print("Dataset: 10,000 candles (realistic OHLCV data)")
        print("=" * 60)
