"""
Unit tests for adaptive_ema.py - KAMA (Kaufman's Adaptive Moving Average)

Test Coverage:
1. Efficiency Ratio (ER) calculation
2. Adaptive Smoothing Constant (ASC) calculation
3. KAMA calculation
4. Regime-aware KAMA
5. Edge cases (perfect trend, pure noise, transitions)
6. Mathematical validation (bounds checking)
7. Performance benchmarks

References:
- FLUXHERO_REQUIREMENTS.md: Feature 2 success criteria
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.computation.adaptive_ema import (  # noqa: E402
    calculate_efficiency_ratio,
    calculate_adaptive_smoothing_constant,
    calculate_kama,
    calculate_kama_with_regime_adjustment,
    validate_kama_bounds,
)


class TestEfficiencyRatio:
    """Test Efficiency Ratio (ER) calculation."""

    def test_perfect_trend_high_er(self):
        """Perfect uptrend should have ER close to 1.0."""
        # Create perfect uptrend: +1 per bar
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0,
                          106.0, 107.0, 108.0, 109.0, 110.0], dtype=np.float64)
        er = calculate_efficiency_ratio(prices, period=10)

        # Last ER should be close to 1.0 (perfect trend)
        assert not np.isnan(er[-1])
        assert er[-1] > 0.95, f"Expected ER > 0.95 for perfect trend, got {er[-1]}"

    def test_pure_noise_low_er(self):
        """Random oscillations should have ER close to 0.0."""
        # Create choppy market: alternating +1/-1
        prices = np.array([100.0, 101.0, 100.0, 101.0, 100.0, 101.0,
                          100.0, 101.0, 100.0, 101.0, 100.0], dtype=np.float64)
        er = calculate_efficiency_ratio(prices, period=10)

        # Last ER should be close to 0.0 (pure noise, no net movement)
        assert not np.isnan(er[-1])
        assert er[-1] < 0.05, f"Expected ER < 0.05 for noise, got {er[-1]}"

    def test_er_bounds_0_to_1(self):
        """ER must always be between 0 and 1."""
        # Use realistic price data
        np.random.seed(42)
        prices = 100.0 + np.cumsum(np.random.randn(100)) * 0.5
        prices = prices.astype(np.float64)
        er = calculate_efficiency_ratio(prices, period=10)

        # Check all non-NaN values are in [0, 1]
        valid_er = er[~np.isnan(er)]
        assert np.all(valid_er >= 0.0), "ER has values < 0"
        assert np.all(valid_er <= 1.0), "ER has values > 1"

    def test_er_first_values_nan(self):
        """First 'period' values should be NaN."""
        prices = np.arange(1.0, 21.0, dtype=np.float64)
        er = calculate_efficiency_ratio(prices, period=10)

        # First 10 should be NaN
        assert np.all(np.isnan(er[:10]))
        # 11th onwards should have values
        assert not np.isnan(er[10])

    def test_er_constant_prices(self):
        """Constant prices should give ER = 1.0 (no volatility, perfect 'trend')."""
        prices = np.full(20, 100.0, dtype=np.float64)
        er = calculate_efficiency_ratio(prices, period=10)

        # Should be 1.0 (division by zero case handled)
        valid_er = er[~np.isnan(er)]
        assert np.all(valid_er == 1.0), "Constant prices should give ER = 1.0"


class TestAdaptiveSmoothingConstant:
    """Test Adaptive Smoothing Constant (ASC) calculation."""

    def test_asc_at_er_extremes(self):
        """ASC at ER=0 and ER=1 should match theoretical bounds."""
        er = np.array([0.0, 1.0], dtype=np.float64)
        asc = calculate_adaptive_smoothing_constant(er, fast_period=2, slow_period=30)

        # Calculate expected values
        sc_fast = 2.0 / (2 + 1)  # 0.6667
        sc_slow = 2.0 / (30 + 1)  # 0.0645
        expected_asc_min = sc_slow ** 2
        expected_asc_max = sc_fast ** 2

        assert np.isclose(asc[0], expected_asc_min, rtol=1e-6), \
            f"ASC at ER=0 should be {expected_asc_min}, got {asc[0]}"
        assert np.isclose(asc[1], expected_asc_max, rtol=1e-6), \
            f"ASC at ER=1 should be {expected_asc_max}, got {asc[1]}"

    def test_asc_monotonic_increase(self):
        """ASC should increase as ER increases."""
        er = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
        asc = calculate_adaptive_smoothing_constant(er)

        # ASC should be monotonically increasing
        for i in range(len(asc) - 1):
            assert asc[i] <= asc[i + 1], \
                f"ASC should increase with ER, but asc[{i}]={asc[i]} > asc[{i+1}]={asc[i+1]}"

    def test_asc_bounds(self):
        """ASC must be between SC_slow² and SC_fast²."""
        # Random ER values
        np.random.seed(42)
        er = np.random.rand(100).astype(np.float64)
        asc = calculate_adaptive_smoothing_constant(er, fast_period=2, slow_period=30)

        sc_fast = 2.0 / (2 + 1)
        sc_slow = 2.0 / (30 + 1)
        asc_min = sc_slow ** 2
        asc_max = sc_fast ** 2

        assert np.all(asc >= asc_min - 1e-10), "ASC below minimum bound"
        assert np.all(asc <= asc_max + 1e-10), "ASC above maximum bound"

    def test_asc_with_nan_er(self):
        """ASC should handle NaN ER values."""
        er = np.array([np.nan, 0.5, np.nan, 0.8], dtype=np.float64)
        asc = calculate_adaptive_smoothing_constant(er)

        assert np.isnan(asc[0])
        assert not np.isnan(asc[1])
        assert np.isnan(asc[2])
        assert not np.isnan(asc[3])


class TestKAMA:
    """Test KAMA calculation."""

    def test_kama_trending_market(self):
        """KAMA should follow price closely in trending market."""
        # Strong uptrend
        prices = np.array([100.0, 102.0, 104.0, 106.0, 108.0, 110.0,
                          112.0, 114.0, 116.0, 118.0, 120.0], dtype=np.float64)
        kama = calculate_kama(prices, er_period=5)

        # KAMA should be within 2% of final price (requirement R2 success criteria)
        valid_kama = kama[~np.isnan(kama)]
        assert len(valid_kama) > 0
        final_kama = valid_kama[-1]
        final_price = prices[-1]
        pct_diff = abs(final_kama - final_price) / final_price * 100
        assert pct_diff < 2.0, \
            f"KAMA should be within 2% of price in trend, got {pct_diff:.2f}%"

    def test_kama_choppy_market(self):
        """KAMA should stay relatively flat in choppy market."""
        # Oscillating market around 100
        prices = np.array([100.0, 101.0, 99.0, 100.5, 99.5, 100.0,
                          101.0, 99.0, 100.0, 101.0, 100.0], dtype=np.float64)
        kama = calculate_kama(prices, er_period=5)

        valid_kama = kama[~np.isnan(kama)]
        # KAMA should stay near 100 with low volatility
        assert np.std(valid_kama) < 1.0, "KAMA should be smooth in choppy market"
        assert np.mean(valid_kama) > 99.0 and np.mean(valid_kama) < 101.0

    def test_kama_initialization(self):
        """First valid KAMA value should equal first valid price."""
        prices = np.arange(1.0, 21.0, dtype=np.float64)
        kama = calculate_kama(prices, er_period=10)

        # KAMA[10] should equal prices[10]
        assert kama[10] == prices[10], \
            f"First KAMA should equal first valid price, got {kama[10]} vs {prices[10]}"

    def test_kama_smooth_transitions(self):
        """KAMA should transition smoothly between regimes (no sudden jumps)."""
        # Create transition: trending -> choppy
        trend = np.arange(100.0, 110.0, 1.0)
        chop = np.array([110.0, 111.0, 109.0, 110.5, 109.5, 110.0])
        prices = np.concatenate([trend, chop]).astype(np.float64)

        kama = calculate_kama(prices, er_period=5)

        valid_kama = kama[~np.isnan(kama)]
        # Check no jumps > 5% of ATR
        kama_changes = np.abs(np.diff(valid_kama))
        max_change = np.max(kama_changes)
        assert max_change < 2.0, \
            f"KAMA should transition smoothly, max change was {max_change}"

    def test_kama_insufficient_data(self):
        """KAMA should return all NaN for insufficient data."""
        prices = np.array([100.0, 101.0, 102.0], dtype=np.float64)
        kama = calculate_kama(prices, er_period=10)

        assert np.all(np.isnan(kama)), "KAMA should be all NaN with insufficient data"


class TestRegimeAwareKAMA:
    """Test regime-aware KAMA calculation."""

    def test_trending_regime_detection(self):
        """Strong trend should be classified as TRENDING (regime=2)."""
        # Perfect uptrend
        prices = np.array([100.0, 102.0, 104.0, 106.0, 108.0, 110.0,
                          112.0, 114.0, 116.0, 118.0, 120.0], dtype=np.float64)
        kama, er, regime = calculate_kama_with_regime_adjustment(
            prices, er_period=5, trend_threshold=0.6, choppy_threshold=0.3
        )

        # Last regime should be TRENDING (2)
        valid_regime = regime[~np.isnan(regime)]
        assert len(valid_regime) > 0
        assert valid_regime[-1] == 2.0, \
            f"Strong trend should be regime=2, got {valid_regime[-1]}"

    def test_choppy_regime_detection(self):
        """Choppy market should be classified as CHOPPY (regime=0)."""
        # Oscillating market
        prices = np.array([100.0, 101.0, 99.0, 100.5, 99.5, 100.0,
                          101.0, 99.0, 100.0, 101.0, 100.0], dtype=np.float64)
        kama, er, regime = calculate_kama_with_regime_adjustment(
            prices, er_period=5, trend_threshold=0.6, choppy_threshold=0.3
        )

        # Last regime should be CHOPPY (0)
        valid_regime = regime[~np.isnan(regime)]
        assert len(valid_regime) > 0
        assert valid_regime[-1] == 0.0, \
            f"Choppy market should be regime=0, got {valid_regime[-1]}"

    def test_neutral_regime_detection(self):
        """Moderate trend should be classified as NEUTRAL (regime=1)."""
        # Gentle uptrend with some noise
        np.random.seed(42)
        prices = 100.0 + np.cumsum(np.random.randn(20) * 0.3 + 0.2)
        prices = prices.astype(np.float64)

        kama, er, regime = calculate_kama_with_regime_adjustment(
            prices, er_period=5, trend_threshold=0.6, choppy_threshold=0.3
        )

        valid_regime = regime[~np.isnan(regime)]
        # Should have some neutral readings (regime=1)
        assert np.any(valid_regime == 1.0), "Should detect NEUTRAL regime"

    def test_regime_consistency_with_er(self):
        """Regime classification should match ER thresholds."""
        prices = np.array([100.0, 102.0, 104.0, 106.0, 108.0, 110.0,
                          112.0, 114.0, 116.0, 118.0, 120.0], dtype=np.float64)
        kama, er, regime = calculate_kama_with_regime_adjustment(
            prices, er_period=5, trend_threshold=0.6, choppy_threshold=0.3
        )

        # Check consistency
        for i in range(len(er)):
            if not np.isnan(er[i]):
                if er[i] > 0.6:
                    assert regime[i] == 2.0, f"ER={er[i]} should give regime=2"
                elif er[i] < 0.3:
                    assert regime[i] == 0.0, f"ER={er[i]} should give regime=0"
                else:
                    assert regime[i] == 1.0, f"ER={er[i]} should give regime=1"


class TestKAMAValidation:
    """Test mathematical validation of KAMA bounds."""

    def test_validate_er_bounds(self):
        """Validate ER stays in [0, 1]."""
        np.random.seed(42)
        prices = 100.0 + np.cumsum(np.random.randn(100)) * 0.5
        prices = prices.astype(np.float64)

        er = calculate_efficiency_ratio(prices, period=10)
        asc = calculate_adaptive_smoothing_constant(er)

        er_valid, asc_valid = validate_kama_bounds(er, asc)
        assert er_valid, "ER validation failed - values outside [0, 1]"

    def test_validate_asc_bounds(self):
        """Validate ASC stays in [SC_slow², SC_fast²]."""
        np.random.seed(42)
        prices = 100.0 + np.cumsum(np.random.randn(100)) * 0.5
        prices = prices.astype(np.float64)

        er = calculate_efficiency_ratio(prices, period=10)
        asc = calculate_adaptive_smoothing_constant(er, fast_period=2, slow_period=30)

        er_valid, asc_valid = validate_kama_bounds(er, asc, fast_period=2, slow_period=30)
        assert asc_valid, "ASC validation failed - values outside bounds"

    def test_validate_both_bounds(self):
        """Validate both ER and ASC bounds together."""
        # Perfect trend
        prices = np.arange(100.0, 120.0, 1.0, dtype=np.float64)
        er = calculate_efficiency_ratio(prices, period=10)
        asc = calculate_adaptive_smoothing_constant(er)

        er_valid, asc_valid = validate_kama_bounds(er, asc)
        assert er_valid and asc_valid, "Validation failed for perfect trend"

    def test_validate_edge_cases(self):
        """Validate bounds for edge cases."""
        # Constant prices
        prices = np.full(20, 100.0, dtype=np.float64)
        er = calculate_efficiency_ratio(prices, period=10)
        asc = calculate_adaptive_smoothing_constant(er)

        er_valid, asc_valid = validate_kama_bounds(er, asc)
        assert er_valid and asc_valid, "Validation failed for constant prices"


class TestKAMAEdgeCases:
    """Test edge cases for KAMA calculations."""

    def test_empty_array(self):
        """Empty array should return empty array."""
        prices = np.array([], dtype=np.float64)
        er = calculate_efficiency_ratio(prices)
        assert len(er) == 0

    def test_single_price(self):
        """Single price should return NaN."""
        prices = np.array([100.0], dtype=np.float64)
        kama = calculate_kama(prices, er_period=10)
        assert np.all(np.isnan(kama))

    def test_negative_prices(self):
        """KAMA should handle negative prices (for some instruments)."""
        prices = np.array([-10.0, -9.0, -8.0, -7.0, -6.0, -5.0,
                          -4.0, -3.0, -2.0, -1.0, 0.0], dtype=np.float64)
        kama = calculate_kama(prices, er_period=5)

        valid_kama = kama[~np.isnan(kama)]
        assert len(valid_kama) > 0, "KAMA should work with negative prices"

    def test_very_large_prices(self):
        """KAMA should handle large prices without overflow."""
        prices = np.array([1e6, 1.01e6, 1.02e6, 1.03e6, 1.04e6, 1.05e6,
                          1.06e6, 1.07e6, 1.08e6, 1.09e6, 1.1e6], dtype=np.float64)
        kama = calculate_kama(prices, er_period=5)

        valid_kama = kama[~np.isnan(kama)]
        assert len(valid_kama) > 0
        assert not np.any(np.isinf(valid_kama)), "KAMA overflow detected"

    def test_prices_with_gaps(self):
        """KAMA should handle price gaps."""
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 110.0,  # Gap up
                          111.0, 112.0, 113.0, 114.0, 115.0], dtype=np.float64)
        kama = calculate_kama(prices, er_period=5)

        valid_kama = kama[~np.isnan(kama)]
        assert len(valid_kama) > 0
        # KAMA should adapt to gap
        assert valid_kama[-1] > 100.0


class TestKAMAPerformance:
    """Test KAMA performance benchmarks."""

    def test_kama_performance_10k_candles(self):
        """KAMA on 10k candles should complete in <100ms (from requirements)."""
        import time

        np.random.seed(42)
        prices = 100.0 + np.cumsum(np.random.randn(10000)) * 0.5
        prices = prices.astype(np.float64)

        # Warmup JIT compilation
        _ = calculate_kama(prices[:100], er_period=10)

        # Benchmark
        start_time = time.time()
        _ = calculate_kama(prices, er_period=10)
        elapsed_ms = (time.time() - start_time) * 1000

        assert elapsed_ms < 100, \
            f"KAMA on 10k candles took {elapsed_ms:.2f}ms, expected <100ms"

    def test_full_kama_suite_performance(self):
        """Full KAMA calculation (ER + ASC + KAMA + regime) on 10k candles <200ms."""
        import time

        np.random.seed(42)
        prices = 100.0 + np.cumsum(np.random.randn(10000)) * 0.5
        prices = prices.astype(np.float64)

        # Warmup
        _ = calculate_kama_with_regime_adjustment(prices[:100], er_period=10)

        # Benchmark
        start_time = time.time()
        kama, er, regime = calculate_kama_with_regime_adjustment(prices, er_period=10)
        elapsed_ms = (time.time() - start_time) * 1000

        assert elapsed_ms < 200, \
            f"Full KAMA suite took {elapsed_ms:.2f}ms, expected <200ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
