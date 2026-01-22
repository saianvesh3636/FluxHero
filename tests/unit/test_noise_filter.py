"""
Unit tests for Market Microstructure Noise Filter module.

Tests cover:
    - Spread-to-volatility ratio calculation and validation
    - Volume validation for normal signals and breakouts
    - Time-of-day filtering for illiquid hours
    - Combined noise filter application
    - Edge cases and performance benchmarks

Requirements tested:
    - R4.1.1-4.1.4: Spread-to-volatility ratio
    - R4.2.1-4.2.3: Volume validation
    - R4.3.1-4.3.3: Time-of-day filtering
"""

import numpy as np
import pytest
from datetime import datetime
import sys
import os
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.strategy.noise_filter import (
    calculate_spread_to_volatility_ratio,
    validate_spread_ratio,
    calculate_average_volume,
    validate_volume,
    is_illiquid_hour,
    is_near_close,
    apply_noise_filter,
    calculate_rejection_reason,
    calculate_price_gap_ratio,
    validate_price_gap,
)


# ============================================================================
# Spread-to-Volatility Ratio Tests
# ============================================================================

class TestSpreadToVolatilityRatio:
    """Test spread-to-volatility ratio calculation (R4.1.1-4.1.3)."""

    def test_basic_sv_ratio_calculation(self):
        """Test basic SV ratio calculation."""
        # Create test data with known spread and volatility
        bid = np.array([100.0, 100.5, 101.0, 101.5, 102.0])
        ask = np.array([100.1, 100.6, 101.1, 101.6, 102.1])  # Spread = 0.1
        close = np.array([100.05, 100.55, 101.05, 101.55, 102.05])

        sv_ratio = calculate_spread_to_volatility_ratio(bid, ask, close, volatility_period=3)

        # First 2 bars should be NaN
        assert np.isnan(sv_ratio[0])
        assert np.isnan(sv_ratio[1])

        # Later bars should have valid ratios
        assert not np.isnan(sv_ratio[2])
        assert sv_ratio[2] > 0

    def test_sv_ratio_with_high_spread(self):
        """Test SV ratio correctly identifies high spreads (R4.1.4)."""
        # Wide spread scenario
        bid = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        ask = np.array([101.0, 101.0, 101.0, 101.0, 101.0])  # Spread = 1.0
        close = np.array([100.5, 100.5, 100.5, 100.5, 100.5])  # Low volatility

        sv_ratio = calculate_spread_to_volatility_ratio(bid, ask, close, volatility_period=3)

        # High spread with low volatility should produce high SV ratio
        assert sv_ratio[-1] > 0.05  # Should exceed normal threshold

    def test_sv_ratio_with_low_spread(self):
        """Test SV ratio with tight spreads."""
        # Tight spread scenario
        bid = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        ask = np.array([100.01, 101.01, 102.01, 103.01, 104.01])  # Spread = 0.01
        close = np.array([100.005, 101.005, 102.005, 103.005, 104.005])

        sv_ratio = calculate_spread_to_volatility_ratio(bid, ask, close, volatility_period=3)

        # Low spread with normal volatility should produce low SV ratio
        assert sv_ratio[-1] < 0.05  # Should be below threshold

    def test_sv_ratio_zero_volatility(self):
        """Test SV ratio handles zero volatility (constant prices)."""
        # Constant prices (zero volatility)
        bid = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        ask = np.array([100.1, 100.1, 100.1, 100.1, 100.1])
        close = np.array([100.05, 100.05, 100.05, 100.05, 100.05])

        sv_ratio = calculate_spread_to_volatility_ratio(bid, ask, close, volatility_period=3)

        # Zero volatility should result in very high ratio (rejection)
        assert sv_ratio[-1] > 100  # Should be flagged as suspicious

    def test_validate_spread_ratio_normal_hours(self):
        """Test spread ratio validation for normal hours."""
        sv_ratio = np.array([0.01, 0.03, 0.05, 0.06, 0.10])

        valid = validate_spread_ratio(sv_ratio, threshold=0.05)

        assert valid[0]  # 0.01 < 0.05
        assert valid[1]  # 0.03 < 0.05
        assert valid[2]  # 0.05 == 0.05
        assert not valid[3]  # 0.06 > 0.05
        assert not valid[4]  # 0.10 > 0.05

    def test_insufficient_data(self):
        """Test SV ratio with insufficient data."""
        bid = np.array([100.0, 100.5])
        ask = np.array([100.1, 100.6])
        close = np.array([100.05, 100.55])

        sv_ratio = calculate_spread_to_volatility_ratio(bid, ask, close, volatility_period=10)

        # All values should be NaN when data < period
        assert all(np.isnan(sv_ratio))


# ============================================================================
# Volume Validation Tests
# ============================================================================

class TestVolumeValidation:
    """Test volume validation logic (R4.2.1-4.2.3)."""

    def test_calculate_average_volume(self):
        """Test average volume calculation (R4.2.1)."""
        volume = np.array([1000, 1500, 2000, 1200, 1800])

        avg_vol = calculate_average_volume(volume, period=3)

        # First 2 bars should be NaN
        assert np.isnan(avg_vol[0])
        assert np.isnan(avg_vol[1])

        # Bar 2: mean([1000, 1500, 2000]) = 1500
        assert abs(avg_vol[2] - 1500.0) < 1e-6

        # Bar 3: mean([1500, 2000, 1200]) = 1566.67
        assert abs(avg_vol[3] - 1566.67) < 0.1

        # Bar 4: mean([2000, 1200, 1800]) = 1666.67
        assert abs(avg_vol[4] - 1666.67) < 0.1

    def test_validate_normal_signal_volume(self):
        """Test volume validation for normal signals (R4.2.2)."""
        volume = np.array([1000, 600, 400, 1500])
        avg_volume = np.array([1000, 1000, 1000, 1000])
        is_breakout = np.array([False, False, False, False])

        valid = validate_volume(volume, avg_volume, is_breakout, normal_threshold=0.5)

        # Normal signals require volume > 0.5 × avg
        assert valid[0]  # 1000 > 500
        assert valid[1]  # 600 > 500
        assert not valid[2]  # 400 < 500
        assert valid[3]  # 1500 > 500

    def test_validate_breakout_signal_volume(self):
        """Test volume validation for breakout signals (R4.2.3)."""
        volume = np.array([2000, 1600, 1400, 1000])
        avg_volume = np.array([1000, 1000, 1000, 1000])
        is_breakout = np.array([True, True, True, True])

        valid = validate_volume(volume, avg_volume, is_breakout, breakout_threshold=1.5)

        # Breakout signals require volume > 1.5 × avg
        assert valid[0]  # 2000 > 1500
        assert valid[1]  # 1600 > 1500
        assert not valid[2]  # 1400 < 1500
        assert not valid[3]  # 1000 < 1500

    def test_validate_mixed_signal_types(self):
        """Test volume validation with mixed normal and breakout signals."""
        volume = np.array([600, 1600, 800, 1200])
        avg_volume = np.array([1000, 1000, 1000, 1000])
        is_breakout = np.array([False, True, False, True])

        valid = validate_volume(volume, avg_volume, is_breakout)

        # [False: 600>500 ✓, True: 1600>1500 ✓, False: 800>500 ✓, True: 1200<1500 ✗]
        assert valid[0]  # Normal signal, 600 > 500
        assert valid[1]  # Breakout signal, 1600 > 1500
        assert valid[2]  # Normal signal, 800 > 500
        assert not valid[3]  # Breakout signal, 1200 < 1500

    def test_validate_volume_with_nan(self):
        """Test volume validation handles NaN average volumes."""
        volume = np.array([1000, 1500, 2000])
        avg_volume = np.array([np.nan, 1000, 1000])
        is_breakout = np.array([False, False, False])

        valid = validate_volume(volume, avg_volume, is_breakout)

        # First bar with NaN average should be invalid
        assert not valid[0]
        assert valid[1]
        assert valid[2]


# ============================================================================
# Time-of-Day Filter Tests
# ============================================================================

class TestTimeOfDayFilter:
    """Test time-of-day filtering logic (R4.3.1-4.3.3)."""

    def test_premarket_hours(self):
        """Test pre-market hour detection (R4.3.1)."""
        # 9:15 AM - pre-market
        dt = datetime(2024, 1, 15, 9, 15)
        assert is_illiquid_hour(dt)

        # 9:29 AM - pre-market
        dt = datetime(2024, 1, 15, 9, 29)
        assert is_illiquid_hour(dt)

        # 9:30 AM - market open
        dt = datetime(2024, 1, 15, 9, 30)
        assert not is_illiquid_hour(dt)

    def test_lunch_hours(self):
        """Test lunch hour detection (R4.3.1)."""
        # 11:59 AM - before lunch
        dt = datetime(2024, 1, 15, 11, 59)
        assert not is_illiquid_hour(dt)

        # 12:00 PM - lunch start
        dt = datetime(2024, 1, 15, 12, 0)
        assert is_illiquid_hour(dt)

        # 12:30 PM - lunch
        dt = datetime(2024, 1, 15, 12, 30)
        assert is_illiquid_hour(dt)

        # 1:00 PM - lunch end
        dt = datetime(2024, 1, 15, 13, 0)
        assert not is_illiquid_hour(dt)

    def test_afterhours(self):
        """Test after-hours detection (R4.3.1)."""
        # 3:59 PM - normal hours
        dt = datetime(2024, 1, 15, 15, 59)
        assert not is_illiquid_hour(dt)

        # 4:00 PM - after-hours
        dt = datetime(2024, 1, 15, 16, 0)
        assert is_illiquid_hour(dt)

        # 5:00 PM - after-hours
        dt = datetime(2024, 1, 15, 17, 0)
        assert is_illiquid_hour(dt)

    def test_normal_trading_hours(self):
        """Test normal trading hour detection."""
        # 10:30 AM - normal
        dt = datetime(2024, 1, 15, 10, 30)
        assert not is_illiquid_hour(dt)

        # 2:00 PM - normal
        dt = datetime(2024, 1, 15, 14, 0)
        assert not is_illiquid_hour(dt)

    def test_near_close_detection(self):
        """Test near-close detection (R4.3.3)."""
        # 3:44 PM - 16 minutes before close
        dt = datetime(2024, 1, 15, 15, 44)
        assert not is_near_close(dt, minutes_before_close=15)

        # 3:45 PM - exactly 15 minutes before close
        dt = datetime(2024, 1, 15, 15, 45)
        assert is_near_close(dt, minutes_before_close=15)

        # 3:50 PM - 10 minutes before close
        dt = datetime(2024, 1, 15, 15, 50)
        assert is_near_close(dt, minutes_before_close=15)

        # 3:59 PM - 1 minute before close
        dt = datetime(2024, 1, 15, 15, 59)
        assert is_near_close(dt, minutes_before_close=15)

        # 4:00 PM - market close
        dt = datetime(2024, 1, 15, 16, 0)
        assert not is_near_close(dt, minutes_before_close=15)


# ============================================================================
# Combined Noise Filter Tests
# ============================================================================

class TestCombinedNoiseFilter:
    """Test combined noise filter application."""

    def test_all_filters_pass(self):
        """Test signal passes all filters."""
        sv_ratio = np.array([0.01, 0.02, 0.03, 0.01, 0.02])
        volume = np.array([1000, 1500, 2000, 1200, 1800])
        avg_volume = np.array([1000, 1000, 1000, 1000, 1000])
        is_breakout = np.array([False, False, False, False, False])
        is_illiquid = np.array([False, False, False, False, False])

        valid = apply_noise_filter(
            sv_ratio, volume, avg_volume, is_breakout, is_illiquid
        )

        # All signals should pass (good SV ratio, good volume, normal hours)
        assert valid[0]  # 0.01 < 0.05, 1000 > 500
        assert valid[1]  # 0.02 < 0.05, 1500 > 500
        assert valid[2]  # 0.03 < 0.05, 2000 > 500

    def test_sv_ratio_rejection_normal_hours(self):
        """Test signal rejection due to high SV ratio in normal hours."""
        sv_ratio = np.array([0.01, 0.06, 0.03])
        volume = np.array([1000, 1500, 2000])
        avg_volume = np.array([1000, 1000, 1000])
        is_breakout = np.array([False, False, False])
        is_illiquid = np.array([False, False, False])

        valid = apply_noise_filter(
            sv_ratio, volume, avg_volume, is_breakout, is_illiquid
        )

        assert valid[0]  # Pass
        assert not valid[1]  # Reject: SV ratio 0.06 > 0.05
        assert valid[2]  # Pass

    def test_sv_ratio_rejection_illiquid_hours(self):
        """Test stricter SV ratio threshold during illiquid hours (R4.3.2)."""
        sv_ratio = np.array([0.01, 0.03, 0.01])
        volume = np.array([1000, 1500, 2000])
        avg_volume = np.array([1000, 1000, 1000])
        is_breakout = np.array([False, False, False])
        is_illiquid = np.array([False, True, False])

        valid = apply_noise_filter(
            sv_ratio, volume, avg_volume, is_breakout, is_illiquid
        )

        assert valid[0]  # Normal hours: 0.01 < 0.05
        assert not valid[1]  # Illiquid hours: 0.03 > 0.025 (stricter)
        assert valid[2]  # Normal hours: 0.01 < 0.05

    def test_volume_rejection_normal_signal(self):
        """Test signal rejection due to low volume (normal signal)."""
        sv_ratio = np.array([0.01, 0.01, 0.01])
        volume = np.array([1000, 400, 600])
        avg_volume = np.array([1000, 1000, 1000])
        is_breakout = np.array([False, False, False])
        is_illiquid = np.array([False, False, False])

        valid = apply_noise_filter(
            sv_ratio, volume, avg_volume, is_breakout, is_illiquid
        )

        assert valid[0]  # 1000 > 500
        assert not valid[1]  # 400 < 500 (reject)
        assert valid[2]  # 600 > 500

    def test_volume_rejection_breakout_signal(self):
        """Test signal rejection due to low volume (breakout signal)."""
        sv_ratio = np.array([0.01, 0.01, 0.01])
        volume = np.array([2000, 1400, 1600])
        avg_volume = np.array([1000, 1000, 1000])
        is_breakout = np.array([True, True, True])
        is_illiquid = np.array([False, False, False])

        valid = apply_noise_filter(
            sv_ratio, volume, avg_volume, is_breakout, is_illiquid
        )

        assert valid[0]  # 2000 > 1500
        assert not valid[1]  # 1400 < 1500 (reject)
        assert valid[2]  # 1600 > 1500

    def test_multiple_rejection_criteria(self):
        """Test signals rejected for different reasons."""
        sv_ratio = np.array([0.06, 0.01, 0.01, 0.01])
        volume = np.array([1000, 400, 1400, 1000])
        avg_volume = np.array([1000, 1000, 1000, 1000])
        is_breakout = np.array([False, False, True, False])
        is_illiquid = np.array([False, False, False, False])

        valid = apply_noise_filter(
            sv_ratio, volume, avg_volume, is_breakout, is_illiquid
        )

        assert not valid[0]  # Reject: high SV ratio
        assert not valid[1]  # Reject: low volume (normal)
        assert not valid[2]  # Reject: low volume (breakout, 1400 < 1500)
        assert valid[3]  # Pass all filters

    def test_nan_data_rejection(self):
        """Test signals with NaN data are rejected."""
        sv_ratio = np.array([np.nan, 0.01, 0.01])
        volume = np.array([1000, 1000, 1000])
        avg_volume = np.array([1000, np.nan, 1000])
        is_breakout = np.array([False, False, False])
        is_illiquid = np.array([False, False, False])

        valid = apply_noise_filter(
            sv_ratio, volume, avg_volume, is_breakout, is_illiquid
        )

        assert not valid[0]  # NaN SV ratio
        assert not valid[1]  # NaN avg volume
        assert valid[2]  # Valid data


# ============================================================================
# Rejection Reason Tests
# ============================================================================

class TestRejectionReason:
    """Test rejection reason calculation for debugging."""

    def test_rejection_reason_codes(self):
        """Test rejection reason codes are correctly assigned."""
        sv_ratio = np.array([0.01, 0.06, 0.01, 0.01, np.nan])
        volume = np.array([1000, 1000, 400, 1400, 1000])
        avg_volume = np.array([1000, 1000, 1000, 1000, 1000])
        is_breakout = np.array([False, False, False, True, False])
        is_illiquid = np.array([False, False, False, False, False])

        reasons = calculate_rejection_reason(
            sv_ratio, volume, avg_volume, is_breakout, is_illiquid
        )

        assert reasons[0] == 0  # Valid signal
        assert reasons[1] == 1  # High SV ratio (normal hours)
        assert reasons[2] == 3  # Low volume (normal signal)
        assert reasons[3] == 4  # Low volume (breakout signal)
        assert reasons[4] == 5  # Insufficient data (NaN)

    def test_rejection_reason_illiquid_hours(self):
        """Test rejection reason for illiquid hours."""
        sv_ratio = np.array([0.03, 0.01])
        volume = np.array([1000, 1000])
        avg_volume = np.array([1000, 1000])
        is_breakout = np.array([False, False])
        is_illiquid = np.array([True, False])

        reasons = calculate_rejection_reason(
            sv_ratio, volume, avg_volume, is_breakout, is_illiquid
        )

        assert reasons[0] == 2  # High SV ratio (illiquid hours, 0.03 > 0.025)
        assert reasons[1] == 0  # Valid (normal hours, 0.01 < 0.05)


# ============================================================================
# Success Criteria Tests (from FLUXHERO_REQUIREMENTS.md)
# ============================================================================

class TestSuccessCriteria:
    """Test success criteria from requirements."""

    def test_wide_spread_low_vol_rejection(self):
        """
        Success criterion: Wide spread (0.10) + low vol (1.0) → Signal rejected
        SV_Ratio = 0.10 / 1.0 = 0.10 = 10% > 5% threshold
        """
        # Simulate low volatility (std = 1.0)
        close = np.array([100.0, 100.5, 101.0, 100.5, 100.0])  # std ≈ 0.4

        # Wide spread
        bid = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        ask = np.array([100.4, 100.4, 100.4, 100.4, 100.4])  # Spread = 0.4

        sv_ratio = calculate_spread_to_volatility_ratio(bid, ask, close, volatility_period=3)

        # SV ratio should be high (0.4 / ~0.4 ≈ 1.0 = 100%)
        assert sv_ratio[-1] > 0.05  # Should exceed threshold

    def test_tight_spread_normal_vol_acceptance(self):
        """
        Success criterion: Tight spread (0.02) + normal vol (2.0) → Signal allowed
        SV_Ratio = 0.02 / 2.0 = 0.01 = 1% < 5% threshold
        """
        # Simulate normal volatility (trending prices)
        close = np.array([100.0, 102.0, 104.0, 106.0, 108.0])

        # Tight spread
        bid = np.array([100.0, 102.0, 104.0, 106.0, 108.0])
        ask = np.array([100.02, 102.02, 104.02, 106.02, 108.02])  # Spread = 0.02

        sv_ratio = calculate_spread_to_volatility_ratio(bid, ask, close, volatility_period=3)

        # SV ratio should be low
        assert sv_ratio[-1] < 0.05  # Should pass threshold

    def test_low_volume_breakout_rejection(self):
        """
        Success criterion: Low volume breakout → Signal rejected
        Breakout requires volume > 1.5 × average
        """
        volume = np.array([1000, 1000, 1200])  # 1200 < 1.5 × 1000
        avg_volume = np.array([1000, 1000, 1000])
        is_breakout = np.array([False, False, True])

        valid = validate_volume(volume, avg_volume, is_breakout)

        assert not valid[2]  # Breakout rejected due to low volume


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_arrays(self):
        """Test functions handle empty arrays gracefully."""
        empty = np.array([])

        sv_ratio = calculate_spread_to_volatility_ratio(empty, empty, empty)
        assert len(sv_ratio) == 0

        avg_vol = calculate_average_volume(empty)
        assert len(avg_vol) == 0

    def test_single_value_arrays(self):
        """Test functions handle single-value arrays."""
        single = np.array([100.0])

        sv_ratio = calculate_spread_to_volatility_ratio(single, single, single, volatility_period=10)
        assert len(sv_ratio) == 1
        assert np.isnan(sv_ratio[0])

    def test_negative_volumes(self):
        """Test volume validation with negative volumes (should be invalid)."""
        volume = np.array([-100, 1000, 2000])
        avg_volume = np.array([1000, 1000, 1000])
        is_breakout = np.array([False, False, False])

        valid = validate_volume(volume, avg_volume, is_breakout)

        assert not valid[0]  # Negative volume should fail

    def test_very_large_spreads(self):
        """Test SV ratio with very large spreads."""
        bid = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        ask = np.array([150.0, 150.0, 150.0, 150.0, 150.0])  # Spread = 50.0
        close = np.array([125.0, 125.0, 125.0, 125.0, 125.0])

        sv_ratio = calculate_spread_to_volatility_ratio(bid, ask, close, volatility_period=3)

        # Very large spread should produce very high ratio
        assert sv_ratio[-1] > 100  # Should be flagged


# ============================================================================
# Performance Benchmark Tests
# ============================================================================

class TestPerformance:
    """Test performance benchmarks for noise filter."""

    def test_noise_filter_performance_10k_candles(self):
        """Test full noise filter on 10k candles completes in <50ms."""
        import time

        n = 10000
        sv_ratio = np.random.uniform(0.01, 0.10, n)
        volume = np.random.uniform(500, 2000, n)
        avg_volume = np.random.uniform(800, 1200, n)
        is_breakout = np.random.choice([True, False], n)
        is_illiquid = np.random.choice([True, False], n)

        # Warm up JIT compilation
        _ = apply_noise_filter(
            sv_ratio[:100], volume[:100], avg_volume[:100],
            is_breakout[:100], is_illiquid[:100]
        )

        # Benchmark
        start = time.perf_counter()
        _ = apply_noise_filter(
            sv_ratio, volume, avg_volume, is_breakout, is_illiquid
        )
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms

        print(f"\nNoise filter (10k candles): {elapsed:.2f}ms")
        assert elapsed < 50, f"Performance too slow: {elapsed:.2f}ms > 50ms"

    def test_sv_ratio_calculation_performance(self):
        """Test SV ratio calculation performance."""
        import time

        n = 10000
        bid = np.random.uniform(100, 200, n)
        ask = bid + np.random.uniform(0.01, 0.1, n)
        close = (bid + ask) / 2

        # Warm up
        _ = calculate_spread_to_volatility_ratio(bid[:100], ask[:100], close[:100])

        # Benchmark
        start = time.perf_counter()
        _ = calculate_spread_to_volatility_ratio(bid, ask, close)
        elapsed = (time.perf_counter() - start) * 1000

        print(f"\nSV ratio calculation (10k candles): {elapsed:.2f}ms")
        assert elapsed < 100, f"Performance too slow: {elapsed:.2f}ms"


# ============================================================================
# Price Gap Filter Tests
# ============================================================================

class TestPriceGapFilter:
    """Test price gap filter functions (Phase 16 - Retail optimization)."""

    def test_calculate_price_gap_ratio_basic(self):
        """Test basic price gap ratio calculation."""
        open_prices = np.array([100.0, 102.0, 101.0, 103.0])
        close_prices = np.array([101.0, 103.0, 102.0, 104.0])

        gap_ratio = calculate_price_gap_ratio(open_prices, close_prices)

        # First bar should be NaN (no previous close)
        assert np.isnan(gap_ratio[0])

        # Bar 1: |102.0 - 101.0| / 101.0 = 0.0099 (0.99%)
        assert np.isclose(gap_ratio[1], 0.0099, atol=0.0001)

        # Bar 2: |101.0 - 103.0| / 103.0 = 0.0194 (1.94%)
        assert np.isclose(gap_ratio[2], 0.0194, atol=0.0001)

        # Bar 3: |103.0 - 102.0| / 102.0 = 0.0098 (0.98%)
        assert np.isclose(gap_ratio[3], 0.0098, atol=0.0001)

    def test_calculate_price_gap_ratio_large_gap(self):
        """Test gap calculation with large overnight gap."""
        open_prices = np.array([100.0, 105.0])  # 5% gap up
        close_prices = np.array([101.0, 106.0])

        gap_ratio = calculate_price_gap_ratio(open_prices, close_prices)

        # Bar 1: |105.0 - 101.0| / 101.0 = 0.0396 (3.96%)
        assert gap_ratio[1] > 0.02  # Exceeds 2% threshold
        assert np.isclose(gap_ratio[1], 0.0396, atol=0.0001)

    def test_calculate_price_gap_ratio_gap_down(self):
        """Test gap calculation with gap down."""
        open_prices = np.array([100.0, 97.0])  # 3% gap down
        close_prices = np.array([101.0, 98.0])

        gap_ratio = calculate_price_gap_ratio(open_prices, close_prices)

        # Bar 1: |97.0 - 101.0| / 101.0 = 0.0396 (3.96%)
        assert gap_ratio[1] > 0.02  # Exceeds 2% threshold
        assert np.isclose(gap_ratio[1], 0.0396, atol=0.0001)

    def test_calculate_price_gap_ratio_zero_division(self):
        """Test gap calculation with zero previous close (edge case)."""
        open_prices = np.array([100.0, 101.0])
        close_prices = np.array([0.0, 102.0])  # Zero previous close

        gap_ratio = calculate_price_gap_ratio(open_prices, close_prices)

        # Should handle division by zero gracefully
        assert gap_ratio[1] == 999.0  # High ratio = reject

    def test_calculate_price_gap_ratio_single_bar(self):
        """Test gap calculation with insufficient data."""
        open_prices = np.array([100.0])
        close_prices = np.array([101.0])

        gap_ratio = calculate_price_gap_ratio(open_prices, close_prices)

        # Single bar: all NaN
        assert np.isnan(gap_ratio[0])

    def test_validate_price_gap_within_threshold(self):
        """Test gap validation with gaps within threshold."""
        gap_ratio = np.array([np.nan, 0.01, 0.015, 0.019])

        valid = validate_price_gap(gap_ratio, threshold=0.02)

        # First bar: NaN → rejected
        assert not valid[0]

        # Bars 1-3: all within 2% threshold → accepted
        assert valid[1]  # 1%
        assert valid[2]  # 1.5%
        assert valid[3]  # 1.9%

    def test_validate_price_gap_exceeds_threshold(self):
        """Test gap validation with gaps exceeding threshold."""
        gap_ratio = np.array([np.nan, 0.01, 0.025, 0.03])

        valid = validate_price_gap(gap_ratio, threshold=0.02)

        # First bar: NaN → rejected
        assert not valid[0]

        # Bar 1: 1% → accepted
        assert valid[1]

        # Bars 2-3: exceed 2% threshold → rejected
        assert not valid[2]  # 2.5%
        assert not valid[3]  # 3%

    def test_validate_price_gap_custom_threshold(self):
        """Test gap validation with custom threshold."""
        gap_ratio = np.array([np.nan, 0.005, 0.008, 0.012])

        valid = validate_price_gap(gap_ratio, threshold=0.01)

        # First bar: NaN → rejected
        assert not valid[0]

        # Bar 1: 0.5% → accepted
        assert valid[1]

        # Bar 2: 0.8% → accepted
        assert valid[2]

        # Bar 3: 1.2% → rejected (exceeds 1% threshold)
        assert not valid[3]

    def test_price_gap_filter_earnings_scenario(self):
        """Test gap filter for earnings announcement scenario."""
        # Simulate normal trading, then earnings gap
        open_prices = np.array([100.0, 100.5, 101.0, 107.0])  # 5.9% gap on bar 3
        close_prices = np.array([100.2, 100.8, 101.2, 107.5])

        gap_ratio = calculate_price_gap_ratio(open_prices, close_prices)
        valid = validate_price_gap(gap_ratio, threshold=0.02)

        # Normal gaps accepted
        assert valid[1]  # Small gap
        assert valid[2]  # Small gap

        # Earnings gap rejected
        assert not valid[3]  # Large gap (5.9%)

    def test_price_gap_filter_empty_array(self):
        """Test gap filter with empty arrays."""
        open_prices = np.array([])
        close_prices = np.array([])

        gap_ratio = calculate_price_gap_ratio(open_prices, close_prices)
        valid = validate_price_gap(gap_ratio)

        # Should return empty arrays
        assert len(gap_ratio) == 0
        assert len(valid) == 0

    def test_price_gap_filter_performance(self):
        """Test gap filter performance on large dataset."""
        n = 10_000
        np.random.seed(42)

        # Generate realistic price data with occasional gaps
        base_price = 100.0
        close_prices = np.zeros(n)
        open_prices = np.zeros(n)

        close_prices[0] = base_price
        open_prices[0] = base_price

        for i in range(1, n):
            # Normal price change (90% of time)
            if np.random.rand() > 0.1:
                close_prices[i] = close_prices[i-1] * (1 + np.random.randn() * 0.01)
                open_prices[i] = close_prices[i-1] * (1 + np.random.randn() * 0.005)
            else:
                # Occasional large gap (10% of time)
                close_prices[i] = close_prices[i-1] * (1 + np.random.randn() * 0.03)
                open_prices[i] = close_prices[i-1] * (1 + np.random.randn() * 0.02)

        # Benchmark
        start = time.perf_counter()
        gap_ratio = calculate_price_gap_ratio(open_prices, close_prices)
        valid = validate_price_gap(gap_ratio)
        elapsed = (time.perf_counter() - start) * 1000

        print(f"\nPrice gap filter (10k candles): {elapsed:.2f}ms")
        assert elapsed < 50, f"Performance too slow: {elapsed:.2f}ms"

        # Verify some gaps were rejected
        num_rejected = np.sum(~valid[1:])  # Exclude first bar (NaN)
        assert num_rejected > 0, "Expected some gaps to be rejected"


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Test integrated noise filter workflow."""

    def test_full_noise_filter_workflow(self):
        """Test complete noise filter workflow with realistic data."""
        # Generate realistic market data
        n = 100
        np.random.seed(42)

        # Prices
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        bid = close - np.random.uniform(0.01, 0.05, n)
        ask = close + np.random.uniform(0.01, 0.05, n)

        # Volume
        volume = np.random.uniform(500, 2000, n)

        # Signals
        is_breakout = np.random.choice([True, False], n, p=[0.1, 0.9])
        is_illiquid = np.random.choice([True, False], n, p=[0.2, 0.8])

        # Step 1: Calculate SV ratio
        sv_ratio = calculate_spread_to_volatility_ratio(bid, ask, close, volatility_period=20)

        # Step 2: Calculate average volume
        avg_volume = calculate_average_volume(volume, period=20)

        # Step 3: Apply noise filter
        valid = apply_noise_filter(
            sv_ratio, volume, avg_volume, is_breakout, is_illiquid
        )

        # Step 4: Get rejection reasons for debugging
        reasons = calculate_rejection_reason(
            sv_ratio, volume, avg_volume, is_breakout, is_illiquid
        )

        # Verify some signals passed and some failed
        num_valid = np.sum(valid[20:])  # Skip first 20 bars (insufficient data)
        assert num_valid > 0, "No signals passed filter"
        assert num_valid < (n - 20), "All signals passed filter (unlikely)"

        # Verify rejection reasons are populated
        assert np.any(reasons == 0), "No valid signals (reason 0)"
        assert np.any(reasons > 0), "No rejected signals"

        print(f"\nIntegration test: {num_valid}/{n-20} signals passed filter")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
