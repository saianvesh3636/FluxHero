"""
Unit tests for volatility-adaptive smoothing module.

Tests cover:
- ATR moving average calculation
- Volatility state classification (LOW/NORMAL/HIGH)
- Dynamic period adjustment
- Multi-timeframe volatility spike detection
- Volatility-alpha linkage
- Adaptive EMA with volatility
- Performance benchmarks
"""

import os

# Import volatility functions
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from backend.computation.indicators import calculate_atr
from backend.computation.volatility import (
    VOL_STATE_HIGH,
    VOL_STATE_LOW,
    VOL_STATE_NORMAL,
    adjust_period_for_volatility,
    calculate_adaptive_ema_with_volatility,
    calculate_atr_ma,
    calculate_volatility_alpha,
    classify_volatility_state,
    detect_volatility_spike,
    get_position_size_multiplier,
    get_stop_loss_multiplier,
)

# ========================
# ATR Moving Average Tests
# ========================


def test_calculate_atr_ma_basic():
    """Test basic ATR moving average calculation."""
    atr = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 2.5, 2.0, 1.5, 1.0], dtype=np.float64)
    atr_ma = calculate_atr_ma(atr, period=3)

    # First 2 values should be NaN
    assert np.isnan(atr_ma[0])
    assert np.isnan(atr_ma[1])

    # Third value should be average of first 3 values
    assert np.isclose(atr_ma[2], (1.0 + 1.5 + 2.0) / 3.0)

    # Fourth value should be average of values 1-3
    assert np.isclose(atr_ma[3], (1.5 + 2.0 + 2.5) / 3.0)


def test_calculate_atr_ma_handles_nan():
    """Test that ATR MA handles NaN values correctly."""
    atr = np.array([1.0, np.nan, 2.0, 3.0, 4.0], dtype=np.float64)
    atr_ma = calculate_atr_ma(atr, period=3)

    # Should skip NaN values in calculation
    assert not np.isnan(atr_ma[4])


def test_calculate_atr_ma_insufficient_data():
    """Test ATR MA with insufficient data."""
    atr = np.array([1.0, 2.0], dtype=np.float64)
    atr_ma = calculate_atr_ma(atr, period=5)

    # All values should be NaN
    assert np.all(np.isnan(atr_ma))


# ==================================
# Volatility State Classification Tests
# ==================================


def test_classify_volatility_state_low():
    """Test classification of low volatility state."""
    atr = np.array([0.8, 0.9, 1.0], dtype=np.float64)
    atr_ma = np.array([2.0, 2.0, 2.0], dtype=np.float64)

    states = classify_volatility_state(atr, atr_ma)

    # Ratios: 0.4, 0.45, 0.5
    # 0.8/2.0 = 0.4 < 0.5 → LOW
    # 0.9/2.0 = 0.45 < 0.5 → LOW
    # 1.0/2.0 = 0.5, which is NOT < 0.5, so NORMAL
    assert states[0] == VOL_STATE_LOW
    assert states[1] == VOL_STATE_LOW
    assert states[2] == VOL_STATE_NORMAL  # Exactly at threshold


def test_classify_volatility_state_normal():
    """Test classification of normal volatility state."""
    atr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    atr_ma = np.array([2.0, 2.0, 2.0], dtype=np.float64)

    states = classify_volatility_state(atr, atr_ma)

    # Ratios: 0.5, 1.0, 1.5
    # 1.0/2.0 = 0.5, which is NOT < 0.5, so NORMAL
    # 2.0/2.0 = 1.0 (NORMAL)
    # 3.0/2.0 = 1.5, which is NOT > 1.5, so NORMAL (at boundary)
    assert states[0] == VOL_STATE_NORMAL  # At low threshold
    assert states[1] == VOL_STATE_NORMAL
    assert states[2] == VOL_STATE_NORMAL  # Exactly at high threshold


def test_classify_volatility_state_high():
    """Test classification of high volatility state."""
    atr = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    atr_ma = np.array([2.0, 2.0, 2.0], dtype=np.float64)

    states = classify_volatility_state(atr, atr_ma)

    # All should be HIGH (ATR > 1.5 × ATR_MA)
    assert np.all(states == VOL_STATE_HIGH)


def test_classify_volatility_state_custom_thresholds():
    """Test volatility classification with custom thresholds."""
    atr = np.array([1.5, 2.0, 4.5], dtype=np.float64)
    atr_ma = np.array([2.0, 2.0, 2.0], dtype=np.float64)

    # Custom thresholds: low=0.7, high=2.0
    states = classify_volatility_state(atr, atr_ma, low_threshold=0.7, high_threshold=2.0)

    # Ratios: 0.75, 1.0, 2.25
    # 1.5/2.0 = 0.75, which is NOT < 0.7, so NORMAL
    # 2.0/2.0 = 1.0 (NORMAL)
    # 4.5/2.0 = 2.25 > 2.0 (HIGH)
    assert states[0] == VOL_STATE_NORMAL  # 0.75 is NOT < 0.7
    assert states[1] == VOL_STATE_NORMAL  # 2.0/2.0 = 1.0
    assert states[2] == VOL_STATE_HIGH  # 4.5/2.0 = 2.25 > 2.0


def test_classify_volatility_state_handles_nan():
    """Test that state classification handles NaN values."""
    atr = np.array([1.0, np.nan, 3.0], dtype=np.float64)
    atr_ma = np.array([2.0, 2.0, 2.0], dtype=np.float64)

    states = classify_volatility_state(atr, atr_ma)

    # NaN should default to NORMAL
    assert states[1] == VOL_STATE_NORMAL


# ============================
# Period Adjustment Tests
# ============================


def test_adjust_period_for_high_volatility():
    """Test period adjustment for high volatility (shorten by 30%)."""
    adjusted = adjust_period_for_volatility(20, VOL_STATE_HIGH)

    # Should be 20 × 0.7 = 14
    assert adjusted == 14


def test_adjust_period_for_low_volatility():
    """Test period adjustment for low volatility (lengthen by 30%)."""
    adjusted = adjust_period_for_volatility(20, VOL_STATE_LOW)

    # Should be 20 × 1.3 = 26
    assert adjusted == 26


def test_adjust_period_for_normal_volatility():
    """Test period adjustment for normal volatility (no change)."""
    adjusted = adjust_period_for_volatility(20, VOL_STATE_NORMAL)

    # Should remain 20
    assert adjusted == 20


def test_adjust_period_minimum_enforcement():
    """Test that adjusted period never goes below minimum of 2."""
    adjusted = adjust_period_for_volatility(2, VOL_STATE_HIGH)

    # 2 × 0.7 = 1.4, should be clamped to 2
    assert adjusted >= 2


def test_adjust_period_custom_multipliers():
    """Test period adjustment with custom multipliers."""
    adjusted_high = adjust_period_for_volatility(
        100, VOL_STATE_HIGH, low_vol_multiplier=1.5, high_vol_multiplier=0.5
    )
    adjusted_low = adjust_period_for_volatility(
        100, VOL_STATE_LOW, low_vol_multiplier=1.5, high_vol_multiplier=0.5
    )

    assert adjusted_high == 50  # 100 × 0.5
    assert adjusted_low == 150  # 100 × 1.5


# ====================================
# Volatility Spike Detection Tests
# ====================================


def test_detect_volatility_spike_basic():
    """Test basic volatility spike detection."""
    atr_5min = np.array([2.0, 4.5, 3.5, 2.0], dtype=np.float64)
    atr_1hour = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64)

    spikes = detect_volatility_spike(atr_5min, atr_1hour, spike_threshold=2.0)

    assert not spikes[0]  # 2.0 / 2.0 = 1.0 (not spike)
    assert spikes[1]  # 4.5 / 2.0 = 2.25 > 2.0 (spike)
    assert not spikes[2]  # 3.5 / 2.0 = 1.75 (not > 2.0)
    assert not spikes[3]  # 2.0 / 2.0 = 1.0 (not spike)


def test_detect_volatility_spike_no_spikes():
    """Test spike detection when no spikes occur."""
    atr_5min = np.array([1.5, 1.8, 1.6, 1.7], dtype=np.float64)
    atr_1hour = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64)

    spikes = detect_volatility_spike(atr_5min, atr_1hour)

    # No values exceed 2× threshold
    assert np.all(~spikes)


def test_detect_volatility_spike_custom_threshold():
    """Test spike detection with custom threshold."""
    atr_5min = np.array([3.0, 6.0], dtype=np.float64)
    atr_1hour = np.array([2.0, 2.0], dtype=np.float64)

    spikes = detect_volatility_spike(atr_5min, atr_1hour, spike_threshold=2.5)

    assert not spikes[0]  # 3.0 / 2.0 = 1.5 (not > 2.5)
    assert spikes[1]  # 6.0 / 2.0 = 3.0 > 2.5


def test_detect_volatility_spike_handles_nan():
    """Test spike detection with NaN values."""
    atr_5min = np.array([np.nan, 4.0], dtype=np.float64)
    atr_1hour = np.array([2.0, 2.0], dtype=np.float64)

    spikes = detect_volatility_spike(atr_5min, atr_1hour)

    assert not spikes[0]  # NaN should not trigger spike


# ==============================
# Volatility-Alpha Linkage Tests
# ==============================


def test_calculate_volatility_alpha_low_vol():
    """Test alpha calculation for low volatility (should be minimum)."""
    atr = np.array([1.0], dtype=np.float64)
    atr_ma = np.array([2.0], dtype=np.float64)

    alphas = calculate_volatility_alpha(atr, atr_ma, min_alpha=0.1, max_alpha=0.6)

    # Ratio = 1.0/2.0 = 0.5 (at low threshold) → min alpha
    assert np.isclose(alphas[0], 0.1, atol=0.01)


def test_calculate_volatility_alpha_high_vol():
    """Test alpha calculation for high volatility (should be maximum)."""
    atr = np.array([3.0], dtype=np.float64)
    atr_ma = np.array([2.0], dtype=np.float64)

    alphas = calculate_volatility_alpha(atr, atr_ma, min_alpha=0.1, max_alpha=0.6)

    # Ratio = 3.0/2.0 = 1.5 (at high threshold) → max alpha
    assert np.isclose(alphas[0], 0.6, atol=0.01)


def test_calculate_volatility_alpha_normal_vol():
    """Test alpha calculation for normal volatility (should be midpoint)."""
    atr = np.array([2.0], dtype=np.float64)
    atr_ma = np.array([2.0], dtype=np.float64)

    alphas = calculate_volatility_alpha(atr, atr_ma, min_alpha=0.1, max_alpha=0.6)

    # Ratio = 2.0/2.0 = 1.0 (middle) → mid alpha (~0.35)
    assert 0.3 < alphas[0] < 0.4


def test_calculate_volatility_alpha_interpolation():
    """Test that alpha interpolates correctly between min and max."""
    atr = np.array([1.25, 1.5, 1.75], dtype=np.float64)
    atr_ma = np.array([2.0, 2.0, 2.0], dtype=np.float64)

    alphas = calculate_volatility_alpha(atr, atr_ma, min_alpha=0.1, max_alpha=0.6)

    # Ratios: 0.625, 0.75, 0.875 (normalized to [0.5, 1.5])
    # Should be monotonically increasing
    assert alphas[0] < alphas[1] < alphas[2]
    assert 0.1 <= alphas[0] <= 0.6
    assert 0.1 <= alphas[1] <= 0.6
    assert 0.1 <= alphas[2] <= 0.6


def test_calculate_volatility_alpha_handles_nan():
    """Test alpha calculation with NaN values."""
    atr = np.array([np.nan, 2.0], dtype=np.float64)
    atr_ma = np.array([2.0, 2.0], dtype=np.float64)

    alphas = calculate_volatility_alpha(atr, atr_ma)

    # NaN should result in default mid-range alpha
    assert not np.isnan(alphas[0])
    assert 0.1 <= alphas[0] <= 0.6


# ===============================
# Adaptive EMA with Volatility Tests
# ===============================


def test_adaptive_ema_basic():
    """Test basic adaptive EMA calculation."""
    prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0], dtype=np.float64)
    atr = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=np.float64)
    atr_ma = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float64)

    ema = calculate_adaptive_ema_with_volatility(prices, atr, atr_ma, base_period=3)

    # Should have values after initialization period
    assert not np.isnan(ema[-1])


def test_adaptive_ema_high_volatility_follows_price():
    """Test that adaptive EMA follows price closely in high volatility."""
    # Strong uptrend with high volatility
    prices = np.linspace(100, 120, 30).astype(np.float64)
    atr = np.full(30, 5.0, dtype=np.float64)  # High ATR
    atr_ma = np.full(30, 2.0, dtype=np.float64)  # ATR > 1.5 × ATR_MA

    ema = calculate_adaptive_ema_with_volatility(prices, atr, atr_ma, base_period=10)

    valid_ema = ema[~np.isnan(ema)]
    valid_prices = prices[~np.isnan(ema)]

    # In high vol, EMA should stay within 5% of price
    if len(valid_ema) > 0:
        percent_diff = np.abs((valid_ema[-1] - valid_prices[-1]) / valid_prices[-1])
        assert percent_diff < 0.05


def test_adaptive_ema_low_volatility_smooths():
    """Test that adaptive EMA smooths noise in low volatility."""
    # Noisy sideways price action with low volatility
    np.random.seed(42)
    prices = 100.0 + np.random.randn(30) * 0.5
    prices = prices.astype(np.float64)
    atr = np.full(30, 0.5, dtype=np.float64)  # Low ATR
    atr_ma = np.full(30, 2.0, dtype=np.float64)  # ATR < 0.5 × ATR_MA

    ema = calculate_adaptive_ema_with_volatility(prices, atr, atr_ma, base_period=10)

    valid_ema = ema[~np.isnan(ema)]

    # EMA should be smoother than prices (lower std dev)
    if len(valid_ema) > 5:
        ema_volatility = np.std(np.diff(valid_ema))
        price_volatility = np.std(np.diff(prices[-len(valid_ema) :]))
        assert ema_volatility < price_volatility


def test_adaptive_ema_insufficient_data():
    """Test adaptive EMA with insufficient data."""
    prices = np.array([100.0, 101.0], dtype=np.float64)
    atr = np.array([1.0, 1.5], dtype=np.float64)
    atr_ma = np.array([2.0, 2.0], dtype=np.float64)

    ema = calculate_adaptive_ema_with_volatility(prices, atr, atr_ma, base_period=10)

    # All values should be NaN
    assert np.all(np.isnan(ema))


# ===================================
# Multiplier Utility Function Tests
# ===================================


def test_get_stop_loss_multiplier_spike():
    """Test stop loss multiplier during volatility spike."""
    assert get_stop_loss_multiplier(True) == 1.5


def test_get_stop_loss_multiplier_normal():
    """Test stop loss multiplier during normal volatility."""
    assert get_stop_loss_multiplier(False) == 1.0


def test_get_position_size_multiplier_spike():
    """Test position size multiplier during volatility spike."""
    assert get_position_size_multiplier(True) == 0.7


def test_get_position_size_multiplier_normal():
    """Test position size multiplier during normal volatility."""
    assert get_position_size_multiplier(False) == 1.0


# ==========================
# Integration Tests
# ==========================


def test_full_volatility_workflow():
    """Test complete volatility analysis workflow."""
    # Generate synthetic price data
    np.random.seed(42)
    n = 100
    high = 100 + np.cumsum(np.random.randn(n) * 0.5)
    low = high - np.abs(np.random.randn(n) * 0.3)
    close = low + (high - low) * np.random.rand(n)

    high = high.astype(np.float64)
    low = low.astype(np.float64)
    close = close.astype(np.float64)

    # Step 1: Calculate ATR
    atr = calculate_atr(high, low, close, period=14)

    # Step 2: Calculate ATR moving average
    atr_ma = calculate_atr_ma(atr, period=50)

    # Step 3: Classify volatility states
    vol_states = classify_volatility_state(atr, atr_ma)

    # Step 4: Adjust periods based on volatility
    base_period = 20
    adjusted_periods = np.array(
        [adjust_period_for_volatility(base_period, state) for state in vol_states]
    )

    # Step 5: Detect volatility spikes
    spikes = detect_volatility_spike(atr, atr_ma)

    # Step 6: Calculate adaptive alphas
    alphas = calculate_volatility_alpha(atr, atr_ma)

    # Step 7: Calculate adaptive EMA
    ema = calculate_adaptive_ema_with_volatility(close, atr, atr_ma, base_period=20)

    # Assertions
    assert len(vol_states) == n
    assert len(adjusted_periods) == n
    assert len(spikes) == n
    assert len(alphas) == n
    assert len(ema) == n

    # Check that we have valid values after warm-up period
    assert np.sum(~np.isnan(atr)) > 0
    assert np.sum(~np.isnan(atr_ma)) > 0
    assert np.sum(~np.isnan(ema)) > 0


def test_volatility_state_persistence():
    """Test that volatility states can persist over multiple bars."""
    # Create sustained high volatility period
    atr = np.array([3.0, 3.2, 3.1, 3.3, 3.2, 3.4, 3.3], dtype=np.float64)
    atr_ma = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float64)

    states = classify_volatility_state(atr, atr_ma)

    # Ratios: 1.5, 1.6, 1.55, 1.65, 1.6, 1.7, 1.65
    # First is exactly 1.5 (at threshold, so NORMAL)
    # Rest are > 1.5 (HIGH)
    assert states[0] == VOL_STATE_NORMAL  # 3.0/2.0 = 1.5 (at threshold)
    assert np.all(states[1:] == VOL_STATE_HIGH)  # All others > 1.5


# ==========================
# Performance Benchmarks
# ==========================


def test_performance_volatility_suite_10k_candles():
    """Test that full volatility suite completes in <200ms for 10k candles."""
    n = 10000
    np.random.seed(42)

    high = 100 + np.cumsum(np.random.randn(n) * 0.5)
    low = high - np.abs(np.random.randn(n) * 0.3)
    close = low + (high - low) * np.random.rand(n)
    prices = close.astype(np.float64)
    high = high.astype(np.float64)
    low = low.astype(np.float64)

    start = time.time()

    # Calculate ATR
    atr = calculate_atr(high, low, close, period=14)

    # Calculate ATR MA
    atr_ma = calculate_atr_ma(atr, period=50)

    # Classify volatility states
    classify_volatility_state(atr, atr_ma)

    # Detect spikes
    detect_volatility_spike(atr, atr_ma)

    # Calculate alphas
    calculate_volatility_alpha(atr, atr_ma)

    # Calculate adaptive EMA
    calculate_adaptive_ema_with_volatility(prices, atr, atr_ma, base_period=20)

    elapsed = time.time() - start

    print(f"\n10k candles volatility suite: {elapsed * 1000:.2f}ms")
    assert elapsed < 0.2, f"Volatility suite took {elapsed * 1000:.2f}ms (target: <200ms)"


def test_performance_individual_functions():
    """Test individual function performance."""
    n = 10000
    np.random.seed(42)

    atr = np.abs(np.random.randn(n)).astype(np.float64) * 2.0
    atr_ma = np.full(n, 2.0, dtype=np.float64)
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5).astype(np.float64)

    # Test ATR MA
    start = time.time()
    calculate_atr_ma(atr, period=50)
    elapsed_atr_ma = time.time() - start

    # Test state classification
    start = time.time()
    classify_volatility_state(atr, atr_ma)
    elapsed_classify = time.time() - start

    # Test alpha calculation
    start = time.time()
    calculate_volatility_alpha(atr, atr_ma)
    elapsed_alpha = time.time() - start

    # Test adaptive EMA
    start = time.time()
    calculate_adaptive_ema_with_volatility(prices, atr, atr_ma, base_period=20)
    elapsed_ema = time.time() - start

    print("\nIndividual function performance (10k candles):")
    print(f"  ATR MA: {elapsed_atr_ma * 1000:.2f}ms")
    print(f"  Classify: {elapsed_classify * 1000:.2f}ms")
    print(f"  Alpha: {elapsed_alpha * 1000:.2f}ms")
    print(f"  Adaptive EMA: {elapsed_ema * 1000:.2f}ms")

    # Each should be reasonably fast
    assert elapsed_atr_ma < 0.1
    assert elapsed_classify < 0.05
    assert elapsed_alpha < 0.05
    assert elapsed_ema < 0.1


# ==========================
# Edge Case Tests
# ==========================


def test_edge_case_empty_arrays():
    """Test functions with empty arrays."""
    empty = np.array([], dtype=np.float64)

    atr_ma = calculate_atr_ma(empty, period=10)
    assert len(atr_ma) == 0

    alphas = calculate_volatility_alpha(empty, empty)
    assert len(alphas) == 0


def test_edge_case_single_value():
    """Test functions with single value."""
    single = np.array([1.0], dtype=np.float64)

    atr_ma = calculate_atr_ma(single, period=5)
    assert len(atr_ma) == 1
    assert np.isnan(atr_ma[0])


def test_edge_case_constant_atr():
    """Test with constant ATR values."""
    atr = np.full(20, 2.0, dtype=np.float64)
    atr_ma = calculate_atr_ma(atr, period=10)

    # ATR MA should also be constant after warm-up
    valid_ma = atr_ma[~np.isnan(atr_ma)]
    assert np.allclose(valid_ma, 2.0)


def test_edge_case_zero_atr_ma():
    """Test handling of zero ATR MA (division by zero)."""
    atr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    atr_ma = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    # Should not crash, should handle gracefully
    states = classify_volatility_state(atr, atr_ma)
    assert len(states) == 3
    # All should remain NORMAL (default) since we skip zero ATR_MA
    assert np.all(states == VOL_STATE_NORMAL)

    alphas = calculate_volatility_alpha(atr, atr_ma)
    assert len(alphas) == 3
    # Should have default mid-range alphas
    assert not np.any(np.isnan(alphas))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
