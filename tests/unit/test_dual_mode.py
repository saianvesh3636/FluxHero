"""
Unit tests for Dual-Mode Strategy Engine.

Tests cover:
- Trend-following signal generation
- Mean-reversion signal generation
- Trailing stop calculation
- Fixed stop loss calculation
- Position sizing
- Signal blending
- Regime-based size adjustment
- Performance tracking
- Dynamic weight adjustment

Reference: FLUXHERO_REQUIREMENTS.md Feature 6
"""

import numpy as np
import pytest
from backend.strategy.dual_mode import (
    # Signal functions
    generate_trend_following_signals,
    generate_mean_reversion_signals,
    blend_signals,

    # Stop and sizing functions
    calculate_trailing_stop,
    calculate_fixed_stop_loss,
    calculate_position_size,
    adjust_size_for_regime,

    # Classes
    StrategyPerformance,
    DualModeStrategy,

    # Constants
    SIGNAL_NONE,
    SIGNAL_LONG,
    SIGNAL_SHORT,
    SIGNAL_EXIT_LONG,
    MODE_TREND_FOLLOWING,
    MODE_MEAN_REVERSION,
    MODE_NEUTRAL,
)


# ============================================================================
# Trend-Following Signal Tests
# ============================================================================

def test_trend_following_long_entry():
    """Test trend-following long entry signal (R6.1.1)."""
    # Create price data that crosses above KAMA + 0.5*ATR
    prices = np.array([100.0, 101.0, 102.0, 103.0, 105.0, 107.0])
    kama = np.array([100.0, 100.5, 101.0, 101.5, 102.0, 102.5])
    atr = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

    signals = generate_trend_following_signals(prices, kama, atr)

    # Entry threshold = KAMA + 0.5*ATR = 102.0 + 1.0 = 103.0 at index 4
    # Price crosses above at index 4 (103.0 <= 103.0, 105.0 > 103.0)
    # Should generate long signal
    assert np.any(signals == SIGNAL_LONG)


def test_trend_following_long_exit():
    """Test trend-following long exit signal (R6.1.2)."""
    # Create scenario: enter long, then price crosses below KAMA - 0.3*ATR
    prices = np.array([100.0, 105.0, 106.0, 107.0, 103.0, 101.0])
    kama = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    atr = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

    signals = generate_trend_following_signals(prices, kama, atr)

    # Should have entry and exit signals
    assert np.any(signals == SIGNAL_LONG)
    assert np.any(signals == SIGNAL_EXIT_LONG)


def test_trend_following_short_entry():
    """Test trend-following short entry signal."""
    # Create price data that crosses below KAMA - 0.5*ATR
    prices = np.array([100.0, 99.0, 98.0, 97.0, 95.0, 93.0])
    kama = np.array([100.0, 99.5, 99.0, 98.5, 98.0, 97.5])
    atr = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

    signals = generate_trend_following_signals(prices, kama, atr)

    # Should generate short signal
    assert np.any(signals == SIGNAL_SHORT)


def test_trend_following_no_signal_in_range():
    """Test that no signals are generated when price stays within bands."""
    # Price oscillates around KAMA within the entry bands
    prices = np.array([100.0, 100.5, 100.2, 100.8, 100.3])
    kama = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    atr = np.array([2.0, 2.0, 2.0, 2.0, 2.0])

    signals = generate_trend_following_signals(prices, kama, atr)

    # No entry signals should be generated
    assert not np.any(signals == SIGNAL_LONG)
    assert not np.any(signals == SIGNAL_SHORT)


def test_trend_following_nan_handling():
    """Test that NaN values don't generate signals."""
    prices = np.array([np.nan, 100.0, 105.0, 110.0])
    kama = np.array([100.0, np.nan, 102.0, 103.0])
    atr = np.array([2.0, 2.0, np.nan, 2.0])

    signals = generate_trend_following_signals(prices, kama, atr)

    # Signals at NaN positions should be SIGNAL_NONE
    assert signals[0] == SIGNAL_NONE
    assert signals[1] == SIGNAL_NONE
    assert signals[2] == SIGNAL_NONE


# ============================================================================
# Mean-Reversion Signal Tests
# ============================================================================

def test_mean_reversion_long_entry():
    """Test mean-reversion long entry (R6.2.1)."""
    # RSI < 30 and price at lower Bollinger Band
    prices = np.array([100.0, 99.0, 98.0, 97.0, 95.0])
    rsi = np.array([50.0, 40.0, 32.0, 28.0, 25.0])
    bb_lower = np.array([96.0, 96.0, 96.0, 96.0, 96.0])
    bb_middle = np.array([100.0, 100.0, 100.0, 100.0, 100.0])

    signals = generate_mean_reversion_signals(
        prices, rsi, bb_lower, bb_middle
    )

    # Should generate long signal when RSI < 30 and price <= lower band
    assert np.any(signals == SIGNAL_LONG)


def test_mean_reversion_long_exit_middle_band():
    """Test mean-reversion exit when price returns to middle band (R6.2.2)."""
    # Enter oversold, then return to middle
    prices = np.array([100.0, 95.0, 96.0, 98.0, 100.0])
    rsi = np.array([50.0, 25.0, 30.0, 45.0, 50.0])
    bb_lower = np.array([96.0, 96.0, 96.0, 96.0, 96.0])
    bb_middle = np.array([100.0, 100.0, 100.0, 100.0, 100.0])

    signals = generate_mean_reversion_signals(
        prices, rsi, bb_lower, bb_middle
    )

    # Should have entry and exit signals
    assert np.any(signals == SIGNAL_LONG)
    assert np.any(signals == SIGNAL_EXIT_LONG)


def test_mean_reversion_long_exit_rsi_overbought():
    """Test mean-reversion exit when RSI becomes overbought (R6.2.2)."""
    # Enter oversold, RSI rises to >70
    prices = np.array([100.0, 95.0, 96.0, 97.0, 98.0])
    rsi = np.array([50.0, 25.0, 40.0, 60.0, 75.0])
    bb_lower = np.array([96.0, 96.0, 96.0, 96.0, 96.0])
    bb_middle = np.array([100.0, 100.0, 100.0, 100.0, 100.0])

    signals = generate_mean_reversion_signals(
        prices, rsi, bb_lower, bb_middle
    )

    # Should exit when RSI > 70
    assert np.any(signals == SIGNAL_EXIT_LONG)


def test_mean_reversion_no_entry_without_both_conditions():
    """Test that entry requires BOTH RSI < 30 AND price at lower band."""
    # RSI < 30 but price not at lower band
    prices = np.array([100.0, 99.0, 98.0, 97.5])
    rsi = np.array([50.0, 35.0, 28.0, 25.0])
    bb_lower = np.array([95.0, 95.0, 95.0, 95.0])
    bb_middle = np.array([100.0, 100.0, 100.0, 100.0])

    signals = generate_mean_reversion_signals(
        prices, rsi, bb_lower, bb_middle
    )

    # No entry signal should be generated
    assert not np.any(signals == SIGNAL_LONG)


# ============================================================================
# Trailing Stop Tests
# ============================================================================

def test_trailing_stop_long():
    """Test trailing stop for long position (R6.1.3)."""
    prices = np.array([100.0, 102.0, 105.0, 103.0, 106.0])
    atr = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    entry_idx = 0

    stops = calculate_trailing_stop(prices, atr, entry_idx, is_long=True)

    # Stop = Peak - 2.5*ATR
    # At idx 0: peak=100, stop=100-5=95
    # At idx 2: peak=105, stop=105-5=100
    # At idx 4: peak=106, stop=106-5=101
    assert stops[0] == pytest.approx(95.0)
    assert stops[2] == pytest.approx(100.0)
    assert stops[4] == pytest.approx(101.0)

    # Stop should never decrease for long
    for i in range(1, len(stops)):
        if not np.isnan(stops[i]) and not np.isnan(stops[i-1]):
            assert stops[i] >= stops[i-1]


def test_trailing_stop_short():
    """Test trailing stop for short position."""
    prices = np.array([100.0, 98.0, 95.0, 97.0, 94.0])
    atr = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    entry_idx = 0

    stops = calculate_trailing_stop(prices, atr, entry_idx, is_long=False)

    # Stop = Trough + 2.5*ATR
    # At idx 0: trough=100, stop=100+5=105
    # At idx 2: trough=95, stop=95+5=100
    # At idx 4: trough=94, stop=94+5=99
    assert stops[0] == pytest.approx(105.0)
    assert stops[2] == pytest.approx(100.0)
    assert stops[4] == pytest.approx(99.0)

    # Stop should never increase for short
    for i in range(1, len(stops)):
        if not np.isnan(stops[i]) and not np.isnan(stops[i-1]):
            assert stops[i] <= stops[i-1]


def test_trailing_stop_invalid_entry_idx():
    """Test trailing stop with invalid entry index."""
    prices = np.array([100.0, 102.0, 105.0])
    atr = np.array([2.0, 2.0, 2.0])

    # Out of bounds entry index
    stops = calculate_trailing_stop(prices, atr, entry_idx=10, is_long=True)
    assert np.all(np.isnan(stops))


# ============================================================================
# Fixed Stop Loss Tests
# ============================================================================

def test_fixed_stop_loss_long():
    """Test fixed stop loss for long position (R6.2.3)."""
    entry_price = 100.0
    stop = calculate_fixed_stop_loss(entry_price, is_long=True, stop_pct=0.03)

    # Long stop = Entry * (1 - 0.03) = 100 * 0.97 = 97.0
    assert stop == pytest.approx(97.0)


def test_fixed_stop_loss_short():
    """Test fixed stop loss for short position."""
    entry_price = 100.0
    stop = calculate_fixed_stop_loss(entry_price, is_long=False, stop_pct=0.03)

    # Short stop = Entry * (1 + 0.03) = 100 * 1.03 = 103.0
    assert stop == pytest.approx(103.0)


def test_fixed_stop_loss_custom_percentage():
    """Test fixed stop loss with custom percentage."""
    entry_price = 100.0
    stop = calculate_fixed_stop_loss(entry_price, is_long=True, stop_pct=0.05)

    # 5% stop: 100 * 0.95 = 95.0
    assert stop == pytest.approx(95.0)


# ============================================================================
# Position Sizing Tests
# ============================================================================

def test_position_size_trend_following():
    """Test position sizing for trend-following (1% risk, R6.1.4)."""
    capital = 10000.0
    entry_price = 100.0
    stop_price = 97.0  # 3% stop
    risk_pct = 0.01  # 1% risk

    size = calculate_position_size(
        capital, entry_price, stop_price, risk_pct, is_long=True
    )

    # Risk amount = 10000 * 0.01 = 100
    # Price risk = 100 - 97 = 3
    # Shares = 100 / 3 = 33.33
    assert size == pytest.approx(33.33, rel=0.01)


def test_position_size_mean_reversion():
    """Test position sizing for mean-reversion (0.75% risk, R6.2.4)."""
    capital = 10000.0
    entry_price = 100.0
    stop_price = 97.0  # 3% stop
    risk_pct = 0.0075  # 0.75% risk

    size = calculate_position_size(
        capital, entry_price, stop_price, risk_pct, is_long=True
    )

    # Risk amount = 10000 * 0.0075 = 75
    # Price risk = 100 - 97 = 3
    # Shares = 75 / 3 = 25
    assert size == pytest.approx(25.0)


def test_position_size_zero_risk():
    """Test position sizing with zero price risk."""
    capital = 10000.0
    entry_price = 100.0
    stop_price = 100.0  # No risk
    risk_pct = 0.01

    size = calculate_position_size(
        capital, entry_price, stop_price, risk_pct, is_long=True
    )

    # Should return 0 to avoid division by zero
    assert size == 0.0


def test_position_size_short():
    """Test position sizing for short position."""
    capital = 10000.0
    entry_price = 100.0
    stop_price = 103.0  # 3% stop
    risk_pct = 0.01

    size = calculate_position_size(
        capital, entry_price, stop_price, risk_pct, is_long=False
    )

    # Risk amount = 10000 * 0.01 = 100
    # Price risk = |100 - 103| = 3
    # Shares = 100 / 3 = 33.33
    assert size == pytest.approx(33.33, rel=0.01)


# ============================================================================
# Signal Blending Tests
# ============================================================================

def test_blend_signals_agreement_required():
    """Test signal blending with agreement requirement (R6.3.3)."""
    trend_signals = np.array([SIGNAL_NONE, SIGNAL_LONG, SIGNAL_LONG, SIGNAL_NONE])
    mr_signals = np.array([SIGNAL_NONE, SIGNAL_LONG, SIGNAL_SHORT, SIGNAL_NONE])

    blended = blend_signals(trend_signals, mr_signals, require_agreement=True)

    # Only index 1 has agreement (both LONG)
    assert blended[0] == SIGNAL_NONE
    assert blended[1] == SIGNAL_LONG
    assert blended[2] == SIGNAL_NONE  # Disagreement
    assert blended[3] == SIGNAL_NONE


def test_blend_signals_no_agreement_required():
    """Test signal blending without agreement requirement (50/50 weight)."""
    trend_signals = np.array([SIGNAL_NONE, SIGNAL_LONG, SIGNAL_NONE, SIGNAL_SHORT])
    mr_signals = np.array([SIGNAL_NONE, SIGNAL_NONE, SIGNAL_LONG, SIGNAL_NONE])

    blended = blend_signals(trend_signals, mr_signals, require_agreement=False)

    # Take whichever signal is present
    assert blended[0] == SIGNAL_NONE
    assert blended[1] == SIGNAL_LONG  # From trend
    assert blended[2] == SIGNAL_LONG  # From MR
    assert blended[3] == SIGNAL_SHORT  # From trend


def test_blend_signals_all_none():
    """Test blending when both strategies have no signals."""
    trend_signals = np.array([SIGNAL_NONE, SIGNAL_NONE, SIGNAL_NONE])
    mr_signals = np.array([SIGNAL_NONE, SIGNAL_NONE, SIGNAL_NONE])

    blended = blend_signals(trend_signals, mr_signals)

    assert np.all(blended == SIGNAL_NONE)


# ============================================================================
# Regime-Based Size Adjustment Tests
# ============================================================================

def test_adjust_size_neutral_regime():
    """Test size reduction for neutral regime (R6.3.2)."""
    base_size = 100.0
    adjusted = adjust_size_for_regime(base_size, MODE_NEUTRAL)

    # 30% reduction: 100 * 0.7 = 70
    assert adjusted == pytest.approx(70.0)


def test_adjust_size_trend_regime():
    """Test no size adjustment for trend regime."""
    base_size = 100.0
    adjusted = adjust_size_for_regime(base_size, MODE_TREND_FOLLOWING)

    # No adjustment
    assert adjusted == pytest.approx(100.0)


def test_adjust_size_mean_reversion_regime():
    """Test no size adjustment for mean-reversion regime."""
    base_size = 100.0
    adjusted = adjust_size_for_regime(base_size, MODE_MEAN_REVERSION)

    # No adjustment
    assert adjusted == pytest.approx(100.0)


# ============================================================================
# Performance Tracking Tests
# ============================================================================

def test_strategy_performance_win_rate():
    """Test win rate calculation (R6.4.1)."""
    perf = StrategyPerformance()

    # Add trades: 3 wins, 2 losses
    perf.add_trade(100.0, MODE_TREND_FOLLOWING)
    perf.add_trade(-50.0, MODE_TREND_FOLLOWING)
    perf.add_trade(75.0, MODE_TREND_FOLLOWING)
    perf.add_trade(-25.0, MODE_TREND_FOLLOWING)
    perf.add_trade(150.0, MODE_TREND_FOLLOWING)

    win_rate = perf.get_win_rate()
    assert win_rate == pytest.approx(0.6)  # 3/5 = 60%


def test_strategy_performance_total_return():
    """Test total return calculation."""
    perf = StrategyPerformance()

    perf.add_trade(100.0, MODE_TREND_FOLLOWING)
    perf.add_trade(-50.0, MODE_TREND_FOLLOWING)
    perf.add_trade(75.0, MODE_TREND_FOLLOWING)

    total_return = perf.get_total_return()
    assert total_return == pytest.approx(125.0)


def test_strategy_performance_max_drawdown():
    """Test max drawdown calculation."""
    perf = StrategyPerformance()

    # Sequence: +100, -150 (DD from 100 to -50 = 150%), +200
    perf.add_trade(100.0, MODE_TREND_FOLLOWING)
    perf.add_trade(-150.0, MODE_TREND_FOLLOWING)
    perf.add_trade(200.0, MODE_TREND_FOLLOWING)

    max_dd = perf.get_max_drawdown()
    # Peak = 100, trough = -50, DD = (100 - (-50)) / 100 = 1.5
    assert max_dd == pytest.approx(1.5)


def test_strategy_performance_no_trades():
    """Test performance metrics with no trades."""
    perf = StrategyPerformance()

    assert perf.get_win_rate() == 0.0
    assert perf.get_total_return() == 0.0
    assert perf.get_max_drawdown() == 0.0
    assert perf.get_sharpe_ratio() == 0.0


# ============================================================================
# DualModeStrategy Tests
# ============================================================================

def test_dual_mode_get_active_mode():
    """Test mode selection based on regime (R6.1.5, R6.2.5)."""
    strategy = DualModeStrategy()

    # REGIME_STRONG_TREND (2) → MODE_TREND_FOLLOWING
    assert strategy.get_active_mode(2) == MODE_TREND_FOLLOWING

    # REGIME_MEAN_REVERSION (0) → MODE_MEAN_REVERSION
    assert strategy.get_active_mode(0) == MODE_MEAN_REVERSION

    # REGIME_NEUTRAL (1) → MODE_NEUTRAL
    assert strategy.get_active_mode(1) == MODE_NEUTRAL


def test_dual_mode_update_performance():
    """Test performance tracking updates."""
    strategy = DualModeStrategy()

    # Add trades to different modes
    strategy.update_performance(100.0, MODE_TREND_FOLLOWING)
    strategy.update_performance(-50.0, MODE_MEAN_REVERSION)
    strategy.update_performance(75.0, MODE_NEUTRAL)

    # Check that trades were recorded
    assert strategy.trend_perf.wins == 1
    assert strategy.mr_perf.losses == 1
    assert strategy.neutral_perf.wins == 1


def test_dual_mode_rebalance_weights():
    """Test dynamic weight adjustment (R6.4.2, R6.4.3)."""
    strategy = DualModeStrategy()

    # Add 20 losing trades to trend mode
    for _ in range(20):
        strategy.update_performance(-10.0, MODE_TREND_FOLLOWING)

    # Add 20 winning trades to MR mode
    for _ in range(20):
        strategy.update_performance(10.0, MODE_MEAN_REVERSION)

    # Rebalance
    strategy.rebalance_weights(min_trades=20)

    # Trend weight should be reduced, MR weight should be maintained/increased
    assert strategy.trend_weight < 1.0
    assert strategy.mr_weight >= 0.5


def test_dual_mode_performance_summary():
    """Test comprehensive performance summary."""
    strategy = DualModeStrategy()

    # Add some trades
    strategy.update_performance(100.0, MODE_TREND_FOLLOWING)
    strategy.update_performance(-50.0, MODE_TREND_FOLLOWING)
    strategy.update_performance(75.0, MODE_MEAN_REVERSION)

    summary = strategy.get_performance_summary()

    # Check structure
    assert 'trend_following' in summary
    assert 'mean_reversion' in summary
    assert 'neutral' in summary

    # Check trend-following stats
    assert summary['trend_following']['total_trades'] == 2
    assert summary['trend_following']['win_rate'] == 0.5
    assert summary['trend_following']['total_return'] == 50.0

    # Check mean-reversion stats
    assert summary['mean_reversion']['total_trades'] == 1
    assert summary['mean_reversion']['win_rate'] == 1.0


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_trend_following_workflow():
    """Test complete trend-following workflow."""
    # Create trending market data
    np.random.seed(42)
    n = 100
    prices = np.cumsum(np.random.randn(n) * 0.5 + 0.1) + 100
    kama = np.convolve(prices, np.ones(10)/10, mode='same')
    atr = np.ones(n) * 2.0

    # Generate signals
    signals = generate_trend_following_signals(prices, kama, atr)

    # Should have some long entries in uptrend
    assert np.any(signals == SIGNAL_LONG)

    # Find first long entry
    entry_indices = np.where(signals == SIGNAL_LONG)[0]
    if len(entry_indices) > 0:
        entry_idx = entry_indices[0]

        # Calculate trailing stop
        stops = calculate_trailing_stop(prices, atr, entry_idx, is_long=True)

        # Stop should be calculated from entry point onward
        assert not np.isnan(stops[entry_idx])

        # Calculate position size
        size = calculate_position_size(
            capital=10000.0,
            entry_price=prices[entry_idx],
            stop_price=stops[entry_idx],
            risk_pct=0.01,
            is_long=True
        )

        assert size > 0


def test_full_mean_reversion_workflow():
    """Test complete mean-reversion workflow."""
    # Create mean-reverting data with clear oversold conditions
    n = 100
    prices = np.ones(n) * 100.0
    rsi = np.ones(n) * 50.0
    bb_middle = np.ones(n) * 100.0
    bb_lower = np.ones(n) * 95.0

    # Create oversold condition: price drops to lower band, RSI < 30
    prices[20] = 95.0  # Touch lower band
    rsi[20] = 25.0     # Oversold

    # Then price recovers to middle band
    prices[25] = 100.0  # Back to middle
    rsi[25] = 50.0

    # Generate signals
    signals = generate_mean_reversion_signals(
        prices, rsi, bb_lower, bb_middle
    )

    # Should have entry signal
    assert signals[20] == SIGNAL_LONG or np.any(signals == SIGNAL_LONG)

    # Should have exit signal
    assert np.any(signals == SIGNAL_EXIT_LONG)


def test_regime_switching():
    """Test strategy adapts to regime changes."""
    strategy = DualModeStrategy()

    # Start in strong trend (regime 2)
    mode = strategy.get_active_mode(regime=2)
    assert mode == MODE_TREND_FOLLOWING

    # Switch to mean reversion (regime 0)
    mode = strategy.get_active_mode(regime=0)
    assert mode == MODE_MEAN_REVERSION

    # Switch to neutral (regime 1)
    mode = strategy.get_active_mode(regime=1)
    assert mode == MODE_NEUTRAL


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_empty_arrays():
    """Test handling of empty input arrays."""
    empty = np.array([])

    signals = generate_trend_following_signals(empty, empty, empty)
    assert len(signals) == 0

    signals = generate_mean_reversion_signals(empty, empty, empty, empty)
    assert len(signals) == 0


def test_single_value_arrays():
    """Test handling of single-value arrays."""
    single = np.array([100.0])

    signals = generate_trend_following_signals(single, single, single)
    assert len(signals) == 1
    assert signals[0] == SIGNAL_NONE


def test_all_nan_arrays():
    """Test handling of all-NaN arrays."""
    nans = np.full(10, np.nan)

    signals = generate_trend_following_signals(nans, nans, nans)
    assert np.all(signals == SIGNAL_NONE)


# ============================================================================
# Performance Benchmarks
# ============================================================================

def test_performance_trend_signals_10k_candles():
    """Test trend signal generation performance on 10k candles."""
    import time

    n = 10000
    prices = np.cumsum(np.random.randn(n) * 0.5) + 100
    kama = np.convolve(prices, np.ones(20)/20, mode='same')
    atr = np.ones(n) * 2.0

    # Warmup JIT
    generate_trend_following_signals(prices[:100], kama[:100], atr[:100])

    # Benchmark
    start = time.time()
    signals = generate_trend_following_signals(prices, kama, atr)
    elapsed = time.time() - start

    # Should complete in <100ms
    assert elapsed < 0.1
    assert len(signals) == n


def test_performance_mr_signals_10k_candles():
    """Test mean-reversion signal generation performance on 10k candles."""
    import time

    n = 10000
    prices = 100 + 10 * np.sin(np.linspace(0, 100*np.pi, n))
    rsi = 50 + 20 * np.sin(np.linspace(0, 100*np.pi, n))
    bb_lower = np.ones(n) * 90
    bb_middle = np.ones(n) * 100

    # Warmup JIT
    generate_mean_reversion_signals(
        prices[:100], rsi[:100], bb_lower[:100], bb_middle[:100]
    )

    # Benchmark
    start = time.time()
    signals = generate_mean_reversion_signals(
        prices, rsi, bb_lower, bb_middle
    )
    elapsed = time.time() - start

    # Should complete in <100ms
    assert elapsed < 0.1
    assert len(signals) == n


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
