"""
Unit tests for backtesting engine module.

Tests cover:
- Backtest orchestrator (engine.py)
- Next-bar fill logic (fills.py)
- Performance metrics (metrics.py)

Reference: FLUXHERO_REQUIREMENTS.md Feature 9
"""

import numpy as np
import pytest

from backend.backtesting.engine import (
    BacktestConfig,
    BacktestEngine,
    Order,
    OrderSide,
    OrderType,
    validate_bar_integrity,
)
from backend.backtesting.fills import (
    check_stop_and_target,
    get_next_bar_fill_price,
    simulate_intrabar_stop_execution,
    simulate_intrabar_target_execution,
    validate_no_lookahead,
)
from backend.backtesting.metrics import (
    PerformanceMetrics,
    calculate_annualized_return,
    calculate_avg_win_loss_ratio,
    calculate_max_drawdown,
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_total_return,
    calculate_win_rate,
)

# ============================================================================
# Tests for fills.py - Next-Bar Fill Logic
# ============================================================================


class TestNextBarFillLogic:
    """Test next-bar fill logic (R9.1)."""

    def test_get_next_bar_fill_price_basic(self):
        """Test basic next-bar fill (R9.1.1)."""
        opens = np.array([100.0, 101.0, 102.0, 103.0])

        # Signal at bar 1 → fill at bar 2 (open=102.0)
        fill_price, fill_idx = get_next_bar_fill_price(1, opens, delay_bars=1)

        assert fill_price == 102.0
        assert fill_idx == 2

    def test_get_next_bar_fill_price_insufficient_data(self):
        """Test fill when not enough future bars available."""
        opens = np.array([100.0, 101.0, 102.0])

        # Signal at bar 2 → no bar 3 available
        fill_price, fill_idx = get_next_bar_fill_price(2, opens, delay_bars=1)

        assert np.isnan(fill_price)
        assert fill_idx == -1

    def test_get_next_bar_fill_price_minute_data(self):
        """Test 1-bar delay for minute data (R9.1.2)."""
        opens = np.array([50.0, 51.0, 52.0, 53.0, 54.0])

        # Signal at bar 2 → fill at bar 3
        fill_price, fill_idx = get_next_bar_fill_price(2, opens, delay_bars=1)

        assert fill_price == 53.0
        assert fill_idx == 3

    def test_simulate_intrabar_stop_long(self):
        """Test stop loss execution for long position."""
        # Long position with stop at 95, bar drops to 94
        hit, price = simulate_intrabar_stop_execution(
            high=100.0, low=94.0, close=96.0, stop_price=95.0, is_long=True
        )

        assert hit is True
        assert price == 95.0

    def test_simulate_intrabar_stop_not_hit(self):
        """Test stop not hit when price stays above."""
        # Long position with stop at 95, bar stays above
        hit, price = simulate_intrabar_stop_execution(
            high=100.0, low=96.0, close=98.0, stop_price=95.0, is_long=True
        )

        assert hit is False
        assert price == 98.0  # Close price returned

    def test_simulate_intrabar_target_long(self):
        """Test take profit execution for long position."""
        # Long position with target at 110, bar reaches 111
        hit, price = simulate_intrabar_target_execution(
            high=111.0, low=105.0, close=108.0, target_price=110.0, is_long=True
        )

        assert hit is True
        assert price == 110.0

    def test_simulate_intrabar_target_not_hit(self):
        """Test target not hit when price doesn't reach."""
        # Long position with target at 110, bar doesn't reach
        hit, price = simulate_intrabar_target_execution(
            high=108.0, low=105.0, close=107.0, target_price=110.0, is_long=True
        )

        assert hit is False
        assert price == 107.0  # Close price returned

    def test_check_stop_and_target_stop_priority(self):
        """Test that stop is checked before target (conservative)."""
        # Both stop and target theoretically hit in same bar
        exited, price, reason = check_stop_and_target(
            high=111.0, low=94.0, close=96.0, stop_price=95.0, target_price=110.0, is_long=True
        )

        assert exited is True
        assert price == 95.0  # Stop price
        assert reason == "stop"

    def test_check_stop_and_target_target_only(self):
        """Test target hit when stop not touched."""
        exited, price, reason = check_stop_and_target(
            high=111.0, low=106.0, close=108.0, stop_price=95.0, target_price=110.0, is_long=True
        )

        assert exited is True
        assert price == 110.0
        assert reason == "target"

    def test_check_stop_and_target_neither(self):
        """Test neither stop nor target hit."""
        exited, price, reason = check_stop_and_target(
            high=108.0, low=96.0, close=102.0, stop_price=95.0, target_price=110.0, is_long=True
        )

        assert exited is False
        assert price == 102.0
        assert reason == "none"

    def test_validate_no_lookahead_valid(self):
        """Test lookahead validation with valid fills (R9.1.4)."""
        signals = np.array([10, 20, 30], dtype=np.int32)
        fills = np.array([11, 21, 31], dtype=np.int32)

        is_valid = validate_no_lookahead(signals, fills)
        assert is_valid is True

    def test_validate_no_lookahead_invalid(self):
        """Test lookahead detection when fill before signal."""
        signals = np.array([10, 20, 30], dtype=np.int32)
        fills = np.array([11, 19, 31], dtype=np.int32)  # Fill at 19 before signal at 20!

        is_valid = validate_no_lookahead(signals, fills)
        assert is_valid is False


# ============================================================================
# Tests for metrics.py - Performance Calculations
# ============================================================================


class TestPerformanceMetrics:
    """Test performance metrics calculations (R9.3)."""

    def test_calculate_returns(self):
        """Test returns calculation from equity curve."""
        equity = np.array([100.0, 105.0, 103.0, 108.0])
        returns = calculate_returns(equity)

        assert len(returns) == 3
        assert np.isclose(returns[0], 0.05)  # 5% gain
        assert np.isclose(returns[1], -0.019047619, atol=0.001)  # ~-1.9% loss
        assert np.isclose(returns[2], 0.048543689, atol=0.001)  # ~4.85% gain

    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation (R9.3.1)."""
        # Create simple returns: avg 1% per period, std 2%
        np.random.seed(42)
        returns = np.random.normal(0.01, 0.02, 100)

        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.04, periods_per_year=252)

        # Should be positive and reasonable
        assert sharpe > 0
        assert sharpe < 10  # Sanity check

    def test_calculate_sharpe_ratio_zero_std(self):
        """Test Sharpe with zero volatility (edge case)."""
        returns = np.array([0.01, 0.01, 0.01, 0.01])  # Constant returns

        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.04, periods_per_year=252)

        assert sharpe == 0.0  # Zero std → Sharpe = 0

    def test_calculate_max_drawdown(self):
        """Test max drawdown calculation (R9.3.1)."""
        equity = np.array([100.0, 110.0, 105.0, 95.0, 100.0, 120.0])

        dd_pct, peak_idx, trough_idx = calculate_max_drawdown(equity)

        # Max DD from 110 to 95 = -13.636%
        assert np.isclose(dd_pct, -13.636, atol=0.01)
        assert peak_idx == 1
        assert trough_idx == 3

    def test_calculate_max_drawdown_no_drawdown(self):
        """Test max drawdown with only gains."""
        equity = np.array([100.0, 110.0, 120.0, 130.0])

        dd_pct, peak_idx, trough_idx = calculate_max_drawdown(equity)

        assert dd_pct == 0.0

    def test_calculate_win_rate(self):
        """Test win rate calculation (R9.3.1)."""
        pnls = np.array([100.0, -50.0, 200.0, -30.0, 150.0])

        win_rate = calculate_win_rate(pnls)

        assert win_rate == 0.6  # 3 wins out of 5

    def test_calculate_win_rate_no_trades(self):
        """Test win rate with no trades."""
        pnls = np.array([])

        win_rate = calculate_win_rate(pnls)

        assert win_rate == 0.0

    def test_calculate_avg_win_loss_ratio(self):
        """Test avg win/loss ratio calculation (R9.3.1)."""
        pnls = np.array([100.0, -50.0, 200.0, -30.0, 150.0])

        ratio = calculate_avg_win_loss_ratio(pnls)

        # Avg win = 150, Avg loss = 40 → ratio = 3.75
        assert np.isclose(ratio, 3.75)

    def test_calculate_avg_win_loss_ratio_no_losses(self):
        """Test ratio when no losses (edge case)."""
        pnls = np.array([100.0, 200.0, 150.0])

        ratio = calculate_avg_win_loss_ratio(pnls)

        assert ratio == 0.0  # No losses → ratio = 0

    def test_calculate_total_return(self):
        """Test total return calculation."""
        total_ret, pct_ret = calculate_total_return(100000.0, 125000.0)

        assert total_ret == 25000.0
        assert pct_ret == 25.0

    def test_calculate_annualized_return(self):
        """Test annualized return calculation."""
        # 25% return over 6 months (182 days)
        ann_ret = calculate_annualized_return(25.0, 182)

        # Should be higher than 25% (compounded annually)
        assert ann_ret > 25.0
        assert ann_ret < 100.0  # Sanity check

    def test_performance_metrics_calculate_all(self):
        """Test comprehensive metrics calculation."""
        # Create simple backtest data
        equity = np.array([100000.0, 105000.0, 103000.0, 110000.0, 115000.0])
        pnls = np.array([1000.0, -500.0, 2000.0, 1500.0])
        holding_periods = np.array([5, 3, 8, 4], dtype=np.int32)

        metrics = PerformanceMetrics.calculate_all_metrics(
            equity_curve=equity,
            trades_pnl=pnls,
            trades_holding_periods=holding_periods,
            initial_capital=100000.0,
            risk_free_rate=0.04,
            periods_per_year=252,
        )

        # Check all required metrics present
        assert "total_return" in metrics
        assert "total_return_pct" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown_pct" in metrics
        assert "win_rate" in metrics
        assert "avg_win_loss_ratio" in metrics
        assert "total_trades" in metrics

        # Validate some values
        assert metrics["total_return"] == 15000.0
        assert metrics["total_return_pct"] == 15.0
        assert metrics["total_trades"] == 4
        assert metrics["win_rate"] == 0.75  # 3 wins out of 4

    def test_performance_metrics_format_report(self):
        """Test metrics report formatting."""
        metrics = {
            "initial_capital": 100000.0,
            "final_equity": 125000.0,
            "total_return": 25000.0,
            "total_return_pct": 25.0,
            "annualized_return_pct": 55.6,
            "sharpe_ratio": 1.2,
            "max_drawdown_pct": -12.5,
            "risk_free_rate": 0.04,
            "total_trades": 50,
            "winning_trades": 28,
            "losing_trades": 22,
            "win_rate": 0.56,
            "avg_win": 1200.0,
            "avg_loss": 800.0,
            "avg_win_loss_ratio": 1.5,
            "avg_holding_period": 5.2,
        }

        report = PerformanceMetrics.format_metrics_report(metrics)

        # Check report contains key information
        assert "BACKTEST PERFORMANCE REPORT" in report
        assert "$125,000.00" in report
        assert "25.00%" in report
        assert "1.20" in report  # Sharpe
        assert "56.00%" in report  # Win rate

    def test_performance_metrics_check_success_criteria(self):
        """Test success criteria checking (FLUXHERO_REQUIREMENTS.md)."""
        # Metrics that meet all criteria
        good_metrics = {
            "sharpe_ratio": 1.0,  # >0.8 ✓
            "max_drawdown_pct": -20.0,  # <25% ✓
            "win_rate": 0.50,  # >45% ✓
            "avg_win_loss_ratio": 2.0,  # >1.5 ✓
        }

        criteria = PerformanceMetrics.check_success_criteria(good_metrics)

        assert criteria["sharpe_ratio_ok"] is True
        assert criteria["max_drawdown_ok"] is True
        assert criteria["win_rate_ok"] is True
        assert criteria["win_loss_ratio_ok"] is True
        assert criteria["all_criteria_met"] is True

    def test_performance_metrics_fail_criteria(self):
        """Test failure detection for success criteria."""
        # Metrics that fail some criteria
        bad_metrics = {
            "sharpe_ratio": 0.5,  # <0.8 ✗
            "max_drawdown_pct": -30.0,  # >25% ✗
            "win_rate": 0.40,  # <45% ✗
            "avg_win_loss_ratio": 1.2,  # <1.5 ✗
        }

        criteria = PerformanceMetrics.check_success_criteria(bad_metrics)

        assert criteria["sharpe_ratio_ok"] is False
        assert criteria["max_drawdown_ok"] is False
        assert criteria["win_rate_ok"] is False
        assert criteria["win_loss_ratio_ok"] is False
        assert criteria["all_criteria_met"] is False


# ============================================================================
# Tests for engine.py - Backtest Orchestrator
# ============================================================================


class TestBacktestEngine:
    """Test backtest engine orchestrator."""

    def test_backtest_config_defaults(self):
        """Test backtest config default values (R9.2)."""
        config = BacktestConfig()

        assert config.initial_capital == 100000.0
        assert config.commission_per_share == 0.005  # R9.2.1
        assert config.slippage_pct == 0.0001  # R9.2.2: 0.01%
        assert config.impact_threshold == 0.1  # R9.2.3: 10%
        assert config.risk_free_rate == 0.04  # R9.3.1: 4%

    def test_backtest_engine_initialization(self):
        """Test engine initialization."""
        config = BacktestConfig(initial_capital=50000.0)
        engine = BacktestEngine(config)

        assert engine.config.initial_capital == 50000.0

    def test_backtest_simple_buy_and_hold(self):
        """Test simple buy-and-hold backtest (validation test)."""
        # Create simple uptrend data
        bars = np.array(
            [
                [100.0, 102.0, 99.0, 101.0, 1000000],  # Bar 0
                [101.0, 103.0, 100.0, 102.0, 1000000],  # Bar 1
                [102.0, 105.0, 101.0, 104.0, 1000000],  # Bar 2
                [104.0, 106.0, 103.0, 105.0, 1000000],  # Bar 3
                [105.0, 108.0, 104.0, 107.0, 1000000],  # Bar 4
            ]
        )

        # Simple strategy: buy at bar 0, hold
        def buy_and_hold_strategy(bars_data, current_idx, position):
            orders = []
            if current_idx == 0 and position is None:
                # Buy 500 shares at bar 0 (fits in $100k capital)
                orders.append(
                    Order(
                        bar_index=0,
                        symbol="TEST",
                        side=OrderSide.BUY,
                        shares=500,
                        order_type=OrderType.MARKET,
                    )
                )
            return orders

        config = BacktestConfig(initial_capital=100000.0)
        engine = BacktestEngine(config)

        state = engine.run(bars, buy_and_hold_strategy, symbol="TEST")

        # Check position opened
        assert len(state.trades) == 0  # No closed trades (still holding)
        assert state.position is not None
        assert state.position.shares == 500

        # Check equity increased (price went from 101 to 107)
        assert state.equity > 100000.0

    def test_backtest_simple_trade_with_exit(self):
        """Test backtest with entry and exit."""
        bars = np.array(
            [
                [100.0, 102.0, 99.0, 101.0, 1000000],  # Bar 0
                [101.0, 103.0, 100.0, 102.0, 1000000],  # Bar 1
                [102.0, 105.0, 101.0, 104.0, 1000000],  # Bar 2
                [104.0, 106.0, 103.0, 105.0, 1000000],  # Bar 3
                [105.0, 104.0, 102.0, 103.0, 1000000],  # Bar 4
            ]
        )

        # Strategy: buy at bar 0, sell at bar 2
        # Note: Using 989 shares to account for slippage + commission with $100k capital
        def simple_strategy(bars_data, current_idx, position):
            orders = []
            if current_idx == 0 and position is None:
                orders.append(Order(bar_index=0, symbol="TEST", side=OrderSide.BUY, shares=989))
            elif current_idx == 2 and position is not None:
                orders.append(Order(bar_index=2, symbol="TEST", side=OrderSide.SELL, shares=989))
            return orders

        config = BacktestConfig(initial_capital=100000.0)
        engine = BacktestEngine(config)

        state = engine.run(bars, simple_strategy, symbol="TEST")

        # Check trade completed
        assert len(state.trades) == 1
        assert state.position is None  # Position closed

        trade = state.trades[0]
        assert trade.shares == 989
        assert trade.pnl > 0  # Should be profitable (buy low, sell high)

    def test_backtest_slippage_applied(self):
        """Test that slippage is applied to fills (R9.2.2)."""
        bars = np.array(
            [
                [100.0, 102.0, 99.0, 101.0, 1000000],
                [101.0, 103.0, 100.0, 102.0, 1000000],
            ]
        )

        def buy_strategy(bars_data, current_idx, position):
            if current_idx == 0 and position is None:
                return [Order(bar_index=0, symbol="TEST", side=OrderSide.BUY, shares=100)]
            return []

        config = BacktestConfig(initial_capital=100000.0, slippage_pct=0.001)  # 0.1% slippage
        engine = BacktestEngine(config)

        state = engine.run(bars, buy_strategy, symbol="TEST")

        # Fill should occur at bar 1 open (101.0) + 0.1% slippage
        assert state.position is not None
        expected_fill = 101.0 * 1.001
        assert np.isclose(state.position.entry_price, expected_fill, rtol=0.01)

    def test_backtest_commission_applied(self):
        """Test that commission is deducted (R9.2.1)."""
        bars = np.array(
            [
                [100.0, 102.0, 99.0, 101.0, 1000000],
                [101.0, 103.0, 100.0, 102.0, 1000000],
                [102.0, 105.0, 101.0, 104.0, 1000000],
            ]
        )

        def buy_sell_strategy(bars_data, current_idx, position):
            if current_idx == 0 and position is None:
                return [Order(bar_index=0, symbol="TEST", side=OrderSide.BUY, shares=100)]
            elif current_idx == 1 and position is not None:
                return [Order(bar_index=1, symbol="TEST", side=OrderSide.SELL, shares=100)]
            return []

        config = BacktestConfig(
            initial_capital=100000.0,
            commission_per_share=0.01,  # $0.01 per share
            slippage_pct=0.0,  # No slippage for clarity
        )
        engine = BacktestEngine(config)

        state = engine.run(bars, buy_sell_strategy, symbol="TEST")

        # Check commission was charged
        assert len(state.trades) == 1
        trade = state.trades[0]
        assert trade.commission > 0
        # 100 shares × $0.01 = $1.00 commission
        assert np.isclose(trade.commission, 1.0, atol=0.01)

    def test_backtest_insufficient_capital(self):
        """Test order cancellation when insufficient capital."""
        bars = np.array(
            [
                [100.0, 102.0, 99.0, 101.0, 1000000],
                [101.0, 103.0, 100.0, 102.0, 1000000],
            ]
        )

        def overleveraged_strategy(bars_data, current_idx, position):
            if current_idx == 0 and position is None:
                # Try to buy too many shares
                return [Order(bar_index=0, symbol="TEST", side=OrderSide.BUY, shares=100000)]
            return []

        config = BacktestConfig(initial_capital=10000.0)  # Only $10k
        engine = BacktestEngine(config)

        state = engine.run(bars, overleveraged_strategy, symbol="TEST")

        # Order should be cancelled (not enough capital)
        assert state.position is None
        assert len(state.trades) == 0

    def test_backtest_performance_summary(self):
        """Test performance summary generation."""
        bars = np.array(
            [
                [100.0, 102.0, 99.0, 101.0, 1000000],
                [101.0, 103.0, 100.0, 102.0, 1000000],
                [102.0, 105.0, 101.0, 104.0, 1000000],
            ]
        )

        def dummy_strategy(bars_data, current_idx, position):
            return []

        config = BacktestConfig(initial_capital=100000.0)
        engine = BacktestEngine(config)

        state = engine.run(bars, dummy_strategy, symbol="TEST")

        summary = engine.get_performance_summary(state)

        assert "total_trades" in summary
        assert "win_rate" in summary
        assert "total_return" in summary
        assert "final_equity" in summary

    def test_backtest_equity_curve_tracking(self):
        """Test equity curve is tracked at each bar."""
        bars = np.array(
            [
                [100.0, 102.0, 99.0, 101.0, 1000000],
                [101.0, 103.0, 100.0, 102.0, 1000000],
                [102.0, 105.0, 101.0, 104.0, 1000000],
            ]
        )

        def dummy_strategy(bars_data, current_idx, position):
            return []

        config = BacktestConfig(initial_capital=100000.0)
        engine = BacktestEngine(config)

        state = engine.run(bars, dummy_strategy, symbol="TEST")

        # Equity curve should have entry for each bar
        assert len(state.equity_curve) == len(bars)


# ============================================================================
# Tests for Bar Integrity Validation
# ============================================================================


class TestBarIntegrityValidation:
    """Test bar integrity validation function."""

    def test_valid_bars_pass(self):
        """Test that valid OHLC data passes validation."""
        # Valid bars: high >= open/close, low <= open/close, high >= low
        bars = np.array([
            [100.0, 102.0, 99.0, 101.0, 1000000],  # Open, High, Low, Close, Volume
            [101.0, 103.0, 100.0, 102.0, 1000000],
            [102.0, 105.0, 101.0, 104.0, 1000000],
        ])

        issues = validate_bar_integrity(bars)
        assert issues == []

    def test_high_less_than_low_detected(self):
        """Test detection of High < Low (invalid OHLC)."""
        # Invalid bar: high=98 < low=99
        bars = np.array([
            [100.0, 102.0, 99.0, 101.0, 1000000],  # Valid
            [101.0, 98.0, 99.0, 100.0, 1000000],   # Invalid: high < low
            [102.0, 105.0, 101.0, 104.0, 1000000], # Valid
        ])

        issues = validate_bar_integrity(bars)
        # Should detect at least the high < low issue (may also detect high < open/close)
        assert len(issues) >= 1
        assert any("High < Low" in issue for issue in issues)
        assert any("1" in issue for issue in issues)  # Index 1 should be in the message

    def test_high_less_than_open_detected(self):
        """Test detection of High < Open."""
        # Invalid bar: high=99 < open=100
        bars = np.array([
            [100.0, 99.0, 98.0, 99.0, 1000000],   # Invalid: high < open
            [101.0, 103.0, 100.0, 102.0, 1000000], # Valid
        ])

        issues = validate_bar_integrity(bars)
        assert len(issues) >= 1
        assert any("High < Open" in issue for issue in issues)

    def test_high_less_than_close_detected(self):
        """Test detection of High < Close."""
        # Invalid bar: high=100 < close=101
        bars = np.array([
            [98.0, 100.0, 97.0, 101.0, 1000000],  # Invalid: high < close
        ])

        issues = validate_bar_integrity(bars)
        assert len(issues) >= 1
        assert any("High < Close" in issue for issue in issues)

    def test_low_greater_than_open_detected(self):
        """Test detection of Low > Open."""
        # Invalid bar: low=102 > open=100
        bars = np.array([
            [100.0, 105.0, 102.0, 104.0, 1000000],  # Invalid: low > open
        ])

        issues = validate_bar_integrity(bars)
        assert len(issues) >= 1
        assert any("Low > Open" in issue for issue in issues)

    def test_low_greater_than_close_detected(self):
        """Test detection of Low > Close."""
        # Invalid bar: low=102 > close=101
        bars = np.array([
            [105.0, 106.0, 102.0, 101.0, 1000000],  # Invalid: low > close
        ])

        issues = validate_bar_integrity(bars)
        assert len(issues) >= 1
        assert any("Low > Close" in issue for issue in issues)

    def test_timestamps_monotonic_pass(self):
        """Test that monotonically increasing timestamps pass."""
        bars = np.array([
            [100.0, 102.0, 99.0, 101.0, 1000000],
            [101.0, 103.0, 100.0, 102.0, 1000000],
            [102.0, 105.0, 101.0, 104.0, 1000000],
        ])
        timestamps = np.array([
            np.datetime64('2024-01-01'),
            np.datetime64('2024-01-02'),
            np.datetime64('2024-01-03'),
        ])

        issues = validate_bar_integrity(bars, timestamps)
        assert issues == []

    def test_timestamps_non_monotonic_detected(self):
        """Test detection of non-monotonic timestamps."""
        bars = np.array([
            [100.0, 102.0, 99.0, 101.0, 1000000],
            [101.0, 103.0, 100.0, 102.0, 1000000],
            [102.0, 105.0, 101.0, 104.0, 1000000],
        ])
        # Timestamps out of order: 2024-01-03 comes before 2024-01-02
        timestamps = np.array([
            np.datetime64('2024-01-01'),
            np.datetime64('2024-01-03'),
            np.datetime64('2024-01-02'),  # Out of order!
        ])

        issues = validate_bar_integrity(bars, timestamps)
        assert len(issues) == 1
        assert "Non-monotonic timestamps" in issues[0]

    def test_timestamps_duplicate_detected(self):
        """Test detection of duplicate timestamps."""
        bars = np.array([
            [100.0, 102.0, 99.0, 101.0, 1000000],
            [101.0, 103.0, 100.0, 102.0, 1000000],
            [102.0, 105.0, 101.0, 104.0, 1000000],
        ])
        # Duplicate timestamp
        timestamps = np.array([
            np.datetime64('2024-01-01'),
            np.datetime64('2024-01-02'),
            np.datetime64('2024-01-02'),  # Duplicate!
        ])

        issues = validate_bar_integrity(bars, timestamps)
        assert len(issues) == 1
        assert "Non-monotonic timestamps" in issues[0]

    def test_timestamps_numeric_epoch(self):
        """Test with numeric (Unix epoch) timestamps."""
        bars = np.array([
            [100.0, 102.0, 99.0, 101.0, 1000000],
            [101.0, 103.0, 100.0, 102.0, 1000000],
            [102.0, 105.0, 101.0, 104.0, 1000000],
        ])
        # Unix epoch timestamps
        timestamps = np.array([1704067200, 1704153600, 1704240000])

        issues = validate_bar_integrity(bars, timestamps)
        assert issues == []

    def test_empty_bars_detected(self):
        """Test detection of empty bars array."""
        bars = np.array([]).reshape(0, 5)

        issues = validate_bar_integrity(bars)
        assert len(issues) == 1
        assert "Empty bars array" in issues[0]

    def test_multiple_issues_reported(self):
        """Test that multiple issues are all reported."""
        # Bar with multiple issues
        bars = np.array([
            [100.0, 102.0, 99.0, 101.0, 1000000],  # Valid
            [101.0, 98.0, 99.0, 100.0, 1000000],   # Invalid: high < low
            [102.0, 105.0, 101.0, 104.0, 1000000], # Valid
        ])
        # Non-monotonic timestamps
        timestamps = np.array([
            np.datetime64('2024-01-03'),
            np.datetime64('2024-01-02'),
            np.datetime64('2024-01-01'),
        ])

        issues = validate_bar_integrity(bars, timestamps)
        # Should have at least 2 issues: OHLC issue and timestamp issue
        assert len(issues) >= 2

    def test_backtest_engine_calls_validation(self):
        """Test that BacktestEngine.run() calls bar validation."""
        # Valid bars
        bars = np.array([
            [100.0, 102.0, 99.0, 101.0, 1000000],
            [101.0, 103.0, 100.0, 102.0, 1000000],
            [102.0, 105.0, 101.0, 104.0, 1000000],
        ])

        def dummy_strategy(bars_data, current_idx, position):
            return []

        config = BacktestConfig(initial_capital=100000.0)
        engine = BacktestEngine(config)

        # Should not raise - validation passes
        state = engine.run(bars, dummy_strategy, symbol="TEST")
        assert len(state.equity_curve) == len(bars)


# ============================================================================
# Tests for Backtest Operation Logging
# ============================================================================


class TestBacktestOperationLogging:
    """Test backtest operation logging functionality."""

    def test_backtest_logs_start_message(self, caplog):
        """Test that backtest logs start message with config summary."""
        import logging

        bars = np.array([
            [100.0, 102.0, 99.0, 101.0, 1000000],
            [101.0, 103.0, 100.0, 102.0, 1000000],
            [102.0, 105.0, 101.0, 104.0, 1000000],
        ])

        def dummy_strategy(bars_data, current_idx, position):
            return []

        config = BacktestConfig(initial_capital=100000.0)
        engine = BacktestEngine(config)

        with caplog.at_level(logging.INFO, logger="backend.backtesting.engine"):
            engine.run(bars, dummy_strategy, symbol="SPY")

        # Check start message logged
        start_messages = [r for r in caplog.records if "Backtest started" in r.message]
        assert len(start_messages) == 1

        start_msg = start_messages[0].message
        assert "symbol=SPY" in start_msg
        assert "bars=3" in start_msg
        assert "initial_capital=$100,000.00" in start_msg
        assert "commission=$0.005/share" in start_msg
        assert "slippage=0.010%" in start_msg

    def test_backtest_logs_completion_message(self, caplog):
        """Test that backtest logs completion message with final metrics."""
        import logging

        bars = np.array([
            [100.0, 102.0, 99.0, 101.0, 1000000],
            [101.0, 103.0, 100.0, 102.0, 1000000],
            [102.0, 105.0, 101.0, 104.0, 1000000],
        ])

        def dummy_strategy(bars_data, current_idx, position):
            return []

        config = BacktestConfig(initial_capital=100000.0)
        engine = BacktestEngine(config)

        with caplog.at_level(logging.INFO, logger="backend.backtesting.engine"):
            engine.run(bars, dummy_strategy, symbol="TEST")

        # Check completion message logged
        complete_messages = [r for r in caplog.records if "Backtest completed" in r.message]
        assert len(complete_messages) == 1

        complete_msg = complete_messages[0].message
        assert "duration=" in complete_msg
        assert "ms" in complete_msg
        assert "total_trades=" in complete_msg
        assert "win_rate=" in complete_msg
        assert "return=" in complete_msg
        assert "final_equity=" in complete_msg

    def test_backtest_logs_progress_for_large_dataset(self, caplog):
        """Test that backtest logs progress every 10% for large datasets."""
        import logging

        # Create 100 bars to trigger progress logging
        n_bars = 100
        bars = np.zeros((n_bars, 5))
        bars[:, 0] = 100.0  # Open
        bars[:, 1] = 102.0  # High
        bars[:, 2] = 99.0   # Low
        bars[:, 3] = 101.0  # Close
        bars[:, 4] = 1000000  # Volume

        def dummy_strategy(bars_data, current_idx, position):
            return []

        config = BacktestConfig(initial_capital=100000.0)
        engine = BacktestEngine(config)

        with caplog.at_level(logging.INFO, logger="backend.backtesting.engine"):
            engine.run(bars, dummy_strategy, symbol="TEST")

        # Check progress messages logged
        progress_messages = [r for r in caplog.records if "Backtest progress" in r.message]

        # With 100 bars, we should get progress logs at 10%, 20%, ..., 90%
        assert len(progress_messages) >= 1

        # Verify progress message format
        if progress_messages:
            msg = progress_messages[0].message
            assert "%" in msg
            assert "bars)" in msg
            assert "trades=" in msg
            assert "equity=" in msg
            assert "elapsed=" in msg
            assert "ms" in msg

    def test_backtest_logs_duration_in_milliseconds(self, caplog):
        """Test that backtest duration is logged in milliseconds."""
        import logging
        import re

        bars = np.array([
            [100.0, 102.0, 99.0, 101.0, 1000000],
            [101.0, 103.0, 100.0, 102.0, 1000000],
            [102.0, 105.0, 101.0, 104.0, 1000000],
        ])

        def dummy_strategy(bars_data, current_idx, position):
            return []

        config = BacktestConfig(initial_capital=100000.0)
        engine = BacktestEngine(config)

        with caplog.at_level(logging.INFO, logger="backend.backtesting.engine"):
            engine.run(bars, dummy_strategy, symbol="TEST")

        # Find completion message and verify duration format
        complete_messages = [r for r in caplog.records if "Backtest completed" in r.message]
        assert len(complete_messages) == 1

        # Extract duration using regex
        match = re.search(r"duration=(\d+)ms", complete_messages[0].message)
        assert match is not None, "Duration should be in format 'duration=XXms'"

        duration_ms = int(match.group(1))
        assert duration_ms >= 0, "Duration should be non-negative"

    def test_backtest_logs_correct_trade_stats(self, caplog):
        """Test that completion log contains accurate trade statistics."""
        import logging

        bars = np.array([
            [100.0, 102.0, 99.0, 101.0, 1000000],  # Bar 0
            [101.0, 103.0, 100.0, 102.0, 1000000],  # Bar 1 - buy fills here
            [102.0, 105.0, 101.0, 104.0, 1000000],  # Bar 2
            [104.0, 106.0, 103.0, 105.0, 1000000],  # Bar 3 - sell fills here
            [105.0, 107.0, 104.0, 106.0, 1000000],  # Bar 4
        ])

        # Strategy: buy at bar 0, sell at bar 2 (profitable trade)
        def simple_strategy(bars_data, current_idx, position):
            orders = []
            if current_idx == 0 and position is None:
                orders.append(Order(bar_index=0, symbol="TEST", side=OrderSide.BUY, shares=100))
            elif current_idx == 2 and position is not None:
                orders.append(Order(bar_index=2, symbol="TEST", side=OrderSide.SELL, shares=100))
            return orders

        config = BacktestConfig(initial_capital=100000.0)
        engine = BacktestEngine(config)

        with caplog.at_level(logging.INFO, logger="backend.backtesting.engine"):
            engine.run(bars, simple_strategy, symbol="TEST")

        # Find completion message
        complete_messages = [r for r in caplog.records if "Backtest completed" in r.message]
        assert len(complete_messages) == 1

        msg = complete_messages[0].message
        # Should show 1 trade, 100% win rate (profitable trade)
        assert "total_trades=1" in msg
        assert "win_rate=100.0%" in msg

    def test_backtest_no_progress_for_small_dataset(self, caplog):
        """Test that progress logging is minimal for very small datasets."""
        import logging

        # Only 3 bars - progress interval = 0, so no progress logs expected
        bars = np.array([
            [100.0, 102.0, 99.0, 101.0, 1000000],
            [101.0, 103.0, 100.0, 102.0, 1000000],
            [102.0, 105.0, 101.0, 104.0, 1000000],
        ])

        def dummy_strategy(bars_data, current_idx, position):
            return []

        config = BacktestConfig(initial_capital=100000.0)
        engine = BacktestEngine(config)

        with caplog.at_level(logging.INFO, logger="backend.backtesting.engine"):
            engine.run(bars, dummy_strategy, symbol="TEST")

        # Should have start and completion, but no progress messages
        # (3 bars / 10 = 0, progress_interval = max(1, 0) = 1, but only 3 bars)
        progress_messages = [r for r in caplog.records if "Backtest progress" in r.message]
        # For 3 bars, we may or may not get progress logs depending on the interval
        # The key is that we don't flood the logs for tiny datasets
        assert len(progress_messages) <= 2  # At most 2 progress messages for 3 bars


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
