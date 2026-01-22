"""
Unit tests for backend/risk/kill_switch.py.

Tests drawdown circuit breakers, equity tracking, and risk monitoring.

Requirements tested:
- R11.3.1: Drawdown tracking from equity peak
- R11.3.2: 15% drawdown actions (reduce sizes, tighten stops, alert)
- R11.3.3: 20% drawdown actions (close all, disable trading)
- R11.4.1: Real-time risk monitoring
- R11.4.2: Daily risk report generation

Author: FluxHero
Date: 2026-01-20
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from backend.risk.kill_switch import (  # noqa: E402
    DrawdownCircuitBreaker,
    DrawdownCircuitBreakerConfig,
    DrawdownLevel,
    EquityTracker,
    Position,
    TradingStatus,
    calculate_correlation_matrix,
    calculate_risk_metrics,
    format_daily_risk_report,
    generate_daily_risk_report,
)

# ============================================================================
# EquityTracker Tests (R11.3.1)
# ============================================================================

def test_equity_tracker_initialization():
    """Test EquityTracker initialization."""
    tracker = EquityTracker(100000.0)

    assert tracker.equity_peak == 100000.0
    assert tracker.current_equity == 100000.0
    assert tracker.calculate_drawdown() == 0.0
    assert tracker.calculate_drawdown_pct() == 0.0


def test_equity_tracker_update_with_profit():
    """Test equity update when making profit."""
    tracker = EquityTracker(100000.0)

    # Profit: equity increases
    dd_pct = tracker.update_equity(105000.0)

    assert tracker.current_equity == 105000.0
    assert tracker.equity_peak == 105000.0  # Peak updated
    assert dd_pct == 0.0  # No drawdown


def test_equity_tracker_update_with_loss():
    """Test equity update when taking loss."""
    tracker = EquityTracker(100000.0)

    # Loss: equity decreases
    dd_pct = tracker.update_equity(95000.0)

    assert tracker.current_equity == 95000.0
    assert tracker.equity_peak == 100000.0  # Peak unchanged
    assert dd_pct == 0.05  # 5% drawdown


def test_equity_tracker_drawdown_calculation():
    """Test drawdown calculation accuracy."""
    tracker = EquityTracker(100000.0)

    # Lose 15%
    tracker.update_equity(85000.0)

    assert tracker.calculate_drawdown() == 15000.0
    assert tracker.calculate_drawdown_pct() == 0.15


def test_equity_tracker_peak_recovery():
    """Test peak update after recovery from drawdown."""
    tracker = EquityTracker(100000.0)

    # Drawdown
    tracker.update_equity(85000.0)
    assert tracker.equity_peak == 100000.0

    # Recovery to new peak
    tracker.update_equity(110000.0)
    assert tracker.equity_peak == 110000.0
    assert tracker.calculate_drawdown_pct() == 0.0


def test_equity_tracker_reset_peak():
    """Test manual peak reset."""
    tracker = EquityTracker(100000.0)

    tracker.reset_peak(120000.0)

    assert tracker.equity_peak == 120000.0


def test_equity_tracker_history():
    """Test equity history tracking."""
    tracker = EquityTracker(100000.0)

    ts1 = datetime(2026, 1, 1, 10, 0, 0)
    ts2 = datetime(2026, 1, 1, 11, 0, 0)

    tracker.update_equity(105000.0, ts1)
    tracker.update_equity(102000.0, ts2)

    assert len(tracker.equity_history) == 2
    assert tracker.equity_history[0] == (ts1, 105000.0)
    assert tracker.equity_history[1] == (ts2, 102000.0)


# ============================================================================
# DrawdownCircuitBreaker Tests (R11.3.2, R11.3.3)
# ============================================================================

def test_circuit_breaker_initialization():
    """Test circuit breaker initialization."""
    breaker = DrawdownCircuitBreaker()

    assert breaker.trading_status == TradingStatus.ACTIVE
    assert breaker.manual_review_required is False
    assert len(breaker.alerts) == 0


def test_circuit_breaker_normal_drawdown():
    """Test normal operation below 15% drawdown."""
    breaker = DrawdownCircuitBreaker()

    level = breaker.check_drawdown_level(0.10)  # 10% drawdown

    assert level == DrawdownLevel.NORMAL


def test_circuit_breaker_warning_drawdown():
    """Test warning level at 15% drawdown (R11.3.2)."""
    breaker = DrawdownCircuitBreaker()

    level = breaker.check_drawdown_level(0.16)  # 16% drawdown

    assert level == DrawdownLevel.WARNING


def test_circuit_breaker_critical_drawdown():
    """Test critical level at 20% drawdown (R11.3.3)."""
    breaker = DrawdownCircuitBreaker()

    level = breaker.check_drawdown_level(0.21)  # 21% drawdown

    assert level == DrawdownLevel.CRITICAL


def test_circuit_breaker_warning_actions():
    """Test actions at 15% drawdown (R11.3.2)."""
    breaker = DrawdownCircuitBreaker()

    # Trigger warning
    status, alerts = breaker.update_trading_status(0.16)

    assert status == TradingStatus.REDUCED
    assert len(alerts) == 1
    assert "WARNING" in alerts[0]
    assert "15%" in alerts[0]


def test_circuit_breaker_critical_actions():
    """Test actions at 20% drawdown (R11.3.3)."""
    breaker = DrawdownCircuitBreaker()

    # Trigger critical
    status, alerts = breaker.update_trading_status(0.21)

    assert status == TradingStatus.DISABLED
    assert breaker.manual_review_required is True
    assert len(alerts) == 1
    assert "CRITICAL" in alerts[0]
    assert "20%" in alerts[0]


def test_circuit_breaker_position_size_multiplier():
    """Test position size reduction at warning level (R11.3.2)."""
    breaker = DrawdownCircuitBreaker()

    # Normal
    breaker.trading_status = TradingStatus.ACTIVE
    assert breaker.get_position_size_multiplier() == 1.0

    # Warning: 50% reduction
    breaker.trading_status = TradingStatus.REDUCED
    assert breaker.get_position_size_multiplier() == 0.5

    # Critical: no trading
    breaker.trading_status = TradingStatus.DISABLED
    assert breaker.get_position_size_multiplier() == 0.0


def test_circuit_breaker_stop_loss_multiplier():
    """Test stop loss tightening at warning level (R11.3.2)."""
    breaker = DrawdownCircuitBreaker()

    # Normal: 2.5× ATR
    breaker.trading_status = TradingStatus.ACTIVE
    assert breaker.get_stop_loss_multiplier() == 2.5

    # Warning: tighten to 2.0× ATR
    breaker.trading_status = TradingStatus.REDUCED
    assert breaker.get_stop_loss_multiplier() == 2.0


def test_circuit_breaker_can_open_position():
    """Test position opening permissions."""
    breaker = DrawdownCircuitBreaker()

    # Active: can trade
    breaker.trading_status = TradingStatus.ACTIVE
    can_trade, reason = breaker.can_open_new_position()
    assert can_trade is True

    # Reduced: can trade with reduced size
    breaker.trading_status = TradingStatus.REDUCED
    can_trade, reason = breaker.can_open_new_position()
    assert can_trade is True

    # Disabled: cannot trade
    breaker.trading_status = TradingStatus.DISABLED
    can_trade, reason = breaker.can_open_new_position()
    assert can_trade is False
    assert "disabled" in reason.lower()


def test_circuit_breaker_should_close_all():
    """Test close all positions at critical level (R11.3.3)."""
    breaker = DrawdownCircuitBreaker()

    # Normal/Warning: don't close
    breaker.trading_status = TradingStatus.ACTIVE
    assert breaker.should_close_all_positions() is False

    breaker.trading_status = TradingStatus.REDUCED
    assert breaker.should_close_all_positions() is False

    # Critical: close all
    breaker.trading_status = TradingStatus.DISABLED
    assert breaker.should_close_all_positions() is True


def test_circuit_breaker_manual_review():
    """Test manual review requirement (R11.3.3)."""
    breaker = DrawdownCircuitBreaker()

    # Trigger critical
    breaker.update_trading_status(0.21)
    assert breaker.manual_review_required is True

    # Cannot trade until review
    can_trade, _ = breaker.can_open_new_position()
    assert can_trade is False

    # Acknowledge review
    breaker.acknowledge_manual_review()
    assert breaker.manual_review_required is False
    assert breaker.trading_status == TradingStatus.ACTIVE


def test_circuit_breaker_recovery_from_warning():
    """Test recovery from warning to normal."""
    breaker = DrawdownCircuitBreaker()

    # Trigger warning
    breaker.update_trading_status(0.16)
    assert breaker.trading_status == TradingStatus.REDUCED

    # Recover to normal
    status, alerts = breaker.update_trading_status(0.10)
    assert status == TradingStatus.ACTIVE
    assert len(alerts) == 1
    assert "recovered" in alerts[0].lower()


def test_circuit_breaker_alerts_tracking():
    """Test alert tracking."""
    breaker = DrawdownCircuitBreaker()

    # Generate multiple alerts
    breaker.update_trading_status(0.16)  # Warning
    breaker.update_trading_status(0.21)  # Critical

    alerts = breaker.get_recent_alerts(10)
    assert len(alerts) == 2
    assert all(isinstance(alert, tuple) for alert in alerts)
    assert all(len(alert) == 2 for alert in alerts)


# ============================================================================
# Risk Metrics Tests (R11.4.1)
# ============================================================================

def test_calculate_risk_metrics_no_positions():
    """Test risk metrics with no open positions."""
    metrics = calculate_risk_metrics(
        account_balance=100000.0,
        equity_peak=105000.0,
        current_equity=100000.0,
        open_positions=[]
    )

    assert metrics.equity_peak == 105000.0
    assert metrics.current_equity == 100000.0
    assert metrics.current_drawdown == 5000.0
    assert abs(metrics.current_drawdown_pct - 0.047619) < 0.0001  # ~4.76%
    assert metrics.drawdown_level == DrawdownLevel.NORMAL
    assert metrics.num_positions == 0
    assert metrics.total_exposure == 0.0
    assert metrics.total_risk_deployed == 0.0


def test_calculate_risk_metrics_with_positions():
    """Test risk metrics with open positions."""
    positions = [
        Position("SPY", 100, 400.0, 410.0, 395.0),  # Risk: 100 × 5 = 500
        Position("QQQ", 50, 300.0, 305.0, 291.0),   # Risk: 50 × 9 = 450
    ]

    metrics = calculate_risk_metrics(
        account_balance=100000.0,
        equity_peak=100000.0,
        current_equity=100000.0,
        open_positions=positions
    )

    assert metrics.num_positions == 2
    assert metrics.total_exposure == 100*410 + 50*305  # 56250
    assert metrics.exposure_pct == 0.5625  # 56.25%
    assert metrics.total_risk_deployed == 950.0  # 500 + 450
    assert metrics.worst_case_loss == 950.0
    assert metrics.largest_position_symbol == "SPY"
    assert metrics.largest_position_value == 41000.0


def test_calculate_risk_metrics_warning_level():
    """Test risk metrics at warning drawdown level."""
    metrics = calculate_risk_metrics(
        account_balance=85000.0,
        equity_peak=100000.0,
        current_equity=85000.0,
        open_positions=[]
    )

    assert metrics.current_drawdown_pct == 0.15
    assert metrics.drawdown_level == DrawdownLevel.WARNING


def test_calculate_risk_metrics_critical_level():
    """Test risk metrics at critical drawdown level."""
    metrics = calculate_risk_metrics(
        account_balance=80000.0,
        equity_peak=100000.0,
        current_equity=80000.0,
        open_positions=[]
    )

    assert metrics.current_drawdown_pct == 0.20
    assert metrics.drawdown_level == DrawdownLevel.CRITICAL


# ============================================================================
# Correlation Matrix Tests (R11.4.1)
# ============================================================================

def test_correlation_matrix_perfect_positive():
    """Test correlation matrix with perfectly correlated assets."""
    prices = {
        "SPY": np.array([100.0, 101.0, 102.0, 103.0, 104.0]),
        "QQQ": np.array([200.0, 202.0, 204.0, 206.0, 208.0]),
    }

    matrix = calculate_correlation_matrix(prices)

    assert matrix[("SPY", "SPY")] == 1.0
    assert matrix[("QQQ", "QQQ")] == 1.0
    assert abs(matrix[("SPY", "QQQ")] - 1.0) < 0.0001  # Perfect correlation
    assert abs(matrix[("QQQ", "SPY")] - 1.0) < 0.0001  # Symmetric


def test_correlation_matrix_negative():
    """Test correlation matrix with negatively correlated assets."""
    prices = {
        "STOCK": np.array([100.0, 101.0, 102.0, 103.0, 104.0]),
        "INVERSE": np.array([200.0, 198.0, 196.0, 194.0, 192.0]),
    }

    matrix = calculate_correlation_matrix(prices)

    assert matrix[("STOCK", "INVERSE")] < 0  # Negative correlation
    assert matrix[("INVERSE", "STOCK")] < 0


def test_correlation_matrix_independent():
    """Test correlation matrix with independent assets."""
    np.random.seed(42)
    prices = {
        "ASSET1": np.random.randn(100) + 100,
        "ASSET2": np.random.randn(100) + 200,
    }

    matrix = calculate_correlation_matrix(prices)

    # Should be close to 0 for independent series
    assert abs(matrix[("ASSET1", "ASSET2")]) < 0.2


def test_correlation_matrix_insufficient_data():
    """Test correlation matrix with insufficient data."""
    prices = {
        "SPY": np.array([100.0]),  # Only 1 point
        "QQQ": np.array([200.0]),
    }

    matrix = calculate_correlation_matrix(prices)

    assert matrix[("SPY", "QQQ")] == 0.0  # Default to 0


# ============================================================================
# Daily Risk Report Tests (R11.4.2)
# ============================================================================

def test_generate_daily_risk_report():
    """Test daily risk report generation (R11.4.2)."""
    tracker = EquityTracker(100000.0)
    tracker.update_equity(95000.0)

    breaker = DrawdownCircuitBreaker()
    breaker.update_trading_status(0.05)

    positions = [Position("SPY", 100, 400.0, 410.0, 395.0)]

    report = generate_daily_risk_report(
        account_balance=95000.0,
        equity_tracker=tracker,
        open_positions=positions,
        circuit_breaker=breaker
    )

    assert report.account_balance == 95000.0
    assert report.equity_peak == 100000.0
    assert report.current_drawdown_pct == 0.05
    assert report.num_positions == 1
    assert report.total_exposure == 41000.0
    assert report.trading_status == TradingStatus.ACTIVE
    assert report.largest_position == "SPY"


def test_format_daily_risk_report():
    """Test report formatting."""
    tracker = EquityTracker(100000.0)
    breaker = DrawdownCircuitBreaker()

    report = generate_daily_risk_report(
        account_balance=100000.0,
        equity_tracker=tracker,
        open_positions=[],
        circuit_breaker=breaker
    )

    formatted = format_daily_risk_report(report)

    assert "Daily Risk Report" in formatted
    assert "Account Status" in formatted
    assert "Portfolio Exposure" in formatted
    assert "Risk Metrics" in formatted
    assert "$100,000.00" in formatted


def test_daily_risk_report_with_alerts():
    """Test report with alerts included."""
    tracker = EquityTracker(100000.0)
    tracker.update_equity(85000.0)

    breaker = DrawdownCircuitBreaker()
    breaker.update_trading_status(0.15)  # Trigger warning alert

    report = generate_daily_risk_report(
        account_balance=85000.0,
        equity_tracker=tracker,
        open_positions=[],
        circuit_breaker=breaker
    )

    assert len(report.alerts) > 0
    assert "WARNING" in report.alerts[0]


# ============================================================================
# Position Data Class Tests
# ============================================================================

def test_position_market_value():
    """Test position market value calculation."""
    pos = Position("SPY", 100, 400.0, 410.0, 395.0)

    assert pos.market_value == 41000.0


def test_position_risk_amount():
    """Test position risk amount calculation."""
    pos = Position("SPY", 100, 400.0, 410.0, 395.0)

    assert pos.risk_amount == 500.0  # 100 × (400 - 395)


def test_position_unrealized_pnl():
    """Test unrealized P&L calculation."""
    pos = Position("SPY", 100, 400.0, 410.0, 395.0)

    assert pos.unrealized_pnl == 1000.0  # 100 × (410 - 400)


def test_position_short():
    """Test position calculations for short positions."""
    pos = Position("SPY", -100, 400.0, 390.0, 405.0)

    assert pos.market_value == 39000.0  # abs(-100 × 390)
    assert pos.risk_amount == 500.0  # abs(-100 × (400 - 405))
    assert pos.unrealized_pnl == 1000.0  # -100 × (390 - 400) = 1000 profit


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_drawdown_workflow():
    """Test complete drawdown monitoring workflow."""
    # Initialize
    tracker = EquityTracker(100000.0)
    breaker = DrawdownCircuitBreaker()

    # Normal trading
    tracker.update_equity(105000.0)
    status, alerts = breaker.update_trading_status(tracker.calculate_drawdown_pct())
    assert status == TradingStatus.ACTIVE

    # Market downturn: 16% drawdown
    tracker.update_equity(88200.0)  # 16% down from peak
    status, alerts = breaker.update_trading_status(tracker.calculate_drawdown_pct())
    assert status == TradingStatus.REDUCED
    assert len(alerts) == 1

    # Further decline: 21% drawdown
    tracker.update_equity(82950.0)  # 21% down from peak
    status, alerts = breaker.update_trading_status(tracker.calculate_drawdown_pct())
    assert status == TradingStatus.DISABLED
    assert breaker.should_close_all_positions() is True


def test_risk_monitoring_with_multiple_positions():
    """Test risk monitoring with portfolio of positions."""
    positions = [
        Position("SPY", 100, 400.0, 410.0, 395.0),
        Position("QQQ", 50, 300.0, 305.0, 291.0),
        Position("IWM", 200, 180.0, 185.0, 175.0),
    ]

    metrics = calculate_risk_metrics(
        account_balance=100000.0,
        equity_peak=105000.0,
        current_equity=100000.0,
        open_positions=positions
    )

    # Verify all metrics calculated correctly
    assert metrics.num_positions == 3
    assert metrics.total_exposure > 0
    assert metrics.total_risk_deployed > 0
    assert metrics.largest_position_symbol in ["SPY", "QQQ", "IWM"]


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_zero_equity_peak():
    """Test handling of zero equity peak."""
    tracker = EquityTracker(0.0)

    dd_pct = tracker.calculate_drawdown_pct()

    assert dd_pct == 0.0  # Should handle gracefully


def test_custom_config():
    """Test custom circuit breaker configuration."""
    config = DrawdownCircuitBreakerConfig(
        warning_drawdown_pct=0.10,  # 10% warning
        critical_drawdown_pct=0.15,  # 15% critical
        warning_size_reduction=0.75,  # 75% reduction
        warning_stop_multiplier=1.5,
    )

    breaker = DrawdownCircuitBreaker(config)

    # Warning at 10%
    level = breaker.check_drawdown_level(0.11)
    assert level == DrawdownLevel.WARNING

    # Critical at 15%
    level = breaker.check_drawdown_level(0.16)
    assert level == DrawdownLevel.CRITICAL

    # Size multiplier
    breaker.trading_status = TradingStatus.REDUCED
    assert breaker.get_position_size_multiplier() == 0.25  # 1.0 - 0.75

    # Stop multiplier
    assert breaker.get_stop_loss_multiplier() == 1.5


def test_empty_position_list():
    """Test risk calculations with empty position list."""
    metrics = calculate_risk_metrics(
        account_balance=100000.0,
        equity_peak=100000.0,
        current_equity=100000.0,
        open_positions=[]
    )

    assert metrics.num_positions == 0
    assert metrics.total_exposure == 0.0
    assert metrics.largest_position_symbol is None


# ============================================================================
# Success Criteria Tests (from FLUXHERO_REQUIREMENTS.md)
# ============================================================================

def test_success_criteria_five_losing_trades():
    """Test that 5 consecutive 1% losses don't trigger circuit breaker."""
    tracker = EquityTracker(100000.0)
    breaker = DrawdownCircuitBreaker()

    # 5 × 1% losses = 5% drawdown (with slippage ~6%)
    for i in range(5):
        new_equity = tracker.current_equity * 0.988  # Slightly worse than 1%
        tracker.update_equity(new_equity)

    dd_pct = tracker.calculate_drawdown_pct()

    # Should be under 15% threshold
    assert dd_pct < 0.15

    # Should still be active
    status, _ = breaker.update_trading_status(dd_pct)
    assert status == TradingStatus.ACTIVE


def test_success_criteria_15_percent_drawdown():
    """Test 15% drawdown triggers size reduction (R11.3.2)."""
    tracker = EquityTracker(100000.0)
    breaker = DrawdownCircuitBreaker()

    # 15% drawdown
    tracker.update_equity(85000.0)
    status, alerts = breaker.update_trading_status(tracker.calculate_drawdown_pct())

    # Verify actions taken
    assert status == TradingStatus.REDUCED
    assert breaker.get_position_size_multiplier() == 0.5  # 50% reduction
    assert breaker.get_stop_loss_multiplier() == 2.0  # Tightened
    assert len(alerts) > 0


def test_success_criteria_20_percent_drawdown():
    """Test 20% drawdown disables trading (R11.3.3)."""
    tracker = EquityTracker(100000.0)
    breaker = DrawdownCircuitBreaker()

    # 20% drawdown
    tracker.update_equity(80000.0)
    status, alerts = breaker.update_trading_status(tracker.calculate_drawdown_pct())

    # Verify actions taken
    assert status == TradingStatus.DISABLED
    assert breaker.should_close_all_positions() is True
    can_trade, _ = breaker.can_open_new_position()
    assert can_trade is False
    assert breaker.manual_review_required is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
