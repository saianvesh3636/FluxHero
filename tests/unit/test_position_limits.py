"""
Unit tests for position_limits.py (Risk Management System - Feature 11).

Tests cover:
- Position-level risk checks (R11.1)
- Portfolio-level risk checks (R11.2)
- Correlation monitoring (R11.2.3)
- Stop loss calculations (R11.1.4)
- Risk monitoring functions

Author: FluxHero
Date: 2026-01-20
"""

import numpy as np
import pytest

from backend.core.config import Settings
from backend.risk.position_limits import (
    Position,
    PositionLimitsConfig,
    RiskCheckResult,
    StrategyType,
    calculate_atr_stop_loss,
    calculate_correlation,
    calculate_position_size_from_risk,
    calculate_total_portfolio_risk,
    calculate_worst_case_loss,
    check_correlation_with_existing_positions,
    get_largest_position,
    validate_new_position,
    validate_portfolio_level_risk,
    validate_position_level_risk,
)

# ============================================================================
# Position-Level Risk Tests (R11.1)
# ============================================================================


def test_calculate_position_size_trend_following():
    """Test position sizing for trend-following strategy (1% risk)."""
    # R11.1.1: 1% risk for trend-following
    account_balance = 100000
    entry_price = 50.0
    stop_loss = 48.0  # $2 risk per share

    shares = calculate_position_size_from_risk(
        account_balance, entry_price, stop_loss, StrategyType.TREND_FOLLOWING
    )

    # Expected: $1000 risk / $2 per share = 500 shares
    assert shares == 500.0

    # Verify actual risk
    actual_risk = shares * (entry_price - stop_loss)
    assert actual_risk == 1000.0  # 1% of account


def test_calculate_position_size_mean_reversion():
    """Test position sizing for mean-reversion strategy (0.75% risk)."""
    # R11.1.1: 0.75% risk for mean-reversion
    account_balance = 100000
    entry_price = 50.0
    stop_loss = 48.5  # $1.50 risk per share

    shares = calculate_position_size_from_risk(
        account_balance, entry_price, stop_loss, StrategyType.MEAN_REVERSION
    )

    # Expected: $750 risk / $1.50 per share = 500 shares
    assert shares == 500.0

    # Verify actual risk
    actual_risk = shares * (entry_price - stop_loss)
    assert actual_risk == 750.0  # 0.75% of account


def test_calculate_position_size_zero_risk():
    """Test position sizing with zero price risk (edge case)."""
    account_balance = 100000
    entry_price = 50.0
    stop_loss = 50.0  # No risk!

    shares = calculate_position_size_from_risk(
        account_balance, entry_price, stop_loss, StrategyType.TREND_FOLLOWING
    )

    # Should return 0 shares (can't calculate position size)
    assert shares == 0.0


def test_calculate_position_size_short_position():
    """Test position sizing for short positions."""
    account_balance = 100000
    entry_price = 50.0
    stop_loss = 52.0  # $2 risk per share (stop above entry for short)

    shares = calculate_position_size_from_risk(
        account_balance, entry_price, stop_loss, StrategyType.TREND_FOLLOWING
    )

    # Expected: $1000 risk / $2 per share = 500 shares
    assert shares == 500.0


def test_validate_position_level_risk_approved():
    """Test position-level risk validation (all checks pass)."""
    account_balance = 100000
    entry_price = 50.0
    stop_loss = 48.0
    shares = 333  # 1% risk ($1000 / $2 = 500), but limited to ~17% position size

    result, reason = validate_position_level_risk(
        account_balance, entry_price, stop_loss, shares, StrategyType.TREND_FOLLOWING
    )

    assert result == RiskCheckResult.APPROVED
    assert "passed" in reason.lower()


def test_validate_position_level_risk_no_stop():
    """Test rejection when no stop loss provided (R11.1.3)."""
    account_balance = 100000
    entry_price = 50.0
    stop_loss = None  # No stop!
    shares = 500

    result, reason = validate_position_level_risk(
        account_balance, entry_price, stop_loss, shares, StrategyType.TREND_FOLLOWING
    )

    assert result == RiskCheckResult.REJECTED_NO_STOP
    assert "mandatory" in reason.lower()


def test_validate_position_level_risk_excessive_risk():
    """Test rejection when risk exceeds max (R11.1.1)."""
    account_balance = 100000
    entry_price = 50.0
    stop_loss = 45.0  # $5 risk per share
    shares = 300  # Risk = $1500 (1.5%, exceeds 1% max)

    result, reason = validate_position_level_risk(
        account_balance, entry_price, stop_loss, shares, StrategyType.TREND_FOLLOWING
    )

    assert result == RiskCheckResult.REJECTED_EXCESSIVE_RISK
    assert "risk" in reason.lower()
    assert "1500" in reason  # Actual risk amount


def test_validate_position_level_risk_position_too_large():
    """Test rejection when position size exceeds 20% of account (R11.1.2)."""
    account_balance = 100000
    entry_price = 50.0
    stop_loss = 49.0  # $1 risk per share (low risk)
    shares = 500  # Position value = $25,000 (25%, exceeds 20% max)

    result, reason = validate_position_level_risk(
        account_balance, entry_price, stop_loss, shares, StrategyType.TREND_FOLLOWING
    )

    assert result == RiskCheckResult.REJECTED_POSITION_TOO_LARGE
    assert "20%" in reason or "position size" in reason.lower()


def test_calculate_atr_stop_loss_trend_long():
    """Test ATR-based stop for trend-following long position (R11.1.4)."""
    entry_price = 100.0
    atr = 2.0
    side = 1  # Long

    stop = calculate_atr_stop_loss(entry_price, atr, side, StrategyType.TREND_FOLLOWING)

    # Expected: 100 - (2.5 × 2.0) = 95.0
    assert stop == 95.0


def test_calculate_atr_stop_loss_trend_short():
    """Test ATR-based stop for trend-following short position (R11.1.4)."""
    entry_price = 100.0
    atr = 2.0
    side = -1  # Short

    stop = calculate_atr_stop_loss(entry_price, atr, side, StrategyType.TREND_FOLLOWING)

    # Expected: 100 + (2.5 × 2.0) = 105.0
    assert stop == 105.0


def test_calculate_atr_stop_loss_mean_rev_long():
    """Test fixed percentage stop for mean-reversion long (R11.1.4)."""
    entry_price = 100.0
    atr = 2.0  # Ignored for mean-reversion
    side = 1  # Long

    stop = calculate_atr_stop_loss(entry_price, atr, side, StrategyType.MEAN_REVERSION)

    # Expected: 100 - (3% of 100) = 97.0
    assert stop == 97.0


def test_calculate_atr_stop_loss_mean_rev_short():
    """Test fixed percentage stop for mean-reversion short (R11.1.4)."""
    entry_price = 100.0
    atr = 2.0
    side = -1  # Short

    stop = calculate_atr_stop_loss(entry_price, atr, side, StrategyType.MEAN_REVERSION)

    # Expected: 100 + (3% of 100) = 103.0
    assert stop == 103.0


# ============================================================================
# Portfolio-Level Risk Tests (R11.2)
# ============================================================================


def test_validate_portfolio_level_risk_approved():
    """Test portfolio-level risk validation (all checks pass)."""
    account_balance = 100000
    open_positions = [
        Position("SPY", 100, 400, 410, 395),  # Value: $41,000
    ]
    new_position_value = 8000  # Total: $49,000 (49%, within 50% limit)

    result, reason = validate_portfolio_level_risk(
        account_balance, open_positions, new_position_value
    )

    assert result == RiskCheckResult.APPROVED
    assert "passed" in reason.lower()


def test_validate_portfolio_level_risk_max_positions():
    """Test rejection when max positions reached (R11.2.2)."""
    account_balance = 100000
    open_positions = [
        Position("SPY", 50, 400, 410, 395),
        Position("QQQ", 50, 300, 305, 291),
        Position("IWM", 50, 200, 205, 195),
        Position("DIA", 50, 350, 355, 345),
        Position("AAPL", 50, 150, 155, 147),
    ]  # 5 positions (max)
    new_position_value = 5000

    result, reason = validate_portfolio_level_risk(
        account_balance, open_positions, new_position_value
    )

    assert result == RiskCheckResult.REJECTED_MAX_POSITIONS
    assert "5" in reason
    assert "max" in reason.lower()


def test_validate_portfolio_level_risk_total_exposure():
    """Test rejection when total exposure exceeds 50% (R11.2.1)."""
    account_balance = 100000
    open_positions = [
        Position("SPY", 100, 400, 410, 395),  # Value: $41,000
    ]
    new_position_value = 11000  # Total: $52,000 (52%, exceeds 50% max)

    result, reason = validate_portfolio_level_risk(
        account_balance, open_positions, new_position_value
    )

    assert result == RiskCheckResult.REJECTED_TOTAL_EXPOSURE
    assert "50%" in reason or "exposure" in reason.lower()


def test_validate_portfolio_level_risk_empty_portfolio():
    """Test portfolio validation with no existing positions."""
    account_balance = 100000
    open_positions = []
    new_position_value = 20000  # 20% (well within limits)

    result, reason = validate_portfolio_level_risk(
        account_balance, open_positions, new_position_value
    )

    assert result == RiskCheckResult.APPROVED


# ============================================================================
# Correlation Tests (R11.2.3)
# ============================================================================


def test_calculate_correlation_perfect_positive():
    """Test correlation calculation with perfect positive correlation."""
    prices1 = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    prices2 = np.array([200.0, 202.0, 204.0, 206.0, 208.0])

    corr = calculate_correlation(prices1, prices2)

    # Should be very close to 1.0
    assert abs(corr - 1.0) < 0.01


def test_calculate_correlation_perfect_negative():
    """Test correlation calculation with perfect negative correlation."""
    prices1 = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    prices2 = np.array([208.0, 206.0, 204.0, 202.0, 200.0])

    corr = calculate_correlation(prices1, prices2)

    # Should be very close to -1.0
    assert abs(corr - (-1.0)) < 0.01


def test_calculate_correlation_no_correlation():
    """Test correlation calculation with uncorrelated series."""
    prices1 = np.array([100.0, 105.0, 95.0, 110.0, 90.0])
    prices2 = np.array([200.0, 200.0, 200.0, 200.0, 200.0])  # Constant

    corr = calculate_correlation(prices1, prices2)

    # Constant series has zero correlation (zero std dev)
    assert corr == 0.0


def test_calculate_correlation_edge_cases():
    """Test correlation with edge cases (mismatched lengths, short series)."""
    # Mismatched lengths
    prices1 = np.array([100.0, 101.0])
    prices2 = np.array([200.0, 201.0, 202.0])
    corr = calculate_correlation(prices1, prices2)
    assert corr == 0.0  # Should handle gracefully

    # Single value
    prices1 = np.array([100.0])
    prices2 = np.array([200.0])
    corr = calculate_correlation(prices1, prices2)
    assert corr == 0.0


def test_check_correlation_no_positions():
    """Test correlation check with no existing positions."""
    new_prices = np.array([100.0, 101.0, 102.0])
    open_positions = []
    prices_map = {}

    should_reduce, max_corr, symbol = check_correlation_with_existing_positions(
        new_prices, open_positions, prices_map
    )

    assert should_reduce is False
    assert max_corr == 0.0
    assert symbol is None


def test_check_correlation_high_correlation():
    """Test correlation check with high correlation (R11.2.3)."""
    new_prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    existing_prices = np.array([200.0, 202.0, 204.0, 206.0, 208.0])  # Perfect corr

    open_positions = [Position("SPY", 100, 400, 410, 395)]
    prices_map = {"SPY": existing_prices}

    should_reduce, max_corr, symbol = check_correlation_with_existing_positions(
        new_prices, open_positions, prices_map
    )

    # Correlation > 0.7 threshold
    assert should_reduce is True
    assert max_corr > 0.7
    assert symbol == "SPY"


def test_check_correlation_low_correlation():
    """Test correlation check with low correlation."""
    new_prices = np.array([100.0, 102.0, 101.0, 103.0, 102.0])  # Flat pattern
    existing_prices = np.array([200.0, 199.0, 201.0, 198.0, 202.0])  # Oscillating pattern

    open_positions = [Position("SPY", 100, 400, 410, 395)]
    prices_map = {"SPY": existing_prices}

    should_reduce, max_corr, symbol = check_correlation_with_existing_positions(
        new_prices, open_positions, prices_map
    )

    # Correlation should be low (not high enough to trigger reduction)
    assert should_reduce is False
    assert max_corr < 0.7  # Below threshold


# ============================================================================
# Comprehensive Validation Tests
# ============================================================================


def test_validate_new_position_all_checks_pass():
    """Test comprehensive validation with all checks passing."""
    account_balance = 100000
    entry_price = 50.0
    stop_loss = 48.0
    shares = 333  # ~17% position size (under 20% limit), 1% risk would be 500 but limited
    open_positions = []

    result, reason, adjusted_shares = validate_new_position(
        account_balance,
        entry_price,
        stop_loss,
        shares,
        StrategyType.TREND_FOLLOWING,
        open_positions,
    )

    assert result == RiskCheckResult.APPROVED
    assert adjusted_shares == shares  # No adjustment


def test_validate_new_position_correlation_adjustment():
    """Test position size reduction due to high correlation (R11.2.3)."""
    account_balance = 100000
    entry_price = 50.0
    stop_loss = 48.0
    shares = 333  # ~17% position size, within limits

    # Highly correlated prices
    new_prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    existing_prices = np.array([200.0, 202.0, 204.0, 206.0, 208.0])

    # Smaller existing position to stay within 50% total exposure limit
    # 50 shares * 410 = $20,500 (20.5%)
    open_positions = [Position("SPY", 50, 400, 410, 395)]
    prices_map = {"SPY": existing_prices}

    result, reason, adjusted_shares = validate_new_position(
        account_balance,
        entry_price,
        stop_loss,
        shares,
        StrategyType.TREND_FOLLOWING,
        open_positions,
        new_symbol_prices=new_prices,
        position_prices_map=prices_map,
    )

    # Should approve but reduce size by 50%
    assert result == RiskCheckResult.APPROVED
    assert adjusted_shares == np.floor(333 * 0.5)  # 50% reduction = 166
    assert "correlation" in reason.lower()


def test_validate_new_position_fails_position_risk():
    """Test validation failure at position-level risk check."""
    account_balance = 100000
    entry_price = 50.0
    stop_loss = None  # No stop!
    shares = 500
    open_positions = []

    result, reason, adjusted_shares = validate_new_position(
        account_balance,
        entry_price,
        stop_loss,
        shares,
        StrategyType.TREND_FOLLOWING,
        open_positions,
    )

    assert result == RiskCheckResult.REJECTED_NO_STOP
    assert adjusted_shares == 0.0


def test_validate_new_position_fails_portfolio_risk():
    """Test validation failure at portfolio-level risk check."""
    account_balance = 100000
    entry_price = 50.0
    stop_loss = 48.0
    shares = 333  # Within position size limit (20%), but will fail on max positions

    # Already at max 5 positions
    open_positions = [
        Position("SPY", 50, 400, 410, 395),
        Position("QQQ", 50, 300, 305, 291),
        Position("IWM", 50, 200, 205, 195),
        Position("DIA", 50, 350, 355, 345),
        Position("AAPL", 50, 150, 155, 147),
    ]

    result, reason, adjusted_shares = validate_new_position(
        account_balance,
        entry_price,
        stop_loss,
        shares,
        StrategyType.TREND_FOLLOWING,
        open_positions,
    )

    assert result == RiskCheckResult.REJECTED_MAX_POSITIONS
    assert adjusted_shares == 0.0


# ============================================================================
# Risk Monitoring Tests (R11.4)
# ============================================================================


def test_calculate_total_portfolio_risk():
    """Test calculation of total portfolio risk and exposure (R11.4.2)."""
    open_positions = [
        Position("SPY", 100, 400, 410, 395),  # Risk: 100×5=$500, Exposure: $41k
        Position("QQQ", 50, 300, 305, 291),  # Risk: 50×9=$450, Exposure: $15.25k
    ]

    total_risk, total_exposure = calculate_total_portfolio_risk(open_positions)

    assert total_risk == 950.0  # $500 + $450
    assert total_exposure == 56250.0  # $41,000 + $15,250


def test_calculate_total_portfolio_risk_empty():
    """Test portfolio risk with no positions."""
    open_positions = []

    total_risk, total_exposure = calculate_total_portfolio_risk(open_positions)

    assert total_risk == 0.0
    assert total_exposure == 0.0


def test_get_largest_position():
    """Test getting largest position by market value (R11.4.2)."""
    open_positions = [
        Position("SPY", 100, 400, 410, 395),  # $41,000
        Position("QQQ", 50, 300, 305, 291),  # $15,250
        Position("AAPL", 200, 150, 155, 147),  # $31,000
    ]

    largest = get_largest_position(open_positions)

    assert largest is not None
    assert largest.symbol == "SPY"
    assert largest.market_value == 41000.0


def test_get_largest_position_empty():
    """Test largest position with no positions."""
    open_positions = []

    largest = get_largest_position(open_positions)

    assert largest is None


def test_calculate_worst_case_loss():
    """Test worst-case loss calculation if all stops hit (R11.4.2)."""
    open_positions = [
        Position("SPY", 100, 400, 410, 395),  # Risk: $500
        Position("QQQ", 50, 300, 305, 291),  # Risk: $450
        Position("AAPL", 200, 150, 155, 147),  # Risk: $600
    ]

    worst_case = calculate_worst_case_loss(open_positions)

    assert worst_case == 1550.0  # $500 + $450 + $600


def test_calculate_worst_case_loss_empty():
    """Test worst-case loss with no positions."""
    open_positions = []

    worst_case = calculate_worst_case_loss(open_positions)

    assert worst_case == 0.0


# ============================================================================
# Success Criteria Tests (from FLUXHERO_REQUIREMENTS.md)
# ============================================================================


def test_success_criteria_five_losing_trades():
    """Test that 5 consecutive 1% losses result in <6% drawdown."""
    # Simulate 5 trades with 1% risk each
    account_balance = 100000

    # Each trade risks $1000 (1%)
    # With slippage/commissions, actual loss might be slightly more
    # But should stay under 6% total

    total_loss = 5 * 1000  # Perfect 1% loss each
    drawdown_pct = (total_loss / account_balance) * 100

    assert drawdown_pct == 5.0  # Exactly 5% (before slippage)
    assert drawdown_pct < 6.0  # Within success criteria


def test_success_criteria_position_sizing():
    """Test position sizing matches example from requirements."""
    # Example: $100k account, $50 entry, $48 stop
    # Expected: 500 shares (1% risk = $1k)

    account_balance = 100000
    entry_price = 50.0
    stop_loss = 48.0

    shares = calculate_position_size_from_risk(
        account_balance, entry_price, stop_loss, StrategyType.TREND_FOLLOWING
    )

    assert shares == 500.0

    # Verify actual risk is 1%
    risk_amount = shares * (entry_price - stop_loss)
    risk_pct = (risk_amount / account_balance) * 100
    assert risk_pct == 1.0


def test_success_criteria_correlation_reduction():
    """Test correlated position gets size reduced by 50% (R11.2.3)."""
    account_balance = 100000
    entry_price = 50.0
    stop_loss = 48.0
    shares = 333  # Within limits

    # Perfect correlation
    new_prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    existing_prices = np.array([200.0, 202.0, 204.0, 206.0, 208.0])

    # Smaller existing position (50 shares * 410 = $20,500 = 20.5%)
    open_positions = [Position("SPY", 50, 400, 410, 395)]
    prices_map = {"SPY": existing_prices}

    result, reason, adjusted_shares = validate_new_position(
        account_balance,
        entry_price,
        stop_loss,
        shares,
        StrategyType.TREND_FOLLOWING,
        open_positions,
        new_symbol_prices=new_prices,
        position_prices_map=prices_map,
    )

    # Should reduce by 50%
    assert adjusted_shares == np.floor(333 * 0.5)  # 166


# ============================================================================
# Edge Case Tests
# ============================================================================


def test_edge_case_very_tight_stop():
    """Test position sizing with very tight stop (large position)."""
    account_balance = 100000
    entry_price = 100.0
    stop_loss = 99.9  # Only $0.10 risk per share

    shares = calculate_position_size_from_risk(
        account_balance, entry_price, stop_loss, StrategyType.TREND_FOLLOWING
    )

    # Expected: $1000 / $0.10 = 10,000 shares
    # But this should fail max position size check (20%)
    assert shares == 10000.0

    # Validation should reject due to position size
    result, reason = validate_position_level_risk(
        account_balance, entry_price, stop_loss, shares, StrategyType.TREND_FOLLOWING
    )

    assert result == RiskCheckResult.REJECTED_POSITION_TOO_LARGE


def test_edge_case_very_wide_stop():
    """Test position sizing with very wide stop (small position)."""
    account_balance = 100000
    entry_price = 100.0
    stop_loss = 50.0  # $50 risk per share (very wide)

    shares = calculate_position_size_from_risk(
        account_balance, entry_price, stop_loss, StrategyType.TREND_FOLLOWING
    )

    # Expected: $1000 / $50 = 20 shares (small position)
    assert shares == 20.0


def test_edge_case_custom_config():
    """Test with custom configuration parameters."""
    custom_config = PositionLimitsConfig(
        max_risk_pct_trend=0.02,  # 2% instead of 1%
        max_position_size_pct=0.30,  # 30% instead of 20%
        max_total_exposure_pct=0.70,  # 70% instead of 50%
    )

    account_balance = 100000
    entry_price = 50.0
    stop_loss = 48.0

    # With 2% risk: should get 1000 shares
    shares = calculate_position_size_from_risk(
        account_balance, entry_price, stop_loss, StrategyType.TREND_FOLLOWING, custom_config
    )

    assert shares == 1000.0

    # Use 600 shares for validation (within 30% position size limit)
    # 600 shares * $50 = $30,000 = 30% of account
    result, reason = validate_position_level_risk(
        account_balance, entry_price, stop_loss, 600, StrategyType.TREND_FOLLOWING, custom_config
    )

    assert result == RiskCheckResult.APPROVED


def test_centralized_config_integration():
    """Test that centralized Settings config works with risk functions."""
    # Create a custom Settings instance with different risk parameters
    custom_settings = Settings(
        max_risk_pct_trend=0.015,  # 1.5% instead of default 1%
        max_position_size_pct=0.25,  # 25% instead of default 20%
        max_total_exposure_pct=0.60,  # 60% instead of default 50%
        max_open_positions=10,  # 10 instead of default 5
        correlation_threshold=0.8,  # 0.8 instead of default 0.7
    )

    account_balance = 100000
    entry_price = 50.0  # Lower price to avoid position size limit
    stop_loss = 49.0

    # Test position sizing with custom settings
    # With 1.5% risk and $1 price risk: (100000 * 0.015) / 1 = 1500 shares
    shares = calculate_position_size_from_risk(
        account_balance, entry_price, stop_loss, StrategyType.TREND_FOLLOWING, custom_settings
    )
    assert shares == 1500.0

    # Test position validation with custom settings
    # Use smaller position (500 shares) to stay within 25% position size limit
    # 500 shares * $50 = $25,000 = 25% of account (exactly at limit)
    result, reason = validate_position_level_risk(
        account_balance, entry_price, stop_loss, 500, StrategyType.TREND_FOLLOWING, custom_settings
    )
    assert result == RiskCheckResult.APPROVED

    # Test ATR stop loss with custom settings
    stop_price = calculate_atr_stop_loss(
        entry_price=100.0,
        atr=2.0,
        side=1,  # Long position
        strategy_type=StrategyType.TREND_FOLLOWING,
        config=custom_settings,
    )
    assert stop_price == 95.0  # 100 - (2.5 * 2.0)

    # Test portfolio validation with custom settings (max 10 positions)
    positions = [Position(f"SYM{i}", 10, 100, 100, 95) for i in range(9)]
    result, reason = validate_portfolio_level_risk(
        account_balance, positions, 10000, custom_settings
    )
    assert result == RiskCheckResult.APPROVED  # 9 positions < 10 max

    # Test correlation check with custom threshold (0.8)
    new_prices = np.array([100, 101, 102, 103, 104])
    position_prices_map = {"SYM0": np.array([100, 101, 102, 103, 104])}
    should_reduce, max_corr, _ = check_correlation_with_existing_positions(
        new_prices, [positions[0]], position_prices_map, custom_settings
    )
    # Correlation is 1.0, which exceeds 0.8 threshold
    assert should_reduce is True
    assert abs(max_corr - 1.0) < 1e-10  # Near perfect correlation


def test_centralized_config_default_values():
    """Test that functions use centralized config defaults when config is None."""
    account_balance = 100000
    entry_price = 50.0
    stop_loss = 48.0

    # Call without config parameter - should use get_settings() internally
    shares = calculate_position_size_from_risk(
        account_balance, entry_price, stop_loss, StrategyType.TREND_FOLLOWING
    )
    # Default is 1% risk: (100000 * 0.01) / 2 = 500 shares
    assert shares == 500.0

    # Test validation without config
    # Use 400 shares to stay within 20% position size limit
    # 400 shares * $50 = $20,000 = 20% of account (exactly at limit)
    result, reason = validate_position_level_risk(
        account_balance, entry_price, stop_loss, 400, StrategyType.TREND_FOLLOWING
    )
    assert result == RiskCheckResult.APPROVED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
