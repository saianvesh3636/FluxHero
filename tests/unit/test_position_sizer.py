"""
Unit tests for position_sizer.py

Tests position sizing calculations, risk limits, and kill-switch functionality.

Author: FluxHero Team
Date: 2026-01-20
"""

import pytest

from backend.execution.position_sizer import (
    AccountState,
    PositionSize,
    PositionSizer,
    PositionSizeResult,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def basic_account():
    """Basic account state for testing."""
    return AccountState(
        balance=10000.0,
        cash=8000.0,
        deployed_value=2000.0,
        daily_pnl=0.0,
        num_positions=1,
        session_start_balance=10000.0,
    )


@pytest.fixture
def position_sizer():
    """Standard position sizer with default parameters."""
    return PositionSizer(
        risk_pct=0.01,  # 1% risk
        max_position_pct=0.20,  # 20% max position
        max_deployment_pct=0.50,  # 50% max deployment
        kill_switch_pct=0.03,  # 3% kill-switch
        kill_switch_enabled=True,
    )


# ============================================================================
# Test Initialization
# ============================================================================


def test_position_sizer_initialization():
    """Test PositionSizer initialization with default parameters."""
    sizer = PositionSizer()

    assert sizer.risk_pct == 0.01
    assert sizer.max_position_pct == 0.20
    assert sizer.max_deployment_pct == 0.50
    assert sizer.kill_switch_pct == 0.03
    assert sizer.kill_switch_enabled is True
    assert sizer.kill_switch_triggered is False
    assert sizer.kill_switch_trigger_time is None


def test_position_sizer_custom_parameters():
    """Test PositionSizer with custom parameters."""
    sizer = PositionSizer(
        risk_pct=0.02,
        max_position_pct=0.15,
        max_deployment_pct=0.60,
        kill_switch_pct=0.05,
        kill_switch_enabled=False,
    )

    assert sizer.risk_pct == 0.02
    assert sizer.max_position_pct == 0.15
    assert sizer.max_deployment_pct == 0.60
    assert sizer.kill_switch_pct == 0.05
    assert sizer.kill_switch_enabled is False


# ============================================================================
# Test 1% Risk Rule (R10.3.1)
# ============================================================================


def test_basic_risk_calculation(position_sizer, basic_account):
    """Test basic 1% risk rule calculation."""
    # R10.3.1: shares = (account_balance × 0.01) / (entry_price - stop_loss_price)
    # shares = (10000 × 0.01) / (100 - 98) = 100 / 2 = 50
    # BUT position limit (20%): 10000 × 0.20 / 100 = 20 shares
    # Takes minimum: 20 shares (position limit wins)
    result = position_sizer.calculate_position_size(
        account=basic_account,
        entry_price=100.0,
        stop_loss_price=98.0,
        strategy="trend",
    )

    assert result.shares == 20  # Limited by 20% position size
    assert result.risk_amount == 40.0  # 20 shares × $2 risk = $40
    assert result.position_value == 2000.0  # 20 shares × $100 = $2000
    assert result.result_code == PositionSizeResult.SUCCESS


def test_tight_stop_loss_larger_position(position_sizer, basic_account):
    """Test that tighter stop loss allows larger position."""
    # Tighter stop (1% vs 2%) should allow 2x position size
    # Risk-based: 100 / 1 = 100 shares
    # Position limit: 20 shares (still wins)
    result = position_sizer.calculate_position_size(
        account=basic_account,
        entry_price=100.0,
        stop_loss_price=99.0,  # Tight 1% stop
        strategy="trend",
    )

    # Still limited by 20% position size limit
    assert result.shares == 20
    assert result.risk_amount == 20.0  # 20 shares × $1 risk
    assert result.result_code == PositionSizeResult.SUCCESS


def test_wide_stop_loss_smaller_position(position_sizer, basic_account):
    """Test that wider stop loss reduces position size."""
    # Wide 5% stop should give smaller position
    result = position_sizer.calculate_position_size(
        account=basic_account,
        entry_price=100.0,
        stop_loss_price=95.0,  # Wide 5% stop
        strategy="trend",
    )

    assert result.shares == 20  # Smaller position due to wide stop
    assert result.risk_amount == 100.0
    assert result.result_code == PositionSizeResult.SUCCESS


# ============================================================================
# Test Max Position Size Limit (R10.3.2 - 20% per position)
# ============================================================================


def test_max_position_size_limit(position_sizer, basic_account):
    """Test 20% max position size limit."""
    # With very tight stop (0.1%), risk-based sizing would be huge
    # But should be capped at 20% of account = $2000
    result = position_sizer.calculate_position_size(
        account=basic_account,
        entry_price=100.0,
        stop_loss_price=99.90,  # 0.1% stop
        strategy="trend",
    )

    # Max position: 10000 × 0.20 = $2000 / $100 = 20 shares
    assert result.shares == 20
    assert result.position_value == 2000.0
    assert result.result_code == PositionSizeResult.SUCCESS


def test_position_limit_with_expensive_stock(position_sizer, basic_account):
    """Test position limit with expensive stock."""
    # Stock at $500, max position = $2000 / $500 = 4 shares
    result = position_sizer.calculate_position_size(
        account=basic_account,
        entry_price=500.0,
        stop_loss_price=490.0,  # $10 risk
        strategy="trend",
    )

    # Risk-based: (10000 × 0.01) / 10 = 10 shares
    # Position limit: (10000 × 0.20) / 500 = 4 shares
    # Should take minimum: 4 shares
    assert result.shares == 4
    assert result.position_value == 2000.0


# ============================================================================
# Test Max Deployment Limit (R10.3.2 - 50% total)
# ============================================================================


def test_max_deployment_limit(position_sizer):
    """Test 50% max total deployment limit."""
    # Account already has 40% deployed, can only add 10% more
    account = AccountState(
        balance=10000.0,
        cash=6000.0,
        deployed_value=4000.0,  # 40% already deployed
        daily_pnl=0.0,
        num_positions=2,
        session_start_balance=10000.0,
    )

    result = position_sizer.calculate_position_size(
        account=account,
        entry_price=100.0,
        stop_loss_price=98.0,
        strategy="trend",
    )

    # Max deployment: 10000 × 0.50 = $5000
    # Already deployed: $4000
    # Available: $1000 / $100 = 10 shares
    assert result.shares == 10
    assert result.position_value == 1000.0
    assert result.result_code == PositionSizeResult.SUCCESS


def test_deployment_limit_exceeded(position_sizer):
    """Test rejection when deployment limit exceeded."""
    # Account already at 50% deployment limit
    account = AccountState(
        balance=10000.0,
        cash=5000.0,
        deployed_value=5000.0,  # 50% deployed (at limit)
        daily_pnl=0.0,
        num_positions=3,
        session_start_balance=10000.0,
    )

    result = position_sizer.calculate_position_size(
        account=account,
        entry_price=100.0,
        stop_loss_price=98.0,
        strategy="trend",
    )

    assert result.shares == 0
    assert result.result_code == PositionSizeResult.EXCEEDS_TOTAL_DEPLOYMENT


# ============================================================================
# Test Whole Share Rounding (R10.3.3)
# ============================================================================


def test_round_down_to_whole_shares(position_sizer, basic_account):
    """Test that fractional shares are rounded down."""
    # Entry price that would give fractional shares
    result = position_sizer.calculate_position_size(
        account=basic_account,
        entry_price=99.99,  # Odd price
        stop_loss_price=98.0,
        strategy="trend",
    )

    # Position limit: 10000 × 0.20 / 99.99 = 20.002... → rounds to 20
    assert result.shares == 20
    assert isinstance(result.shares, int)


def test_minimum_one_share(position_sizer):
    """Test that positions smaller than 1 share are rejected."""
    # Very small account
    account = AccountState(
        balance=100.0,
        cash=100.0,
        deployed_value=0.0,
        daily_pnl=0.0,
        num_positions=0,
        session_start_balance=100.0,
    )

    result = position_sizer.calculate_position_size(
        account=account,
        entry_price=200.0,  # Stock too expensive
        stop_loss_price=198.0,
        strategy="trend",
    )

    # Risk amount: 100 × 0.01 = $1
    # Risk per share: $2
    # Shares: 1 / 2 = 0.5 → rounds to 0 → rejected
    assert result.shares == 0
    assert result.result_code == PositionSizeResult.INSUFFICIENT_CAPITAL


# ============================================================================
# Test Kill-Switch (R10.4.1, R10.4.2, R10.4.3)
# ============================================================================


def test_kill_switch_triggers_at_3_percent_loss(position_sizer, basic_account):
    """Test kill-switch triggers at -3% daily loss (R10.4.2)."""
    # Set daily P&L to -3% loss
    basic_account.daily_pnl = -300.0  # -3% of $10,000
    basic_account.session_start_balance = 10000.0

    result = position_sizer.calculate_position_size(
        account=basic_account,
        entry_price=100.0,
        stop_loss_price=98.0,
        strategy="trend",
    )

    assert result.shares == 0
    assert result.result_code == PositionSizeResult.KILL_SWITCH_ACTIVE
    assert position_sizer.kill_switch_triggered is True
    assert "Kill-switch active" in result.reason


def test_kill_switch_not_triggered_below_threshold(position_sizer, basic_account):
    """Test kill-switch doesn't trigger below -3% threshold."""
    # Set daily P&L to -2.9% loss (below threshold)
    basic_account.daily_pnl = -290.0

    result = position_sizer.calculate_position_size(
        account=basic_account,
        entry_price=100.0,
        stop_loss_price=98.0,
        strategy="trend",
    )

    assert result.shares > 0  # Should still allow trades
    assert result.result_code == PositionSizeResult.SUCCESS
    assert position_sizer.kill_switch_triggered is False


def test_kill_switch_manual_reset(position_sizer, basic_account):
    """Test manual reset of kill-switch (R10.4.3)."""
    # Trigger kill-switch
    basic_account.daily_pnl = -350.0  # -3.5%

    result = position_sizer.calculate_position_size(
        account=basic_account,
        entry_price=100.0,
        stop_loss_price=98.0,
    )
    assert result.shares == 0

    # Manual reset
    position_sizer.reset_kill_switch()

    assert position_sizer.kill_switch_triggered is False
    assert position_sizer.kill_switch_trigger_time is None

    # Now should allow trades (but still at loss)
    basic_account.daily_pnl = -100.0  # Reduced loss
    result = position_sizer.calculate_position_size(
        account=basic_account,
        entry_price=100.0,
        stop_loss_price=98.0,
    )
    assert result.shares > 0


def test_disable_kill_switch(position_sizer, basic_account):
    """Test disabling kill-switch (R10.4.3 manual override)."""
    # Set daily P&L to -4% loss
    basic_account.daily_pnl = -400.0

    # Disable kill-switch
    position_sizer.disable_kill_switch()

    result = position_sizer.calculate_position_size(
        account=basic_account,
        entry_price=100.0,
        stop_loss_price=98.0,
    )

    # Should allow trades despite loss
    assert result.shares > 0
    assert result.result_code == PositionSizeResult.SUCCESS


def test_enable_kill_switch(position_sizer):
    """Test re-enabling kill-switch."""
    position_sizer.disable_kill_switch()
    assert position_sizer.kill_switch_enabled is False

    position_sizer.enable_kill_switch()
    assert position_sizer.kill_switch_enabled is True


# ============================================================================
# Test Cash Constraint
# ============================================================================


def test_insufficient_cash(position_sizer):
    """Test position sizing with insufficient cash."""
    # Low cash available but already near deployment limit
    account = AccountState(
        balance=10000.0,
        cash=1000.0,  # Only $1000 cash
        deployed_value=4000.0,  # 40% deployed (10% more available before 50% limit)
        daily_pnl=0.0,
        num_positions=3,
        session_start_balance=10000.0,
    )

    result = position_sizer.calculate_position_size(
        account=account,
        entry_price=100.0,
        stop_loss_price=98.0,
    )

    # Deployment limit: 10000 × 0.50 = 5000, deployed = 4000, available = 1000
    # Cash available: $1000
    # Both allow: 1000 / 100 = 10 shares
    assert result.shares == 10
    assert result.position_value == 1000.0


def test_zero_cash_available(position_sizer):
    """Test rejection with zero cash."""
    account = AccountState(
        balance=10000.0,
        cash=0.0,  # No cash
        deployed_value=10000.0,
        daily_pnl=0.0,
        num_positions=5,
        session_start_balance=10000.0,
    )

    result = position_sizer.calculate_position_size(
        account=account,
        entry_price=100.0,
        stop_loss_price=98.0,
    )

    assert result.shares == 0
    assert result.result_code == PositionSizeResult.INSUFFICIENT_CAPITAL


# ============================================================================
# Test Edge Cases
# ============================================================================


def test_invalid_entry_price(position_sizer, basic_account):
    """Test handling of invalid entry price."""
    result = position_sizer.calculate_position_size(
        account=basic_account,
        entry_price=0.0,  # Invalid
        stop_loss_price=98.0,
    )

    assert result.shares == 0
    assert result.result_code == PositionSizeResult.INVALID_RISK


def test_invalid_stop_loss_price(position_sizer, basic_account):
    """Test handling of invalid stop loss price."""
    result = position_sizer.calculate_position_size(
        account=basic_account,
        entry_price=100.0,
        stop_loss_price=-5.0,  # Invalid
    )

    assert result.shares == 0
    assert result.result_code == PositionSizeResult.INVALID_RISK


def test_entry_equals_stop(position_sizer, basic_account):
    """Test handling of entry price equal to stop loss."""
    result = position_sizer.calculate_position_size(
        account=basic_account,
        entry_price=100.0,
        stop_loss_price=100.0,  # No risk
    )

    assert result.shares == 0
    assert result.result_code == PositionSizeResult.INVALID_RISK
    assert "entry price equals stop loss" in result.reason


def test_short_position_sizing(position_sizer, basic_account):
    """Test position sizing for short positions (stop above entry)."""
    # Short: entry at 100, stop at 102
    result = position_sizer.calculate_position_size(
        account=basic_account,
        entry_price=100.0,
        stop_loss_price=102.0,  # Stop above entry (short)
        strategy="mean_reversion",
    )

    # Risk per share: |100 - 102| = 2
    # Risk-based: 100 / 2 = 50 shares
    # Position limit: 20 shares (wins)
    assert result.shares == 20
    assert result.risk_amount == 40.0  # 20 shares × $2 risk
    assert result.result_code == PositionSizeResult.SUCCESS


# ============================================================================
# Test Utility Methods
# ============================================================================


def test_get_max_shares(position_sizer, basic_account):
    """Test get_max_shares utility method."""
    max_shares = position_sizer.get_max_shares(
        account=basic_account,
        entry_price=100.0,
    )

    # Position limit: (10000 × 0.20) / 100 = 20 shares
    # Deployment limit: (10000 × 0.50 - 2000) / 100 = 30 shares
    # Cash limit: 8000 / 100 = 80 shares
    # Minimum: 20 shares
    assert max_shares == 20


def test_get_max_shares_deployment_constrained(position_sizer):
    """Test get_max_shares with deployment constraint."""
    account = AccountState(
        balance=10000.0,
        cash=8000.0,
        deployed_value=4500.0,  # Close to limit
        daily_pnl=0.0,
        num_positions=2,
        session_start_balance=10000.0,
    )

    max_shares = position_sizer.get_max_shares(
        account=account,
        entry_price=100.0,
    )

    # Deployment limit: (10000 × 0.50 - 4500) / 100 = 5 shares
    assert max_shares == 5


def test_get_risk_metrics(position_sizer, basic_account):
    """Test get_risk_metrics monitoring."""
    metrics = position_sizer.get_risk_metrics(basic_account)

    assert metrics["deployment_pct"] == 0.20  # 2000 / 10000
    assert metrics["deployment_used"] == 2000.0
    assert metrics["deployment_available"] == 3000.0  # 5000 - 2000
    assert metrics["daily_pnl"] == 0.0
    assert metrics["daily_pnl_pct"] == 0.0
    assert metrics["kill_switch_triggered"] is False
    assert metrics["kill_switch_distance"] > 0  # Positive = safe


def test_get_risk_metrics_near_kill_switch(position_sizer, basic_account):
    """Test risk metrics near kill-switch threshold."""
    basic_account.daily_pnl = -280.0  # -2.8%

    metrics = position_sizer.get_risk_metrics(basic_account)

    assert metrics["daily_pnl"] == -280.0
    assert metrics["daily_pnl_pct"] == -0.028
    assert (
        abs(metrics["kill_switch_distance"] - 0.002) < 0.0001
    )  # ~0.2% away from -3% (floating point tolerance)


# ============================================================================
# Test PositionSize Dataclass
# ============================================================================


def test_position_size_dataclass():
    """Test PositionSize dataclass creation."""
    pos = PositionSize(
        shares=50,
        risk_amount=100.0,
        position_value=5000.0,
        result_code=PositionSizeResult.SUCCESS,
        reason="Test position",
    )

    assert pos.shares == 50
    assert pos.risk_amount == 100.0
    assert pos.position_value == 5000.0
    assert pos.result_code == PositionSizeResult.SUCCESS
    assert pos.reason == "Test position"


# ============================================================================
# Test AccountState Dataclass
# ============================================================================


def test_account_state_dataclass():
    """Test AccountState dataclass creation."""
    account = AccountState(
        balance=10000.0,
        cash=8000.0,
        deployed_value=2000.0,
        daily_pnl=-50.0,
        num_positions=2,
        session_start_balance=10050.0,
    )

    assert account.balance == 10000.0
    assert account.cash == 8000.0
    assert account.deployed_value == 2000.0
    assert account.daily_pnl == -50.0
    assert account.num_positions == 2
    assert account.session_start_balance == 10050.0


# ============================================================================
# Test Success Criteria (from FLUXHERO_REQUIREMENTS.md)
# ============================================================================


def test_success_criteria_1_percent_risk_rule():
    """
    Success criteria: 1% risk rule correctly limits risk.

    From R10.3.1: shares = (account_balance × 0.01) / (entry_price - stop_loss_price)
    """
    sizer = PositionSizer(risk_pct=0.01)
    account = AccountState(
        balance=50000.0,
        cash=50000.0,
        deployed_value=0.0,
        daily_pnl=0.0,
        num_positions=0,
        session_start_balance=50000.0,
    )

    result = sizer.calculate_position_size(
        account=account,
        entry_price=150.0,
        stop_loss_price=147.0,  # $3 risk
    )

    # Risk amount: 50000 × 0.01 = $500
    # Risk-based shares: 500 / 3 = 166.66... → 166 shares
    # Position limit: 50000 × 0.20 / 150 = 66.66... → 66 shares (wins)
    # Actual risk: 66 × 3 = $198
    assert result.shares == 66
    assert result.risk_amount == 198.0


def test_success_criteria_max_position_20_percent():
    """
    Success criteria: Max position size is 20% of account.

    From R10.3.2: Never exceed 20% of account per position
    """
    sizer = PositionSizer(max_position_pct=0.20)
    account = AccountState(
        balance=100000.0,
        cash=100000.0,
        deployed_value=0.0,
        daily_pnl=0.0,
        num_positions=0,
        session_start_balance=100000.0,
    )

    result = sizer.calculate_position_size(
        account=account,
        entry_price=50.0,
        stop_loss_price=49.90,  # Tiny 0.2% stop
    )

    # Max position: 100000 × 0.20 = $20,000
    max_position_value = 20000.0
    assert result.position_value <= max_position_value


def test_success_criteria_max_deployment_50_percent():
    """
    Success criteria: Max total deployment is 50% of account.

    From R10.3.2: Never exceed 50% total deployed capital
    """
    sizer = PositionSizer(max_deployment_pct=0.50)
    account = AccountState(
        balance=100000.0,
        cash=80000.0,
        deployed_value=40000.0,  # 40% deployed
        daily_pnl=0.0,
        num_positions=2,
        session_start_balance=100000.0,
    )

    result = sizer.calculate_position_size(
        account=account,
        entry_price=100.0,
        stop_loss_price=98.0,
    )

    # Max deployment: 50% = $50,000
    # Already deployed: $40,000
    # Available: $10,000
    # Max shares: 10000 / 100 = 100
    assert result.shares <= 100
    assert (account.deployed_value + result.position_value) <= 50000.0


def test_success_criteria_kill_switch_at_3_percent():
    """
    Success criteria: Kill-switch triggers at -3% daily loss.

    From R10.4.2: Stop trading if daily loss > 3%
    """
    sizer = PositionSizer(kill_switch_pct=0.03)
    account = AccountState(
        balance=9700.0,  # Down from $10,000
        cash=9700.0,
        deployed_value=0.0,
        daily_pnl=-300.0,  # -3% loss
        num_positions=0,
        session_start_balance=10000.0,
    )

    result = sizer.calculate_position_size(
        account=account,
        entry_price=100.0,
        stop_loss_price=98.0,
    )

    assert result.shares == 0
    assert result.result_code == PositionSizeResult.KILL_SWITCH_ACTIVE
    assert sizer.kill_switch_triggered is True
