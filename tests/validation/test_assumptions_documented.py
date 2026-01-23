"""
Validation Tests for Documented Assumptions.

This module verifies that the assumptions documented in docs/ASSUMPTIONS.md
match the actual values in the codebase. This ensures documentation stays
in sync with implementation.

Reference: enhancement_tasks.md Phase 24 - Create assumptions document
"""

import numpy as np

from backend.backtesting.engine import BacktestConfig
from backend.backtesting.fills import get_next_bar_fill_price
from backend.core.config import get_settings
from backend.risk.position_limits import (
    StrategyType,
    calculate_position_size_from_risk,
)


class TestCommissionAssumptions:
    """Verify commission model matches documentation ($0.005/share)."""

    def test_default_commission_per_share(self):
        """
        ASSUMPTION: Commission is $0.005 per share (Alpaca-like).

        Documented in: docs/ASSUMPTIONS.md - Commission Model section
        Reference: R9.2.1
        """
        config = BacktestConfig()
        assert config.commission_per_share == 0.005, (
            "Default commission should be $0.005/share as documented"
        )

    def test_commission_round_trip_cost(self):
        """
        Verify round-trip commission calculation example from docs.

        Example from docs:
        - 100-share trade = $1.00 round-trip commission
        - 1000-share trade = $10.00 round-trip commission
        """
        commission_per_share = 0.005

        # 100 shares: entry + exit
        shares_100 = 100
        round_trip_100 = shares_100 * commission_per_share * 2
        assert round_trip_100 == 1.00, "100 shares round-trip should be $1.00"

        # 1000 shares: entry + exit
        shares_1000 = 1000
        round_trip_1000 = shares_1000 * commission_per_share * 2
        assert round_trip_1000 == 10.00, "1000 shares round-trip should be $10.00"


class TestSlippageAssumptions:
    """Verify slippage model matches documentation (0.01% base)."""

    def test_default_base_slippage(self):
        """
        ASSUMPTION: Base slippage is 0.01% (1 basis point).

        Documented in: docs/ASSUMPTIONS.md - Slippage Model section
        Reference: R9.2.2
        """
        config = BacktestConfig()
        assert config.slippage_pct == 0.0001, (
            "Default slippage should be 0.0001 (0.01%) as documented"
        )

    def test_default_impact_penalty(self):
        """
        ASSUMPTION: Impact penalty is 0.05% for large orders.

        Documented in: docs/ASSUMPTIONS.md - Slippage Model section
        Reference: R9.2.3
        """
        config = BacktestConfig()
        assert config.impact_penalty_pct == 0.0005, (
            "Default impact penalty should be 0.0005 (0.05%) as documented"
        )

    def test_default_impact_threshold(self):
        """
        ASSUMPTION: Impact threshold is 10% of average volume.

        Documented in: docs/ASSUMPTIONS.md - Slippage Model section
        Reference: R9.2.3
        """
        config = BacktestConfig()
        assert config.impact_threshold == 0.1, (
            "Default impact threshold should be 0.1 (10%) as documented"
        )

    def test_slippage_direction(self):
        """
        Verify slippage direction from documentation.

        From docs:
        - BUY orders: fill_price = open_price * (1 + slippage)
        - SELL orders: fill_price = open_price * (1 - slippage)
        """
        open_price = 100.0
        slippage = 0.0001  # 0.01%

        # BUY: price goes up (worse for buyer)
        buy_fill = open_price * (1 + slippage)
        assert buy_fill == 100.01, "BUY fill should be higher than open"

        # SELL: price goes down (worse for seller)
        sell_fill = open_price * (1 - slippage)
        assert sell_fill == 99.99, "SELL fill should be lower than open"


class TestFillAssumptions:
    """Verify fill timing matches documentation (next-bar open)."""

    def test_default_fill_delay_is_one_bar(self):
        """
        ASSUMPTION: Orders fill 1 bar after signal generation.

        Documented in: docs/ASSUMPTIONS.md - Order Fill Assumptions section
        Reference: R9.1.1
        """
        # Test with simple data
        open_prices = np.array([100.0, 101.0, 102.0, 103.0])

        # Signal on bar 1, should fill at bar 2 open (default delay=1)
        fill_price, fill_bar = get_next_bar_fill_price(
            signal_bar_index=1,
            open_prices=open_prices,
            delay_bars=1,  # Default documented value
        )

        assert fill_bar == 2, "Fill should occur 1 bar after signal"
        assert fill_price == 102.0, "Fill price should be next bar open"


class TestRiskFreeRateAssumptions:
    """Verify risk-free rate matches documentation (4.0%)."""

    def test_default_risk_free_rate(self):
        """
        ASSUMPTION: Risk-free rate is 4.0% annual.

        Documented in: docs/ASSUMPTIONS.md - Risk-Free Rate section
        Reference: R9.3.1
        """
        config = BacktestConfig()
        assert config.risk_free_rate == 0.04, (
            "Default risk-free rate should be 0.04 (4%) as documented"
        )


class TestPositionSizingAssumptions:
    """Verify position sizing matches documentation."""

    def test_trend_following_risk_percent(self):
        """
        ASSUMPTION: Trend following uses 1% risk per trade.

        Documented in: docs/ASSUMPTIONS.md - Position Sizing section
        Reference: R11.1.1
        """
        settings = get_settings()
        assert settings.max_risk_pct_trend == 0.01, (
            "Trend following risk should be 0.01 (1%) as documented"
        )

    def test_mean_reversion_risk_percent(self):
        """
        ASSUMPTION: Mean reversion uses 0.75% risk per trade.

        Documented in: docs/ASSUMPTIONS.md - Position Sizing section
        Reference: R11.1.1
        """
        settings = get_settings()
        assert settings.max_risk_pct_mean_rev == 0.0075, (
            "Mean reversion risk should be 0.0075 (0.75%) as documented"
        )

    def test_max_single_position_percent(self):
        """
        ASSUMPTION: Maximum single position is 20% of account.

        Documented in: docs/ASSUMPTIONS.md - Position Sizing section
        Reference: R11.1.2
        """
        settings = get_settings()
        assert settings.max_position_size_pct == 0.20, (
            "Max single position should be 0.20 (20%) as documented"
        )

    def test_max_total_deployment_percent(self):
        """
        ASSUMPTION: Maximum total deployment is 50% of account.

        Documented in: docs/ASSUMPTIONS.md - Position Sizing section
        Reference: R11.1.2
        """
        settings = get_settings()
        assert settings.max_total_exposure_pct == 0.50, (
            "Max total deployment should be 0.50 (50%) as documented"
        )

    def test_max_open_positions(self):
        """
        ASSUMPTION: Maximum concurrent positions is 5.

        Documented in: docs/ASSUMPTIONS.md - Position Sizing section
        Reference: R11.2.2
        """
        settings = get_settings()
        assert settings.max_open_positions == 5, "Max open positions should be 5 as documented"

    def test_trend_stop_loss_atr_multiple(self):
        """
        ASSUMPTION: Trend following stop loss is 2.5x ATR.

        Documented in: docs/ASSUMPTIONS.md - Position Sizing section
        Reference: R11.1.4
        """
        settings = get_settings()
        assert settings.trend_stop_atr_multiplier == 2.5, (
            "Trend stop ATR multiple should be 2.5 as documented"
        )

    def test_mean_reversion_stop_loss_percent(self):
        """
        ASSUMPTION: Mean reversion stop loss is 3% fixed.

        Documented in: docs/ASSUMPTIONS.md - Position Sizing section
        Reference: R11.1.4
        """
        settings = get_settings()
        assert settings.mean_rev_stop_pct == 0.03, (
            "Mean reversion stop should be 0.03 (3%) as documented"
        )


class TestPositionSizingFormula:
    """Verify position sizing formula from documentation."""

    def test_risk_based_position_sizing_formula(self):
        """
        Verify the documented formula:
        shares = (account_balance * risk_pct) / |entry_price - stop_loss|

        Example:
        - Account: $100,000
        - Risk: 1% (trend following)
        - Entry: $100
        - Stop: $97.50 (2.5% below)
        - Risk amount: $1,000
        - Price risk: $2.50
        - Shares: 1000 / 2.50 = 400
        """
        account_balance = 100_000.0
        entry_price = 100.0
        stop_price = 97.50

        # Calculate using the actual function
        shares = calculate_position_size_from_risk(
            account_balance=account_balance,
            entry_price=entry_price,
            stop_loss=stop_price,
            strategy_type=StrategyType.TREND_FOLLOWING,
        )

        # Manual calculation: (100000 * 0.01) / 2.50 = 400
        expected_shares = 400.0

        assert shares == expected_shares, (
            f"Position size should be {expected_shares} shares, got {shares}"
        )


class TestCorrelationAssumptions:
    """Verify correlation handling matches documentation."""

    def test_correlation_threshold(self):
        """
        ASSUMPTION: Assets with correlation > 0.7 are considered correlated.

        Documented in: docs/ASSUMPTIONS.md - Market Assumptions section
        """
        settings = get_settings()
        assert settings.correlation_threshold == 0.7, (
            "Correlation threshold should be 0.7 as documented"
        )

    def test_correlation_size_reduction(self):
        """
        ASSUMPTION: Correlated assets get 50% position size reduction.

        Documented in: docs/ASSUMPTIONS.md - Market Assumptions section
        """
        settings = get_settings()
        assert settings.correlation_size_reduction == 0.50, (
            "Correlation size reduction should be 0.50 (50%) as documented"
        )


class TestReturnsTypeAssumption:
    """Verify returns calculation type matches documentation."""

    def test_simple_returns_used(self):
        """
        ASSUMPTION: Simple returns (not log returns) are used.

        Formula: return[t] = (equity[t] - equity[t-1]) / equity[t-1]

        Documented in: docs/ASSUMPTIONS.md - Return Calculations section
        """
        from backend.backtesting.metrics import calculate_returns

        # Test with known values
        equity = np.array([100.0, 110.0, 99.0])
        returns = calculate_returns(equity)

        # Simple returns
        expected_simple = np.array([0.10, -0.10])

        # Log returns would be different
        expected_log = np.array([np.log(110 / 100), np.log(99 / 110)])

        # Verify simple returns are used (not log)
        np.testing.assert_array_almost_equal(
            returns,
            expected_simple,
            decimal=10,
            err_msg="Should use simple returns, not log returns",
        )

        # Confirm these differ from log returns
        assert not np.allclose(returns, expected_log), "Simple and log returns should differ"
