"""
Validation Tests for Metric Calculations.

This module validates metric calculations against hand-calculated expected values
to ensure mathematical correctness. Each test includes worked examples in comments.

Reference: FLUXHERO_REQUIREMENTS.md R9.3 - Metrics Calculation
Reference: enhancement_tasks.md Phase 24 - Quality Control & Validation Framework

Key validation approach:
1. Use simple, hand-verifiable numbers
2. Include step-by-step calculations in comments
3. Test both typical cases and edge cases
4. Validate against known formulas
"""

import numpy as np

from backend.backtesting.metrics import (
    MetricSanityError,
    PerformanceMetrics,
    calculate_annualized_return,
    calculate_avg_holding_period,
    calculate_avg_win_loss_ratio,
    calculate_max_drawdown,
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_total_return,
    calculate_win_rate,
    validate_metric_sanity,
)


class TestCalculateReturnsValidation:
    """Validate returns calculation with hand-calculated values."""

    def test_simple_returns_hand_calculated(self):
        """
        Test simple returns calculation.

        Hand calculation:
        - Equity: [100, 110, 99, 108]
        - Return[0] = (110 - 100) / 100 = 0.10 (10%)
        - Return[1] = (99 - 110) / 110 = -0.10 (exactly -10%)
        - Return[2] = (108 - 99) / 99 = 0.090909... (9.09%)
        """
        equity = np.array([100.0, 110.0, 99.0, 108.0])
        returns = calculate_returns(equity)

        assert len(returns) == 3

        # Hand-calculated expected values
        expected_return_0 = (110.0 - 100.0) / 100.0  # = 0.10
        expected_return_1 = (99.0 - 110.0) / 110.0  # = -0.1
        expected_return_2 = (108.0 - 99.0) / 99.0  # = 0.090909...

        np.testing.assert_almost_equal(returns[0], expected_return_0, decimal=10)
        np.testing.assert_almost_equal(returns[1], expected_return_1, decimal=10)
        np.testing.assert_almost_equal(returns[2], expected_return_2, decimal=6)

    def test_returns_with_round_percentages(self):
        """
        Test returns with values designed for exact percentages.

        Hand calculation:
        - Equity: [1000, 1050, 1102.5, 1047.375]
        - Return[0] = (1050 - 1000) / 1000 = 0.05 (5%)
        - Return[1] = (1102.5 - 1050) / 1050 = 0.05 (5%)
        - Return[2] = (1047.375 - 1102.5) / 1102.5 = -0.05 (-5%)
        """
        equity = np.array([1000.0, 1050.0, 1102.5, 1047.375])
        returns = calculate_returns(equity)

        np.testing.assert_almost_equal(returns[0], 0.05, decimal=10)
        np.testing.assert_almost_equal(returns[1], 0.05, decimal=10)
        np.testing.assert_almost_equal(returns[2], -0.05, decimal=10)

    def test_returns_flat_equity(self):
        """
        Test returns when equity doesn't change.

        Hand calculation:
        - Equity: [100, 100, 100, 100]
        - All returns = 0
        """
        equity = np.array([100.0, 100.0, 100.0, 100.0])
        returns = calculate_returns(equity)

        assert len(returns) == 3
        np.testing.assert_array_equal(returns, [0.0, 0.0, 0.0])

    def test_returns_single_period(self):
        """Test returns with minimal equity curve (2 points)."""
        equity = np.array([100.0, 120.0])
        returns = calculate_returns(equity)

        assert len(returns) == 1
        np.testing.assert_almost_equal(returns[0], 0.20, decimal=10)

    def test_returns_empty_and_single_value(self):
        """Test edge cases: empty array and single value."""
        # Empty array
        empty = calculate_returns(np.array([]))
        assert len(empty) == 0

        # Single value
        single = calculate_returns(np.array([100.0]))
        assert len(single) == 0


class TestCalculateSharpeRatioValidation:
    """Validate Sharpe ratio calculation with hand-calculated values."""

    def test_sharpe_ratio_hand_calculated(self):
        """
        Test Sharpe ratio with hand-calculated example.

        Hand calculation:
        - Returns: [0.01, 0.02, -0.01, 0.03, 0.01] (5 daily returns)
        - Mean return = (0.01 + 0.02 - 0.01 + 0.03 + 0.01) / 5 = 0.012
        - Variance = ((0.01-0.012)^2 + (0.02-0.012)^2 + (-0.01-0.012)^2 +
                      (0.03-0.012)^2 + (0.01-0.012)^2) / 5
                   = (0.000004 + 0.000064 + 0.000484 + 0.000324 + 0.000004) / 5
                   = 0.00088 / 5 = 0.000176
        - Std dev = sqrt(0.000176) = 0.01327...

        Annualized (252 periods):
        - Annual return = 0.012 * 252 = 3.024 = 302.4%
        - Annual std = 0.01327 * sqrt(252) = 0.2106...
        - Risk-free rate = 0.04 = 4%
        - Sharpe = (3.024 - 0.04) / 0.2106 = 14.16...
        """
        returns = np.array([0.01, 0.02, -0.01, 0.03, 0.01])
        risk_free_rate = 0.04
        periods_per_year = 252

        sharpe = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)

        # Hand-calculated components
        mean_return = np.mean(returns)  # 0.012
        std_return = np.std(returns)  # population std

        annual_return = mean_return * periods_per_year
        annual_std = std_return * np.sqrt(periods_per_year)
        expected_sharpe = (annual_return - risk_free_rate) / annual_std

        np.testing.assert_almost_equal(sharpe, expected_sharpe, decimal=10)

    def test_sharpe_ratio_known_values(self):
        """
        Test Sharpe with precisely constructed returns.

        Setup: 252 days of 0.1% daily return (25.2% annual), 2% std
        - Annual return = 0.001 * 252 = 0.252 = 25.2%
        - Annual std = 0.02 * sqrt(252) = 0.3175
        - Risk-free = 4%
        - Sharpe = (0.252 - 0.04) / 0.3175 = 0.6677
        """
        # Create 252 returns with exact mean and std
        np.random.seed(42)
        n = 252
        target_mean = 0.001  # 0.1% daily
        target_std = 0.02  # 2% daily std

        # Generate standardized normal, then scale
        z = np.random.randn(n)
        z = (z - z.mean()) / z.std()  # Exact standardization
        returns = target_mean + target_std * z

        # Verify inputs
        np.testing.assert_almost_equal(np.mean(returns), target_mean, decimal=10)
        np.testing.assert_almost_equal(np.std(returns), target_std, decimal=10)

        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.04, periods_per_year=252)

        # Expected calculation
        annual_return = target_mean * 252  # 0.252
        annual_std = target_std * np.sqrt(252)  # ~0.3175
        expected_sharpe = (annual_return - 0.04) / annual_std

        np.testing.assert_almost_equal(sharpe, expected_sharpe, decimal=6)

    def test_sharpe_ratio_zero_volatility(self):
        """
        Test Sharpe when all returns are identical (zero volatility).

        With zero std dev, Sharpe should return 0.0 to avoid division by zero.
        """
        returns = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.04, periods_per_year=252)
        assert sharpe == 0.0

    def test_sharpe_ratio_negative(self):
        """
        Test Sharpe ratio can be negative when returns < risk-free rate.

        Hand calculation:
        - Returns: [-0.001, -0.001, -0.001, -0.001] (-0.1% daily)
        - Mean = -0.001
        - Std = 0 (all same)
        - This returns 0.0 due to zero std

        Try with some variance:
        - Returns: [-0.002, -0.001, -0.001, -0.002] (avg -0.15% daily)
        - Mean = -0.0015
        - Annual return = -0.0015 * 252 = -0.378 = -37.8%
        - Sharpe = (-0.378 - 0.04) / (std * sqrt(252)) < 0
        """
        returns = np.array([-0.002, -0.001, -0.001, -0.002])
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.04, periods_per_year=252)
        assert sharpe < 0  # Negative Sharpe expected

    def test_sharpe_ratio_empty_returns(self):
        """Test Sharpe with empty returns array."""
        returns = np.array([])
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.04, periods_per_year=252)
        assert sharpe == 0.0


class TestCalculateMaxDrawdownValidation:
    """Validate max drawdown calculation with hand-calculated values."""

    def test_max_drawdown_hand_calculated(self):
        """
        Test max drawdown with hand-calculated example.

        Equity curve: [100, 120, 90, 110, 80, 100]

        Peak tracking:
        - i=0: equity=100, peak=100, dd=0%
        - i=1: equity=120, peak=120 (new peak), dd=0%
        - i=2: equity=90, peak=120, dd=(90-120)/120 = -25%
        - i=3: equity=110, peak=120, dd=(110-120)/120 = -8.33%
        - i=4: equity=80, peak=120, dd=(80-120)/120 = -33.33% (MAX!)
        - i=5: equity=100, peak=120, dd=(100-120)/120 = -16.67%

        Max DD = -33.33%, peak_idx=1 (120), trough_idx=4 (80)
        """
        equity = np.array([100.0, 120.0, 90.0, 110.0, 80.0, 100.0])
        dd_pct, peak_idx, trough_idx = calculate_max_drawdown(equity)

        expected_dd_pct = ((80.0 - 120.0) / 120.0) * 100.0  # -33.333...%
        np.testing.assert_almost_equal(dd_pct, expected_dd_pct, decimal=6)
        assert peak_idx == 1
        assert trough_idx == 4

    def test_max_drawdown_simple_case(self):
        """
        Test simple drawdown case.

        Equity: [100, 110, 105, 95, 100, 120]
        Max DD from 110 to 95 = (95-110)/110 = -13.636%
        """
        equity = np.array([100.0, 110.0, 105.0, 95.0, 100.0, 120.0])
        dd_pct, peak_idx, trough_idx = calculate_max_drawdown(equity)

        expected_dd_pct = ((95.0 - 110.0) / 110.0) * 100.0  # -13.636...%
        np.testing.assert_almost_equal(dd_pct, expected_dd_pct, decimal=3)
        assert peak_idx == 1
        assert trough_idx == 3

    def test_max_drawdown_no_drawdown(self):
        """
        Test when equity only increases (no drawdown).

        Equity: [100, 110, 120, 130]
        Each point is a new peak, so drawdown is always 0%.
        """
        equity = np.array([100.0, 110.0, 120.0, 130.0])
        dd_pct, peak_idx, trough_idx = calculate_max_drawdown(equity)

        assert dd_pct == 0.0
        # When no drawdown, indices point to start
        assert peak_idx == 0
        assert trough_idx == 0

    def test_max_drawdown_monotonic_decrease(self):
        """
        Test when equity only decreases.

        Equity: [100, 90, 80, 70]
        First point is peak, max DD at end: (70-100)/100 = -30%
        """
        equity = np.array([100.0, 90.0, 80.0, 70.0])
        dd_pct, peak_idx, trough_idx = calculate_max_drawdown(equity)

        expected_dd_pct = ((70.0 - 100.0) / 100.0) * 100.0  # -30%
        np.testing.assert_almost_equal(dd_pct, expected_dd_pct, decimal=10)
        assert peak_idx == 0
        assert trough_idx == 3

    def test_max_drawdown_multiple_drawdowns(self):
        """
        Test with multiple drawdown periods to find the maximum.

        Equity: [100, 110, 100, 120, 90, 130]

        Drawdown analysis:
        - First DD: 100/110 = -9.09%
        - Second DD: 90/120 = -25% (MAX)

        Max DD = -25%, peak_idx=3 (120), trough_idx=4 (90)
        """
        equity = np.array([100.0, 110.0, 100.0, 120.0, 90.0, 130.0])
        dd_pct, peak_idx, trough_idx = calculate_max_drawdown(equity)

        expected_dd_pct = ((90.0 - 120.0) / 120.0) * 100.0  # -25%
        np.testing.assert_almost_equal(dd_pct, expected_dd_pct, decimal=10)
        assert peak_idx == 3
        assert trough_idx == 4

    def test_max_drawdown_empty(self):
        """Test with empty equity curve."""
        dd_pct, peak_idx, trough_idx = calculate_max_drawdown(np.array([]))
        assert dd_pct == 0.0
        assert peak_idx == -1
        assert trough_idx == -1


class TestCalculateWinRateValidation:
    """Validate win rate calculation with hand-calculated values."""

    def test_win_rate_hand_calculated(self):
        """
        Test win rate with hand-calculated values.

        PnLs: [100, -50, 200, -30, 150]
        Wins (pnl > 0): 100, 200, 150 = 3 trades
        Losses (pnl <= 0): -50, -30 = 2 trades
        Win rate = 3 / 5 = 0.6 = 60%
        """
        pnls = np.array([100.0, -50.0, 200.0, -30.0, 150.0])
        win_rate = calculate_win_rate(pnls)
        assert win_rate == 0.6

    def test_win_rate_all_wins(self):
        """
        Test win rate when all trades are winners.

        PnLs: [50, 100, 25, 75]
        All 4 are wins, win rate = 4/4 = 1.0 = 100%
        """
        pnls = np.array([50.0, 100.0, 25.0, 75.0])
        win_rate = calculate_win_rate(pnls)
        assert win_rate == 1.0

    def test_win_rate_all_losses(self):
        """
        Test win rate when all trades are losses.

        PnLs: [-50, -100, -25, -75]
        All 4 are losses, win rate = 0/4 = 0.0 = 0%
        """
        pnls = np.array([-50.0, -100.0, -25.0, -75.0])
        win_rate = calculate_win_rate(pnls)
        assert win_rate == 0.0

    def test_win_rate_breakeven_is_loss(self):
        """
        Test that breakeven (pnl = 0) is counted as a loss.

        PnLs: [100, 0, -50]
        Wins: 100 (1 trade)
        Not wins: 0, -50 (2 trades)
        Win rate = 1/3 = 0.333...
        """
        pnls = np.array([100.0, 0.0, -50.0])
        win_rate = calculate_win_rate(pnls)
        np.testing.assert_almost_equal(win_rate, 1.0 / 3.0, decimal=10)

    def test_win_rate_empty(self):
        """Test win rate with no trades."""
        pnls = np.array([])
        win_rate = calculate_win_rate(pnls)
        assert win_rate == 0.0

    def test_win_rate_single_trade(self):
        """Test win rate with single winning and losing trade."""
        assert calculate_win_rate(np.array([100.0])) == 1.0
        assert calculate_win_rate(np.array([-100.0])) == 0.0


class TestCalculateAvgWinLossRatioValidation:
    """Validate average win/loss ratio with hand-calculated values."""

    def test_avg_win_loss_ratio_hand_calculated(self):
        """
        Test avg win/loss ratio with hand-calculated values.

        PnLs: [100, -50, 200, -30, 150]
        Wins: 100, 200, 150 → avg = (100 + 200 + 150) / 3 = 150
        Losses: |-50|, |-30| → avg = (50 + 30) / 2 = 40
        Ratio = 150 / 40 = 3.75
        """
        pnls = np.array([100.0, -50.0, 200.0, -30.0, 150.0])
        ratio = calculate_avg_win_loss_ratio(pnls)
        np.testing.assert_almost_equal(ratio, 3.75, decimal=10)

    def test_avg_win_loss_ratio_equal_averages(self):
        """
        Test when avg win equals avg loss.

        PnLs: [100, -100]
        Avg win = 100, Avg loss = 100
        Ratio = 100 / 100 = 1.0
        """
        pnls = np.array([100.0, -100.0])
        ratio = calculate_avg_win_loss_ratio(pnls)
        assert ratio == 1.0

    def test_avg_win_loss_ratio_no_losses(self):
        """
        Test when there are no losses (all wins).

        Returns 0.0 because division would be by zero.
        """
        pnls = np.array([100.0, 200.0, 50.0])
        ratio = calculate_avg_win_loss_ratio(pnls)
        assert ratio == 0.0

    def test_avg_win_loss_ratio_no_wins(self):
        """
        Test when there are no wins (all losses).

        Returns 0.0 because there's no average win to calculate.
        """
        pnls = np.array([-100.0, -200.0, -50.0])
        ratio = calculate_avg_win_loss_ratio(pnls)
        assert ratio == 0.0

    def test_avg_win_loss_ratio_precise_calculation(self):
        """
        Test with values designed for precise ratio.

        PnLs: [200, -100, 300, -50]
        Wins: 200, 300 → avg = 250
        Losses: 100, 50 → avg = 75
        Ratio = 250 / 75 = 3.333...
        """
        pnls = np.array([200.0, -100.0, 300.0, -50.0])
        ratio = calculate_avg_win_loss_ratio(pnls)
        expected = 250.0 / 75.0
        np.testing.assert_almost_equal(ratio, expected, decimal=10)

    def test_avg_win_loss_ratio_empty(self):
        """Test with no trades."""
        pnls = np.array([])
        ratio = calculate_avg_win_loss_ratio(pnls)
        assert ratio == 0.0


class TestCalculateTotalReturnValidation:
    """Validate total return calculation with hand-calculated values."""

    def test_total_return_hand_calculated(self):
        """
        Test total return calculation.

        Initial: 100,000
        Final: 125,000
        Return = 125,000 - 100,000 = 25,000
        Percent = (25,000 / 100,000) * 100 = 25%
        """
        total_ret, pct_ret = calculate_total_return(100000.0, 125000.0)
        assert total_ret == 25000.0
        assert pct_ret == 25.0

    def test_total_return_loss(self):
        """
        Test total return with loss.

        Initial: 100,000
        Final: 80,000
        Return = 80,000 - 100,000 = -20,000
        Percent = (-20,000 / 100,000) * 100 = -20%
        """
        total_ret, pct_ret = calculate_total_return(100000.0, 80000.0)
        assert total_ret == -20000.0
        assert pct_ret == -20.0

    def test_total_return_breakeven(self):
        """
        Test total return with no change.

        Initial: 100,000
        Final: 100,000
        Return = 0
        Percent = 0%
        """
        total_ret, pct_ret = calculate_total_return(100000.0, 100000.0)
        assert total_ret == 0.0
        assert pct_ret == 0.0

    def test_total_return_double(self):
        """
        Test total return when equity doubles.

        Initial: 50,000
        Final: 100,000
        Return = 50,000
        Percent = 100%
        """
        total_ret, pct_ret = calculate_total_return(50000.0, 100000.0)
        assert total_ret == 50000.0
        assert pct_ret == 100.0


class TestCalculateAnnualizedReturnValidation:
    """Validate annualized return (CAGR) calculation with hand-calculated values."""

    def test_annualized_return_hand_calculated(self):
        """
        Test CAGR calculation.

        Total return: 21% over 365 days (1 year)
        CAGR = (1 + 0.21)^(365/365) - 1 = 0.21 = 21%
        """
        ann_ret = calculate_annualized_return(21.0, 365)
        np.testing.assert_almost_equal(ann_ret, 21.0, decimal=10)

    def test_annualized_return_half_year(self):
        """
        Test CAGR for half-year period.

        Total return: 10% over 182.5 days
        CAGR = (1 + 0.10)^(365/182.5) - 1 = 1.10^2 - 1 = 0.21 = 21%
        """
        ann_ret = calculate_annualized_return(10.0, 182)
        # (1.10)^(365/182) - 1 = 1.10^2.0055 - 1 ≈ 0.211
        expected = ((1.0 + 0.10) ** (365.0 / 182.0) - 1.0) * 100.0
        np.testing.assert_almost_equal(ann_ret, expected, decimal=6)

    def test_annualized_return_two_years(self):
        """
        Test CAGR for two-year period.

        Total return: 44% over 730 days (2 years)
        CAGR = (1 + 0.44)^(365/730) - 1 = 1.44^0.5 - 1 = 0.2 = 20%
        """
        ann_ret = calculate_annualized_return(44.0, 730)
        expected = ((1.0 + 0.44) ** (365.0 / 730.0) - 1.0) * 100.0  # 20%
        np.testing.assert_almost_equal(ann_ret, expected, decimal=6)

    def test_annualized_return_negative(self):
        """
        Test CAGR with negative return.

        Total return: -19% over 365 days
        CAGR = (1 - 0.19)^1 - 1 = -0.19 = -19%
        """
        ann_ret = calculate_annualized_return(-19.0, 365)
        np.testing.assert_almost_equal(ann_ret, -19.0, decimal=10)

    def test_annualized_return_zero_days(self):
        """Test CAGR with zero days returns 0."""
        ann_ret = calculate_annualized_return(25.0, 0)
        assert ann_ret == 0.0


class TestCalculateAvgHoldingPeriodValidation:
    """Validate average holding period calculation."""

    def test_avg_holding_period_hand_calculated(self):
        """
        Test average holding period.

        Holding periods: [5, 10, 3, 8, 12] bars
        Average = (5 + 10 + 3 + 8 + 12) / 5 = 38 / 5 = 7.6 bars
        """
        periods = np.array([5, 10, 3, 8, 12], dtype=np.int32)
        avg = calculate_avg_holding_period(periods)
        assert avg == 7.6

    def test_avg_holding_period_single(self):
        """Test with single holding period."""
        periods = np.array([10], dtype=np.int32)
        avg = calculate_avg_holding_period(periods)
        assert avg == 10.0

    def test_avg_holding_period_empty(self):
        """Test with no trades."""
        periods = np.array([], dtype=np.int32)
        avg = calculate_avg_holding_period(periods)
        assert avg == 0.0


class TestPerformanceMetricsIntegrationValidation:
    """Validate PerformanceMetrics.calculate_all_metrics with complete example."""

    def test_calculate_all_metrics_hand_verified(self):
        """
        Test full metrics calculation with hand-verified inputs.

        Setup:
        - Initial capital: 10,000
        - Equity curve: [10000, 10500, 10200, 10800, 10300, 11000]
        - Trades PnL: [500, -300, 600, -500, 700] (corresponding to equity changes)
        - Holding periods: [10, 5, 15, 8, 12] bars

        Expected results (hand-calculated):
        - Total return: 11,000 - 10,000 = 1,000 (10%)
        - Final equity: 11,000
        - Win rate: 3 wins / 5 trades = 0.6
        - Avg win: (500 + 600 + 700) / 3 = 600
        - Avg loss: (300 + 500) / 2 = 400
        - Avg win/loss ratio: 600 / 400 = 1.5
        - Avg holding period: (10 + 5 + 15 + 8 + 12) / 5 = 10 bars
        """
        equity_curve = np.array([10000.0, 10500.0, 10200.0, 10800.0, 10300.0, 11000.0])
        trades_pnl = np.array([500.0, -300.0, 600.0, -500.0, 700.0])
        trades_holding_periods = np.array([10, 5, 15, 8, 12], dtype=np.int32)
        initial_capital = 10000.0

        metrics = PerformanceMetrics.calculate_all_metrics(
            equity_curve=equity_curve,
            trades_pnl=trades_pnl,
            trades_holding_periods=trades_holding_periods,
            initial_capital=initial_capital,
            risk_free_rate=0.04,
            periods_per_year=252,
        )

        # Validate return metrics
        assert metrics["initial_capital"] == 10000.0
        assert metrics["final_equity"] == 11000.0
        assert metrics["total_return"] == 1000.0
        assert metrics["total_return_pct"] == 10.0

        # Validate trade statistics
        assert metrics["total_trades"] == 5
        assert metrics["winning_trades"] == 3
        assert metrics["losing_trades"] == 2
        np.testing.assert_almost_equal(metrics["win_rate"], 0.6, decimal=10)
        np.testing.assert_almost_equal(metrics["avg_win"], 600.0, decimal=10)
        np.testing.assert_almost_equal(metrics["avg_loss"], 400.0, decimal=10)
        np.testing.assert_almost_equal(metrics["avg_win_loss_ratio"], 1.5, decimal=10)
        np.testing.assert_almost_equal(metrics["avg_holding_period"], 10.0, decimal=10)

    def test_check_success_criteria_passing(self):
        """Test success criteria with passing metrics."""
        good_metrics = {
            "sharpe_ratio": 1.2,  # > 0.8 ✓
            "max_drawdown_pct": -15.0,  # > -25% ✓
            "win_rate": 0.55,  # > 0.45 ✓
            "avg_win_loss_ratio": 2.0,  # > 1.5 ✓
        }
        criteria = PerformanceMetrics.check_success_criteria(good_metrics)

        assert criteria["sharpe_ratio_ok"] is True
        assert criteria["max_drawdown_ok"] is True
        assert criteria["win_rate_ok"] is True
        assert criteria["win_loss_ratio_ok"] is True
        assert criteria["all_criteria_met"] is True

    def test_check_success_criteria_failing(self):
        """Test success criteria with failing metrics."""
        bad_metrics = {
            "sharpe_ratio": 0.5,  # < 0.8 ✗
            "max_drawdown_pct": -30.0,  # < -25% ✗
            "win_rate": 0.40,  # < 0.45 ✗
            "avg_win_loss_ratio": 1.2,  # < 1.5 ✗
        }
        criteria = PerformanceMetrics.check_success_criteria(bad_metrics)

        assert criteria["sharpe_ratio_ok"] is False
        assert criteria["max_drawdown_ok"] is False
        assert criteria["win_rate_ok"] is False
        assert criteria["win_loss_ratio_ok"] is False
        assert criteria["all_criteria_met"] is False

    def test_check_success_criteria_edge_values(self):
        """Test success criteria at exact boundary values."""
        # Exactly at thresholds (should fail since > not >=)
        edge_metrics = {
            "sharpe_ratio": 0.8,  # not > 0.8
            "max_drawdown_pct": -25.0,  # not > -25%
            "win_rate": 0.45,  # not > 0.45
            "avg_win_loss_ratio": 1.5,  # not > 1.5
        }
        criteria = PerformanceMetrics.check_success_criteria(edge_metrics)
        assert criteria["all_criteria_met"] is False

        # Just above thresholds (should pass)
        above_metrics = {
            "sharpe_ratio": 0.81,
            "max_drawdown_pct": -24.9,
            "win_rate": 0.46,
            "avg_win_loss_ratio": 1.51,
        }
        criteria = PerformanceMetrics.check_success_criteria(above_metrics)
        assert criteria["all_criteria_met"] is True


class TestMetricSanityChecksValidation:
    """Validate metric sanity checks catch invalid values and log appropriately."""

    def test_sanity_check_valid_metrics_pass(self):
        """
        Test that valid metrics pass all sanity checks.

        Valid metrics have:
        - Sharpe ratio in [-5, +5]
        - Win rate in [0, 1]
        - Max drawdown in [-100, 0]
        - Non-negative ratios and counts
        """

        valid_metrics = {
            "sharpe_ratio": 1.5,
            "win_rate": 0.55,
            "max_drawdown_pct": -15.0,
            "avg_win_loss_ratio": 2.0,
            "total_trades": 100,
            "winning_trades": 55,
            "losing_trades": 45,
            "initial_capital": 10000.0,
            "final_equity": 11000.0,
            "total_return": 1000.0,
            "annualized_return_pct": 25.0,
            "avg_holding_period": 5.0,
        }
        violations = validate_metric_sanity(valid_metrics)
        assert len(violations) == 0

    def test_sanity_check_extreme_sharpe_ratio(self):
        """
        Test that extreme Sharpe ratios trigger warnings.

        Sharpe ratios outside [-5, +5] are highly unusual and warrant investigation.
        A Sharpe of 10 would indicate 10 standard deviations of excess return,
        which is extremely unlikely in real trading.
        """

        # Extremely high Sharpe (suspicious)
        high_sharpe_metrics = _create_valid_metrics()
        high_sharpe_metrics["sharpe_ratio"] = 10.0
        violations = validate_metric_sanity(high_sharpe_metrics)
        assert any("Extreme Sharpe ratio" in v for v in violations)

        # Extremely low Sharpe (suspicious)
        low_sharpe_metrics = _create_valid_metrics()
        low_sharpe_metrics["sharpe_ratio"] = -8.0
        violations = validate_metric_sanity(low_sharpe_metrics)
        assert any("Extreme Sharpe ratio" in v for v in violations)

    def test_sanity_check_invalid_win_rate(self):
        """
        Test that invalid win rates are caught as critical errors.

        Win rate must be a valid probability in [0, 1].
        Values outside this range are mathematically impossible.
        """

        # Win rate > 1 (impossible)
        high_win_rate = _create_valid_metrics()
        high_win_rate["win_rate"] = 1.5
        violations = validate_metric_sanity(high_win_rate)
        assert any("Invalid win rate" in v for v in violations)

        # Win rate < 0 (impossible)
        low_win_rate = _create_valid_metrics()
        low_win_rate["win_rate"] = -0.2
        violations = validate_metric_sanity(low_win_rate)
        assert any("Invalid win rate" in v for v in violations)

        # Should raise when raise_on_critical=True
        import pytest

        with pytest.raises(MetricSanityError):
            validate_metric_sanity(high_win_rate, raise_on_critical=True)

    def test_sanity_check_invalid_max_drawdown(self):
        """
        Test that invalid max drawdown values are caught.

        Max drawdown must be <= 0 (it's a loss measure)
        and >= -100% (can't lose more than 100% of capital)
        """

        # Positive drawdown (impossible - drawdown is always negative or zero)
        positive_dd = _create_valid_metrics()
        positive_dd["max_drawdown_pct"] = 5.0
        violations = validate_metric_sanity(positive_dd)
        assert any("Invalid max drawdown" in v and "<= 0" in v for v in violations)

        # Drawdown exceeding -100% (impossible)
        huge_dd = _create_valid_metrics()
        huge_dd["max_drawdown_pct"] = -120.0
        violations = validate_metric_sanity(huge_dd)
        assert any("Invalid max drawdown" in v and "exceed -100%" in v for v in violations)

        # Should raise when raise_on_critical=True
        import pytest

        with pytest.raises(MetricSanityError):
            validate_metric_sanity(positive_dd, raise_on_critical=True)

    def test_sanity_check_very_large_drawdown_warning(self):
        """
        Test that very large (but valid) drawdowns trigger warnings.

        Drawdowns exceeding -50% are unusual and may indicate problems,
        though they are technically possible.
        """

        large_dd = _create_valid_metrics()
        large_dd["max_drawdown_pct"] = -75.0
        violations = validate_metric_sanity(large_dd)
        assert any("Very large max drawdown" in v for v in violations)

    def test_sanity_check_negative_ratios(self):
        """
        Test that negative win/loss ratios are caught.

        Avg win/loss ratio must be >= 0 (can't have negative ratio).
        """

        import pytest

        negative_ratio = _create_valid_metrics()
        negative_ratio["avg_win_loss_ratio"] = -1.0
        violations = validate_metric_sanity(negative_ratio)
        assert any("Invalid avg win/loss ratio" in v for v in violations)

        with pytest.raises(MetricSanityError):
            validate_metric_sanity(negative_ratio, raise_on_critical=True)

    def test_sanity_check_negative_trade_count(self):
        """
        Test that negative trade counts are caught.

        Trade counts must be >= 0 (can't have negative trades).
        """

        import pytest

        negative_trades = _create_valid_metrics()
        negative_trades["total_trades"] = -5
        violations = validate_metric_sanity(negative_trades)
        assert any("Invalid total trades" in v for v in violations)

        with pytest.raises(MetricSanityError):
            validate_metric_sanity(negative_trades, raise_on_critical=True)

    def test_sanity_check_trade_count_mismatch(self):
        """
        Test that trade count mismatches are caught.

        winning_trades + losing_trades must equal total_trades.
        """

        import pytest

        mismatch = _create_valid_metrics()
        mismatch["total_trades"] = 100
        mismatch["winning_trades"] = 60
        mismatch["losing_trades"] = 30  # Should be 40
        violations = validate_metric_sanity(mismatch)
        assert any("Trade count mismatch" in v for v in violations)

        with pytest.raises(MetricSanityError):
            validate_metric_sanity(mismatch, raise_on_critical=True)

    def test_sanity_check_equity_return_mismatch(self):
        """
        Test that equity/return mismatches are caught.

        final_equity should equal initial_capital + total_return.
        """

        import pytest

        mismatch = _create_valid_metrics()
        mismatch["initial_capital"] = 10000.0
        mismatch["final_equity"] = 12000.0
        mismatch["total_return"] = 1000.0  # Should be 2000.0
        violations = validate_metric_sanity(mismatch)
        assert any("Equity/return mismatch" in v for v in violations)

        with pytest.raises(MetricSanityError):
            validate_metric_sanity(mismatch, raise_on_critical=True)

    def test_sanity_check_extreme_annualized_return(self):
        """
        Test that extreme annualized returns trigger warnings.

        Returns > 500% or < -100% are unusual for real strategies.
        """

        high_return = _create_valid_metrics()
        high_return["annualized_return_pct"] = 600.0
        violations = validate_metric_sanity(high_return)
        assert any("Extreme annualized return" in v for v in violations)

        # Very negative (total wipeout and then some)
        low_return = _create_valid_metrics()
        low_return["annualized_return_pct"] = -150.0
        violations = validate_metric_sanity(low_return)
        assert any("Extreme annualized return" in v for v in violations)

    def test_sanity_check_negative_holding_period(self):
        """
        Test that negative holding periods are caught.

        Average holding period must be >= 0.
        """

        import pytest

        negative_holding = _create_valid_metrics()
        negative_holding["avg_holding_period"] = -5.0
        violations = validate_metric_sanity(negative_holding)
        assert any("Invalid avg holding period" in v for v in violations)

        with pytest.raises(MetricSanityError):
            validate_metric_sanity(negative_holding, raise_on_critical=True)

    def test_calculate_all_metrics_runs_sanity_checks(self):
        """
        Test that calculate_all_metrics runs sanity checks by default.

        With valid inputs, no warnings should be logged.
        """
        equity_curve = np.array([10000.0, 10500.0, 10200.0, 10800.0, 10300.0, 11000.0])
        trades_pnl = np.array([500.0, -300.0, 600.0, -500.0, 700.0])
        trades_holding_periods = np.array([10, 5, 15, 8, 12], dtype=np.int32)

        # This should not raise - all metrics should be valid
        metrics = PerformanceMetrics.calculate_all_metrics(
            equity_curve=equity_curve,
            trades_pnl=trades_pnl,
            trades_holding_periods=trades_holding_periods,
            initial_capital=10000.0,
            enable_sanity_checks=True,
        )

        # Verify metrics were calculated
        assert metrics["total_trades"] == 5
        assert metrics["win_rate"] == 0.6

    def test_calculate_all_metrics_sanity_checks_can_be_disabled(self):
        """Test that sanity checks can be disabled."""
        equity_curve = np.array([10000.0, 11000.0])
        trades_pnl = np.array([1000.0])
        trades_holding_periods = np.array([10], dtype=np.int32)

        # Should work with sanity checks disabled
        metrics = PerformanceMetrics.calculate_all_metrics(
            equity_curve=equity_curve,
            trades_pnl=trades_pnl,
            trades_holding_periods=trades_holding_periods,
            initial_capital=10000.0,
            enable_sanity_checks=False,
        )
        assert metrics["total_return"] == 1000.0


def _create_valid_metrics() -> dict:
    """Create a valid metrics dictionary for testing."""
    return {
        "sharpe_ratio": 1.5,
        "win_rate": 0.55,
        "max_drawdown_pct": -15.0,
        "avg_win_loss_ratio": 2.0,
        "total_trades": 100,
        "winning_trades": 55,
        "losing_trades": 45,
        "initial_capital": 10000.0,
        "final_equity": 11000.0,
        "total_return": 1000.0,
        "annualized_return_pct": 25.0,
        "avg_holding_period": 5.0,
    }
