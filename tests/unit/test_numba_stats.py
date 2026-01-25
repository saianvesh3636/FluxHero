"""
Unit tests for Numba-optimized performance metrics.

Tests cover all Tier 1 metrics with various edge cases including:
- Normal operation with realistic data
- Edge cases (empty arrays, single values, all positive/negative)
- Mathematical correctness vs manual calculations

Reference: /Users/anvesh/.claude/plans/swirling-tumbling-cloud.md
"""

import numpy as np
import pytest

from backend.analytics.numba_stats import (
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_profit_factor,
    calculate_value_at_risk,
    calculate_cvar,
    calculate_kelly_criterion,
    calculate_recovery_factor,
    calculate_ulcer_index,
    calculate_alpha_beta,
    calculate_consecutive_wins_losses,
    calculate_skewness,
    calculate_kurtosis,
    calculate_tail_ratio,
    calculate_information_ratio,
    calculate_r_squared,
    calculate_tier1_metrics,
)


class TestSortinoRatio:
    """Tests for Sortino ratio calculation."""

    def test_sortino_positive_returns(self):
        """Sortino should be positive for net positive returns."""
        returns = np.array([0.01, -0.005, 0.02, 0.015, -0.01, 0.008])
        sortino = calculate_sortino_ratio(returns, target_return=0.0)

        assert sortino > 0, "Sortino should be positive for net positive returns"

    def test_sortino_all_positive(self):
        """Sortino should be very high when all returns are positive."""
        returns = np.array([0.01, 0.02, 0.015, 0.008, 0.012])
        sortino = calculate_sortino_ratio(returns, target_return=0.0)

        # With no downside, sortino should be capped at high value
        assert sortino >= 5.0, "Sortino should be high with no downside"

    def test_sortino_all_negative(self):
        """Sortino should be negative when all returns are negative."""
        returns = np.array([-0.01, -0.02, -0.015, -0.008])
        sortino = calculate_sortino_ratio(returns, target_return=0.0)

        assert sortino < 0, "Sortino should be negative for all negative returns"

    def test_sortino_empty_array(self):
        """Sortino should return 0 for empty array."""
        returns = np.array([])
        sortino = calculate_sortino_ratio(returns)

        assert sortino == 0.0, "Sortino should be 0 for empty array"

    def test_sortino_single_value(self):
        """Sortino should return 0 for single value."""
        returns = np.array([0.01])
        sortino = calculate_sortino_ratio(returns)

        assert sortino == 0.0, "Sortino should be 0 for single value"


class TestCalmarRatio:
    """Tests for Calmar ratio calculation."""

    def test_calmar_basic(self):
        """Calmar should be positive for profitable strategy with drawdown."""
        returns = np.array([0.01, 0.02, -0.015, 0.01, 0.02])
        equity = np.array([100000, 101000, 103020, 101475, 102490, 104540])

        calmar = calculate_calmar_ratio(returns, equity)

        assert calmar > 0, "Calmar should be positive for profitable strategy"

    def test_calmar_no_drawdown(self):
        """Calmar should be high when there's no drawdown."""
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        equity = np.array([100000, 101000, 102010, 103030, 104060])

        calmar = calculate_calmar_ratio(returns, equity)

        assert calmar >= 5.0, "Calmar should be high with no drawdown"

    def test_calmar_deep_drawdown(self):
        """Calmar should be lower with deep drawdowns."""
        returns = np.array([0.02, -0.15, 0.05, 0.03])
        equity = np.array([100000, 102000, 86700, 91035, 93766])

        calmar = calculate_calmar_ratio(returns, equity)

        assert calmar < 1.0, "Calmar should be low with deep drawdown"

    def test_calmar_empty(self):
        """Calmar should return 0 for empty arrays."""
        returns = np.array([])
        equity = np.array([100000])

        calmar = calculate_calmar_ratio(returns, equity)

        assert calmar == 0.0, "Calmar should be 0 for empty data"


class TestProfitFactor:
    """Tests for profit factor calculation."""

    def test_profit_factor_profitable(self):
        """Profit factor > 1 for profitable trades."""
        pnls = np.array([100.0, -50.0, 200.0, -75.0, 150.0])
        pf = calculate_profit_factor(pnls)

        # (100 + 200 + 150) / (50 + 75) = 450 / 125 = 3.6
        expected = 450.0 / 125.0
        assert abs(pf - expected) < 0.01, f"Expected {expected}, got {pf}"

    def test_profit_factor_unprofitable(self):
        """Profit factor < 1 for unprofitable trades."""
        pnls = np.array([50.0, -100.0, 30.0, -80.0])
        pf = calculate_profit_factor(pnls)

        # (50 + 30) / (100 + 80) = 80 / 180 = 0.44
        assert pf < 1.0, "Profit factor should be < 1 for unprofitable trades"

    def test_profit_factor_no_losses(self):
        """Profit factor should be high with no losses."""
        pnls = np.array([100.0, 50.0, 200.0])
        pf = calculate_profit_factor(pnls)

        assert pf >= 10.0, "Profit factor should be high with no losses"

    def test_profit_factor_no_wins(self):
        """Profit factor should be 0 with no wins."""
        pnls = np.array([-100.0, -50.0, -200.0])
        pf = calculate_profit_factor(pnls)

        assert pf == 0.0, "Profit factor should be 0 with no wins"

    def test_profit_factor_empty(self):
        """Profit factor should be 0 for empty array."""
        pnls = np.array([])
        pf = calculate_profit_factor(pnls)

        assert pf == 0.0, "Profit factor should be 0 for empty array"


class TestValueAtRisk:
    """Tests for Value at Risk calculation."""

    def test_var_95_basic(self):
        """VaR 95% should be a negative value in left tail."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)

        var = calculate_value_at_risk(returns, 0.95)

        assert var < 0, "VaR should be negative (worst case loss)"
        assert var > -0.10, "VaR should be reasonable (not extreme)"

    def test_var_higher_confidence_more_extreme(self):
        """VaR at higher confidence should be more extreme."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)

        var_95 = calculate_value_at_risk(returns, 0.95)
        var_99 = calculate_value_at_risk(returns, 0.99)

        assert var_99 < var_95, "VaR 99% should be more extreme than VaR 95%"

    def test_var_empty(self):
        """VaR should return 0 for insufficient data."""
        returns = np.array([0.01])
        var = calculate_value_at_risk(returns)

        assert var == 0.0, "VaR should be 0 for insufficient data"


class TestCVaR:
    """Tests for Conditional Value at Risk calculation."""

    def test_cvar_more_extreme_than_var(self):
        """CVaR should be more extreme than VaR."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)

        var = calculate_value_at_risk(returns, 0.95)
        cvar = calculate_cvar(returns, 0.95)

        assert cvar <= var, "CVaR should be more extreme (more negative) than VaR"


class TestKellyCriterion:
    """Tests for Kelly Criterion calculation."""

    def test_kelly_basic(self):
        """Kelly should be positive for profitable edge."""
        win_rate = 0.55
        avg_win = 150.0
        avg_loss = 100.0

        kelly = calculate_kelly_criterion(win_rate, avg_win, avg_loss)

        # Kelly = 0.55 - (0.45 / 1.5) = 0.55 - 0.30 = 0.25
        expected = 0.25
        assert abs(kelly - expected) < 0.01, f"Expected {expected}, got {kelly}"

    def test_kelly_no_edge(self):
        """Kelly should be 0 or negative with no edge."""
        win_rate = 0.50
        avg_win = 100.0
        avg_loss = 100.0

        kelly = calculate_kelly_criterion(win_rate, avg_win, avg_loss)

        assert kelly == 0.0, "Kelly should be 0 with no edge"

    def test_kelly_invalid_inputs(self):
        """Kelly should handle invalid inputs gracefully."""
        assert calculate_kelly_criterion(0.5, 100, 0) == 0.0
        assert calculate_kelly_criterion(0, 100, 100) == 0.0
        assert calculate_kelly_criterion(1.0, 100, 100) == 0.0


class TestRecoveryFactor:
    """Tests for Recovery Factor calculation."""

    def test_recovery_basic(self):
        """Recovery factor should be return / drawdown."""
        total_return = 0.30  # 30%
        max_drawdown = -0.10  # -10%

        rf = calculate_recovery_factor(total_return, max_drawdown)

        assert abs(rf - 3.0) < 0.01, "Recovery factor should be 3.0"

    def test_recovery_no_drawdown(self):
        """Recovery factor should be high with no drawdown."""
        rf = calculate_recovery_factor(0.20, 0.0)

        assert rf >= 10.0, "Recovery factor should be high with no drawdown"


class TestUlcerIndex:
    """Tests for Ulcer Index calculation."""

    def test_ulcer_basic(self):
        """Ulcer index should be positive for equity with drawdowns."""
        equity = np.array([100, 105, 102, 98, 103, 108, 105], dtype=np.float64)
        ui = calculate_ulcer_index(equity)

        assert ui > 0, "Ulcer index should be positive"

    def test_ulcer_no_drawdown(self):
        """Ulcer index should be 0 with no drawdowns."""
        equity = np.array([100, 101, 102, 103, 104], dtype=np.float64)
        ui = calculate_ulcer_index(equity)

        assert ui == 0.0, "Ulcer index should be 0 with no drawdowns"


class TestAlphaBeta:
    """Tests for Alpha and Beta calculation."""

    def test_alpha_beta_basic(self):
        """Alpha and beta should be calculated correctly."""
        np.random.seed(42)
        benchmark = np.random.normal(0.0005, 0.01, 100)
        strategy = benchmark * 1.2 + np.random.normal(0.0002, 0.005, 100)

        alpha, beta = calculate_alpha_beta(strategy, benchmark)

        # Strategy has positive alpha (outperformance) and beta > 1 (higher sensitivity)
        assert beta > 1.0, "Beta should be > 1 for leveraged exposure"

    def test_alpha_beta_uncorrelated(self):
        """Uncorrelated returns should have beta near 0."""
        np.random.seed(42)
        strategy = np.random.normal(0.001, 0.01, 100)
        benchmark = np.random.normal(0.0005, 0.01, 100)

        alpha, beta = calculate_alpha_beta(strategy, benchmark)

        # With random uncorrelated data, beta should be close to 0
        assert abs(beta) < 1.0, "Beta should be near 0 for uncorrelated returns"


class TestConsecutiveWinsLosses:
    """Tests for consecutive wins/losses calculation."""

    def test_consecutive_basic(self):
        """Should correctly count consecutive wins and losses."""
        pnls = np.array([100, 50, -30, -20, -10, 80, 60, 40], dtype=np.float64)
        wins, losses = calculate_consecutive_wins_losses(pnls)

        assert wins == 3, f"Expected 3 consecutive wins, got {wins}"
        assert losses == 3, f"Expected 3 consecutive losses, got {losses}"

    def test_consecutive_all_wins(self):
        """All wins should give max consecutive wins = total trades."""
        pnls = np.array([100, 50, 80, 60], dtype=np.float64)
        wins, losses = calculate_consecutive_wins_losses(pnls)

        assert wins == 4, "All wins should give consecutive wins = 4"
        assert losses == 0, "All wins should give consecutive losses = 0"

    def test_consecutive_empty(self):
        """Empty array should return (0, 0)."""
        pnls = np.array([], dtype=np.float64)
        wins, losses = calculate_consecutive_wins_losses(pnls)

        assert wins == 0 and losses == 0


class TestSkewnessKurtosis:
    """Tests for skewness and kurtosis calculations."""

    def test_skewness_positive(self):
        """Returns with positive skew should have positive skewness."""
        # Create right-skewed data
        returns = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.05], dtype=np.float64)
        skew = calculate_skewness(returns)

        assert skew > 0, "Should have positive skewness"

    def test_kurtosis_normal(self):
        """Normal distribution should have excess kurtosis near 0."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 10000)
        kurt = calculate_kurtosis(returns)

        # Excess kurtosis of normal distribution is 0
        assert abs(kurt) < 0.5, "Normal distribution should have excess kurtosis near 0"


class TestTailRatio:
    """Tests for tail ratio calculation."""

    def test_tail_ratio_symmetric(self):
        """Symmetric distribution should have tail ratio near 1."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 1000)
        tail = calculate_tail_ratio(returns)

        assert 0.8 < tail < 1.2, "Symmetric distribution should have tail ratio near 1"


class TestTier1Aggregation:
    """Tests for the aggregation function."""

    def test_tier1_metrics_complete(self):
        """All Tier 1 metrics should be calculated."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)
        equity = np.zeros(253, dtype=np.float64)
        equity[0] = 100000
        for i, r in enumerate(returns):
            equity[i + 1] = equity[i] * (1 + r)

        pnls = np.random.normal(200, 500, 50)

        metrics = calculate_tier1_metrics(
            returns=returns,
            equity_curve=equity,
            pnls=pnls,
        )

        # Check all expected keys exist
        expected_keys = [
            "sortino_ratio",
            "calmar_ratio",
            "profit_factor",
            "value_at_risk_95",
            "cvar_95",
            "kelly_criterion",
            "recovery_factor",
            "ulcer_index",
            "max_consecutive_wins",
            "max_consecutive_losses",
            "skewness",
            "kurtosis",
            "tail_ratio",
            "alpha",
            "beta",
            "information_ratio",
            "r_squared",
        ]

        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"
            assert metrics[key] is not None, f"Key {key} is None"
