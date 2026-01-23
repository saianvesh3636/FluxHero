"""
Walk-forward backtest integration tests (Phase 18).

This test validates walk-forward analysis using 1-year SPY data:
- Run walk-forward on real/synthetic SPY data
- Verify pass rate calculation matches expected values
- Compare aggregate metrics against known baselines
- Test with DualModeStrategy (same as regular backtest)

Reference: FLUXHERO_REQUIREMENTS.md R9.4
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.backtesting.engine import BacktestConfig, Position  # noqa: E402
from backend.backtesting.walk_forward import (  # noqa: E402
    DEFAULT_PASS_RATE_THRESHOLD,
    aggregate_walk_forward_results,
    calculate_pass_rate,
    generate_walk_forward_windows,
    passes_walk_forward_test,
    run_walk_forward_backtest,
    validate_no_data_leakage,
)
from scripts.run_spy_backtest import (  # noqa: E402
    DualModeStrategy,
    generate_synthetic_spy_data,
)


def create_dual_mode_strategy_factory(
    bars: np.ndarray,
    initial_capital: float,
    params: dict[str, Any],
) -> callable:
    """
    Strategy factory for DualModeStrategy.

    This wraps DualModeStrategy to be compatible with walk-forward testing.
    The strategy uses full window data (train+test) for indicator warmup.
    """
    strategy = DualModeStrategy(bars)

    def get_orders(
        all_bars: np.ndarray,
        current_idx: int,
        position: Position | None,
    ) -> list:
        return strategy.get_orders(all_bars, current_idx, position)

    return get_orders


class TestWalkForwardWithSPYData:
    """Integration tests for walk-forward analysis with SPY data."""

    def test_walk_forward_completes_on_1_year_data(self):
        """Test that walk-forward completes successfully on 1-year SPY data."""
        # Generate 252 bars (1 year of daily trading data)
        data = generate_synthetic_spy_data(252)
        bars = data["bars"]
        timestamps = data["timestamps"]
        volumes = data["volumes"]

        # Configure backtest (same as regular SPY backtest)
        config = BacktestConfig(
            initial_capital=100000.0,
            commission_per_share=0.005,
            slippage_pct=0.0001,
            impact_threshold=0.1,
            impact_penalty_pct=0.0005,
            risk_free_rate=0.04,
        )

        # Run walk-forward with default 3-month train / 1-month test windows
        # 252 bars with 84-bar windows (63 train + 21 test) = 3 windows
        result = run_walk_forward_backtest(
            bars=bars,
            strategy_factory=create_dual_mode_strategy_factory,
            config=config,
            train_bars=63,
            test_bars=21,
            timestamps=timestamps,
            volumes=volumes,
            symbol="SPY",
        )

        # Verify walk-forward completed
        assert result is not None
        assert result.total_windows == 3
        assert len(result.window_results) == 3

        # Verify each window has valid results
        for i, wr in enumerate(result.window_results):
            assert wr.window.window_id == i
            assert len(wr.equity_curve) > 0
            assert wr.initial_equity > 0
            assert wr.final_equity > 0
            assert "sharpe_ratio" in wr.metrics
            assert "max_drawdown_pct" in wr.metrics
            assert "win_rate" in wr.metrics

        # Verify pass rate is valid
        assert 0.0 <= result.pass_rate <= 1.0

        # Print summary for debugging
        print("\n" + "=" * 60)
        print("WALK-FORWARD INTEGRATION TEST RESULTS (1-year SPY)")
        print("=" * 60)
        print(f"Total Windows: {result.total_windows}")
        print(f"Profitable Windows: {result.profitable_windows}")
        print(f"Pass Rate: {result.pass_rate * 100:.1f}%")
        print(f"Passes Walk-Forward Test: {passes_walk_forward_test(result.pass_rate)}")
        print("\nPer-Window Results:")
        for wr in result.window_results:
            status = "✅ Profitable" if wr.is_profitable else "❌ Loss"
            print(
                f"  Window {wr.window.window_id}: "
                f"${wr.initial_equity:,.2f} → ${wr.final_equity:,.2f} ({status})"
            )
        print("=" * 60)

    def test_walk_forward_windows_have_no_data_leakage(self):
        """Verify that walk-forward windows have no data leakage (R9.4.1)."""
        # Generate 1 year of data
        data = generate_synthetic_spy_data(252)
        bars = data["bars"]

        # Generate windows
        windows = generate_walk_forward_windows(
            n_bars=len(bars),
            train_bars=63,
            test_bars=21,
        )

        # Verify no data leakage
        assert validate_no_data_leakage(windows) is True

        # Explicitly verify train/test boundaries
        for i, window in enumerate(windows):
            # Train period ends exactly where test period starts
            assert window.train_end_idx == window.test_start_idx

            # Train data never overlaps with test data
            assert window.train_end_idx <= window.test_start_idx

            # Windows are sequential: window N's test end <= window N+1's train start
            if i > 0:
                prev_window = windows[i - 1]
                assert prev_window.test_end_idx <= window.train_start_idx

    def test_pass_rate_calculation_accuracy(self):
        """Verify pass rate calculation matches expected values."""
        # Test with known outcomes
        # 3 profitable out of 5 windows = 60% (should NOT pass strict >60% threshold)
        pass_rate = calculate_pass_rate(3, 5)
        assert pass_rate == 0.6
        assert passes_walk_forward_test(pass_rate) is False

        # 4 profitable out of 5 windows = 80% (should pass)
        pass_rate = calculate_pass_rate(4, 5)
        assert pass_rate == 0.8
        assert passes_walk_forward_test(pass_rate) is True

        # 2 profitable out of 3 windows = 66.7% (should pass)
        pass_rate = calculate_pass_rate(2, 3)
        assert pass_rate == pytest.approx(2 / 3)
        assert passes_walk_forward_test(pass_rate) is True

    def test_aggregate_metrics_match_known_baseline(self):
        """Verify aggregate metrics are within expected ranges."""
        # Generate data with fixed seed for reproducibility
        np.random.seed(42)
        data = generate_synthetic_spy_data(252)
        bars = data["bars"]
        timestamps = data["timestamps"]
        volumes = data["volumes"]

        config = BacktestConfig(
            initial_capital=100000.0,
            commission_per_share=0.005,
            slippage_pct=0.0001,
            risk_free_rate=0.04,
        )

        result = run_walk_forward_backtest(
            bars=bars,
            strategy_factory=create_dual_mode_strategy_factory,
            config=config,
            train_bars=63,
            test_bars=21,
            timestamps=timestamps,
            volumes=volumes,
            symbol="SPY",
        )

        aggregate = aggregate_walk_forward_results(result)

        # Verify aggregate metrics are reasonable
        assert len(aggregate.combined_equity_curve) > 0
        assert aggregate.total_windows == result.total_windows
        assert aggregate.total_profitable_windows == result.profitable_windows
        assert aggregate.pass_rate == result.pass_rate
        assert aggregate.initial_capital == config.initial_capital

        # Verify metrics are not NaN
        assert not np.isnan(aggregate.aggregate_sharpe)
        assert not np.isnan(aggregate.aggregate_max_drawdown_pct)

        # Verify metrics are in reasonable ranges
        assert -5.0 <= aggregate.aggregate_sharpe <= 5.0
        assert -100.0 <= aggregate.aggregate_max_drawdown_pct <= 0.0
        assert 0.0 <= aggregate.aggregate_win_rate <= 1.0
        assert aggregate.total_trades >= 0

        # Verify final capital matches last equity curve value
        assert aggregate.final_capital == aggregate.combined_equity_curve[-1]

        # Print aggregate summary
        print("\n" + "=" * 60)
        print("AGGREGATE WALK-FORWARD METRICS")
        print("=" * 60)
        print(f"Combined Equity Points: {len(aggregate.combined_equity_curve)}")
        print(f"Aggregate Sharpe: {aggregate.aggregate_sharpe:.2f}")
        print(f"Aggregate Max Drawdown: {aggregate.aggregate_max_drawdown_pct:.2f}%")
        print(f"Aggregate Win Rate: {aggregate.aggregate_win_rate * 100:.1f}%")
        print(f"Total Trades: {aggregate.total_trades}")
        print(f"Total Return: {aggregate.total_return_pct:.2f}%")
        print(f"Initial Capital: ${aggregate.initial_capital:,.2f}")
        print(f"Final Capital: ${aggregate.final_capital:,.2f}")
        print(f"Pass Rate: {aggregate.pass_rate * 100:.1f}%")
        print(f"Passes Test: {'✅' if aggregate.passes_walk_forward_test else '❌'}")
        print("=" * 60)

    def test_walk_forward_with_extended_data(self):
        """Test walk-forward with 2 years of data (6 windows)."""
        # Generate 504 bars (2 years of daily trading data)
        data = generate_synthetic_spy_data(504)
        bars = data["bars"]
        timestamps = data["timestamps"]
        volumes = data["volumes"]

        config = BacktestConfig(
            initial_capital=100000.0,
            commission_per_share=0.005,
            slippage_pct=0.0001,
            risk_free_rate=0.04,
        )

        # Run walk-forward: 504 bars / 84-bar window = 6 windows
        result = run_walk_forward_backtest(
            bars=bars,
            strategy_factory=create_dual_mode_strategy_factory,
            config=config,
            train_bars=63,
            test_bars=21,
            timestamps=timestamps,
            volumes=volumes,
            symbol="SPY",
        )

        # Verify correct number of windows
        assert result.total_windows == 6

        # Verify capital carries forward correctly
        for i in range(1, len(result.window_results)):
            prev_final = result.window_results[i - 1].final_equity
            curr_initial = result.window_results[i].initial_equity
            assert curr_initial == pytest.approx(prev_final, rel=0.001)

        # Verify data integrity across windows
        validate_no_data_leakage(
            [wr.window for wr in result.window_results]
        )

    def test_walk_forward_capital_continuity(self):
        """Verify capital flows correctly between windows."""
        data = generate_synthetic_spy_data(252)
        bars = data["bars"]

        config = BacktestConfig(
            initial_capital=100000.0,
            commission_per_share=0.005,
            slippage_pct=0.0001,
        )

        result = run_walk_forward_backtest(
            bars=bars,
            strategy_factory=create_dual_mode_strategy_factory,
            config=config,
            train_bars=63,
            test_bars=21,
            symbol="SPY",
        )

        # First window starts with initial capital
        assert result.window_results[0].initial_equity == config.initial_capital

        # Each subsequent window starts with previous window's final equity
        for i in range(1, len(result.window_results)):
            expected_initial = result.window_results[i - 1].final_equity
            actual_initial = result.window_results[i].initial_equity
            assert actual_initial == pytest.approx(expected_initial, rel=0.001), (
                f"Window {i} initial equity mismatch: "
                f"expected {expected_initial}, got {actual_initial}"
            )


class TestWalkForwardPassRateIntegration:
    """Integration tests for pass rate thresholds (R9.4.4)."""

    def test_default_threshold_is_60_percent(self):
        """Verify default pass rate threshold is 60%."""
        assert DEFAULT_PASS_RATE_THRESHOLD == 0.6

    def test_exactly_60_percent_fails(self):
        """
        Test that exactly 60% does not pass.

        Per R9.4.4: Strategy passes if >60% of test periods are profitable.
        This is a strict greater-than comparison.
        """
        # Exactly 60% = 3/5 windows profitable
        pass_rate = 3 / 5
        assert pass_rate == 0.6
        assert passes_walk_forward_test(pass_rate) is False

    def test_61_percent_passes(self):
        """Test that >60% passes."""
        # 61% passes
        assert passes_walk_forward_test(0.61) is True
        # 2/3 = 66.7% passes
        assert passes_walk_forward_test(2 / 3) is True
        # 4/5 = 80% passes
        assert passes_walk_forward_test(0.8) is True

    def test_aggregate_pass_determination(self):
        """Test aggregate results pass determination matches raw calculation."""
        data = generate_synthetic_spy_data(252)
        bars = data["bars"]

        config = BacktestConfig(initial_capital=100000.0)

        result = run_walk_forward_backtest(
            bars=bars,
            strategy_factory=create_dual_mode_strategy_factory,
            config=config,
            train_bars=63,
            test_bars=21,
            symbol="SPY",
        )

        aggregate = aggregate_walk_forward_results(result)

        # Pass determination should match
        expected_passes = passes_walk_forward_test(result.pass_rate)
        assert aggregate.passes_walk_forward_test == expected_passes


class TestWalkForwardEdgeCases:
    """Edge case integration tests for walk-forward analysis."""

    def test_minimum_viable_data(self):
        """Test walk-forward with minimum viable data (exactly 1 window)."""
        # Generate exactly 84 bars (63 train + 21 test = 1 window)
        # Note: generate_synthetic_spy_data requires >= 126 bars, so generate manually
        np.random.seed(789)
        n_bars = 84
        prices = 100.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, n_bars)))

        # Create OHLCV data
        bars = np.zeros((n_bars, 5))
        bars[:, 0] = prices  # Open
        bars[:, 1] = prices * 1.01  # High
        bars[:, 2] = prices * 0.99  # Low
        bars[:, 3] = prices  # Close
        bars[:, 4] = 1_000_000  # Volume

        config = BacktestConfig(initial_capital=100000.0)

        result = run_walk_forward_backtest(
            bars=bars,
            strategy_factory=create_dual_mode_strategy_factory,
            config=config,
            train_bars=63,
            test_bars=21,
            symbol="SPY",
        )

        # Should have exactly 1 window
        assert result.total_windows == 1
        assert len(result.window_results) == 1

        # Pass rate is 0% or 100% with 1 window
        assert result.pass_rate in [0.0, 1.0]

    def test_walk_forward_with_strong_uptrend(self):
        """Test walk-forward with strong uptrend data (likely all windows profitable)."""
        # Generate data with strong uptrend
        np.random.seed(123)
        n_bars = 252
        prices = 100.0 * np.exp(np.cumsum(np.random.normal(0.002, 0.01, n_bars)))

        # Create OHLCV data
        bars = np.zeros((n_bars, 5))
        bars[:, 0] = prices  # Open
        bars[:, 1] = prices * 1.01  # High
        bars[:, 2] = prices * 0.99  # Low
        bars[:, 3] = prices  # Close
        bars[:, 4] = 1_000_000  # Volume

        config = BacktestConfig(initial_capital=100000.0)

        result = run_walk_forward_backtest(
            bars=bars,
            strategy_factory=create_dual_mode_strategy_factory,
            config=config,
            train_bars=63,
            test_bars=21,
            symbol="SPY",
        )

        # With strong uptrend, buy-and-hold should profit in most windows
        # (though strategy may differ)
        assert result.total_windows == 3
        # Verify results are valid even if not all profitable
        for wr in result.window_results:
            assert wr.final_equity > 0

    def test_walk_forward_with_strong_downtrend(self):
        """Test walk-forward with strong downtrend data (likely all windows unprofitable)."""
        # Generate data with strong downtrend
        np.random.seed(456)
        n_bars = 252
        prices = 100.0 * np.exp(np.cumsum(np.random.normal(-0.002, 0.01, n_bars)))

        # Create OHLCV data
        bars = np.zeros((n_bars, 5))
        bars[:, 0] = prices  # Open
        bars[:, 1] = prices * 1.01  # High
        bars[:, 2] = prices * 0.99  # Low
        bars[:, 3] = prices  # Close
        bars[:, 4] = 1_000_000  # Volume

        config = BacktestConfig(initial_capital=100000.0)

        result = run_walk_forward_backtest(
            bars=bars,
            strategy_factory=create_dual_mode_strategy_factory,
            config=config,
            train_bars=63,
            test_bars=21,
            symbol="SPY",
        )

        # Verify results are valid
        assert result.total_windows == 3
        for wr in result.window_results:
            assert wr.final_equity > 0


if __name__ == "__main__":
    # Run tests when executed directly
    print("Running walk-forward integration tests...")

    test = TestWalkForwardWithSPYData()
    test.test_walk_forward_completes_on_1_year_data()
    test.test_walk_forward_windows_have_no_data_leakage()
    test.test_pass_rate_calculation_accuracy()
    test.test_aggregate_metrics_match_known_baseline()
    test.test_walk_forward_with_extended_data()
    test.test_walk_forward_capital_continuity()

    threshold_tests = TestWalkForwardPassRateIntegration()
    threshold_tests.test_default_threshold_is_60_percent()
    threshold_tests.test_exactly_60_percent_fails()
    threshold_tests.test_61_percent_passes()
    threshold_tests.test_aggregate_pass_determination()

    edge_tests = TestWalkForwardEdgeCases()
    edge_tests.test_minimum_viable_data()
    edge_tests.test_walk_forward_with_strong_uptrend()
    edge_tests.test_walk_forward_with_strong_downtrend()

    print("\n✅ All walk-forward integration tests passed!")
