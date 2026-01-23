"""
Benchmark Comparison Tests for Strategy Performance.

This module compares strategy returns against benchmark indices:
- Buy-and-hold SPY returns
- SPY total return over the same period

Flags strategies that significantly underperform benchmarks, helping identify
when active trading is adding value versus when passive investing would be better.

Note: These tests are designed to "flag" underperformance (via warnings) rather than
hard-fail. In many market conditions (strong bull markets), buy-and-hold will outperform
active strategies. The goal is visibility into relative performance, not enforcement.

Reference: enhancement_tasks.md Phase 24 - Quality Control & Validation Framework

Usage:
    pytest tests/regression/test_benchmark_comparison.py -v
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.backtesting.engine import BacktestConfig, BacktestEngine  # noqa: E402
from backend.backtesting.metrics import PerformanceMetrics  # noqa: E402
from scripts.run_spy_backtest import DualModeStrategy, generate_synthetic_spy_data  # noqa: E402


def calculate_buy_and_hold_return(
    prices: np.ndarray,
    initial_capital: float,
    commission_per_share: float = 0.005,
) -> dict:
    """
    Calculate buy-and-hold strategy returns.

    Simulates buying at the first price and holding until the end.
    Includes commission costs for entry and exit.

    Args:
        prices: Array of closing prices
        initial_capital: Starting capital
        commission_per_share: Commission per share traded

    Returns:
        Dict with buy-and-hold performance metrics
    """
    entry_price = prices[0]
    exit_price = prices[-1]
    num_days = len(prices)

    # Calculate shares we can buy with initial capital (accounting for commission)
    # shares * (entry_price + commission) = initial_capital
    # Approximate: buy what we can afford
    shares = int(initial_capital / (entry_price + commission_per_share))

    # Entry cost
    entry_cost = shares * entry_price + shares * commission_per_share

    # Exit proceeds
    exit_proceeds = shares * exit_price - shares * commission_per_share

    # Calculate returns
    total_return = exit_proceeds - entry_cost
    total_return_pct = (total_return / entry_cost) * 100.0

    # Price return (without costs, for comparison)
    price_return_pct = ((exit_price / entry_price) - 1) * 100.0

    # Build equity curve for buy-and-hold
    equity_curve = np.zeros(num_days)
    for i in range(num_days):
        current_value = shares * prices[i]
        # Subtract entry commission (already paid)
        if i == 0:
            equity_curve[i] = initial_capital - shares * commission_per_share
        else:
            equity_curve[i] = current_value + (initial_capital - entry_cost)

    # Final equity includes exit commission
    final_equity = exit_proceeds + (initial_capital - entry_cost)

    # Calculate max drawdown for buy-and-hold
    peak = equity_curve[0]
    max_drawdown = 0.0
    for i in range(1, num_days):
        if equity_curve[i] > peak:
            peak = equity_curve[i]
        drawdown = (peak - equity_curve[i]) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    return {
        "shares_bought": shares,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "entry_cost": entry_cost,
        "exit_proceeds": exit_proceeds,
        "total_return": total_return,
        "total_return_pct": total_return_pct,
        "price_return_pct": price_return_pct,
        "final_equity": final_equity,
        "max_drawdown_pct": max_drawdown * 100.0,
        "num_days": num_days,
    }


def run_strategy_backtest(num_days: int = 252) -> tuple[dict, np.ndarray]:
    """
    Run the dual-mode strategy backtest.

    Args:
        num_days: Number of trading days

    Returns:
        Tuple of (metrics dict, closing prices array)
    """
    config = BacktestConfig(
        initial_capital=100000.0,
        commission_per_share=0.005,
        slippage_pct=0.0001,
        impact_threshold=0.1,
        impact_penalty_pct=0.0005,
        risk_free_rate=0.04,
    )

    data = generate_synthetic_spy_data(num_days)
    bars = data["bars"]
    timestamps = data["timestamps"]
    volumes = data["volumes"]
    close_prices = data["close"]

    strategy = DualModeStrategy(bars)

    engine = BacktestEngine(config)
    state = engine.run(
        bars=bars,
        strategy_func=strategy.get_orders,
        symbol="SPY",
        timestamps=timestamps,
        volumes=volumes,
    )

    equity_curve = np.array(state.equity_curve)
    trades_pnl = np.array([t.pnl for t in state.trades]) if state.trades else np.array([])
    trades_holding_periods = (
        np.array([t.holding_bars for t in state.trades], dtype=np.int32)
        if state.trades
        else np.array([], dtype=np.int32)
    )

    metrics = PerformanceMetrics.calculate_all_metrics(
        equity_curve=equity_curve,
        trades_pnl=trades_pnl,
        trades_holding_periods=trades_holding_periods,
        initial_capital=config.initial_capital,
        risk_free_rate=config.risk_free_rate,
    )

    return metrics, close_prices


class TestBenchmarkComparison:
    """Test suite comparing strategy performance against benchmarks."""

    # Threshold for significant underperformance (percentage points)
    UNDERPERFORMANCE_THRESHOLD_PCT = 10.0

    # Risk-adjusted underperformance threshold (Sharpe difference)
    SHARPE_UNDERPERFORMANCE_THRESHOLD = 0.5

    @pytest.fixture
    def backtest_results(self) -> tuple[dict, np.ndarray]:
        """Run backtest and return results."""
        return run_strategy_backtest(num_days=252)

    @pytest.fixture
    def strategy_metrics(self, backtest_results) -> dict:
        """Extract strategy metrics from backtest results."""
        metrics, _ = backtest_results
        return metrics

    @pytest.fixture
    def close_prices(self, backtest_results) -> np.ndarray:
        """Extract close prices from backtest results."""
        _, prices = backtest_results
        return prices

    @pytest.fixture
    def buy_hold_metrics(self, close_prices) -> dict:
        """Calculate buy-and-hold benchmark metrics."""
        return calculate_buy_and_hold_return(
            prices=close_prices,
            initial_capital=100000.0,
            commission_per_share=0.005,
        )

    def test_strategy_vs_buy_and_hold_return(
        self, strategy_metrics, buy_hold_metrics
    ):
        """
        Compare strategy return vs buy-and-hold and flag significant underperformance.

        This test always passes but issues a warning if strategy significantly
        underperforms. The warning is captured in test output for visibility.
        """
        strategy_return = strategy_metrics["total_return_pct"]
        buy_hold_return = buy_hold_metrics["total_return_pct"]

        underperformance = buy_hold_return - strategy_return

        # Flag underperformance via warning (does not fail the test)
        if underperformance > self.UNDERPERFORMANCE_THRESHOLD_PCT:
            warnings.warn(
                f"Strategy significantly underperforms buy-and-hold:\n"
                f"  Strategy return: {strategy_return:.2f}%\n"
                f"  Buy-and-hold return: {buy_hold_return:.2f}%\n"
                f"  Underperformance: {underperformance:.2f}% "
                f"(threshold: {self.UNDERPERFORMANCE_THRESHOLD_PCT}%)",
                UserWarning,
                stacklevel=1,
            )

        # Test passes but records the comparison
        assert True, "Benchmark comparison completed"

    def test_strategy_vs_spy_price_return(self, strategy_metrics, buy_hold_metrics):
        """
        Compare strategy return vs SPY price return and flag significant underperformance.

        Price return is the raw percentage change without transaction costs.
        """
        strategy_return = strategy_metrics["total_return_pct"]
        spy_price_return = buy_hold_metrics["price_return_pct"]

        underperformance = spy_price_return - strategy_return

        # Allow more slack for price return since strategy has costs
        adjusted_threshold = self.UNDERPERFORMANCE_THRESHOLD_PCT + 2.0

        # Flag underperformance via warning
        if underperformance > adjusted_threshold:
            warnings.warn(
                f"Strategy significantly underperforms SPY price return:\n"
                f"  Strategy return: {strategy_return:.2f}%\n"
                f"  SPY price return: {spy_price_return:.2f}%\n"
                f"  Underperformance: {underperformance:.2f}% "
                f"(threshold: {adjusted_threshold}%)",
                UserWarning,
                stacklevel=1,
            )

        # Test passes but records the comparison
        assert True, "Price return comparison completed"

    def test_strategy_max_drawdown_reasonable(
        self, strategy_metrics, buy_hold_metrics
    ):
        """
        Test that strategy max drawdown is not significantly worse than buy-and-hold.

        A good active strategy should provide better risk control.
        """
        # Strategy max drawdown is negative, so use absolute value
        strategy_dd = abs(strategy_metrics["max_drawdown_pct"])
        buy_hold_dd = buy_hold_metrics["max_drawdown_pct"]

        # Strategy drawdown should not exceed buy-and-hold by more than 50%
        drawdown_ratio_threshold = 1.5

        if buy_hold_dd > 0:
            drawdown_ratio = strategy_dd / buy_hold_dd
            if drawdown_ratio > drawdown_ratio_threshold:
                warnings.warn(
                    f"Strategy drawdown significantly worse than buy-and-hold:\n"
                    f"  Strategy max drawdown: {strategy_dd:.2f}%\n"
                    f"  Buy-and-hold max drawdown: {buy_hold_dd:.2f}%\n"
                    f"  Ratio: {drawdown_ratio:.2f}x (threshold: {drawdown_ratio_threshold}x)",
                    UserWarning,
                    stacklevel=1,
                )

        # This test always passes - drawdown comparison is informational
        assert True, "Drawdown comparison completed"

    def test_strategy_generates_alpha(self, strategy_metrics, buy_hold_metrics):
        """
        Test that strategy generates positive alpha (excess return).

        Note: This is an informational test - it will not fail if alpha is negative,
        but will warn. True alpha calculation requires proper risk adjustment.
        """
        strategy_return = strategy_metrics["total_return_pct"]
        buy_hold_return = buy_hold_metrics["total_return_pct"]

        alpha = strategy_return - buy_hold_return

        # This is informational - we just report the alpha
        if alpha < 0:
            warnings.warn(
                f"Strategy alpha is negative: {alpha:.2f}% "
                f"(strategy: {strategy_return:.2f}%, buy-hold: {buy_hold_return:.2f}%)",
                UserWarning,
                stacklevel=1,
            )

        # Test passes but records alpha
        assert True, f"Alpha calculation completed: {alpha:.2f}%"

    def test_risk_adjusted_performance(self, strategy_metrics, buy_hold_metrics):
        """
        Test risk-adjusted performance using a simple return/risk ratio.

        Compares return per unit of drawdown risk.
        """
        strategy_return = strategy_metrics["total_return_pct"]
        strategy_dd = abs(strategy_metrics["max_drawdown_pct"])

        buy_hold_return = buy_hold_metrics["total_return_pct"]
        buy_hold_dd = buy_hold_metrics["max_drawdown_pct"]

        # Avoid division by zero
        if strategy_dd == 0:
            strategy_dd = 0.01
        if buy_hold_dd == 0:
            buy_hold_dd = 0.01

        strategy_risk_adj = strategy_return / strategy_dd
        buy_hold_risk_adj = buy_hold_return / buy_hold_dd

        # Strategy should have at least 50% of buy-and-hold risk-adjusted return
        min_ratio = 0.5

        if buy_hold_risk_adj > 0:
            ratio = strategy_risk_adj / buy_hold_risk_adj
            if ratio < min_ratio:
                warnings.warn(
                    f"Strategy risk-adjusted return is low:\n"
                    f"  Strategy return/drawdown: {strategy_risk_adj:.2f}\n"
                    f"  Buy-and-hold return/drawdown: {buy_hold_risk_adj:.2f}\n"
                    f"  Ratio: {ratio:.2f} (minimum: {min_ratio})",
                    UserWarning,
                    stacklevel=1,
                )

        # Test passes but records comparison
        assert True, "Risk-adjusted comparison completed"


class TestBenchmarkComparisonExtended:
    """Extended benchmark comparison tests with multiple time periods."""

    @pytest.mark.parametrize("num_days", [126, 252, 504])
    def test_strategy_across_time_periods(self, num_days):
        """
        Compare strategy performance across different time periods.

        Flags underperformance via warnings for visibility but does not fail.

        Args:
            num_days: Number of trading days (126=6mo, 252=1yr, 504=2yr)
        """
        strategy_metrics, close_prices = run_strategy_backtest(num_days=num_days)
        buy_hold_metrics = calculate_buy_and_hold_return(
            prices=close_prices,
            initial_capital=100000.0,
            commission_per_share=0.005,
        )

        strategy_return = strategy_metrics["total_return_pct"]
        buy_hold_return = buy_hold_metrics["total_return_pct"]

        # More lenient threshold for shorter periods (higher variance)
        if num_days < 200:
            threshold = 15.0
        elif num_days < 400:
            threshold = 10.0
        else:
            threshold = 8.0

        underperformance = buy_hold_return - strategy_return

        # Flag underperformance via warning
        if underperformance > threshold:
            warnings.warn(
                f"Strategy underperforms buy-and-hold over {num_days} days:\n"
                f"  Strategy return: {strategy_return:.2f}%\n"
                f"  Buy-and-hold return: {buy_hold_return:.2f}%\n"
                f"  Underperformance: {underperformance:.2f}% (threshold: {threshold}%)",
                UserWarning,
                stacklevel=1,
            )

        # Test passes but records comparison
        assert True, f"Period comparison completed ({num_days} days)"


class TestBenchmarkReporting:
    """Test suite for benchmark comparison reporting."""

    def test_generate_comparison_report(self):
        """Generate and verify comparison report format."""
        strategy_metrics, close_prices = run_strategy_backtest(num_days=252)
        buy_hold_metrics = calculate_buy_and_hold_return(
            prices=close_prices,
            initial_capital=100000.0,
            commission_per_share=0.005,
        )

        report = generate_comparison_report(strategy_metrics, buy_hold_metrics)

        # Verify report contains expected sections
        assert "BENCHMARK COMPARISON REPORT" in report
        assert "Strategy Performance" in report
        assert "Buy-and-Hold Benchmark" in report
        assert "Comparison" in report


def generate_comparison_report(strategy_metrics: dict, buy_hold_metrics: dict) -> str:
    """
    Generate a formatted comparison report.

    Args:
        strategy_metrics: Dict of strategy performance metrics
        buy_hold_metrics: Dict of buy-and-hold benchmark metrics

    Returns:
        Formatted report string
    """
    strategy_return = strategy_metrics["total_return_pct"]
    strategy_dd = abs(strategy_metrics["max_drawdown_pct"])
    strategy_sharpe = strategy_metrics["sharpe_ratio"]
    strategy_final = strategy_metrics["final_equity"]
    total_trades = strategy_metrics["total_trades"]

    buy_hold_return = buy_hold_metrics["total_return_pct"]
    buy_hold_dd = buy_hold_metrics["max_drawdown_pct"]
    buy_hold_final = buy_hold_metrics["final_equity"]
    price_return = buy_hold_metrics["price_return_pct"]

    alpha = strategy_return - buy_hold_return

    lines = [
        "",
        "=" * 80,
        "BENCHMARK COMPARISON REPORT",
        "=" * 80,
        "",
        "Strategy Performance:",
        f"  Total Return:    {strategy_return:>10.2f}%",
        f"  Max Drawdown:    {strategy_dd:>10.2f}%",
        f"  Sharpe Ratio:    {strategy_sharpe:>10.2f}",
        f"  Final Equity:    ${strategy_final:>12,.2f}",
        f"  Total Trades:    {total_trades:>10d}",
        "",
        "Buy-and-Hold Benchmark:",
        f"  Total Return:    {buy_hold_return:>10.2f}%",
        f"  Price Return:    {price_return:>10.2f}%",
        f"  Max Drawdown:    {buy_hold_dd:>10.2f}%",
        f"  Final Equity:    ${buy_hold_final:>12,.2f}",
        "",
        "Comparison:",
        f"  Alpha (excess):  {alpha:>10.2f}%",
        f"  Outperforming:   {'Yes' if alpha > 0 else 'No':>10s}",
        "",
        "=" * 80,
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    # Run comparison and print report
    print("Running benchmark comparison...")

    strategy_metrics, close_prices = run_strategy_backtest(num_days=252)
    buy_hold_metrics = calculate_buy_and_hold_return(
        prices=close_prices,
        initial_capital=100000.0,
        commission_per_share=0.005,
    )

    report = generate_comparison_report(strategy_metrics, buy_hold_metrics)
    print(report)

    # Run pytest
    sys.exit(pytest.main([__file__, "-v"]))
