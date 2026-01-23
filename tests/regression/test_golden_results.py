"""
Golden Test Suite for Backtest Regression Testing.

This module runs backtests with deterministic data (fixed seed) and compares
results against pre-computed golden baselines. Alerts on >1% deviation to
catch unintended changes in backtest logic.

Reference: enhancement_tasks.md Phase 24 - Quality Control & Validation Framework

Usage:
    # Run tests
    pytest tests/regression/test_golden_results.py -v

    # Regenerate golden baseline (if intentionally changing behavior)
    python tests/regression/test_golden_results.py --generate-baseline
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.backtesting.engine import BacktestConfig, BacktestEngine  # noqa: E402
from backend.backtesting.metrics import PerformanceMetrics  # noqa: E402
from scripts.run_spy_backtest import DualModeStrategy, generate_synthetic_spy_data  # noqa: E402

# Path to golden results file
GOLDEN_RESULTS_PATH = Path(__file__).parent / "golden_results.json"

# Default deviation threshold (1%)
DEFAULT_DEVIATION_THRESHOLD_PCT = 1.0


def load_golden_results() -> dict:
    """Load golden results from JSON file."""
    if not GOLDEN_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Golden results file not found: {GOLDEN_RESULTS_PATH}\n"
            "Run with --generate-baseline to create it."
        )
    with open(GOLDEN_RESULTS_PATH) as f:
        return json.load(f)


def save_golden_results(data: dict) -> None:
    """Save golden results to JSON file."""
    with open(GOLDEN_RESULTS_PATH, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved golden results to {GOLDEN_RESULTS_PATH}")


def run_deterministic_backtest() -> tuple[dict, dict]:
    """
    Run backtest with deterministic data (fixed seed).

    Returns:
        Tuple of (metrics dict, config dict)
    """
    # Configuration matches golden baseline
    config = BacktestConfig(
        initial_capital=100000.0,
        commission_per_share=0.005,
        slippage_pct=0.0001,
        impact_threshold=0.1,
        impact_penalty_pct=0.0005,
        risk_free_rate=0.04,
    )

    # Generate synthetic data with fixed seed (seed=42 is set inside generate_synthetic_spy_data)
    data = generate_synthetic_spy_data(252)
    bars = data["bars"]
    timestamps = data["timestamps"]
    volumes = data["volumes"]

    # Initialize strategy
    strategy = DualModeStrategy(bars)

    # Run backtest
    engine = BacktestEngine(config)
    state = engine.run(
        bars=bars,
        strategy_func=strategy.get_orders,
        symbol="SPY",
        timestamps=timestamps,
        volumes=volumes,
    )

    # Calculate metrics
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

    config_dict = {
        "random_seed": 42,
        "num_days": 252,
        "initial_capital": config.initial_capital,
        "commission_per_share": config.commission_per_share,
        "slippage_pct": config.slippage_pct,
        "impact_threshold": config.impact_threshold,
        "impact_penalty_pct": config.impact_penalty_pct,
        "risk_free_rate": config.risk_free_rate,
        "symbol": "SPY",
    }

    return metrics, config_dict


def generate_golden_baseline() -> None:
    """
    Generate golden baseline results.

    Run this when intentionally changing backtest behavior to update
    the expected results.
    """
    print("Generating golden baseline...")
    print("Running deterministic backtest...")

    metrics, config_dict = run_deterministic_backtest()

    # Create golden results structure
    from datetime import datetime

    golden_data = {
        "description": "Golden test baseline results for regression testing",
        "version": "1.0.0",
        "generated_date": datetime.now().strftime("%Y-%m-%d"),
        "test_config": config_dict,
        "deviation_threshold_pct": DEFAULT_DEVIATION_THRESHOLD_PCT,
        "expected_metrics": metrics,
    }

    save_golden_results(golden_data)

    print("\nGenerated golden baseline with metrics:")
    print(f"  Total Return: {metrics['total_return_pct']:.4f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown: {metrics['max_drawdown_pct']:.4f}%")
    print(f"  Win Rate: {metrics['win_rate']:.4f}")
    print(f"  Total Trades: {metrics['total_trades']}")
    print(f"  Final Equity: ${metrics['final_equity']:,.2f}")


def calculate_deviation_pct(actual: float, expected: float) -> float:
    """Calculate percentage deviation between actual and expected values."""
    if expected == 0:
        return 0.0 if actual == 0 else 100.0
    return abs((actual - expected) / expected) * 100.0


class TestGoldenResults:
    """Test suite for golden result regression testing."""

    @pytest.fixture
    def golden_data(self) -> dict:
        """Load golden results fixture."""
        return load_golden_results()

    @pytest.fixture
    def current_metrics(self) -> dict:
        """Run backtest and return current metrics."""
        metrics, _ = run_deterministic_backtest()
        return metrics

    def test_golden_file_exists(self):
        """Test that golden results file exists."""
        assert GOLDEN_RESULTS_PATH.exists(), (
            f"Golden results file not found at {GOLDEN_RESULTS_PATH}. "
            "Run with --generate-baseline to create it."
        )

    def test_golden_file_has_expected_metrics(self, golden_data):
        """Test that golden file contains expected metrics."""
        assert "expected_metrics" in golden_data
        metrics = golden_data["expected_metrics"]

        # Should not just have the comment
        assert "_comment" not in metrics or len(metrics) > 1, (
            "Golden file has no metrics. Run with --generate-baseline to populate."
        )

        # Required metric keys
        required_keys = [
            "total_return_pct",
            "sharpe_ratio",
            "max_drawdown_pct",
            "win_rate",
            "total_trades",
            "final_equity",
        ]
        for key in required_keys:
            assert key in metrics, f"Missing required metric: {key}"

    def test_total_return_within_threshold(self, golden_data, current_metrics):
        """Test that total return is within deviation threshold."""
        expected = golden_data["expected_metrics"]["total_return_pct"]
        actual = current_metrics["total_return_pct"]
        threshold = golden_data.get("deviation_threshold_pct", DEFAULT_DEVIATION_THRESHOLD_PCT)

        deviation = calculate_deviation_pct(actual, expected)

        assert deviation <= threshold, (
            f"Total return deviation {deviation:.2f}% exceeds threshold {threshold}%\n"
            f"Expected: {expected:.4f}%, Actual: {actual:.4f}%"
        )

    def test_sharpe_ratio_within_threshold(self, golden_data, current_metrics):
        """Test that Sharpe ratio is within deviation threshold."""
        expected = golden_data["expected_metrics"]["sharpe_ratio"]
        actual = current_metrics["sharpe_ratio"]
        threshold = golden_data.get("deviation_threshold_pct", DEFAULT_DEVIATION_THRESHOLD_PCT)

        deviation = calculate_deviation_pct(actual, expected)

        assert deviation <= threshold, (
            f"Sharpe ratio deviation {deviation:.2f}% exceeds threshold {threshold}%\n"
            f"Expected: {expected:.4f}, Actual: {actual:.4f}"
        )

    def test_max_drawdown_within_threshold(self, golden_data, current_metrics):
        """Test that max drawdown is within deviation threshold."""
        expected = golden_data["expected_metrics"]["max_drawdown_pct"]
        actual = current_metrics["max_drawdown_pct"]
        threshold = golden_data.get("deviation_threshold_pct", DEFAULT_DEVIATION_THRESHOLD_PCT)

        deviation = calculate_deviation_pct(actual, expected)

        assert deviation <= threshold, (
            f"Max drawdown deviation {deviation:.2f}% exceeds threshold {threshold}%\n"
            f"Expected: {expected:.4f}%, Actual: {actual:.4f}%"
        )

    def test_win_rate_within_threshold(self, golden_data, current_metrics):
        """Test that win rate is within deviation threshold."""
        expected = golden_data["expected_metrics"]["win_rate"]
        actual = current_metrics["win_rate"]
        threshold = golden_data.get("deviation_threshold_pct", DEFAULT_DEVIATION_THRESHOLD_PCT)

        deviation = calculate_deviation_pct(actual, expected)

        assert deviation <= threshold, (
            f"Win rate deviation {deviation:.2f}% exceeds threshold {threshold}%\n"
            f"Expected: {expected:.4f}, Actual: {actual:.4f}"
        )

    def test_total_trades_exact_match(self, golden_data, current_metrics):
        """Test that total trades matches exactly (no deviation allowed)."""
        expected = golden_data["expected_metrics"]["total_trades"]
        actual = current_metrics["total_trades"]

        assert actual == expected, (
            f"Total trades mismatch. Expected: {expected}, Actual: {actual}\n"
            "This indicates a change in signal generation or order execution logic."
        )

    def test_final_equity_within_threshold(self, golden_data, current_metrics):
        """Test that final equity is within deviation threshold."""
        expected = golden_data["expected_metrics"]["final_equity"]
        actual = current_metrics["final_equity"]
        threshold = golden_data.get("deviation_threshold_pct", DEFAULT_DEVIATION_THRESHOLD_PCT)

        deviation = calculate_deviation_pct(actual, expected)

        assert deviation <= threshold, (
            f"Final equity deviation {deviation:.2f}% exceeds threshold {threshold}%\n"
            f"Expected: ${expected:,.2f}, Actual: ${actual:,.2f}"
        )

    def test_annualized_return_within_threshold(self, golden_data, current_metrics):
        """Test that annualized return is within deviation threshold."""
        expected = golden_data["expected_metrics"].get("annualized_return_pct", 0)
        actual = current_metrics["annualized_return_pct"]
        threshold = golden_data.get("deviation_threshold_pct", DEFAULT_DEVIATION_THRESHOLD_PCT)

        # Skip if expected value not in golden file
        if expected == 0 and "annualized_return_pct" not in golden_data["expected_metrics"]:
            pytest.skip("annualized_return_pct not in golden baseline")

        deviation = calculate_deviation_pct(actual, expected)

        assert deviation <= threshold, (
            f"Annualized return deviation {deviation:.2f}% exceeds threshold {threshold}%\n"
            f"Expected: {expected:.4f}%, Actual: {actual:.4f}%"
        )

    def test_avg_win_loss_ratio_within_threshold(self, golden_data, current_metrics):
        """Test that avg win/loss ratio is within deviation threshold."""
        expected = golden_data["expected_metrics"].get("avg_win_loss_ratio", 0)
        actual = current_metrics["avg_win_loss_ratio"]
        threshold = golden_data.get("deviation_threshold_pct", DEFAULT_DEVIATION_THRESHOLD_PCT)

        # Skip if expected value not in golden file or both are 0
        if expected == 0 and actual == 0:
            pytest.skip("No wins or losses to compare")

        if expected == 0:
            pytest.skip("avg_win_loss_ratio not in golden baseline")

        deviation = calculate_deviation_pct(actual, expected)

        assert deviation <= threshold, (
            f"Avg win/loss ratio deviation {deviation:.2f}% exceeds threshold {threshold}%\n"
            f"Expected: {expected:.4f}, Actual: {actual:.4f}"
        )


class TestGoldenResultsIntegrity:
    """Test suite for golden file integrity and consistency."""

    def test_config_matches_test_parameters(self):
        """Test that golden file config matches test parameters."""
        golden_data = load_golden_results()
        config = golden_data["test_config"]

        # Verify config values match what we use in tests
        assert config["random_seed"] == 42
        assert config["num_days"] == 252
        assert config["initial_capital"] == 100000.0
        assert config["commission_per_share"] == 0.005
        assert config["slippage_pct"] == 0.0001
        assert config["symbol"] == "SPY"

    def test_deviation_threshold_is_reasonable(self):
        """Test that deviation threshold is reasonable (between 0.1% and 5%)."""
        golden_data = load_golden_results()
        threshold = golden_data.get("deviation_threshold_pct", DEFAULT_DEVIATION_THRESHOLD_PCT)

        assert 0.1 <= threshold <= 5.0, (
            f"Deviation threshold {threshold}% is outside reasonable range (0.1-5%)"
        )


def print_comparison_report(golden_data: dict, current_metrics: dict) -> None:
    """Print a comparison report between golden and current metrics."""
    expected = golden_data["expected_metrics"]
    threshold = golden_data.get("deviation_threshold_pct", DEFAULT_DEVIATION_THRESHOLD_PCT)

    print("\n" + "=" * 80)
    print("GOLDEN TEST COMPARISON REPORT")
    print("=" * 80)
    print(f"Deviation Threshold: {threshold}%")
    print()

    metrics_to_compare = [
        ("total_return_pct", "Total Return (%)", "{:.4f}"),
        ("sharpe_ratio", "Sharpe Ratio", "{:.4f}"),
        ("max_drawdown_pct", "Max Drawdown (%)", "{:.4f}"),
        ("win_rate", "Win Rate", "{:.4f}"),
        ("total_trades", "Total Trades", "{:.0f}"),
        ("final_equity", "Final Equity ($)", "{:,.2f}"),
        ("annualized_return_pct", "Annualized Return (%)", "{:.4f}"),
        ("avg_win_loss_ratio", "Avg Win/Loss Ratio", "{:.4f}"),
    ]

    all_passed = True
    for key, label, fmt in metrics_to_compare:
        if key not in expected:
            continue

        exp_val = expected[key]
        act_val = current_metrics.get(key, 0)
        deviation = calculate_deviation_pct(act_val, exp_val)
        passed = deviation <= threshold if key != "total_trades" else act_val == exp_val
        status = "PASS" if passed else "FAIL"
        symbol = "+" if passed else "X"

        if not passed:
            all_passed = False

        exp_str = fmt.format(exp_val)
        act_str = fmt.format(act_val)
        print(
            f"[{symbol}] {label:25s} | Expected: {exp_str:>15s} | Actual: {act_str:>15s} | "
            f"Deviation: {deviation:>6.2f}% | {status}"
        )

    print()
    if all_passed:
        print("RESULT: All metrics within acceptable deviation threshold")
    else:
        print("RESULT: Some metrics exceed deviation threshold - INVESTIGATION REQUIRED")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Golden test suite for backtest regression")
    parser.add_argument(
        "--generate-baseline",
        action="store_true",
        help="Generate golden baseline (use when intentionally changing behavior)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comparison report without failing tests",
    )
    args = parser.parse_args()

    if args.generate_baseline:
        generate_golden_baseline()
    elif args.compare:
        golden_data = load_golden_results()
        metrics, _ = run_deterministic_backtest()
        print_comparison_report(golden_data, metrics)
    else:
        # Run pytest
        sys.exit(pytest.main([__file__, "-v"]))
