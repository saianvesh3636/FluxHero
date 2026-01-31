"""
Execute Walk-Forward Grid Search Optimization for Dual-Mode Strategy.

This script runs walk-forward analysis with parameter optimization:
- Divides data into consecutive train/test windows
- Re-optimizes parameters on each training window
- Evaluates out-of-sample performance on test windows
- Reports aggregate metrics and pass rate

Usage:
    python scripts/run_grid_search.py
    make backtest-optimize
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime, timedelta  # noqa: E402

import numpy as np  # noqa: E402

from backend.backtesting.engine import BacktestConfig  # noqa: E402
from backend.backtesting.grid_search import (  # noqa: E402
    DualModeGridConfig,
    create_dual_mode_optimizer,
    create_dual_mode_strategy_factory,
)
from backend.backtesting.walk_forward import (  # noqa: E402
    aggregate_walk_forward_results,
    run_walk_forward_backtest,
)


def generate_synthetic_spy_data(num_days: int = 504) -> dict:
    """
    Generate synthetic SPY-like OHLCV data for backtesting.

    Simulates realistic market conditions with multiple regimes:
    - Starting price: $450
    - Daily volatility: ~1%
    - Volume: ~80M average with noise

    Args:
        num_days: Number of trading days (default: 504 = 2 years)

    Returns:
        Dict with 'bars', 'timestamps', 'volumes' arrays
    """
    np.random.seed(42)  # For reproducibility

    # Generate timestamps (trading days only)
    start_date = datetime(2022, 1, 1)
    timestamps = np.array([
        (start_date + timedelta(days=i)).timestamp()
        for i in range(num_days)
    ])

    # Generate price series with drift and volatility
    starting_price = 450.0
    daily_return_mean = 0.08 / 252  # ~8% annual return
    daily_volatility = 0.012  # ~1.2% daily volatility

    # Generate returns with different market regimes
    returns = np.random.normal(daily_return_mean, daily_volatility, num_days)

    # Create varying regimes throughout the data
    # Year 1: Mixed conditions
    returns[0:63] += 0.002       # Q1: Uptrend
    returns[63:126] = np.random.normal(0, 0.008, 63)  # Q2: Choppy
    returns[126:189] -= 0.001   # Q3: Mild downtrend
    returns[189:252] += 0.001   # Q4: Recovery

    # Year 2: Different pattern
    returns[252:315] = np.random.normal(0.0005, 0.015, 63)  # Q1: High vol uptrend
    returns[315:378] -= 0.002   # Q2: Correction
    returns[378:441] += 0.003   # Q3: Strong rally
    returns[441:504] = np.random.normal(0, 0.01, min(63, num_days - 441))  # Q4: Consolidation

    # Calculate closing prices
    close = starting_price * np.exp(np.cumsum(returns))

    # Generate OHLC from close
    daily_range = 0.006  # ~0.6% intraday range
    high = close * (1 + np.abs(np.random.normal(0, daily_range, num_days)))
    low = close * (1 - np.abs(np.random.normal(0, daily_range, num_days)))
    open_price = close * np.exp(np.random.normal(0, 0.003, num_days))

    # Ensure OHLC relationships are valid
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    # Generate volume (80M average with ±20% noise)
    volume = np.abs(np.random.normal(80_000_000, 16_000_000, num_days))

    # Create OHLCV bars array (N, 5)
    bars = np.column_stack([open_price, high, low, close, volume])

    return {
        'bars': bars,
        'timestamps': timestamps,
        'volumes': volume,
        'close': close,
    }


def main():
    """Execute walk-forward grid search optimization."""
    print("=" * 80)
    print("FluxHero - Walk-Forward Grid Search Optimization")
    print("=" * 80)
    print()

    # Generate synthetic SPY data (2 years = 504 trading days)
    print("Generating synthetic SPY data (2 years, 504 trading days)...")
    data = generate_synthetic_spy_data(504)
    bars = data['bars']
    timestamps = data['timestamps']
    close_prices = data['close']

    print(f"Data range: {datetime.fromtimestamp(timestamps[0]).date()} to {datetime.fromtimestamp(timestamps[-1]).date()}")
    print(f"Starting price: ${close_prices[0]:.2f}")
    print(f"Ending price: ${close_prices[-1]:.2f}")
    buy_hold_return = ((close_prices[-1] / close_prices[0]) - 1) * 100
    print(f"Buy & Hold return: {buy_hold_return:.2f}%")
    print()

    # Configure backtest
    config = BacktestConfig(
        initial_capital=100000.0,
        commission_per_share=0.005,
        slippage_pct=0.0001,
        impact_threshold=0.1,
        impact_penalty_pct=0.0005,
        risk_free_rate=0.04,
    )

    print("Backtest configuration:")
    print(f"  Initial capital: ${config.initial_capital:,.0f}")
    print(f"  Commission: ${config.commission_per_share}/share")
    print(f"  Risk-free rate: {config.risk_free_rate * 100:.1f}%")
    print()

    # Configure grid search
    grid_config = DualModeGridConfig(
        rsi_oversold=(30.0, 35.0, 40.0),
        rsi_overbought=(60.0, 65.0, 70.0),
        bb_entry_threshold=(0.10, 0.15, 0.20),
        atr_entry_mult=(0.3, 0.5, 0.7),
        stop_atr_mult=(1.5, 2.0, 2.5),
        target_metric="sharpe_ratio",
        use_or_logic=True,
    )

    print("Grid search configuration:")
    print(f"  RSI oversold: {grid_config.rsi_oversold}")
    print(f"  RSI overbought: {grid_config.rsi_overbought}")
    print(f"  BB entry threshold: {grid_config.bb_entry_threshold}")
    print(f"  ATR entry multiplier: {grid_config.atr_entry_mult}")
    print(f"  Stop ATR multiplier: {grid_config.stop_atr_mult}")
    print(f"  Target metric: {grid_config.target_metric}")
    total_combos = (
        len(grid_config.rsi_oversold) *
        len(grid_config.rsi_overbought) *
        len(grid_config.bb_entry_threshold) *
        len(grid_config.atr_entry_mult) *
        len(grid_config.stop_atr_mult)
    )
    print(f"  Total parameter combinations: {total_combos}")
    print()

    # Walk-forward configuration
    train_bars = 126  # 6 months training
    test_bars = 63    # 3 months testing
    print("Walk-forward configuration:")
    print(f"  Training period: {train_bars} bars (~6 months)")
    print(f"  Testing period: {test_bars} bars (~3 months)")
    print()

    # Create optimizer and strategy factory
    optimizer = create_dual_mode_optimizer(grid_config)
    strategy_factory = create_dual_mode_strategy_factory()

    # Initial parameters (used if optimization fails)
    initial_params = {
        "rsi_oversold": 35.0,
        "rsi_overbought": 65.0,
        "bb_entry_threshold": 0.15,
        "atr_entry_mult": 0.5,
        "stop_atr_mult": 2.0,
        "use_or_logic": True,
    }

    # Run walk-forward backtest with optimization
    print("Running walk-forward backtest with parameter optimization...")
    print("(This may take a few moments as each window optimizes parameters)")
    print()

    result = run_walk_forward_backtest(
        bars=bars,
        strategy_factory=strategy_factory,
        config=config,
        train_bars=train_bars,
        test_bars=test_bars,
        timestamps=timestamps,
        symbol="SPY",
        initial_params=initial_params,
        optimizer=optimizer,
    )

    # Aggregate results
    print()
    print("Aggregating walk-forward results...")
    aggregate = aggregate_walk_forward_results(result)

    # Display results
    print()
    print("=" * 80)
    print("WALK-FORWARD OPTIMIZATION RESULTS")
    print("=" * 80)
    print()

    print("Per-Window Results:")
    print("-" * 60)
    for i, window_result in enumerate(result.window_results):
        status = "PROFIT" if window_result.is_profitable else "LOSS"
        ret = ((window_result.final_equity - window_result.initial_equity) /
               window_result.initial_equity * 100)
        sharpe = window_result.metrics.get("sharpe_ratio", 0.0)
        params = window_result.strategy_params
        print(f"  Window {i+1}: {status:6} | Return: {ret:+6.2f}% | Sharpe: {sharpe:+5.2f}")
        print(f"           Params: RSI({params.get('rsi_oversold', '-')}/{params.get('rsi_overbought', '-')}), "
              f"BB({params.get('bb_entry_threshold', '-')}), ATR({params.get('atr_entry_mult', '-')}/{params.get('stop_atr_mult', '-')})")

    print()
    print("Aggregate Metrics:")
    print("-" * 60)
    print(f"  Total Windows:     {aggregate.total_windows}")
    print(f"  Profitable Windows: {aggregate.total_profitable_windows}")
    print(f"  Pass Rate:         {aggregate.pass_rate:.1%}")
    print(f"  Passes Test:       {'YES' if aggregate.passes_walk_forward_test else 'NO'} (>60% required)")
    print()
    print(f"  Aggregate Sharpe:  {aggregate.aggregate_sharpe:.2f}")
    print(f"  Aggregate MaxDD:   {aggregate.aggregate_max_drawdown_pct:.2f}%")
    print(f"  Aggregate Win Rate: {aggregate.aggregate_win_rate:.1%}")
    print(f"  Total Trades:      {aggregate.total_trades}")
    print()
    print(f"  Initial Capital:   ${aggregate.initial_capital:,.2f}")
    print(f"  Final Capital:     ${aggregate.final_capital:,.2f}")
    print(f"  Total Return:      {aggregate.total_return_pct:.2f}%")
    print(f"  vs Buy & Hold:     {buy_hold_return:.2f}%")

    print()
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print()

    # Check success criteria
    success = True
    issues = []

    # Check pass rate
    if aggregate.passes_walk_forward_test:
        print(f"  PASS Pass Rate: {aggregate.pass_rate:.1%} > 60%")
    else:
        print(f"  FAIL Pass Rate: {aggregate.pass_rate:.1%} <= 60%")
        success = False
        issues.append("Pass rate below 60%")

    # Check OOS Sharpe
    if aggregate.aggregate_sharpe > 0:
        print(f"  PASS OOS Sharpe: {aggregate.aggregate_sharpe:.2f} > 0")
    else:
        print(f"  FAIL OOS Sharpe: {aggregate.aggregate_sharpe:.2f} <= 0")
        success = False
        issues.append("Negative out-of-sample Sharpe ratio")

    # Check for overfitting (compare train vs test Sharpe gap)
    # We'll estimate by looking at return consistency
    returns_std = np.std(aggregate.per_window_returns) if aggregate.per_window_returns else 0
    if returns_std < 15.0:  # Returns don't vary too much between windows
        print(f"  PASS Return consistency: StdDev {returns_std:.1f}% < 15%")
    else:
        print(f"  WARN Return consistency: StdDev {returns_std:.1f}% >= 15% (possible overfitting)")

    print()
    if success:
        print("SUCCESS: All primary validation criteria met!")
    else:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")

    print("=" * 80)

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
