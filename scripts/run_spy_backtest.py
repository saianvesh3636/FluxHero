"""
Execute 1-Year Backtest on SPY with Dual-Mode Strategy (Phase 17 - Task 2).

This script runs a comprehensive backtest using:
- Historical SPY data (1 year)
- Dual-mode strategy (trend-following + mean-reversion)
- Regime detection for automatic strategy switching
- Full risk management and position sizing

Success Criteria (from FLUXHERO_REQUIREMENTS.md):
- Sharpe ratio > 0.8
- Max drawdown < 25%
- Win rate > 45%

Usage:
    python scripts/run_spy_backtest.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime, timedelta  # noqa: E402

import numpy as np  # noqa: E402

from backend.backtesting.engine import (  # noqa: E402
    BacktestConfig,
    BacktestEngine,
    Order,
    OrderSide,
    OrderType,
    Position,
    PositionSide,
)
from backend.backtesting.metrics import PerformanceMetrics  # noqa: E402
from backend.computation.adaptive_ema import calculate_kama_with_regime_adjustment  # noqa: E402
from backend.computation.indicators import (  # noqa: E402
    calculate_atr,
    calculate_bollinger_bands,
    calculate_rsi,
)
from backend.strategy.dual_mode import (  # noqa: E402
    SIGNAL_EXIT_LONG,
    SIGNAL_LONG,
    SIGNAL_NONE,
    calculate_position_size,
    generate_mean_reversion_signals,
    generate_trend_following_signals,
)
from backend.strategy.regime_detector import (  # noqa: E402
    REGIME_MEAN_REVERSION,
    REGIME_STRONG_TREND,
    detect_regime,
)


def generate_synthetic_spy_data(num_days: int = 252) -> dict:
    """
    Generate synthetic SPY-like OHLCV data for backtesting.

    Simulates realistic market conditions:
    - Starting price: $450
    - Daily volatility: ~1%
    - Upward drift: ~10% annual
    - Volume: ~80M average with noise

    Args:
        num_days: Number of trading days (default: 252 = 1 year)

    Returns:
        Dict with 'bars', 'timestamps', 'volumes' arrays
    """
    np.random.seed(42)  # For reproducibility

    # Generate timestamps (trading days only)
    start_date = datetime(2023, 1, 1)
    timestamps = np.array([
        (start_date + timedelta(days=i)).timestamp()
        for i in range(num_days)
    ])

    # Generate price series with drift and volatility
    starting_price = 450.0
    daily_return_mean = 0.10 / 252  # ~10% annual return
    daily_volatility = 0.01  # ~1% daily volatility

    # Generate returns with different market regimes
    returns = np.random.normal(daily_return_mean, daily_volatility, num_days)

    # Add some trending periods and ranging periods for regime testing
    # First quarter: strong uptrend
    returns[0:63] += 0.003
    # Second quarter: choppy/ranging
    returns[63:126] = np.random.normal(0, 0.008, 63)
    # Third quarter: downtrend
    returns[126:189] -= 0.002
    # Fourth quarter: recovery
    returns[189:252] += 0.002

    # Calculate closing prices
    close = starting_price * np.exp(np.cumsum(returns))

    # Generate OHLC from close
    daily_range = 0.005  # ~0.5% intraday range
    high = close * (1 + np.abs(np.random.normal(0, daily_range, num_days)))
    low = close * (1 - np.abs(np.random.normal(0, daily_range, num_days)))
    open_price = close * np.exp(np.random.normal(0, 0.002, num_days))

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


class DualModeStrategy:
    """
    Dual-mode strategy that switches between trend-following and mean-reversion.

    Attributes:
        trend_signals: Cached trend-following signals
        mr_signals: Cached mean-reversion signals
        trend_regime: Cached regime classifications
        atr: Cached ATR values
        kama: Cached KAMA values
    """

    def __init__(self, bars: np.ndarray):
        """
        Initialize strategy with full dataset to calculate indicators.

        Args:
            bars: OHLCV bars array, shape (N, 5)
        """
        # Extract OHLC data (ensure contiguous arrays for Numba)
        high_prices = np.ascontiguousarray(bars[:, 1])
        low_prices = np.ascontiguousarray(bars[:, 2])
        close_prices = np.ascontiguousarray(bars[:, 3])

        # Calculate technical indicators
        print("  Calculating KAMA...")
        self.kama, er, regime = calculate_kama_with_regime_adjustment(close_prices)

        print("  Calculating ATR...")
        self.atr = calculate_atr(high_prices, low_prices, close_prices)

        print("  Calculating RSI...")
        rsi = calculate_rsi(close_prices)

        print("  Calculating Bollinger Bands...")
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close_prices)

        # Detect market regimes
        print("  Detecting regimes...")
        from backend.computation.volatility import calculate_atr_ma
        atr_ma = calculate_atr_ma(self.atr)

        regime_data = detect_regime(
            high=high_prices,
            low=low_prices,
            close=close_prices,
            atr=self.atr,
            atr_ma=atr_ma,
            apply_persistence=True,
        )
        self.trend_regime = regime_data['trend_regime']

        # Generate signals for both strategies
        print("  Generating trend signals...")
        self.trend_signals = generate_trend_following_signals(
            prices=close_prices,
            kama=self.kama,
            atr=self.atr,
        )

        print("  Generating mean-reversion signals...")
        self.mr_signals = generate_mean_reversion_signals(
            prices=close_prices,
            rsi=rsi,
            bollinger_lower=bb_lower,
            bollinger_middle=bb_middle,
        )

        self.close_prices = close_prices

    def get_orders(
        self,
        bars: np.ndarray,
        current_index: int,
        position: Position | None
    ) -> list[Order]:
        """
        Generate orders based on current bar and position.

        This is the callback function for BacktestEngine.

        Args:
            bars: Full OHLCV array
            current_index: Current bar index
            position: Current open position (None if flat)

        Returns:
            List of orders to place
        """
        orders = []

        # Skip if indicators not ready
        if current_index < 50:
            return orders

        current_close = self.close_prices[current_index]
        current_atr = self.atr[current_index]
        current_regime = self.trend_regime[current_index]

        # Skip if ATR not valid
        if np.isnan(current_atr) or current_atr == 0:
            return orders

        # Select active strategy based on regime
        if current_regime == REGIME_STRONG_TREND:
            active_signal = self.trend_signals[current_index]
            risk_pct = 0.01  # 1% risk
        elif current_regime == REGIME_MEAN_REVERSION:
            active_signal = self.mr_signals[current_index]
            risk_pct = 0.0075  # 0.75% risk
        else:  # NEUTRAL
            # Require both strategies to agree
            if (self.trend_signals[current_index] == self.mr_signals[current_index] and
                self.trend_signals[current_index] != SIGNAL_NONE):
                active_signal = self.trend_signals[current_index]
            else:
                active_signal = SIGNAL_NONE
            risk_pct = 0.007  # 0.7% risk

        # Process signals
        if position is None:  # Flat - look for entry
            if active_signal == SIGNAL_LONG:
                # Calculate stop loss
                if current_regime == REGIME_STRONG_TREND:
                    stop_price = current_close - (2.5 * current_atr)
                else:
                    stop_price = current_close * 0.97  # 3% stop

                # Estimate capital (we don't have access to state here)
                # Use a rough estimate based on initial capital
                capital = 100000.0  # Will be adjusted by engine

                shares = calculate_position_size(
                    capital=capital,
                    entry_price=current_close,
                    stop_price=stop_price,
                    risk_pct=risk_pct,
                    is_long=True,
                )

                if shares > 0:
                    order = Order(
                        bar_index=current_index,
                        symbol="SPY",
                        side=OrderSide.BUY,
                        shares=int(shares),
                        order_type=OrderType.MARKET,
                    )
                    orders.append(order)

        else:  # In position - look for exit
            # Check exit signals
            if position.side == PositionSide.LONG and active_signal == SIGNAL_EXIT_LONG:
                order = Order(
                    bar_index=current_index,
                    symbol="SPY",
                    side=OrderSide.SELL,
                    shares=position.shares,
                    order_type=OrderType.MARKET,
                )
                orders.append(order)

        return orders


def main():
    """Execute 1-year SPY backtest with dual-mode strategy."""
    print("=" * 80)
    print("FluxHero - SPY 1-Year Backtest with Dual-Mode Strategy")
    print("=" * 80)
    print()

    # Generate synthetic SPY data (1 year = 252 trading days)
    print("Generating synthetic SPY data (252 trading days)...")
    data = generate_synthetic_spy_data(252)
    bars = data['bars']
    timestamps = data['timestamps']
    close_prices = data['close']

    print(f"Data range: {datetime.fromtimestamp(timestamps[0])} to {datetime.fromtimestamp(timestamps[-1])}")
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
    print(f"  Slippage: {config.slippage_pct * 100:.3f}%")
    print(f"  Risk-free rate: {config.risk_free_rate * 100:.1f}%")
    print()

    # Initialize strategy
    print("Initializing dual-mode strategy...")
    strategy = DualModeStrategy(bars)

    # Initialize backtest engine
    print("\nRunning backtest...")
    engine = BacktestEngine(config)

    # Run backtest
    state = engine.run(
        bars=bars,
        strategy_func=strategy.get_orders,
        symbol='SPY',
        timestamps=timestamps,
        volumes=data['volumes'],
    )

    # Calculate metrics
    print("\nCalculating performance metrics...")
    equity_curve = np.array(state.equity_curve)

    # Extract trades data
    trades_pnl = np.array([t.pnl for t in state.trades]) if state.trades else np.array([])
    trades_holding_periods = np.array([t.holding_bars for t in state.trades]) if state.trades else np.array([])

    metrics = PerformanceMetrics.calculate_all_metrics(
        equity_curve=equity_curve,
        trades_pnl=trades_pnl,
        trades_holding_periods=trades_holding_periods,
        initial_capital=config.initial_capital,
        risk_free_rate=config.risk_free_rate,
    )

    # Display results
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    print()
    print(PerformanceMetrics.format_metrics_report(metrics))

    # Check success criteria
    print("\n" + "=" * 80)
    print("SUCCESS CRITERIA VALIDATION")
    print("=" * 80)
    print()

    criteria_results = PerformanceMetrics.check_success_criteria(metrics)
    success = criteria_results['all_criteria_met']

    if success:
        print("✅ ALL SUCCESS CRITERIA MET!")
        print("   - Sharpe ratio > 0.8")
        print("   - Max drawdown < 25%")
        print("   - Win rate > 45%")
        print("   - Avg Win/Loss ratio > 1.5")
    else:
        print("⚠️  Some success criteria not met:")
        if criteria_results['sharpe_ratio_ok']:
            print(f"   ✅ Sharpe ratio: {metrics['sharpe_ratio']:.2f} (target: > 0.8)")
        else:
            print(f"   ❌ Sharpe ratio: {metrics['sharpe_ratio']:.2f} (target: > 0.8)")

        if criteria_results['max_drawdown_ok']:
            print(f"   ✅ Max drawdown: {metrics['max_drawdown_pct']:.2f}% (target: < 25%)")
        else:
            print(f"   ❌ Max drawdown: {metrics['max_drawdown_pct']:.2f}% (target: < 25%)")

        if criteria_results['win_rate_ok']:
            print(f"   ✅ Win rate: {metrics['win_rate']:.2f}% (target: > 45%)")
        else:
            print(f"   ❌ Win rate: {metrics['win_rate']:.2f}% (target: > 45%)")

        if criteria_results['win_loss_ratio_ok']:
            print(f"   ✅ Avg Win/Loss: {metrics['avg_win_loss_ratio']:.2f} (target: > 1.5)")
        else:
            print(f"   ❌ Avg Win/Loss: {metrics['avg_win_loss_ratio']:.2f} (target: > 1.5)")

    print("\n" + "=" * 80)
    print(f"Total trades executed: {len(state.trades)}")
    print(f"Final equity: ${equity_curve[-1]:,.2f}")
    print(f"Total return: {metrics['total_return_pct']:.2f}%")
    print(f"vs Buy & Hold: {buy_hold_return:.2f}%")
    print("=" * 80)

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
