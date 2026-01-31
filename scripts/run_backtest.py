#!/usr/bin/env python3
"""
Easy-to-Use Backtest Runner with Yahoo Finance Data.

Fetches real market data and runs backtests with clear comparisons.
Supports multiple strategy selection.

Usage:
    # Basic backtest - shows interactive strategy menu
    python scripts/run_backtest.py

    # Select strategy via CLI (by number or name)
    python scripts/run_backtest.py --strategy 1          # Dual-Mode (default)
    python scripts/run_backtest.py --strategy 2          # Trend-Following Only
    python scripts/run_backtest.py --strategy 3          # Mean-Reversion Only
    python scripts/run_backtest.py --strategy 4          # AMM System
    python scripts/run_backtest.py --strategy 5          # Golden Adaptive

    # Or by name
    python scripts/run_backtest.py --strategy dual_mode
    python scripts/run_backtest.py --strategy trend_only

    # Skip menu and use default strategy
    python scripts/run_backtest.py --no-menu

    # Specific symbol and date range
    python scripts/run_backtest.py --symbol AAPL --start 2023-01-01

    # Compare multiple symbols
    python scripts/run_backtest.py --symbols SPY,QQQ,IWM

    # Quick test mode (last 6 months)
    python scripts/run_backtest.py --quick

    # Full analysis with report
    python scripts/run_backtest.py --full --report --monte-carlo

    # Compare ALL strategies on same symbol
    python scripts/run_backtest.py --all-strategies --symbol SPY
    python scripts/run_backtest.py --all-strategies --report  # With combined chart

    # Using Makefile
    make backtest-live              # SPY, 1 year
    make backtest-live SYMBOL=AAPL  # Custom symbol
    make backtest-all               # Compare all strategies

Available Strategies:
    [1] Dual-Mode       - Switches trend/mean-reversion based on regime
    [2] Trend-Only      - Pure trend-following using KAMA + ATR
    [3] Mean-Reversion  - Pure RSI + Bollinger Bands
    [4] AMM             - Multi-indicator weighted system
    [5] Golden          - 4-dimensional confidence-weighted system
"""

import argparse
import sys
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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
from backend.data.yahoo_fetcher import YahooFinanceFetcher, YahooFinanceError  # noqa: E402
from backend.strategy.dual_mode import (  # noqa: E402
    SIGNAL_EXIT_LONG,
    SIGNAL_EXIT_SHORT,
    SIGNAL_LONG,
    SIGNAL_NONE,
    SIGNAL_SHORT,
    calculate_position_size,
    generate_mean_reversion_signals,
    generate_trend_following_signals,
)
from backend.strategy.regime_detector import (  # noqa: E402
    REGIME_MEAN_REVERSION,
    REGIME_NEUTRAL,
    REGIME_STRONG_TREND,
    detect_regime,
)

# =============================================================================
# AVAILABLE STRATEGIES
# =============================================================================

STRATEGIES = {
    "1": {
        "name": "dual_mode",
        "title": "Dual-Mode Strategy",
        "description": "Switches between trend-following (KAMA+ATR) and mean-reversion (RSI+BB) based on market regime",
    },
    "2": {
        "name": "trend_only",
        "title": "Trend-Following Only",
        "description": "Pure trend-following using KAMA crossovers with ATR-based entries and trailing stops",
    },
    "3": {
        "name": "mean_reversion",
        "title": "Mean-Reversion Only",
        "description": "Pure mean-reversion using RSI oversold/overbought with Bollinger Band entries",
    },
    "4": {
        "name": "amm",
        "title": "AMM (Adaptive Market Measure)",
        "description": "Multi-indicator weighted system: SMA + RSI + Momentum + Bollinger with Z-score normalization",
    },
    "5": {
        "name": "golden",
        "title": "Golden Adaptive (4D System)",
        "description": "4-dimensional system using Fractal + Efficiency + Volatility + Volume with confidence-weighted sizing",
    },
}


def print_strategy_menu():
    """Print available strategies menu."""
    print("\n" + "=" * 70)
    print("AVAILABLE STRATEGIES")
    print("=" * 70)
    for key, info in STRATEGIES.items():
        print(f"\n  [{key}] {info['title']}")
        print(f"      {info['description']}")
    print("\n" + "=" * 70)


def get_strategy_choice() -> str:
    """Prompt user to select a strategy."""
    print_strategy_menu()
    while True:
        choice = input("\nSelect strategy (1-5) or press Enter for default [1]: ").strip()
        if choice == "":
            return "1"
        if choice in STRATEGIES:
            return choice
        print(f"Invalid choice '{choice}'. Please enter 1-5.")

# Optional analytics imports for HTML reports
try:
    from backend.analytics import (
        QuantStatsAdapter,
        TearsheetGenerator,
        run_monte_carlo_analysis,
        MonteCarloSimulator,
    )
    import quantstats as qs
    HAS_ANALYTICS = True
except ImportError:
    HAS_ANALYTICS = False
    qs = None


def run_monte_carlo_validation(
    result: dict,
    n_simulations: int = 5000,
    verbose: bool = True,
) -> dict:
    """
    Run Monte Carlo validation to check if strategy beats random.

    This is a critical test: if a strategy can't beat random resampling
    of its own returns, it has no real edge.

    Args:
        result: Backtest result dict
        n_simulations: Number of Monte Carlo simulations
        verbose: Print detailed output

    Returns:
        dict with validation results
    """
    if not HAS_ANALYTICS:
        return {"error": "Analytics module not available"}

    # Calculate returns from equity curve
    equity = np.array(result['equity_curve'])
    returns = np.diff(equity) / equity[:-1]
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    if len(returns) < 20:
        return {"error": "Not enough data for Monte Carlo validation"}

    # Run Monte Carlo simulation
    simulator = MonteCarloSimulator(returns)
    mc_result = simulator.run(
        n_simulations=n_simulations,
        n_periods=len(returns),
        initial_capital=result['initial_capital'],
        seed=42,
    )

    # Strategy's actual performance
    strategy_return = result['total_return_pct'] / 100.0  # Convert to decimal

    # Comparison metrics
    beats_median = strategy_return > mc_result.median_return
    beats_75th = strategy_return > mc_result.percentile_75
    percentile_rank = float(np.mean(mc_result.final_equities < result['final_equity'])) * 100

    # Edge calculation: how much better than random median
    edge_vs_random = (strategy_return - mc_result.median_return) * 100  # In percentage points

    # Statistical significance: strategy should beat at least 60% of random paths
    is_significant = percentile_rank > 60

    # Final verdict
    has_edge = beats_median and is_significant

    validation = {
        # Strategy metrics
        "strategy_return_pct": result['total_return_pct'],
        "strategy_sharpe": result['sharpe_ratio'],

        # Monte Carlo metrics
        "mc_median_return_pct": mc_result.median_return * 100,
        "mc_mean_return_pct": mc_result.mean_return * 100,
        "mc_5th_percentile_pct": mc_result.percentile_5 * 100,
        "mc_25th_percentile_pct": mc_result.percentile_25 * 100,
        "mc_75th_percentile_pct": mc_result.percentile_75 * 100,
        "mc_95th_percentile_pct": mc_result.percentile_95 * 100,

        # Comparison
        "beats_median": beats_median,
        "beats_75th_percentile": beats_75th,
        "percentile_rank": percentile_rank,
        "edge_vs_random_pct": edge_vs_random,
        "is_statistically_significant": is_significant,

        # Verdict
        "has_edge": has_edge,
        "verdict": "STRATEGY HAS EDGE" if has_edge else "NO EDGE (Strategy is random)",

        # Raw result for further analysis
        "mc_result": mc_result,
        "n_simulations": n_simulations,
    }

    if verbose:
        print("\n" + "=" * 60)
        print("MONTE CARLO VALIDATION")
        print("=" * 60)
        print(f"Simulations: {n_simulations:,}")
        print()
        print("Strategy vs Random Comparison:")
        print("-" * 40)
        print(f"  Strategy Return:      {result['total_return_pct']:>+8.2f}%")
        print(f"  MC Median Return:     {mc_result.median_return * 100:>+8.2f}%")
        print(f"  MC 75th Percentile:   {mc_result.percentile_75 * 100:>+8.2f}%")
        print()
        print(f"  Edge vs Random:       {edge_vs_random:>+8.2f}%")
        print(f"  Percentile Rank:      {percentile_rank:>8.1f}%")
        print("    (Strategy beats X% of random simulations)")
        print()
        print("Monte Carlo Distribution:")
        print("-" * 40)
        print(f"  5th percentile:       {mc_result.percentile_5 * 100:>+8.2f}%")
        print(f"  25th percentile:      {mc_result.percentile_25 * 100:>+8.2f}%")
        print(f"  Median (50th):        {mc_result.median_return * 100:>+8.2f}%")
        print(f"  75th percentile:      {mc_result.percentile_75 * 100:>+8.2f}%")
        print(f"  95th percentile:      {mc_result.percentile_95 * 100:>+8.2f}%")
        print()
        print("Validation:")
        print("-" * 40)

        # Beats median?
        status = "PASS" if beats_median else "FAIL"
        print(f"  [{'+' if beats_median else '-'}] Beats MC Median: {status}")

        # Beats 75th percentile? (strong edge)
        status = "PASS" if beats_75th else "FAIL"
        print(f"  [{'+' if beats_75th else '-'}] Beats MC 75th Percentile: {status}")

        # Statistically significant?
        status = "PASS" if is_significant else "FAIL"
        print(f"  [{'+' if is_significant else '-'}] Statistically Significant (>60%): {status}")

        print()
        if has_edge:
            print("  VERDICT: STRATEGY HAS EDGE")
            print(f"           Beats {percentile_rank:.1f}% of random simulations")
        else:
            print("  VERDICT: NO EDGE DETECTED")
            print("           Strategy performance is indistinguishable from random")
        print("=" * 60)

    return validation


class BaseStrategy:
    """Base class for all strategies."""

    def __init__(self, bars: np.ndarray):
        self.bars = bars
        high_prices = np.ascontiguousarray(bars[:, 1])
        low_prices = np.ascontiguousarray(bars[:, 2])
        close_prices = np.ascontiguousarray(bars[:, 3])

        self.high_prices = high_prices
        self.low_prices = low_prices
        self.close_prices = close_prices
        self.symbol = "UNKNOWN"

        # Calculate common indicators
        self.kama, self.er, _ = calculate_kama_with_regime_adjustment(close_prices)
        self.atr = calculate_atr(high_prices, low_prices, close_prices)
        self.rsi = calculate_rsi(close_prices)
        self.bb_upper, self.bb_middle, self.bb_lower = calculate_bollinger_bands(close_prices)

        # Detect regimes
        from backend.computation.volatility import calculate_atr_ma
        atr_ma = calculate_atr_ma(self.atr)
        regime_data = detect_regime(
            high=high_prices, low=low_prices, close=close_prices,
            atr=self.atr, atr_ma=atr_ma, apply_persistence=True,
        )
        self.trend_regime = regime_data['trend_regime']

    def get_orders(self, bars: np.ndarray, current_index: int, position: Position | None) -> list[Order]:
        raise NotImplementedError


class DualModeStrategy(BaseStrategy):
    """Dual-mode strategy: switches between trend-following and mean-reversion based on regime."""

    def __init__(self, bars: np.ndarray, **kwargs):
        super().__init__(bars)

        # Generate both signal types
        self.trend_signals = generate_trend_following_signals(
            prices=self.close_prices, kama=self.kama, atr=self.atr,
        )
        self.mr_signals = generate_mean_reversion_signals(
            prices=self.close_prices, rsi=self.rsi,
            bollinger_lower=self.bb_lower, bollinger_middle=self.bb_middle,
            bollinger_upper=self.bb_upper,
            rsi_oversold=kwargs.get('rsi_oversold', 35.0),
            rsi_overbought=kwargs.get('rsi_overbought', 65.0),
            bb_entry_threshold=kwargs.get('bb_entry_threshold', 0.15),
            use_or_logic=kwargs.get('use_or_logic', True),
        )

    def get_orders(self, bars: np.ndarray, current_index: int, position: Position | None) -> list[Order]:
        orders = []
        if current_index < 50:
            return orders

        current_close = self.close_prices[current_index]
        current_atr = self.atr[current_index]
        current_regime = self.trend_regime[current_index]

        if np.isnan(current_atr) or current_atr == 0:
            return orders

        # Select strategy based on regime
        if current_regime == REGIME_STRONG_TREND:
            active_signal = self.trend_signals[current_index]
            risk_pct, stop_mult = 0.01, 2.5
        elif current_regime == REGIME_MEAN_REVERSION:
            active_signal = self.mr_signals[current_index]
            risk_pct, stop_mult = 0.0075, None  # Fixed 3% stop
        else:
            if self.trend_signals[current_index] == self.mr_signals[current_index] != SIGNAL_NONE:
                active_signal = self.trend_signals[current_index]
            else:
                active_signal = SIGNAL_NONE
            risk_pct, stop_mult = 0.007, 2.0

        return self._generate_orders(current_index, position, active_signal, risk_pct, stop_mult)

    def _generate_orders(self, idx, position, signal, risk_pct, stop_mult):
        orders = []
        close = self.close_prices[idx]
        atr = self.atr[idx]

        if position is None and signal == SIGNAL_LONG:
            stop = close - (stop_mult * atr) if stop_mult else close * 0.97
            shares = int(calculate_position_size(100000.0, close, stop, risk_pct, True))
            if shares > 0:
                orders.append(Order(idx, self.symbol, OrderSide.BUY, shares, OrderType.MARKET))
        elif position is None and signal == SIGNAL_SHORT:
            stop = close + (stop_mult * atr) if stop_mult else close * 1.03
            shares = int(calculate_position_size(100000.0, close, stop, risk_pct, False))
            if shares > 0:
                orders.append(Order(idx, self.symbol, OrderSide.SELL, shares, OrderType.MARKET))
        elif position and position.side == PositionSide.LONG and signal == SIGNAL_EXIT_LONG:
            orders.append(Order(idx, self.symbol, OrderSide.SELL, position.shares, OrderType.MARKET))
        elif position and position.side == PositionSide.SHORT and signal == SIGNAL_EXIT_SHORT:
            orders.append(Order(idx, self.symbol, OrderSide.BUY, position.shares, OrderType.MARKET))

        return orders


class TrendOnlyStrategy(BaseStrategy):
    """Pure trend-following strategy using KAMA + ATR breakouts."""

    def __init__(self, bars: np.ndarray, **kwargs):
        super().__init__(bars)
        self.signals = generate_trend_following_signals(
            prices=self.close_prices, kama=self.kama, atr=self.atr,
            entry_multiplier=kwargs.get('entry_mult', 0.5),
            exit_multiplier=kwargs.get('exit_mult', 0.3),
        )

    def get_orders(self, bars: np.ndarray, current_index: int, position: Position | None) -> list[Order]:
        orders = []
        if current_index < 50:
            return orders

        close = self.close_prices[current_index]
        atr = self.atr[current_index]
        signal = self.signals[current_index]

        if np.isnan(atr) or atr == 0:
            return orders

        if position is None and signal == SIGNAL_LONG:
            stop = close - (2.5 * atr)
            shares = int(calculate_position_size(100000.0, close, stop, 0.01, True))
            if shares > 0:
                orders.append(Order(current_index, self.symbol, OrderSide.BUY, shares, OrderType.MARKET))
        elif position is None and signal == SIGNAL_SHORT:
            stop = close + (2.5 * atr)
            shares = int(calculate_position_size(100000.0, close, stop, 0.01, False))
            if shares > 0:
                orders.append(Order(current_index, self.symbol, OrderSide.SELL, shares, OrderType.MARKET))
        elif position and position.side == PositionSide.LONG and signal == SIGNAL_EXIT_LONG:
            orders.append(Order(current_index, self.symbol, OrderSide.SELL, position.shares, OrderType.MARKET))
        elif position and position.side == PositionSide.SHORT and signal == SIGNAL_EXIT_SHORT:
            orders.append(Order(current_index, self.symbol, OrderSide.BUY, position.shares, OrderType.MARKET))

        return orders


class MeanReversionStrategy(BaseStrategy):
    """Pure mean-reversion strategy using RSI + Bollinger Bands."""

    def __init__(self, bars: np.ndarray, **kwargs):
        super().__init__(bars)
        self.signals = generate_mean_reversion_signals(
            prices=self.close_prices, rsi=self.rsi,
            bollinger_lower=self.bb_lower, bollinger_middle=self.bb_middle,
            bollinger_upper=self.bb_upper,
            rsi_oversold=kwargs.get('rsi_oversold', 35.0),
            rsi_overbought=kwargs.get('rsi_overbought', 65.0),
            bb_entry_threshold=kwargs.get('bb_entry_threshold', 0.15),
            use_or_logic=kwargs.get('use_or_logic', True),
        )

    def get_orders(self, bars: np.ndarray, current_index: int, position: Position | None) -> list[Order]:
        orders = []
        if current_index < 50:
            return orders

        close = self.close_prices[current_index]
        signal = self.signals[current_index]

        if position is None and signal == SIGNAL_LONG:
            stop = close * 0.97  # 3% fixed stop
            shares = int(calculate_position_size(100000.0, close, stop, 0.0075, True))
            if shares > 0:
                orders.append(Order(current_index, self.symbol, OrderSide.BUY, shares, OrderType.MARKET))
        elif position is None and signal == SIGNAL_SHORT:
            stop = close * 1.03
            shares = int(calculate_position_size(100000.0, close, stop, 0.0075, False))
            if shares > 0:
                orders.append(Order(current_index, self.symbol, OrderSide.SELL, shares, OrderType.MARKET))
        elif position and position.side == PositionSide.LONG and signal == SIGNAL_EXIT_LONG:
            orders.append(Order(current_index, self.symbol, OrderSide.SELL, position.shares, OrderType.MARKET))
        elif position and position.side == PositionSide.SHORT and signal == SIGNAL_EXIT_SHORT:
            orders.append(Order(current_index, self.symbol, OrderSide.BUY, position.shares, OrderType.MARKET))

        return orders


class AMMStrategyWrapper(BaseStrategy):
    """AMM (Adaptive Market Measure) - Multi-indicator weighted system."""

    def __init__(self, bars: np.ndarray, **kwargs):
        super().__init__(bars)
        try:
            from backend.amm_system.strategy import AMMStrategy, AMMConfig
            config = AMMConfig()
            self._amm = AMMStrategy(bars=bars, initial_capital=100000.0, config=config)
            self._available = True
        except ImportError:
            self._available = False

    def get_orders(self, bars: np.ndarray, current_index: int, position: Position | None) -> list[Order]:
        if not self._available:
            return []
        return self._amm.get_orders(bars, current_index, position)


class GoldenStrategyWrapper(BaseStrategy):
    """Golden Adaptive (4D System) - Confidence-weighted multidimensional strategy."""

    def __init__(self, bars: np.ndarray, **kwargs):
        super().__init__(bars)
        try:
            from backend.golden_system.strategy import GoldenAdaptiveStrategy
            self._golden = GoldenAdaptiveStrategy(bars=bars, initial_capital=100000.0)
            self._available = True
        except ImportError:
            self._available = False

    def get_orders(self, bars: np.ndarray, current_index: int, position: Position | None) -> list[Order]:
        if not self._available:
            return []
        return self._golden.get_orders(bars, current_index, position)


def create_strategy(strategy_name: str, bars: np.ndarray, **kwargs) -> BaseStrategy:
    """Factory function to create strategy by name."""
    strategies = {
        "dual_mode": DualModeStrategy,
        "trend_only": TrendOnlyStrategy,
        "mean_reversion": MeanReversionStrategy,
        "amm": AMMStrategyWrapper,
        "golden": GoldenStrategyWrapper,
    }

    strategy_class = strategies.get(strategy_name, DualModeStrategy)
    return strategy_class(bars, **kwargs)


def run_backtest(
    symbol: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 100000.0,
    verbose: bool = True,
    show_diagnostics: bool = False,
    run_monte_carlo: bool = False,
    mc_simulations: int = 5000,
    strategy_name: str = "dual_mode",
) -> dict:
    """
    Run backtest on a symbol with real Yahoo Finance data.

    Returns dict with metrics and comparison data.
    """
    # Fetch data
    fetcher = YahooFinanceFetcher()

    if verbose:
        print(f"\nFetching {symbol} data from Yahoo Finance...")

    try:
        data = fetcher.fetch_historical_data(symbol, start_date, end_date)
    except YahooFinanceError as e:
        print(f"Error fetching data: {e}")
        return None

    bars = data['bars']
    timestamps = data['timestamps']
    dates = data['dates']

    if verbose:
        print(f"  Loaded {len(bars)} trading days")
        print(f"  Date range: {dates[0]} to {dates[-1]}")
        print(f"  Price range: ${bars[0, 3]:.2f} to ${bars[-1, 3]:.2f}")

    # Calculate buy-and-hold return
    buy_hold_return = ((bars[-1, 3] / bars[0, 3]) - 1) * 100

    # Configure backtest
    config = BacktestConfig(
        initial_capital=initial_capital,
        commission_per_share=0.005,
        slippage_pct=0.0001,
        impact_threshold=0.1,
        impact_penalty_pct=0.0005,
        risk_free_rate=0.04,
    )

    # Initialize strategy
    strategy_info = None
    for key, info in STRATEGIES.items():
        if info['name'] == strategy_name:
            strategy_info = info
            break

    if verbose:
        strategy_title = strategy_info['title'] if strategy_info else strategy_name
        print(f"\nInitializing strategy: {strategy_title}...")

    strategy = create_strategy(strategy_name, bars)
    strategy.symbol = symbol

    # Show diagnostics if requested
    if show_diagnostics:
        print("\n" + "-" * 40)
        print("SIGNAL DIAGNOSTICS")
        print("-" * 40)

        # Count signals
        trend_long = np.sum(strategy.trend_signals == SIGNAL_LONG)
        trend_exit = np.sum(strategy.trend_signals == SIGNAL_EXIT_LONG)
        mr_long = np.sum(strategy.mr_signals == SIGNAL_LONG)
        mr_exit = np.sum(strategy.mr_signals == SIGNAL_EXIT_LONG)

        print(f"  Trend signals: {trend_long} entries, {trend_exit} exits")
        print(f"  Mean-rev signals: {mr_long} entries, {mr_exit} exits")

        # Regime breakdown
        regime_trend = np.sum(strategy.trend_regime == REGIME_STRONG_TREND)
        regime_mr = np.sum(strategy.trend_regime == REGIME_MEAN_REVERSION)
        regime_neutral = np.sum(strategy.trend_regime == REGIME_NEUTRAL)
        total = len(strategy.trend_regime)

        print(f"\n  Regime breakdown:")
        print(f"    Strong Trend:    {regime_trend:4d} bars ({regime_trend/total*100:5.1f}%)")
        print(f"    Mean Reversion:  {regime_mr:4d} bars ({regime_mr/total*100:5.1f}%)")
        print(f"    Neutral:         {regime_neutral:4d} bars ({regime_neutral/total*100:5.1f}%)")

        # RSI stats
        close_prices = np.ascontiguousarray(bars[:, 3])
        rsi = calculate_rsi(close_prices)
        valid_rsi = rsi[~np.isnan(rsi)]
        if len(valid_rsi) > 0:
            print(f"\n  RSI stats:")
            print(f"    Range: {np.min(valid_rsi):.1f} - {np.max(valid_rsi):.1f}")
            print(f"    Bars with RSI < 35: {np.sum(valid_rsi < 35)}")
            print(f"    Bars with RSI > 65: {np.sum(valid_rsi > 65)}")

    # Run backtest
    if verbose:
        print("Running backtest...")

    engine = BacktestEngine(config)
    state = engine.run(
        bars=bars,
        strategy_func=strategy.get_orders,
        symbol=symbol,
        timestamps=timestamps,
        volumes=bars[:, 4],
    )

    # Calculate metrics
    equity_curve = np.array(state.equity_curve)
    trades_pnl = np.array([t.pnl for t in state.trades]) if state.trades else np.array([])
    trades_holding = np.array([t.holding_bars for t in state.trades]) if state.trades else np.array([])

    metrics = PerformanceMetrics.calculate_all_metrics(
        equity_curve=equity_curve,
        trades_pnl=trades_pnl,
        trades_holding_periods=trades_holding,
        initial_capital=initial_capital,
        risk_free_rate=config.risk_free_rate,
    )

    # Build result
    result = {
        'symbol': symbol,
        'strategy_name': strategy_name,
        'strategy_title': strategy_info['title'] if strategy_info else strategy_name,
        'start_date': dates[0],
        'end_date': dates[-1],
        'trading_days': len(bars),
        'initial_capital': initial_capital,
        'final_equity': equity_curve[-1],
        'total_return_pct': metrics['total_return_pct'],
        'buy_hold_return_pct': buy_hold_return,
        'excess_return_pct': metrics['total_return_pct'] - buy_hold_return,
        'sharpe_ratio': metrics['sharpe_ratio'],
        'max_drawdown_pct': metrics['max_drawdown_pct'],
        'total_trades': len(state.trades),
        'win_rate': metrics['win_rate'],
        'avg_win_loss_ratio': metrics['avg_win_loss_ratio'],
        'metrics': metrics,
        'equity_curve': equity_curve,
        'trades': state.trades,
        'mc_validation': None,
    }

    # Run Monte Carlo validation if requested
    if run_monte_carlo and HAS_ANALYTICS:
        mc_validation = run_monte_carlo_validation(
            result=result,
            n_simulations=mc_simulations,
            verbose=verbose,
        )
        result['mc_validation'] = mc_validation

    return result


def print_results(result: dict, show_details: bool = True):
    """Print formatted backtest results."""
    if result is None:
        return

    strategy_title = result.get('strategy_title', result.get('strategy_name', 'Unknown'))
    print("\n" + "=" * 70)
    print(f"BACKTEST RESULTS: {result['symbol']} | {strategy_title}")
    print("=" * 70)

    print(f"\nPeriod: {result['start_date']} to {result['end_date']} ({result['trading_days']} days)")

    print("\n" + "-" * 40)
    print("RETURNS COMPARISON")
    print("-" * 40)
    print(f"  Strategy Return:    {result['total_return_pct']:+8.2f}%")
    print(f"  Buy & Hold Return:  {result['buy_hold_return_pct']:+8.2f}%")
    print(f"  Excess Return:      {result['excess_return_pct']:+8.2f}%")

    # Determine winner
    if result['total_return_pct'] > result['buy_hold_return_pct']:
        print(f"  --> Strategy OUTPERFORMED by {result['excess_return_pct']:.2f}%")
    elif result['total_return_pct'] < result['buy_hold_return_pct']:
        print(f"  --> Strategy UNDERPERFORMED by {-result['excess_return_pct']:.2f}%")
    else:
        print(f"  --> Strategy MATCHED buy & hold")

    print("\n" + "-" * 40)
    print("RISK METRICS")
    print("-" * 40)
    print(f"  Sharpe Ratio:       {result['sharpe_ratio']:+8.2f}")
    print(f"  Max Drawdown:       {result['max_drawdown_pct']:8.2f}%")

    print("\n" + "-" * 40)
    print("TRADE STATISTICS")
    print("-" * 40)
    print(f"  Total Trades:       {result['total_trades']:8d}")
    print(f"  Win Rate:           {result['win_rate']:8.1f}%")
    if result['avg_win_loss_ratio'] > 0:
        print(f"  Avg Win/Loss:       {result['avg_win_loss_ratio']:8.2f}x")

    print("\n" + "-" * 40)
    print("CAPITAL")
    print("-" * 40)
    print(f"  Initial:            ${result['initial_capital']:,.2f}")
    print(f"  Final:              ${result['final_equity']:,.2f}")
    print(f"  P&L:                ${result['final_equity'] - result['initial_capital']:+,.2f}")

    # Success criteria check
    print("\n" + "-" * 40)
    print("SUCCESS CRITERIA")
    print("-" * 40)

    checks = [
        ("Sharpe > 0.8", result['sharpe_ratio'] > 0.8),
        ("Max DD < 25%", abs(result['max_drawdown_pct']) < 25),
        ("Win Rate > 45%", result['win_rate'] > 45),
        ("Beat Buy&Hold", result['total_return_pct'] > result['buy_hold_return_pct']),
    ]

    # Add Monte Carlo validation if available
    mc_validation = result.get('mc_validation')
    if mc_validation and 'has_edge' in mc_validation:
        checks.append(("Beats Random (MC)", mc_validation['has_edge']))

    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        symbol = "+" if passed else "-"
        print(f"  [{symbol}] {name}: {status}")

    passed_count = sum(1 for _, p in checks if p)
    print(f"\n  Score: {passed_count}/{len(checks)} criteria met")

    # Show MC summary if available
    if mc_validation and 'has_edge' in mc_validation:
        print()
        if mc_validation['has_edge']:
            print(f"  Monte Carlo: Strategy has edge (beats {mc_validation['percentile_rank']:.1f}% of random)")
        else:
            print(f"  Monte Carlo: NO EDGE - strategy is random noise")
            print(f"              (only beats {mc_validation['percentile_rank']:.1f}% of random simulations)")

    print("=" * 70)

    # Print QuantStats metrics if available
    if HAS_ANALYTICS:
        print_quantstats_metrics(result)


def print_quantstats_metrics(result: dict, benchmark: str = "SPY"):
    """
    Print key metrics using QuantStats calculations.

    This provides the same metrics that appear in the HTML report.
    """
    if not HAS_ANALYTICS or qs is None:
        return

    import pandas as pd

    # Calculate returns from equity curve
    equity = np.array(result['equity_curve'])
    returns = np.diff(equity) / equity[:-1]
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    if len(returns) < 5:
        return

    # Create pandas Series (QuantStats expects this)
    index = pd.date_range(end=pd.Timestamp.now(), periods=len(returns), freq='D')
    returns_series = pd.Series(returns, index=index, name="Strategy")

    print("\n" + "=" * 60)
    print("QUANTSTATS METRICS")
    print("=" * 60)

    try:
        # Core performance metrics
        print("\nPerformance:")
        print("-" * 40)
        print(f"  Total Return:         {qs.stats.comp(returns_series) * 100:>+8.2f}%")
        print(f"  CAGR:                 {qs.stats.cagr(returns_series) * 100:>+8.2f}%")
        print(f"  Sharpe Ratio:         {qs.stats.sharpe(returns_series):>+8.2f}")
        print(f"  Sortino Ratio:        {qs.stats.sortino(returns_series):>+8.2f}")
        print(f"  Calmar Ratio:         {qs.stats.calmar(returns_series):>+8.2f}")

        # Risk metrics
        print("\nRisk:")
        print("-" * 40)
        print(f"  Max Drawdown:         {qs.stats.max_drawdown(returns_series) * 100:>8.2f}%")
        print(f"  Volatility (Ann.):    {qs.stats.volatility(returns_series) * 100:>8.2f}%")
        print(f"  VaR (95%):            {qs.stats.var(returns_series) * 100:>8.2f}%")
        print(f"  CVaR (95%):           {qs.stats.cvar(returns_series) * 100:>8.2f}%")

        # Win/Loss stats
        print("\nTrade Statistics:")
        print("-" * 40)
        print(f"  Win Rate:             {qs.stats.win_rate(returns_series) * 100:>8.1f}%")
        print(f"  Win/Loss Ratio:       {qs.stats.win_loss_ratio(returns_series):>8.2f}")
        print(f"  Profit Factor:        {qs.stats.profit_factor(returns_series):>8.2f}")
        print(f"  Payoff Ratio:         {qs.stats.payoff_ratio(returns_series):>8.2f}")

        # Distribution
        print("\nDistribution:")
        print("-" * 40)
        print(f"  Skewness:             {qs.stats.skew(returns_series):>+8.2f}")
        print(f"  Kurtosis:             {qs.stats.kurtosis(returns_series):>+8.2f}")
        print(f"  Best Day:             {qs.stats.best(returns_series) * 100:>+8.2f}%")
        print(f"  Worst Day:            {qs.stats.worst(returns_series) * 100:>+8.2f}%")

        print("=" * 60)

    except Exception as e:
        print(f"  Error calculating QuantStats metrics: {e}")


def generate_html_report(
    result: dict,
    benchmark: str = "SPY",
    output_path: str | None = None,
    show_monte_carlo: bool = False,
    open_browser: bool = True,
) -> str | None:
    """
    Generate HTML tearsheet report using QuantStats.

    Args:
        result: Backtest result dict from run_backtest()
        benchmark: Benchmark symbol for comparison
        output_path: Custom output path (auto-generated if None)
        show_monte_carlo: Include Monte Carlo analysis

    Returns:
        Path to generated HTML file, or None if failed
    """
    if not HAS_ANALYTICS:
        print("Warning: Analytics module not available. Install quantstats:")
        print("  pip install quantstats")
        return None

    # Calculate returns from equity curve
    equity = np.array(result['equity_curve'])
    returns = np.diff(equity) / equity[:-1]

    # Clean any NaN/inf values
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    if len(returns) < 5:
        print("Warning: Not enough data for HTML report (need at least 5 periods)")
        return None

    # Generate filename if not provided
    if output_path is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"backtest_{result['symbol']}_{timestamp}.html"

    # Build title with strategy name and MC validation status if available
    strategy_title = result.get('strategy_title', result.get('strategy_name', 'Strategy'))
    mc_validation = result.get('mc_validation')
    if mc_validation and 'has_edge' in mc_validation:
        edge_status = "HAS EDGE" if mc_validation['has_edge'] else "NO EDGE"
        title = f"FluxHero: {result['symbol']} - {strategy_title} [{edge_status}]"
    else:
        title = f"FluxHero: {result['symbol']} - {strategy_title}"

    print(f"\nGenerating HTML report...")
    print(f"  Benchmark: {benchmark}")

    try:
        generator = TearsheetGenerator()
        report_path = generator.generate_tearsheet(
            returns=returns,
            benchmark_symbol=benchmark,
            title=title,
            output_filename=output_path,
        )

        print(f"  Report saved: {report_path}")

        # Open in browser
        if open_browser:
            report_url = f"file://{report_path.absolute()}"
            print(f"  Opening in browser...")
            webbrowser.open(report_url)

        # Show Monte Carlo summary if validation was run
        if mc_validation and 'has_edge' in mc_validation:
            print(f"\n  Monte Carlo Validation:")
            print(f"    Percentile Rank: {mc_validation['percentile_rank']:.1f}%")
            print(f"    Edge vs Random: {mc_validation['edge_vs_random_pct']:+.2f}%")
            print(f"    Verdict: {mc_validation['verdict']}")

        # Run additional Monte Carlo if requested and not already done
        if show_monte_carlo and not mc_validation:
            print("\nRunning Monte Carlo simulation...")
            run_monte_carlo_analysis(
                returns=returns,
                n_simulations=5000,
                n_periods=len(returns),
                initial_capital=result['initial_capital'],
                print_results=True,
            )

        return str(report_path)

    except Exception as e:
        print(f"  Error generating report: {e}")

        # Try without benchmark
        try:
            print("  Retrying without benchmark comparison...")
            report_path = generator.generate_tearsheet(
                returns=returns,
                benchmark_symbol=None,
                title=title,
                output_filename=output_path,
            )
            print(f"  Report saved: {report_path}")

            # Open in browser
            if open_browser:
                report_url = f"file://{report_path.absolute()}"
                print(f"  Opening in browser...")
                webbrowser.open(report_url)

            return str(report_path)
        except Exception as e2:
            print(f"  Failed to generate report: {e2}")
            return None


def print_comparison_table(results: list[dict]):
    """Print comparison table for multiple symbols."""
    if not results:
        return

    strategy_title = results[0].get('strategy_title', 'Strategy')
    print("\n" + "=" * 90)
    print(f"MULTI-SYMBOL COMPARISON | {strategy_title}")
    print("=" * 90)

    # Header
    print(f"\n{'Symbol':<8} {'Strategy':>10} {'Buy&Hold':>10} {'Excess':>10} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>7} {'WinRate':>8}")
    print("-" * 90)

    for r in results:
        print(f"{r['symbol']:<8} "
              f"{r['total_return_pct']:>+9.2f}% "
              f"{r['buy_hold_return_pct']:>+9.2f}% "
              f"{r['excess_return_pct']:>+9.2f}% "
              f"{r['sharpe_ratio']:>+7.2f} "
              f"{r['max_drawdown_pct']:>7.2f}% "
              f"{r['total_trades']:>7d} "
              f"{r['win_rate']:>7.1f}%")

    print("-" * 90)

    # Summary stats
    avg_excess = np.mean([r['excess_return_pct'] for r in results])
    avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
    beat_count = sum(1 for r in results if r['excess_return_pct'] > 0)

    print(f"\nSummary:")
    print(f"  Beat Buy&Hold: {beat_count}/{len(results)} symbols")
    print(f"  Avg Excess Return: {avg_excess:+.2f}%")
    print(f"  Avg Sharpe Ratio: {avg_sharpe:+.2f}")
    print("=" * 90)


def run_all_strategies(
    symbol: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 100000.0,
    verbose: bool = True,
    run_monte_carlo: bool = False,
) -> list[dict]:
    """
    Run ALL strategies on the same symbol and return results.

    Returns list of result dicts, one per strategy.
    """
    results = []

    for key, info in STRATEGIES.items():
        strategy_name = info['name']
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running [{key}] {info['title']}...")
            print(f"{'='*60}")

        try:
            result = run_backtest(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                verbose=False,  # Suppress individual verbose output
                show_diagnostics=False,
                run_monte_carlo=run_monte_carlo,
                strategy_name=strategy_name,
            )
            if result:
                results.append(result)
                if verbose:
                    print(f"  Return: {result['total_return_pct']:+.2f}%  |  "
                          f"Sharpe: {result['sharpe_ratio']:+.2f}  |  "
                          f"Trades: {result['total_trades']}")
        except Exception as e:
            if verbose:
                print(f"  Error: {e}")

    return results


def print_strategy_comparison_table(results: list[dict], sort_by: str = "sharpe_ratio"):
    """
    Print comparison table for all strategies, sorted by performance.

    Args:
        results: List of backtest results
        sort_by: Metric to sort by (sharpe_ratio, total_return_pct, excess_return_pct)
    """
    if not results:
        return

    # Sort results by specified metric (descending)
    sorted_results = sorted(results, key=lambda r: r.get(sort_by, 0), reverse=True)

    symbol = results[0].get('symbol', 'UNKNOWN')
    print("\n" + "=" * 100)
    print(f"STRATEGY COMPARISON: {symbol} (Sorted by {sort_by.replace('_', ' ').title()})")
    print("=" * 100)

    # Header
    print(f"\n{'Rank':<5} {'Strategy':<25} {'Return':>10} {'B&H':>10} {'Excess':>10} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>7} {'Win%':>7}")
    print("-" * 100)

    for rank, r in enumerate(sorted_results, 1):
        strategy_title = r.get('strategy_title', r.get('strategy_name', 'Unknown'))[:24]
        print(f"{rank:<5} "
              f"{strategy_title:<25} "
              f"{r['total_return_pct']:>+9.2f}% "
              f"{r['buy_hold_return_pct']:>+9.2f}% "
              f"{r['excess_return_pct']:>+9.2f}% "
              f"{r['sharpe_ratio']:>+7.2f} "
              f"{r['max_drawdown_pct']:>7.2f}% "
              f"{r['total_trades']:>7d} "
              f"{r['win_rate']:>6.1f}%")

    print("-" * 100)

    # Best/Worst summary
    best = sorted_results[0]
    worst = sorted_results[-1]
    beat_bh = sum(1 for r in results if r['excess_return_pct'] > 0)

    print(f"\nSummary:")
    print(f"  Best Strategy:  {best.get('strategy_title', best.get('strategy_name'))} "
          f"({best['total_return_pct']:+.2f}%, Sharpe: {best['sharpe_ratio']:+.2f})")
    print(f"  Worst Strategy: {worst.get('strategy_title', worst.get('strategy_name'))} "
          f"({worst['total_return_pct']:+.2f}%, Sharpe: {worst['sharpe_ratio']:+.2f})")
    print(f"  Beat Buy&Hold:  {beat_bh}/{len(results)} strategies")

    # Monte Carlo summary if available
    mc_results = [r for r in results if r.get('mc_validation') and r['mc_validation'].get('has_edge') is not None]
    if mc_results:
        has_edge_count = sum(1 for r in mc_results if r['mc_validation']['has_edge'])
        print(f"  Has Real Edge:  {has_edge_count}/{len(mc_results)} strategies (Monte Carlo validated)")

    print("=" * 100)


def generate_combined_equity_chart(
    results: list[dict],
    output_path: str | None = None,
    open_browser: bool = True,
) -> str | None:
    """
    Generate a combined equity curve chart showing all strategies vs buy-and-hold.

    Args:
        results: List of backtest results from different strategies
        output_path: Custom output path (auto-generated if None)
        open_browser: Open the chart in browser

    Returns:
        Path to generated HTML file
    """
    if not results:
        return None

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Warning: plotly not installed. Install with: pip install plotly")
        # Fall back to matplotlib
        return _generate_matplotlib_chart(results, output_path, open_browser)

    symbol = results[0].get('symbol', 'UNKNOWN')
    trading_days = results[0].get('trading_days', len(results[0]['equity_curve']))

    # Create figure with secondary y-axis for drawdown
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Equity Curves', 'Drawdown'),
        row_heights=[0.7, 0.3],
    )

    # Color palette for strategies
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#95190C']

    # Calculate buy-and-hold equity curve
    initial_capital = results[0]['initial_capital']
    first_equity = results[0]['equity_curve']
    bh_return = results[0]['buy_hold_return_pct'] / 100
    bh_equity = [initial_capital * (1 + bh_return * i / (len(first_equity) - 1))
                 for i in range(len(first_equity))]

    # Add buy-and-hold line
    fig.add_trace(
        go.Scatter(
            y=bh_equity,
            name=f"Buy & Hold ({results[0]['buy_hold_return_pct']:+.1f}%)",
            line=dict(color='gray', width=2, dash='dash'),
            opacity=0.7,
        ),
        row=1, col=1
    )

    # Add each strategy's equity curve
    for i, r in enumerate(results):
        strategy_name = r.get('strategy_title', r.get('strategy_name', f'Strategy {i+1}'))
        color = colors[i % len(colors)]

        # Equity curve
        fig.add_trace(
            go.Scatter(
                y=r['equity_curve'],
                name=f"{strategy_name} ({r['total_return_pct']:+.1f}%)",
                line=dict(color=color, width=2),
            ),
            row=1, col=1
        )

        # Calculate and add drawdown
        equity = np.array(r['equity_curve'])
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak * 100

        fig.add_trace(
            go.Scatter(
                y=drawdown,
                name=f"{strategy_name} DD",
                line=dict(color=color, width=1),
                showlegend=False,
            ),
            row=2, col=1
        )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Strategy Comparison: {symbol}<br><sup>All strategies vs Buy & Hold</sup>",
            x=0.5,
        ),
        height=700,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
        ),
    )

    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig.update_xaxes(title_text="Trading Days", row=2, col=1)

    # Generate filename if not provided
    if output_path is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"strategy_comparison_{symbol}_{timestamp}.html"

    # Save to HTML
    fig.write_html(output_path)
    print(f"\nCombined chart saved: {output_path}")

    if open_browser:
        import webbrowser
        report_url = f"file://{Path(output_path).absolute()}"
        print(f"Opening in browser...")
        webbrowser.open(report_url)

    return output_path


def _generate_matplotlib_chart(
    results: list[dict],
    output_path: str | None = None,
    open_browser: bool = True,
) -> str | None:
    """Fallback matplotlib chart if plotly is not available."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    except ImportError:
        print("Warning: matplotlib not installed. Cannot generate chart.")
        return None

    symbol = results[0].get('symbol', 'UNKNOWN')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']

    # Buy-and-hold line
    initial_capital = results[0]['initial_capital']
    first_equity = results[0]['equity_curve']
    bh_return = results[0]['buy_hold_return_pct'] / 100
    bh_equity = [initial_capital * (1 + bh_return * i / (len(first_equity) - 1))
                 for i in range(len(first_equity))]
    ax1.plot(bh_equity, 'k--', alpha=0.5, label=f"Buy & Hold ({results[0]['buy_hold_return_pct']:+.1f}%)")

    # Plot each strategy
    for i, r in enumerate(results):
        strategy_name = r.get('strategy_title', r.get('strategy_name', f'Strategy {i+1}'))
        color = colors[i % len(colors)]

        ax1.plot(r['equity_curve'], color=color, linewidth=1.5,
                 label=f"{strategy_name} ({r['total_return_pct']:+.1f}%)")

        # Drawdown
        equity = np.array(r['equity_curve'])
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak * 100
        ax2.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color=color)

    ax1.set_ylabel('Equity ($)')
    ax1.set_title(f'Strategy Comparison: {symbol}')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Trading Days')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Generate filename if not provided
    if output_path is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"strategy_comparison_{symbol}_{timestamp}.png"

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nCombined chart saved: {output_path}")

    if open_browser:
        import webbrowser
        report_url = f"file://{Path(output_path).absolute()}"
        print(f"Opening in browser...")
        webbrowser.open(report_url)

    return output_path


def generate_combined_quantstats_report(
    results: list[dict],
    benchmark: str = "SPY",
    output_path: str | None = None,
    open_browser: bool = True,
) -> str | None:
    """
    Generate a combined QuantStats-style HTML report comparing all strategies.

    This creates a custom HTML report with:
    - All equity curves on one chart
    - Performance metrics table for all strategies
    - Monthly returns heatmap for best strategy
    - Drawdown comparison
    """
    if not HAS_ANALYTICS:
        print("Warning: QuantStats not available for combined report")
        return generate_combined_equity_chart(results, output_path, open_browser)

    import pandas as pd

    symbol = results[0].get('symbol', 'UNKNOWN')

    # Build returns DataFrame with all strategies
    returns_dict = {}
    for r in results:
        strategy_name = r.get('strategy_title', r.get('strategy_name', 'Unknown'))
        equity = np.array(r['equity_curve'])
        returns = np.diff(equity) / equity[:-1]
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        returns_dict[strategy_name] = returns

    # Find the minimum length and truncate all to match
    min_len = min(len(r) for r in returns_dict.values())
    for key in returns_dict:
        returns_dict[key] = returns_dict[key][:min_len]

    # Create DataFrame
    index = pd.date_range(end=pd.Timestamp.now(), periods=min_len, freq='D')
    returns_df = pd.DataFrame(returns_dict, index=index)

    # Generate filename if not provided
    if output_path is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"strategy_comparison_{symbol}_{timestamp}.html"

    # Generate combined chart first
    chart_path = generate_combined_equity_chart(results, None, False)

    # Build custom HTML report
    html_content = _build_combined_html_report(results, returns_df, symbol, chart_path)

    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"\nCombined report saved: {output_path}")

    if open_browser:
        report_url = f"file://{Path(output_path).absolute()}"
        print(f"Opening in browser...")
        webbrowser.open(report_url)

    return output_path


def _build_combined_html_report(
    results: list[dict],
    returns_df,
    symbol: str,
    chart_path: str | None,
) -> str:
    """Build custom HTML report comparing all strategies."""
    import pandas as pd

    # Sort results by Sharpe ratio
    sorted_results = sorted(results, key=lambda r: r.get('sharpe_ratio', 0), reverse=True)
    best = sorted_results[0]

    # Build metrics table rows
    metrics_rows = ""
    for rank, r in enumerate(sorted_results, 1):
        strategy = r.get('strategy_title', r.get('strategy_name', 'Unknown'))
        mc_badge = ""
        if r.get('mc_validation') and r['mc_validation'].get('has_edge') is not None:
            if r['mc_validation']['has_edge']:
                mc_badge = '<span style="color: green; font-weight: bold;">HAS EDGE</span>'
            else:
                mc_badge = '<span style="color: red;">NO EDGE</span>'

        metrics_rows += f"""
        <tr>
            <td>{rank}</td>
            <td><strong>{strategy}</strong></td>
            <td style="color: {'green' if r['total_return_pct'] > 0 else 'red'}">{r['total_return_pct']:+.2f}%</td>
            <td style="color: {'green' if r['excess_return_pct'] > 0 else 'red'}">{r['excess_return_pct']:+.2f}%</td>
            <td style="color: {'green' if r['sharpe_ratio'] > 0.5 else 'orange' if r['sharpe_ratio'] > 0 else 'red'}">{r['sharpe_ratio']:+.2f}</td>
            <td style="color: {'red' if abs(r['max_drawdown_pct']) > 20 else 'orange' if abs(r['max_drawdown_pct']) > 10 else 'green'}">{r['max_drawdown_pct']:.2f}%</td>
            <td>{r['total_trades']}</td>
            <td>{r['win_rate']:.1f}%</td>
            <td>{mc_badge}</td>
        </tr>
        """

    # Embed chart if it exists
    chart_embed = ""
    if chart_path and Path(chart_path).exists():
        chart_path_obj = Path(chart_path)
        if chart_path_obj.suffix.lower() == '.html':
            # HTML chart (from plotly) - embed as iframe
            with open(chart_path, 'r') as f:
                chart_content = f.read()
            chart_embed = f'<div class="chart-container">{chart_content}</div>'
        elif chart_path_obj.suffix.lower() in ('.png', '.jpg', '.jpeg'):
            # Image chart (from matplotlib) - embed as base64
            import base64
            with open(chart_path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
            img_type = 'png' if chart_path_obj.suffix.lower() == '.png' else 'jpeg'
            chart_embed = f'''
            <div class="chart-container">
                <img src="data:image/{img_type};base64,{img_data}"
                     style="max-width: 100%; height: auto;"
                     alt="Strategy Comparison Chart">
            </div>'''

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Strategy Comparison: {symbol}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #2E86AB;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .summary-cards {{
            display: flex;
            gap: 20px;
            margin: 20px 0;
            flex-wrap: wrap;
        }}
        .card {{
            background: linear-gradient(135deg, #2E86AB, #1a5276);
            color: white;
            padding: 20px;
            border-radius: 10px;
            min-width: 200px;
            flex: 1;
        }}
        .card.winner {{
            background: linear-gradient(135deg, #27ae60, #1e8449);
        }}
        .card h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            opacity: 0.9;
        }}
        .card .value {{
            font-size: 28px;
            font-weight: bold;
        }}
        .card .sub {{
            font-size: 12px;
            opacity: 0.8;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #333;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        tr:first-child {{
            background: #e8f5e9;
        }}
        .chart-container {{
            margin: 30px 0;
            border: 1px solid #ddd;
            border-radius: 5px;
            overflow: hidden;
        }}
        .chart-container iframe {{
            width: 100%;
            height: 700px;
            border: none;
        }}
        .footer {{
            text-align: center;
            color: #888;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Strategy Comparison: {symbol}</h1>
        <p>Comparing {len(results)} strategies on {results[0].get('trading_days', 'N/A')} trading days
           ({results[0].get('start_date', 'N/A')} to {results[0].get('end_date', 'N/A')})</p>

        <div class="summary-cards">
            <div class="card winner">
                <h3>BEST STRATEGY</h3>
                <div class="value">{best.get('strategy_title', best.get('strategy_name'))}</div>
                <div class="sub">Return: {best['total_return_pct']:+.2f}% | Sharpe: {best['sharpe_ratio']:+.2f}</div>
            </div>
            <div class="card">
                <h3>BUY & HOLD</h3>
                <div class="value">{results[0]['buy_hold_return_pct']:+.2f}%</div>
                <div class="sub">{symbol} benchmark return</div>
            </div>
            <div class="card">
                <h3>BEAT BENCHMARK</h3>
                <div class="value">{sum(1 for r in results if r['excess_return_pct'] > 0)}/{len(results)}</div>
                <div class="sub">strategies outperformed</div>
            </div>
            <div class="card">
                <h3>BEST SHARPE</h3>
                <div class="value">{best['sharpe_ratio']:+.2f}</div>
                <div class="sub">risk-adjusted return</div>
            </div>
        </div>

        <h2>Performance Ranking</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Strategy</th>
                    <th>Return</th>
                    <th>vs B&H</th>
                    <th>Sharpe</th>
                    <th>Max DD</th>
                    <th>Trades</th>
                    <th>Win Rate</th>
                    <th>Monte Carlo</th>
                </tr>
            </thead>
            <tbody>
                {metrics_rows}
            </tbody>
        </table>

        {chart_embed}

        <div class="footer">
            Generated by FluxHero Backtest Runner | {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
    """

    return html


def main():
    parser = argparse.ArgumentParser(
        description="Easy-to-use backtest runner with Yahoo Finance data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_backtest.py                           # SPY, last 1 year
  python scripts/run_backtest.py --symbol AAPL             # AAPL, last 1 year
  python scripts/run_backtest.py --symbols SPY,QQQ,IWM     # Compare multiple
  python scripts/run_backtest.py --quick                   # Quick test (6 months)
  python scripts/run_backtest.py --start 2022-01-01        # Custom start date
        """
    )

    parser.add_argument(
        '--symbol', '-s',
        default='SPY',
        help='Symbol to backtest (default: SPY)'
    )
    parser.add_argument(
        '--symbols',
        help='Comma-separated symbols for comparison (e.g., SPY,QQQ,IWM)'
    )
    parser.add_argument(
        '--start',
        help='Start date YYYY-MM-DD (default: 1 year ago)'
    )
    parser.add_argument(
        '--end',
        help='End date YYYY-MM-DD (default: today)'
    )
    parser.add_argument(
        '--capital', '-c',
        type=float,
        default=100000.0,
        help='Initial capital (default: $100,000)'
    )
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Quick mode: last 6 months only'
    )
    parser.add_argument(
        '--full', '-f',
        action='store_true',
        help='Full analysis: 2 years with detailed output'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal output'
    )
    parser.add_argument(
        '--diagnostics', '-d',
        action='store_true',
        help='Show signal and regime diagnostics'
    )
    parser.add_argument(
        '--report', '-r',
        action='store_true',
        help='Generate HTML tearsheet report (requires quantstats)'
    )
    parser.add_argument(
        '--report-output',
        type=str,
        help='Custom output path for HTML report'
    )
    parser.add_argument(
        '--benchmark',
        type=str,
        default='SPY',
        help='Benchmark symbol for report comparison (default: SPY)'
    )
    parser.add_argument(
        '--monte-carlo',
        action='store_true',
        help='Include Monte Carlo analysis in report'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['dual_mode', 'trend_only', 'mean_reversion', 'amm', 'golden', '1', '2', '3', '4', '5'],
        help='Strategy to use (name or number 1-5). Interactive menu if omitted.'
    )
    parser.add_argument(
        '--no-menu',
        action='store_true',
        help='Skip interactive strategy menu (use default: dual_mode)'
    )
    parser.add_argument(
        '--all-strategies',
        action='store_true',
        help='Run and compare ALL strategies on the same symbol'
    )

    args = parser.parse_args()

    # Determine date range
    end_date = args.end or datetime.now().strftime('%Y-%m-%d')

    if args.start:
        start_date = args.start
    elif args.quick:
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    elif args.full:
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    else:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    # Determine symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    else:
        symbols = [args.symbol.upper()]

    print("=" * 70)
    print("FluxHero Backtest Runner")
    print("=" * 70)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Capital: ${args.capital:,.0f}")

    # Handle --all-strategies mode
    if args.all_strategies:
        print(f"\nMode: COMPARE ALL STRATEGIES")
        print("Running all 5 strategies on each symbol...")

        all_results = []
        for symbol in symbols:
            print(f"\n{'#'*70}")
            print(f"# Comparing strategies for: {symbol}")
            print(f"{'#'*70}")

            strategy_results = run_all_strategies(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                initial_capital=args.capital,
                verbose=not args.quiet,
                run_monte_carlo=args.monte_carlo,
            )

            if strategy_results:
                all_results.extend(strategy_results)

                # Print comparison table for this symbol
                print_strategy_comparison_table(strategy_results)

                # Generate combined chart/report if requested
                if args.report:
                    generate_combined_quantstats_report(
                        results=strategy_results,
                        benchmark=args.benchmark if symbol != args.benchmark else 'QQQ',
                        open_browser=True,
                    )

        # Summary across all symbols
        if len(symbols) > 1 and all_results:
            print(f"\n{'='*70}")
            print("OVERALL SUMMARY ACROSS ALL SYMBOLS")
            print(f"{'='*70}")
            best_overall = max(all_results, key=lambda r: r.get('sharpe_ratio', 0))
            print(f"Best Overall: {best_overall.get('strategy_title')} on {best_overall['symbol']}")
            print(f"  Return: {best_overall['total_return_pct']:+.2f}%  |  Sharpe: {best_overall['sharpe_ratio']:+.2f}")

        return 0 if all_results else 1

    # Single strategy mode - determine which strategy to use
    if args.strategy:
        # Handle both number and name inputs
        if args.strategy in STRATEGIES:
            strategy_choice = args.strategy
            strategy_name = STRATEGIES[args.strategy]['name']
        else:
            strategy_name = args.strategy
            strategy_choice = None
            for key, info in STRATEGIES.items():
                if info['name'] == strategy_name:
                    strategy_choice = key
                    break
    elif args.no_menu:
        # Use default without menu
        strategy_choice = "1"
        strategy_name = "dual_mode"
    else:
        # Interactive menu
        strategy_choice = get_strategy_choice()
        strategy_name = STRATEGIES[strategy_choice]['name']

    # Display selected strategy
    strategy_info = STRATEGIES.get(strategy_choice, {"title": strategy_name, "description": ""})
    print(f"\nStrategy: [{strategy_choice}] {strategy_info['title']}")
    print(f"          {strategy_info['description']}")

    # Run backtests
    results = []
    for symbol in symbols:
        result = run_backtest(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_capital=args.capital,
            verbose=not args.quiet,
            show_diagnostics=args.diagnostics,
            run_monte_carlo=args.monte_carlo,
            mc_simulations=5000,
            strategy_name=strategy_name,
        )
        if result:
            results.append(result)
            if len(symbols) == 1:
                print_results(result, show_details=not args.quiet)

    # Print comparison for multiple symbols
    if len(results) > 1:
        print_comparison_table(results)

    # Generate HTML reports if requested
    if args.report and results:
        print("\n" + "=" * 70)
        print("GENERATING HTML REPORTS")
        print("=" * 70)

        for result in results:
            # Determine output path
            if args.report_output and len(results) == 1:
                output_path = args.report_output
            else:
                output_path = None  # Auto-generate

            report_path = generate_html_report(
                result=result,
                benchmark=args.benchmark if result['symbol'] != args.benchmark else 'QQQ',
                output_path=output_path,
                show_monte_carlo=args.monte_carlo,
            )

            if report_path:
                print(f"\n  Open report: file://{Path(report_path).absolute()}")

    # Summary
    if results:
        total_pnl = sum(r['final_equity'] - r['initial_capital'] for r in results)
        print(f"\nTotal P&L across all symbols: ${total_pnl:+,.2f}")

    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
