"""
Runner for AMM strategy backtests with QuantStats integration.

Provides functions for single backtests, walk-forward testing, and report generation.
"""

import numpy as np
from typing import Any

from backend.backtesting.engine import BacktestConfig, BacktestEngine
from backend.backtesting.walk_forward import (
    run_walk_forward_backtest,
    aggregate_walk_forward_results,
)
from backend.analytics.quantstats_wrapper import create_adapter_from_backtest
from backend.amm_system.strategy import (
    AMMConfig,
    AMMStrategy,
    create_amm_strategy_factory,
)
from backend.amm_system.optimizer import (
    GridSearchConfig,
    create_amm_optimizer,
)


def run_amm_backtest(
    bars: np.ndarray,
    symbol: str = "SPY",
    config: AMMConfig | None = None,
    backtest_config: BacktestConfig | None = None,
    timestamps: np.ndarray | None = None,
    volumes: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Run a single AMM strategy backtest.

    Parameters
    ----------
    bars : np.ndarray
        OHLCV data with shape (n_bars, 5)
    symbol : str
        Trading symbol
    config : AMMConfig, optional
        Strategy configuration
    backtest_config : BacktestConfig, optional
        Backtest engine configuration
    timestamps : np.ndarray, optional
        Bar timestamps
    volumes : np.ndarray, optional
        Volume data (if not in bars)

    Returns
    -------
    dict
        Results containing:
        - 'state': BacktestState object
        - 'summary': Performance summary dict
        - 'all_metrics': Full QuantStats metrics
        - 'config': Strategy configuration used
    """
    strategy_config = config or AMMConfig()
    bt_config = backtest_config or BacktestConfig()

    # Create strategy
    strategy = AMMStrategy(
        bars=bars,
        initial_capital=bt_config.initial_capital,
        config=strategy_config,
        symbol=symbol,
    )

    # Run backtest
    engine = BacktestEngine(bt_config)
    state = engine.run(
        bars=bars,
        strategy_func=strategy.get_orders,
        symbol=symbol,
        timestamps=timestamps,
        volumes=volumes,
    )

    # Get performance summary
    summary = engine.get_performance_summary(state)

    # Get QuantStats metrics if we have trades
    all_metrics = {}
    if len(state.trades) > 0 and len(state.equity_curve) > 1:
        trades_pnl = [t.pnl for t in state.trades]
        adapter = create_adapter_from_backtest(
            equity_curve=state.equity_curve,
            trades_pnl=trades_pnl,
            risk_free_rate=bt_config.risk_free_rate,
        )
        all_metrics = adapter.get_all_metrics()

    return {
        "state": state,
        "summary": summary,
        "all_metrics": all_metrics,
        "config": strategy_config.to_dict(),
    }


def run_amm_walk_forward(
    bars: np.ndarray,
    train_bars: int = 252,
    test_bars: int = 63,
    symbol: str = "SPY",
    config: AMMConfig | None = None,
    backtest_config: BacktestConfig | None = None,
    optimizer_config: GridSearchConfig | None = None,
    timestamps: np.ndarray | None = None,
    volumes: np.ndarray | None = None,
    optimize: bool = True,
    pass_threshold: float = 0.6,
) -> dict[str, Any]:
    """
    Run walk-forward backtest with optional parameter optimization.

    Parameters
    ----------
    bars : np.ndarray
        OHLCV data with shape (n_bars, 5)
    train_bars : int
        Number of bars for training window (default 252 = ~1 year)
    test_bars : int
        Number of bars for test window (default 63 = ~1 quarter)
    symbol : str
        Trading symbol
    config : AMMConfig, optional
        Base strategy configuration
    backtest_config : BacktestConfig, optional
        Backtest engine configuration
    optimizer_config : GridSearchConfig, optional
        Grid search configuration (only used if optimize=True)
    timestamps : np.ndarray, optional
        Bar timestamps
    volumes : np.ndarray, optional
        Volume data
    optimize : bool
        Whether to run parameter optimization on each training window
    pass_threshold : float
        Minimum pass rate to consider strategy valid

    Returns
    -------
    dict
        Results containing:
        - 'pass_rate': Fraction of profitable test windows
        - 'passes': Whether strategy passes walk-forward test
        - 'sharpe': Aggregate Sharpe ratio
        - 'max_drawdown': Aggregate max drawdown
        - 'total_return': Total return across all windows
        - 'win_rate': Aggregate win rate
        - 'total_trades': Total number of trades
        - 'window_results': Per-window results
        - 'all_metrics': Full QuantStats metrics on combined equity curve
        - 'config': Strategy configuration used
    """
    strategy_config = config or AMMConfig()
    bt_config = backtest_config or BacktestConfig()

    # Create strategy factory
    strategy_factory = create_amm_strategy_factory(strategy_config)

    # Create optimizer if requested
    optimizer = None
    if optimize:
        optimizer = create_amm_optimizer(
            config=optimizer_config,
            base_strategy_config=strategy_config,
        )

    # Run walk-forward backtest
    wf_result = run_walk_forward_backtest(
        bars=bars,
        strategy_factory=strategy_factory,
        config=bt_config,
        train_bars=train_bars,
        test_bars=test_bars,
        timestamps=timestamps,
        volumes=volumes,
        symbol=symbol,
        initial_params=strategy_config.to_dict(),
        optimizer=optimizer,
    )

    # Aggregate results
    aggregate = aggregate_walk_forward_results(wf_result, pass_threshold)

    # Get QuantStats metrics on combined equity curve
    # Note: Walk-forward results don't store individual trades, only aggregated metrics
    # We can still compute metrics from the combined equity curve
    all_metrics = {}
    if len(aggregate.combined_equity_curve) > 1:
        equity_arr = np.array(aggregate.combined_equity_curve)
        # Calculate returns from equity curve for QuantStats
        returns = np.diff(equity_arr) / equity_arr[:-1]
        # Create adapter with empty trades (metrics will be calculated from equity/returns)
        adapter = create_adapter_from_backtest(
            equity_curve=aggregate.combined_equity_curve,
            trades_pnl=[],  # Individual trades not available in walk-forward results
            risk_free_rate=bt_config.risk_free_rate,
        )
        all_metrics = adapter.get_all_metrics()

    return {
        "pass_rate": aggregate.pass_rate,
        "passes": aggregate.passes_walk_forward_test,
        "sharpe": aggregate.aggregate_sharpe,
        "max_drawdown": aggregate.aggregate_max_drawdown_pct,
        "total_return": aggregate.total_return_pct,
        "win_rate": aggregate.aggregate_win_rate,
        "total_trades": aggregate.total_trades,
        "total_windows": aggregate.total_windows,
        "profitable_windows": aggregate.total_profitable_windows,
        "initial_capital": aggregate.initial_capital,
        "final_capital": aggregate.final_capital,
        "window_results": wf_result.window_results,
        "all_metrics": all_metrics,
        "config": strategy_config.to_dict(),
    }


def generate_amm_report(
    result: dict[str, Any],
    output_path: str | None = None,
) -> str:
    """
    Generate a performance report from backtest results.

    Parameters
    ----------
    result : dict
        Results from run_amm_backtest or run_amm_walk_forward
    output_path : str, optional
        Path to save report. If None, only returns report string.

    Returns
    -------
    str
        Formatted performance report
    """
    lines = []
    lines.append("=" * 60)
    lines.append("AMM STRATEGY PERFORMANCE REPORT")
    lines.append("=" * 60)
    lines.append("")

    # Determine if this is walk-forward or single backtest
    is_walk_forward = "pass_rate" in result

    if is_walk_forward:
        lines.append("WALK-FORWARD RESULTS")
        lines.append("-" * 40)
        lines.append(f"Pass Rate:         {result['pass_rate']:.1%}")
        lines.append(f"Passes Test:       {'YES' if result['passes'] else 'NO'}")
        lines.append(f"Total Windows:     {result['total_windows']}")
        lines.append(f"Profitable Windows:{result['profitable_windows']}")
        lines.append("")
        lines.append("AGGREGATE METRICS")
        lines.append("-" * 40)
        lines.append(f"Sharpe Ratio:      {result['sharpe']:.2f}")
        lines.append(f"Max Drawdown:      {result['max_drawdown']:.1%}")
        lines.append(f"Total Return:      {result['total_return']:.1%}")
        lines.append(f"Win Rate:          {result['win_rate']:.1%}")
        lines.append(f"Total Trades:      {result['total_trades']}")
        lines.append(f"Initial Capital:   ${result['initial_capital']:,.2f}")
        lines.append(f"Final Capital:     ${result['final_capital']:,.2f}")
    else:
        summary = result.get("summary", {})
        lines.append("SINGLE BACKTEST RESULTS")
        lines.append("-" * 40)
        lines.append(f"Total Return:      {summary.get('total_return_pct', 0):.1%}")
        lines.append(f"Sharpe Ratio:      {summary.get('sharpe_ratio', 0):.2f}")
        lines.append(f"Max Drawdown:      {summary.get('max_drawdown_pct', 0):.1%}")
        lines.append(f"Win Rate:          {summary.get('win_rate', 0):.1%}")
        lines.append(f"Total Trades:      {summary.get('total_trades', 0)}")
        lines.append(f"Initial Capital:   ${summary.get('initial_capital', 0):,.2f}")
        lines.append(f"Final Capital:     ${summary.get('final_capital', 0):,.2f}")

    # Add QuantStats metrics if available
    metrics = result.get("all_metrics", {})
    if metrics:
        lines.append("")
        lines.append("QUANTSTATS METRICS")
        lines.append("-" * 40)

        key_metrics = [
            ("sortino", "Sortino Ratio"),
            ("calmar", "Calmar Ratio"),
            ("avg_win", "Avg Win"),
            ("avg_loss", "Avg Loss"),
            ("profit_factor", "Profit Factor"),
            ("payoff_ratio", "Payoff Ratio"),
            ("ulcer_index", "Ulcer Index"),
            ("var", "Value at Risk"),
            ("cvar", "CVaR"),
        ]

        for key, label in key_metrics:
            if key in metrics:
                value = metrics[key]
                if isinstance(value, float):
                    if "ratio" in key.lower() or key in ["profit_factor", "payoff_ratio"]:
                        lines.append(f"{label:18} {value:.2f}")
                    elif key in ["var", "cvar", "ulcer_index"]:
                        lines.append(f"{label:18} {value:.2%}")
                    else:
                        lines.append(f"{label:18} {value:.4f}")

    # Add strategy configuration
    config = result.get("config", {})
    if config:
        lines.append("")
        lines.append("STRATEGY CONFIGURATION")
        lines.append("-" * 40)
        lines.append(f"SMA Period:        {config.get('sma_period', 'N/A')}")
        lines.append(f"RSI Period:        {config.get('rsi_period', 'N/A')}")
        lines.append(f"Momentum Period:   {config.get('mom_period', 'N/A')}")
        lines.append(f"Bollinger Period:  {config.get('boll_period', 'N/A')}")
        lines.append(
            f"Weights (SMA/RSI/MOM/BOLL): "
            f"{config.get('w_sma', 0):.2f}/{config.get('w_rsi', 0):.2f}/"
            f"{config.get('w_mom', 0):.2f}/{config.get('w_boll', 0):.2f}"
        )
        lines.append(f"Z-Score Lookback:  {config.get('zscore_lookback', 'N/A')}")
        lines.append(f"EMA Span:          {config.get('ema_span', 'N/A')}")
        lines.append(f"Entry Threshold:   {config.get('entry_threshold', 'N/A')}")
        lines.append(f"Risk Per Trade:    {config.get('risk_per_trade', 0):.1%}")
        lines.append(f"ATR Stop Multiple: {config.get('atr_stop_mult', 'N/A')}")
        lines.append(f"Regime Filter:     {'ON' if config.get('use_regime_filter', True) else 'OFF'}")
        if config.get('use_regime_filter', True):
            lines.append(f"ER Thresholds:     trend>{config.get('er_trend_threshold', 0.3):.2f}, range<{config.get('er_range_threshold', 0.2):.2f}")

    lines.append("")
    lines.append("=" * 60)

    report = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)

    return report
