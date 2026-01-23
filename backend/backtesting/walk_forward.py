"""
Walk-Forward Testing Module for FluxHero Trading System.

This module implements walk-forward analysis, a technique to validate trading
strategies by dividing historical data into consecutive train/test windows
and evaluating out-of-sample performance.

Features:
- WalkForwardWindow dataclass with train/test indices and dates
- Window generation with configurable train/test periods
- Rolling window execution with optional parameter re-optimization
- Edge case handling (insufficient data, uneven final window, date gaps)

Reference:
- FLUXHERO_REQUIREMENTS.md R9.4 Walk-Forward Testing
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

from backend.backtesting.engine import BacktestConfig, BacktestEngine, BacktestState
from backend.backtesting.metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """Represents a single walk-forward train/test window.

    Attributes:
        window_id: Sequential identifier for this window (0-indexed)
        train_start_idx: Start bar index for training period (inclusive)
        train_end_idx: End bar index for training period (exclusive)
        test_start_idx: Start bar index for testing period (inclusive)
        test_end_idx: End bar index for testing period (exclusive)
        train_start_date: Start date of training period (if available)
        train_end_date: End date of training period (if available)
        test_start_date: Start date of testing period (if available)
        test_end_date: End date of testing period (if available)
    """

    window_id: int
    train_start_idx: int
    train_end_idx: int
    test_start_idx: int
    test_end_idx: int
    train_start_date: datetime | None = None
    train_end_date: datetime | None = None
    test_start_date: datetime | None = None
    test_end_date: datetime | None = None

    @property
    def train_size(self) -> int:
        """Number of bars in the training period."""
        return self.train_end_idx - self.train_start_idx

    @property
    def test_size(self) -> int:
        """Number of bars in the testing period."""
        return self.test_end_idx - self.test_start_idx

    def __repr__(self) -> str:
        if self.train_start_date and self.test_end_date:
            train_end = self.train_end_date.date() if self.train_end_date else "N/A"
            test_start = self.test_start_date.date() if self.test_start_date else "N/A"
            return (
                f"WalkForwardWindow(id={self.window_id}, "
                f"train={self.train_start_date.date()}-{train_end}, "
                f"test={test_start}-{self.test_end_date.date()})"
            )
        return (
            f"WalkForwardWindow(id={self.window_id}, "
            f"train_idx=[{self.train_start_idx}:{self.train_end_idx}], "
            f"test_idx=[{self.test_start_idx}:{self.test_end_idx}])"
        )


class InsufficientDataError(Exception):
    """Raised when there is insufficient data for walk-forward testing."""

    pass


def generate_walk_forward_windows(
    n_bars: int,
    train_bars: int = 63,
    test_bars: int = 21,
    timestamps: NDArray | None = None,
    min_test_bars: int | None = None,
) -> list[WalkForwardWindow]:
    """
    Generate walk-forward train/test windows.

    Creates consecutive non-overlapping windows where each window has a training
    period followed by a test period. The test period of one window is immediately
    followed by the training period of the next window.

    Default configuration:
    - 3-month train (63 trading days)
    - 1-month test (21 trading days)
    - Total window size: 84 bars

    Parameters
    ----------
    n_bars : int
        Total number of bars in the dataset
    train_bars : int
        Number of bars for training period (default: 63 = ~3 months)
    test_bars : int
        Number of bars for testing period (default: 21 = ~1 month)
    timestamps : Optional[NDArray]
        Array of bar timestamps (Unix epoch). If provided, dates are included
        in the window objects.
    min_test_bars : Optional[int]
        Minimum bars required for the final test period. If the final window
        would have fewer test bars, it is excluded. Default is test_bars // 2.

    Returns
    -------
    list[WalkForwardWindow]
        List of walk-forward windows with train/test indices and dates

    Raises
    ------
    InsufficientDataError
        If n_bars is less than train_bars + min_test_bars
    ValueError
        If train_bars or test_bars is not positive

    Examples
    --------
    >>> # Generate windows for 252 bars (1 year of daily data)
    >>> windows = generate_walk_forward_windows(252, train_bars=63, test_bars=21)
    >>> len(windows)
    3
    >>> windows[0].train_start_idx, windows[0].train_end_idx
    (0, 63)
    >>> windows[0].test_start_idx, windows[0].test_end_idx
    (63, 84)
    """
    # Validate inputs
    if train_bars <= 0:
        raise ValueError(f"train_bars must be positive, got {train_bars}")
    if test_bars <= 0:
        raise ValueError(f"test_bars must be positive, got {test_bars}")

    # Set default minimum test bars
    if min_test_bars is None:
        min_test_bars = test_bars // 2
        # Ensure at least 1 bar for very small test periods
        min_test_bars = max(1, min_test_bars)

    # Check minimum data requirement
    min_required = train_bars + min_test_bars
    if n_bars < min_required:
        raise InsufficientDataError(
            f"Insufficient data for walk-forward testing: have {n_bars} bars, "
            f"need at least {min_required} (train={train_bars} + min_test={min_test_bars})"
        )

    windows: list[WalkForwardWindow] = []
    window_id = 0
    start_idx = 0

    while start_idx + train_bars < n_bars:
        # Calculate train period
        train_start = start_idx
        train_end = start_idx + train_bars

        # Calculate test period
        test_start = train_end
        test_end = min(test_start + test_bars, n_bars)

        # Check if test period has enough bars
        actual_test_bars = test_end - test_start
        if actual_test_bars < min_test_bars:
            # Skip this window - not enough test data
            break

        # Extract dates if timestamps provided
        train_start_date = None
        train_end_date = None
        test_start_date = None
        test_end_date = None

        if timestamps is not None and len(timestamps) >= n_bars:
            train_start_date = datetime.fromtimestamp(float(timestamps[train_start]))
            # train_end is exclusive, so use train_end - 1 for the date
            train_end_date = datetime.fromtimestamp(float(timestamps[train_end - 1]))
            test_start_date = datetime.fromtimestamp(float(timestamps[test_start]))
            test_end_date = datetime.fromtimestamp(float(timestamps[test_end - 1]))

        window = WalkForwardWindow(
            window_id=window_id,
            train_start_idx=train_start,
            train_end_idx=train_end,
            test_start_idx=test_start,
            test_end_idx=test_end,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            test_start_date=test_start_date,
            test_end_date=test_end_date,
        )
        windows.append(window)

        # Move to next window (test period becomes available for next window's data)
        start_idx = test_end
        window_id += 1

    return windows


def validate_no_data_leakage(windows: list[WalkForwardWindow]) -> bool:
    """
    Validate that there is no data leakage between windows.

    Checks that:
    1. Test periods do not overlap with training periods
    2. Windows are sequential and non-overlapping
    3. Training never uses future data

    Parameters
    ----------
    windows : list[WalkForwardWindow]
        List of walk-forward windows to validate

    Returns
    -------
    bool
        True if no data leakage detected

    Raises
    ------
    ValueError
        If data leakage is detected
    """
    if not windows:
        return True

    for i, window in enumerate(windows):
        # Check train/test ordering within window
        if window.train_end_idx > window.test_start_idx:
            raise ValueError(
                f"Window {window.window_id}: Training period overlaps with test period "
                f"(train_end={window.train_end_idx} > test_start={window.test_start_idx})"
            )

        if window.train_start_idx >= window.train_end_idx:
            raise ValueError(
                f"Window {window.window_id}: Invalid training period "
                f"(start={window.train_start_idx} >= end={window.train_end_idx})"
            )

        if window.test_start_idx >= window.test_end_idx:
            raise ValueError(
                f"Window {window.window_id}: Invalid test period "
                f"(start={window.test_start_idx} >= end={window.test_end_idx})"
            )

        # Check sequential ordering between windows
        if i > 0:
            prev_window = windows[i - 1]
            if window.train_start_idx < prev_window.test_end_idx:
                raise ValueError(
                    f"Window {window.window_id}: Training starts before previous test ends "
                    f"(train_start={window.train_start_idx} < "
                    f"prev_test_end={prev_window.test_end_idx})"
                )

    return True


def check_date_gaps(
    timestamps: NDArray,
    max_gap_days: int = 5,
) -> list[tuple[int, int, float]]:
    """
    Check for large gaps in the timestamp data.

    Identifies periods where trading data is missing, which could affect
    walk-forward analysis quality.

    Parameters
    ----------
    timestamps : NDArray
        Array of bar timestamps (Unix epoch)
    max_gap_days : int
        Maximum acceptable gap in trading days (default: 5)

    Returns
    -------
    list[tuple[int, int, float]]
        List of (bar_index, next_bar_index, gap_days) for each gap exceeding max_gap_days
    """
    if len(timestamps) < 2:
        return []

    gaps: list[tuple[int, int, float]] = []
    seconds_per_day = 86400.0
    max_gap_seconds = max_gap_days * seconds_per_day

    for i in range(len(timestamps) - 1):
        gap = timestamps[i + 1] - timestamps[i]
        if gap > max_gap_seconds:
            gap_days = gap / seconds_per_day
            gaps.append((i, i + 1, gap_days))

    return gaps


@dataclass
class WalkForwardWindowResult:
    """Result from a single walk-forward test window.

    Attributes:
        window: The walk-forward window configuration
        metrics: Performance metrics from the test period
        initial_equity: Starting equity for this window
        final_equity: Ending equity for this window
        equity_curve: Equity values during test period
        is_profitable: True if final_equity > initial_equity
        strategy_params: Strategy parameters used (may differ if re-optimized)
    """

    window: WalkForwardWindow
    metrics: dict[str, Any]
    initial_equity: float
    final_equity: float
    equity_curve: list[float]
    is_profitable: bool
    strategy_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Complete result from walk-forward backtest.

    Attributes:
        window_results: Results for each individual window
        total_windows: Total number of windows tested
        profitable_windows: Number of profitable windows
        config: Backtest configuration used
    """

    window_results: list[WalkForwardWindowResult]
    total_windows: int
    profitable_windows: int
    config: BacktestConfig

    @property
    def pass_rate(self) -> float:
        """Percentage of profitable windows (0.0 to 1.0)."""
        if self.total_windows == 0:
            return 0.0
        return self.profitable_windows / self.total_windows


StrategyFactory = Callable[[NDArray, float, dict[str, Any]], Callable]
"""Type alias for strategy factory function.

A strategy factory takes (bars, initial_capital, params) and returns
a strategy function compatible with BacktestEngine.run().
"""

OptimizerFunc = Callable[[NDArray, BacktestConfig], dict[str, Any]]
"""Type alias for parameter optimizer function.

An optimizer takes (train_bars, config) and returns optimized strategy parameters.
"""

# Default pass rate threshold: strategy must have >60% profitable windows
DEFAULT_PASS_RATE_THRESHOLD = 0.6


def calculate_pass_rate(profitable_windows: int, total_windows: int) -> float:
    """
    Calculate the pass rate (percentage of profitable test windows).

    Parameters
    ----------
    profitable_windows : int
        Number of windows where final_equity > initial_equity
    total_windows : int
        Total number of test windows

    Returns
    -------
    float
        Pass rate as a decimal (0.0 to 1.0)

    Examples
    --------
    >>> calculate_pass_rate(3, 5)
    0.6
    >>> calculate_pass_rate(0, 5)
    0.0
    >>> calculate_pass_rate(0, 0)
    0.0

    Reference: FLUXHERO_REQUIREMENTS.md R9.4.4
    """
    if total_windows == 0:
        return 0.0
    return profitable_windows / total_windows


def passes_walk_forward_test(
    pass_rate: float,
    threshold: float = DEFAULT_PASS_RATE_THRESHOLD,
) -> bool:
    """
    Determine if a strategy passes the walk-forward test.

    A strategy passes if the pass rate is STRICTLY GREATER than the threshold.
    By default, this means >60% of test windows must be profitable.

    Parameters
    ----------
    pass_rate : float
        Percentage of profitable windows (0.0 to 1.0)
    threshold : float
        Minimum pass rate threshold (default: 0.6 = 60%)

    Returns
    -------
    bool
        True if pass_rate > threshold (strategy passes)

    Examples
    --------
    >>> passes_walk_forward_test(0.65)  # 65% > 60%
    True
    >>> passes_walk_forward_test(0.60)  # 60% is NOT > 60%
    False
    >>> passes_walk_forward_test(0.55)  # 55% < 60%
    False

    Notes
    -----
    Per R9.4.4, strategy passes if >60% of test periods are profitable.
    This is a strict greater-than comparison, so exactly 60% does NOT pass.

    Reference: FLUXHERO_REQUIREMENTS.md R9.4.4
    """
    return pass_rate > threshold


def run_walk_forward_backtest(
    bars: NDArray,
    strategy_factory: StrategyFactory,
    config: BacktestConfig | None = None,
    train_bars: int = 63,
    test_bars: int = 21,
    timestamps: NDArray | None = None,
    volumes: NDArray | None = None,
    symbol: str = "SPY",
    initial_params: dict[str, Any] | None = None,
    optimizer: OptimizerFunc | None = None,
    min_test_bars: int | None = None,
) -> WalkForwardResult:
    """
    Run walk-forward backtest across multiple train/test windows.

    Executes backtest on each test window sequentially. Optionally re-optimizes
    strategy parameters on each train window before testing.

    Parameters
    ----------
    bars : NDArray
        OHLCV data, shape (N, 5) where columns are [open, high, low, close, volume]
    strategy_factory : StrategyFactory
        Factory function that creates a strategy function.
        Signature: (bars, initial_capital, params) -> strategy_func
        The returned strategy_func should be compatible with BacktestEngine.run()
    config : BacktestConfig, optional
        Backtest configuration. If None, uses default config.
    train_bars : int
        Number of bars for training period (default: 63 = ~3 months)
    test_bars : int
        Number of bars for testing period (default: 21 = ~1 month)
    timestamps : NDArray, optional
        Timestamps for each bar (for trade logging and date tracking)
    volumes : NDArray, optional
        Volume data if not included in bars array
    symbol : str
        Trading symbol (default: 'SPY')
    initial_params : dict, optional
        Initial strategy parameters. Used for all windows if no optimizer provided.
    optimizer : OptimizerFunc, optional
        Function to optimize strategy parameters on train data.
        If provided, called on each train window to get optimized parameters.
        Signature: (train_bars, config) -> optimized_params
    min_test_bars : int, optional
        Minimum bars required for final test period. Default is test_bars // 2.

    Returns
    -------
    WalkForwardResult
        Complete walk-forward results with per-window metrics

    Raises
    ------
    InsufficientDataError
        If not enough bars for at least one walk-forward window
    ValueError
        If train_bars or test_bars is not positive

    Examples
    --------
    >>> # Basic walk-forward without optimization
    >>> def my_strategy_factory(bars, capital, params):
    ...     strategy = MyStrategy(bars, capital, **params)
    ...     return strategy.get_orders
    ...
    >>> result = run_walk_forward_backtest(
    ...     bars=price_data,
    ...     strategy_factory=my_strategy_factory,
    ...     initial_params={"risk_pct": 0.01}
    ... )
    >>> print(f"Pass rate: {result.pass_rate:.1%}")

    >>> # Walk-forward with parameter optimization
    >>> def optimize_params(train_bars, config):
    ...     # Run optimization on train data
    ...     best_params = optimize_on_data(train_bars)
    ...     return best_params
    ...
    >>> result = run_walk_forward_backtest(
    ...     bars=price_data,
    ...     strategy_factory=my_strategy_factory,
    ...     optimizer=optimize_params
    ... )
    """
    # Use default config if not provided
    if config is None:
        config = BacktestConfig()

    # Use empty params if not provided
    if initial_params is None:
        initial_params = {}

    n_bars = len(bars)

    # Generate walk-forward windows
    windows = generate_walk_forward_windows(
        n_bars=n_bars,
        train_bars=train_bars,
        test_bars=test_bars,
        timestamps=timestamps,
        min_test_bars=min_test_bars,
    )

    logger.info(
        f"Starting walk-forward backtest: {len(windows)} windows, "
        f"train={train_bars} bars, test={test_bars} bars"
    )

    # Validate no data leakage
    validate_no_data_leakage(windows)

    window_results: list[WalkForwardWindowResult] = []
    profitable_count = 0

    # Track cumulative capital across windows
    current_capital = config.initial_capital

    for window in windows:
        logger.debug(f"Processing window {window.window_id}: {window}")

        # Extract train data for this window
        train_data = bars[window.train_start_idx : window.train_end_idx]

        # Determine strategy parameters for this window
        if optimizer is not None:
            # Re-optimize on train data
            logger.debug(f"Optimizing parameters on train window {window.window_id}")
            params = optimizer(train_data, config)
        else:
            # Use initial/previous parameters
            params = initial_params.copy()

        # Extract test data for this window
        test_data = bars[window.test_start_idx : window.test_end_idx]
        test_timestamps = None
        if timestamps is not None:
            test_timestamps = timestamps[window.test_start_idx : window.test_end_idx]
        test_volumes = None
        if volumes is not None:
            test_volumes = volumes[window.test_start_idx : window.test_end_idx]

        # Create strategy using factory with test data
        # Strategy needs full context up to test period for indicator warmup
        # Include train data + test data for proper indicator calculation
        full_window_data = bars[window.train_start_idx : window.test_end_idx]
        strategy_func = strategy_factory(full_window_data, current_capital, params)

        # Create a wrapper that adjusts indices for the test-only portion
        test_offset = window.train_end_idx - window.train_start_idx

        def create_test_wrapper(orig_func: Callable, offset: int) -> Callable:
            """Create wrapper that maps test indices to full window indices."""

            def wrapper(test_bars: NDArray, idx: int, position: Any) -> list:
                # Map test index to full window index
                full_idx = idx + offset
                return orig_func(full_window_data, full_idx, position)

            return wrapper

        test_strategy_func = create_test_wrapper(strategy_func, test_offset)

        # Run backtest on test period
        window_config = BacktestConfig(
            initial_capital=current_capital,
            commission_per_share=config.commission_per_share,
            slippage_pct=config.slippage_pct,
            impact_threshold=config.impact_threshold,
            impact_penalty_pct=config.impact_penalty_pct,
            risk_free_rate=config.risk_free_rate,
            max_position_size=config.max_position_size,
            enable_sanity_checks=config.enable_sanity_checks,
        )

        engine = BacktestEngine(window_config)
        state: BacktestState = engine.run(
            bars=test_data,
            strategy_func=test_strategy_func,
            symbol=symbol,
            timestamps=test_timestamps,
            volumes=test_volumes,
        )

        # Calculate metrics for this window
        trades_pnl = np.array([t.pnl for t in state.trades]) if state.trades else np.array([])
        trades_holding = (
            np.array([t.holding_bars for t in state.trades]) if state.trades else np.array([])
        )

        metrics = PerformanceMetrics.calculate_all_metrics(
            equity_curve=np.array(state.equity_curve),
            trades_pnl=trades_pnl,
            trades_holding_periods=trades_holding,
            initial_capital=current_capital,
            risk_free_rate=config.risk_free_rate,
            enable_sanity_checks=config.enable_sanity_checks,
        )

        # Determine profitability
        final_equity = state.equity_curve[-1] if state.equity_curve else current_capital
        is_profitable = final_equity > current_capital

        if is_profitable:
            profitable_count += 1

        # Create window result
        window_result = WalkForwardWindowResult(
            window=window,
            metrics=metrics,
            initial_equity=current_capital,
            final_equity=final_equity,
            equity_curve=state.equity_curve.copy(),
            is_profitable=is_profitable,
            strategy_params=params.copy(),
        )
        window_results.append(window_result)

        logger.info(
            f"Window {window.window_id}: "
            f"${current_capital:,.2f} -> ${final_equity:,.2f} "
            f"({'profitable' if is_profitable else 'loss'}), "
            f"{len(state.trades)} trades, Sharpe={metrics['sharpe_ratio']:.2f}"
        )

        # Update capital for next window (carry forward)
        current_capital = final_equity

    # Create final result
    result = WalkForwardResult(
        window_results=window_results,
        total_windows=len(windows),
        profitable_windows=profitable_count,
        config=config,
    )

    logger.info(
        f"Walk-forward complete: {result.profitable_windows}/{result.total_windows} "
        f"profitable ({result.pass_rate:.1%})"
    )

    return result


@dataclass
class AggregateWalkForwardMetrics:
    """Aggregated metrics from walk-forward testing.

    Attributes:
        combined_equity_curve: Concatenated equity values from all test periods
        aggregate_sharpe: Sharpe ratio calculated from combined equity curve
        aggregate_max_drawdown_pct: Maximum drawdown across combined equity curve
        aggregate_win_rate: Win rate across all trades from all windows
        total_trades: Total number of trades across all windows
        total_profitable_windows: Number of windows with positive returns
        total_windows: Total number of windows tested
        pass_rate: Percentage of profitable windows (0.0 to 1.0)
        passes_walk_forward_test: True if pass_rate > 0.6 (60% threshold)
        initial_capital: Starting capital
        final_capital: Ending capital after all windows
        total_return_pct: Total return percentage
        per_window_returns: List of return percentages for each window
    """

    combined_equity_curve: list[float]
    aggregate_sharpe: float
    aggregate_max_drawdown_pct: float
    aggregate_win_rate: float
    total_trades: int
    total_profitable_windows: int
    total_windows: int
    pass_rate: float
    passes_walk_forward_test: bool
    initial_capital: float
    final_capital: float
    total_return_pct: float
    per_window_returns: list[float]


def aggregate_walk_forward_results(
    result: WalkForwardResult,
    pass_threshold: float = 0.6,
) -> AggregateWalkForwardMetrics:
    """
    Aggregate results from walk-forward testing across all windows.

    Combines equity curves from all test periods and calculates aggregate
    performance metrics to evaluate overall strategy robustness.

    Parameters
    ----------
    result : WalkForwardResult
        Complete walk-forward results containing all window results
    pass_threshold : float
        Minimum pass rate required for strategy to pass (default: 0.6 = 60%)

    Returns
    -------
    AggregateWalkForwardMetrics
        Aggregated metrics including:
        - Combined equity curve from all test periods
        - Aggregate Sharpe ratio
        - Aggregate max drawdown
        - Aggregate win rate across all trades
        - Pass rate and pass/fail status

    Examples
    --------
    >>> result = run_walk_forward_backtest(bars, strategy_factory)
    >>> aggregate = aggregate_walk_forward_results(result)
    >>> print(f"Pass rate: {aggregate.pass_rate:.1%}")
    >>> print(f"Passes test: {aggregate.passes_walk_forward_test}")
    >>> print(f"Aggregate Sharpe: {aggregate.aggregate_sharpe:.2f}")

    Notes
    -----
    The combined equity curve is constructed by normalizing each window's
    equity to start at the ending value of the previous window, creating
    a continuous capital growth curve.

    Reference: FLUXHERO_REQUIREMENTS.md R9.4.3
    """
    from backend.backtesting.metrics import (
        calculate_max_drawdown,
        calculate_returns,
        calculate_sharpe_ratio,
        calculate_win_rate,
    )

    if not result.window_results:
        return AggregateWalkForwardMetrics(
            combined_equity_curve=[result.config.initial_capital],
            aggregate_sharpe=0.0,
            aggregate_max_drawdown_pct=0.0,
            aggregate_win_rate=0.0,
            total_trades=0,
            total_profitable_windows=0,
            total_windows=0,
            pass_rate=0.0,
            passes_walk_forward_test=False,
            initial_capital=result.config.initial_capital,
            final_capital=result.config.initial_capital,
            total_return_pct=0.0,
            per_window_returns=[],
        )

    # Combine equity curves from all windows
    # Each window's equity curve is scaled to continue from the previous window's end
    combined_equity: list[float] = []
    all_trades_pnl: list[float] = []
    per_window_returns: list[float] = []

    for i, window_result in enumerate(result.window_results):
        # Calculate per-window return
        if window_result.initial_equity > 0:
            window_return = (
                (window_result.final_equity - window_result.initial_equity)
                / window_result.initial_equity
            ) * 100.0
        else:
            window_return = 0.0
        per_window_returns.append(window_return)

        # Add equity curve points
        # For first window, include all points
        # For subsequent windows, skip the first point (which equals previous final)
        equity_curve = window_result.equity_curve
        if i == 0:
            combined_equity.extend(equity_curve)
        else:
            # Skip first point to avoid duplicating the junction point
            if len(equity_curve) > 1:
                combined_equity.extend(equity_curve[1:])
            elif len(equity_curve) == 1:
                # Only one point, still add it if different from last
                if combined_equity and equity_curve[0] != combined_equity[-1]:
                    combined_equity.extend(equity_curve)

        # Collect trade P&Ls from window metrics
        if "trades_pnl" in window_result.metrics:
            all_trades_pnl.extend(window_result.metrics["trades_pnl"])
        # Alternative: count from win/loss counts if trades_pnl not available
        elif "winning_trades" in window_result.metrics and "losing_trades" in window_result.metrics:
            # Reconstruct P&Ls for win rate calculation (only signs matter)
            wins = window_result.metrics.get("winning_trades", 0)
            losses = window_result.metrics.get("losing_trades", 0)
            all_trades_pnl.extend([1.0] * wins)  # Positive for wins
            all_trades_pnl.extend([-1.0] * losses)  # Negative for losses

    # Convert to numpy arrays for metric calculations
    combined_equity_arr = np.array(combined_equity, dtype=np.float64)
    trades_pnl_arr = np.array(all_trades_pnl, dtype=np.float64)

    # Calculate aggregate metrics
    # Sharpe ratio from combined equity curve
    if len(combined_equity_arr) > 1:
        returns = calculate_returns(combined_equity_arr)
        aggregate_sharpe = calculate_sharpe_ratio(
            returns,
            risk_free_rate=result.config.risk_free_rate,
            periods_per_year=252,
        )
    else:
        aggregate_sharpe = 0.0

    # Max drawdown from combined equity curve
    if len(combined_equity_arr) > 0:
        max_dd_pct, _, _ = calculate_max_drawdown(combined_equity_arr)
        aggregate_max_drawdown = max_dd_pct
    else:
        aggregate_max_drawdown = 0.0

    # Win rate from all trades
    if len(trades_pnl_arr) > 0:
        aggregate_win_rate = calculate_win_rate(trades_pnl_arr)
    else:
        aggregate_win_rate = 0.0

    # Calculate totals
    total_trades = len(all_trades_pnl)
    total_profitable_windows = result.profitable_windows
    total_windows = result.total_windows

    # Pass rate
    pass_rate = result.pass_rate

    # Initial and final capital
    initial_capital = result.config.initial_capital
    final_capital = combined_equity[-1] if combined_equity else initial_capital

    # Total return
    if initial_capital > 0:
        total_return_pct = ((final_capital - initial_capital) / initial_capital) * 100.0
    else:
        total_return_pct = 0.0

    # Determine if strategy passes walk-forward test
    # Per R9.4.4: strategy passes if >60% of test periods are profitable
    # Using strict greater-than comparison
    passes_test = passes_walk_forward_test(pass_rate, threshold=pass_threshold)

    logger.info(
        f"Aggregate walk-forward metrics: "
        f"Sharpe={aggregate_sharpe:.2f}, "
        f"MaxDD={aggregate_max_drawdown:.2f}%, "
        f"WinRate={aggregate_win_rate:.2%}, "
        f"PassRate={pass_rate:.1%}, "
        f"Passes={'YES' if passes_test else 'NO'}"
    )

    return AggregateWalkForwardMetrics(
        combined_equity_curve=combined_equity,
        aggregate_sharpe=aggregate_sharpe,
        aggregate_max_drawdown_pct=aggregate_max_drawdown,
        aggregate_win_rate=aggregate_win_rate,
        total_trades=total_trades,
        total_profitable_windows=total_profitable_windows,
        total_windows=total_windows,
        pass_rate=pass_rate,
        passes_walk_forward_test=passes_test,
        initial_capital=initial_capital,
        final_capital=final_capital,
        total_return_pct=total_return_pct,
        per_window_returns=per_window_returns,
    )
