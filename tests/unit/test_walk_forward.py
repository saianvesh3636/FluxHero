"""
Unit tests for walk-forward testing module.

Tests cover:
- WalkForwardWindow dataclass
- Window generation with various data lengths
- No data leakage validation
- Rolling window execution (run_walk_forward_backtest)
- Edge cases (insufficient data, uneven final window, date gaps)

Reference: FLUXHERO_REQUIREMENTS.md R9.4.1, R9.4.2
"""

from datetime import datetime
from typing import Any

import numpy as np
import pytest

from backend.backtesting.engine import BacktestConfig, Order, Position
from backend.backtesting.walk_forward import (
    InsufficientDataError,
    WalkForwardResult,
    WalkForwardWindow,
    WalkForwardWindowResult,
    check_date_gaps,
    generate_walk_forward_windows,
    run_walk_forward_backtest,
    validate_no_data_leakage,
)


class TestWalkForwardWindow:
    """Test WalkForwardWindow dataclass."""

    def test_window_creation(self):
        """Test basic window creation."""
        window = WalkForwardWindow(
            window_id=0,
            train_start_idx=0,
            train_end_idx=63,
            test_start_idx=63,
            test_end_idx=84,
        )

        assert window.window_id == 0
        assert window.train_start_idx == 0
        assert window.train_end_idx == 63
        assert window.test_start_idx == 63
        assert window.test_end_idx == 84
        assert window.train_start_date is None
        assert window.test_end_date is None

    def test_window_train_size(self):
        """Test train_size property."""
        window = WalkForwardWindow(
            window_id=0,
            train_start_idx=0,
            train_end_idx=63,
            test_start_idx=63,
            test_end_idx=84,
        )
        assert window.train_size == 63

    def test_window_test_size(self):
        """Test test_size property."""
        window = WalkForwardWindow(
            window_id=0,
            train_start_idx=0,
            train_end_idx=63,
            test_start_idx=63,
            test_end_idx=84,
        )
        assert window.test_size == 21

    def test_window_with_dates(self):
        """Test window creation with dates."""
        train_start = datetime(2023, 1, 1)
        train_end = datetime(2023, 3, 31)
        test_start = datetime(2023, 4, 1)
        test_end = datetime(2023, 4, 30)

        window = WalkForwardWindow(
            window_id=0,
            train_start_idx=0,
            train_end_idx=63,
            test_start_idx=63,
            test_end_idx=84,
            train_start_date=train_start,
            train_end_date=train_end,
            test_start_date=test_start,
            test_end_date=test_end,
        )

        assert window.train_start_date == train_start
        assert window.test_end_date == test_end

    def test_window_repr_without_dates(self):
        """Test string representation without dates."""
        window = WalkForwardWindow(
            window_id=0,
            train_start_idx=0,
            train_end_idx=63,
            test_start_idx=63,
            test_end_idx=84,
        )
        repr_str = repr(window)
        assert "id=0" in repr_str
        assert "train_idx=[0:63]" in repr_str
        assert "test_idx=[63:84]" in repr_str

    def test_window_repr_with_dates(self):
        """Test string representation with dates."""
        window = WalkForwardWindow(
            window_id=0,
            train_start_idx=0,
            train_end_idx=63,
            test_start_idx=63,
            test_end_idx=84,
            train_start_date=datetime(2023, 1, 1),
            train_end_date=datetime(2023, 3, 31),
            test_start_date=datetime(2023, 4, 1),
            test_end_date=datetime(2023, 4, 30),
        )
        repr_str = repr(window)
        assert "id=0" in repr_str
        assert "2023-01-01" in repr_str
        assert "2023-04-30" in repr_str


class TestGenerateWalkForwardWindows:
    """Test window generation function."""

    def test_12_month_synthetic_data(self):
        """Test window generation with 12-month (252 bars) synthetic data."""
        # 252 trading days = 1 year
        # With 63-bar train + 21-bar test = 84-bar window
        # 252 / 84 = 3 full windows
        windows = generate_walk_forward_windows(n_bars=252, train_bars=63, test_bars=21)

        assert len(windows) == 3

        # Verify first window
        assert windows[0].window_id == 0
        assert windows[0].train_start_idx == 0
        assert windows[0].train_end_idx == 63
        assert windows[0].test_start_idx == 63
        assert windows[0].test_end_idx == 84

        # Verify second window starts after first test ends
        assert windows[1].train_start_idx == 84
        assert windows[1].train_end_idx == 147
        assert windows[1].test_start_idx == 147
        assert windows[1].test_end_idx == 168

        # Verify third window
        assert windows[2].train_start_idx == 168
        assert windows[2].train_end_idx == 231
        assert windows[2].test_start_idx == 231
        assert windows[2].test_end_idx == 252

    def test_4_month_minimal_case(self):
        """Test minimal case with 4-month data (84 bars = 1 window)."""
        # 84 bars = exactly 1 window (63 train + 21 test)
        windows = generate_walk_forward_windows(n_bars=84, train_bars=63, test_bars=21)

        assert len(windows) == 1
        assert windows[0].train_size == 63
        assert windows[0].test_size == 21

    def test_1_plus_year_multiple_windows(self):
        """Test with more than 1 year of data (multiple windows)."""
        # 504 bars = 2 years = 6 windows
        windows = generate_walk_forward_windows(n_bars=504, train_bars=63, test_bars=21)

        assert len(windows) == 6

        # Verify all windows are sequential
        for i in range(1, len(windows)):
            assert windows[i].train_start_idx == windows[i - 1].test_end_idx

    def test_insufficient_data(self):
        """Test error when insufficient data for even one window."""
        # Need at least 63 (train) + 10 (min_test default = 21//2) = 73 bars
        with pytest.raises(InsufficientDataError) as exc_info:
            generate_walk_forward_windows(n_bars=70, train_bars=63, test_bars=21)

        assert "Insufficient data" in str(exc_info.value)
        assert "70 bars" in str(exc_info.value)

    def test_uneven_final_window(self):
        """Test handling of uneven final window."""
        # 100 bars: 1 full window (84), 16 remaining
        # 16 < 63 train, so only 1 window
        windows = generate_walk_forward_windows(n_bars=100, train_bars=63, test_bars=21)

        assert len(windows) == 1
        # First window gets full test period
        assert windows[0].test_end_idx == 84

    def test_partial_final_test_window(self):
        """Test partial test period in final window."""
        # 170 bars: 2 full windows (168), then 2 bars remaining
        # 2 bars < min_test (10), so only 2 windows
        windows = generate_walk_forward_windows(n_bars=170, train_bars=63, test_bars=21)

        assert len(windows) == 2

    def test_min_test_bars_custom(self):
        """Test custom minimum test bars."""
        # 250 bars: after 2 full windows (168), 82 bars remaining
        # Next train period: 168-231 (63 bars)
        # Next test period: 231-250 (19 bars)

        # With min_test_bars=20, 19 < 20, so 2 windows
        windows = generate_walk_forward_windows(
            n_bars=250, train_bars=63, test_bars=21, min_test_bars=20
        )
        assert len(windows) == 2

        # With min_test_bars=10, 19 >= 10, so 3 windows with truncated test
        windows = generate_walk_forward_windows(
            n_bars=250, train_bars=63, test_bars=21, min_test_bars=10
        )
        assert len(windows) == 3
        assert windows[2].test_size == 19  # Truncated test period (250 - 231)

    def test_with_timestamps(self):
        """Test window generation with timestamps."""
        n_bars = 84
        # Generate timestamps: 1 day apart starting from 2023-01-01
        base_ts = datetime(2023, 1, 1).timestamp()
        timestamps = np.array([base_ts + i * 86400 for i in range(n_bars)])

        windows = generate_walk_forward_windows(
            n_bars=n_bars, train_bars=63, test_bars=21, timestamps=timestamps
        )

        assert len(windows) == 1
        assert windows[0].train_start_date is not None
        assert windows[0].test_end_date is not None
        assert windows[0].train_start_date.date() == datetime(2023, 1, 1).date()

    def test_invalid_train_bars(self):
        """Test error with non-positive train_bars."""
        with pytest.raises(ValueError) as exc_info:
            generate_walk_forward_windows(n_bars=100, train_bars=0, test_bars=21)
        assert "train_bars must be positive" in str(exc_info.value)

    def test_invalid_test_bars(self):
        """Test error with non-positive test_bars."""
        with pytest.raises(ValueError) as exc_info:
            generate_walk_forward_windows(n_bars=100, train_bars=63, test_bars=0)
        assert "test_bars must be positive" in str(exc_info.value)


class TestValidateNoDataLeakage:
    """Test data leakage validation."""

    def test_valid_windows_no_leakage(self):
        """Test that properly generated windows have no leakage."""
        windows = generate_walk_forward_windows(n_bars=252, train_bars=63, test_bars=21)

        # Should not raise
        result = validate_no_data_leakage(windows)
        assert result is True

    def test_empty_windows_list(self):
        """Test empty windows list passes validation."""
        result = validate_no_data_leakage([])
        assert result is True

    def test_overlapping_train_test(self):
        """Test detection of train/test overlap within window."""
        # Manually create invalid window with train overlapping test
        bad_window = WalkForwardWindow(
            window_id=0,
            train_start_idx=0,
            train_end_idx=70,  # Overlaps with test_start
            test_start_idx=60,  # Should be >= train_end
            test_end_idx=84,
        )

        with pytest.raises(ValueError) as exc_info:
            validate_no_data_leakage([bad_window])
        assert "overlaps with test period" in str(exc_info.value)

    def test_invalid_train_period(self):
        """Test detection of invalid train period (start >= end)."""
        bad_window = WalkForwardWindow(
            window_id=0,
            train_start_idx=63,  # Invalid: start >= end
            train_end_idx=63,
            test_start_idx=63,
            test_end_idx=84,
        )

        with pytest.raises(ValueError) as exc_info:
            validate_no_data_leakage([bad_window])
        assert "Invalid training period" in str(exc_info.value)

    def test_invalid_test_period(self):
        """Test detection of invalid test period (start >= end)."""
        bad_window = WalkForwardWindow(
            window_id=0,
            train_start_idx=0,
            train_end_idx=63,
            test_start_idx=84,  # Invalid: start >= end
            test_end_idx=84,
        )

        with pytest.raises(ValueError) as exc_info:
            validate_no_data_leakage([bad_window])
        assert "Invalid test period" in str(exc_info.value)

    def test_overlapping_windows(self):
        """Test detection of overlapping consecutive windows."""
        window1 = WalkForwardWindow(
            window_id=0,
            train_start_idx=0,
            train_end_idx=63,
            test_start_idx=63,
            test_end_idx=84,
        )
        # Window 2 starts training before window 1's test ends
        window2 = WalkForwardWindow(
            window_id=1,
            train_start_idx=80,  # Overlaps with window1 test (ends at 84)
            train_end_idx=143,
            test_start_idx=143,
            test_end_idx=164,
        )

        with pytest.raises(ValueError) as exc_info:
            validate_no_data_leakage([window1, window2])
        assert "Training starts before previous test ends" in str(exc_info.value)


class TestCheckDateGaps:
    """Test date gap detection."""

    def test_no_gaps(self):
        """Test data with no significant gaps."""
        # Daily data, 1 day apart
        timestamps = np.array([i * 86400 for i in range(100)])
        gaps = check_date_gaps(timestamps, max_gap_days=5)
        assert len(gaps) == 0

    def test_detect_large_gap(self):
        """Test detection of gap larger than threshold."""
        # Data with a 10-day gap
        timestamps = np.array(
            [
                0,
                86400,
                172800,  # Days 0, 1, 2
                172800 + 10 * 86400,  # Day 12 (10-day gap)
                172800 + 11 * 86400,  # Day 13
            ]
        )
        gaps = check_date_gaps(timestamps, max_gap_days=5)

        assert len(gaps) == 1
        assert gaps[0][0] == 2  # Gap starts at index 2
        assert gaps[0][1] == 3  # Gap ends at index 3
        assert gaps[0][2] == pytest.approx(10.0, rel=0.01)  # 10-day gap

    def test_multiple_gaps(self):
        """Test detection of multiple gaps."""
        # Data with two gaps
        day = 86400
        timestamps = np.array(
            [
                0,
                day,
                2 * day,  # Days 0-2
                10 * day,  # Day 10 (8-day gap)
                11 * day,
                12 * day,  # Days 11-12
                25 * day,  # Day 25 (13-day gap)
                26 * day,
            ]
        )
        gaps = check_date_gaps(timestamps, max_gap_days=5)

        assert len(gaps) == 2

    def test_empty_timestamps(self):
        """Test with empty timestamps array."""
        gaps = check_date_gaps(np.array([]), max_gap_days=5)
        assert len(gaps) == 0

    def test_single_timestamp(self):
        """Test with single timestamp."""
        gaps = check_date_gaps(np.array([86400]), max_gap_days=5)
        assert len(gaps) == 0

    def test_custom_max_gap(self):
        """Test custom max_gap_days threshold."""
        day = 86400
        timestamps = np.array([0, day, 2 * day, 6 * day, 7 * day])  # 4-day gap at index 2

        # With max_gap_days=3, detect the 4-day gap
        gaps = check_date_gaps(timestamps, max_gap_days=3)
        assert len(gaps) == 1

        # With max_gap_days=5, 4-day gap is acceptable
        gaps = check_date_gaps(timestamps, max_gap_days=5)
        assert len(gaps) == 0


# =============================================================================
# Helper functions for walk-forward execution tests
# =============================================================================


def generate_synthetic_bars(
    n_bars: int,
    base_price: float = 100.0,
    volatility: float = 0.02,
    trend: float = 0.0001,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic OHLCV bar data for testing.

    Creates realistic price data with configurable trend and volatility.

    Parameters
    ----------
    n_bars : int
        Number of bars to generate
    base_price : float
        Starting price level
    volatility : float
        Daily volatility (fraction, e.g. 0.02 = 2%)
    trend : float
        Daily trend (fraction, e.g. 0.0001 = 0.01% per day)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        OHLCV array of shape (n_bars, 5)
    """
    rng = np.random.default_rng(seed)

    # Generate close prices with trend and random walk
    returns = rng.normal(trend, volatility, n_bars)
    prices = base_price * np.exp(np.cumsum(returns))

    # Generate OHLC from close prices
    opens = np.roll(prices, 1)
    opens[0] = base_price

    # High/Low based on volatility
    intraday_vol = volatility * 0.5
    highs = np.maximum(opens, prices) * (1 + rng.uniform(0, intraday_vol, n_bars))
    lows = np.minimum(opens, prices) * (1 - rng.uniform(0, intraday_vol, n_bars))

    # Volume
    base_volume = 1_000_000
    volumes = rng.uniform(0.5, 1.5, n_bars) * base_volume

    bars = np.column_stack([opens, highs, lows, prices, volumes])
    return bars


def simple_strategy_factory(
    bars: np.ndarray,
    initial_capital: float,
    params: dict[str, Any],
) -> callable:
    """Create a simple buy-and-hold strategy for testing.

    This strategy buys on the first bar and holds throughout the test.
    """
    bought = False

    def get_orders(
        all_bars: np.ndarray,
        current_idx: int,
        position: Position | None,
    ) -> list[Order]:
        nonlocal bought
        from backend.backtesting.engine import OrderSide, OrderType

        if not bought and position is None:
            bought = True
            # Buy 100 shares on first bar
            return [
                Order(
                    bar_index=current_idx,
                    symbol="TEST",
                    side=OrderSide.BUY,
                    shares=100,
                    order_type=OrderType.MARKET,
                )
            ]
        return []

    return get_orders


def alternating_strategy_factory(
    bars: np.ndarray,
    initial_capital: float,
    params: dict[str, Any],
) -> callable:
    """Create a strategy that alternates between profitable and losing windows.

    Uses params["make_profit"] to determine behavior.
    """
    traded = False

    def get_orders(
        all_bars: np.ndarray,
        current_idx: int,
        position: Position | None,
    ) -> list[Order]:
        nonlocal traded
        from backend.backtesting.engine import OrderSide, OrderType

        make_profit = params.get("make_profit", True)

        if not traded and position is None:
            traded = True
            # Buy shares - profit/loss depends on market direction
            shares = 100 if make_profit else 1  # Small position for losses
            return [
                Order(
                    bar_index=current_idx,
                    symbol="TEST",
                    side=OrderSide.BUY,
                    shares=shares,
                    order_type=OrderType.MARKET,
                )
            ]
        return []

    return get_orders


class TestWalkForwardWindowResult:
    """Test WalkForwardWindowResult dataclass."""

    def test_window_result_creation(self):
        """Test basic window result creation."""
        window = WalkForwardWindow(
            window_id=0,
            train_start_idx=0,
            train_end_idx=63,
            test_start_idx=63,
            test_end_idx=84,
        )
        result = WalkForwardWindowResult(
            window=window,
            metrics={"sharpe_ratio": 1.5, "total_return_pct": 5.0},
            initial_equity=100000.0,
            final_equity=105000.0,
            equity_curve=[100000.0, 102000.0, 105000.0],
            is_profitable=True,
            strategy_params={"risk_pct": 0.01},
        )

        assert result.window.window_id == 0
        assert result.is_profitable is True
        assert result.initial_equity == 100000.0
        assert result.final_equity == 105000.0
        assert result.metrics["sharpe_ratio"] == 1.5


class TestWalkForwardResult:
    """Test WalkForwardResult dataclass."""

    def test_pass_rate_all_profitable(self):
        """Test pass rate when all windows are profitable."""
        config = BacktestConfig()
        result = WalkForwardResult(
            window_results=[],
            total_windows=3,
            profitable_windows=3,
            config=config,
        )
        assert result.pass_rate == 1.0

    def test_pass_rate_none_profitable(self):
        """Test pass rate when no windows are profitable."""
        config = BacktestConfig()
        result = WalkForwardResult(
            window_results=[],
            total_windows=3,
            profitable_windows=0,
            config=config,
        )
        assert result.pass_rate == 0.0

    def test_pass_rate_partial(self):
        """Test pass rate with partial profitability."""
        config = BacktestConfig()
        result = WalkForwardResult(
            window_results=[],
            total_windows=4,
            profitable_windows=3,
            config=config,
        )
        assert result.pass_rate == 0.75

    def test_pass_rate_zero_windows(self):
        """Test pass rate with zero windows."""
        config = BacktestConfig()
        result = WalkForwardResult(
            window_results=[],
            total_windows=0,
            profitable_windows=0,
            config=config,
        )
        assert result.pass_rate == 0.0


class TestRunWalkForwardBacktest:
    """Test run_walk_forward_backtest function."""

    def test_basic_walk_forward(self):
        """Test basic walk-forward execution without optimization."""
        # Generate 252 bars (1 year) - should create 3 windows
        bars = generate_synthetic_bars(252, trend=0.0002)  # Slight uptrend

        result = run_walk_forward_backtest(
            bars=bars,
            strategy_factory=simple_strategy_factory,
            train_bars=63,
            test_bars=21,
            symbol="TEST",
        )

        # Should have 3 windows
        assert result.total_windows == 3
        assert len(result.window_results) == 3

        # Each window should have metrics
        for wr in result.window_results:
            assert "sharpe_ratio" in wr.metrics
            assert "total_return_pct" in wr.metrics
            assert wr.initial_equity > 0
            assert wr.final_equity > 0
            assert len(wr.equity_curve) > 0

    def test_walk_forward_with_custom_config(self):
        """Test walk-forward with custom backtest config."""
        bars = generate_synthetic_bars(168, trend=0.0001)  # 2 windows

        config = BacktestConfig(
            initial_capital=50000.0,
            commission_per_share=0.01,
            slippage_pct=0.0002,
        )

        result = run_walk_forward_backtest(
            bars=bars,
            strategy_factory=simple_strategy_factory,
            config=config,
            train_bars=63,
            test_bars=21,
        )

        assert result.total_windows == 2
        # First window should start with configured capital
        assert result.window_results[0].initial_equity == 50000.0

    def test_walk_forward_with_initial_params(self):
        """Test walk-forward with initial strategy parameters."""
        bars = generate_synthetic_bars(84)  # 1 window

        initial_params = {"risk_pct": 0.02, "mode": "aggressive"}

        result = run_walk_forward_backtest(
            bars=bars,
            strategy_factory=simple_strategy_factory,
            train_bars=63,
            test_bars=21,
            initial_params=initial_params,
        )

        assert result.total_windows == 1
        # Parameters should be passed through
        assert result.window_results[0].strategy_params == initial_params

    def test_walk_forward_with_optimizer(self):
        """Test walk-forward with parameter optimization."""
        bars = generate_synthetic_bars(168)  # 2 windows

        optimization_calls = []

        def mock_optimizer(
            train_bars: np.ndarray,
            config: BacktestConfig,
        ) -> dict[str, Any]:
            # Track optimizer calls
            optimization_calls.append(len(train_bars))
            return {"optimized": True, "train_size": len(train_bars)}

        result = run_walk_forward_backtest(
            bars=bars,
            strategy_factory=simple_strategy_factory,
            train_bars=63,
            test_bars=21,
            optimizer=mock_optimizer,
        )

        # Optimizer should be called for each window
        assert len(optimization_calls) == 2
        assert all(size == 63 for size in optimization_calls)

        # Each window should have optimized params
        for wr in result.window_results:
            assert wr.strategy_params.get("optimized") is True

    def test_walk_forward_insufficient_data(self):
        """Test error when insufficient data."""
        bars = generate_synthetic_bars(70)  # Less than 63 + 10 (min test)

        with pytest.raises(InsufficientDataError):
            run_walk_forward_backtest(
                bars=bars,
                strategy_factory=simple_strategy_factory,
                train_bars=63,
                test_bars=21,
            )

    def test_walk_forward_capital_carry_forward(self):
        """Test that capital carries forward between windows."""
        # Use uptrending data to ensure profits
        bars = generate_synthetic_bars(168, trend=0.003)  # Strong uptrend

        result = run_walk_forward_backtest(
            bars=bars,
            strategy_factory=simple_strategy_factory,
            train_bars=63,
            test_bars=21,
        )

        # Second window should start with first window's final equity
        if len(result.window_results) >= 2:
            first_final = result.window_results[0].final_equity
            second_initial = result.window_results[1].initial_equity
            assert second_initial == pytest.approx(first_final, rel=0.001)

    def test_walk_forward_profitability_tracking(self):
        """Test that profitability is tracked correctly."""
        # Generate trending data where buy-and-hold should profit
        bars = generate_synthetic_bars(168, trend=0.005)  # Very strong uptrend

        result = run_walk_forward_backtest(
            bars=bars,
            strategy_factory=simple_strategy_factory,
            train_bars=63,
            test_bars=21,
        )

        # With strong uptrend, windows should be profitable
        for wr in result.window_results:
            if wr.final_equity > wr.initial_equity:
                assert wr.is_profitable == True  # noqa: E712 - numpy bool comparison
            else:
                assert wr.is_profitable == False  # noqa: E712 - numpy bool comparison

        # profitable_windows should match count
        profitable_count = sum(1 for wr in result.window_results if wr.is_profitable)
        assert result.profitable_windows == profitable_count

    def test_walk_forward_with_timestamps(self):
        """Test walk-forward with timestamps provided."""
        n_bars = 168
        bars = generate_synthetic_bars(n_bars)

        # Generate timestamps
        base_ts = datetime(2023, 1, 1).timestamp()
        timestamps = np.array([base_ts + i * 86400 for i in range(n_bars)])

        result = run_walk_forward_backtest(
            bars=bars,
            strategy_factory=simple_strategy_factory,
            train_bars=63,
            test_bars=21,
            timestamps=timestamps,
        )

        # Windows should have dates
        for wr in result.window_results:
            assert wr.window.train_start_date is not None
            assert wr.window.test_end_date is not None

    def test_walk_forward_single_window(self):
        """Test walk-forward with exactly one window."""
        bars = generate_synthetic_bars(84)  # Exactly 63 + 21

        result = run_walk_forward_backtest(
            bars=bars,
            strategy_factory=simple_strategy_factory,
            train_bars=63,
            test_bars=21,
        )

        assert result.total_windows == 1
        assert result.window_results[0].window.window_id == 0
        assert result.window_results[0].window.train_size == 63
        assert result.window_results[0].window.test_size == 21

    def test_walk_forward_equity_curve_per_window(self):
        """Test that equity curves are captured per window."""
        bars = generate_synthetic_bars(84)

        result = run_walk_forward_backtest(
            bars=bars,
            strategy_factory=simple_strategy_factory,
            train_bars=63,
            test_bars=21,
        )

        # Equity curve should have entries for test period
        wr = result.window_results[0]
        assert len(wr.equity_curve) == 21  # Test period length
        # Curve should start with initial equity
        assert wr.equity_curve[0] == pytest.approx(wr.initial_equity, rel=0.01)

    def test_walk_forward_metrics_calculation(self):
        """Test that metrics are calculated for each window."""
        bars = generate_synthetic_bars(84, trend=0.001)

        result = run_walk_forward_backtest(
            bars=bars,
            strategy_factory=simple_strategy_factory,
            train_bars=63,
            test_bars=21,
        )

        wr = result.window_results[0]
        metrics = wr.metrics

        # All standard metrics should be present
        expected_keys = [
            "sharpe_ratio",
            "max_drawdown_pct",
            "win_rate",
            "total_return_pct",
            "total_trades",
            "initial_capital",
            "final_equity",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"

    def test_walk_forward_default_config(self):
        """Test that default config is used when not provided."""
        bars = generate_synthetic_bars(84)

        result = run_walk_forward_backtest(
            bars=bars,
            strategy_factory=simple_strategy_factory,
            train_bars=63,
            test_bars=21,
        )

        # Default initial capital should be used
        assert result.window_results[0].initial_equity == 100000.0
