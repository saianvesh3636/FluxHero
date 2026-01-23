"""
Walk-Forward Testing Module for FluxHero Trading System.

This module implements walk-forward analysis, a technique to validate trading
strategies by dividing historical data into consecutive train/test windows
and evaluating out-of-sample performance.

Features:
- WalkForwardWindow dataclass with train/test indices and dates
- Window generation with configurable train/test periods
- Edge case handling (insufficient data, uneven final window, date gaps)

Reference:
- FLUXHERO_REQUIREMENTS.md R9.4 Walk-Forward Testing
"""

from dataclasses import dataclass
from datetime import datetime

from numpy.typing import NDArray


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
