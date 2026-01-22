"""
Fill Logic for Backtesting Module.

This module implements next-bar fill logic with realistic timing delays,
simulating the time between signal generation and order execution.

Features:
- Next-bar fill: Signal on bar N → Fill at bar N+1 open (R9.1.1)
- Configurable delay for different timeframes (R9.1.2, R9.1.3)
- No look-ahead bias (strict time-series split) - R9.1.4

Reference:
- FLUXHERO_REQUIREMENTS.md Feature 9.1: Next-Bar Fill Logic
"""

import numpy as np
from numba import njit


@njit(cache=True)
def get_next_bar_fill_price(
    signal_bar_index: int, open_prices: np.ndarray, delay_bars: int = 1
) -> tuple[float, int]:
    """
    Get fill price at next bar's open.

    Implements R9.1.1: Signal on bar N → Fill at bar N+1 open price.
    This simulates realistic 60s delay for hourly data, preventing look-ahead bias.

    Parameters
    ----------
    signal_bar_index : int
        Bar index where signal was generated
    open_prices : np.ndarray (float64)
        Array of open prices
    delay_bars : int
        Number of bars to delay (default: 1)
        - Minute data: 1 bar (R9.1.2)
        - Daily data: 1 bar (R9.1.3)

    Returns
    -------
    Tuple[float, int]
        (fill_price, fill_bar_index)
        Returns (NaN, -1) if not enough data for fill

    Examples
    --------
    >>> opens = np.array([100.0, 101.0, 102.0, 103.0])
    >>> price, idx = get_next_bar_fill_price(1, opens, delay_bars=1)
    >>> price  # 102.0 (open of bar 2)
    >>> idx    # 2
    """
    fill_bar_index = signal_bar_index + delay_bars

    # Check if fill bar is available (R9.1.4: no peeking into future)
    if fill_bar_index >= len(open_prices):
        return np.nan, -1

    fill_price = open_prices[fill_bar_index]

    # Return NaN if fill price is invalid
    if np.isnan(fill_price):
        return np.nan, -1

    return fill_price, fill_bar_index


@njit(cache=True)
def simulate_intrabar_stop_execution(
    high: float, low: float, close: float, stop_price: float, is_long: bool
) -> tuple[bool, float]:
    """
    Simulate stop loss execution within a bar.

    Assumes stop is hit if price touches stop level during the bar.
    Conservative fill: uses stop price (not worst price in bar).

    Parameters
    ----------
    high : float
        High price of bar
    low : float
        Low price of bar
    close : float
        Close price of bar (fallback if stop not clearly hit)
    stop_price : float
        Stop loss price
    is_long : bool
        True if long position, False if short

    Returns
    -------
    Tuple[bool, float]
        (stop_hit, exit_price)

    Examples
    --------
    >>> # Long position with stop at 95, bar drops to 94
    >>> hit, price = simulate_intrabar_stop_execution(100.0, 94.0, 96.0, 95.0, True)
    >>> hit    # True
    >>> price  # 95.0 (stop price)

    >>> # Long position with stop at 95, bar stays above
    >>> hit, price = simulate_intrabar_stop_execution(100.0, 96.0, 98.0, 95.0, True)
    >>> hit    # False
    >>> price  # 98.0 (close price, not executed)
    """
    if is_long:
        # Long position: stop below entry, hit if low touches stop
        if low <= stop_price:
            return True, stop_price
    else:
        # Short position: stop above entry, hit if high touches stop
        if high >= stop_price:
            return True, stop_price

    # Stop not hit
    return False, close


@njit(cache=True)
def simulate_intrabar_target_execution(
    high: float, low: float, close: float, target_price: float, is_long: bool
) -> tuple[bool, float]:
    """
    Simulate take profit execution within a bar.

    Assumes target is hit if price reaches target level during the bar.
    Optimistic fill: uses target price.

    Parameters
    ----------
    high : float
        High price of bar
    low : float
        Low price of bar
    close : float
        Close price of bar (fallback if target not hit)
    target_price : float
        Take profit price
    is_long : bool
        True if long position, False if short

    Returns
    -------
    Tuple[bool, float]
        (target_hit, exit_price)

    Examples
    --------
    >>> # Long position with target at 110, bar reaches 111
    >>> hit, price = simulate_intrabar_target_execution(111.0, 105.0, 108.0, 110.0, True)
    >>> hit    # True
    >>> price  # 110.0 (target price)

    >>> # Long position with target at 110, bar doesn't reach
    >>> hit, price = simulate_intrabar_target_execution(108.0, 105.0, 107.0, 110.0, True)
    >>> hit    # False
    >>> price  # 107.0 (close price, not executed)
    """
    if is_long:
        # Long position: target above entry, hit if high reaches target
        if high >= target_price:
            return True, target_price
    else:
        # Short position: target below entry, hit if low reaches target
        if low <= target_price:
            return True, target_price

    # Target not hit
    return False, close


@njit(cache=True)
def check_stop_and_target(
    high: float,
    low: float,
    close: float,
    stop_price: float | None,
    target_price: float | None,
    is_long: bool,
) -> tuple[bool, float, str]:
    """
    Check both stop loss and take profit in order of execution.

    Priority logic:
    1. If both stop and target are hit in same bar, assume stop hit first (conservative)
    2. If only one is hit, use that exit
    3. If neither hit, position remains open

    Parameters
    ----------
    high : float
        High price of bar
    low : float
        Low price of bar
    close : float
        Close price of bar
    stop_price : Optional[float]
        Stop loss price (None if no stop)
    target_price : Optional[float]
        Take profit price (None if no target)
    is_long : bool
        True if long position, False if short

    Returns
    -------
    Tuple[bool, float, str]
        (exited, exit_price, exit_reason)
        exit_reason: 'stop', 'target', or 'none'

    Examples
    --------
    >>> # Long with stop 95, target 110, bar hits stop
    >>> exited, price, reason = check_stop_and_target(100.0, 94.0, 96.0, 95.0, 110.0, True)
    >>> exited  # True
    >>> price   # 95.0
    >>> reason  # 'stop'

    >>> # Long with stop 95, target 110, bar hits target
    >>> exited, price, reason = check_stop_and_target(111.0, 106.0, 108.0, 95.0, 110.0, True)
    >>> exited  # True
    >>> price   # 110.0
    >>> reason  # 'target'
    """
    # Check stop loss first (conservative: assume stop hit before target if both touched)
    if stop_price is not None and not np.isnan(stop_price):
        stop_hit, stop_exit_price = simulate_intrabar_stop_execution(
            high, low, close, stop_price, is_long
        )
        if stop_hit:
            return True, stop_exit_price, "stop"

    # Check take profit second
    if target_price is not None and not np.isnan(target_price):
        target_hit, target_exit_price = simulate_intrabar_target_execution(
            high, low, close, target_price, is_long
        )
        if target_hit:
            return True, target_exit_price, "target"

    # Neither hit
    return False, close, "none"


@njit(cache=True)
def validate_no_lookahead(signal_indices: np.ndarray, fill_indices: np.ndarray) -> bool:
    """
    Validate that no look-ahead bias exists (R9.1.4).

    Ensures all fills occur AFTER their corresponding signals.

    Parameters
    ----------
    signal_indices : np.ndarray (int32)
        Bar indices where signals were generated
    fill_indices : np.ndarray (int32)
        Bar indices where fills occurred

    Returns
    -------
    bool
        True if no look-ahead bias detected (all fills after signals)

    Examples
    --------
    >>> signals = np.array([10, 20, 30], dtype=np.int32)
    >>> fills = np.array([11, 21, 31], dtype=np.int32)
    >>> validate_no_lookahead(signals, fills)  # True

    >>> signals = np.array([10, 20, 30], dtype=np.int32)
    >>> fills = np.array([11, 19, 31], dtype=np.int32)  # Fill at 19 before signal at 20!
    >>> validate_no_lookahead(signals, fills)  # False
    """
    if len(signal_indices) != len(fill_indices):
        return False

    for i in range(len(signal_indices)):
        # Fill must occur AFTER signal (fill_index > signal_index)
        if fill_indices[i] <= signal_indices[i]:
            return False

    return True
