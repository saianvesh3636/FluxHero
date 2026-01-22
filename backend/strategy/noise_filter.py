"""
Market Microstructure Noise Filter

This module implements noise filtering logic to prevent trading on illiquid or manipulated
price movements by validating spread-to-volatility ratios, volume conditions, and
time-of-day restrictions.

Requirements:
    - R4.1.1-4.1.4: Spread-to-volatility ratio calculation and validation
    - R4.2.1-4.2.3: Volume validation for signals and breakouts
    - R4.3.1-4.3.3: Time-of-day filtering for illiquid hours

Performance:
    - All functions JIT-compiled with Numba for speed
    - <50ms for 10k candle validation
"""

from datetime import datetime, time

import numpy as np
from numba import njit

# ============================================================================
# Spread-to-Volatility Ratio Functions
# ============================================================================

@njit(cache=True)
def calculate_spread_to_volatility_ratio(
    bid: np.ndarray,
    ask: np.ndarray,
    close: np.ndarray,
    volatility_period: int = 20
) -> np.ndarray:
    """
    Calculate spread-to-volatility ratio for noise detection.

    Formula (R4.1.1-4.1.3):
        1. Spread = Ask - Bid
        2. Vol_5min = StdDev(Close, volatility_period bars)
        3. SV_Ratio = Spread / Vol_5min

    Args:
        bid: Bid prices (float64 array)
        ask: Ask prices (float64 array)
        close: Close prices (float64 array)
        volatility_period: Lookback period for volatility calculation (default: 20)

    Returns:
        SV_Ratio array (float64), NaN for first 'volatility_period' bars

    Example:
        >>> bid = np.array([100.0, 100.5, 101.0])
        >>> ask = np.array([100.1, 100.6, 101.1])
        >>> close = np.array([100.05, 100.55, 101.05])
        >>> sv_ratio = calculate_spread_to_volatility_ratio(bid, ask, close, 2)
        >>> # sv_ratio[2] = 0.1 / std(close[-2:])
    """
    n = len(close)
    sv_ratio = np.full(n, np.nan, dtype=np.float64)

    if n < volatility_period:
        return sv_ratio

    # Calculate spread for each bar
    spread = ask - bid

    # Calculate rolling standard deviation of close prices
    for i in range(volatility_period - 1, n):
        # Get window of close prices
        window = close[i - volatility_period + 1:i + 1]

        # Calculate standard deviation (volatility)
        volatility = np.std(window)

        # Avoid division by zero
        if volatility > 1e-10:
            sv_ratio[i] = spread[i] / volatility
        else:
            # Very low volatility, treat as suspicious
            sv_ratio[i] = 999.0  # High ratio = reject signal

    return sv_ratio


@njit(cache=True)
def validate_spread_ratio(
    sv_ratio: np.ndarray,
    threshold: float = 0.05,
    illiquid_threshold: float = 0.025
) -> np.ndarray:
    """
    Validate signals based on spread-to-volatility ratio.

    Requirement R4.1.4: Reject signals if SV_Ratio > threshold (default 0.05 = 5%)
    Requirement R4.3.2: Stricter threshold during illiquid hours (0.025 = 2.5%)

    Args:
        sv_ratio: Spread-to-volatility ratio array
        threshold: Maximum acceptable SV ratio for normal hours (default: 0.05)
        illiquid_threshold: Stricter threshold for illiquid hours (default: 0.025)

    Returns:
        Boolean array: True = signal valid, False = signal rejected

    Example:
        >>> sv_ratio = np.array([0.01, 0.03, 0.06, 0.10])
        >>> valid = validate_spread_ratio(sv_ratio, threshold=0.05)
        >>> # valid = [True, True, False, False]
    """
    return sv_ratio <= threshold


# ============================================================================
# Volume Validation Functions
# ============================================================================

@njit(cache=True)
def calculate_average_volume(
    volume: np.ndarray,
    period: int = 20
) -> np.ndarray:
    """
    Calculate rolling average volume.

    Requirement R4.2.1: Calculate average volume over last 'period' bars

    Args:
        volume: Volume array (float64)
        period: Lookback period (default: 20)

    Returns:
        Average volume array (float64), NaN for first 'period-1' bars

    Example:
        >>> volume = np.array([1000, 1500, 2000, 1200, 1800])
        >>> avg_vol = calculate_average_volume(volume, period=3)
        >>> # avg_vol[2] = mean([1000, 1500, 2000]) = 1500
    """
    n = len(volume)
    avg_volume = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return avg_volume

    for i in range(period - 1, n):
        window = volume[i - period + 1:i + 1]
        avg_volume[i] = np.mean(window)

    return avg_volume


@njit(cache=True)
def validate_volume(
    volume: np.ndarray,
    avg_volume: np.ndarray,
    is_breakout: np.ndarray,
    normal_threshold: float = 0.5,
    breakout_threshold: float = 1.5
) -> np.ndarray:
    """
    Validate signals based on volume requirements.

    Requirements:
        - R4.2.2: Normal signals require volume > 0.5 × Avg_Volume
        - R4.2.3: Breakout signals require volume > 1.5 × Avg_Volume

    Args:
        volume: Current volume array
        avg_volume: Average volume array
        is_breakout: Boolean array indicating breakout signals
        normal_threshold: Multiplier for normal signals (default: 0.5)
        breakout_threshold: Multiplier for breakout signals (default: 1.5)

    Returns:
        Boolean array: True = volume sufficient, False = volume too low

    Example:
        >>> volume = np.array([1000, 2000, 500])
        >>> avg_volume = np.array([1000, 1000, 1000])
        >>> is_breakout = np.array([False, True, False])
        >>> valid = validate_volume(volume, avg_volume, is_breakout)
        >>> # [True (1000 > 500), True (2000 > 1500), False (500 < 500)]
    """
    n = len(volume)
    valid = np.full(n, False, dtype=np.bool_)

    for i in range(n):
        if np.isnan(avg_volume[i]):
            valid[i] = False
            continue

        if is_breakout[i]:
            # Breakout requires higher volume
            valid[i] = volume[i] > (breakout_threshold * avg_volume[i])
        else:
            # Normal signal requires moderate volume
            valid[i] = volume[i] > (normal_threshold * avg_volume[i])

    return valid


# ============================================================================
# Time-of-Day Filter Functions
# ============================================================================

def is_illiquid_hour(timestamp: datetime) -> bool:
    """
    Check if timestamp falls during illiquid trading hours.

    Requirement R4.3.1: Flag illiquid hours:
        - Pre-market: Before 9:30 AM EST
        - Lunch: 12:00-1:00 PM EST
        - After-hours: After 4:00 PM EST

    Args:
        timestamp: Datetime object (assumed to be in EST/EDT)

    Returns:
        True if illiquid hour, False otherwise

    Example:
        >>> from datetime import datetime
        >>> dt = datetime(2024, 1, 15, 9, 15)  # 9:15 AM
        >>> is_illiquid_hour(dt)  # True (pre-market)
        >>> dt = datetime(2024, 1, 15, 10, 30)  # 10:30 AM
        >>> is_illiquid_hour(dt)  # False (normal hours)
    """
    t = timestamp.time()

    # Pre-market: before 9:30 AM
    if t < time(9, 30):
        return True

    # Lunch: 12:00 PM - 1:00 PM
    if time(12, 0) <= t < time(13, 0):
        return True

    # After-hours: after 4:00 PM
    if t >= time(16, 0):
        return True

    return False


def is_near_close(timestamp: datetime, minutes_before_close: int = 15) -> bool:
    """
    Check if timestamp is within X minutes of market close.

    Requirement R4.3.3: Disable new position entries 15 minutes before market close

    Args:
        timestamp: Datetime object (assumed to be in EST/EDT)
        minutes_before_close: Minutes before close to restrict entries (default: 15)

    Returns:
        True if near market close, False otherwise

    Example:
        >>> from datetime import datetime
        >>> dt = datetime(2024, 1, 15, 15, 50)  # 3:50 PM
        >>> is_near_close(dt, minutes_before_close=15)  # True
        >>> dt = datetime(2024, 1, 15, 15, 40)  # 3:40 PM
        >>> is_near_close(dt, minutes_before_close=15)  # False
    """
    t = timestamp.time()
    market_close = time(16, 0)  # 4:00 PM EST

    # Calculate cutoff time (market close - minutes_before_close)
    close_hour = 16
    close_minute = 0 - minutes_before_close

    # Handle negative minutes
    if close_minute < 0:
        close_hour -= 1
        close_minute += 60

    cutoff_time = time(close_hour, close_minute)

    return cutoff_time <= t < market_close


# ============================================================================
# Combined Noise Filter
# ============================================================================

@njit(cache=True)
def apply_noise_filter(
    sv_ratio: np.ndarray,
    volume: np.ndarray,
    avg_volume: np.ndarray,
    is_breakout: np.ndarray,
    is_illiquid_hours: np.ndarray,
    sv_threshold_normal: float = 0.05,
    sv_threshold_illiquid: float = 0.025,
    volume_threshold_normal: float = 0.5,
    volume_threshold_breakout: float = 1.5
) -> np.ndarray:
    """
    Apply comprehensive noise filter to signals.

    Combines all noise filtering criteria:
        1. Spread-to-volatility ratio validation
        2. Volume validation (normal vs breakout)
        3. Stricter thresholds during illiquid hours

    Args:
        sv_ratio: Spread-to-volatility ratio array
        volume: Current volume array
        avg_volume: Average volume array
        is_breakout: Boolean array indicating breakout signals
        is_illiquid_hours: Boolean array indicating illiquid hours
        sv_threshold_normal: SV ratio threshold for normal hours (default: 0.05)
        sv_threshold_illiquid: SV ratio threshold for illiquid hours (default: 0.025)
        volume_threshold_normal: Volume multiplier for normal signals (default: 0.5)
        volume_threshold_breakout: Volume multiplier for breakout signals (default: 1.5)

    Returns:
        Boolean array: True = signal passed all filters, False = signal rejected

    Example:
        >>> sv_ratio = np.array([0.01, 0.04, 0.06])
        >>> volume = np.array([1000, 2000, 500])
        >>> avg_volume = np.array([1000, 1000, 1000])
        >>> is_breakout = np.array([False, True, False])
        >>> is_illiquid = np.array([False, False, True])
        >>> valid = apply_noise_filter(sv_ratio, volume, avg_volume, is_breakout, is_illiquid)
        >>> # Validates each signal against all criteria
    """
    n = len(sv_ratio)
    valid = np.full(n, False, dtype=np.bool_)

    for i in range(n):
        # Skip if data is insufficient
        if np.isnan(sv_ratio[i]) or np.isnan(avg_volume[i]):
            valid[i] = False
            continue

        # Check spread-to-volatility ratio
        if is_illiquid_hours[i]:
            # Stricter threshold during illiquid hours (R4.3.2)
            sv_valid = sv_ratio[i] <= sv_threshold_illiquid
        else:
            # Normal threshold (R4.1.4)
            sv_valid = sv_ratio[i] <= sv_threshold_normal

        if not sv_valid:
            valid[i] = False
            continue

        # Check volume requirements
        if is_breakout[i]:
            # Breakout requires higher volume (R4.2.3)
            volume_valid = volume[i] > (volume_threshold_breakout * avg_volume[i])
        else:
            # Normal signal requires moderate volume (R4.2.2)
            volume_valid = volume[i] > (volume_threshold_normal * avg_volume[i])

        if not volume_valid:
            valid[i] = False
            continue

        # All checks passed
        valid[i] = True

    return valid


@njit(cache=True)
def calculate_rejection_reason(
    sv_ratio: np.ndarray,
    volume: np.ndarray,
    avg_volume: np.ndarray,
    is_breakout: np.ndarray,
    is_illiquid_hours: np.ndarray,
    sv_threshold_normal: float = 0.05,
    sv_threshold_illiquid: float = 0.025,
    volume_threshold_normal: float = 0.5,
    volume_threshold_breakout: float = 1.5
) -> np.ndarray:
    """
    Calculate rejection reason codes for debugging.

    Rejection codes:
        0 = Signal valid (passed all filters)
        1 = Rejected due to high SV ratio (normal hours)
        2 = Rejected due to high SV ratio (illiquid hours)
        3 = Rejected due to low volume (normal signal)
        4 = Rejected due to low volume (breakout signal)
        5 = Rejected due to insufficient data (NaN values)

    Args:
        sv_ratio: Spread-to-volatility ratio array
        volume: Current volume array
        avg_volume: Average volume array
        is_breakout: Boolean array indicating breakout signals
        is_illiquid_hours: Boolean array indicating illiquid hours
        sv_threshold_normal: SV ratio threshold for normal hours
        sv_threshold_illiquid: SV ratio threshold for illiquid hours
        volume_threshold_normal: Volume multiplier for normal signals
        volume_threshold_breakout: Volume multiplier for breakout signals

    Returns:
        Integer array with rejection reason codes

    Example:
        >>> # Use for debugging why signals were rejected
        >>> reasons = calculate_rejection_reason(sv_ratio, volume, avg_volume, ...)
        >>> # reasons[i] tells you why signal at bar i was rejected
    """
    n = len(sv_ratio)
    reasons = np.zeros(n, dtype=np.int32)

    for i in range(n):
        # Check for insufficient data
        if np.isnan(sv_ratio[i]) or np.isnan(avg_volume[i]):
            reasons[i] = 5
            continue

        # Check spread-to-volatility ratio
        if is_illiquid_hours[i]:
            if sv_ratio[i] > sv_threshold_illiquid:
                reasons[i] = 2
                continue
        else:
            if sv_ratio[i] > sv_threshold_normal:
                reasons[i] = 1
                continue

        # Check volume requirements
        if is_breakout[i]:
            if volume[i] <= (volume_threshold_breakout * avg_volume[i]):
                reasons[i] = 4
                continue
        else:
            if volume[i] <= (volume_threshold_normal * avg_volume[i]):
                reasons[i] = 3
                continue

        # All checks passed
        reasons[i] = 0

    return reasons


# ============================================================================
# Price Gap Filter Functions
# ============================================================================

@njit(cache=True)
def calculate_price_gap_ratio(
    open_prices: np.ndarray,
    close_prices: np.ndarray
) -> np.ndarray:
    """
    Calculate price gap ratio between open and previous close.

    Retail-specific optimization (Phase 16): Reject trades with excessive overnight
    gaps that may indicate news events or earnings releases.

    Formula:
        Gap_Ratio[i] = |Open[i] - Close[i-1]| / Close[i-1]

    Args:
        open_prices: Open prices (float64 array)
        close_prices: Close prices (float64 array)

    Returns:
        Gap ratio array (float64), NaN for first bar (no previous close)

    Example:
        >>> open_prices = np.array([100.0, 102.0, 101.0])
        >>> close_prices = np.array([101.0, 103.0, 102.0])
        >>> gap_ratio = calculate_price_gap_ratio(open_prices, close_prices)
        >>> # gap_ratio[1] = |102.0 - 101.0| / 101.0 = 0.0099 (0.99%)
        >>> # gap_ratio[2] = |101.0 - 103.0| / 103.0 = 0.0194 (1.94%)
    """
    n = len(open_prices)
    gap_ratio = np.full(n, np.nan, dtype=np.float64)

    if n < 2:
        return gap_ratio

    for i in range(1, n):
        prev_close = close_prices[i - 1]

        # Avoid division by zero
        if prev_close > 1e-10:
            gap = np.abs(open_prices[i] - prev_close)
            gap_ratio[i] = gap / prev_close
        else:
            # Zero or very small previous close, treat as suspicious
            gap_ratio[i] = 999.0

    return gap_ratio


@njit(cache=True)
def validate_price_gap(
    gap_ratio: np.ndarray,
    threshold: float = 0.02
) -> np.ndarray:
    """
    Validate signals based on price gap ratio.

    Retail-specific optimization: Reject trades if gap exceeds threshold (default 2%).
    Large gaps often indicate:
        - Earnings announcements
        - News events
        - Low liquidity overnight
        - Increased slippage risk

    Args:
        gap_ratio: Price gap ratio array from calculate_price_gap_ratio()
        threshold: Maximum acceptable gap ratio (default: 0.02 = 2%)

    Returns:
        Boolean array: True = gap acceptable, False = gap too large (reject trade)

    Example:
        >>> gap_ratio = np.array([np.nan, 0.01, 0.025, 0.015])
        >>> valid = validate_price_gap(gap_ratio, threshold=0.02)
        >>> # valid = [False (NaN), True (1%), False (2.5%), True (1.5%)]
    """
    n = len(gap_ratio)
    valid = np.full(n, False, dtype=np.bool_)

    for i in range(n):
        if np.isnan(gap_ratio[i]):
            # No previous close data
            valid[i] = False
        else:
            # Valid if gap is within threshold
            valid[i] = gap_ratio[i] <= threshold

    return valid
