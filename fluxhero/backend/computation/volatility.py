"""
Volatility-Adaptive Smoothing Module using Numba JIT Compilation

This module implements volatility-based adaptive adjustments for technical indicators.
It dynamically adjusts indicator lookback periods and alpha values based on ATR,
allowing faster reactions in high-volatility environments.

Key Features:
- 14-period ATR baseline calculation
- Volatility state classification (LOW/NORMAL/HIGH)
- Dynamic period adjustment (shorten in high vol, lengthen in low vol)
- Multi-timeframe volatility checker (5-min vs 1-hour)
- Volatility-alpha linkage using regression

Performance targets: <200ms for 10,000 candles
"""

import numpy as np
from numba import njit


# Volatility states (constants for regime classification)
VOL_STATE_LOW = 0
VOL_STATE_NORMAL = 1
VOL_STATE_HIGH = 2


@njit(cache=True)
def calculate_atr_ma(atr: np.ndarray, period: int = 50) -> np.ndarray:
    """
    Calculate Simple Moving Average of ATR for baseline comparison.

    Used to determine relative volatility states by comparing current ATR
    to its long-term average.

    Args:
        atr: 1D array of ATR values (float64)
        period: Number of periods for moving average (default: 50)

    Returns:
        1D array of ATR moving average values (initial values are NaN)

    Example:
        >>> atr = np.array([1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
        >>> atr_ma = calculate_atr_ma(atr, period=3)
    """
    n = len(atr)
    atr_ma = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return atr_ma

    # Calculate SMA for each valid window
    for i in range(period - 1, n):
        window_sum = 0.0
        count = 0
        for j in range(i - period + 1, i + 1):
            if not np.isnan(atr[j]):
                window_sum += atr[j]
                count += 1

        if count > 0:
            atr_ma[i] = window_sum / count

    return atr_ma


@njit(cache=True)
def classify_volatility_state(
    atr: np.ndarray,
    atr_ma: np.ndarray,
    low_threshold: float = 0.5,
    high_threshold: float = 1.5
) -> np.ndarray:
    """
    Classify volatility state for each bar based on ATR vs ATR_MA ratio.

    Volatility States:
        - LOW (0): ATR < low_threshold × ATR_MA (default: 0.5)
        - NORMAL (1): low_threshold × ATR_MA ≤ ATR ≤ high_threshold × ATR_MA
        - HIGH (2): ATR > high_threshold × ATR_MA (default: 1.5)

    Requirements:
        - R3.1.2: Define volatility states based on ATR/ATR_MA ratio

    Args:
        atr: 1D array of ATR values (float64)
        atr_ma: 1D array of ATR moving average values (float64)
        low_threshold: Multiplier for low volatility detection (default: 0.5)
        high_threshold: Multiplier for high volatility detection (default: 1.5)

    Returns:
        1D array of volatility states (0=LOW, 1=NORMAL, 2=HIGH)

    Example:
        >>> atr = np.array([1.0, 2.0, 3.0, 2.5, 1.5])
        >>> atr_ma = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        >>> states = classify_volatility_state(atr, atr_ma)
        >>> # Result: [0, 1, 2, 1, 1] (LOW, NORMAL, HIGH, NORMAL, NORMAL)
    """
    n = len(atr)
    states = np.full(n, VOL_STATE_NORMAL, dtype=np.int32)

    for i in range(n):
        if np.isnan(atr[i]) or np.isnan(atr_ma[i]) or atr_ma[i] == 0:
            continue

        ratio = atr[i] / atr_ma[i]

        if ratio < low_threshold:
            states[i] = VOL_STATE_LOW
        elif ratio > high_threshold:
            states[i] = VOL_STATE_HIGH
        else:
            states[i] = VOL_STATE_NORMAL

    return states


@njit(cache=True)
def adjust_period_for_volatility(
    base_period: int,
    vol_state: int,
    low_vol_multiplier: float = 1.3,
    high_vol_multiplier: float = 0.7
) -> int:
    """
    Adjust indicator lookback period based on volatility state.

    Strategy:
        - High volatility: Shorten period (×0.7) for faster reactions
        - Low volatility: Lengthen period (×1.3) to smooth noise
        - Normal volatility: Use base period

    Requirements:
        - R3.1.3: Adjust indicator periods dynamically based on volatility

    Args:
        base_period: Original indicator period
        vol_state: Current volatility state (0=LOW, 1=NORMAL, 2=HIGH)
        low_vol_multiplier: Period multiplier for low volatility (default: 1.3)
        high_vol_multiplier: Period multiplier for high volatility (default: 0.7)

    Returns:
        Adjusted period (integer, minimum 2)

    Example:
        >>> adjust_period_for_volatility(20, VOL_STATE_HIGH)
        14  # 20 × 0.7 = 14
        >>> adjust_period_for_volatility(20, VOL_STATE_LOW)
        26  # 20 × 1.3 = 26
    """
    if vol_state == VOL_STATE_HIGH:
        adjusted = int(base_period * high_vol_multiplier)
    elif vol_state == VOL_STATE_LOW:
        adjusted = int(base_period * low_vol_multiplier)
    else:
        adjusted = base_period

    # Ensure minimum period of 2
    return max(adjusted, 2)


@njit(cache=True)
def detect_volatility_spike(
    atr_short: np.ndarray,
    atr_long: np.ndarray,
    spike_threshold: float = 2.0
) -> np.ndarray:
    """
    Detect volatility spikes by comparing short and long timeframe ATR.

    A volatility spike occurs when short-term ATR exceeds long-term ATR
    by the threshold multiplier (default: 2×).

    Requirements:
        - R3.2.1: Compare current 5-min ATR vs 1-hour ATR
        - R3.2.2: If 5-min ATR > 2× hourly ATR → Flag "volatility spike"

    Args:
        atr_short: ATR from shorter timeframe (e.g., 5-min) (float64)
        atr_long: ATR from longer timeframe (e.g., 1-hour) (float64)
        spike_threshold: Multiplier for spike detection (default: 2.0)

    Returns:
        1D boolean array (True = spike detected, False = normal)

    Note:
        During volatility spikes, recommended actions:
        - Widen stops by 1.5×
        - Reduce position size by 30%
        - Increase signal confirmation requirements

    Example:
        >>> atr_5min = np.array([2.0, 4.0, 3.5, 2.5])
        >>> atr_1hour = np.array([2.0, 2.0, 2.0, 2.0])
        >>> spikes = detect_volatility_spike(atr_5min, atr_1hour)
        >>> # Result: [False, True, True, False]
    """
    n = len(atr_short)
    spikes = np.zeros(n, dtype=np.bool_)

    for i in range(n):
        if np.isnan(atr_short[i]) or np.isnan(atr_long[i]):
            continue

        if atr_long[i] > 0 and atr_short[i] > spike_threshold * atr_long[i]:
            spikes[i] = True

    return spikes


@njit(cache=True)
def calculate_volatility_alpha(
    atr: np.ndarray,
    atr_ma: np.ndarray,
    min_alpha: float = 0.1,
    max_alpha: float = 0.6
) -> np.ndarray:
    """
    Calculate adaptive alpha (smoothing constant) based on ATR.

    Strategy:
        - High volatility (ATR rising) → High α (0.4-0.6) for fast reactions
        - Low volatility (ATR falling) → Low α (0.1-0.2) to smooth noise
        - Linear interpolation between min and max based on ATR/ATR_MA ratio

    Requirements:
        - R3.3.1: High volatility → Use high α (0.4-0.6)
        - R3.3.2: Low volatility → Use low α (0.1-0.2)
        - R3.3.3: Regression-based α = f(ATR) relationship

    Args:
        atr: 1D array of ATR values (float64)
        atr_ma: 1D array of ATR moving average (float64)
        min_alpha: Minimum alpha for low volatility (default: 0.1)
        max_alpha: Maximum alpha for high volatility (default: 0.6)

    Returns:
        1D array of adaptive alpha values (range: min_alpha to max_alpha)

    Formula:
        ratio = ATR / ATR_MA
        normalized_ratio = (ratio - 0.5) / (1.5 - 0.5)  # Map [0.5, 1.5] to [0, 1]
        alpha = min_alpha + (max_alpha - min_alpha) × normalized_ratio

    Example:
        >>> atr = np.array([1.0, 2.0, 3.0])
        >>> atr_ma = np.array([2.0, 2.0, 2.0])
        >>> alphas = calculate_volatility_alpha(atr, atr_ma)
        >>> # ATR/ATR_MA ratios: [0.5, 1.0, 1.5]
        >>> # Alphas: [0.1, 0.35, 0.6]
    """
    n = len(atr)
    alphas = np.full(n, (min_alpha + max_alpha) / 2.0, dtype=np.float64)

    # Thresholds from classify_volatility_state
    low_threshold = 0.5
    high_threshold = 1.5

    for i in range(n):
        if np.isnan(atr[i]) or np.isnan(atr_ma[i]) or atr_ma[i] == 0:
            continue

        ratio = atr[i] / atr_ma[i]

        # Clamp ratio to [low_threshold, high_threshold]
        ratio = max(low_threshold, min(high_threshold, ratio))

        # Normalize ratio to [0, 1] range
        normalized_ratio = (ratio - low_threshold) / (high_threshold - low_threshold)

        # Calculate alpha using linear interpolation
        alphas[i] = min_alpha + (max_alpha - min_alpha) * normalized_ratio

    return alphas


@njit(cache=True)
def calculate_adaptive_ema_with_volatility(
    prices: np.ndarray,
    atr: np.ndarray,
    atr_ma: np.ndarray,
    base_period: int = 20
) -> np.ndarray:
    """
    Calculate EMA with volatility-adaptive alpha.

    This combines the EMA calculation with volatility-based alpha adjustment,
    allowing the indicator to respond faster during high volatility and
    smooth out noise during low volatility.

    Requirements:
        - R3.3: Volatility-Alpha Linkage implementation

    Args:
        prices: 1D array of closing prices (float64)
        atr: 1D array of ATR values (float64)
        atr_ma: 1D array of ATR moving average (float64)
        base_period: Base EMA period (used for initialization, default: 20)

    Returns:
        1D array of adaptive EMA values

    Example:
        >>> prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        >>> atr = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        >>> atr_ma = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        >>> ema = calculate_adaptive_ema_with_volatility(prices, atr, atr_ma, 20)
    """
    n = len(prices)
    ema = np.full(n, np.nan, dtype=np.float64)

    if n < base_period:
        return ema

    # Calculate adaptive alphas
    alphas = calculate_volatility_alpha(atr, atr_ma)

    # Initialize EMA with SMA of first 'base_period' values
    sma = 0.0
    for i in range(base_period):
        sma += prices[i]
    ema[base_period - 1] = sma / base_period

    # Calculate adaptive EMA using volatility-based alpha
    for i in range(base_period, n):
        if not np.isnan(alphas[i]):
            ema[i] = (prices[i] * alphas[i]) + (ema[i - 1] * (1.0 - alphas[i]))
        else:
            # Fallback to standard EMA if alpha is invalid
            alpha = 2.0 / (base_period + 1.0)
            ema[i] = (prices[i] * alpha) + (ema[i - 1] * (1.0 - alpha))

    return ema


@njit(cache=True)
def get_stop_loss_multiplier(is_spike: bool) -> float:
    """
    Get stop loss multiplier for volatility spike conditions.

    Requirements:
        - R3.2.3: During volatility spikes, widen stops by 1.5×

    Args:
        is_spike: Boolean indicating if volatility spike is detected

    Returns:
        Stop loss multiplier (1.5 if spike, 1.0 if normal)

    Example:
        >>> get_stop_loss_multiplier(True)
        1.5
        >>> get_stop_loss_multiplier(False)
        1.0
    """
    return 1.5 if is_spike else 1.0


@njit(cache=True)
def get_position_size_multiplier(is_spike: bool) -> float:
    """
    Get position size multiplier for volatility spike conditions.

    Requirements:
        - R3.2.3: During volatility spikes, reduce position size by 30%

    Args:
        is_spike: Boolean indicating if volatility spike is detected

    Returns:
        Position size multiplier (0.7 if spike, 1.0 if normal)

    Example:
        >>> get_position_size_multiplier(True)
        0.7  # Reduce to 70% of normal size
        >>> get_position_size_multiplier(False)
        1.0
    """
    return 0.7 if is_spike else 1.0
