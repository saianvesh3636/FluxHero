"""
Anticipatory Indicators for Regime Change Detection.

This module provides leading indicators that can anticipate regime changes
before they occur, allowing for proactive position adjustments.

Indicators:
- Volatility Squeeze: Detects when BB contracts inside Keltner Channels
- RSI/Price Divergence: Detects momentum divergence from price
- Volume Exhaustion: Detects declining volume in trends
- Regime Derivatives: Velocity and acceleration of ER/ADX

All functions are Numba JIT-compiled for performance.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def calculate_keltner_channels(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    ema_period: int = 20,
    atr_period: int = 10,
    atr_mult: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Keltner Channels.

    Keltner Channels are volatility-based bands set above and below an EMA.
    Used with Bollinger Bands to detect volatility squeezes.

    Parameters
    ----------
    close : np.ndarray
        Closing prices
    high : np.ndarray
        High prices
    low : np.ndarray
        Low prices
    ema_period : int
        Period for the center EMA (default: 20)
    atr_period : int
        Period for ATR calculation (default: 10)
    atr_mult : float
        ATR multiplier for channel width (default: 1.5)

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (upper_channel, middle_ema, lower_channel)
    """
    n = len(close)
    upper = np.full(n, np.nan, dtype=np.float64)
    middle = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)

    if n < max(ema_period, atr_period):
        return upper, middle, lower

    # Calculate EMA (middle line)
    alpha = 2.0 / (ema_period + 1.0)
    ema = np.full(n, np.nan, dtype=np.float64)

    # Initialize with SMA
    sma = 0.0
    for i in range(ema_period):
        sma += close[i]
    ema[ema_period - 1] = sma / ema_period

    for i in range(ema_period, n):
        ema[i] = close[i] * alpha + ema[i - 1] * (1.0 - alpha)

    # Calculate ATR
    atr = np.full(n, np.nan, dtype=np.float64)
    tr_sum = 0.0

    for i in range(1, n):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
        if i < atr_period:
            tr_sum += tr
            if i == atr_period - 1:
                atr[i] = tr_sum / atr_period
        else:
            atr[i] = (atr[i - 1] * (atr_period - 1) + tr) / atr_period

    # Calculate channels
    for i in range(max(ema_period, atr_period) - 1, n):
        if not np.isnan(ema[i]) and not np.isnan(atr[i]):
            middle[i] = ema[i]
            upper[i] = ema[i] + atr_mult * atr[i]
            lower[i] = ema[i] - atr_mult * atr[i]

    return upper, middle, lower


@njit(cache=True)
def calculate_squeeze(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    bb_period: int = 20,
    bb_std: float = 2.0,
    kc_period: int = 20,
    kc_atr_period: int = 10,
    kc_mult: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate Volatility Squeeze indicator.

    A squeeze occurs when Bollinger Bands contract inside Keltner Channels,
    indicating low volatility that often precedes significant price moves.

    Parameters
    ----------
    close : np.ndarray
        Closing prices
    high : np.ndarray
        High prices
    low : np.ndarray
        Low prices
    bb_period : int
        Bollinger Band period (default: 20)
    bb_std : float
        Bollinger Band standard deviation multiplier (default: 2.0)
    kc_period : int
        Keltner Channel EMA period (default: 20)
    kc_atr_period : int
        Keltner Channel ATR period (default: 10)
    kc_mult : float
        Keltner Channel ATR multiplier (default: 1.5)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (squeeze_on, squeeze_intensity)
        - squeeze_on: 1.0 if squeeze active, 0.0 otherwise
        - squeeze_intensity: 0-1 score (higher = tighter squeeze)
    """
    n = len(close)
    squeeze_on = np.zeros(n, dtype=np.float64)
    squeeze_intensity = np.zeros(n, dtype=np.float64)

    warmup = max(bb_period, kc_period, kc_atr_period)
    if n < warmup:
        return squeeze_on, squeeze_intensity

    # Calculate Bollinger Bands
    bb_upper = np.full(n, np.nan, dtype=np.float64)
    bb_lower = np.full(n, np.nan, dtype=np.float64)

    for i in range(bb_period - 1, n):
        sma = 0.0
        for j in range(bb_period):
            sma += close[i - j]
        sma /= bb_period

        variance = 0.0
        for j in range(bb_period):
            diff = close[i - j] - sma
            variance += diff * diff
        std = np.sqrt(variance / bb_period)

        bb_upper[i] = sma + bb_std * std
        bb_lower[i] = sma - bb_std * std

    # Calculate Keltner Channels
    kc_upper, kc_middle, kc_lower = calculate_keltner_channels(
        close, high, low, kc_period, kc_atr_period, kc_mult
    )

    # Detect squeeze
    for i in range(warmup - 1, n):
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(kc_upper[i]) or np.isnan(kc_lower[i])):
            continue

        # Squeeze is ON when BB is inside KC
        bb_inside_kc = bb_lower[i] > kc_lower[i] and bb_upper[i] < kc_upper[i]

        if bb_inside_kc:
            squeeze_on[i] = 1.0

            # Calculate intensity (how tight is the squeeze)
            bb_width = bb_upper[i] - bb_lower[i]
            kc_width = kc_upper[i] - kc_lower[i]

            if kc_width > 0:
                # Intensity = 1 - (BB width / KC width), clamped to 0-1
                intensity = 1.0 - (bb_width / kc_width)
                squeeze_intensity[i] = max(0.0, min(1.0, intensity))

    return squeeze_on, squeeze_intensity


@njit(cache=True)
def calculate_divergence(
    prices: np.ndarray,
    rsi: np.ndarray,
    lookback: int = 14,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate RSI/Price Divergence indicator.

    Divergence occurs when price makes new highs/lows but RSI doesn't confirm,
    often signaling a potential reversal.

    Parameters
    ----------
    prices : np.ndarray
        Closing prices
    rsi : np.ndarray
        RSI values (0-100)
    lookback : int
        Lookback period for finding local extremes (default: 14)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (bullish_div, bearish_div)
        - bullish_div: 0-1 score (higher = stronger bullish divergence)
        - bearish_div: 0-1 score (higher = stronger bearish divergence)
    """
    n = len(prices)
    bullish_div = np.zeros(n, dtype=np.float64)
    bearish_div = np.zeros(n, dtype=np.float64)

    if n < lookback * 2:
        return bullish_div, bearish_div

    for i in range(lookback * 2, n):
        # Skip if RSI is NaN
        if np.isnan(rsi[i]) or np.isnan(rsi[i - lookback]):
            continue

        # Find local extremes in the lookback window
        price_max_idx = i - lookback
        price_min_idx = i - lookback
        rsi_max_idx = i - lookback
        rsi_min_idx = i - lookback

        for j in range(i - lookback, i + 1):
            if prices[j] > prices[price_max_idx]:
                price_max_idx = j
            if prices[j] < prices[price_min_idx]:
                price_min_idx = j
            if not np.isnan(rsi[j]):
                if rsi[j] > rsi[rsi_max_idx]:
                    rsi_max_idx = j
                if rsi[j] < rsi[rsi_min_idx]:
                    rsi_min_idx = j

        # Bearish divergence: Price making higher high, RSI making lower high
        if (prices[i] >= prices[price_max_idx] * 0.995 and  # Near or at price high
            rsi[i] < rsi[rsi_max_idx] - 5.0):  # RSI significantly lower
            # Score based on RSI gap
            rsi_gap = rsi[rsi_max_idx] - rsi[i]
            bearish_div[i] = min(1.0, rsi_gap / 20.0)  # Normalize to 0-1

        # Bullish divergence: Price making lower low, RSI making higher low
        if (prices[i] <= prices[price_min_idx] * 1.005 and  # Near or at price low
            rsi[i] > rsi[rsi_min_idx] + 5.0):  # RSI significantly higher
            # Score based on RSI gap
            rsi_gap = rsi[i] - rsi[rsi_min_idx]
            bullish_div[i] = min(1.0, rsi_gap / 20.0)  # Normalize to 0-1

    return bullish_div, bearish_div


@njit(cache=True)
def calculate_volume_exhaustion(
    close: np.ndarray,
    volume: np.ndarray,
    lookback: int = 20,
) -> np.ndarray:
    """
    Calculate Volume Exhaustion indicator.

    Detects when a trend is losing steam - price continues in trend direction
    but volume is declining, suggesting exhaustion.

    Parameters
    ----------
    close : np.ndarray
        Closing prices
    volume : np.ndarray
        Trading volume
    lookback : int
        Lookback period for trend and volume analysis (default: 20)

    Returns
    -------
    np.ndarray
        Exhaustion score (0-1, higher = more exhausted)
    """
    n = len(close)
    exhaustion = np.zeros(n, dtype=np.float64)

    if n < lookback * 2:
        return exhaustion

    for i in range(lookback * 2, n):
        # Calculate price trend direction and strength
        price_start = close[i - lookback]
        price_end = close[i]
        price_change = (price_end - price_start) / price_start

        # Calculate volume trend
        vol_first_half = 0.0
        vol_second_half = 0.0
        half = lookback // 2

        for j in range(i - lookback, i - half):
            vol_first_half += volume[j]
        for j in range(i - half, i + 1):
            vol_second_half += volume[j]

        vol_first_half /= (lookback - half)
        vol_second_half /= (half + 1)

        # Detect exhaustion conditions
        # Uptrend with declining volume
        if price_change > 0.02:  # 2% uptrend
            if vol_second_half < vol_first_half * 0.8:  # Volume dropped 20%+
                vol_decline = 1.0 - (vol_second_half / vol_first_half)
                exhaustion[i] = min(1.0, vol_decline * 1.5)

        # Downtrend with declining volume
        elif price_change < -0.02:  # 2% downtrend
            if vol_second_half < vol_first_half * 0.8:
                vol_decline = 1.0 - (vol_second_half / vol_first_half)
                exhaustion[i] = min(1.0, vol_decline * 1.5)

    return exhaustion


@njit(cache=True)
def calculate_regime_derivatives(
    efficiency_ratio: np.ndarray,
    adx: np.ndarray,
    smooth: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate velocity and acceleration of regime indicators.

    Provides early warning of regime changes by measuring the rate of change
    and acceleration of efficiency ratio and ADX.

    Parameters
    ----------
    efficiency_ratio : np.ndarray
        Efficiency ratio values (0-1)
    adx : np.ndarray
        ADX values (0-100)
    smooth : int
        Smoothing period for derivatives (default: 5)

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (er_velocity, er_acceleration, adx_velocity, adx_acceleration)
        - Velocity: First derivative (rate of change per bar)
        - Acceleration: Second derivative (change in rate of change)
    """
    n = len(efficiency_ratio)
    er_velocity = np.zeros(n, dtype=np.float64)
    er_acceleration = np.zeros(n, dtype=np.float64)
    adx_velocity = np.zeros(n, dtype=np.float64)
    adx_acceleration = np.zeros(n, dtype=np.float64)

    if n < smooth * 3:
        return er_velocity, er_acceleration, adx_velocity, adx_acceleration

    # Calculate smoothed first derivatives (velocity)
    for i in range(smooth, n):
        # Skip if any values are NaN
        skip = False
        for j in range(smooth + 1):
            if np.isnan(efficiency_ratio[i - j]) or np.isnan(adx[i - j]):
                skip = True
                break
        if skip:
            continue

        # Calculate average rate of change over smooth period
        er_sum = 0.0
        adx_sum = 0.0
        for j in range(smooth):
            er_sum += efficiency_ratio[i - j] - efficiency_ratio[i - j - 1]
            adx_sum += adx[i - j] - adx[i - j - 1]

        er_velocity[i] = er_sum / smooth
        adx_velocity[i] = adx_sum / smooth

    # Calculate second derivatives (acceleration)
    for i in range(smooth * 2, n):
        # Skip if velocity values are zero (indicates NaN propagation)
        if er_velocity[i] == 0.0 and er_velocity[i - smooth] == 0.0:
            continue

        er_vel_sum = 0.0
        adx_vel_sum = 0.0
        for j in range(smooth):
            er_vel_sum += er_velocity[i - j] - er_velocity[i - j - 1]
            adx_vel_sum += adx_velocity[i - j] - adx_velocity[i - j - 1]

        er_acceleration[i] = er_vel_sum / smooth
        adx_acceleration[i] = adx_vel_sum / smooth

    return er_velocity, er_acceleration, adx_velocity, adx_acceleration


@njit(cache=True)
def calculate_all_anticipation_indicators(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    rsi: np.ndarray,
    efficiency_ratio: np.ndarray,
    adx: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Calculate all anticipation indicators in one pass.

    Parameters
    ----------
    close : np.ndarray
        Closing prices
    high : np.ndarray
        High prices
    low : np.ndarray
        Low prices
    volume : np.ndarray
        Trading volume
    rsi : np.ndarray
        RSI values
    efficiency_ratio : np.ndarray
        Efficiency ratio values
    adx : np.ndarray
        ADX values

    Returns
    -------
    dict containing all indicator arrays
    """
    # This is a convenience wrapper - in Numba we can't return dicts easily
    # So we return a tuple and the Python wrapper converts to dict
    squeeze_on, squeeze_intensity = calculate_squeeze(close, high, low)
    bullish_div, bearish_div = calculate_divergence(close, rsi)
    exhaustion = calculate_volume_exhaustion(close, volume)
    er_vel, er_acc, adx_vel, adx_acc = calculate_regime_derivatives(efficiency_ratio, adx)

    return (
        squeeze_on,
        squeeze_intensity,
        bullish_div,
        bearish_div,
        exhaustion,
        er_vel,
        er_acc,
        adx_vel,
        adx_acc,
    )


def get_all_anticipation_indicators(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    rsi: np.ndarray,
    efficiency_ratio: np.ndarray,
    adx: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Python wrapper to get all anticipation indicators as a dict.

    Parameters
    ----------
    close : np.ndarray
        Closing prices
    high : np.ndarray
        High prices
    low : np.ndarray
        Low prices
    volume : np.ndarray
        Trading volume
    rsi : np.ndarray
        RSI values
    efficiency_ratio : np.ndarray
        Efficiency ratio values
    adx : np.ndarray
        ADX values

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing all indicator arrays:
        - squeeze_on: Volatility squeeze active (0 or 1)
        - squeeze_intensity: Squeeze tightness (0-1)
        - bullish_divergence: Bullish RSI divergence score (0-1)
        - bearish_divergence: Bearish RSI divergence score (0-1)
        - volume_exhaustion: Trend exhaustion score (0-1)
        - er_velocity: Efficiency ratio rate of change
        - er_acceleration: Efficiency ratio acceleration
        - adx_velocity: ADX rate of change
        - adx_acceleration: ADX acceleration
    """
    squeeze_on, squeeze_intensity = calculate_squeeze(close, high, low)
    bullish_div, bearish_div = calculate_divergence(close, rsi)
    exhaustion = calculate_volume_exhaustion(close, volume)
    er_vel, er_acc, adx_vel, adx_acc = calculate_regime_derivatives(efficiency_ratio, adx)

    return {
        "squeeze_on": squeeze_on,
        "squeeze_intensity": squeeze_intensity,
        "bullish_divergence": bullish_div,
        "bearish_divergence": bearish_div,
        "volume_exhaustion": exhaustion,
        "er_velocity": er_vel,
        "er_acceleration": er_acc,
        "adx_velocity": adx_vel,
        "adx_acceleration": adx_acc,
    }
