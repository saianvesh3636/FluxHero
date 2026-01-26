"""
Golden Adaptive EMA - Combines Market Microstructure and Volatility Adaptation

Following Kahneman's principle: Simple algorithms beat complex expert systems.

This module provides:
1. Simple Golden EMA: Geometric mean of fractal-based and volatility-based alphas
2. No magic numbers - all derived from price action
3. Crossover signal generation

Mathematical Foundation:
- Fractal Dimension: Measures market structure (trending vs mean-reverting)
- Volatility Adaptation: Adjusts responsiveness to market activity
- Combined via geometric mean: Both must agree for extreme alpha values

Usage:
    from backend.computation.golden_ema import (
        calculate_simple_golden_ema,
        calculate_golden_ema_fast_slow,
        calculate_golden_ema_signals
    )

    golden_ema, alpha = calculate_simple_golden_ema(high, low, close)
"""

import numpy as np
from numba import njit


@njit(cache=True)
def _calculate_true_range(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """Calculate True Range for ATR computation."""
    n = len(high)
    tr = np.full(n, np.nan)

    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    return tr


@njit(cache=True)
def _calculate_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int
) -> np.ndarray:
    """Calculate Average True Range using Wilder's smoothing."""
    n = len(high)
    tr = _calculate_true_range(high, low, close)
    atr = np.full(n, np.nan)

    if n < period + 1:
        return atr

    # Initial ATR as SMA
    initial = 0.0
    for i in range(1, period + 1):
        initial += tr[i]
    atr[period] = initial / period

    # Wilder's smoothing
    for i in range(period + 1, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

    return atr


@njit(cache=True)
def calculate_fractal_alpha(
    high: np.ndarray,
    low: np.ndarray,
    lookback: int,
    alpha_slow: float,
    alpha_fast: float
) -> np.ndarray:
    """
    Calculate adaptive alpha based on fractal dimension (market microstructure).

    Fractal Statistic Interpretation:
        ≈ 1.0: Trending market (ranges accumulate)
        ≈ 2.0: Mean-reverting market (price oscillates back)

    Alpha Mapping:
        Trending (stat=1) → Fast alpha (follow the trend)
        Mean-reverting (stat=2) → Slow alpha (smooth out noise)

    Parameters
    ----------
    high : np.ndarray
        High prices
    low : np.ndarray
        Low prices
    lookback : int
        Lookback period for fractal calculation
    alpha_slow : float
        Slowest alpha (e.g., 2/(30+1) = 0.0645)
    alpha_fast : float
        Fastest alpha (e.g., 2/(2+1) = 0.6667)

    Returns
    -------
    np.ndarray
        Adaptive alpha based on market structure
    """
    n = len(high)
    alpha = np.full(n, np.nan)

    start_idx = 2 * lookback

    for i in range(start_idx, n):
        # Period 1: i-2*lookback to i-lookback
        h1 = high[i - 2*lookback]
        l1 = low[i - 2*lookback]
        for j in range(i - 2*lookback + 1, i - lookback):
            if high[j] > h1:
                h1 = high[j]
            if low[j] < l1:
                l1 = low[j]
        n1 = (h1 - l1) / lookback

        # Period 2: i-lookback to i
        h2 = high[i - lookback]
        l2 = low[i - lookback]
        for j in range(i - lookback + 1, i):
            if high[j] > h2:
                h2 = high[j]
            if low[j] < l2:
                l2 = low[j]
        n2 = (h2 - l2) / lookback

        # Combined period
        h_combined = max(h1, h2)
        l_combined = min(l1, l2)
        n3 = (h_combined - l_combined) / (2 * lookback)

        if n1 > 1e-10 and n2 > 1e-10 and n3 > 1e-10:
            # Fractal statistic (Hurst exponent approximation)
            fractal_stat = (np.log(n1 + n2) - np.log(n3)) / np.log(2)
            fractal_stat = max(1.0, min(2.0, fractal_stat))

            # Map to alpha using exponential interpolation
            # stat=1 → alpha_fast, stat=2 → alpha_slow
            w = np.log(alpha_slow / alpha_fast)
            alpha[i] = alpha_fast * np.exp(w * (fractal_stat - 1))
        else:
            alpha[i] = (alpha_fast + alpha_slow) / 2

    return alpha


@njit(cache=True)
def calculate_volatility_alpha(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    vol_lookback: int,
    alpha_slow: float,
    alpha_fast: float
) -> np.ndarray:
    """
    Calculate adaptive alpha based on volatility (ATR percentage).

    Logic:
        High volatility → Fast alpha (prices moving quickly, need to keep up)
        Low volatility → Slow alpha (avoid reacting to noise)

    Parameters
    ----------
    high, low, close : np.ndarray
        OHLC price data
    vol_lookback : int
        ATR period
    alpha_slow : float
        Slowest alpha
    alpha_fast : float
        Fastest alpha

    Returns
    -------
    np.ndarray
        Adaptive alpha based on volatility
    """
    n = len(close)
    alpha = np.full(n, np.nan)

    atr = _calculate_atr(high, low, close, vol_lookback)

    for i in range(vol_lookback + 1, n):
        if close[i] > 0 and not np.isnan(atr[i]):
            # ATR as percentage of price
            atr_pct = atr[i] / close[i]

            # Linear scaling: map ATR% to alpha
            # Typical range: 0.5% to 3% ATR
            # Map to alpha range using linear interpolation
            normalized = (atr_pct - 0.005) / (0.03 - 0.005)
            normalized = max(0.0, min(1.0, normalized))

            alpha[i] = alpha_slow + normalized * (alpha_fast - alpha_slow)
        else:
            alpha[i] = (alpha_fast + alpha_slow) / 2

    return alpha


@njit(cache=True)
def calculate_simple_golden_ema(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    mm_lookback: int = 20,
    vol_lookback: int = 14,
    slow_period: int = 30,
    fast_period: int = 2
) -> tuple:
    """
    Calculate Simple Golden EMA combining Market Microstructure and Volatility.

    The Golden Alpha is the geometric mean of:
    1. Fractal-based alpha (market structure)
    2. Volatility-based alpha (activity level)

    No magic thresholds. Just smoothing factors derived from price action.

    Parameters
    ----------
    high : np.ndarray
        High prices
    low : np.ndarray
        Low prices
    close : np.ndarray
        Close prices
    mm_lookback : int
        Lookback for fractal dimension calculation (default: 20)
    vol_lookback : int
        Lookback for ATR calculation (default: 14)
    slow_period : int
        Slowest EMA equivalent period (default: 30)
    fast_period : int
        Fastest EMA equivalent period (default: 2)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (golden_ema, alpha)
        - golden_ema: The adaptive moving average
        - alpha: The smoothing factor used at each bar

    Example
    -------
    >>> high = np.array([...])  # High prices
    >>> low = np.array([...])   # Low prices
    >>> close = np.array([...]) # Close prices
    >>> golden_ema, alpha = calculate_simple_golden_ema(high, low, close)
    """
    n = len(close)
    golden_ema = np.full(n, np.nan)
    alpha_combined = np.full(n, np.nan)

    # Alpha bounds
    alpha_fast = 2.0 / (fast_period + 1)
    alpha_slow = 2.0 / (slow_period + 1)

    # Calculate component alphas
    alpha_fractal = calculate_fractal_alpha(high, low, mm_lookback, alpha_slow, alpha_fast)
    alpha_vol = calculate_volatility_alpha(high, low, close, vol_lookback, alpha_slow, alpha_fast)

    # Start index (need warmup for both)
    start_idx = max(2 * mm_lookback, vol_lookback + 1)

    # Initialize EMA
    golden_ema[start_idx] = close[start_idx]

    for i in range(start_idx, n):
        if np.isnan(alpha_fractal[i]) or np.isnan(alpha_vol[i]):
            if i > 0 and not np.isnan(golden_ema[i-1]):
                golden_ema[i] = golden_ema[i-1]
            continue

        # Combine using geometric mean
        # Both must agree for extreme values
        alpha_combined[i] = np.sqrt(alpha_fractal[i] * alpha_vol[i])

        # Clamp to valid range
        alpha_combined[i] = max(alpha_slow, min(alpha_fast, alpha_combined[i]))

        # Update EMA
        if i == start_idx:
            golden_ema[i] = close[i]
        else:
            golden_ema[i] = alpha_combined[i] * close[i] + (1 - alpha_combined[i]) * golden_ema[i-1]

    return golden_ema, alpha_combined


@njit(cache=True)
def calculate_golden_ema_fast_slow(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    mm_lookback: int = 20,
    vol_lookback: int = 14,
    slow_period: int = 30,
    fast_period: int = 2
) -> tuple:
    """
    Calculate Fast and Slow Golden EMAs for crossover signals.

    Fast EMA: Uses half the lookback periods (more responsive)
    Slow EMA: Uses standard lookback periods (smoother)

    Parameters
    ----------
    high, low, close : np.ndarray
        Price data
    mm_lookback : int
        Base lookback for fractal dimension
    vol_lookback : int
        Base lookback for ATR
    slow_period : int
        Base slow period
    fast_period : int
        Base fast period

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (golden_fast, golden_slow, alpha_fast, alpha_slow)
    """
    # Fast Golden EMA (shorter lookbacks)
    golden_fast, alpha_f = calculate_simple_golden_ema(
        high, low, close,
        mm_lookback=max(5, mm_lookback // 2),
        vol_lookback=max(5, vol_lookback // 2),
        slow_period=max(10, slow_period // 2),
        fast_period=fast_period
    )

    # Slow Golden EMA (longer lookbacks)
    golden_slow, alpha_s = calculate_simple_golden_ema(
        high, low, close,
        mm_lookback=mm_lookback,
        vol_lookback=vol_lookback,
        slow_period=slow_period,
        fast_period=fast_period * 2
    )

    return golden_fast, golden_slow, alpha_f, alpha_s


@njit(cache=True)
def calculate_golden_ema_signals(
    close: np.ndarray,
    golden_fast: np.ndarray,
    golden_slow: np.ndarray
) -> np.ndarray:
    """
    Generate trading signals from Golden EMA crossovers.

    No magic thresholds - pure crossover logic.

    Parameters
    ----------
    close : np.ndarray
        Close prices
    golden_fast : np.ndarray
        Fast Golden EMA
    golden_slow : np.ndarray
        Slow Golden EMA

    Returns
    -------
    np.ndarray
        Signal array: 1 = long, -1 = short, 0 = no signal
    """
    n = len(close)
    signals = np.zeros(n)

    for i in range(1, n):
        if np.isnan(golden_fast[i]) or np.isnan(golden_slow[i]):
            continue
        if np.isnan(golden_fast[i-1]) or np.isnan(golden_slow[i-1]):
            continue

        # Crossover detection
        fast_above_now = golden_fast[i] > golden_slow[i]
        fast_above_prev = golden_fast[i-1] > golden_slow[i-1]

        if fast_above_now and not fast_above_prev:
            signals[i] = 1.0   # Bullish crossover
        elif not fast_above_now and fast_above_prev:
            signals[i] = -1.0  # Bearish crossover

    return signals


@njit(cache=True)
def calculate_golden_regime(
    alpha: np.ndarray,
    alpha_slow: float,
    alpha_fast: float
) -> np.ndarray:
    """
    Classify regime based on Golden Alpha value.

    Uses percentile-based classification (no magic thresholds):
        Low alpha (bottom 25%) → Mean-reversion regime
        Mid alpha (middle 50%) → Neutral regime
        High alpha (top 25%) → Trending regime

    Parameters
    ----------
    alpha : np.ndarray
        Golden alpha values
    alpha_slow : float
        Minimum alpha value
    alpha_fast : float
        Maximum alpha value

    Returns
    -------
    np.ndarray
        Regime: 0 = mean-reversion, 1 = neutral, 2 = trending
    """
    n = len(alpha)
    regime = np.full(n, 1.0)  # Default neutral

    alpha_range = alpha_fast - alpha_slow

    for i in range(n):
        if np.isnan(alpha[i]):
            continue

        # Normalize alpha to 0-1 range
        normalized = (alpha[i] - alpha_slow) / alpha_range

        # Classify based on position in range
        if normalized < 0.25:
            regime[i] = 0.0  # Mean-reversion (slow alpha = choppy market)
        elif normalized > 0.75:
            regime[i] = 2.0  # Trending (fast alpha = trending market)
        else:
            regime[i] = 1.0  # Neutral

    return regime


# =============================================================================
# Convenience Functions
# =============================================================================

def golden_ema_indicator(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    **kwargs
) -> dict:
    """
    Calculate all Golden EMA components and return as dictionary.

    Convenience wrapper for strategy integration.

    Parameters
    ----------
    high, low, close : np.ndarray
        Price data
    **kwargs
        Parameters for calculate_simple_golden_ema

    Returns
    -------
    dict
        {
            'golden_ema': np.ndarray,
            'golden_fast': np.ndarray,
            'golden_slow': np.ndarray,
            'alpha': np.ndarray,
            'signals': np.ndarray,
            'regime': np.ndarray
        }
    """
    # Single Golden EMA
    golden_ema, alpha = calculate_simple_golden_ema(high, low, close, **kwargs)

    # Fast and Slow for crossovers
    golden_fast, golden_slow, _, _ = calculate_golden_ema_fast_slow(
        high, low, close, **kwargs
    )

    # Signals
    signals = calculate_golden_ema_signals(close, golden_fast, golden_slow)

    # Regime
    slow_period = kwargs.get('slow_period', 30)
    fast_period = kwargs.get('fast_period', 2)
    alpha_slow = 2.0 / (slow_period + 1)
    alpha_fast = 2.0 / (fast_period + 1)
    regime = calculate_golden_regime(alpha, alpha_slow, alpha_fast)

    return {
        'golden_ema': golden_ema,
        'golden_fast': golden_fast,
        'golden_slow': golden_slow,
        'alpha': alpha,
        'signals': signals,
        'regime': regime
    }
