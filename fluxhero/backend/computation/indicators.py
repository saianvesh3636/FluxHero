"""
JIT-Compiled Technical Indicators using Numba

This module provides high-performance implementations of core technical indicators
using Numba's @njit (No-Python JIT) compilation for near-C++ speeds.

All functions are decorated with @njit(cache=True) for optimal performance:
- EMA: Exponential Moving Average with configurable alpha
- RSI: Relative Strength Index for overbought/oversold detection
- ATR: Average True Range for volatility measurement

Performance targets (10,000 candles):
- EMA: <100ms
- Full indicator suite: <500ms
"""

import numpy as np
from numba import njit


@njit(cache=True)
def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average using Numba JIT compilation.

    Formula:
        α (alpha) = 2 / (period + 1)
        EMA[today] = (Price[today] × α) + (EMA[yesterday] × (1 - α))

    Args:
        prices: 1D array of closing prices (float64)
        period: Number of periods for EMA calculation (e.g., 12, 26, 50)

    Returns:
        1D array of EMA values (same length as prices, initial values are NaN)

    Example:
        >>> prices = np.array([10.0, 11.0, 12.0, 11.0, 13.0, 14.0])
        >>> ema = calculate_ema(prices, period=5)
    """
    n = len(prices)
    ema = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return ema

    # Calculate alpha (smoothing factor)
    alpha = 2.0 / (period + 1.0)

    # Initialize EMA with SMA of first 'period' values
    sma = 0.0
    for i in range(period):
        sma += prices[i]
    ema[period - 1] = sma / period

    # Calculate EMA for remaining values
    for i in range(period, n):
        ema[i] = (prices[i] * alpha) + (ema[i - 1] * (1.0 - alpha))

    return ema


@njit(cache=True)
def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index using Numba JIT compilation.

    Formula:
        RS = Average Gain / Average Loss
        RSI = 100 - (100 / (1 + RS))

    Args:
        prices: 1D array of closing prices (float64)
        period: Number of periods for RSI calculation (default: 14)

    Returns:
        1D array of RSI values (range: 0-100, initial values are NaN)
        - RSI > 70: Overbought condition
        - RSI < 30: Oversold condition

    Example:
        >>> prices = np.array([44.0, 44.34, 44.09, 43.61, 44.33, 44.83])
        >>> rsi = calculate_rsi(prices, period=14)
    """
    n = len(prices)
    rsi = np.full(n, np.nan, dtype=np.float64)

    if n < period + 1:
        return rsi

    # Calculate price changes
    deltas = np.diff(prices)

    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Calculate initial average gain and loss (SMA for first period)
    avg_gain = 0.0
    avg_loss = 0.0
    for i in range(period):
        avg_gain += gains[i]
        avg_loss += losses[i]
    avg_gain /= period
    avg_loss /= period

    # Calculate RSI for first valid point
    if avg_loss == 0.0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    # Calculate RSI using smoothed moving average (Wilder's smoothing)
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period

        if avg_loss == 0.0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


@njit(cache=True)
def calculate_true_range(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """
    Calculate True Range for volatility measurement.

    True Range captures complete price movement including gaps:
        TR = MAX of:
            1. High - Low (current range)
            2. |High - Previous Close| (gap up)
            3. |Low - Previous Close| (gap down)

    Args:
        high: 1D array of high prices (float64)
        low: 1D array of low prices (float64)
        close: 1D array of closing prices (float64)

    Returns:
        1D array of True Range values (first value is NaN)

    Example:
        >>> high = np.array([45.0, 46.0, 47.0])
        >>> low = np.array([44.0, 45.0, 46.0])
        >>> close = np.array([44.5, 45.5, 46.5])
        >>> tr = calculate_true_range(high, low, close)
    """
    n = len(high)
    tr = np.full(n, np.nan, dtype=np.float64)

    if n < 2:
        return tr

    # First TR is just high - low (no previous close)
    tr[0] = high[0] - low[0]

    # Calculate TR for remaining bars
    for i in range(1, n):
        method1 = high[i] - low[i]
        method2 = abs(high[i] - close[i - 1])
        method3 = abs(low[i] - close[i - 1])
        tr[i] = max(method1, method2, method3)

    return tr


@njit(cache=True)
def calculate_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14
) -> np.ndarray:
    """
    Calculate Average True Range using Numba JIT compilation.

    ATR is a smoothed moving average of True Range, measuring volatility.
    Uses Wilder's smoothing (similar to EMA but with period-specific alpha).

    Formula:
        ATR[today] = ((ATR[yesterday] × (period - 1)) + TR[today]) / period

    Args:
        high: 1D array of high prices (float64)
        low: 1D array of low prices (float64)
        close: 1D array of closing prices (float64)
        period: Number of periods for ATR calculation (default: 14)

    Returns:
        1D array of ATR values (initial values are NaN)

    Common uses:
        - Stop-loss placement: Stop = Entry - (ATR × 2)
        - Position sizing: Shares = Risk $ / ATR
        - Volatility regime detection: Compare ATR vs ATR_MA(50)

    Example:
        >>> high = np.array([45.0, 46.0, 47.0, 48.0])
        >>> low = np.array([44.0, 45.0, 46.0, 47.0])
        >>> close = np.array([44.5, 45.5, 46.5, 47.5])
        >>> atr = calculate_atr(high, low, close, period=14)
    """
    n = len(high)
    atr = np.full(n, np.nan, dtype=np.float64)

    if n < period + 1:
        return atr

    # Calculate True Range first
    tr = calculate_true_range(high, low, close)

    # Calculate initial ATR as SMA of first 'period' TR values
    initial_atr = 0.0
    for i in range(1, period + 1):
        initial_atr += tr[i]
    atr[period] = initial_atr / period

    # Calculate ATR using Wilder's smoothing
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


@njit(cache=True)
def calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Simple Moving Average using Numba JIT compilation.

    SMA treats all prices equally (unlike EMA which weights recent prices more).

    Formula:
        SMA = Sum(Prices over period) / period

    Args:
        prices: 1D array of closing prices (float64)
        period: Number of periods for SMA calculation

    Returns:
        1D array of SMA values (initial values are NaN)

    Example:
        >>> prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        >>> sma = calculate_sma(prices, period=3)
    """
    n = len(prices)
    sma = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return sma

    # Calculate SMA for each valid window
    for i in range(period - 1, n):
        window_sum = 0.0
        for j in range(i - period + 1, i + 1):
            window_sum += prices[j]
        sma[i] = window_sum / period

    return sma


@njit(cache=True)
def calculate_bollinger_bands(
    prices: np.ndarray,
    period: int = 20,
    num_std: float = 2.0
) -> tuple:
    """
    Calculate Bollinger Bands using Numba JIT compilation.

    Bollinger Bands measure volatility and provide dynamic support/resistance.

    Formula:
        Middle Band = SMA(period)
        Upper Band = Middle Band + (num_std × StdDev)
        Lower Band = Middle Band - (num_std × StdDev)

    Args:
        prices: 1D array of closing prices (float64)
        period: Number of periods for moving average (default: 20)
        num_std: Number of standard deviations (default: 2.0)

    Returns:
        Tuple of (upper_band, middle_band, lower_band) arrays

    Example:
        >>> prices = np.array([10.0, 11.0, 12.0, 11.5, 13.0, 14.0])
        >>> upper, middle, lower = calculate_bollinger_bands(prices, period=5)
    """
    n = len(prices)
    middle_band = np.full(n, np.nan, dtype=np.float64)
    upper_band = np.full(n, np.nan, dtype=np.float64)
    lower_band = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return upper_band, middle_band, lower_band

    # Calculate for each valid window
    for i in range(period - 1, n):
        # Calculate mean
        window_sum = 0.0
        for j in range(i - period + 1, i + 1):
            window_sum += prices[j]
        mean = window_sum / period
        middle_band[i] = mean

        # Calculate standard deviation
        variance = 0.0
        for j in range(i - period + 1, i + 1):
            diff = prices[j] - mean
            variance += diff * diff
        std_dev = np.sqrt(variance / period)

        # Calculate bands
        upper_band[i] = mean + (num_std * std_dev)
        lower_band[i] = mean - (num_std * std_dev)

    return upper_band, middle_band, lower_band
