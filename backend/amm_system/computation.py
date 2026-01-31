"""
Numba-compiled indicator calculations for the AMM strategy.

All functions use @njit(cache=True) for performance optimization.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate Simple Moving Average."""
    n = len(prices)
    sma = np.empty(n, dtype=np.float64)
    sma[:] = np.nan

    for i in range(period - 1, n):
        total = 0.0
        for j in range(period):
            total += prices[i - j]
        sma[i] = total / period

    return sma


@njit(cache=True)
def calculate_sma_deviation(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate price deviation from SMA as a percentage.

    Returns (price - SMA) / SMA, scaled to represent relative distance.
    Positive values = price above SMA, negative = price below SMA.
    """
    n = len(prices)
    deviation = np.empty(n, dtype=np.float64)
    deviation[:] = np.nan

    sma = calculate_sma(prices, period)

    for i in range(period - 1, n):
        if sma[i] > 0:
            deviation[i] = (prices[i] - sma[i]) / sma[i]

    return deviation


@njit(cache=True)
def calculate_rsi_normalized(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate RSI normalized to [-1, 1] range.

    Standard RSI is 0-100. This rescales: (RSI - 50) / 50
    So RSI=70 becomes +0.4, RSI=30 becomes -0.4.
    """
    n = len(prices)
    rsi_norm = np.empty(n, dtype=np.float64)
    rsi_norm[:] = np.nan

    if n < period + 1:
        return rsi_norm

    # Calculate price changes
    changes = np.empty(n, dtype=np.float64)
    changes[0] = 0.0
    for i in range(1, n):
        changes[i] = prices[i] - prices[i - 1]

    # Initialize averages with SMA of first 'period' gains/losses
    avg_gain = 0.0
    avg_loss = 0.0
    for i in range(1, period + 1):
        if changes[i] > 0:
            avg_gain += changes[i]
        else:
            avg_loss += abs(changes[i])
    avg_gain /= period
    avg_loss /= period

    # First RSI value
    if avg_loss == 0:
        rsi_norm[period] = 1.0  # RSI = 100
    else:
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi_norm[period] = (rsi - 50.0) / 50.0

    # Calculate subsequent RSI values using Wilder's smoothing
    for i in range(period + 1, n):
        change = changes[i]
        if change > 0:
            gain = change
            loss = 0.0
        else:
            gain = 0.0
            loss = abs(change)

        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0:
            rsi_norm[i] = 1.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
            rsi_norm[i] = (rsi - 50.0) / 50.0

    return rsi_norm


@njit(cache=True)
def calculate_momentum(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate N-day momentum as percentage change.

    Returns (price - price_n_days_ago) / price_n_days_ago
    """
    n = len(prices)
    momentum = np.empty(n, dtype=np.float64)
    momentum[:] = np.nan

    for i in range(period, n):
        if prices[i - period] > 0:
            momentum[i] = (prices[i] - prices[i - period]) / prices[i - period]

    return momentum


@njit(cache=True)
def calculate_bollinger_pct_b(prices: np.ndarray, period: int = 20, num_std: float = 2.0) -> np.ndarray:
    """
    Calculate Bollinger %B - a mean-reversion indicator.

    %B = (Price - Lower Band) / (Upper Band - Lower Band)

    Interpretation:
    - %B > 1.0: Price above upper band (overbought)
    - %B < 0.0: Price below lower band (oversold)
    - %B = 0.5: Price at middle band

    Normalized to [-1, 1] range: (pct_b - 0.5) * 2
    So: overbought -> +1, oversold -> -1, middle -> 0
    """
    n = len(prices)
    pct_b_norm = np.empty(n, dtype=np.float64)
    pct_b_norm[:] = np.nan

    for i in range(period - 1, n):
        # Calculate SMA (middle band)
        total = 0.0
        for j in range(period):
            total += prices[i - j]
        sma = total / period

        # Calculate standard deviation
        variance = 0.0
        for j in range(period):
            diff = prices[i - j] - sma
            variance += diff * diff
        std = np.sqrt(variance / period)

        if std > 1e-10:
            upper = sma + num_std * std
            lower = sma - num_std * std
            band_width = upper - lower

            if band_width > 1e-10:
                pct_b = (prices[i] - lower) / band_width
                # Normalize to [-1, 1]: 0->-1, 0.5->0, 1->+1
                pct_b_norm[i] = (pct_b - 0.5) * 2.0
                # Clip to [-1, 1] range
                if pct_b_norm[i] > 1.0:
                    pct_b_norm[i] = 1.0
                elif pct_b_norm[i] < -1.0:
                    pct_b_norm[i] = -1.0

    return pct_b_norm


@njit(cache=True)
def calculate_zscore_rolling(values: np.ndarray, lookback: int) -> np.ndarray:
    """
    Calculate rolling z-score normalization.

    For each point: (value - rolling_mean) / rolling_std
    """
    n = len(values)
    zscore = np.empty(n, dtype=np.float64)
    zscore[:] = np.nan

    for i in range(lookback - 1, n):
        # Skip if any value in the window is NaN
        has_nan = False
        for j in range(lookback):
            if np.isnan(values[i - j]):
                has_nan = True
                break

        if has_nan:
            continue

        # Calculate mean
        total = 0.0
        for j in range(lookback):
            total += values[i - j]
        mean = total / lookback

        # Calculate std
        variance = 0.0
        for j in range(lookback):
            diff = values[i - j] - mean
            variance += diff * diff
        std = np.sqrt(variance / lookback)

        if std > 1e-10:
            zscore[i] = (values[i] - mean) / std
        else:
            zscore[i] = 0.0

    return zscore


@njit(cache=True)
def calculate_ema_smooth(values: np.ndarray, span: int) -> np.ndarray:
    """
    Apply EMA smoothing to a series.

    Alpha = 2 / (span + 1)
    """
    n = len(values)
    ema = np.empty(n, dtype=np.float64)
    ema[:] = np.nan

    alpha = 2.0 / (span + 1)

    # Find first non-NaN value
    first_valid = -1
    for i in range(n):
        if not np.isnan(values[i]):
            first_valid = i
            break

    if first_valid == -1:
        return ema

    ema[first_valid] = values[first_valid]

    for i in range(first_valid + 1, n):
        if np.isnan(values[i]):
            ema[i] = ema[i - 1]
        else:
            ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]

    return ema


@njit(cache=True)
def combine_indicators_weighted(
    sma_dev: np.ndarray,
    rsi_norm: np.ndarray,
    momentum: np.ndarray,
    boll_pct_b: np.ndarray,
    w_sma: float,
    w_rsi: float,
    w_mom: float,
    w_boll: float,
) -> np.ndarray:
    """
    Combine all indicators with weights.

    Indicators:
    - sma_dev: Trend indicator (price vs SMA)
    - rsi_norm: Momentum/overbought-oversold
    - momentum: Pure trend/momentum
    - boll_pct_b: Mean-reversion indicator (Bollinger %B)

    Note: boll_pct_b is INVERTED for signal combination because:
    - High %B (overbought) should reduce bullish signal
    - Low %B (oversold) should reduce bearish signal

    Returns weighted sum, handling NaN values gracefully.
    """
    n = len(sma_dev)
    combined = np.empty(n, dtype=np.float64)
    combined[:] = np.nan

    for i in range(n):
        # Skip if any indicator is NaN
        if (
            np.isnan(sma_dev[i])
            or np.isnan(rsi_norm[i])
            or np.isnan(momentum[i])
            or np.isnan(boll_pct_b[i])
        ):
            continue

        # Invert Bollinger %B for counter-trend signal
        # When price is overbought (+1), this subtracts from bullish signals
        # When price is oversold (-1), this adds to bullish signals
        boll_counter = -boll_pct_b[i]

        combined[i] = (
            w_sma * sma_dev[i]
            + w_rsi * rsi_norm[i]
            + w_mom * momentum[i]
            + w_boll * boll_counter
        )

    return combined


def compute_amm_indicators(
    bars: np.ndarray,
    sma_period: int = 50,
    rsi_period: int = 14,
    mom_period: int = 20,
    boll_period: int = 20,
    w_sma: float = 0.25,
    w_rsi: float = 0.25,
    w_mom: float = 0.25,
    w_boll: float = 0.25,
    zscore_lookback: int = 50,
    ema_span: int = 10,
) -> dict:
    """
    Main entry point: compute all AMM indicators.

    The AMM combines trend and mean-reversion indicators:
    - SMA Deviation: Trend (price position relative to moving average)
    - RSI: Momentum/overbought-oversold
    - Momentum: Pure trend
    - Bollinger %B: Mean-reversion (provides counter-trend balance)

    Parameters
    ----------
    bars : np.ndarray
        OHLCV data with shape (n_bars, 5). Column order: Open, High, Low, Close, Volume
    sma_period : int
        Period for SMA deviation (default: 50, ~2 months)
    rsi_period : int
        Period for RSI calculation (default: 14, standard)
    mom_period : int
        Period for momentum calculation (default: 20, ~1 month)
    boll_period : int
        Period for Bollinger %B calculation (default: 20, standard)
    w_sma, w_rsi, w_mom, w_boll : float
        Weights for each indicator (should sum to 1.0)
        Default: equal weights (0.25 each)
    zscore_lookback : int
        Lookback period for z-score normalization (default: 50)
    ema_span : int
        Span for EMA smoothing (default: 10, responsive)

    Returns
    -------
    dict
        Dictionary containing all computed indicators:
        - 'sma_deviation': Price deviation from SMA (trend)
        - 'rsi_normalized': RSI scaled to [-1, 1] (momentum)
        - 'momentum': N-day momentum (trend)
        - 'bollinger_pct_b': Bollinger %B scaled to [-1, 1] (mean-reversion)
        - 'combined_raw': Weighted combination of indicators
        - 'combined_zscore': Z-score normalized combined signal
        - 'signal': Final EMA-smoothed signal
    """
    # Extract close prices (column 3)
    close = bars[:, 3].astype(np.float64)

    # Calculate individual indicators
    sma_dev = calculate_sma_deviation(close, sma_period)
    rsi_norm = calculate_rsi_normalized(close, rsi_period)
    momentum = calculate_momentum(close, mom_period)
    boll_pct_b = calculate_bollinger_pct_b(close, boll_period)

    # Combine with weights
    combined_raw = combine_indicators_weighted(
        sma_dev, rsi_norm, momentum, boll_pct_b, w_sma, w_rsi, w_mom, w_boll
    )

    # Apply z-score normalization
    combined_zscore = calculate_zscore_rolling(combined_raw, zscore_lookback)

    # Apply EMA smoothing for final signal
    signal = calculate_ema_smooth(combined_zscore, ema_span)

    return {
        "sma_deviation": sma_dev,
        "rsi_normalized": rsi_norm,
        "momentum": momentum,
        "bollinger_pct_b": boll_pct_b,
        "combined_raw": combined_raw,
        "combined_zscore": combined_zscore,
        "signal": signal,
    }
