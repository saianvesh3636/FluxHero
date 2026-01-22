"""
Regime Detection System for FluxHero Trading System.

This module implements market regime detection using ADX (Average Directional Index)
and linear regression to classify markets as trending or mean-reverting.

Features:
- ADX calculation for trend strength measurement (R5.1.1)
- Linear regression slope and R² for trend quality (R5.1.2)
- Regime classification: STRONG_TREND, MEAN_REVERSION, NEUTRAL (R5.1.3)
- Volatility regime detection: HIGH_VOL, LOW_VOL (R5.2.1)
- Regime persistence tracking with 3-bar confirmation (R5.2.3)
- Multi-asset correlation analysis (R5.3.1-3)

Performance: ADX calculation <100ms for 10k candles (Numba JIT)

Reference:
- FLUXHERO_REQUIREMENTS.md Feature 5: Regime Detection System
"""

import numpy as np
from numba import njit

# Regime constants
REGIME_MEAN_REVERSION = 0
REGIME_NEUTRAL = 1
REGIME_STRONG_TREND = 2

# Volatility regime constants
VOL_LOW = 0
VOL_NORMAL = 1
VOL_HIGH = 2


@njit(cache=True)
def calculate_directional_movement(
    high: np.ndarray,
    low: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate directional movement (+DM and -DM) for ADX calculation.

    Directional movement measures the portion of the day's range that is in the
    direction of the trend.

    Formula:
        +DM = High[today] - High[yesterday] if positive, else 0
        -DM = Low[yesterday] - Low[today] if positive, else 0

    Special rules:
        - If +DM > -DM and +DM > 0, then +DM = value, -DM = 0
        - If -DM > +DM and -DM > 0, then -DM = value, +DM = 0
        - If +DM = -DM, both are 0 (no clear direction)

    Args:
        high: Array of high prices (float64)
        low: Array of low prices (float64)

    Returns:
        Tuple of (+DM array, -DM array)
        First value is NaN (no previous bar to compare)

    Reference: R5.1.1 - Directional movement for ADX calculation

    Example:
        >>> high = np.array([100.0, 102.0, 101.0, 103.0])
        >>> low = np.array([98.0, 99.0, 98.5, 100.0])
        >>> plus_dm, minus_dm = calculate_directional_movement(high, low)
        >>> # Bar 1: +DM = 2.0, -DM = 0 (up move)
        >>> # Bar 2: +DM = 0, -DM = 0.5 (down move)
    """
    n = len(high)
    plus_dm = np.full(n, np.nan, dtype=np.float64)
    minus_dm = np.full(n, np.nan, dtype=np.float64)

    for i in range(1, n):
        high_diff = high[i] - high[i - 1]
        low_diff = low[i - 1] - low[i]

        # Determine which direction is stronger
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
            minus_dm[i] = 0.0
        elif low_diff > high_diff and low_diff > 0:
            plus_dm[i] = 0.0
            minus_dm[i] = low_diff
        else:
            plus_dm[i] = 0.0
            minus_dm[i] = 0.0

    return plus_dm, minus_dm


@njit(cache=True)
def calculate_directional_indicators(
    plus_dm: np.ndarray,
    minus_dm: np.ndarray,
    atr: np.ndarray,
    period: int = 14,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate +DI and -DI (Directional Indicators) using smoothed DM and ATR.

    Formula:
        +DI = 100 × (Smoothed +DM / ATR)
        -DI = 100 × (Smoothed -DM / ATR)

    Smoothing uses Wilder's method (same as ATR):
        Smoothed_DM[i] = (Smoothed_DM[i-1] × (period-1) + DM[i]) / period

    Args:
        plus_dm: Array of +DM values (float64)
        minus_dm: Array of -DM values (float64)
        atr: Array of ATR values (float64)
        period: Smoothing period (default: 14)

    Returns:
        Tuple of (+DI array, -DI array), values 0-100
        First 'period' values are NaN

    Reference: R5.1.1 - Directional indicators for ADX

    Example:
        >>> # +DM = [5, 3, 4], ATR = [10, 10, 10]
        >>> # Smoothed +DM = 4.0 → +DI = 100 × 4/10 = 40
    """
    n = len(plus_dm)
    plus_di = np.full(n, np.nan, dtype=np.float64)
    minus_di = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return plus_di, minus_di

    # Initialize first smoothed values as SMA
    smoothed_plus_dm = 0.0
    smoothed_minus_dm = 0.0

    # Find first valid index (skip NaN values)
    first_valid = period
    for i in range(1, n):
        if not np.isnan(plus_dm[i]):
            first_valid = i
            break

    if first_valid + period > n:
        return plus_di, minus_di

    # Calculate initial SMA for smoothing
    for i in range(first_valid, first_valid + period):
        if not np.isnan(plus_dm[i]):
            smoothed_plus_dm += plus_dm[i]
            smoothed_minus_dm += minus_dm[i]

    smoothed_plus_dm /= period
    smoothed_minus_dm /= period

    # Calculate first DI values
    if not np.isnan(atr[first_valid + period - 1]) and atr[first_valid + period - 1] > 0:
        plus_di[first_valid + period - 1] = 100.0 * smoothed_plus_dm / atr[first_valid + period - 1]
        minus_di[first_valid + period - 1] = 100.0 * smoothed_minus_dm / atr[first_valid + period - 1]

    # Calculate subsequent values using Wilder's smoothing
    for i in range(first_valid + period, n):
        if not np.isnan(plus_dm[i]) and not np.isnan(minus_dm[i]):
            smoothed_plus_dm = (smoothed_plus_dm * (period - 1) + plus_dm[i]) / period
            smoothed_minus_dm = (smoothed_minus_dm * (period - 1) + minus_dm[i]) / period

            if not np.isnan(atr[i]) and atr[i] > 0:
                plus_di[i] = 100.0 * smoothed_plus_dm / atr[i]
                minus_di[i] = 100.0 * smoothed_minus_dm / atr[i]

    return plus_di, minus_di


@njit(cache=True)
def calculate_adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """
    Calculate ADX (Average Directional Index) for trend strength measurement.

    ADX measures trend strength regardless of direction (always positive).

    Interpretation:
        - ADX > 25 → Strong trend (trending market)
        - ADX < 20 → Weak trend (ranging/choppy market)
        - ADX > 40 → Very strong trend
        - ADX < 15 → No trend

    Formula:
        1. Calculate +DM and -DM (directional movement)
        2. Calculate +DI and -DI (directional indicators)
        3. DX = 100 × |+DI - -DI| / (+DI + -DI)
        4. ADX = Smoothed average of DX over 'period' bars

    Args:
        high: Array of high prices (float64)
        low: Array of low prices (float64)
        close: Array of close prices (float64, unused but kept for API consistency)
        atr: Array of ATR values (float64, pre-calculated)
        period: ADX period (default: 14)

    Returns:
        Array of ADX values (float64), range 0-100
        First 2×period values are NaN (need data for DM smoothing + ADX smoothing)

    Reference: R5.1.1 - ADX >25 = trending, <20 = ranging

    Example:
        >>> # Strong uptrend: +DI >> -DI
        >>> # DX = |60 - 20| / (60 + 20) = 50
        >>> # ADX smooths DX over 14 bars → ~48
    """
    n = len(high)
    adx = np.full(n, np.nan, dtype=np.float64)

    if n < 2 * period:
        return adx

    # Step 1: Calculate directional movement
    plus_dm, minus_dm = calculate_directional_movement(high, low)

    # Step 2: Calculate directional indicators
    plus_di, minus_di = calculate_directional_indicators(plus_dm, minus_dm, atr, period)

    # Step 3: Calculate DX (Directional Index)
    dx = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(plus_di[i]) and not np.isnan(minus_di[i]):
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum

    # Step 4: Calculate ADX (smoothed DX)
    # Find first valid DX
    first_valid = -1
    for i in range(n):
        if not np.isnan(dx[i]):
            first_valid = i
            break

    if first_valid == -1 or first_valid + period >= n:
        return adx

    # Initialize ADX as SMA of first 'period' DX values
    adx_value = 0.0
    for i in range(first_valid, first_valid + period):
        adx_value += dx[i]
    adx_value /= period
    adx[first_valid + period - 1] = adx_value

    # Calculate subsequent ADX values using Wilder's smoothing
    for i in range(first_valid + period, n):
        if not np.isnan(dx[i]):
            adx_value = (adx_value * (period - 1) + dx[i]) / period
            adx[i] = adx_value

    return adx


@njit(cache=True)
def calculate_linear_regression(
    prices: np.ndarray,
    period: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate linear regression slope and R² over rolling window.

    Linear regression fits a straight line to price data:
        Price = slope × time + intercept

    Slope: Rate of price change per bar (positive = uptrend, negative = downtrend)
    R²: Quality of fit (0 = no fit, 1 = perfect fit)

    Interpretation:
        - R² > 0.7 → Strong linear trend (prices follow straight line)
        - R² < 0.3 → No clear trend (choppy/random movement)
        - Slope > 0 with high R² → Strong uptrend
        - Slope < 0 with high R² → Strong downtrend

    Formula (using least squares):
        slope = Σ((x - x̄)(y - ȳ)) / Σ((x - x̄)²)
        R² = 1 - (SS_res / SS_tot)

        Where:
        - x = time (0, 1, 2, ..., period-1)
        - y = prices
        - SS_res = Σ(y - y_pred)² (residual sum of squares)
        - SS_tot = Σ(y - ȳ)² (total sum of squares)

    Args:
        prices: Array of prices (float64)
        period: Lookback period for regression (default: 50)

    Returns:
        Tuple of (slope array, R² array)
        First 'period-1' values are NaN

    Reference: R5.1.2 - R² >0.7 = strong trend, <0.3 = no trend

    Example:
        >>> prices = np.array([100, 101, 102, 103, 104])  # Perfect uptrend
        >>> slope, r_squared = calculate_linear_regression(prices, period=5)
        >>> # slope ≈ 1.0 (rises $1 per bar)
        >>> # r_squared ≈ 1.0 (perfect fit)
    """
    n = len(prices)
    slopes = np.full(n, np.nan, dtype=np.float64)
    r_squared = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return slopes, r_squared

    # Pre-calculate x values (time: 0, 1, 2, ..., period-1)
    x = np.arange(period, dtype=np.float64)
    x_mean = np.mean(x)
    x_diff = x - x_mean
    x_diff_sq_sum = np.sum(x_diff ** 2)

    # Rolling window calculation
    for i in range(period - 1, n):
        window = prices[i - period + 1:i + 1]

        # Skip if any NaN in window
        if np.any(np.isnan(window)):
            continue

        y = window
        y_mean = np.mean(y)
        y_diff = y - y_mean

        # Calculate slope
        numerator = np.sum(x_diff * y_diff)
        slope = numerator / x_diff_sq_sum
        slopes[i] = slope

        # Calculate R²
        # Predicted values
        intercept = y_mean - slope * x_mean
        y_pred = slope * x + intercept

        # Sum of squares
        ss_res = np.sum((y - y_pred) ** 2)  # Residual
        ss_tot = np.sum(y_diff ** 2)  # Total

        # R² calculation (handle edge case where ss_tot = 0)
        if ss_tot > 0:
            r_squared[i] = 1.0 - (ss_res / ss_tot)
        else:
            # Constant prices → R² = 1.0 if slope = 0, else 0.0
            r_squared[i] = 1.0 if abs(slope) < 1e-10 else 0.0

    return slopes, r_squared


@njit(cache=True)
def classify_trend_regime(
    adx: np.ndarray,
    r_squared: np.ndarray,
    adx_trend_threshold: float = 25.0,
    adx_ranging_threshold: float = 20.0,
    r2_trend_threshold: float = 0.6,
    r2_ranging_threshold: float = 0.4,
) -> np.ndarray:
    """
    Classify market regime based on ADX and R² values.

    Combines trend strength (ADX) and trend quality (R²) to determine if
    market is trending or mean-reverting.

    Classification Logic:
        - STRONG_TREND (2): ADX > 25 AND R² > 0.6
        - MEAN_REVERSION (0): ADX < 20 AND R² < 0.4
        - NEUTRAL (1): Everything else (transition/unclear)

    Args:
        adx: Array of ADX values (float64)
        r_squared: Array of R² values (float64)
        adx_trend_threshold: ADX threshold for trending (default: 25.0)
        adx_ranging_threshold: ADX threshold for ranging (default: 20.0)
        r2_trend_threshold: R² threshold for trending (default: 0.6)
        r2_ranging_threshold: R² threshold for ranging (default: 0.4)

    Returns:
        Array of regime codes (int32):
        - 0 = MEAN_REVERSION (choppy, range-bound)
        - 1 = NEUTRAL (transition/unclear)
        - 2 = STRONG_TREND (strong directional move)

    Reference: R5.1.3 - Combine ADX + R² for regime classification

    Example:
        >>> adx = np.array([32.0, 18.0, 25.0])
        >>> r2 = np.array([0.75, 0.25, 0.5])
        >>> regime = classify_trend_regime(adx, r2)
        >>> # [STRONG_TREND, MEAN_REVERSION, NEUTRAL]
    """
    n = len(adx)
    regime = np.full(n, REGIME_NEUTRAL, dtype=np.int32)

    for i in range(n):
        if np.isnan(adx[i]) or np.isnan(r_squared[i]):
            continue

        # Strong trend: High ADX + High R²
        if adx[i] > adx_trend_threshold and r_squared[i] > r2_trend_threshold:
            regime[i] = REGIME_STRONG_TREND
        # Mean reversion: Low ADX + Low R²
        elif adx[i] < adx_ranging_threshold and r_squared[i] < r2_ranging_threshold:
            regime[i] = REGIME_MEAN_REVERSION
        # Everything else is neutral/transition
        else:
            regime[i] = REGIME_NEUTRAL

    return regime


@njit(cache=True)
def classify_volatility_regime(
    atr: np.ndarray,
    atr_ma: np.ndarray,
    high_vol_threshold: float = 1.5,
    low_vol_threshold: float = 0.7,
) -> np.ndarray:
    """
    Classify volatility regime based on ATR relative to its moving average.

    Compares current ATR to 50-period ATR average to determine if volatility
    is unusually high or low.

    Classification:
        - HIGH_VOL (2): ATR > 1.5 × ATR_MA
        - LOW_VOL (0): ATR < 0.7 × ATR_MA
        - NORMAL (1): Between thresholds

    Args:
        atr: Array of ATR values (float64)
        atr_ma: Array of ATR moving average values (float64, typically 50-period)
        high_vol_threshold: Multiplier for high volatility (default: 1.5)
        low_vol_threshold: Multiplier for low volatility (default: 0.7)

    Returns:
        Array of volatility regime codes (int32):
        - 0 = LOW_VOL (calm markets)
        - 1 = VOL_NORMAL (typical volatility)
        - 2 = HIGH_VOL (elevated volatility)

    Reference: R5.2.1 - ATR > 1.5×ATR_MA = HIGH_VOL, <0.7×ATR_MA = LOW_VOL

    Example:
        >>> atr = np.array([3.0, 1.0, 2.0])
        >>> atr_ma = np.array([2.0, 2.0, 2.0])
        >>> vol_regime = classify_volatility_regime(atr, atr_ma)
        >>> # [HIGH_VOL, LOW_VOL, NORMAL]
    """
    n = len(atr)
    vol_regime = np.full(n, VOL_NORMAL, dtype=np.int32)

    for i in range(n):
        if np.isnan(atr[i]) or np.isnan(atr_ma[i]) or atr_ma[i] == 0:
            continue

        ratio = atr[i] / atr_ma[i]

        if ratio > high_vol_threshold:
            vol_regime[i] = VOL_HIGH
        elif ratio < low_vol_threshold:
            vol_regime[i] = VOL_LOW
        else:
            vol_regime[i] = VOL_NORMAL

    return vol_regime


@njit(cache=True)
def apply_regime_persistence(
    regime: np.ndarray,
    confirmation_bars: int = 3,
) -> np.ndarray:
    """
    Apply regime persistence filter to prevent whipsaws.

    Requires 'confirmation_bars' consecutive bars in new regime before
    officially switching. This prevents rapid regime flip-flopping.

    Logic:
        - Track how long current regime has persisted
        - Only change regime if new regime persists for 'confirmation_bars'
        - Smooth out single-bar regime switches

    Args:
        regime: Array of raw regime codes (int32)
        confirmation_bars: Number of consecutive bars required to confirm (default: 3)

    Returns:
        Array of confirmed regime codes (int32)
        Same length as input, with smoothed regime transitions

    Reference: R5.2.3 - Require 3 consecutive bars to confirm regime change

    Example:
        >>> regime = np.array([2, 2, 0, 0, 0, 2, 2, 2, 2])
        >>> confirmed = apply_regime_persistence(regime, confirmation_bars=3)
        >>> # [2, 2, 2, 2, 2, 2, 2, 2, 2] - need 3 consecutive 0s to switch
        >>> # Actual: [2, 2, 2, 0, 0, 0, 0, 2, 2] - switches after 3 bars
    """
    n = len(regime)
    confirmed_regime = np.copy(regime)

    if n < confirmation_bars:
        return confirmed_regime

    # Start with first non-NaN regime
    current_regime = regime[0]
    consecutive_count = 1

    for i in range(1, n):
        # Check if regime is same as previous
        if regime[i] == regime[i - 1]:
            consecutive_count += 1
        else:
            consecutive_count = 1

        # Only switch if new regime persists for confirmation_bars
        if consecutive_count >= confirmation_bars:
            current_regime = regime[i]

        confirmed_regime[i] = current_regime

    return confirmed_regime


@njit(cache=True)
def calculate_correlation_matrix(
    returns_matrix: np.ndarray,
) -> np.ndarray:
    """
    Calculate correlation matrix for multiple assets.

    Correlation measures how assets move together:
        - Correlation = 1.0 → Perfect positive correlation (move together)
        - Correlation = 0.0 → No correlation (independent)
        - Correlation = -1.0 → Perfect negative correlation (move opposite)

    High correlation (>0.8) indicates markets moving together (risk-on/risk-off),
    reducing diversification benefits.

    Formula:
        Corr(X, Y) = Cov(X, Y) / (StdDev(X) × StdDev(Y))

    Args:
        returns_matrix: 2D array of returns (float64)
                       Shape: (n_bars, n_assets)
                       Each column is returns for one asset

    Returns:
        2D correlation matrix (float64)
        Shape: (n_assets, n_assets)
        Diagonal = 1.0 (asset correlated with itself)
        Symmetric matrix (corr(A,B) = corr(B,A))

    Reference: R5.3.1-3 - Correlation >0.8 = risk-on/risk-off

    Example:
        >>> # 3 assets, 5 bars of returns
        >>> returns = np.array([[0.01, 0.01, -0.01],
        ...                     [0.02, 0.02, -0.02],
        ...                     [-0.01, -0.01, 0.01]])
        >>> corr = calculate_correlation_matrix(returns)
        >>> # Asset 0 and 1: corr ≈ 1.0 (move together)
        >>> # Asset 0 and 2: corr ≈ -1.0 (move opposite)
    """
    n_bars, n_assets = returns_matrix.shape
    corr_matrix = np.eye(n_assets, dtype=np.float64)

    # Calculate mean returns for each asset
    means = np.zeros(n_assets, dtype=np.float64)
    for j in range(n_assets):
        valid_count = 0
        total = 0.0
        for i in range(n_bars):
            if not np.isnan(returns_matrix[i, j]):
                total += returns_matrix[i, j]
                valid_count += 1
        if valid_count > 0:
            means[j] = total / valid_count

    # Calculate correlation for each pair
    for j1 in range(n_assets):
        for j2 in range(j1 + 1, n_assets):
            # Calculate covariance
            cov = 0.0
            count = 0
            for i in range(n_bars):
                if not np.isnan(returns_matrix[i, j1]) and not np.isnan(returns_matrix[i, j2]):
                    cov += (returns_matrix[i, j1] - means[j1]) * (returns_matrix[i, j2] - means[j2])
                    count += 1

            if count > 1:
                cov /= (count - 1)  # Sample covariance

                # Calculate standard deviations
                std1 = 0.0
                std2 = 0.0
                count = 0
                for i in range(n_bars):
                    if not np.isnan(returns_matrix[i, j1]) and not np.isnan(returns_matrix[i, j2]):
                        std1 += (returns_matrix[i, j1] - means[j1]) ** 2
                        std2 += (returns_matrix[i, j2] - means[j2]) ** 2
                        count += 1

                if count > 1:
                    std1 = np.sqrt(std1 / (count - 1))
                    std2 = np.sqrt(std2 / (count - 1))

                    # Calculate correlation
                    if std1 > 0 and std2 > 0:
                        corr = cov / (std1 * std2)
                        corr_matrix[j1, j2] = corr
                        corr_matrix[j2, j1] = corr  # Symmetric

    return corr_matrix


def detect_regime(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    atr_ma: np.ndarray,
    adx_period: int = 14,
    regression_period: int = 50,
    apply_persistence: bool = True,
    confirmation_bars: int = 3,
) -> dict:
    """
    Complete regime detection pipeline combining all metrics.

    This is the main entry point for regime detection. It calculates:
    - ADX for trend strength
    - Linear regression for trend quality
    - Trend regime classification
    - Volatility regime classification
    - Regime persistence filtering

    Args:
        high: Array of high prices (float64)
        low: Array of low prices (float64)
        close: Array of close prices (float64)
        atr: Array of ATR values (float64, pre-calculated)
        atr_ma: Array of ATR moving average (float64, typically 50-period)
        adx_period: Period for ADX calculation (default: 14)
        regression_period: Period for linear regression (default: 50)
        apply_persistence: Whether to apply regime persistence filter (default: True)
        confirmation_bars: Bars required to confirm regime change (default: 3)

    Returns:
        Dictionary with:
        - 'adx': ADX values
        - 'r_squared': R² values
        - 'regression_slope': Linear regression slopes
        - 'trend_regime': Raw trend regime codes
        - 'trend_regime_confirmed': Persistence-filtered trend regime
        - 'volatility_regime': Volatility regime codes

    Example:
        >>> result = detect_regime(high, low, close, atr, atr_ma)
        >>> if result['trend_regime_confirmed'][-1] == REGIME_STRONG_TREND:
        ...     print("Strong trend detected, use trend-following strategy")
    """
    # Calculate ADX
    adx = calculate_adx(high, low, close, atr, period=adx_period)

    # Calculate linear regression
    regression_slope, r_squared = calculate_linear_regression(close, period=regression_period)

    # Classify trend regime
    trend_regime = classify_trend_regime(adx, r_squared)

    # Apply persistence filter if requested
    if apply_persistence:
        trend_regime_confirmed = apply_regime_persistence(trend_regime, confirmation_bars)
    else:
        trend_regime_confirmed = trend_regime

    # Classify volatility regime
    volatility_regime = classify_volatility_regime(atr, atr_ma)

    return {
        'adx': adx,
        'r_squared': r_squared,
        'regression_slope': regression_slope,
        'trend_regime': trend_regime,
        'trend_regime_confirmed': trend_regime_confirmed,
        'volatility_regime': volatility_regime,
    }
