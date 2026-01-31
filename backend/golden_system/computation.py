"""
Golden Adaptive Indicator - Full Four-Dimensional System

The Complex Three-Tier System (Layer 2):
Combines 4 orthogonal dimensions into a unified adaptive alpha with confidence scoring.

Dimensions:
1. Fractal (Market Microstructure) - Trending vs Mean-reverting
2. Efficiency Ratio - Directional efficiency of price movement
3. Volatility - Activity level (ATR-based)
4. Volume - Confirmation of moves

Key Innovation: Confidence scoring based on dimension agreement.
When all 4 dimensions agree → High confidence (trade aggressively)
When dimensions disagree → Low confidence (reduce size or skip)

This module is SELF-CONTAINED - no dependencies on the simple golden_ema system.
Can be removed entirely without affecting other code.

Usage:
    from backend.golden_system import compute_golden_adaptive_indicators

    indicators = compute_golden_adaptive_indicators(bars)
    alpha = indicators['alpha']
    confidence = indicators['confidence']
    regime = indicators['regime']
"""

import numpy as np
from numba import njit


# =============================================================================
# Core Calculations (Self-Contained)
# =============================================================================

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


# =============================================================================
# Dimension 1: Fractal Alpha (Market Microstructure)
# =============================================================================

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
            w = np.log(alpha_slow / alpha_fast)
            alpha[i] = alpha_fast * np.exp(w * (fractal_stat - 1))
        else:
            alpha[i] = (alpha_fast + alpha_slow) / 2

    return alpha


# =============================================================================
# Dimension 2: Efficiency Ratio Alpha
# =============================================================================

@njit(cache=True)
def calculate_efficiency_ratio_alpha(
    close: np.ndarray,
    er_period: int,
    alpha_slow: float,
    alpha_fast: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate adaptive alpha based on Efficiency Ratio.

    ER = |Price Change| / Sum of |Daily Changes|
        0 = Pure noise (price went nowhere despite movement)
        1 = Perfect efficiency (price moved in one direction)

    Alpha Mapping:
        High ER → Fast alpha (efficient trend, follow it)
        Low ER → Slow alpha (noise, smooth it out)
    """
    n = len(close)
    alpha = np.full(n, np.nan)
    er = np.full(n, np.nan)

    for i in range(er_period, n):
        # Direction (net change)
        direction = abs(close[i] - close[i - er_period])

        # Volatility (sum of absolute changes)
        volatility = 0.0
        for j in range(i - er_period + 1, i + 1):
            volatility += abs(close[j] - close[j - 1])

        # Efficiency Ratio
        if volatility > 1e-10:
            er[i] = direction / volatility
        else:
            er[i] = 0.0

        # Map to alpha using squared ER (KAMA approach)
        sc = er[i] * er[i]
        alpha[i] = alpha_slow + sc * (alpha_fast - alpha_slow)

    return alpha, er


# =============================================================================
# Dimension 3: Volatility Alpha
# =============================================================================

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
            normalized = (atr_pct - 0.005) / (0.03 - 0.005)
            normalized = max(0.0, min(1.0, normalized))

            alpha[i] = alpha_slow + normalized * (alpha_fast - alpha_slow)
        else:
            alpha[i] = (alpha_fast + alpha_slow) / 2

    return alpha


# =============================================================================
# Dimension 4: Volume Alpha
# =============================================================================

@njit(cache=True)
def calculate_volume_alpha(
    volume: np.ndarray,
    vol_period: int,
    alpha_slow: float,
    alpha_fast: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate adaptive alpha based on relative volume.

    Logic:
        High volume → More conviction in price moves → Faster alpha
        Low volume → Less conviction → Slower alpha
    """
    n = len(volume)
    alpha = np.full(n, np.nan)
    rel_vol = np.full(n, np.nan)

    for i in range(vol_period, n):
        # Average volume
        avg_vol = 0.0
        for j in range(i - vol_period, i):
            avg_vol += volume[j]
        avg_vol /= vol_period

        # Relative volume (current vs average)
        if avg_vol > 1e-10:
            rel_vol[i] = volume[i] / avg_vol
        else:
            rel_vol[i] = 1.0

        # Map to alpha
        rv_clamped = max(0.5, min(2.0, rel_vol[i]))
        rv_normalized = (rv_clamped - 0.5) / 1.5

        alpha[i] = alpha_slow + rv_normalized * (alpha_fast - alpha_slow)

    return alpha, rel_vol


# =============================================================================
# Combined Four-Dimension Alpha
# =============================================================================

@njit(cache=True)
def calculate_four_dimension_alpha(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    fractal_lookback: int = 20,
    vol_lookback: int = 14,
    er_period: int = 10,
    volume_period: int = 20,
    slow_period: int = 30,
    fast_period: int = 2,
    w_fractal: float = 0.30,
    w_efficiency: float = 0.30,
    w_volatility: float = 0.25,
    w_volume: float = 0.15
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Golden Alpha combining 4 orthogonal dimensions.

    Returns
    -------
    tuple of 8 arrays:
        alpha_combined, confidence, regime,
        alpha_fractal, alpha_vol, alpha_er, alpha_volume, er_values
    """
    n = len(close)

    # Alpha bounds
    alpha_fast = 2.0 / (fast_period + 1)
    alpha_slow = 2.0 / (slow_period + 1)

    # Calculate each dimension's alpha
    alpha_fractal = calculate_fractal_alpha(
        high, low, fractal_lookback, alpha_slow, alpha_fast
    )
    alpha_vol = calculate_volatility_alpha(
        high, low, close, vol_lookback, alpha_slow, alpha_fast
    )
    alpha_er, er_values = calculate_efficiency_ratio_alpha(
        close, er_period, alpha_slow, alpha_fast
    )
    alpha_volume, rel_vol = calculate_volume_alpha(
        volume, volume_period, alpha_slow, alpha_fast
    )

    # Output arrays
    alpha_combined = np.full(n, np.nan)
    confidence = np.full(n, np.nan)
    regime = np.full(n, 1.0)  # Default neutral

    # Start index
    start_idx = max(2 * fractal_lookback, vol_lookback + 1, er_period, volume_period)

    for i in range(start_idx, n):
        if (np.isnan(alpha_fractal[i]) or np.isnan(alpha_vol[i]) or
            np.isnan(alpha_er[i]) or np.isnan(alpha_volume[i])):
            continue

        # Weighted average
        alpha_combined[i] = (
            w_fractal * alpha_fractal[i] +
            w_efficiency * alpha_er[i] +
            w_volatility * alpha_vol[i] +
            w_volume * alpha_volume[i]
        )
        alpha_combined[i] = max(alpha_slow, min(alpha_fast, alpha_combined[i]))

        # Confidence score (dimension agreement)
        alphas = np.array([alpha_fractal[i], alpha_vol[i], alpha_er[i], alpha_volume[i]])
        alpha_std = np.std(alphas)
        alpha_range = alpha_fast - alpha_slow
        confidence[i] = max(0.0, 1.0 - (alpha_std / (alpha_range / 2)))

        # Regime classification
        fractal_normalized = (alpha_fractal[i] - alpha_slow) / alpha_range
        er_val = er_values[i] if not np.isnan(er_values[i]) else 0.5

        if fractal_normalized > 0.6 and er_val > 0.5:
            regime[i] = 2.0  # Strong trend
        elif fractal_normalized < 0.3 and er_val < 0.3:
            regime[i] = 0.0  # Mean-reversion
        else:
            regime[i] = 1.0  # Neutral

    return (alpha_combined, confidence, regime,
            alpha_fractal, alpha_vol, alpha_er, alpha_volume, er_values)


@njit(cache=True)
def calculate_golden_ema_from_alpha(
    close: np.ndarray,
    alpha: np.ndarray
) -> np.ndarray:
    """Calculate EMA using pre-computed adaptive alpha."""
    n = len(close)
    ema = np.full(n, np.nan)

    start_idx = 0
    for i in range(n):
        if not np.isnan(alpha[i]):
            start_idx = i
            ema[i] = close[i]
            break

    for i in range(start_idx + 1, n):
        if np.isnan(alpha[i]):
            ema[i] = ema[i-1]
        else:
            ema[i] = alpha[i] * close[i] + (1 - alpha[i]) * ema[i-1]

    return ema


# =============================================================================
# Main Entry Point
# =============================================================================

def compute_golden_adaptive_indicators(
    bars: np.ndarray,
    fractal_lookback: int = 20,
    vol_lookback: int = 14,
    er_period: int = 10,
    volume_period: int = 20,
    slow_period: int = 30,
    fast_period: int = 2,
    w_fractal: float = 0.30,
    w_efficiency: float = 0.30,
    w_volatility: float = 0.25,
    w_volume: float = 0.15
) -> dict:
    """
    Compute all Golden Adaptive indicators (Layer 2 of Three-Tier System).

    Parameters
    ----------
    bars : np.ndarray
        OHLCV data with shape (N, 5): [open, high, low, close, volume]

    Returns
    -------
    dict
        {
            'golden_ema': np.ndarray - The adaptive moving average
            'golden_ema_fast': np.ndarray - Faster version for crossovers
            'golden_ema_slow': np.ndarray - Slower version for crossovers
            'alpha': np.ndarray - Combined adaptive alpha
            'confidence': np.ndarray - Dimension agreement score (0-1)
            'regime': np.ndarray - Detected regime (0=MR, 1=neutral, 2=trend)
            'alpha_fractal': np.ndarray - Fractal dimension alpha
            'alpha_volatility': np.ndarray - Volatility alpha
            'alpha_efficiency': np.ndarray - Efficiency ratio alpha
            'alpha_volume': np.ndarray - Volume alpha
            'efficiency_ratio': np.ndarray - Raw ER values
        }
    """
    high = bars[:, 1]
    low = bars[:, 2]
    close = bars[:, 3]
    volume = bars[:, 4]

    alpha_fast = 2.0 / (fast_period + 1)
    alpha_slow = 2.0 / (slow_period + 1)

    (alpha, confidence, regime,
     alpha_fractal, alpha_vol, alpha_er, alpha_volume, er_values) = calculate_four_dimension_alpha(
        high, low, close, volume,
        fractal_lookback, vol_lookback, er_period, volume_period,
        slow_period, fast_period,
        w_fractal, w_efficiency, w_volatility, w_volume
    )

    # Golden EMA
    golden_ema = calculate_golden_ema_from_alpha(close, alpha)

    # Fast and slow for crossovers
    fast_alpha = np.clip(alpha * 1.5, alpha_slow, alpha_fast)
    slow_alpha_arr = np.clip(alpha * 0.6, alpha_slow, alpha_fast)
    golden_ema_fast = calculate_golden_ema_from_alpha(close, fast_alpha)
    golden_ema_slow = calculate_golden_ema_from_alpha(close, slow_alpha_arr)

    return {
        'golden_ema': golden_ema,
        'golden_ema_fast': golden_ema_fast,
        'golden_ema_slow': golden_ema_slow,
        'alpha': alpha,
        'confidence': confidence,
        'regime': regime,
        'alpha_fractal': alpha_fractal,
        'alpha_volatility': alpha_vol,
        'alpha_efficiency': alpha_er,
        'alpha_volume': alpha_volume,
        'efficiency_ratio': er_values,
    }


def generate_golden_signals(
    close: np.ndarray,
    golden_ema: np.ndarray,
    golden_ema_fast: np.ndarray,
    golden_ema_slow: np.ndarray,
    confidence: np.ndarray,
    regime: np.ndarray,
    lookback: int = 50,  # Rolling window for extremes (same as other indicator lookbacks)
) -> np.ndarray:
    """
    Generate trading signals from Golden Adaptive indicators.

    TRUE NO MAGIC NUMBERS:
    - Crossovers: Signal when lines cross (no threshold)
    - Mean-reversion: Signal at rolling min/max (extreme of recent distribution)
    - Confidence: Scales signal strength (no minimum threshold)

    Signal Logic:
    - Trending regime: EMA crossovers and price-EMA crossovers
    - Mean-reversion regime: Price at rolling high/low relative to EMA
    - Neutral regime: EMA crossovers only

    Parameters
    ----------
    lookback : int
        Rolling window for calculating extremes (uses same window as indicators)

    Returns
    -------
    np.ndarray
        Signal array: positive = long strength, negative = short strength
        Signal magnitude = confidence (dimension agreement)
    """
    n = len(close)
    signals = np.zeros(n)

    # Pre-calculate rolling deviation statistics for MR signals
    # No magic threshold - use rolling min/max
    deviations = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(golden_ema[i]) and golden_ema[i] > 0:
            deviations[i] = (close[i] - golden_ema[i]) / golden_ema[i]

    for i in range(lookback, n):
        # Skip if indicators not ready
        if (np.isnan(golden_ema[i]) or np.isnan(golden_ema_fast[i]) or
            np.isnan(golden_ema_slow[i]) or np.isnan(confidence[i])):
            continue

        if np.isnan(golden_ema_fast[i-1]) or np.isnan(golden_ema_slow[i-1]):
            continue

        current_regime = regime[i]
        conf = confidence[i]

        if current_regime == 2.0:  # Trending - follow crossovers
            # Price crossing EMA
            if close[i] > golden_ema[i] and close[i-1] <= golden_ema[i-1]:
                signals[i] = conf  # Bullish
            elif close[i] < golden_ema[i] and close[i-1] >= golden_ema[i-1]:
                signals[i] = -conf  # Bearish
            # EMA crossovers
            elif golden_ema_fast[i] > golden_ema_slow[i] and golden_ema_fast[i-1] <= golden_ema_slow[i-1]:
                signals[i] = conf
            elif golden_ema_fast[i] < golden_ema_slow[i] and golden_ema_fast[i-1] >= golden_ema_slow[i-1]:
                signals[i] = -conf

        elif current_regime == 0.0:  # Mean-reversion - fade extremes
            # Get rolling window of deviations
            window = deviations[i-lookback:i]
            valid_window = window[~np.isnan(window)]

            if len(valid_window) > 0:
                rolling_min = np.min(valid_window)
                rolling_max = np.max(valid_window)
                current_dev = deviations[i]

                if not np.isnan(current_dev):
                    # Signal at extremes: current deviation equals rolling min or max
                    if current_dev <= rolling_min:
                        signals[i] = conf  # Oversold - go long
                    elif current_dev >= rolling_max:
                        signals[i] = -conf  # Overbought - go short

        else:  # Neutral (1.0) - crossovers only
            fast_above_now = golden_ema_fast[i] > golden_ema_slow[i]
            fast_above_prev = golden_ema_fast[i-1] > golden_ema_slow[i-1]

            if fast_above_now and not fast_above_prev:
                signals[i] = conf
            elif not fast_above_now and fast_above_prev:
                signals[i] = -conf

    return signals


# =============================================================================
# Analysis
# =============================================================================

def analyze_dimension_contribution(indicators: dict) -> dict:
    """Analyze how each dimension contributes to the final alpha."""
    alpha = indicators['alpha']
    valid_mask = ~np.isnan(alpha)

    def stats(arr, mask):
        valid = arr[mask & ~np.isnan(arr)]
        if len(valid) == 0:
            return {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan}
        return {
            'mean': float(np.mean(valid)),
            'std': float(np.std(valid)),
            'min': float(np.min(valid)),
            'max': float(np.max(valid)),
        }

    return {
        'combined_alpha': stats(alpha, valid_mask),
        'fractal_alpha': stats(indicators['alpha_fractal'], valid_mask),
        'volatility_alpha': stats(indicators['alpha_volatility'], valid_mask),
        'efficiency_alpha': stats(indicators['alpha_efficiency'], valid_mask),
        'volume_alpha': stats(indicators['alpha_volume'], valid_mask),
        'confidence': stats(indicators['confidence'], valid_mask),
        'regime_distribution': {
            'mean_reversion': float(np.sum(indicators['regime'][valid_mask] == 0) / np.sum(valid_mask)),
            'neutral': float(np.sum(indicators['regime'][valid_mask] == 1) / np.sum(valid_mask)),
            'trending': float(np.sum(indicators['regime'][valid_mask] == 2) / np.sum(valid_mask)),
        }
    }


def print_dimension_analysis(analysis: dict) -> None:
    """Pretty print the dimension analysis."""
    print("\n" + "=" * 70)
    print("  GOLDEN ADAPTIVE - 4 DIMENSION ANALYSIS")
    print("=" * 70)

    print("\n[Alpha Statistics]")
    print(f"  {'Dimension':<20} {'Mean':<10} {'Std':<10} {'Range':<20}")
    print("-" * 60)

    for key, label in [
        ('combined_alpha', 'Combined'),
        ('fractal_alpha', 'Fractal'),
        ('volatility_alpha', 'Volatility'),
        ('efficiency_alpha', 'Efficiency'),
        ('volume_alpha', 'Volume'),
    ]:
        s = analysis[key]
        print(f"  {label:<20} {s['mean']:<10.4f} {s['std']:<10.4f} "
              f"{s['min']:.4f} - {s['max']:.4f}")

    print("\n[Confidence]")
    s = analysis['confidence']
    print(f"  Mean: {s['mean']:.3f}, Std: {s['std']:.3f}")

    print("\n[Regime Distribution]")
    r = analysis['regime_distribution']
    print(f"  Mean-Reversion: {r['mean_reversion']*100:.1f}%")
    print(f"  Neutral:        {r['neutral']*100:.1f}%")
    print(f"  Trending:       {r['trending']*100:.1f}%")
    print("=" * 70 + "\n")
