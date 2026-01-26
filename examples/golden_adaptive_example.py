"""
Golden Adaptive Indicator Example: SPY vs AFRM Comparison

This script demonstrates why fixed "magic number" thresholds fail and how
adaptive, percentile-based thresholds solve the problem.

Key Concepts Demonstrated:
1. RSI distributions differ dramatically between assets (SPY vs AFRM)
2. Fixed thresholds (30/70) may never trigger on some assets
3. Percentile-based thresholds adapt to each asset's characteristics
4. Golden Adaptive EMA combines multiple dimensions for robust signals

Run: python -m examples.golden_adaptive_example
"""

import asyncio
import sys
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.data import get_provider
from backend.computation.indicators import (
    calculate_rsi,
    calculate_atr,
    calculate_sma,
    calculate_ema,
    calculate_bollinger_bands,
)
from backend.computation.adaptive_ema import (
    calculate_efficiency_ratio,
    calculate_kama_with_regime_adjustment,
)
from backend.computation.volatility import calculate_atr_ma


# ============================================================================
# PART 1: Data Classes for Calibrated Parameters
# ============================================================================

@dataclass
class AssetStatistics:
    """Statistics for a single asset's indicator distributions."""
    symbol: str
    data_period: str
    n_bars: int

    # RSI statistics
    rsi_mean: float
    rsi_std: float
    rsi_min: float
    rsi_max: float
    rsi_p10: float  # 10th percentile (dynamic oversold)
    rsi_p25: float
    rsi_p50: float  # Median
    rsi_p75: float
    rsi_p90: float  # 90th percentile (dynamic overbought)
    rsi_pct_below_30: float  # % of time RSI < 30 (fixed threshold)
    rsi_pct_above_70: float  # % of time RSI > 70 (fixed threshold)

    # Efficiency Ratio statistics
    er_mean: float
    er_std: float
    er_p10: float
    er_p50: float
    er_p90: float
    er_pct_above_06: float  # % of time ER > 0.6 (fixed "trending" threshold)
    er_pct_below_03: float  # % of time ER < 0.3 (fixed "choppy" threshold)

    # ATR/Volatility statistics
    atr_mean: float
    atr_std: float
    atr_ratio_mean: float  # ATR / ATR_MA
    atr_ratio_p20: float   # Dynamic low volatility threshold
    atr_ratio_p80: float   # Dynamic high volatility threshold

    # ADX statistics (if available)
    adx_mean: float = 0.0
    adx_p25: float = 0.0
    adx_p75: float = 0.0


@dataclass
class CalibratedThresholds:
    """Calibrated thresholds for an asset based on its own distribution."""
    symbol: str
    calibration_date: str
    lookback_days: int

    # RSI thresholds (percentile-based)
    rsi_oversold: float      # 10th percentile
    rsi_overbought: float    # 90th percentile
    rsi_neutral_low: float   # 25th percentile
    rsi_neutral_high: float  # 75th percentile

    # Efficiency Ratio thresholds
    er_trending: float       # 75th percentile
    er_choppy: float         # 25th percentile

    # Volatility thresholds
    vol_low: float           # 20th percentile of ATR ratio
    vol_high: float          # 80th percentile of ATR ratio


# ============================================================================
# PART 2: Fractal Dimension Calculation (from MM_adaptive concept)
# ============================================================================

def calculate_fractal_dimension(
    high: np.ndarray,
    low: np.ndarray,
    lookback: int = 20
) -> np.ndarray:
    """
    Calculate Fractal Dimension using the efficient method.

    This measures market microstructure:
    - Value near 1.0 = Trending (price moves in one direction)
    - Value near 2.0 = Mean-reverting (price oscillates)

    The formula compares price ranges over two consecutive periods:
    - If trending: combined range ≈ sum of individual ranges
    - If mean-reverting: combined range < sum (price retraces)
    """
    n = len(high)
    fractal_dim = np.full(n, np.nan)

    for i in range(2 * lookback, n):
        # Period 1: i-2*lookback to i-lookback
        h1 = np.max(high[i - 2*lookback : i - lookback])
        l1 = np.min(low[i - 2*lookback : i - lookback])
        n1 = (h1 - l1) / lookback  # Average range per bar

        # Period 2: i-lookback to i
        h2 = np.max(high[i - lookback : i])
        l2 = np.min(low[i - lookback : i])
        n2 = (h2 - l2) / lookback

        # Combined period
        h = max(h1, h2)
        l = min(l1, l2)
        n3 = (h - l) / (2 * lookback)

        if n1 > 0 and n2 > 0 and n3 > 0:
            # Fractal dimension formula
            fractal_dim[i] = (np.log(n1 + n2) - np.log(n3)) / np.log(2)

    return fractal_dim


# ============================================================================
# PART 3: Golden Adaptive Alpha Calculation
# ============================================================================

def calculate_golden_alpha(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    # Pre-computed indicators
    fractal_dim: np.ndarray,
    efficiency_ratio: np.ndarray,
    atr: np.ndarray,
    atr_ma: np.ndarray,
    # Calibrated parameters
    fast_alpha: float = 0.6667,   # 2/(2+1)
    slow_alpha: float = 0.0645,   # 2/(30+1)
    # Weights for each dimension
    w_fractal: float = 0.35,
    w_efficiency: float = 0.30,
    w_volatility: float = 0.25,
    w_volume: float = 0.10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Golden Adaptive Alpha by combining 4 orthogonal dimensions:

    1. Fractal Dimension: Market structure (trending vs mean-reverting)
    2. Efficiency Ratio: Movement quality (directional vs noise)
    3. Volatility: Market activity level
    4. Volume: Participation/conviction

    Returns:
        alpha: Adaptive smoothing factor for EMA
        confidence: Agreement score between dimensions (0-1)
        regime: Detected regime (0=MR, 1=neutral, 2=trend)
    """
    n = len(close)
    alpha = np.full(n, np.nan)
    confidence = np.full(n, np.nan)
    regime = np.full(n, 1.0)  # Default neutral

    # Pre-calculate average volume
    avg_volume = calculate_sma(volume.astype(np.float64), period=20)

    for i in range(50, n):  # Need warmup for all indicators
        if np.isnan(fractal_dim[i]) or np.isnan(efficiency_ratio[i]):
            continue
        if np.isnan(atr[i]) or np.isnan(atr_ma[i]) or atr_ma[i] == 0:
            continue

        # ===== DIMENSION 1: Fractal Structure =====
        # Map fractal_dim (1-2) to alpha (fast-slow)
        fd = np.clip(fractal_dim[i], 1.0, 2.0)
        # fd=1 (trend) → alpha_fractal = fast_alpha
        # fd=2 (MR) → alpha_fractal = slow_alpha
        alpha_fractal = fast_alpha * np.exp(np.log(slow_alpha/fast_alpha) * (fd - 1.0))

        # ===== DIMENSION 2: Efficiency Ratio =====
        # High ER → faster response
        er = np.clip(efficiency_ratio[i], 0.0, 1.0)
        alpha_efficiency = slow_alpha + (fast_alpha - slow_alpha) * (er ** 2)

        # ===== DIMENSION 3: Volatility =====
        vol_ratio = atr[i] / atr_ma[i]
        vol_ratio_clamped = np.clip(vol_ratio, 0.5, 2.0)
        vol_score = (vol_ratio_clamped - 0.5) / 1.5  # Normalize to 0-1
        alpha_volatility = slow_alpha + (fast_alpha - slow_alpha) * vol_score

        # ===== DIMENSION 4: Volume =====
        if avg_volume[i] > 0:
            vol_rel = volume[i] / avg_volume[i]
            vol_rel_clamped = np.clip(vol_rel, 0.5, 2.0)
            vol_score_v = (vol_rel_clamped - 0.5) / 1.5
            alpha_volume = slow_alpha + (fast_alpha - slow_alpha) * vol_score_v
        else:
            alpha_volume = (fast_alpha + slow_alpha) / 2

        # ===== COMBINE: Weighted Average =====
        alpha[i] = (
            w_fractal * alpha_fractal +
            w_efficiency * alpha_efficiency +
            w_volatility * alpha_volatility +
            w_volume * alpha_volume
        )

        # Clamp to valid range
        alpha[i] = np.clip(alpha[i], slow_alpha, fast_alpha)

        # ===== CONFIDENCE SCORE =====
        alphas = np.array([alpha_fractal, alpha_efficiency, alpha_volatility, alpha_volume])
        alpha_std = np.std(alphas)
        alpha_range = fast_alpha - slow_alpha
        # Low std = high agreement = high confidence
        confidence[i] = 1.0 - min(1.0, alpha_std / (alpha_range / 2))

        # ===== REGIME CLASSIFICATION =====
        if fd < 1.4 and er > 0.5:
            regime[i] = 2.0  # Strong trend
        elif fd > 1.6 and er < 0.35:
            regime[i] = 0.0  # Mean-reversion
        else:
            regime[i] = 1.0  # Neutral

    return alpha, confidence, regime


def calculate_golden_ema(close: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Calculate EMA using adaptive alpha from Golden Alpha calculation."""
    n = len(close)
    golden_ema = np.full(n, np.nan)

    # Find first valid alpha
    start_idx = 0
    for i in range(n):
        if not np.isnan(alpha[i]):
            golden_ema[i] = close[i]
            start_idx = i
            break

    for i in range(start_idx + 1, n):
        if np.isnan(alpha[i]):
            golden_ema[i] = golden_ema[i-1]
        else:
            golden_ema[i] = alpha[i] * close[i] + (1 - alpha[i]) * golden_ema[i-1]

    return golden_ema


# ============================================================================
# PART 4: Analysis Functions
# ============================================================================

def calculate_asset_statistics(
    symbol: str,
    bars: np.ndarray,
    dates: list,
    period_desc: str = "1 Year"
) -> AssetStatistics:
    """Calculate comprehensive statistics for an asset."""

    high = bars[:, 1]
    low = bars[:, 2]
    close = bars[:, 3]
    volume = bars[:, 4]

    # Calculate indicators
    rsi = calculate_rsi(close, period=14)
    er = calculate_efficiency_ratio(close, period=10)
    atr = calculate_atr(high, low, close, period=14)
    atr_ma = calculate_atr_ma(atr, period=50)

    # Filter out NaN values
    rsi_valid = rsi[~np.isnan(rsi)]
    er_valid = er[~np.isnan(er)]
    atr_valid = atr[~np.isnan(atr)]

    # ATR ratio
    atr_ratio = np.full_like(atr, np.nan)
    for i in range(len(atr)):
        if not np.isnan(atr[i]) and not np.isnan(atr_ma[i]) and atr_ma[i] > 0:
            atr_ratio[i] = atr[i] / atr_ma[i]
    atr_ratio_valid = atr_ratio[~np.isnan(atr_ratio)]

    return AssetStatistics(
        symbol=symbol,
        data_period=period_desc,
        n_bars=len(bars),

        # RSI
        rsi_mean=np.mean(rsi_valid),
        rsi_std=np.std(rsi_valid),
        rsi_min=np.min(rsi_valid),
        rsi_max=np.max(rsi_valid),
        rsi_p10=np.percentile(rsi_valid, 10),
        rsi_p25=np.percentile(rsi_valid, 25),
        rsi_p50=np.percentile(rsi_valid, 50),
        rsi_p75=np.percentile(rsi_valid, 75),
        rsi_p90=np.percentile(rsi_valid, 90),
        rsi_pct_below_30=100 * np.sum(rsi_valid < 30) / len(rsi_valid),
        rsi_pct_above_70=100 * np.sum(rsi_valid > 70) / len(rsi_valid),

        # Efficiency Ratio
        er_mean=np.mean(er_valid),
        er_std=np.std(er_valid),
        er_p10=np.percentile(er_valid, 10),
        er_p50=np.percentile(er_valid, 50),
        er_p90=np.percentile(er_valid, 90),
        er_pct_above_06=100 * np.sum(er_valid > 0.6) / len(er_valid),
        er_pct_below_03=100 * np.sum(er_valid < 0.3) / len(er_valid),

        # ATR
        atr_mean=np.mean(atr_valid),
        atr_std=np.std(atr_valid),
        atr_ratio_mean=np.mean(atr_ratio_valid) if len(atr_ratio_valid) > 0 else 1.0,
        atr_ratio_p20=np.percentile(atr_ratio_valid, 20) if len(atr_ratio_valid) > 0 else 0.8,
        atr_ratio_p80=np.percentile(atr_ratio_valid, 80) if len(atr_ratio_valid) > 0 else 1.2,
    )


def calibrate_thresholds(stats: AssetStatistics) -> CalibratedThresholds:
    """Generate calibrated thresholds from asset statistics."""
    return CalibratedThresholds(
        symbol=stats.symbol,
        calibration_date=datetime.now().strftime("%Y-%m-%d"),
        lookback_days=stats.n_bars,

        rsi_oversold=stats.rsi_p10,
        rsi_overbought=stats.rsi_p90,
        rsi_neutral_low=stats.rsi_p25,
        rsi_neutral_high=stats.rsi_p75,

        er_trending=stats.er_p90,
        er_choppy=stats.er_p10,

        vol_low=stats.atr_ratio_p20,
        vol_high=stats.atr_ratio_p80,
    )


def print_comparison_table(spy_stats: AssetStatistics, afrm_stats: AssetStatistics):
    """Print a comparison table showing why fixed thresholds fail."""

    print("\n" + "="*80)
    print("           INDICATOR DISTRIBUTION COMPARISON: SPY vs AFRM")
    print("="*80)
    print(f"\n{'Metric':<35} {'SPY':>15} {'AFRM':>15} {'Difference':>12}")
    print("-"*80)

    # RSI Section
    print("\n[RSI - Relative Strength Index]")
    print(f"  Mean                             {spy_stats.rsi_mean:>15.2f} {afrm_stats.rsi_mean:>15.2f} {afrm_stats.rsi_mean - spy_stats.rsi_mean:>+12.2f}")
    print(f"  Std Dev                          {spy_stats.rsi_std:>15.2f} {afrm_stats.rsi_std:>15.2f} {afrm_stats.rsi_std - spy_stats.rsi_std:>+12.2f}")
    print(f"  Min                              {spy_stats.rsi_min:>15.2f} {afrm_stats.rsi_min:>15.2f}")
    print(f"  Max                              {spy_stats.rsi_max:>15.2f} {afrm_stats.rsi_max:>15.2f}")
    print(f"  10th Percentile (oversold)       {spy_stats.rsi_p10:>15.2f} {afrm_stats.rsi_p10:>15.2f} {afrm_stats.rsi_p10 - spy_stats.rsi_p10:>+12.2f}")
    print(f"  50th Percentile (median)         {spy_stats.rsi_p50:>15.2f} {afrm_stats.rsi_p50:>15.2f} {afrm_stats.rsi_p50 - spy_stats.rsi_p50:>+12.2f}")
    print(f"  90th Percentile (overbought)     {spy_stats.rsi_p90:>15.2f} {afrm_stats.rsi_p90:>15.2f} {afrm_stats.rsi_p90 - spy_stats.rsi_p90:>+12.2f}")
    print(f"  % Time RSI < 30 (fixed)          {spy_stats.rsi_pct_below_30:>14.1f}% {afrm_stats.rsi_pct_below_30:>14.1f}%")
    print(f"  % Time RSI > 70 (fixed)          {spy_stats.rsi_pct_above_70:>14.1f}% {afrm_stats.rsi_pct_above_70:>14.1f}%")

    # ER Section
    print("\n[Efficiency Ratio]")
    print(f"  Mean                             {spy_stats.er_mean:>15.3f} {afrm_stats.er_mean:>15.3f} {afrm_stats.er_mean - spy_stats.er_mean:>+12.3f}")
    print(f"  Std Dev                          {spy_stats.er_std:>15.3f} {afrm_stats.er_std:>15.3f}")
    print(f"  10th Percentile                  {spy_stats.er_p10:>15.3f} {afrm_stats.er_p10:>15.3f}")
    print(f"  50th Percentile (median)         {spy_stats.er_p50:>15.3f} {afrm_stats.er_p50:>15.3f}")
    print(f"  90th Percentile                  {spy_stats.er_p90:>15.3f} {afrm_stats.er_p90:>15.3f}")
    print(f"  % Time ER > 0.6 (fixed trend)    {spy_stats.er_pct_above_06:>14.1f}% {afrm_stats.er_pct_above_06:>14.1f}%")
    print(f"  % Time ER < 0.3 (fixed choppy)   {spy_stats.er_pct_below_03:>14.1f}% {afrm_stats.er_pct_below_03:>14.1f}%")

    # ATR Section
    print("\n[ATR / Volatility]")
    print(f"  Mean ATR                         {spy_stats.atr_mean:>15.4f} {afrm_stats.atr_mean:>15.4f}")
    print(f"  ATR as % of Price (approx)       {spy_stats.atr_mean/500*100:>14.2f}% {afrm_stats.atr_mean/50*100:>14.2f}%")
    print(f"  ATR Ratio Mean                   {spy_stats.atr_ratio_mean:>15.3f} {afrm_stats.atr_ratio_mean:>15.3f}")
    print(f"  ATR Ratio 20th Pctl (low vol)    {spy_stats.atr_ratio_p20:>15.3f} {afrm_stats.atr_ratio_p20:>15.3f}")
    print(f"  ATR Ratio 80th Pctl (high vol)   {spy_stats.atr_ratio_p80:>15.3f} {afrm_stats.atr_ratio_p80:>15.3f}")

    print("\n" + "="*80)


def print_threshold_comparison(
    spy_stats: AssetStatistics,
    afrm_stats: AssetStatistics,
    spy_calibrated: CalibratedThresholds,
    afrm_calibrated: CalibratedThresholds
):
    """Print comparison of fixed vs calibrated thresholds."""

    print("\n" + "="*80)
    print("             FIXED vs CALIBRATED THRESHOLDS COMPARISON")
    print("="*80)

    print(f"\n{'Threshold':<30} {'Fixed':<12} {'SPY Calib.':<12} {'AFRM Calib.':<12}")
    print("-"*80)

    print("\n[RSI Thresholds]")
    print(f"  Oversold                       {'30.00':<12} {spy_calibrated.rsi_oversold:<12.2f} {afrm_calibrated.rsi_oversold:<12.2f}")
    print(f"  Overbought                     {'70.00':<12} {spy_calibrated.rsi_overbought:<12.2f} {afrm_calibrated.rsi_overbought:<12.2f}")

    print("\n[Efficiency Ratio Thresholds]")
    print(f"  Trending                       {'0.60':<12} {spy_calibrated.er_trending:<12.3f} {afrm_calibrated.er_trending:<12.3f}")
    print(f"  Choppy                         {'0.30':<12} {spy_calibrated.er_choppy:<12.3f} {afrm_calibrated.er_choppy:<12.3f}")

    print("\n[Volatility Ratio Thresholds]")
    print(f"  Low Vol                        {'0.50':<12} {spy_calibrated.vol_low:<12.3f} {afrm_calibrated.vol_low:<12.3f}")
    print(f"  High Vol                       {'1.50':<12} {spy_calibrated.vol_high:<12.3f} {afrm_calibrated.vol_high:<12.3f}")

    print("\n" + "-"*80)
    print("\n[Signal Generation Impact]")

    # Calculate how many signals would be generated
    spy_fixed_oversold = spy_stats.rsi_pct_below_30
    spy_calib_oversold = 10.0  # By definition, 10th percentile = 10%

    afrm_fixed_oversold = afrm_stats.rsi_pct_below_30
    afrm_calib_oversold = 10.0

    print(f"\n  SPY: Fixed RSI<30 triggers {spy_fixed_oversold:.1f}% of time")
    print(f"       Calibrated RSI<{spy_calibrated.rsi_oversold:.1f} triggers ~10% of time")
    print(f"       --> Fixed threshold {'underutilized' if spy_fixed_oversold < 10 else 'overutilized'}")

    print(f"\n  AFRM: Fixed RSI<30 triggers {afrm_fixed_oversold:.1f}% of time")
    print(f"        Calibrated RSI<{afrm_calibrated.rsi_oversold:.1f} triggers ~10% of time")
    print(f"        --> Fixed threshold {'underutilized' if afrm_fixed_oversold < 10 else 'overutilized'}")

    print("\n" + "="*80)


def demonstrate_golden_adaptive(
    symbol: str,
    bars: np.ndarray,
    dates: list
):
    """Demonstrate Golden Adaptive EMA calculation for an asset."""

    high = bars[:, 1]
    low = bars[:, 2]
    close = bars[:, 3]
    volume = bars[:, 4]

    # Calculate all component indicators
    print(f"\n[Calculating Golden Adaptive indicators for {symbol}...]")

    fractal_dim = calculate_fractal_dimension(high, low, lookback=20)
    kama, er, kama_regime = calculate_kama_with_regime_adjustment(close)
    atr = calculate_atr(high, low, close, period=14)
    atr_ma = calculate_atr_ma(atr, period=50)

    # Calculate Golden Alpha
    alpha, confidence, regime = calculate_golden_alpha(
        close, high, low, volume,
        fractal_dim, er, atr, atr_ma
    )

    # Calculate Golden EMA
    golden_ema = calculate_golden_ema(close, alpha)

    # Also calculate standard EMAs for comparison
    ema_20 = calculate_ema(close, period=20)

    # Print sample of results
    print(f"\n{'Date':<12} {'Close':>10} {'EMA(20)':>10} {'GoldenEMA':>10} {'Alpha':>8} {'Conf':>6} {'Regime':>8} {'FracDim':>8}")
    print("-"*90)

    # Show last 10 bars
    for i in range(-10, 0):
        idx = len(close) + i
        regime_str = {0.0: "MR", 1.0: "Neutral", 2.0: "Trend"}.get(regime[idx], "?")
        print(f"{dates[idx]:<12} {close[idx]:>10.2f} {ema_20[idx]:>10.2f} {golden_ema[idx]:>10.2f} "
              f"{alpha[idx]:>8.4f} {confidence[idx]:>6.2f} {regime_str:>8} {fractal_dim[idx]:>8.3f}")

    # Summary statistics
    valid_alpha = alpha[~np.isnan(alpha)]
    valid_conf = confidence[~np.isnan(confidence)]
    valid_regime = regime[~np.isnan(regime)]

    print(f"\n[Golden Adaptive Summary for {symbol}]")
    print(f"  Alpha range: {np.min(valid_alpha):.4f} - {np.max(valid_alpha):.4f} (mean: {np.mean(valid_alpha):.4f})")
    print(f"  Confidence: mean={np.mean(valid_conf):.3f}, min={np.min(valid_conf):.3f}")
    print(f"  Regime distribution:")
    print(f"    - Mean Reversion: {100*np.sum(valid_regime==0)/len(valid_regime):.1f}%")
    print(f"    - Neutral: {100*np.sum(valid_regime==1)/len(valid_regime):.1f}%")
    print(f"    - Trending: {100*np.sum(valid_regime==2)/len(valid_regime):.1f}%")

    return {
        'golden_ema': golden_ema,
        'alpha': alpha,
        'confidence': confidence,
        'regime': regime,
        'fractal_dim': fractal_dim
    }


# ============================================================================
# PART 5: Main Execution
# ============================================================================

async def fetch_data(symbol: str, days: int = 365) -> tuple:
    """Fetch historical data for a symbol."""
    provider = get_provider()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    try:
        data = await provider.fetch_historical_data(
            symbol,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            interval="1d"
        )
        return data.bars, data.dates
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None, None


async def main():
    """Main execution function."""

    print("\n" + "="*80)
    print("       GOLDEN ADAPTIVE INDICATOR EXAMPLE: SPY vs AFRM")
    print("       Demonstrating Why Fixed Thresholds Fail")
    print("="*80)

    # Fetch data for both symbols
    print("\n[Fetching historical data...]")

    spy_bars, spy_dates = await fetch_data("SPY", days=365)
    afrm_bars, afrm_dates = await fetch_data("AFRM", days=365)

    if spy_bars is None or afrm_bars is None:
        print("Failed to fetch data. Please check your internet connection.")
        return

    print(f"  SPY: {len(spy_bars)} bars loaded")
    print(f"  AFRM: {len(afrm_bars)} bars loaded")

    # Calculate statistics
    print("\n[Calculating indicator statistics...]")
    spy_stats = calculate_asset_statistics("SPY", spy_bars, spy_dates)
    afrm_stats = calculate_asset_statistics("AFRM", afrm_bars, afrm_dates)

    # Print comparison
    print_comparison_table(spy_stats, afrm_stats)

    # Calibrate thresholds
    spy_calibrated = calibrate_thresholds(spy_stats)
    afrm_calibrated = calibrate_thresholds(afrm_stats)

    # Print threshold comparison
    print_threshold_comparison(spy_stats, afrm_stats, spy_calibrated, afrm_calibrated)

    # Demonstrate Golden Adaptive
    print("\n" + "="*80)
    print("                    GOLDEN ADAPTIVE EMA DEMONSTRATION")
    print("="*80)

    spy_golden = demonstrate_golden_adaptive("SPY", spy_bars, spy_dates)
    afrm_golden = demonstrate_golden_adaptive("AFRM", afrm_bars, afrm_dates)

    # Final summary
    print("\n" + "="*80)
    print("                              KEY INSIGHTS")
    print("="*80)
    print("""
    1. FIXED THRESHOLDS FAIL ACROSS ASSETS
       - SPY and AFRM have completely different RSI distributions
       - Using RSI < 30 for both would generate vastly different signal frequencies
       - AFRM is more volatile, so its RSI swings more widely

    2. PERCENTILE-BASED THRESHOLDS ADAPT
       - 10th percentile for oversold adapts to each asset's range
       - SPY's 10th percentile might be 38, AFRM's might be 25
       - Both generate signals ~10% of the time (consistent behavior)

    3. GOLDEN ADAPTIVE COMBINES MULTIPLE DIMENSIONS
       - Fractal Dimension: Market structure (trending vs mean-reverting)
       - Efficiency Ratio: Movement quality (directional vs noise)
       - Volatility: Market activity level
       - Volume: Participation confirmation

    4. CONFIDENCE SCORE INDICATES SIGNAL QUALITY
       - High confidence = all dimensions agree
       - Low confidence = dimensions conflict, be cautious

    5. IMPLEMENTATION STRATEGY
       - Run calibration weekly/monthly
       - Store calibrated parameters per asset
       - Use Golden Adaptive for entries/exits
       - Scale position size by confidence score
    """)
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
