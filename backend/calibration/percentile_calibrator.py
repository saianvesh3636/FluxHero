"""
Percentile Calibrator - Derive Thresholds from Data Distribution

The core principle: Replace magic numbers with percentiles.

Instead of:
    if rsi < 30:  # Magic number!

Use:
    if rsi < calibrated_params.rsi_oversold:  # 10th percentile of RSI distribution

This ensures:
1. Thresholds adapt to each asset's behavior
2. Consistent signal frequency across assets (10th percentile always triggers ~10%)
3. No overfitting to historical data

Usage:
    from backend.calibration import PercentileCalibrator

    calibrator = PercentileCalibrator()
    params = calibrator.calibrate(symbol="SPY", bars=price_data)
"""

import numpy as np
from datetime import datetime
from typing import Optional

from backend.calibration.parameter_store import (
    CalibratedParameters,
    IndicatorDistribution,
    CalibrationConfig,
)
from backend.computation.indicators import (
    calculate_rsi,
    calculate_atr,
    calculate_sma,
)
from backend.computation.adaptive_ema import (
    calculate_efficiency_ratio,
    calculate_kama_with_regime_adjustment,
)
from backend.computation.volatility import calculate_atr_ma
from backend.computation.golden_ema import calculate_simple_golden_ema


class PercentileCalibrator:
    """
    Calibrate indicator thresholds using percentile-based analysis.

    Principle: Let the data define what "oversold", "overbought", "trending" mean
    for each specific asset, rather than using universal magic numbers.
    """

    def __init__(self, config: Optional[CalibrationConfig] = None):
        self.config = config or CalibrationConfig()

    def calibrate(
        self,
        symbol: str,
        bars: np.ndarray,
        dates: Optional[list[str]] = None
    ) -> CalibratedParameters:
        """
        Calibrate all parameters for an asset from its price data.

        Parameters
        ----------
        symbol : str
            Asset symbol (e.g., "SPY", "AFRM")
        bars : np.ndarray
            OHLCV data with shape (N, 5): [open, high, low, close, volume]
        dates : list[str], optional
            Date strings for each bar

        Returns
        -------
        CalibratedParameters
            Fully calibrated parameters derived from data
        """
        if len(bars) < self.config.min_bars_required:
            raise ValueError(
                f"Insufficient data: {len(bars)} bars, need at least {self.config.min_bars_required}"
            )

        # Extract OHLCV
        high = bars[:, 1]
        low = bars[:, 2]
        close = bars[:, 3]
        volume = bars[:, 4]

        # Calculate all indicators
        rsi = calculate_rsi(close, period=14)
        er = calculate_efficiency_ratio(close, period=10)
        atr = calculate_atr(high, low, close, period=14)
        atr_ma = calculate_atr_ma(atr, period=50)

        # ATR ratio (current volatility vs average)
        atr_ratio = self._safe_divide(atr, atr_ma)

        # Golden EMA alpha
        _, alpha = calculate_simple_golden_ema(high, low, close)

        # Calculate distributions
        rsi_dist = IndicatorDistribution.from_array(rsi)
        er_dist = IndicatorDistribution.from_array(er)
        vol_dist = IndicatorDistribution.from_array(atr_ratio)
        alpha_dist = IndicatorDistribution.from_array(alpha)

        # Calibrate ATR multipliers
        atr_entry, atr_exit, atr_stop = self._calibrate_atr_multipliers(
            high, low, close, atr
        )

        # Build calibrated parameters
        params = CalibratedParameters(
            # Metadata
            symbol=symbol.upper(),
            calibration_date=datetime.now().strftime("%Y-%m-%d"),
            lookback_bars=len(bars),
            data_start_date=dates[0] if dates else "unknown",
            data_end_date=dates[-1] if dates else "unknown",

            # RSI thresholds
            rsi_oversold=rsi_dist.p10,
            rsi_overbought=rsi_dist.p90,
            rsi_extreme_oversold=rsi_dist.p5,
            rsi_extreme_overbought=rsi_dist.p95,
            rsi_neutral_low=rsi_dist.p25,
            rsi_neutral_high=rsi_dist.p75,

            # Efficiency Ratio thresholds
            er_choppy=er_dist.p25,
            er_trending=er_dist.p75,
            er_strong_trend=er_dist.p90,

            # Volatility thresholds
            vol_low=vol_dist.p25 if not np.isnan(vol_dist.p25) else 0.7,
            vol_normal_low=vol_dist.p40 if hasattr(vol_dist, 'p40') else (vol_dist.p25 + vol_dist.p50) / 2,
            vol_normal_high=vol_dist.p50 + (vol_dist.p75 - vol_dist.p50) / 2,
            vol_high=vol_dist.p75 if not np.isnan(vol_dist.p75) else 1.3,
            vol_extreme=vol_dist.p95 if not np.isnan(vol_dist.p95) else 2.0,

            # Alpha thresholds
            alpha_slow_regime=alpha_dist.p25 if not np.isnan(alpha_dist.p25) else 0.1,
            alpha_fast_regime=alpha_dist.p75 if not np.isnan(alpha_dist.p75) else 0.5,

            # ATR multipliers
            atr_entry_multiplier=atr_entry,
            atr_exit_multiplier=atr_exit,
            atr_stop_multiplier=atr_stop,

            # Store distributions for reference
            rsi_distribution=rsi_dist,
            er_distribution=er_dist,
            vol_ratio_distribution=vol_dist,
            alpha_distribution=alpha_dist,
        )

        return params

    def _safe_divide(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Safely divide arrays, returning NaN for invalid divisions."""
        result = np.full_like(a, np.nan)
        valid = (b != 0) & ~np.isnan(a) & ~np.isnan(b)
        result[valid] = a[valid] / b[valid]
        return result

    def _calibrate_atr_multipliers(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        atr: np.ndarray
    ) -> tuple[float, float, float]:
        """
        Calibrate ATR multipliers based on actual price movement patterns.

        Analyzes how price moves relative to ATR to determine appropriate
        entry, exit, and stop multipliers.

        Returns
        -------
        tuple[float, float, float]
            (entry_multiplier, exit_multiplier, stop_multiplier)
        """
        n = len(close)
        lookback = min(self.config.atr_lookback_for_multipliers, n - 50)

        # Analyze price movements relative to ATR
        daily_moves = []    # Daily |close - close_prev| / ATR
        swing_moves = []    # 5-day swings / ATR
        drawdowns = []      # Drawdowns from local highs / ATR

        for i in range(50, n):
            if np.isnan(atr[i]) or atr[i] == 0:
                continue

            # Daily move as multiple of ATR
            if i > 0:
                daily_move = abs(close[i] - close[i-1]) / atr[i]
                daily_moves.append(daily_move)

            # 5-day swing
            if i >= 5:
                swing = abs(close[i] - close[i-5]) / atr[i]
                swing_moves.append(swing)

            # Drawdown from 20-day high
            if i >= 20:
                high_20 = np.max(high[i-20:i])
                dd = (high_20 - low[i]) / atr[i]
                drawdowns.append(dd)

        daily_moves = np.array(daily_moves)
        swing_moves = np.array(swing_moves)
        drawdowns = np.array(drawdowns)

        # Entry multiplier: Based on typical daily noise
        # Want entry band to filter out ~50% of daily noise
        entry_mult = np.percentile(daily_moves, 50) if len(daily_moves) > 0 else 0.5

        # Exit multiplier: Smaller than entry for tighter exits
        # Based on ~30th percentile of daily moves
        exit_mult = np.percentile(daily_moves, 30) if len(daily_moves) > 0 else 0.3

        # Stop multiplier: Based on typical drawdowns
        # Want stop to not get hit by normal retracements (~75th percentile)
        stop_mult = np.percentile(drawdowns, 75) if len(drawdowns) > 0 else 2.5

        # Clamp to reasonable ranges
        entry_mult = max(0.2, min(1.5, entry_mult))
        exit_mult = max(0.1, min(1.0, exit_mult))
        stop_mult = max(1.5, min(4.0, stop_mult))

        return float(entry_mult), float(exit_mult), float(stop_mult)

    def compare_fixed_vs_calibrated(
        self,
        symbol: str,
        bars: np.ndarray,
        params: CalibratedParameters
    ) -> dict:
        """
        Compare signal frequency between fixed and calibrated thresholds.

        Returns statistics showing why fixed thresholds fail.
        """
        close = bars[:, 3]
        high = bars[:, 1]
        low = bars[:, 2]

        rsi = calculate_rsi(close, period=14)
        er = calculate_efficiency_ratio(close, period=10)
        atr = calculate_atr(high, low, close, period=14)
        atr_ma = calculate_atr_ma(atr, period=50)
        atr_ratio = self._safe_divide(atr, atr_ma)

        # Filter valid values
        rsi_valid = rsi[~np.isnan(rsi)]
        er_valid = er[~np.isnan(er)]
        atr_ratio_valid = atr_ratio[~np.isnan(atr_ratio)]

        comparison = {
            "symbol": symbol,
            "n_bars": len(bars),
            "rsi": {
                "fixed_oversold_30": {
                    "threshold": 30,
                    "pct_triggered": 100 * np.sum(rsi_valid < 30) / len(rsi_valid),
                },
                "calibrated_oversold": {
                    "threshold": params.rsi_oversold,
                    "pct_triggered": 100 * np.sum(rsi_valid < params.rsi_oversold) / len(rsi_valid),
                },
                "fixed_overbought_70": {
                    "threshold": 70,
                    "pct_triggered": 100 * np.sum(rsi_valid > 70) / len(rsi_valid),
                },
                "calibrated_overbought": {
                    "threshold": params.rsi_overbought,
                    "pct_triggered": 100 * np.sum(rsi_valid > params.rsi_overbought) / len(rsi_valid),
                },
            },
            "er": {
                "fixed_trending_0.6": {
                    "threshold": 0.6,
                    "pct_triggered": 100 * np.sum(er_valid > 0.6) / len(er_valid),
                },
                "calibrated_trending": {
                    "threshold": params.er_trending,
                    "pct_triggered": 100 * np.sum(er_valid > params.er_trending) / len(er_valid),
                },
                "fixed_choppy_0.3": {
                    "threshold": 0.3,
                    "pct_triggered": 100 * np.sum(er_valid < 0.3) / len(er_valid),
                },
                "calibrated_choppy": {
                    "threshold": params.er_choppy,
                    "pct_triggered": 100 * np.sum(er_valid < params.er_choppy) / len(er_valid),
                },
            },
            "volatility": {
                "fixed_high_1.5": {
                    "threshold": 1.5,
                    "pct_triggered": 100 * np.sum(atr_ratio_valid > 1.5) / len(atr_ratio_valid),
                },
                "calibrated_high": {
                    "threshold": params.vol_high,
                    "pct_triggered": 100 * np.sum(atr_ratio_valid > params.vol_high) / len(atr_ratio_valid),
                },
            },
        }

        return comparison

    def print_comparison(self, comparison: dict) -> None:
        """Pretty print the fixed vs calibrated comparison."""
        print(f"\n{'='*70}")
        print(f"  FIXED vs CALIBRATED THRESHOLDS: {comparison['symbol']}")
        print(f"  Data: {comparison['n_bars']} bars")
        print(f"{'='*70}")

        print("\n[RSI Thresholds]")
        rsi = comparison['rsi']
        print(f"  Oversold:")
        print(f"    Fixed (30):       triggers {rsi['fixed_oversold_30']['pct_triggered']:.1f}% of time")
        print(f"    Calibrated ({rsi['calibrated_oversold']['threshold']:.1f}): triggers {rsi['calibrated_oversold']['pct_triggered']:.1f}% of time")
        print(f"  Overbought:")
        print(f"    Fixed (70):       triggers {rsi['fixed_overbought_70']['pct_triggered']:.1f}% of time")
        print(f"    Calibrated ({rsi['calibrated_overbought']['threshold']:.1f}): triggers {rsi['calibrated_overbought']['pct_triggered']:.1f}% of time")

        print("\n[Efficiency Ratio Thresholds]")
        er = comparison['er']
        print(f"  Trending:")
        print(f"    Fixed (0.6):      triggers {er['fixed_trending_0.6']['pct_triggered']:.1f}% of time")
        print(f"    Calibrated ({er['calibrated_trending']['threshold']:.3f}): triggers {er['calibrated_trending']['pct_triggered']:.1f}% of time")
        print(f"  Choppy:")
        print(f"    Fixed (0.3):      triggers {er['fixed_choppy_0.3']['pct_triggered']:.1f}% of time")
        print(f"    Calibrated ({er['calibrated_choppy']['threshold']:.3f}): triggers {er['calibrated_choppy']['pct_triggered']:.1f}% of time")

        print("\n[Volatility Thresholds]")
        vol = comparison['volatility']
        print(f"  High Volatility:")
        print(f"    Fixed (1.5x):     triggers {vol['fixed_high_1.5']['pct_triggered']:.1f}% of time")
        print(f"    Calibrated ({vol['calibrated_high']['threshold']:.2f}x): triggers {vol['calibrated_high']['pct_triggered']:.1f}% of time")

        print(f"\n{'='*70}")
