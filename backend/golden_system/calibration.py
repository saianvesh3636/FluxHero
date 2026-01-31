"""
Golden System Calibration - Layer 1 of Three-Tier System

Advanced calibration for the complex Golden Adaptive system:
1. Percentile-based threshold estimation (no magic numbers)
2. Walk-forward optimization
3. Dimension weight optimization
4. Rolling recalibration

This module is SELF-CONTAINED - can be removed without affecting other code.

Usage:
    from backend.golden_system import GoldenCalibrator

    calibrator = GoldenCalibrator()
    params = calibrator.calibrate(bars, symbol="SPY")
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Callable

from backend.golden_system.computation import (
    compute_golden_adaptive_indicators,
    analyze_dimension_contribution,
)


@dataclass
class GoldenParameters:
    """
    Calibrated parameters for the Golden Adaptive Strategy.

    All thresholds derived from data - no magic numbers.
    """
    # Metadata
    symbol: str
    calibration_date: str
    lookback_bars: int

    # Indicator periods (can be optimized)
    fractal_lookback: int = 20
    vol_lookback: int = 14
    er_period: int = 10
    volume_period: int = 20
    slow_period: int = 30
    fast_period: int = 2

    # Dimension weights (sum to 1.0)
    w_fractal: float = 0.30
    w_efficiency: float = 0.30
    w_volatility: float = 0.25
    w_volume: float = 0.15

    # No threshold parameters needed - signals use crossovers and rolling extremes

    # ATR multipliers for stops (calibrated from price movement)
    atr_stop_trending: float = 2.5
    atr_stop_mr: float = 1.5
    atr_stop_neutral: float = 2.0

    # Risk:Reward ratios (calibrated)
    rr_trending: float = 3.0
    rr_mr: float = 1.5
    rr_neutral: float = 2.0

    # Position sizing
    risk_per_trade: float = 0.01
    max_position_pct: float = 0.20

    # Regime distribution from calibration data
    regime_pct_trending: float = 0.0
    regime_pct_mr: float = 0.0
    regime_pct_neutral: float = 0.0

    # Validation metrics
    in_sample_sharpe: float = 0.0
    out_sample_sharpe: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'calibration_date': self.calibration_date,
            'lookback_bars': self.lookback_bars,
            'fractal_lookback': self.fractal_lookback,
            'vol_lookback': self.vol_lookback,
            'er_period': self.er_period,
            'volume_period': self.volume_period,
            'slow_period': self.slow_period,
            'fast_period': self.fast_period,
            'w_fractal': self.w_fractal,
            'w_efficiency': self.w_efficiency,
            'w_volatility': self.w_volatility,
            'w_volume': self.w_volume,
            'atr_stop_trending': self.atr_stop_trending,
            'atr_stop_mr': self.atr_stop_mr,
            'atr_stop_neutral': self.atr_stop_neutral,
            'rr_trending': self.rr_trending,
            'rr_mr': self.rr_mr,
            'rr_neutral': self.rr_neutral,
        }


class GoldenCalibrator:
    """
    Calibrate parameters for the Golden Adaptive Strategy.

    Uses data-driven estimation instead of magic numbers.
    """

    def __init__(
        self,
        min_bars: int = 252,  # 1 year minimum
    ):
        self.min_bars = min_bars

    def calibrate(
        self,
        bars: np.ndarray,
        symbol: str = "",
    ) -> GoldenParameters:
        """
        Calibrate all parameters from historical data.

        Parameters
        ----------
        bars : np.ndarray
            OHLCV data
        symbol : str
            Asset symbol

        Returns
        -------
        GoldenParameters
            Calibrated parameters
        """
        if len(bars) < self.min_bars:
            raise ValueError(f"Need at least {self.min_bars} bars, got {len(bars)}")

        # Start with default indicator parameters
        params = GoldenParameters(
            symbol=symbol.upper(),
            calibration_date=datetime.now().strftime("%Y-%m-%d"),
            lookback_bars=len(bars),
        )

        # Compute indicators with default parameters
        indicators = compute_golden_adaptive_indicators(
            bars,
            fractal_lookback=params.fractal_lookback,
            vol_lookback=params.vol_lookback,
            er_period=params.er_period,
            volume_period=params.volume_period,
            slow_period=params.slow_period,
            fast_period=params.fast_period,
            w_fractal=params.w_fractal,
            w_efficiency=params.w_efficiency,
            w_volatility=params.w_volatility,
            w_volume=params.w_volume,
        )

        # Note: confidence_percentile and deviation_percentile are user-configurable
        # They define what percentile thresholds to use (no magic numbers)
        # Default: confidence_percentile=25 (trade top 75%), deviation_percentile=80

        # Analyze regime distribution
        analysis = analyze_dimension_contribution(indicators)
        params.regime_pct_trending = analysis['regime_distribution']['trending']
        params.regime_pct_mr = analysis['regime_distribution']['mean_reversion']
        params.regime_pct_neutral = analysis['regime_distribution']['neutral']

        # Calibrate ATR multipliers from price movement
        atr_mults = self._calibrate_atr_multipliers(bars)
        params.atr_stop_trending = atr_mults['trending']
        params.atr_stop_mr = atr_mults['mr']
        params.atr_stop_neutral = atr_mults['neutral']

        # Calibrate R:R ratios from winning trade analysis
        rr_ratios = self._calibrate_reward_ratios(bars)
        params.rr_trending = rr_ratios['trending']
        params.rr_mr = rr_ratios['mr']
        params.rr_neutral = rr_ratios['neutral']

        return params

    def _calibrate_atr_multipliers(self, bars: np.ndarray) -> dict:
        """
        Calibrate ATR stop multipliers from actual price movements.

        Analyzes how far price typically retraces to set stops that
        don't get hit by normal noise.
        """
        high = bars[:, 1]
        low = bars[:, 2]
        close = bars[:, 3]
        n = len(bars)

        # Calculate ATR
        atr = self._calculate_atr(high, low, close, period=14)

        # Analyze retracements
        retracements_up = []  # After up moves
        retracements_down = []  # After down moves

        lookback = 5  # 5-bar swing

        for i in range(lookback + 20, n - lookback):
            if np.isnan(atr[i]) or atr[i] <= 0:
                continue

            # Detect swing high/low
            is_swing_high = True
            is_swing_low = True

            for j in range(i - lookback, i + lookback + 1):
                if j != i:
                    if high[j] >= high[i]:
                        is_swing_high = False
                    if low[j] <= low[i]:
                        is_swing_low = False

            if is_swing_high:
                # Measure retracement after swing high
                min_after = np.min(low[i:i+lookback])
                retrace = (high[i] - min_after) / atr[i]
                retracements_down.append(retrace)

            if is_swing_low:
                # Measure retracement after swing low
                max_after = np.max(high[i:i+lookback])
                retrace = (max_after - low[i]) / atr[i]
                retracements_up.append(retrace)

        # Set stops at 75th percentile of retracements
        # (only 25% of normal moves would hit the stop)
        all_retracements = retracements_up + retracements_down

        if len(all_retracements) > 10:
            p75 = np.percentile(all_retracements, 75)
            p50 = np.percentile(all_retracements, 50)
            p90 = np.percentile(all_retracements, 90)
        else:
            # Fallback to defaults
            p50, p75, p90 = 1.5, 2.0, 2.5

        return {
            'mr': max(1.0, min(2.5, p50)),        # Tighter for MR
            'neutral': max(1.5, min(3.0, p75)),   # Medium
            'trending': max(2.0, min(4.0, p90)),  # Wider for trends
        }

    def _calibrate_reward_ratios(self, bars: np.ndarray) -> dict:
        """
        Calibrate risk:reward ratios from price movement patterns.

        Analyzes how far winning moves typically go relative to
        initial risk (stop distance).
        """
        high = bars[:, 1]
        low = bars[:, 2]
        close = bars[:, 3]
        n = len(bars)

        atr = self._calculate_atr(high, low, close, period=14)

        # Track move extensions after entries
        up_extensions = []
        down_extensions = []

        for i in range(50, n - 20):
            if np.isnan(atr[i]) or atr[i] <= 0:
                continue

            # Simulate long entry: how far does price go up vs down?
            max_up = np.max(high[i:i+20]) - close[i]
            max_down = close[i] - np.min(low[i:i+20])

            if max_down > 0:
                up_extensions.append(max_up / max_down)

            # Simulate short entry
            if max_up > 0:
                down_extensions.append(max_down / max_up)

        all_extensions = up_extensions + down_extensions

        if len(all_extensions) > 10:
            p50 = np.percentile(all_extensions, 50)
            p75 = np.percentile(all_extensions, 75)
        else:
            p50, p75 = 1.5, 2.5

        return {
            'mr': max(1.0, min(2.0, p50)),        # Lower for quick MR trades
            'neutral': max(1.5, min(2.5, p75)),
            'trending': max(2.0, min(4.0, p75 * 1.2)),  # Higher for trends
        }

    def _calculate_atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int
    ) -> np.ndarray:
        """Calculate ATR."""
        n = len(high)
        tr = np.full(n, np.nan)
        atr = np.full(n, np.nan)

        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )

        if n < period + 1:
            return atr

        atr[period] = np.mean(tr[1:period+1])

        for i in range(period + 1, n):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

        return atr

    def optimize_weights(
        self,
        bars: np.ndarray,
        symbol: str = "",
        n_iterations: int = 50,
    ) -> GoldenParameters:
        """
        Optimize dimension weights using random search.

        Tests different weight combinations and finds the one
        that produces the most consistent confidence scores.

        Parameters
        ----------
        bars : np.ndarray
            OHLCV data
        symbol : str
            Asset symbol
        n_iterations : int
            Number of random weight combinations to try

        Returns
        -------
        GoldenParameters
            Parameters with optimized weights
        """
        best_params = None
        best_score = -np.inf

        for _ in range(n_iterations):
            # Generate random weights that sum to 1
            weights = np.random.random(4)
            weights = weights / weights.sum()

            try:
                indicators = compute_golden_adaptive_indicators(
                    bars,
                    w_fractal=weights[0],
                    w_efficiency=weights[1],
                    w_volatility=weights[2],
                    w_volume=weights[3],
                )

                # Score: higher mean confidence with lower std
                confidence = indicators['confidence']
                valid = confidence[~np.isnan(confidence)]

                if len(valid) > 100:
                    score = np.mean(valid) - 0.5 * np.std(valid)

                    if score > best_score:
                        best_score = score
                        best_params = {
                            'w_fractal': weights[0],
                            'w_efficiency': weights[1],
                            'w_volatility': weights[2],
                            'w_volume': weights[3],
                        }
            except Exception:
                continue

        # Calibrate with optimized weights
        params = self.calibrate(bars, symbol)

        if best_params:
            params.w_fractal = best_params['w_fractal']
            params.w_efficiency = best_params['w_efficiency']
            params.w_volatility = best_params['w_volatility']
            params.w_volume = best_params['w_volume']

        return params


class WalkForwardCalibrator:
    """
    Walk-forward calibration for out-of-sample validation.

    Splits data into train/test windows and validates
    parameters on unseen data.
    """

    def __init__(
        self,
        train_bars: int = 252,   # 1 year training
        test_bars: int = 63,     # 3 months testing
        step_bars: int = 21,     # 1 month step
    ):
        self.train_bars = train_bars
        self.test_bars = test_bars
        self.step_bars = step_bars
        self.calibrator = GoldenCalibrator(min_bars=train_bars)

    def run_walk_forward(
        self,
        bars: np.ndarray,
        symbol: str = "",
    ) -> List[dict]:
        """
        Run walk-forward calibration and validation.

        Returns
        -------
        List[dict]
            Results for each walk-forward window
        """
        from backend.golden_system.strategy import backtest_golden_strategy

        n = len(bars)
        results = []

        start = 0
        while start + self.train_bars + self.test_bars <= n:
            train_end = start + self.train_bars
            test_end = train_end + self.test_bars

            # Train window
            train_bars = bars[start:train_end]

            # Test window
            test_bars_data = bars[train_end:test_end]

            # Calibrate on training data
            params = self.calibrator.calibrate(train_bars, symbol)

            # Backtest on test data with calibrated params
            test_result = backtest_golden_strategy(
                test_bars_data,
                symbol=symbol,
                w_fractal=params.w_fractal,
                w_efficiency=params.w_efficiency,
                w_volatility=params.w_volatility,
                w_volume=params.w_volume,
            )

            # Also backtest on training data for comparison
            train_result = backtest_golden_strategy(
                train_bars,
                symbol=symbol,
                w_fractal=params.w_fractal,
                w_efficiency=params.w_efficiency,
                w_volatility=params.w_volatility,
                w_volume=params.w_volume,
            )

            results.append({
                'window_start': start,
                'window_end': test_end,
                'train_sharpe': train_result.sharpe_ratio,
                'test_sharpe': test_result.sharpe_ratio,
                'train_return': train_result.total_return,
                'test_return': test_result.total_return,
                'params': params.to_dict(),
            })

            start += self.step_bars

        return results

    def print_walk_forward_summary(self, results: List[dict]) -> None:
        """Print summary of walk-forward results."""
        if not results:
            print("No walk-forward results")
            return

        print("\n" + "=" * 70)
        print("  WALK-FORWARD CALIBRATION SUMMARY")
        print("=" * 70)

        train_sharpes = [r['train_sharpe'] for r in results]
        test_sharpes = [r['test_sharpe'] for r in results]
        train_returns = [r['train_return'] for r in results]
        test_returns = [r['test_return'] for r in results]

        print(f"\n[Sharpe Ratio]")
        print(f"  Train: Mean={np.mean(train_sharpes):.2f}, Std={np.std(train_sharpes):.2f}")
        print(f"  Test:  Mean={np.mean(test_sharpes):.2f}, Std={np.std(test_sharpes):.2f}")
        print(f"  Degradation: {(np.mean(train_sharpes) - np.mean(test_sharpes)):.2f}")

        print(f"\n[Returns]")
        print(f"  Train: Mean={np.mean(train_returns)*100:.1f}%, Std={np.std(train_returns)*100:.1f}%")
        print(f"  Test:  Mean={np.mean(test_returns)*100:.1f}%, Std={np.std(test_returns)*100:.1f}%")

        # Overfitting indicator: large gap between train and test
        overfit_score = (np.mean(train_sharpes) - np.mean(test_sharpes)) / max(0.1, np.mean(train_sharpes))
        print(f"\n[Overfitting Score]: {overfit_score:.2f}")
        if overfit_score > 0.5:
            print("  WARNING: High overfitting detected!")
        elif overfit_score > 0.25:
            print("  CAUTION: Moderate overfitting")
        else:
            print("  OK: Low overfitting")

        print("=" * 70 + "\n")
