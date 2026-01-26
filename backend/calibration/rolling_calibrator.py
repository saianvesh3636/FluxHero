"""
Rolling Calibrator - Continuous Parameter Recalibration

Implements walk-forward calibration similar to Two Sigma's approach:
1. Calibrate on a training window
2. Apply parameters to the next period
3. Slide forward and repeat

This ensures parameters adapt to changing market conditions without overfitting.

Usage:
    from backend.calibration import RollingCalibrator

    calibrator = RollingCalibrator(
        lookback_bars=252,       # 1 year training window
        recalibrate_every=21     # Monthly recalibration
    )

    # During backtest
    for bar_idx in range(len(bars)):
        if calibrator.should_recalibrate(bar_idx):
            params = calibrator.recalibrate(bars, bar_idx, symbol)
        # Use params for trading decisions
"""

import numpy as np
from datetime import datetime
from typing import Optional, Callable
from dataclasses import dataclass

from backend.calibration.parameter_store import (
    CalibratedParameters,
    ParameterStore,
    CalibrationConfig,
)
from backend.calibration.percentile_calibrator import PercentileCalibrator


@dataclass
class CalibrationEvent:
    """Record of a calibration event for analysis."""
    bar_index: int
    date: str
    params: CalibratedParameters
    reason: str  # "initial", "scheduled", "regime_change", "manual"


class RollingCalibrator:
    """
    Rolling calibration system that recalibrates parameters periodically.

    Key features:
    - Walk-forward calibration (no look-ahead bias)
    - Configurable recalibration frequency
    - Parameter stability tracking
    - Optional regime-triggered recalibration
    """

    def __init__(
        self,
        lookback_bars: int = 252,
        recalibrate_every: int = 21,
        min_bars_for_calibration: int = 100,
        store: Optional[ParameterStore] = None
    ):
        """
        Initialize rolling calibrator.

        Parameters
        ----------
        lookback_bars : int
            Number of bars to use for calibration (default: 252 = 1 year)
        recalibrate_every : int
            Recalibrate every N bars (default: 21 = monthly)
        min_bars_for_calibration : int
            Minimum bars required before first calibration
        store : ParameterStore, optional
            Parameter store for persistence
        """
        self.lookback_bars = lookback_bars
        self.recalibrate_every = recalibrate_every
        self.min_bars = min_bars_for_calibration

        self.config = CalibrationConfig(
            lookback_bars=lookback_bars,
            recalibrate_every_bars=recalibrate_every,
            min_bars_required=min_bars_for_calibration,
        )

        self.percentile_calibrator = PercentileCalibrator(self.config)
        self.store = store or ParameterStore()

        # State tracking
        self.last_calibration_bar: int = -1
        self.current_params: Optional[CalibratedParameters] = None
        self.calibration_history: list[CalibrationEvent] = []
        self.symbol: Optional[str] = None

    def should_recalibrate(self, current_bar: int) -> bool:
        """
        Check if recalibration is needed.

        Returns True if:
        - Never calibrated and have enough bars
        - Enough bars since last calibration
        """
        # First calibration
        if self.current_params is None:
            return current_bar >= self.min_bars

        # Scheduled recalibration
        bars_since_calibration = current_bar - self.last_calibration_bar
        return bars_since_calibration >= self.recalibrate_every

    def recalibrate(
        self,
        bars: np.ndarray,
        current_bar: int,
        symbol: str,
        dates: Optional[list[str]] = None,
        reason: str = "scheduled"
    ) -> CalibratedParameters:
        """
        Recalibrate parameters using data up to current_bar (no look-ahead).

        Parameters
        ----------
        bars : np.ndarray
            Full OHLCV data array
        current_bar : int
            Current bar index (calibration uses data BEFORE this)
        symbol : str
            Asset symbol
        dates : list[str], optional
            Date strings for each bar
        reason : str
            Reason for calibration

        Returns
        -------
        CalibratedParameters
            Newly calibrated parameters
        """
        self.symbol = symbol.upper()

        # Determine calibration window (NO LOOK-AHEAD)
        end_idx = current_bar  # Exclusive - don't include current bar
        start_idx = max(0, end_idx - self.lookback_bars)

        if end_idx - start_idx < self.min_bars:
            raise ValueError(
                f"Insufficient data for calibration: {end_idx - start_idx} bars, "
                f"need {self.min_bars}"
            )

        # Extract training window
        train_bars = bars[start_idx:end_idx]
        train_dates = dates[start_idx:end_idx] if dates else None

        # Calibrate
        params = self.percentile_calibrator.calibrate(
            symbol=symbol,
            bars=train_bars,
            dates=train_dates
        )

        # Calculate stability metric if we have prior params
        if self.current_params is not None:
            params.parameter_stability = self._calculate_stability(
                self.current_params, params
            )

        # Update state
        self.current_params = params
        self.last_calibration_bar = current_bar

        # Record event
        event = CalibrationEvent(
            bar_index=current_bar,
            date=dates[current_bar] if dates and current_bar < len(dates) else datetime.now().strftime("%Y-%m-%d"),
            params=params,
            reason=reason
        )
        self.calibration_history.append(event)

        # Persist
        self.store.save(params)

        return params

    def get_params(self) -> Optional[CalibratedParameters]:
        """Get current calibrated parameters."""
        return self.current_params

    def get_or_calibrate(
        self,
        bars: np.ndarray,
        current_bar: int,
        symbol: str,
        dates: Optional[list[str]] = None
    ) -> CalibratedParameters:
        """
        Get current params or recalibrate if needed.

        Convenience method for use in backtests.
        """
        if self.should_recalibrate(current_bar):
            reason = "initial" if self.current_params is None else "scheduled"
            return self.recalibrate(bars, current_bar, symbol, dates, reason)

        return self.current_params

    def _calculate_stability(
        self,
        old_params: CalibratedParameters,
        new_params: CalibratedParameters
    ) -> float:
        """
        Calculate parameter stability score.

        Returns a score from 0 (completely different) to 1 (identical).
        Useful for detecting regime changes or potential overfitting.
        """
        # Compare key parameters
        comparisons = [
            (old_params.rsi_oversold, new_params.rsi_oversold, 50),      # Normalize by typical range
            (old_params.rsi_overbought, new_params.rsi_overbought, 50),
            (old_params.er_choppy, new_params.er_choppy, 0.5),
            (old_params.er_trending, new_params.er_trending, 0.5),
            (old_params.vol_low, new_params.vol_low, 1.0),
            (old_params.vol_high, new_params.vol_high, 1.0),
            (old_params.atr_stop_multiplier, new_params.atr_stop_multiplier, 2.0),
        ]

        total_diff = 0.0
        for old_val, new_val, normalizer in comparisons:
            if normalizer > 0:
                diff = abs(old_val - new_val) / normalizer
                total_diff += min(diff, 1.0)  # Cap at 1.0

        avg_diff = total_diff / len(comparisons)
        stability = 1.0 - avg_diff

        return max(0.0, min(1.0, stability))

    def get_calibration_summary(self) -> dict:
        """Get summary of calibration history."""
        if not self.calibration_history:
            return {"status": "no calibrations performed"}

        stabilities = [
            e.params.parameter_stability
            for e in self.calibration_history
            if e.params.parameter_stability is not None
        ]

        return {
            "symbol": self.symbol,
            "total_calibrations": len(self.calibration_history),
            "first_calibration_bar": self.calibration_history[0].bar_index,
            "last_calibration_bar": self.calibration_history[-1].bar_index,
            "avg_stability": np.mean(stabilities) if stabilities else None,
            "min_stability": np.min(stabilities) if stabilities else None,
            "calibration_dates": [e.date for e in self.calibration_history[-5:]],  # Last 5
        }

    def reset(self) -> None:
        """Reset calibrator state for new backtest."""
        self.last_calibration_bar = -1
        self.current_params = None
        self.calibration_history = []
        self.symbol = None


class WalkForwardCalibrator:
    """
    Walk-forward calibration for strategy validation.

    Splits data into multiple train/test periods and tracks
    parameter evolution and out-of-sample performance.
    """

    def __init__(
        self,
        train_bars: int = 252,
        test_bars: int = 63,
        step_bars: int = 21
    ):
        """
        Initialize walk-forward calibrator.

        Parameters
        ----------
        train_bars : int
            Training window size (default: 252 = 1 year)
        test_bars : int
            Test window size (default: 63 = 3 months)
        step_bars : int
            Step size between windows (default: 21 = monthly)
        """
        self.train_bars = train_bars
        self.test_bars = test_bars
        self.step_bars = step_bars

    def generate_windows(
        self,
        total_bars: int
    ) -> list[tuple[int, int, int, int]]:
        """
        Generate train/test window indices.

        Returns list of (train_start, train_end, test_start, test_end).
        """
        windows = []
        start = 0

        while start + self.train_bars + self.test_bars <= total_bars:
            train_start = start
            train_end = start + self.train_bars
            test_start = train_end
            test_end = min(test_start + self.test_bars, total_bars)

            windows.append((train_start, train_end, test_start, test_end))
            start += self.step_bars

        return windows

    def run_walk_forward(
        self,
        symbol: str,
        bars: np.ndarray,
        dates: Optional[list[str]] = None,
        evaluate_fn: Optional[Callable] = None
    ) -> dict:
        """
        Run full walk-forward calibration.

        Parameters
        ----------
        symbol : str
            Asset symbol
        bars : np.ndarray
            Full OHLCV data
        dates : list[str], optional
            Date strings
        evaluate_fn : callable, optional
            Function to evaluate strategy on test period
            Signature: evaluate_fn(bars, params) -> dict with 'sharpe', 'returns', etc.

        Returns
        -------
        dict
            Walk-forward results including parameter evolution and performance
        """
        windows = self.generate_windows(len(bars))
        calibrator = PercentileCalibrator()

        results = {
            "symbol": symbol,
            "n_windows": len(windows),
            "train_bars": self.train_bars,
            "test_bars": self.test_bars,
            "windows": []
        }

        previous_params = None

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            # Calibrate on training data
            train_bars_window = bars[train_start:train_end]
            train_dates = dates[train_start:train_end] if dates else None

            params = calibrator.calibrate(symbol, train_bars_window, train_dates)

            # Calculate stability
            if previous_params is not None:
                stability = self._calculate_param_change(previous_params, params)
            else:
                stability = 1.0

            window_result = {
                "window_index": i,
                "train_period": f"{dates[train_start] if dates else train_start} to {dates[train_end-1] if dates else train_end-1}",
                "test_period": f"{dates[test_start] if dates else test_start} to {dates[test_end-1] if dates else test_end-1}",
                "params": {
                    "rsi_oversold": params.rsi_oversold,
                    "rsi_overbought": params.rsi_overbought,
                    "er_choppy": params.er_choppy,
                    "er_trending": params.er_trending,
                    "atr_stop_mult": params.atr_stop_multiplier,
                },
                "stability": stability,
            }

            # Evaluate on test data if function provided
            if evaluate_fn is not None:
                test_bars_window = bars[test_start:test_end]
                eval_result = evaluate_fn(test_bars_window, params)
                window_result["test_performance"] = eval_result

            results["windows"].append(window_result)
            previous_params = params

        # Aggregate statistics
        stabilities = [w["stability"] for w in results["windows"]]
        results["summary"] = {
            "avg_stability": np.mean(stabilities),
            "min_stability": np.min(stabilities),
            "std_stability": np.std(stabilities),
        }

        if evaluate_fn and results["windows"]:
            sharpes = [w.get("test_performance", {}).get("sharpe", 0)
                      for w in results["windows"]
                      if "test_performance" in w]
            if sharpes:
                results["summary"]["avg_oos_sharpe"] = np.mean(sharpes)
                results["summary"]["min_oos_sharpe"] = np.min(sharpes)

        return results

    def _calculate_param_change(
        self,
        old: CalibratedParameters,
        new: CalibratedParameters
    ) -> float:
        """Calculate similarity between parameter sets (0-1)."""
        changes = [
            abs(old.rsi_oversold - new.rsi_oversold) / 50,
            abs(old.rsi_overbought - new.rsi_overbought) / 50,
            abs(old.er_choppy - new.er_choppy) / 0.5,
            abs(old.er_trending - new.er_trending) / 0.5,
            abs(old.atr_stop_multiplier - new.atr_stop_multiplier) / 2,
        ]
        avg_change = np.mean(changes)
        return 1.0 - min(1.0, avg_change)

    def print_walk_forward_results(self, results: dict) -> None:
        """Pretty print walk-forward results."""
        print(f"\n{'='*70}")
        print(f"  WALK-FORWARD CALIBRATION: {results['symbol']}")
        print(f"  Windows: {results['n_windows']} | Train: {results['train_bars']} bars | Test: {results['test_bars']} bars")
        print(f"{'='*70}\n")

        print(f"{'Window':<8} {'Period':<25} {'RSI_OS':<8} {'RSI_OB':<8} {'ER_Ch':<8} {'Stab':<8}")
        print("-"*70)

        for w in results["windows"]:
            print(f"{w['window_index']:<8} {w['test_period'][:25]:<25} "
                  f"{w['params']['rsi_oversold']:<8.1f} "
                  f"{w['params']['rsi_overbought']:<8.1f} "
                  f"{w['params']['er_choppy']:<8.3f} "
                  f"{w['stability']:<8.3f}")

        print("-"*70)
        print(f"\nSummary:")
        print(f"  Average Stability: {results['summary']['avg_stability']:.3f}")
        print(f"  Min Stability:     {results['summary']['min_stability']:.3f}")
        if "avg_oos_sharpe" in results["summary"]:
            print(f"  Avg OOS Sharpe:    {results['summary']['avg_oos_sharpe']:.3f}")
        print(f"{'='*70}\n")
