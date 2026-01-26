"""
Parameter Stability Analyzer - Track Parameter Drift Over Time

Monitors how calibrated parameters change over time to detect:
1. Regime changes (parameters shifting significantly)
2. Potential overfitting (parameters too volatile)
3. Model degradation (parameters becoming unstable)

Usage:
    from backend.validation import ParameterStabilityAnalyzer

    analyzer = ParameterStabilityAnalyzer()
    analyzer.add_calibration(params_jan)
    analyzer.add_calibration(params_feb)
    report = analyzer.generate_report()
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import numpy as np

from backend.calibration import CalibratedParameters


@dataclass
class StabilityReport:
    """Report on parameter stability over time."""

    symbol: str
    n_calibrations: int
    first_calibration: str
    last_calibration: str

    # RSI stability
    rsi_oversold_mean: float
    rsi_oversold_std: float
    rsi_oversold_range: tuple[float, float]
    rsi_oversold_stable: bool  # std < threshold

    rsi_overbought_mean: float
    rsi_overbought_std: float
    rsi_overbought_range: tuple[float, float]
    rsi_overbought_stable: bool

    # ER stability
    er_trending_mean: float
    er_trending_std: float
    er_trending_stable: bool

    er_choppy_mean: float
    er_choppy_std: float
    er_choppy_stable: bool

    # ATR multiplier stability
    atr_stop_mean: float
    atr_stop_std: float
    atr_stop_stable: bool

    # Overall stability score (0-1, higher = more stable)
    overall_stability: float

    # Trend detection (are parameters drifting in one direction?)
    rsi_oversold_trend: str  # "increasing", "decreasing", "stable"
    er_trending_trend: str

    def is_stable(self) -> bool:
        """Check if all parameters are stable."""
        return (
            self.rsi_oversold_stable and
            self.rsi_overbought_stable and
            self.er_trending_stable and
            self.atr_stop_stable
        )


class ParameterStabilityAnalyzer:
    """
    Analyze stability of calibrated parameters over time.

    Tracks parameter values across multiple calibration events
    and calculates stability metrics.
    """

    # Thresholds for determining stability
    RSI_STD_THRESHOLD = 5.0      # RSI stable if std < 5 points
    ER_STD_THRESHOLD = 0.1       # ER stable if std < 0.1
    ATR_STD_THRESHOLD = 0.5      # ATR mult stable if std < 0.5
    TREND_THRESHOLD = 0.7        # Correlation for trend detection

    def __init__(self, symbol: str = ""):
        self.symbol = symbol
        self.calibrations: list[CalibratedParameters] = []
        self.timestamps: list[str] = []

    def add_calibration(self, params: CalibratedParameters) -> None:
        """Add a calibration event."""
        self.calibrations.append(params)
        self.timestamps.append(params.calibration_date)
        if not self.symbol:
            self.symbol = params.symbol

    def clear(self) -> None:
        """Clear all calibrations."""
        self.calibrations = []
        self.timestamps = []

    def generate_report(self) -> StabilityReport:
        """Generate stability report from calibration history."""
        if len(self.calibrations) < 2:
            raise ValueError("Need at least 2 calibrations for stability analysis")

        # Extract parameter series
        rsi_os = np.array([p.rsi_oversold for p in self.calibrations])
        rsi_ob = np.array([p.rsi_overbought for p in self.calibrations])
        er_trend = np.array([p.er_trending for p in self.calibrations])
        er_chop = np.array([p.er_choppy for p in self.calibrations])
        atr_stop = np.array([p.atr_stop_multiplier for p in self.calibrations])

        # Calculate statistics
        rsi_os_std = np.std(rsi_os)
        rsi_ob_std = np.std(rsi_ob)
        er_trend_std = np.std(er_trend)
        er_chop_std = np.std(er_chop)
        atr_stop_std = np.std(atr_stop)

        # Detect trends
        x = np.arange(len(rsi_os))
        rsi_os_corr = np.corrcoef(x, rsi_os)[0, 1] if len(rsi_os) > 2 else 0
        er_trend_corr = np.corrcoef(x, er_trend)[0, 1] if len(er_trend) > 2 else 0

        def trend_label(corr: float) -> str:
            if abs(corr) < self.TREND_THRESHOLD:
                return "stable"
            return "increasing" if corr > 0 else "decreasing"

        # Calculate overall stability (normalized weighted average)
        stability_scores = [
            1 - min(1, rsi_os_std / (self.RSI_STD_THRESHOLD * 2)),
            1 - min(1, rsi_ob_std / (self.RSI_STD_THRESHOLD * 2)),
            1 - min(1, er_trend_std / (self.ER_STD_THRESHOLD * 2)),
            1 - min(1, er_chop_std / (self.ER_STD_THRESHOLD * 2)),
            1 - min(1, atr_stop_std / (self.ATR_STD_THRESHOLD * 2)),
        ]
        overall = np.mean(stability_scores)

        return StabilityReport(
            symbol=self.symbol,
            n_calibrations=len(self.calibrations),
            first_calibration=self.timestamps[0],
            last_calibration=self.timestamps[-1],

            rsi_oversold_mean=float(np.mean(rsi_os)),
            rsi_oversold_std=float(rsi_os_std),
            rsi_oversold_range=(float(np.min(rsi_os)), float(np.max(rsi_os))),
            rsi_oversold_stable=rsi_os_std < self.RSI_STD_THRESHOLD,

            rsi_overbought_mean=float(np.mean(rsi_ob)),
            rsi_overbought_std=float(rsi_ob_std),
            rsi_overbought_range=(float(np.min(rsi_ob)), float(np.max(rsi_ob))),
            rsi_overbought_stable=rsi_ob_std < self.RSI_STD_THRESHOLD,

            er_trending_mean=float(np.mean(er_trend)),
            er_trending_std=float(er_trend_std),
            er_trending_stable=er_trend_std < self.ER_STD_THRESHOLD,

            er_choppy_mean=float(np.mean(er_chop)),
            er_choppy_std=float(er_chop_std),
            er_choppy_stable=er_chop_std < self.ER_STD_THRESHOLD,

            atr_stop_mean=float(np.mean(atr_stop)),
            atr_stop_std=float(atr_stop_std),
            atr_stop_stable=atr_stop_std < self.ATR_STD_THRESHOLD,

            overall_stability=float(overall),

            rsi_oversold_trend=trend_label(rsi_os_corr),
            er_trending_trend=trend_label(er_trend_corr),
        )

    def print_report(self, report: Optional[StabilityReport] = None) -> None:
        """Print formatted stability report."""
        if report is None:
            report = self.generate_report()

        print(f"\n{'='*70}")
        print(f"  PARAMETER STABILITY REPORT: {report.symbol}")
        print(f"  Calibrations: {report.n_calibrations}")
        print(f"  Period: {report.first_calibration} to {report.last_calibration}")
        print(f"{'='*70}\n")

        print(f"{'Parameter':<20} {'Mean':<10} {'Std':<10} {'Range':<20} {'Stable':<8}")
        print("-"*70)

        stable_icon = lambda s: "Yes" if s else "NO"

        print(f"{'RSI Oversold':<20} {report.rsi_oversold_mean:<10.1f} {report.rsi_oversold_std:<10.2f} "
              f"{report.rsi_oversold_range[0]:.1f}-{report.rsi_oversold_range[1]:.1f}{'':>10} "
              f"{stable_icon(report.rsi_oversold_stable):<8}")

        print(f"{'RSI Overbought':<20} {report.rsi_overbought_mean:<10.1f} {report.rsi_overbought_std:<10.2f} "
              f"{report.rsi_overbought_range[0]:.1f}-{report.rsi_overbought_range[1]:.1f}{'':>10} "
              f"{stable_icon(report.rsi_overbought_stable):<8}")

        print(f"{'ER Trending':<20} {report.er_trending_mean:<10.3f} {report.er_trending_std:<10.3f} "
              f"{'':>20} {stable_icon(report.er_trending_stable):<8}")

        print(f"{'ER Choppy':<20} {report.er_choppy_mean:<10.3f} {report.er_choppy_std:<10.3f} "
              f"{'':>20} {stable_icon(report.er_choppy_stable):<8}")

        print(f"{'ATR Stop Mult':<20} {report.atr_stop_mean:<10.2f} {report.atr_stop_std:<10.3f} "
              f"{'':>20} {stable_icon(report.atr_stop_stable):<8}")

        print("-"*70)
        print(f"\n[TRENDS]")
        print(f"  RSI Oversold: {report.rsi_oversold_trend}")
        print(f"  ER Trending:  {report.er_trending_trend}")

        print(f"\n[OVERALL]")
        print(f"  Stability Score: {report.overall_stability:.2f} (0-1, higher=more stable)")
        print(f"  All Parameters Stable: {'Yes' if report.is_stable() else 'NO'}")

        if not report.is_stable():
            print(f"\n[WARNING] Some parameters are unstable!")
            print(f"  Consider: Longer calibration lookback or investigating regime changes.")

        print(f"\n{'='*70}\n")
