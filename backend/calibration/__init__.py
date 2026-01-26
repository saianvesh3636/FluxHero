"""
Calibration Module - Data-Driven Parameter Estimation

Replace magic numbers with statistically-derived thresholds.

Core Principle: Let the data define what "oversold", "trending", "high volatility"
mean for each specific asset, rather than using universal constants.

Components:
- PercentileCalibrator: Derive thresholds from indicator distributions
- RollingCalibrator: Continuous recalibration during backtest/live trading
- WalkForwardCalibrator: Validation via walk-forward analysis
- ParameterStore: Persistence and caching of calibrated parameters

Usage:
    from backend.calibration import (
        PercentileCalibrator,
        RollingCalibrator,
        CalibratedParameters,
        ParameterStore
    )

    # One-time calibration
    calibrator = PercentileCalibrator()
    params = calibrator.calibrate("SPY", bars)

    # Rolling calibration during backtest
    rolling = RollingCalibrator(lookback_bars=252, recalibrate_every=21)
    for bar_idx in range(len(bars)):
        params = rolling.get_or_calibrate(bars, bar_idx, "SPY")
        # Use params.rsi_oversold instead of magic number 30

    # Walk-forward validation
    wf = WalkForwardCalibrator(train_bars=252, test_bars=63)
    results = wf.run_walk_forward("SPY", bars)
"""

from backend.calibration.parameter_store import (
    CalibratedParameters,
    IndicatorDistribution,
    CalibrationConfig,
    ParameterStore,
)

from backend.calibration.percentile_calibrator import PercentileCalibrator

from backend.calibration.rolling_calibrator import (
    RollingCalibrator,
    WalkForwardCalibrator,
    CalibrationEvent,
)

__all__ = [
    # Data classes
    "CalibratedParameters",
    "IndicatorDistribution",
    "CalibrationConfig",
    "CalibrationEvent",
    # Calibrators
    "PercentileCalibrator",
    "RollingCalibrator",
    "WalkForwardCalibrator",
    # Storage
    "ParameterStore",
]
