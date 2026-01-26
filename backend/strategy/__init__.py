"""
Strategy Module - Trading Strategies for FluxHero

Provides:
- DualModeBacktestStrategy: Original strategy with fixed thresholds
- CalibratedBacktestStrategy: Strategy with calibrated (data-driven) thresholds
- Signal generators for trend-following and mean-reversion
- Regime detection and classification
- Noise filtering

Usage:
    # Original strategy (fixed magic numbers)
    from backend.strategy import DualModeBacktestStrategy

    # Calibrated strategy (recommended)
    from backend.strategy import CalibratedBacktestStrategy
    from backend.calibration import PercentileCalibrator

    calibrator = PercentileCalibrator()
    params = calibrator.calibrate("SPY", bars)
    strategy = CalibratedBacktestStrategy(bars, "SPY", params)
"""

from backend.strategy.backtest_strategy import DualModeBacktestStrategy
from backend.strategy.calibrated_backtest_strategy import CalibratedBacktestStrategy

from backend.strategy.dual_mode import (
    SIGNAL_NONE,
    SIGNAL_LONG,
    SIGNAL_SHORT,
    SIGNAL_EXIT_LONG,
    SIGNAL_EXIT_SHORT,
    MODE_TREND_FOLLOWING,
    MODE_MEAN_REVERSION,
    MODE_NEUTRAL,
    generate_trend_following_signals,
    generate_mean_reversion_signals,
    calculate_trailing_stop,
    calculate_fixed_stop_loss,
    calculate_position_size,
)

from backend.strategy.regime_detector import (
    REGIME_MEAN_REVERSION,
    REGIME_NEUTRAL,
    REGIME_STRONG_TREND,
    VOL_LOW,
    VOL_NORMAL,
    VOL_HIGH,
    detect_regime,
    calculate_adx,
    calculate_linear_regression,
    classify_trend_regime,
    classify_volatility_regime,
)

from backend.strategy.signal_generator import (
    SignalExplanation,
    SignalGenerator,
)

__all__ = [
    # Strategies
    "DualModeBacktestStrategy",
    "CalibratedBacktestStrategy",
    # Signal constants
    "SIGNAL_NONE",
    "SIGNAL_LONG",
    "SIGNAL_SHORT",
    "SIGNAL_EXIT_LONG",
    "SIGNAL_EXIT_SHORT",
    # Mode constants
    "MODE_TREND_FOLLOWING",
    "MODE_MEAN_REVERSION",
    "MODE_NEUTRAL",
    # Regime constants
    "REGIME_MEAN_REVERSION",
    "REGIME_NEUTRAL",
    "REGIME_STRONG_TREND",
    "VOL_LOW",
    "VOL_NORMAL",
    "VOL_HIGH",
    # Functions
    "generate_trend_following_signals",
    "generate_mean_reversion_signals",
    "detect_regime",
    "calculate_adx",
    "SignalExplanation",
    "SignalGenerator",
]
