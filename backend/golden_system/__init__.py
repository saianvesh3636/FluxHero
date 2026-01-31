"""
Golden Adaptive System - Complex Three-Tier Trading System

This is the COMPLEX system with:
- Layer 1: Advanced calibration with walk-forward validation
- Layer 2: 4-dimensional Golden Adaptive indicator (fractal, ER, volatility, volume)
- Layer 3: Confidence-weighted strategy with regime-aware trading

This module is SELF-CONTAINED and can be removed without affecting other code.
If backtesting shows this system underperforms, simply delete this folder.

Comparison with Simple System:
------------------------------
Simple System (backend/calibration + backend/computation/golden_ema.py):
- 2 dimensions: fractal + volatility
- Percentile-based threshold calibration
- No confidence scoring

Complex System (this folder):
- 4 dimensions: fractal + efficiency ratio + volatility + volume
- Confidence scoring from dimension agreement
- Walk-forward calibration validation
- Regime-aware position sizing and stops

Usage:
------
    # Layer 2: Compute indicators
    from backend.golden_system import compute_golden_adaptive_indicators

    indicators = compute_golden_adaptive_indicators(bars)
    print(f"Confidence: {indicators['confidence'][-1]:.2f}")
    print(f"Regime: {indicators['regime'][-1]}")

    # Layer 3: Run strategy
    from backend.golden_system import GoldenAdaptiveStrategy, backtest_golden_strategy

    result = backtest_golden_strategy(bars, symbol="SPY")
    print(f"Sharpe: {result.sharpe_ratio:.2f}")

    # Layer 1: Calibrate parameters
    from backend.golden_system import GoldenCalibrator, WalkForwardCalibrator

    calibrator = GoldenCalibrator()
    params = calibrator.calibrate(bars, symbol="SPY")

    # Walk-forward validation
    wf = WalkForwardCalibrator()
    results = wf.run_walk_forward(bars, symbol="SPY")
    wf.print_walk_forward_summary(results)
"""

# Layer 2: Computation
from backend.golden_system.computation import (
    compute_golden_adaptive_indicators,
    generate_golden_signals,
    analyze_dimension_contribution,
    print_dimension_analysis,
    # Individual dimension calculations (for debugging)
    calculate_fractal_alpha,
    calculate_efficiency_ratio_alpha,
    calculate_volatility_alpha,
    calculate_volume_alpha,
    calculate_four_dimension_alpha,
    calculate_golden_ema_from_alpha,
)

# Layer 3: Strategy
from backend.golden_system.strategy import (
    GoldenAdaptiveStrategy,
    GoldenOrder,
    GoldenPosition,
    BacktestResult,
    backtest_golden_strategy,
    print_backtest_result,
)

# Layer 1: Calibration
from backend.golden_system.calibration import (
    GoldenCalibrator,
    GoldenParameters,
    WalkForwardCalibrator,
)

__all__ = [
    # Layer 2
    'compute_golden_adaptive_indicators',
    'generate_golden_signals',
    'analyze_dimension_contribution',
    'print_dimension_analysis',
    'calculate_fractal_alpha',
    'calculate_efficiency_ratio_alpha',
    'calculate_volatility_alpha',
    'calculate_volume_alpha',
    'calculate_four_dimension_alpha',
    'calculate_golden_ema_from_alpha',
    # Layer 3
    'GoldenAdaptiveStrategy',
    'GoldenOrder',
    'GoldenPosition',
    'BacktestResult',
    'backtest_golden_strategy',
    'print_backtest_result',
    # Layer 1
    'GoldenCalibrator',
    'GoldenParameters',
    'WalkForwardCalibrator',
]
