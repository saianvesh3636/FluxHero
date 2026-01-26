"""
Computation Module - High-Performance Technical Indicators

All functions are Numba JIT-compiled for near-C++ performance.

Modules:
- indicators: Core indicators (EMA, RSI, ATR, Bollinger Bands)
- adaptive_ema: KAMA (Kaufman Adaptive Moving Average)
- volatility: Volatility-adaptive smoothing and regime detection
- golden_ema: Golden Adaptive EMA (combines fractal + volatility adaptation)
"""

from backend.computation.indicators import (
    calculate_ema,
    calculate_sma,
    calculate_rsi,
    calculate_atr,
    calculate_true_range,
    calculate_bollinger_bands,
)

from backend.computation.adaptive_ema import (
    calculate_efficiency_ratio,
    calculate_adaptive_smoothing_constant,
    calculate_kama,
    calculate_kama_with_regime_adjustment,
    validate_kama_bounds,
)

from backend.computation.volatility import (
    calculate_atr_ma,
    classify_volatility_state,
    adjust_period_for_volatility,
    detect_volatility_spike,
    calculate_volatility_alpha,
    calculate_adaptive_ema_with_volatility,
    get_stop_loss_multiplier,
    get_position_size_multiplier,
)

from backend.computation.golden_ema import (
    calculate_simple_golden_ema,
    calculate_golden_ema_fast_slow,
    calculate_golden_ema_signals,
    calculate_golden_regime,
    calculate_fractal_alpha,
    calculate_volatility_alpha,
    golden_ema_indicator,
)

__all__ = [
    # Core indicators
    "calculate_ema",
    "calculate_sma",
    "calculate_rsi",
    "calculate_atr",
    "calculate_true_range",
    "calculate_bollinger_bands",
    # KAMA
    "calculate_efficiency_ratio",
    "calculate_adaptive_smoothing_constant",
    "calculate_kama",
    "calculate_kama_with_regime_adjustment",
    "validate_kama_bounds",
    # Volatility
    "calculate_atr_ma",
    "classify_volatility_state",
    "adjust_period_for_volatility",
    "detect_volatility_spike",
    "calculate_volatility_alpha",
    "calculate_adaptive_ema_with_volatility",
    "get_stop_loss_multiplier",
    "get_position_size_multiplier",
    # Golden EMA
    "calculate_simple_golden_ema",
    "calculate_golden_ema_fast_slow",
    "calculate_golden_ema_signals",
    "calculate_golden_regime",
    "calculate_fractal_alpha",
    "golden_ema_indicator",
]
