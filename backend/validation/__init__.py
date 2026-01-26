"""
Validation Module - Cross-Asset and Regime Robustness Testing

Ensures parameters and strategies are not overfit to specific assets or time periods.

Components:
- CrossAssetValidator: Test parameters across multiple assets
- RegimeValidator: Test across different market regimes
- ParameterStabilityAnalyzer: Track parameter drift over time

Usage:
    from backend.validation import CrossAssetValidator

    validator = CrossAssetValidator(assets=["SPY", "QQQ", "IWM", "GLD", "TLT"])
    results = await validator.validate_parameters(params, strategy_class)
"""

from backend.validation.cross_asset_validator import (
    CrossAssetValidator,
    ValidationResult,
)

from backend.validation.parameter_stability import (
    ParameterStabilityAnalyzer,
    StabilityReport,
)

__all__ = [
    "CrossAssetValidator",
    "ValidationResult",
    "ParameterStabilityAnalyzer",
    "StabilityReport",
]
