"""
Adaptive Market Measure (AMM) Strategy Module

A composite indicator strategy combining SMA deviation, RSI, momentum, and ROC
with configurable weights, z-score normalization, and EMA smoothing.
"""

from backend.amm_system.computation import compute_amm_indicators
from backend.amm_system.strategy import (
    AMMConfig,
    AMMStrategy,
    create_amm_strategy_factory,
)
from backend.amm_system.optimizer import (
    GridSearchConfig,
    AMMGridOptimizer,
    create_amm_optimizer,
)

__all__ = [
    "compute_amm_indicators",
    "AMMConfig",
    "AMMStrategy",
    "create_amm_strategy_factory",
    "GridSearchConfig",
    "AMMGridOptimizer",
    "create_amm_optimizer",
]
