"""
FluxHero Analytics Module

Provides advanced performance metrics and reporting using QuantStats with Numba optimization.

Modules:
- numba_stats: JIT-compiled Tier 1 metrics (Sortino, Calmar, VaR, etc.)
- quantstats_wrapper: QuantStats adapter with caching
- report_generator: HTML tearsheet generation
- monte_carlo: Monte Carlo simulations for risk analysis

Usage:
    from backend.analytics import QuantStatsAdapter, TearsheetGenerator

    adapter = QuantStatsAdapter(returns, benchmark_returns)
    metrics = adapter.get_all_metrics()

    generator = TearsheetGenerator()
    report_path = generator.generate_tearsheet(returns, timestamps)

Reference: /Users/anvesh/.claude/plans/swirling-tumbling-cloud.md
"""

from backend.analytics.numba_stats import (
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_profit_factor,
    calculate_value_at_risk,
    calculate_cvar,
    calculate_kelly_criterion,
    calculate_recovery_factor,
    calculate_ulcer_index,
    calculate_alpha_beta,
    calculate_consecutive_wins_losses,
    calculate_skewness,
    calculate_kurtosis,
    calculate_tail_ratio,
    calculate_information_ratio,
    calculate_r_squared,
    calculate_tier1_metrics,
)

from backend.analytics.quantstats_wrapper import (
    QuantStatsAdapter,
    create_adapter_from_backtest,
    fetch_benchmark_returns,
    get_periods_per_year,
    get_market_config,
    MarketType,
    MarketConfig,
    INTERVAL_PERIODS_PER_YEAR,
    MARKET_CONFIGS,
    DEFAULT_MARKET,
)

from backend.analytics.report_generator import (
    TearsheetGenerator,
    get_generator,
)

from backend.analytics.monte_carlo import (
    MonteCarloSimulator,
    MonteCarloResult,
    run_monte_carlo_analysis,
)

__all__ = [
    # Numba stats
    "calculate_sortino_ratio",
    "calculate_calmar_ratio",
    "calculate_profit_factor",
    "calculate_value_at_risk",
    "calculate_cvar",
    "calculate_kelly_criterion",
    "calculate_recovery_factor",
    "calculate_ulcer_index",
    "calculate_alpha_beta",
    "calculate_consecutive_wins_losses",
    "calculate_skewness",
    "calculate_kurtosis",
    "calculate_tail_ratio",
    "calculate_information_ratio",
    "calculate_r_squared",
    "calculate_tier1_metrics",
    # QuantStats wrapper
    "QuantStatsAdapter",
    "create_adapter_from_backtest",
    "fetch_benchmark_returns",
    "get_periods_per_year",
    "get_market_config",
    "MarketType",
    "MarketConfig",
    "INTERVAL_PERIODS_PER_YEAR",
    "MARKET_CONFIGS",
    "DEFAULT_MARKET",
    # Report generator
    "TearsheetGenerator",
    "get_generator",
    # Monte Carlo
    "MonteCarloSimulator",
    "MonteCarloResult",
    "run_monte_carlo_analysis",
]
