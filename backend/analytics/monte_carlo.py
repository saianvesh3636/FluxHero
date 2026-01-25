"""
Monte Carlo Simulation for Strategy Analysis

Provides bootstrap-based Monte Carlo simulations for probabilistic risk analysis.
Uses Numba parallel processing for high-performance simulations.

Features:
- Bootstrap resampling of historical returns
- Parallel execution with Numba
- Probability distributions for outcomes
- Bust probability and goal probability analysis

This module is intended for CLI/backend research use only,
not exposed in frontend.

Usage:
    from backend.analytics.monte_carlo import MonteCarloSimulator

    simulator = MonteCarloSimulator(returns)
    result = simulator.run(n_simulations=10000, n_periods=252)

    print(f"Median return: {result.median_return:.2%}")
    print(f"Bust probability: {result.probability_loss_20pct:.2%}")

Reference: /Users/anvesh/.claude/plans/swirling-tumbling-cloud.md
"""

import logging
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    """Container for Monte Carlo simulation results."""

    # Return statistics
    median_return: float
    mean_return: float
    std_return: float

    # Percentiles (final equity relative to initial)
    percentile_5: float
    percentile_25: float
    percentile_75: float
    percentile_95: float

    # Drawdown statistics
    max_drawdown_median: float
    max_drawdown_95: float

    # Probabilities
    probability_profit: float  # P(final > initial)
    probability_double: float  # P(final > 2x initial)
    probability_loss_10pct: float  # P(final < 0.9x initial)
    probability_loss_20pct: float  # P(final < 0.8x initial)
    probability_loss_50pct: float  # P(final < 0.5x initial)

    # Simulation parameters
    n_simulations: int
    n_periods: int
    initial_capital: float

    # Raw equity paths (optional, can be large)
    equity_paths: np.ndarray | None = None
    final_equities: np.ndarray | None = None


@njit(cache=True, parallel=True)
def _run_monte_carlo_core(
    returns: np.ndarray,
    n_simulations: int,
    n_periods: int,
    initial_capital: float,
    seed: int,
) -> tuple:
    """
    Numba-optimized Monte Carlo core simulation.

    Uses bootstrap resampling with replacement from historical returns.
    Parallel execution across simulations.

    Args:
        returns: Historical returns array
        n_simulations: Number of simulation paths
        n_periods: Number of periods per simulation
        initial_capital: Starting capital
        seed: Random seed for reproducibility

    Returns:
        Tuple of (final_equities, max_drawdowns)
    """
    np.random.seed(seed)
    n_returns = len(returns)

    final_equities = np.zeros(n_simulations, dtype=np.float64)
    max_drawdowns = np.zeros(n_simulations, dtype=np.float64)

    for sim in prange(n_simulations):
        equity = initial_capital
        peak = initial_capital
        max_dd = 0.0

        for period in range(n_periods):
            # Random sample with replacement
            idx = np.random.randint(0, n_returns)
            equity *= 1.0 + returns[idx]

            # Track peak and drawdown
            if equity > peak:
                peak = equity
            if peak > 0:
                dd = (peak - equity) / peak
                if dd > max_dd:
                    max_dd = dd

        final_equities[sim] = equity
        max_drawdowns[sim] = max_dd

    return final_equities, max_drawdowns


@njit(cache=True, parallel=True)
def _run_monte_carlo_with_paths(
    returns: np.ndarray,
    n_simulations: int,
    n_periods: int,
    initial_capital: float,
    seed: int,
) -> tuple:
    """
    Monte Carlo simulation that returns full equity paths.

    More memory intensive but useful for visualization.

    Args:
        returns: Historical returns array
        n_simulations: Number of simulation paths
        n_periods: Number of periods per simulation
        initial_capital: Starting capital
        seed: Random seed

    Returns:
        Tuple of (equity_paths, final_equities, max_drawdowns)
        equity_paths shape: (n_simulations, n_periods + 1)
    """
    np.random.seed(seed)
    n_returns = len(returns)

    equity_paths = np.zeros((n_simulations, n_periods + 1), dtype=np.float64)
    final_equities = np.zeros(n_simulations, dtype=np.float64)
    max_drawdowns = np.zeros(n_simulations, dtype=np.float64)

    for sim in prange(n_simulations):
        equity = initial_capital
        peak = initial_capital
        max_dd = 0.0

        equity_paths[sim, 0] = initial_capital

        for period in range(n_periods):
            idx = np.random.randint(0, n_returns)
            equity *= 1.0 + returns[idx]

            equity_paths[sim, period + 1] = equity

            if equity > peak:
                peak = equity
            if peak > 0:
                dd = (peak - equity) / peak
                if dd > max_dd:
                    max_dd = dd

        final_equities[sim] = equity
        max_drawdowns[sim] = max_dd

    return equity_paths, final_equities, max_drawdowns


class MonteCarloSimulator:
    """
    Monte Carlo simulator for strategy robustness analysis.

    Uses bootstrap resampling of historical returns to simulate
    potential future outcomes.
    """

    def __init__(self, returns: np.ndarray):
        """
        Initialize simulator with historical returns.

        Args:
            returns: Array of historical returns (decimals, e.g., 0.01 = 1%)
        """
        self.returns = np.ascontiguousarray(returns, dtype=np.float64)

        # Validate
        if len(self.returns) < 20:
            logger.warning(
                f"Short return history ({len(self.returns)} periods). "
                "Results may not be statistically significant."
            )

        # Remove any NaN/inf
        self.returns = self.returns[np.isfinite(self.returns)]

    def run(
        self,
        n_simulations: int = 10000,
        n_periods: int = 252,
        initial_capital: float = 100000.0,
        seed: int = 42,
        store_paths: bool = False,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation.

        Args:
            n_simulations: Number of simulation paths (default: 10000)
            n_periods: Number of periods per simulation (default: 252 = 1 year)
            initial_capital: Starting capital (default: 100000)
            seed: Random seed for reproducibility
            store_paths: If True, store full equity paths (memory intensive)

        Returns:
            MonteCarloResult with statistics and probabilities
        """
        logger.info(
            f"Running Monte Carlo: {n_simulations:,} simulations, "
            f"{n_periods} periods, seed={seed}"
        )

        if store_paths:
            equity_paths, final_equities, max_drawdowns = _run_monte_carlo_with_paths(
                self.returns, n_simulations, n_periods, initial_capital, seed
            )
        else:
            equity_paths = None
            final_equities, max_drawdowns = _run_monte_carlo_core(
                self.returns, n_simulations, n_periods, initial_capital, seed
            )

        # Calculate statistics
        final_returns = (final_equities - initial_capital) / initial_capital

        result = MonteCarloResult(
            # Return statistics
            median_return=float(np.median(final_returns)),
            mean_return=float(np.mean(final_returns)),
            std_return=float(np.std(final_returns)),
            # Percentiles
            percentile_5=float(np.percentile(final_returns, 5)),
            percentile_25=float(np.percentile(final_returns, 25)),
            percentile_75=float(np.percentile(final_returns, 75)),
            percentile_95=float(np.percentile(final_returns, 95)),
            # Drawdown statistics
            max_drawdown_median=float(np.median(max_drawdowns)),
            max_drawdown_95=float(np.percentile(max_drawdowns, 95)),
            # Probabilities
            probability_profit=float(np.mean(final_equities > initial_capital)),
            probability_double=float(np.mean(final_equities > 2 * initial_capital)),
            probability_loss_10pct=float(np.mean(final_equities < 0.9 * initial_capital)),
            probability_loss_20pct=float(np.mean(final_equities < 0.8 * initial_capital)),
            probability_loss_50pct=float(np.mean(final_equities < 0.5 * initial_capital)),
            # Parameters
            n_simulations=n_simulations,
            n_periods=n_periods,
            initial_capital=initial_capital,
            # Raw data
            equity_paths=equity_paths,
            final_equities=final_equities,
        )

        logger.info(
            f"Monte Carlo complete: median return={result.median_return:.2%}, "
            f"P(profit)={result.probability_profit:.1%}, "
            f"P(loss>20%)={result.probability_loss_20pct:.1%}"
        )

        return result

    def run_with_goal(
        self,
        goal_return: float = 0.20,
        bust_threshold: float = -0.20,
        n_simulations: int = 10000,
        n_periods: int = 252,
        initial_capital: float = 100000.0,
        seed: int = 42,
    ) -> dict:
        """
        Run Monte Carlo with specific goal and bust thresholds.

        Args:
            goal_return: Target return (e.g., 0.20 = 20% gain)
            bust_threshold: Bust threshold (e.g., -0.20 = 20% loss)
            n_simulations: Number of simulations
            n_periods: Periods per simulation
            initial_capital: Starting capital
            seed: Random seed

        Returns:
            Dictionary with goal and bust probabilities
        """
        result = self.run(n_simulations, n_periods, initial_capital, seed, store_paths=False)

        goal_equity = initial_capital * (1 + goal_return)
        bust_equity = initial_capital * (1 + bust_threshold)

        probability_goal = float(np.mean(result.final_equities >= goal_equity))
        probability_bust = float(np.mean(result.final_equities <= bust_equity))

        return {
            "goal_return": goal_return,
            "bust_threshold": bust_threshold,
            "probability_goal": probability_goal,
            "probability_bust": probability_bust,
            "median_return": result.median_return,
            "mean_return": result.mean_return,
            "percentile_5": result.percentile_5,
            "percentile_95": result.percentile_95,
            "n_simulations": n_simulations,
            "n_periods": n_periods,
        }

    def print_summary(self, result: MonteCarloResult) -> None:
        """Print formatted summary of Monte Carlo results."""
        print("\n" + "=" * 60)
        print("MONTE CARLO SIMULATION RESULTS")
        print("=" * 60)
        print(f"Simulations: {result.n_simulations:,}")
        print(f"Periods: {result.n_periods}")
        print(f"Initial Capital: ${result.initial_capital:,.2f}")
        print()
        print("RETURN DISTRIBUTION")
        print("-" * 40)
        print(f"  Mean Return:    {result.mean_return:>10.2%}")
        print(f"  Median Return:  {result.median_return:>10.2%}")
        print(f"  Std Deviation:  {result.std_return:>10.2%}")
        print()
        print("PERCENTILES")
        print("-" * 40)
        print(f"  5th percentile:  {result.percentile_5:>10.2%}")
        print(f"  25th percentile: {result.percentile_25:>10.2%}")
        print(f"  75th percentile: {result.percentile_75:>10.2%}")
        print(f"  95th percentile: {result.percentile_95:>10.2%}")
        print()
        print("DRAWDOWN ANALYSIS")
        print("-" * 40)
        print(f"  Median Max DD:  {result.max_drawdown_median:>10.2%}")
        print(f"  95th %ile DD:   {result.max_drawdown_95:>10.2%}")
        print()
        print("PROBABILITIES")
        print("-" * 40)
        print(f"  P(Profit):      {result.probability_profit:>10.1%}")
        print(f"  P(Double):      {result.probability_double:>10.1%}")
        print(f"  P(Loss > 10%):  {result.probability_loss_10pct:>10.1%}")
        print(f"  P(Loss > 20%):  {result.probability_loss_20pct:>10.1%}")
        print(f"  P(Loss > 50%):  {result.probability_loss_50pct:>10.1%}")
        print("=" * 60)


def run_monte_carlo_analysis(
    returns: np.ndarray,
    n_simulations: int = 10000,
    n_periods: int = 252,
    initial_capital: float = 100000.0,
    seed: int = 42,
    print_results: bool = True,
) -> MonteCarloResult:
    """
    Convenience function to run Monte Carlo analysis.

    Args:
        returns: Array of historical returns
        n_simulations: Number of simulations
        n_periods: Periods per simulation
        initial_capital: Starting capital
        seed: Random seed
        print_results: If True, print summary to console

    Returns:
        MonteCarloResult object
    """
    simulator = MonteCarloSimulator(returns)
    result = simulator.run(
        n_simulations=n_simulations,
        n_periods=n_periods,
        initial_capital=initial_capital,
        seed=seed,
    )

    if print_results:
        simulator.print_summary(result)

    return result
