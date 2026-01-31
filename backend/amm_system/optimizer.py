"""
Grid search optimizer for AMM strategy parameters.

Optimizes indicator weights and trading thresholds using walk-forward compatible interface.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Any
from itertools import product

from backend.backtesting.engine import BacktestConfig, BacktestEngine
from backend.amm_system.strategy import AMMConfig, AMMStrategy


@dataclass
class GridSearchConfig:
    """Configuration for grid search optimization.

    Reduced search space to avoid overfitting:
    - Fewer weight values (5 instead of 7)
    - Fewer threshold options (3 instead of 4)
    - Fewer EMA spans (2 instead of 3)

    Total combinations: ~50-100 instead of 1000+
    """

    # Weight values to test (must generate combinations that sum to 1.0)
    # Reduced from 7 to 5 values to limit search space
    weight_values: tuple[float, ...] = (0.15, 0.20, 0.25, 0.30, 0.35)

    # Entry thresholds to test
    # Reduced to 3 sensible values
    entry_thresholds: tuple[float, ...] = (0.75, 1.0, 1.5)

    # EMA spans to test
    # Only 2 options: responsive (10) vs smooth (20)
    ema_spans: tuple[int, ...] = (10, 20)

    # Target metric for optimization
    target_metric: str = "sharpe_ratio"

    # Weight sum tolerance
    weight_sum_tolerance: float = 0.01  # Slightly looser tolerance


class AMMGridOptimizer:
    """
    Grid search optimizer for AMM strategy.

    Searches over valid weight combinations (sum to 1.0) and trading parameters.
    """

    def __init__(
        self,
        config: GridSearchConfig | None = None,
        base_strategy_config: AMMConfig | None = None,
    ):
        """
        Initialize the optimizer.

        Parameters
        ----------
        config : GridSearchConfig, optional
            Grid search configuration
        base_strategy_config : AMMConfig, optional
            Base strategy config (non-optimized params use these values)
        """
        self.config = config or GridSearchConfig()
        self.base_strategy_config = base_strategy_config or AMMConfig()

    def generate_weight_combinations(self) -> list[tuple[float, float, float, float]]:
        """
        Generate all valid weight combinations that sum to 1.0.

        Returns list of (w_sma, w_rsi, w_mom, w_roc) tuples.
        """
        valid_combinations = []
        weights = self.config.weight_values
        tolerance = self.config.weight_sum_tolerance

        for combo in product(weights, repeat=4):
            if abs(sum(combo) - 1.0) <= tolerance:
                valid_combinations.append(combo)

        return valid_combinations

    def generate_parameter_grid(self) -> list[dict[str, Any]]:
        """Generate full parameter grid for optimization."""
        weight_combos = self.generate_weight_combinations()
        param_grid = []

        for w_sma, w_rsi, w_mom, w_boll in weight_combos:
            for entry_thresh in self.config.entry_thresholds:
                for ema_span in self.config.ema_spans:
                    param_grid.append(
                        {
                            "w_sma": w_sma,
                            "w_rsi": w_rsi,
                            "w_mom": w_mom,
                            "w_boll": w_boll,
                            "entry_threshold": entry_thresh,
                            "ema_span": ema_span,
                        }
                    )

        return param_grid

    def optimize(
        self,
        train_bars: np.ndarray,
        backtest_config: BacktestConfig,
        symbol: str = "SPY",
    ) -> dict[str, Any]:
        """
        Run grid search optimization on training data.

        Parameters
        ----------
        train_bars : np.ndarray
            Training OHLCV data
        backtest_config : BacktestConfig
            Backtest configuration
        symbol : str
            Trading symbol

        Returns
        -------
        dict
            Best parameters found during optimization
        """
        param_grid = self.generate_parameter_grid()
        best_metric = -np.inf
        best_params = {}

        engine = BacktestEngine(backtest_config)

        for params in param_grid:
            # Create strategy config with current params
            merged = self.base_strategy_config.to_dict()
            merged.update(params)
            strategy_config = AMMConfig.from_dict(merged)

            # Skip if not enough bars for warmup
            if len(train_bars) < AMMStrategy.WARMUP_BARS + 50:
                continue

            # Create strategy
            strategy = AMMStrategy(
                bars=train_bars,
                initial_capital=backtest_config.initial_capital,
                config=strategy_config,
                symbol=symbol,
            )

            # Run backtest
            try:
                state = engine.run(
                    bars=train_bars,
                    strategy_func=strategy.get_orders,
                    symbol=symbol,
                )

                # Calculate target metric
                metric_value = self._calculate_metric(state, backtest_config)

                if metric_value > best_metric:
                    best_metric = metric_value
                    best_params = params.copy()

            except Exception:
                # Skip invalid parameter combinations
                continue

        return best_params

    def _calculate_metric(
        self, state: Any, config: BacktestConfig
    ) -> float:
        """Calculate the target optimization metric from backtest state."""
        if len(state.equity_curve) < 2:
            return -np.inf

        equity = np.array(state.equity_curve, dtype=np.float64)

        # Calculate returns
        returns = np.diff(equity) / equity[:-1]

        if len(returns) < 10:
            return -np.inf

        metric_name = self.config.target_metric

        if metric_name == "sharpe_ratio":
            return self._sharpe_ratio(returns, config.risk_free_rate)
        elif metric_name == "sortino_ratio":
            return self._sortino_ratio(returns, config.risk_free_rate)
        elif metric_name == "total_return":
            return (equity[-1] - equity[0]) / equity[0]
        elif metric_name == "calmar_ratio":
            return self._calmar_ratio(equity, config.risk_free_rate)
        else:
            # Default to Sharpe
            return self._sharpe_ratio(returns, config.risk_free_rate)

    def _sharpe_ratio(
        self, returns: np.ndarray, risk_free_rate: float, periods_per_year: int = 252
    ) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return -np.inf

        daily_rf = risk_free_rate / periods_per_year
        excess_returns = returns - daily_rf
        return (np.mean(excess_returns) / np.std(returns)) * np.sqrt(periods_per_year)

    def _sortino_ratio(
        self, returns: np.ndarray, risk_free_rate: float, periods_per_year: int = 252
    ) -> float:
        """Calculate annualized Sortino ratio."""
        daily_rf = risk_free_rate / periods_per_year
        excess_returns = returns - daily_rf

        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return np.inf if np.mean(excess_returns) > 0 else 0.0

        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return np.inf if np.mean(excess_returns) > 0 else 0.0

        return (np.mean(excess_returns) / downside_std) * np.sqrt(periods_per_year)

    def _calmar_ratio(
        self, equity: np.ndarray, risk_free_rate: float, periods_per_year: int = 252
    ) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown)."""
        total_return = (equity[-1] - equity[0]) / equity[0]
        n_periods = len(equity)
        annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1

        # Calculate max drawdown
        peak = equity[0]
        max_dd = 0.0
        for value in equity:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd

        if max_dd == 0:
            return np.inf if annualized_return > 0 else 0.0

        return annualized_return / max_dd


def create_amm_optimizer(
    config: GridSearchConfig | None = None,
    base_strategy_config: AMMConfig | None = None,
) -> callable:
    """
    Create an optimizer function compatible with walk-forward testing.

    Parameters
    ----------
    config : GridSearchConfig, optional
        Grid search configuration
    base_strategy_config : AMMConfig, optional
        Base strategy configuration

    Returns
    -------
    callable
        Optimizer function with signature:
        (train_bars, backtest_config) -> dict[str, Any]
    """
    optimizer = AMMGridOptimizer(config, base_strategy_config)

    def optimize_func(
        train_bars: np.ndarray,
        backtest_config: BacktestConfig,
    ) -> dict[str, Any]:
        """Optimize strategy parameters on training data."""
        return optimizer.optimize(train_bars, backtest_config)

    return optimize_func
