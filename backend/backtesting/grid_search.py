"""
Grid search optimizer for Dual-Mode strategy parameters.

Optimizes mean-reversion and trend-following thresholds using walk-forward compatible interface.
"""

import numpy as np
from dataclasses import dataclass
from itertools import product
from typing import Any

from backend.backtesting.engine import (
    BacktestConfig,
    BacktestEngine,
    Order,
    OrderSide,
    OrderType,
    Position,
    PositionSide,
)
from backend.computation.adaptive_ema import calculate_kama_with_regime_adjustment
from backend.computation.indicators import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_rsi,
)
from backend.strategy.dual_mode import (
    SIGNAL_EXIT_LONG,
    SIGNAL_LONG,
    SIGNAL_NONE,
    calculate_position_size,
    generate_mean_reversion_signals,
    generate_trend_following_signals,
)
from backend.strategy.regime_detector import (
    REGIME_MEAN_REVERSION,
    REGIME_STRONG_TREND,
    detect_regime,
)


@dataclass
class DualModeGridConfig:
    """Configuration for dual-mode grid search optimization.

    Reduced search space to avoid overfitting:
    - 3 RSI oversold values
    - 3 RSI overbought values
    - 3 BB entry thresholds
    - 3 ATR entry multipliers
    - 3 stop ATR multipliers

    Total combinations: 3^5 = 243 (manageable)
    """

    # RSI thresholds
    rsi_oversold: tuple[float, ...] = (30.0, 35.0, 40.0)
    rsi_overbought: tuple[float, ...] = (60.0, 65.0, 70.0)

    # Bollinger Band entry threshold (%B below this triggers entry)
    bb_entry_threshold: tuple[float, ...] = (0.10, 0.15, 0.20)

    # ATR multiplier for trend-following entry bands
    atr_entry_mult: tuple[float, ...] = (0.3, 0.5, 0.7)

    # ATR multiplier for stop loss
    stop_atr_mult: tuple[float, ...] = (1.5, 2.0, 2.5)

    # Target metric for optimization
    target_metric: str = "sharpe_ratio"

    # Use OR logic (True) or AND logic (False) for mean-reversion entry
    use_or_logic: bool = True


class DualModeGridOptimizer:
    """
    Grid search optimizer for Dual-Mode strategy.

    Searches over RSI thresholds, Bollinger %B entry, and ATR multipliers.
    """

    # Minimum bars needed for indicator warmup
    WARMUP_BARS = 50

    def __init__(
        self,
        config: DualModeGridConfig | None = None,
    ):
        """
        Initialize the optimizer.

        Parameters
        ----------
        config : DualModeGridConfig, optional
            Grid search configuration
        """
        self.config = config or DualModeGridConfig()

    def generate_parameter_grid(self) -> list[dict[str, Any]]:
        """Generate full parameter grid for optimization."""
        param_grid = []

        for rsi_os in self.config.rsi_oversold:
            for rsi_ob in self.config.rsi_overbought:
                for bb_thresh in self.config.bb_entry_threshold:
                    for atr_entry in self.config.atr_entry_mult:
                        for stop_atr in self.config.stop_atr_mult:
                            param_grid.append({
                                "rsi_oversold": rsi_os,
                                "rsi_overbought": rsi_ob,
                                "bb_entry_threshold": bb_thresh,
                                "atr_entry_mult": atr_entry,
                                "stop_atr_mult": stop_atr,
                                "use_or_logic": self.config.use_or_logic,
                            })

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
            Training OHLCV data, shape (N, 5)
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
        best_params: dict[str, Any] = {}

        # Skip if not enough bars for warmup
        if len(train_bars) < self.WARMUP_BARS + 50:
            return best_params

        engine = BacktestEngine(backtest_config)

        for params in param_grid:
            try:
                # Create strategy with current params
                strategy = _DualModeStrategyWithParams(train_bars, params)

                # Run backtest
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

    def _calculate_metric(self, state: Any, config: BacktestConfig) -> float:
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


class _DualModeStrategyWithParams:
    """
    Internal strategy class that accepts parameter overrides.

    Used by the grid optimizer to test different parameter combinations.
    """

    def __init__(self, bars: np.ndarray, params: dict[str, Any]):
        """
        Initialize strategy with full dataset and parameter overrides.

        Args:
            bars: OHLCV bars array, shape (N, 5)
            params: Strategy parameters from grid search
        """
        self.params = params

        # Extract OHLC data (ensure contiguous arrays for Numba)
        high_prices = np.ascontiguousarray(bars[:, 1])
        low_prices = np.ascontiguousarray(bars[:, 2])
        close_prices = np.ascontiguousarray(bars[:, 3])

        # Calculate technical indicators
        self.kama, er, regime = calculate_kama_with_regime_adjustment(close_prices)
        self.atr = calculate_atr(high_prices, low_prices, close_prices)
        rsi = calculate_rsi(close_prices)
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close_prices)

        # Detect market regimes
        from backend.computation.volatility import calculate_atr_ma
        atr_ma = calculate_atr_ma(self.atr)

        regime_data = detect_regime(
            high=high_prices,
            low=low_prices,
            close=close_prices,
            atr=self.atr,
            atr_ma=atr_ma,
            apply_persistence=True,
        )
        self.trend_regime = regime_data['trend_regime']

        # Generate signals with parameterized thresholds
        self.trend_signals = generate_trend_following_signals(
            prices=close_prices,
            kama=self.kama,
            atr=self.atr,
            entry_multiplier=params.get("atr_entry_mult", 0.5),
            exit_multiplier=params.get("atr_entry_mult", 0.5) * 0.6,  # Exit at 60% of entry
        )

        self.mr_signals = generate_mean_reversion_signals(
            prices=close_prices,
            rsi=rsi,
            bollinger_lower=bb_lower,
            bollinger_middle=bb_middle,
            bollinger_upper=bb_upper,
            rsi_oversold=params.get("rsi_oversold", 35.0),
            rsi_overbought=params.get("rsi_overbought", 65.0),
            bb_entry_threshold=params.get("bb_entry_threshold", 0.15),
            use_or_logic=params.get("use_or_logic", True),
        )

        self.close_prices = close_prices
        self.stop_atr_mult = params.get("stop_atr_mult", 2.0)

    def get_orders(
        self,
        bars: np.ndarray,
        current_index: int,
        position: Position | None
    ) -> list[Order]:
        """
        Generate orders based on current bar and position.

        Args:
            bars: Full OHLCV array
            current_index: Current bar index
            position: Current open position (None if flat)

        Returns:
            List of orders to place
        """
        orders: list[Order] = []

        # Skip if indicators not ready
        if current_index < 50:
            return orders

        current_close = self.close_prices[current_index]
        current_atr = self.atr[current_index]
        current_regime = self.trend_regime[current_index]

        # Skip if ATR not valid
        if np.isnan(current_atr) or current_atr == 0:
            return orders

        # Select active strategy based on regime
        if current_regime == REGIME_STRONG_TREND:
            active_signal = self.trend_signals[current_index]
            risk_pct = 0.01  # 1% risk
        elif current_regime == REGIME_MEAN_REVERSION:
            active_signal = self.mr_signals[current_index]
            risk_pct = 0.0075  # 0.75% risk
        else:  # NEUTRAL
            # Require both strategies to agree
            if (self.trend_signals[current_index] == self.mr_signals[current_index] and
                self.trend_signals[current_index] != SIGNAL_NONE):
                active_signal = self.trend_signals[current_index]
            else:
                active_signal = SIGNAL_NONE
            risk_pct = 0.007  # 0.7% risk

        # Process signals
        if position is None:  # Flat - look for entry
            if active_signal == SIGNAL_LONG:
                # Calculate stop loss using parameterized ATR multiplier
                if current_regime == REGIME_STRONG_TREND:
                    stop_price = current_close - (self.stop_atr_mult * current_atr)
                else:
                    stop_price = current_close * 0.97  # 3% stop for mean reversion

                capital = 100000.0  # Estimate

                shares = calculate_position_size(
                    capital=capital,
                    entry_price=current_close,
                    stop_price=stop_price,
                    risk_pct=risk_pct,
                    is_long=True,
                )

                if shares > 0:
                    order = Order(
                        bar_index=current_index,
                        symbol="SPY",
                        side=OrderSide.BUY,
                        shares=int(shares),
                        order_type=OrderType.MARKET,
                    )
                    orders.append(order)

        else:  # In position - look for exit
            if position.side == PositionSide.LONG and active_signal == SIGNAL_EXIT_LONG:
                order = Order(
                    bar_index=current_index,
                    symbol="SPY",
                    side=OrderSide.SELL,
                    shares=position.shares,
                    order_type=OrderType.MARKET,
                )
                orders.append(order)

        return orders


def create_dual_mode_optimizer(
    config: DualModeGridConfig | None = None,
) -> callable:
    """
    Create an optimizer function compatible with walk-forward testing.

    Parameters
    ----------
    config : DualModeGridConfig, optional
        Grid search configuration

    Returns
    -------
    callable
        Optimizer function with signature:
        (train_bars, backtest_config) -> dict[str, Any]
    """
    optimizer = DualModeGridOptimizer(config)

    def optimize_func(
        train_bars: np.ndarray,
        backtest_config: BacktestConfig,
    ) -> dict[str, Any]:
        """Optimize strategy parameters on training data."""
        return optimizer.optimize(train_bars, backtest_config)

    return optimize_func


def create_dual_mode_strategy_factory() -> callable:
    """
    Create a strategy factory function compatible with walk-forward testing.

    Returns
    -------
    callable
        Strategy factory with signature:
        (bars, initial_capital, params) -> strategy_func
    """

    def factory(
        bars: np.ndarray,
        initial_capital: float,
        params: dict[str, Any],
    ) -> callable:
        """Create a strategy function with the given parameters."""
        strategy = _DualModeStrategyWithParams(bars, params)
        return strategy.get_orders

    return factory
