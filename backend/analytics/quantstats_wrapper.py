"""
QuantStats Integration Wrapper for FluxHero

Provides unified access to performance metrics by combining:
1. Numba-optimized Tier 1 metrics (fast, hot path)
2. QuantStats library for comprehensive analytics (60+ metrics)

Features:
- Automatic pandas/numpy conversion
- LRU caching for expensive operations
- Graceful fallback between implementations
- Uses provider system for benchmark data (supports multiple data sources)

Usage:
    adapter = QuantStatsAdapter(returns, benchmark_returns, risk_free_rate=0.04)

    # Get Numba-optimized metrics (fast)
    tier1 = adapter.get_tier1_metrics()

    # Get all metrics including QuantStats (comprehensive)
    all_metrics = adapter.get_all_metrics()

    # Get full QuantStats metrics only
    qs_metrics = adapter.get_quantstats_metrics()

Reference: /Users/anvesh/.claude/plans/swirling-tumbling-cloud.md
"""

import asyncio
import logging
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import quantstats as qs

from backend.analytics.numba_stats import calculate_tier1_metrics

logger = logging.getLogger(__name__)


class QuantStatsAdapter:
    """
    Adapter class for unified QuantStats + Numba metrics access.

    Combines high-performance Numba implementations for frequently-used
    metrics with QuantStats library for comprehensive analytics.
    """

    def __init__(
        self,
        returns: np.ndarray,
        equity_curve: np.ndarray | None = None,
        pnls: np.ndarray | None = None,
        benchmark_returns: np.ndarray | None = None,
        risk_free_rate: float = 0.04,
        periods_per_year: int = 252,
        timestamps: np.ndarray | None = None,
    ):
        """
        Initialize adapter with return series.

        Args:
            returns: Strategy returns array (decimals, e.g., 0.01 = 1%)
            equity_curve: Equity values array (optional, for drawdown metrics)
            pnls: Trade P&Ls array (optional, for trade statistics)
            benchmark_returns: Benchmark returns for comparison (optional)
            risk_free_rate: Annual risk-free rate (default: 4%)
            periods_per_year: Annualization factor (252 for daily)
            timestamps: Optional datetime timestamps for returns
        """
        # Store numpy arrays
        self._returns = np.ascontiguousarray(returns, dtype=np.float64)
        self._equity_curve = (
            np.ascontiguousarray(equity_curve, dtype=np.float64)
            if equity_curve is not None
            else self._compute_equity_from_returns(returns)
        )
        self._pnls = (
            np.ascontiguousarray(pnls, dtype=np.float64)
            if pnls is not None
            else np.array([], dtype=np.float64)
        )
        self._benchmark = (
            np.ascontiguousarray(benchmark_returns, dtype=np.float64)
            if benchmark_returns is not None
            else None
        )
        self._rf = risk_free_rate
        self._periods = periods_per_year

        # Create pandas Series for QuantStats
        if timestamps is not None:
            index = pd.to_datetime(timestamps)
        else:
            index = pd.date_range(end=pd.Timestamp.now(), periods=len(returns), freq="D")

        self._returns_series = pd.Series(returns, index=index)
        self._benchmark_series = (
            pd.Series(benchmark_returns, index=index)
            if benchmark_returns is not None
            else None
        )

        # Cache for computed metrics
        self._cache: dict[str, Any] = {}

    def _compute_equity_from_returns(
        self, returns: np.ndarray, initial_capital: float = 100000.0
    ) -> np.ndarray:
        """Compute equity curve from returns if not provided."""
        equity = np.zeros(len(returns) + 1, dtype=np.float64)
        equity[0] = initial_capital
        for i, r in enumerate(returns):
            equity[i + 1] = equity[i] * (1 + r)
        return equity

    def get_tier1_metrics(self) -> dict[str, float]:
        """
        Get high-priority metrics using Numba implementations.

        Returns:
            Dictionary with Tier 1 metrics:
            - sortino_ratio
            - calmar_ratio
            - profit_factor
            - value_at_risk_95
            - cvar_95
            - kelly_criterion
            - recovery_factor
            - ulcer_index
            - max_consecutive_wins
            - max_consecutive_losses
            - skewness
            - kurtosis
            - tail_ratio
            - alpha, beta (if benchmark provided)
            - information_ratio (if benchmark provided)
            - r_squared (if benchmark provided)
        """
        cache_key = "tier1"
        if cache_key in self._cache:
            return self._cache[cache_key]

        metrics = calculate_tier1_metrics(
            returns=self._returns,
            equity_curve=self._equity_curve,
            pnls=self._pnls,
            benchmark_returns=self._benchmark,
            risk_free_rate=self._rf,
            periods_per_year=self._periods,
        )

        self._cache[cache_key] = metrics
        return metrics

    def get_quantstats_metrics(self) -> dict[str, float]:
        """
        Get all QuantStats metrics (60+).

        This uses the QuantStats library directly for comprehensive analytics.
        Results are cached for performance.

        Returns:
            Dictionary with all QuantStats metrics
        """
        cache_key = "quantstats"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Get basic metrics
            metrics = {
                # Return metrics
                "cagr": qs.stats.cagr(self._returns_series),
                "total_return": qs.stats.comp(self._returns_series),
                "avg_return": float(self._returns_series.mean()),
                "volatility": qs.stats.volatility(self._returns_series),
                "best_day": qs.stats.best(self._returns_series),
                "worst_day": qs.stats.worst(self._returns_series),
                # Risk metrics
                "sharpe": qs.stats.sharpe(self._returns_series, rf=self._rf),
                "sortino": qs.stats.sortino(self._returns_series, rf=self._rf),
                "calmar": qs.stats.calmar(self._returns_series),
                "max_drawdown": qs.stats.max_drawdown(self._returns_series),
                "ulcer_index": qs.stats.ulcer_index(self._returns_series),
                # VaR metrics
                "var_95": qs.stats.var(self._returns_series, sigma=1.65),
                "cvar_95": qs.stats.cvar(self._returns_series, sigma=1.65),
                # Trade statistics
                "win_rate": qs.stats.win_rate(self._returns_series),
                "profit_factor": qs.stats.profit_factor(self._returns_series),
                "payoff_ratio": qs.stats.payoff_ratio(self._returns_series),
                "win_loss_ratio": qs.stats.win_loss_ratio(self._returns_series),
                "avg_win": qs.stats.avg_win(self._returns_series),
                "avg_loss": qs.stats.avg_loss(self._returns_series),
                "consecutive_wins": qs.stats.consecutive_wins(self._returns_series),
                "consecutive_losses": qs.stats.consecutive_losses(self._returns_series),
                # Distribution
                "skew": qs.stats.skew(self._returns_series),
                "kurtosis": qs.stats.kurtosis(self._returns_series),
                "tail_ratio": qs.stats.tail_ratio(self._returns_series),
                # Other
                "kelly_criterion": qs.stats.kelly_criterion(self._returns_series),
                "risk_of_ruin": qs.stats.risk_of_ruin(self._returns_series),
                "recovery_factor": qs.stats.recovery_factor(self._returns_series),
                "outlier_win_ratio": qs.stats.outlier_win_ratio(self._returns_series),
                "outlier_loss_ratio": qs.stats.outlier_loss_ratio(self._returns_series),
            }

            # Add benchmark comparison metrics if benchmark provided
            if self._benchmark_series is not None:
                try:
                    alpha, beta = qs.stats.greeks(
                        self._returns_series, self._benchmark_series
                    )
                    metrics["alpha"] = alpha
                    metrics["beta"] = beta
                    metrics["r_squared"] = qs.stats.r_squared(
                        self._returns_series, self._benchmark_series
                    )
                    metrics["information_ratio"] = qs.stats.information_ratio(
                        self._returns_series, self._benchmark_series
                    )
                    metrics["treynor_ratio"] = qs.stats.treynor_ratio(
                        self._returns_series, self._benchmark_series
                    )
                except Exception as e:
                    logger.warning(f"Failed to calculate benchmark metrics: {e}")
                    metrics["alpha"] = 0.0
                    metrics["beta"] = 1.0
                    metrics["r_squared"] = 0.0
                    metrics["information_ratio"] = 0.0
                    metrics["treynor_ratio"] = 0.0

            # Convert any NaN/inf to 0
            for key, value in metrics.items():
                if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                    metrics[key] = 0.0

            self._cache[cache_key] = metrics
            return metrics

        except Exception as e:
            logger.error(f"Error calculating QuantStats metrics: {e}")
            return {}

    def get_all_metrics(self) -> dict[str, float]:
        """
        Get all available metrics (Numba Tier 1 + QuantStats).

        Combines Numba-optimized metrics with QuantStats for
        comprehensive analytics. Numba metrics take precedence
        for overlapping calculations.

        Returns:
            Dictionary with all metrics (60+)
        """
        cache_key = "all"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Start with QuantStats (comprehensive)
        metrics = self.get_quantstats_metrics()

        # Override with Numba Tier 1 (faster, preferred)
        tier1 = self.get_tier1_metrics()

        # Map Tier 1 names to output names
        metrics["sortino_ratio"] = tier1["sortino_ratio"]
        metrics["calmar_ratio"] = tier1["calmar_ratio"]
        metrics["profit_factor_numba"] = tier1["profit_factor"]
        metrics["value_at_risk_95"] = tier1["value_at_risk_95"]
        metrics["cvar_95_numba"] = tier1["cvar_95"]
        metrics["kelly_criterion_numba"] = tier1["kelly_criterion"]
        metrics["recovery_factor_numba"] = tier1["recovery_factor"]
        metrics["ulcer_index_numba"] = tier1["ulcer_index"]
        metrics["max_consecutive_wins"] = tier1["max_consecutive_wins"]
        metrics["max_consecutive_losses"] = tier1["max_consecutive_losses"]
        metrics["skewness"] = tier1["skewness"]
        metrics["kurtosis_numba"] = tier1["kurtosis"]
        metrics["tail_ratio_numba"] = tier1["tail_ratio"]

        if self._benchmark is not None:
            metrics["alpha_numba"] = tier1["alpha"]
            metrics["beta_numba"] = tier1["beta"]
            metrics["information_ratio_numba"] = tier1["information_ratio"]
            metrics["r_squared_numba"] = tier1["r_squared"]

        self._cache[cache_key] = metrics
        return metrics

    def get_monthly_returns(self) -> pd.DataFrame:
        """
        Get monthly returns pivot table.

        Returns:
            DataFrame with months as rows, years as columns
        """
        try:
            return qs.stats.monthly_returns(self._returns_series)
        except Exception as e:
            logger.error(f"Error calculating monthly returns: {e}")
            return pd.DataFrame()

    def get_drawdown_series(self) -> pd.Series:
        """
        Get drawdown time series.

        Returns:
            Series of drawdown values (0 to -1)
        """
        try:
            return qs.stats.to_drawdown_series(self._returns_series)
        except Exception as e:
            logger.error(f"Error calculating drawdown series: {e}")
            return pd.Series()

    def clear_cache(self) -> None:
        """Clear all cached results."""
        self._cache.clear()


# =============================================================================
# Market Configuration for Annualization
# =============================================================================
# Different markets have different trading hours and days.
# This configuration allows easy extension to futures, crypto, forex, etc.

from dataclasses import dataclass
from enum import Enum


class MarketType(Enum):
    """Supported market types with different trading schedules."""
    US_EQUITIES = "us_equities"      # NYSE/NASDAQ: 6.5 hrs/day, 252 days/year
    CRYPTO = "crypto"                 # 24/7/365
    US_FUTURES = "us_futures"         # ~23 hrs/day, 252 days/year
    FOREX = "forex"                   # 24/5 (Mon-Fri)


@dataclass
class MarketConfig:
    """
    Trading schedule configuration for a market.

    Used to calculate the correct annualization factor for any interval.
    """
    trading_days_per_year: int
    trading_hours_per_day: float
    name: str

    def get_periods_per_day(self, interval: str) -> int:
        """Calculate how many periods of given interval fit in one trading day."""
        interval_minutes = INTERVAL_TO_MINUTES.get(interval)
        if interval_minutes is None:
            raise ValueError(f"Unknown interval: {interval}")

        if interval == "1d":
            return 1
        if interval == "1wk":
            return 1  # Handled separately

        minutes_per_day = self.trading_hours_per_day * 60
        return int(minutes_per_day / interval_minutes)

    def get_periods_per_year(self, interval: str) -> int:
        """
        Calculate annualization factor for a given interval.

        Args:
            interval: Data interval ("1d", "1h", "15m", etc.)

        Returns:
            Number of periods per year for annualization
        """
        if interval == "1wk":
            return self.trading_days_per_year // 5  # Approximate weeks
        if interval == "1d":
            return self.trading_days_per_year

        periods_per_day = self.get_periods_per_day(interval)
        return periods_per_day * self.trading_days_per_year


# Interval duration in minutes
INTERVAL_TO_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "1d": None,   # Special case
    "1wk": None,  # Special case
}

# Pre-configured market schedules
MARKET_CONFIGS = {
    MarketType.US_EQUITIES: MarketConfig(
        trading_days_per_year=252,
        trading_hours_per_day=6.5,  # 9:30 AM - 4:00 PM ET
        name="US Equities (NYSE/NASDAQ)",
    ),
    MarketType.CRYPTO: MarketConfig(
        trading_days_per_year=365,
        trading_hours_per_day=24.0,  # 24/7
        name="Cryptocurrency",
    ),
    MarketType.US_FUTURES: MarketConfig(
        trading_days_per_year=252,
        trading_hours_per_day=23.0,  # Nearly 24 hours with breaks
        name="US Futures (CME)",
    ),
    MarketType.FOREX: MarketConfig(
        trading_days_per_year=260,  # ~52 weeks * 5 days
        trading_hours_per_day=24.0,  # 24 hours Mon-Fri
        name="Forex",
    ),
}

# Default market for backwards compatibility
DEFAULT_MARKET = MarketType.US_EQUITIES


def get_market_config(market: MarketType | str = DEFAULT_MARKET) -> MarketConfig:
    """
    Get market configuration.

    Args:
        market: MarketType enum or string name

    Returns:
        MarketConfig for the specified market
    """
    if isinstance(market, str):
        try:
            market = MarketType(market)
        except ValueError:
            logger.warning(f"Unknown market '{market}', using US_EQUITIES")
            market = MarketType.US_EQUITIES
    return MARKET_CONFIGS[market]


def get_periods_per_year(
    interval: str,
    market: MarketType | str = DEFAULT_MARKET
) -> int:
    """
    Get the annualization factor for a given interval and market.

    Args:
        interval: Data interval ("1d", "1h", "15m", etc.)
        market: Market type (default: US_EQUITIES)

    Returns:
        Number of periods per year for annualization

    Example:
        >>> get_periods_per_year("1d")  # US Equities
        252
        >>> get_periods_per_year("1d", MarketType.CRYPTO)
        365
        >>> get_periods_per_year("1h", MarketType.US_EQUITIES)
        1638
        >>> get_periods_per_year("1h", MarketType.CRYPTO)
        8760
    """
    config = get_market_config(market)
    try:
        return config.get_periods_per_year(interval)
    except ValueError:
        logger.warning(f"Unknown interval '{interval}', defaulting to daily")
        return config.trading_days_per_year


# Legacy constant for backwards compatibility (US Equities daily)
INTERVAL_PERIODS_PER_YEAR = {
    interval: get_periods_per_year(interval, MarketType.US_EQUITIES)
    for interval in INTERVAL_TO_MINUTES.keys()
}


@lru_cache(maxsize=32)
def fetch_benchmark_returns(
    symbol: str = "SPY",
    start_date: str | None = None,
    end_date: str | None = None,
    interval: str = "1d",
) -> np.ndarray:
    """
    Fetch benchmark returns using the configured data provider.

    Uses the provider system to fetch data, allowing switching between
    Yahoo Finance, Alpaca, or other providers without code changes.

    When no date range is specified, fetches maximum available data
    (the provider determines what "max" means for its data source).

    Results are cached to avoid repeated API calls.

    Args:
        symbol: Benchmark symbol (default: SPY)
        start_date: Start date string (YYYY-MM-DD), None for max available
        end_date: End date string (YYYY-MM-DD), None for today
        interval: Data interval ("1d", "1h", "15m", etc.)

    Returns:
        Array of benchmark returns
    """
    try:
        # Import provider here to avoid circular imports
        from backend.data.provider import get_provider

        provider = get_provider()

        # Determine fetch strategy based on date parameters
        use_max_fetch = start_date is None and end_date is None

        if use_max_fetch:
            # Fetch maximum available data - all providers must implement this
            # (e.g., Yahoo uses period="max", Alpaca fetches all available history)
            logger.debug(f"Fetching max available {interval} data for {symbol} via {provider.name}")
            loop = asyncio.new_event_loop()
            try:
                data = loop.run_until_complete(
                    provider.fetch_max_available(symbol, interval=interval)
                )
            finally:
                loop.close()
        else:
            # Specific date range requested
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")

            loop = asyncio.new_event_loop()
            try:
                data = loop.run_until_complete(
                    provider.fetch_historical_data(symbol, start_date, end_date, interval=interval)
                )
            finally:
                loop.close()

        if data.bars is None or len(data.bars) < 2:
            logger.warning(f"Insufficient data for benchmark {symbol}")
            return np.array([])

        # Extract close prices and calculate returns
        # bars format: [open, high, low, close, volume]
        close_prices = data.bars[:, 3]  # Close is at index 3
        returns = np.diff(close_prices) / close_prices[:-1]
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

        logger.debug(f"Fetched {len(returns)} {interval} benchmark returns for {symbol} via {provider.name}")
        return returns

    except Exception as e:
        logger.error(f"Error fetching benchmark returns for {symbol}: {e}")
        # Fallback to QuantStats which uses period="max" internally (daily only)
        if interval == "1d":
            try:
                logger.info(f"Falling back to QuantStats for benchmark {symbol}")
                # QuantStats download_returns fetches max available daily data
                returns = qs.utils.download_returns(symbol)
                if start_date:
                    returns = returns[returns.index >= start_date]
                if end_date:
                    returns = returns[returns.index <= end_date]
                return returns.values
            except Exception as e2:
                logger.error(f"QuantStats fallback also failed: {e2}")
        else:
            logger.warning(f"QuantStats fallback only supports daily data, not {interval}")
        return np.array([])


def create_adapter_from_backtest(
    equity_curve: list[float],
    trades_pnl: list[float],
    timestamps: list[str] | None = None,
    benchmark_symbol: str = "SPY",
    risk_free_rate: float = 0.04,
) -> QuantStatsAdapter:
    """
    Create QuantStatsAdapter from backtest results.

    Convenience function to create adapter from typical backtest output.

    Args:
        equity_curve: List of equity values
        trades_pnl: List of trade P&Ls
        timestamps: List of timestamp strings (ISO format)
        benchmark_symbol: Benchmark for comparison
        risk_free_rate: Annual risk-free rate

    Returns:
        Configured QuantStatsAdapter instance
    """
    equity = np.array(equity_curve, dtype=np.float64)

    # Calculate returns from equity curve
    returns = np.diff(equity) / equity[:-1]
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    pnls = np.array(trades_pnl, dtype=np.float64)

    # Parse timestamps
    ts = None
    if timestamps and len(timestamps) > 1:
        try:
            ts = np.array(pd.to_datetime(timestamps[1:]))  # Skip first (initial equity)
        except Exception:
            ts = None

    # Fetch benchmark
    benchmark = None
    if benchmark_symbol:
        try:
            benchmark = fetch_benchmark_returns(benchmark_symbol)
            # Align to same length as returns
            if len(benchmark) > len(returns):
                benchmark = benchmark[-len(returns):]
            elif len(benchmark) < len(returns):
                # Pad with zeros at start
                padding = np.zeros(len(returns) - len(benchmark))
                benchmark = np.concatenate([padding, benchmark])
        except Exception as e:
            logger.warning(f"Could not fetch benchmark {benchmark_symbol}: {e}")

    return QuantStatsAdapter(
        returns=returns,
        equity_curve=equity,
        pnls=pnls,
        benchmark_returns=benchmark,
        risk_free_rate=risk_free_rate,
        timestamps=ts,
    )
