"""
AMM Strategy class for the Adaptive Market Measure system.

Follows the existing strategy patterns in the codebase.
"""

import numpy as np
from dataclasses import dataclass
from typing import Any, Callable

from backend.backtesting.engine import (
    Order,
    OrderSide,
    OrderType,
    Position,
)
from backend.computation.indicators import calculate_atr
from backend.computation import calculate_efficiency_ratio
from backend.amm_system.computation import compute_amm_indicators


@dataclass
class AMMConfig:
    """Configuration for the AMM strategy."""

    # Indicator periods (reduced from 150 to avoid long warmup)
    sma_period: int = 50  # ~2 months, enough to capture medium-term trend
    rsi_period: int = 14  # Standard RSI period
    mom_period: int = 20  # ~1 month momentum
    boll_period: int = 20  # Standard Bollinger period

    # Weights (should sum to 1.0)
    # Equal weights by default - let optimizer find the best mix
    w_sma: float = 0.25  # Trend component
    w_rsi: float = 0.25  # Momentum/overbought-oversold
    w_mom: float = 0.25  # Pure momentum
    w_boll: float = 0.25  # Mean-reversion component (Bollinger %B)

    # Signal processing
    zscore_lookback: int = 50  # Z-score normalization window
    ema_span: int = 10  # Faster EMA for more responsive signals

    # Trading thresholds
    entry_threshold: float = 1.0  # Enter when signal crosses 1 std dev
    exit_threshold: float = 0.0  # Exit when signal crosses zero

    # Risk management
    risk_per_trade: float = 0.01  # 1% risk per trade
    atr_stop_mult: float = 2.0  # ATR multiplier for stop loss
    atr_period: int = 14  # ATR calculation period
    max_position_pct: float = 0.20  # Max 20% of capital in single position

    # Regime detection (use efficiency ratio for regime)
    use_regime_filter: bool = True  # Enable regime-based filtering
    er_period: int = 20  # Efficiency ratio period
    er_trend_threshold: float = 0.3  # ER > this = trending (use momentum signals)
    er_range_threshold: float = 0.2  # ER < this = ranging (reduce position size)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "sma_period": self.sma_period,
            "rsi_period": self.rsi_period,
            "mom_period": self.mom_period,
            "boll_period": self.boll_period,
            "w_sma": self.w_sma,
            "w_rsi": self.w_rsi,
            "w_mom": self.w_mom,
            "w_boll": self.w_boll,
            "zscore_lookback": self.zscore_lookback,
            "ema_span": self.ema_span,
            "entry_threshold": self.entry_threshold,
            "exit_threshold": self.exit_threshold,
            "risk_per_trade": self.risk_per_trade,
            "atr_stop_mult": self.atr_stop_mult,
            "atr_period": self.atr_period,
            "max_position_pct": self.max_position_pct,
            "use_regime_filter": self.use_regime_filter,
            "er_period": self.er_period,
            "er_trend_threshold": self.er_trend_threshold,
            "er_range_threshold": self.er_range_threshold,
        }

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "AMMConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in params.items() if hasattr(cls, k)})


class AMMStrategy:
    """
    Adaptive Market Measure Strategy.

    Combines trend and mean-reversion indicators with configurable weights:
    - SMA deviation (trend)
    - RSI normalized (momentum)
    - Momentum (trend)
    - Bollinger %B (mean-reversion)

    Features:
    - Z-score normalization for consistent signal scale
    - EMA smoothing to reduce noise
    - Regime detection using Efficiency Ratio
    - ATR-based position sizing and stops
    """

    WARMUP_BARS = 60  # Reduced from 200 (max indicator period is now 50)

    def __init__(
        self,
        bars: np.ndarray,
        initial_capital: float,
        config: AMMConfig | None = None,
        symbol: str = "SPY",
    ):
        """
        Initialize the AMM strategy.

        Parameters
        ----------
        bars : np.ndarray
            OHLCV data with shape (n_bars, 5)
        initial_capital : float
            Starting capital for position sizing
        config : AMMConfig, optional
            Strategy configuration. Uses defaults if not provided.
        symbol : str
            Trading symbol
        """
        self.bars = bars
        self.capital = initial_capital
        self.config = config or AMMConfig()
        self.symbol = symbol

        # Pre-compute all indicators
        self.indicators = compute_amm_indicators(
            bars,
            sma_period=self.config.sma_period,
            rsi_period=self.config.rsi_period,
            mom_period=self.config.mom_period,
            boll_period=self.config.boll_period,
            w_sma=self.config.w_sma,
            w_rsi=self.config.w_rsi,
            w_mom=self.config.w_mom,
            w_boll=self.config.w_boll,
            zscore_lookback=self.config.zscore_lookback,
            ema_span=self.config.ema_span,
        )

        # Pre-compute ATR for stop loss calculation
        high = bars[:, 1].astype(np.float64)
        low = bars[:, 2].astype(np.float64)
        close = bars[:, 3].astype(np.float64)
        self.atr = calculate_atr(high, low, close, self.config.atr_period)

        # Pre-compute Efficiency Ratio for regime detection
        self.efficiency_ratio = calculate_efficiency_ratio(close, self.config.er_period)

    def get_orders(
        self,
        bars: np.ndarray,
        bar_index: int,
        position: Position | None,
    ) -> list[Order]:
        """
        Generate orders for the current bar.

        Uses regime detection (Efficiency Ratio) to adjust behavior:
        - High ER (trending): Normal entry with full position
        - Medium ER: Normal entry with full position
        - Low ER (ranging): Reduced position size (more choppy = more cautious)

        Parameters
        ----------
        bars : np.ndarray
            OHLCV data (same as passed to __init__)
        bar_index : int
            Current bar index
        position : Position | None
            Current open position, or None if flat

        Returns
        -------
        list[Order]
            List of orders to execute
        """
        orders = []

        # Skip warmup period
        if bar_index < self.WARMUP_BARS:
            return orders

        signal = self.indicators["signal"]
        current_signal = signal[bar_index]
        prev_signal = signal[bar_index - 1] if bar_index > 0 else np.nan

        # Skip if signal is not valid
        if np.isnan(current_signal) or np.isnan(prev_signal):
            return orders

        close_price = bars[bar_index, 3]
        threshold = self.config.entry_threshold
        exit_threshold = self.config.exit_threshold

        # Get regime from Efficiency Ratio
        current_er = self.efficiency_ratio[bar_index]
        regime_multiplier = 1.0  # Position size multiplier based on regime

        if self.config.use_regime_filter and not np.isnan(current_er):
            if current_er < self.config.er_range_threshold:
                # Ranging market - reduce position size by 50%
                regime_multiplier = 0.5
            elif current_er > self.config.er_trend_threshold:
                # Trending market - full position
                regime_multiplier = 1.0
            else:
                # Neutral - slightly reduced
                regime_multiplier = 0.75

        if position is None:
            # No position - check for entry signals

            # Long entry: signal crosses above +threshold
            if prev_signal <= threshold and current_signal > threshold:
                stop_loss = self._calculate_stop_loss(close_price, True, bar_index)
                shares = self._calculate_position_size(close_price, stop_loss)
                # Apply regime multiplier
                shares = int(shares * regime_multiplier)
                if shares > 0:
                    orders.append(
                        Order(
                            bar_index=bar_index,
                            symbol=self.symbol,
                            side=OrderSide.BUY,
                            shares=shares,
                            order_type=OrderType.MARKET,
                        )
                    )

            # Short entry: signal crosses below -threshold
            elif prev_signal >= -threshold and current_signal < -threshold:
                stop_loss = self._calculate_stop_loss(close_price, False, bar_index)
                shares = self._calculate_position_size(close_price, stop_loss)
                # Apply regime multiplier
                shares = int(shares * regime_multiplier)
                if shares > 0:
                    orders.append(
                        Order(
                            bar_index=bar_index,
                            symbol=self.symbol,
                            side=OrderSide.SELL,
                            shares=shares,
                            order_type=OrderType.MARKET,
                        )
                    )

        else:
            # Have a position - check for exit signals
            is_long = position.side.value > 0

            if is_long:
                # Exit long: signal crosses below exit threshold (0)
                if prev_signal >= exit_threshold and current_signal < exit_threshold:
                    orders.append(
                        Order(
                            bar_index=bar_index,
                            symbol=self.symbol,
                            side=OrderSide.SELL,
                            shares=position.shares,
                            order_type=OrderType.MARKET,
                        )
                    )
            else:
                # Exit short: signal crosses above exit threshold (0)
                if prev_signal <= exit_threshold and current_signal > exit_threshold:
                    orders.append(
                        Order(
                            bar_index=bar_index,
                            symbol=self.symbol,
                            side=OrderSide.BUY,
                            shares=position.shares,
                            order_type=OrderType.MARKET,
                        )
                    )

        return orders

    def _calculate_stop_loss(
        self, entry_price: float, is_long: bool, bar_index: int
    ) -> float:
        """Calculate stop loss price based on ATR."""
        atr_value = self.atr[bar_index]
        if np.isnan(atr_value):
            atr_value = entry_price * 0.02  # Fallback: 2% of price

        stop_distance = atr_value * self.config.atr_stop_mult

        if is_long:
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def _calculate_position_size(
        self, entry_price: float, stop_price: float
    ) -> int:
        """
        Calculate position size based on risk per trade.

        Uses fixed fractional position sizing:
        Risk = Capital × risk_per_trade
        Shares = Risk / (Entry - Stop)
        """
        risk_amount = self.capital * self.config.risk_per_trade
        stop_distance = abs(entry_price - stop_price)

        if stop_distance <= 0:
            return 0

        shares = int(risk_amount / stop_distance)

        # Apply max position constraint
        max_shares = int(
            (self.capital * self.config.max_position_pct) / entry_price
        )
        shares = min(shares, max_shares)

        return max(shares, 0)

    def update_capital(self, new_capital: float) -> None:
        """Update capital for position sizing calculations."""
        self.capital = new_capital

    def get_indicator_state(self, bar_index: int) -> dict[str, float]:
        """Get indicator values at a specific bar for debugging."""
        return {
            "sma_deviation": float(self.indicators["sma_deviation"][bar_index]),
            "rsi_normalized": float(self.indicators["rsi_normalized"][bar_index]),
            "momentum": float(self.indicators["momentum"][bar_index]),
            "bollinger_pct_b": float(self.indicators["bollinger_pct_b"][bar_index]),
            "combined_raw": float(self.indicators["combined_raw"][bar_index]),
            "combined_zscore": float(self.indicators["combined_zscore"][bar_index]),
            "signal": float(self.indicators["signal"][bar_index]),
            "atr": float(self.atr[bar_index]),
            "efficiency_ratio": float(self.efficiency_ratio[bar_index]),
        }


def create_amm_strategy_factory(
    config: AMMConfig | None = None,
) -> Callable[[np.ndarray, float, dict[str, Any]], Callable]:
    """
    Create an AMM strategy factory compatible with walk-forward testing.

    Parameters
    ----------
    config : AMMConfig, optional
        Base configuration. Parameters from walk-forward optimizer
        will override these values.

    Returns
    -------
    Callable
        Strategy factory function with signature:
        (bars, initial_capital, params) -> strategy_func
    """
    base_config = config or AMMConfig()

    def factory(
        bars: np.ndarray,
        initial_capital: float,
        params: dict[str, Any],
    ) -> Callable[[np.ndarray, int, Position | None], list[Order]]:
        """Create strategy function from parameters."""
        # Merge base config with optimizer params
        merged_params = base_config.to_dict()
        merged_params.update(params)
        strategy_config = AMMConfig.from_dict(merged_params)

        # Create strategy instance
        strategy = AMMStrategy(
            bars=bars,
            initial_capital=initial_capital,
            config=strategy_config,
        )

        # Return the get_orders method bound to this instance
        return strategy.get_orders

    return factory
