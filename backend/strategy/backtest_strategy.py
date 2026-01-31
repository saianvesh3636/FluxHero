"""
Backtest Strategy Adapter for FluxHero API.

Provides a reusable strategy wrapper that integrates:
- Dual-mode signal generation (trend-following + mean-reversion)
- Regime detection (ADX + R-squared)
- Position sizing based on risk
- Stop loss management

This module bridges the gap between the BacktestEngine and the
dual-mode strategy implementation.
"""

import logging

import numpy as np

from backend.backtesting.engine import Order, OrderSide, OrderType, Position, PositionSide
from backend.computation.adaptive_ema import calculate_kama_with_regime_adjustment
from backend.computation.indicators import calculate_atr, calculate_bollinger_bands, calculate_rsi
from backend.computation.volatility import calculate_atr_ma
from backend.strategy.dual_mode import (
    MODE_MEAN_REVERSION,
    MODE_NEUTRAL,
    MODE_TREND_FOLLOWING,
    SIGNAL_EXIT_LONG,
    SIGNAL_EXIT_SHORT,
    SIGNAL_LONG,
    SIGNAL_NONE,
    SIGNAL_SHORT,
    calculate_fixed_stop_loss,
    calculate_position_size,
    generate_mean_reversion_signals,
    generate_trend_following_signals,
)
from backend.strategy.regime_detector import (
    REGIME_MEAN_REVERSION,
    REGIME_NEUTRAL,
    REGIME_STRONG_TREND,
    detect_regime,
)

logger = logging.getLogger(__name__)

# Signal name mappings for debug logging
_SIGNAL_NAMES = {
    SIGNAL_NONE: "NONE",
    SIGNAL_LONG: "LONG",
    SIGNAL_SHORT: "SHORT",
    SIGNAL_EXIT_LONG: "EXIT_LONG",
    SIGNAL_EXIT_SHORT: "EXIT_SHORT",
}

# Regime name mappings for debug logging
_REGIME_NAMES = {
    REGIME_MEAN_REVERSION: "MEAN_REVERSION",
    REGIME_NEUTRAL: "NEUTRAL",
    REGIME_STRONG_TREND: "STRONG_TREND",
}

# Mode name mappings for debug logging
_MODE_NAMES = {
    MODE_TREND_FOLLOWING: "TREND_FOLLOWING",
    MODE_MEAN_REVERSION: "MEAN_REVERSION",
    MODE_NEUTRAL: "NEUTRAL",
}


class DualModeBacktestStrategy:
    """
    Dual-mode strategy adapter for backtest execution.

    Pre-computes all indicators and signals for efficient execution.
    Designed for use with BacktestEngine.

    Strategy Modes:
    - "TREND": Only use trend-following signals
    - "MEAN_REVERSION": Only use mean-reversion signals
    - "DUAL": Automatically switch based on regime detection (default)

    The strategy:
    1. Pre-computes all indicators during initialization
    2. Detects market regime (trending vs mean-reverting)
    3. Generates signals based on active strategy mode
    4. Calculates position size based on risk parameters
    """

    # Warmup period needed for indicators to stabilize
    WARMUP_BARS = 60

    def __init__(
        self,
        bars: np.ndarray,
        initial_capital: float = 100000.0,
        strategy_mode: str = "DUAL",
        trend_risk_pct: float = 0.01,
        mr_risk_pct: float = 0.0075,
    ):
        """
        Initialize strategy with price data.

        Pre-calculates all indicators for performance.

        Args:
            bars: OHLCV array (N, 5) with [open, high, low, close, volume]
            initial_capital: Starting capital for position sizing
            strategy_mode: "TREND", "MEAN_REVERSION", or "DUAL"
            trend_risk_pct: Risk percentage for trend trades (default: 1%)
            mr_risk_pct: Risk percentage for mean-reversion trades (default: 0.75%)
        """
        self.initial_capital = initial_capital
        self.strategy_mode = strategy_mode.upper()
        self.trend_risk_pct = trend_risk_pct
        self.mr_risk_pct = mr_risk_pct

        # Track current capital for position sizing
        self.current_capital = initial_capital

        # Pre-compute all indicators
        self._compute_indicators(bars)

        # Track previous regime for change detection
        self._prev_regime: int | None = None

        logger.info(
            f"Initialized DualModeBacktestStrategy: mode={self.strategy_mode}, "
            f"capital=${initial_capital:,.0f}, bars={len(bars)}"
        )

    def _compute_indicators(self, bars: np.ndarray) -> None:
        """
        Pre-compute all technical indicators and signals.

        This is done once at initialization for performance.
        The BacktestEngine will call get_orders() for each bar,
        and we just look up pre-computed values.
        """
        # Extract OHLC as contiguous arrays (required for Numba)
        high = np.ascontiguousarray(bars[:, 1], dtype=np.float64)
        low = np.ascontiguousarray(bars[:, 2], dtype=np.float64)
        close = np.ascontiguousarray(bars[:, 3], dtype=np.float64)

        self.close = close
        self.high = high
        self.low = low
        n_bars = len(close)

        logger.debug(f"Computing indicators for {n_bars} bars...")

        # Calculate KAMA (Kaufman Adaptive Moving Average)
        self.kama, self.efficiency_ratio, _ = calculate_kama_with_regime_adjustment(close)

        # Calculate ATR (Average True Range) - used for stops and position sizing
        self.atr = calculate_atr(high, low, close, period=14)

        # Calculate ATR moving average for volatility regime
        self.atr_ma = calculate_atr_ma(self.atr, period=50)

        # Calculate RSI (Relative Strength Index)
        self.rsi = calculate_rsi(close, period=14)

        # Calculate Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = calculate_bollinger_bands(
            close, period=20, num_std=2.0
        )

        # Detect market regime
        regime_data = detect_regime(
            high=high,
            low=low,
            close=close,
            atr=self.atr,
            atr_ma=self.atr_ma,
            adx_period=14,
            regression_period=50,
            apply_persistence=True,
            confirmation_bars=3,
        )
        self.regime = regime_data["trend_regime_confirmed"]
        self.adx = regime_data["adx"]
        self.r_squared = regime_data["r_squared"]
        self.volatility_regime = regime_data["volatility_regime"]

        # Generate signals for both strategies
        self.trend_signals = generate_trend_following_signals(
            prices=close,
            kama=self.kama,
            atr=self.atr,
            entry_multiplier=0.5,
            exit_multiplier=0.3,
        )

        self.mr_signals = generate_mean_reversion_signals(
            prices=close,
            rsi=self.rsi,
            bollinger_lower=self.bb_lower,
            bollinger_middle=self.bb_middle,
            rsi_oversold=30.0,
            rsi_overbought=70.0,
        )

        logger.debug("Indicators computed successfully")

    def get_orders(
        self,
        bars: np.ndarray,
        bar_index: int,
        position: Position | None,
    ) -> list[Order]:
        """
        Strategy callback for BacktestEngine.

        Generates orders based on current bar and regime.
        This method is called by BacktestEngine for each bar.

        Args:
            bars: Full OHLCV array (not used, we use pre-computed values)
            bar_index: Current bar index being processed
            position: Current open position (None if flat)

        Returns:
            List of Order objects to execute
        """
        orders: list[Order] = []

        # Skip warmup period while indicators stabilize
        if bar_index < self.WARMUP_BARS:
            return orders

        # Get current values from pre-computed arrays
        current_close = self.close[bar_index]
        current_atr = self.atr[bar_index]
        current_regime = self.regime[bar_index]

        # Skip if any indicator is NaN
        if np.isnan(current_close) or np.isnan(current_atr):
            return orders

        # Log regime changes
        if self._prev_regime is not None and current_regime != self._prev_regime:
            logger.debug(
                "Regime change at bar %d: %s -> %s (ADX=%.2f, R²=%.3f)",
                bar_index,
                _REGIME_NAMES.get(self._prev_regime, "UNKNOWN"),
                _REGIME_NAMES.get(current_regime, "UNKNOWN"),
                self.adx[bar_index],
                self.r_squared[bar_index],
            )
        self._prev_regime = current_regime

        # Determine which signal to use based on strategy mode and regime
        active_signal, risk_pct, active_mode = self._get_active_signal(
            bar_index, current_regime
        )

        # Log signal decisions at DEBUG level (only when there's a signal)
        if active_signal != SIGNAL_NONE:
            logger.debug(
                "Signal at bar %d: %s (mode=%s, regime=%s, risk=%.2f%%)",
                bar_index,
                _SIGNAL_NAMES.get(active_signal, "UNKNOWN"),
                _MODE_NAMES.get(active_mode, "UNKNOWN"),
                _REGIME_NAMES.get(current_regime, "UNKNOWN"),
                risk_pct * 100,
            )

        # ENTRY LOGIC
        if position is None:
            if active_signal == SIGNAL_LONG:
                order = self._create_entry_order(
                    bar_index=bar_index,
                    side=OrderSide.BUY,
                    price=current_close,
                    atr=current_atr,
                    risk_pct=risk_pct,
                    active_mode=active_mode,
                )
                if order is not None:
                    orders.append(order)
                    logger.debug(
                        "LONG entry at bar %d: price=%.2f, shares=%d, "
                        "mode=%s, regime=%s, ATR=%.4f, RSI=%.2f",
                        bar_index,
                        current_close,
                        order.shares,
                        _MODE_NAMES.get(active_mode, "UNKNOWN"),
                        _REGIME_NAMES.get(current_regime, "UNKNOWN"),
                        current_atr,
                        self.rsi[bar_index],
                    )

            elif active_signal == SIGNAL_SHORT:
                # Short selling (if supported)
                # For now, we focus on long-only
                pass

        # EXIT LOGIC
        elif position is not None and position.side == PositionSide.LONG:
            if active_signal == SIGNAL_EXIT_LONG:
                orders.append(
                    Order(
                        bar_index=bar_index,
                        symbol=position.symbol,
                        side=OrderSide.SELL,
                        shares=position.shares,
                        order_type=OrderType.MARKET,
                    )
                )
                logger.debug(
                    "EXIT LONG at bar %d: price=%.2f, entry_price=%.2f, "
                    "shares=%d, regime=%s, RSI=%.2f",
                    bar_index,
                    current_close,
                    position.entry_price,
                    position.shares,
                    _REGIME_NAMES.get(current_regime, "UNKNOWN"),
                    self.rsi[bar_index],
                )

        elif position is not None and position.side == PositionSide.SHORT:
            if active_signal == SIGNAL_EXIT_SHORT:
                orders.append(
                    Order(
                        bar_index=bar_index,
                        symbol=position.symbol,
                        side=OrderSide.BUY,
                        shares=position.shares,
                        order_type=OrderType.MARKET,
                    )
                )
                logger.debug(
                    "EXIT SHORT at bar %d: price=%.2f, entry_price=%.2f, "
                    "shares=%d, regime=%s, RSI=%.2f",
                    bar_index,
                    current_close,
                    position.entry_price,
                    position.shares,
                    _REGIME_NAMES.get(current_regime, "UNKNOWN"),
                    self.rsi[bar_index],
                )

        return orders

    def _get_active_signal(
        self, bar_index: int, current_regime: int
    ) -> tuple[int, float, int]:
        """
        Determine which signal to use based on strategy mode and regime.

        Returns:
            Tuple of (signal, risk_pct, active_mode)
        """
        trend_signal = self.trend_signals[bar_index]
        mr_signal = self.mr_signals[bar_index]

        # Strategy mode: TREND only
        if self.strategy_mode == "TREND":
            return trend_signal, self.trend_risk_pct, MODE_TREND_FOLLOWING

        # Strategy mode: MEAN_REVERSION only
        if self.strategy_mode == "MEAN_REVERSION":
            return mr_signal, self.mr_risk_pct, MODE_MEAN_REVERSION

        # Strategy mode: DUAL (automatic regime-based switching)
        if current_regime == REGIME_STRONG_TREND:
            # Strong trend detected - use trend-following
            if trend_signal != SIGNAL_NONE:
                logger.debug(
                    "DUAL mode at bar %d: STRONG_TREND -> trend-following "
                    "(trend=%s, mr=%s)",
                    bar_index,
                    _SIGNAL_NAMES.get(trend_signal, "UNKNOWN"),
                    _SIGNAL_NAMES.get(mr_signal, "UNKNOWN"),
                )
            return trend_signal, self.trend_risk_pct, MODE_TREND_FOLLOWING

        elif current_regime == REGIME_MEAN_REVERSION:
            # Range-bound market - use mean-reversion
            if mr_signal != SIGNAL_NONE:
                logger.debug(
                    "DUAL mode at bar %d: MEAN_REVERSION -> mean-reversion "
                    "(trend=%s, mr=%s)",
                    bar_index,
                    _SIGNAL_NAMES.get(trend_signal, "UNKNOWN"),
                    _SIGNAL_NAMES.get(mr_signal, "UNKNOWN"),
                )
            return mr_signal, self.mr_risk_pct, MODE_MEAN_REVERSION

        else:  # REGIME_NEUTRAL
            # Neutral regime = range-bound -> use mean-reversion with reduced risk
            # Standard approach: trade the range (buy support, sell resistance)
            if mr_signal != SIGNAL_NONE:
                logger.debug(
                    "DUAL mode at bar %d: NEUTRAL -> mean-reversion signal (%s)",
                    bar_index,
                    _SIGNAL_NAMES.get(mr_signal, "UNKNOWN"),
                )
                # Use 70% of normal MR risk in neutral (slightly more cautious)
                return mr_signal, self.mr_risk_pct * 0.7, MODE_MEAN_REVERSION
            # Check for exit signals
            if trend_signal in (SIGNAL_EXIT_LONG, SIGNAL_EXIT_SHORT):
                logger.debug(
                    "DUAL mode at bar %d: NEUTRAL -> trend exit signal (%s)",
                    bar_index,
                    _SIGNAL_NAMES.get(trend_signal, "UNKNOWN"),
                )
                return trend_signal, self.mr_risk_pct, MODE_NEUTRAL
            if mr_signal in (SIGNAL_EXIT_LONG, SIGNAL_EXIT_SHORT):
                logger.debug(
                    "DUAL mode at bar %d: NEUTRAL -> mr exit signal (%s)",
                    bar_index,
                    _SIGNAL_NAMES.get(mr_signal, "UNKNOWN"),
                )
                return mr_signal, self.mr_risk_pct, MODE_NEUTRAL
            # No clear signal
            return SIGNAL_NONE, 0.0, MODE_NEUTRAL

    def _create_entry_order(
        self,
        bar_index: int,
        side: OrderSide,
        price: float,
        atr: float,
        risk_pct: float,
        active_mode: int,
    ) -> Order | None:
        """
        Create an entry order with proper position sizing.

        Args:
            bar_index: Current bar index
            side: Order side (BUY/SELL)
            price: Current price
            atr: Current ATR value
            risk_pct: Risk percentage for this trade
            active_mode: Active strategy mode

        Returns:
            Order object or None if position size is invalid
        """
        # Calculate stop loss based on active mode
        if active_mode == MODE_TREND_FOLLOWING:
            # Trend-following: ATR-based stop
            if side == OrderSide.BUY:
                stop_price = price - (2.5 * atr)
            else:
                stop_price = price + (2.5 * atr)
        else:
            # Mean-reversion: Fixed percentage stop
            stop_price = calculate_fixed_stop_loss(
                entry_price=price,
                is_long=(side == OrderSide.BUY),
                stop_pct=0.03,
            )

        # Calculate position size based on risk
        shares = calculate_position_size(
            capital=self.current_capital,
            entry_price=price,
            stop_price=stop_price,
            risk_pct=risk_pct,
            is_long=(side == OrderSide.BUY),
        )

        # Validate shares
        shares = int(shares)
        if shares <= 0:
            return None

        # Don't exceed maximum position size (20% of capital)
        max_position_value = self.current_capital * 0.20
        max_shares = int(max_position_value / price)
        shares = min(shares, max_shares)

        if shares <= 0:
            return None

        return Order(
            bar_index=bar_index,
            symbol="",  # Will be set by engine
            side=side,
            shares=shares,
            order_type=OrderType.MARKET,
        )

    def update_capital(self, new_capital: float) -> None:
        """
        Update current capital for position sizing.

        Called after trades to reflect current equity.
        """
        self.current_capital = new_capital

    def get_regime_summary(self) -> dict:
        """
        Get summary of regime distribution in the data.

        Returns:
            Dict with regime counts and percentages
        """
        total = len(self.regime)
        trend_count = np.sum(self.regime == REGIME_STRONG_TREND)
        mr_count = np.sum(self.regime == REGIME_MEAN_REVERSION)
        neutral_count = np.sum(self.regime == REGIME_NEUTRAL)

        return {
            "total_bars": total,
            "trend_bars": int(trend_count),
            "trend_pct": float(trend_count / total * 100) if total > 0 else 0,
            "mean_reversion_bars": int(mr_count),
            "mean_reversion_pct": float(mr_count / total * 100) if total > 0 else 0,
            "neutral_bars": int(neutral_count),
            "neutral_pct": float(neutral_count / total * 100) if total > 0 else 0,
        }

    def get_signal_summary(self) -> dict:
        """
        Get summary of signals generated.

        Returns:
            Dict with signal counts
        """
        return {
            "trend_long_signals": int(np.sum(self.trend_signals == SIGNAL_LONG)),
            "trend_short_signals": int(np.sum(self.trend_signals == SIGNAL_SHORT)),
            "trend_exit_long": int(np.sum(self.trend_signals == SIGNAL_EXIT_LONG)),
            "trend_exit_short": int(np.sum(self.trend_signals == SIGNAL_EXIT_SHORT)),
            "mr_long_signals": int(np.sum(self.mr_signals == SIGNAL_LONG)),
            "mr_short_signals": int(np.sum(self.mr_signals == SIGNAL_SHORT)),
            "mr_exit_long": int(np.sum(self.mr_signals == SIGNAL_EXIT_LONG)),
            "mr_exit_short": int(np.sum(self.mr_signals == SIGNAL_EXIT_SHORT)),
        }
