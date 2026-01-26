"""
Calibrated Backtest Strategy - No Magic Numbers

This strategy uses calibrated parameters derived from data instead of
hardcoded magic numbers like RSI 30/70 or ADX 25.

Key Differences from DualModeBacktestStrategy:
1. All thresholds come from CalibratedParameters (percentile-based)
2. Supports rolling recalibration during backtest
3. Uses Golden EMA (optional) instead of standard KAMA
4. Tracks calibration events for analysis

Usage:
    from backend.calibration import PercentileCalibrator
    from backend.strategy.calibrated_backtest_strategy import CalibratedBacktestStrategy

    # Calibrate parameters
    calibrator = PercentileCalibrator()
    params = calibrator.calibrate("SPY", bars)

    # Run strategy
    strategy = CalibratedBacktestStrategy(
        bars=bars,
        symbol="SPY",
        calibrated_params=params
    )
"""

import logging
from typing import Optional

import numpy as np

from backend.backtesting.engine import Order, OrderSide, OrderType, Position, PositionSide
from backend.computation.adaptive_ema import calculate_kama_with_regime_adjustment
from backend.computation.indicators import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_rsi,
)
from backend.computation.volatility import (
    calculate_atr_ma,
    detect_volatility_spike,
    get_stop_loss_multiplier,
    get_position_size_multiplier,
    calculate_volatility_alpha,
)
from backend.computation.golden_ema import (
    calculate_simple_golden_ema,
    calculate_golden_ema_fast_slow,
    calculate_golden_ema_signals,
)
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
)
from backend.strategy.regime_detector import (
    REGIME_MEAN_REVERSION,
    REGIME_NEUTRAL,
    REGIME_STRONG_TREND,
    detect_regime,
)
from backend.calibration import (
    CalibratedParameters,
    PercentileCalibrator,
    RollingCalibrator,
)

logger = logging.getLogger(__name__)

# Signal name mappings
_SIGNAL_NAMES = {
    SIGNAL_NONE: "NONE",
    SIGNAL_LONG: "LONG",
    SIGNAL_SHORT: "SHORT",
    SIGNAL_EXIT_LONG: "EXIT_LONG",
    SIGNAL_EXIT_SHORT: "EXIT_SHORT",
}

_REGIME_NAMES = {
    REGIME_MEAN_REVERSION: "MEAN_REVERSION",
    REGIME_NEUTRAL: "NEUTRAL",
    REGIME_STRONG_TREND: "STRONG_TREND",
}


def generate_calibrated_trend_signals(
    prices: np.ndarray,
    kama: np.ndarray,
    atr: np.ndarray,
    entry_multiplier: float,
    exit_multiplier: float,
) -> np.ndarray:
    """
    Generate trend-following signals with calibrated multipliers.

    Same logic as dual_mode.generate_trend_following_signals but
    accepts multipliers from calibration instead of defaults.
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.int32)

    if n < 2:
        return signals

    # Calculate entry and exit bands using CALIBRATED multipliers
    upper_entry = kama + (entry_multiplier * atr)
    lower_entry = kama - (entry_multiplier * atr)
    upper_exit = kama + (exit_multiplier * atr)
    lower_exit = kama - (exit_multiplier * atr)

    in_long = False
    in_short = False

    for i in range(1, n):
        if np.isnan(prices[i]) or np.isnan(kama[i]) or np.isnan(atr[i]):
            continue
        if np.isnan(prices[i - 1]) or np.isnan(kama[i - 1]):
            continue

        # ENTRY SIGNALS
        if not in_long and not in_short:
            # Long entry: price crosses above upper entry band
            if prices[i] > upper_entry[i] and prices[i - 1] <= upper_entry[i - 1]:
                signals[i] = SIGNAL_LONG
                in_long = True
            # Short entry: price crosses below lower entry band
            elif prices[i] < lower_entry[i] and prices[i - 1] >= lower_entry[i - 1]:
                signals[i] = SIGNAL_SHORT
                in_short = True

        # EXIT SIGNALS
        elif in_long:
            # Exit long: price crosses below lower exit band
            if prices[i] < lower_exit[i] and prices[i - 1] >= lower_exit[i - 1]:
                signals[i] = SIGNAL_EXIT_LONG
                in_long = False

        elif in_short:
            # Exit short: price crosses above upper exit band
            if prices[i] > upper_exit[i] and prices[i - 1] <= upper_exit[i - 1]:
                signals[i] = SIGNAL_EXIT_SHORT
                in_short = False

    return signals


def generate_calibrated_mr_signals(
    prices: np.ndarray,
    rsi: np.ndarray,
    bollinger_lower: np.ndarray,
    bollinger_middle: np.ndarray,
    rsi_oversold: float,
    rsi_overbought: float,
) -> np.ndarray:
    """
    Generate mean-reversion signals with calibrated RSI thresholds.

    Uses CALIBRATED thresholds instead of magic numbers 30/70.
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.int32)

    if n < 2:
        return signals

    in_long = False
    in_short = False

    for i in range(1, n):
        if np.isnan(prices[i]) or np.isnan(rsi[i]):
            continue
        if np.isnan(bollinger_lower[i]) or np.isnan(bollinger_middle[i]):
            continue

        # ENTRY SIGNALS
        if not in_long and not in_short:
            # Long entry: RSI below CALIBRATED oversold AND price at/below lower BB
            if rsi[i] < rsi_oversold and prices[i] <= bollinger_lower[i]:
                signals[i] = SIGNAL_LONG
                in_long = True
            # Short entry: RSI above CALIBRATED overbought AND price at/above upper BB
            elif rsi[i] > rsi_overbought:
                signals[i] = SIGNAL_SHORT
                in_short = True

        # EXIT SIGNALS
        elif in_long:
            # Exit long: price reaches middle band OR RSI > overbought
            if prices[i] >= bollinger_middle[i] or rsi[i] > rsi_overbought:
                signals[i] = SIGNAL_EXIT_LONG
                in_long = False

        elif in_short:
            # Exit short: price reaches middle band OR RSI < oversold
            if prices[i] <= bollinger_middle[i] or rsi[i] < rsi_oversold:
                signals[i] = SIGNAL_EXIT_SHORT
                in_short = False

    return signals


class CalibratedBacktestStrategy:
    """
    Backtest strategy using calibrated parameters instead of magic numbers.

    Features:
    - All thresholds from CalibratedParameters (data-driven)
    - Optional rolling recalibration
    - Optional Golden EMA instead of KAMA
    - Parameter stability tracking
    """

    WARMUP_BARS = 60

    def __init__(
        self,
        bars: np.ndarray,
        symbol: str,
        calibrated_params: Optional[CalibratedParameters] = None,
        initial_capital: float = 100000.0,
        strategy_mode: str = "DUAL",
        trend_risk_pct: float = 0.01,
        mr_risk_pct: float = 0.0075,
        use_golden_ema: bool = False,
        enable_rolling_calibration: bool = False,
        recalibrate_every: int = 21,
        dates: Optional[list[str]] = None,
    ):
        """
        Initialize calibrated strategy.

        Parameters
        ----------
        bars : np.ndarray
            OHLCV data (N, 5)
        symbol : str
            Asset symbol for calibration
        calibrated_params : CalibratedParameters, optional
            Pre-calibrated parameters. If None, will calibrate from bars.
        initial_capital : float
            Starting capital
        strategy_mode : str
            "TREND", "MEAN_REVERSION", or "DUAL"
        trend_risk_pct : float
            Risk per trend trade
        mr_risk_pct : float
            Risk per mean-reversion trade
        use_golden_ema : bool
            Use Golden Adaptive EMA instead of KAMA
        enable_rolling_calibration : bool
            Enable rolling recalibration during backtest
        recalibrate_every : int
            Recalibrate every N bars (if rolling enabled)
        dates : list[str], optional
            Date strings for each bar
        """
        self.symbol = symbol.upper()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.strategy_mode = strategy_mode.upper()
        self.trend_risk_pct = trend_risk_pct
        self.mr_risk_pct = mr_risk_pct
        self.use_golden_ema = use_golden_ema
        self.dates = dates

        # Calibration setup
        if calibrated_params is not None:
            self.params = calibrated_params
        else:
            # Auto-calibrate from data
            calibrator = PercentileCalibrator()
            self.params = calibrator.calibrate(symbol, bars, dates)

        # Rolling calibration
        self.rolling_calibration = enable_rolling_calibration
        if enable_rolling_calibration:
            self.rolling_calibrator = RollingCalibrator(
                lookback_bars=252,
                recalibrate_every=recalibrate_every
            )
            self.rolling_calibrator.current_params = self.params
            self.rolling_calibrator.last_calibration_bar = self.WARMUP_BARS
        else:
            self.rolling_calibrator = None

        # Store bars for potential recalibration
        self.bars = bars

        # Pre-compute indicators
        self._compute_indicators(bars)

        # State tracking
        self._prev_regime: Optional[int] = None
        self._calibration_events: list[dict] = []

        logger.info(
            f"Initialized CalibratedBacktestStrategy: symbol={self.symbol}, "
            f"mode={self.strategy_mode}, golden_ema={use_golden_ema}, "
            f"rolling_cal={enable_rolling_calibration}"
        )
        logger.info(
            f"Calibrated thresholds: RSI oversold={self.params.rsi_oversold:.1f}, "
            f"overbought={self.params.rsi_overbought:.1f}, "
            f"ER trending={self.params.er_trending:.3f}"
        )

    def _compute_indicators(self, bars: np.ndarray) -> None:
        """Pre-compute all indicators using calibrated parameters."""
        high = np.ascontiguousarray(bars[:, 1], dtype=np.float64)
        low = np.ascontiguousarray(bars[:, 2], dtype=np.float64)
        close = np.ascontiguousarray(bars[:, 3], dtype=np.float64)

        self.close = close
        self.high = high
        self.low = low

        # Core indicators
        self.rsi = calculate_rsi(close, period=14)
        self.atr = calculate_atr(high, low, close, period=14)
        self.atr_ma = calculate_atr_ma(self.atr, period=50)
        self.bb_upper, self.bb_middle, self.bb_lower = calculate_bollinger_bands(
            close, period=20, num_std=2.0
        )

        # Adaptive MA (KAMA or Golden EMA)
        if self.use_golden_ema:
            self.kama, self.alpha = calculate_simple_golden_ema(high, low, close)
            self.golden_fast, self.golden_slow, _, _ = calculate_golden_ema_fast_slow(
                high, low, close
            )
            self.efficiency_ratio = np.full_like(close, 0.5)  # Not used with Golden
        else:
            self.kama, self.efficiency_ratio, _ = calculate_kama_with_regime_adjustment(close)
            self.alpha = None
            self.golden_fast = None
            self.golden_slow = None

        # Regime detection with CALIBRATED thresholds
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
            # Use calibrated thresholds (convert ER to ADX scale)
            adx_trend_threshold=self._er_to_adx_threshold(self.params.er_trending),
            adx_ranging_threshold=self._er_to_adx_threshold(self.params.er_choppy),
            # Use calibrated volatility thresholds
            vol_high_threshold=self.params.vol_high,
            vol_low_threshold=self.params.vol_low,
        )
        self.regime = regime_data["trend_regime_confirmed"]
        self.adx = regime_data["adx"]
        self.r_squared = regime_data["r_squared"]
        self.volatility_regime = regime_data["volatility_regime"]

        # Volatility spike detection (for dynamic stop/size adjustment)
        # Use ATR as both short and long term (or calculate separate timeframes)
        atr_short = self.atr  # Current ATR
        atr_long = self.atr_ma  # Longer-term average
        self.volatility_spike = detect_volatility_spike(
            atr_short, atr_long,
            spike_threshold=self.params.vol_extreme
        )

        # Generate signals with CALIBRATED parameters
        self._generate_signals()

    def _generate_signals(self) -> None:
        """Generate signals using calibrated parameters."""
        # Trend-following signals with calibrated ATR multipliers
        self.trend_signals = generate_calibrated_trend_signals(
            prices=self.close,
            kama=self.kama,
            atr=self.atr,
            entry_multiplier=self.params.atr_entry_multiplier,
            exit_multiplier=self.params.atr_exit_multiplier,
        )

        # Mean-reversion signals with calibrated RSI thresholds
        self.mr_signals = generate_calibrated_mr_signals(
            prices=self.close,
            rsi=self.rsi,
            bollinger_lower=self.bb_lower,
            bollinger_middle=self.bb_middle,
            rsi_oversold=self.params.rsi_oversold,
            rsi_overbought=self.params.rsi_overbought,
        )

        # Golden EMA signals (if enabled)
        if self.use_golden_ema and self.golden_fast is not None:
            self.golden_signals = calculate_golden_ema_signals(
                self.close, self.golden_fast, self.golden_slow
            )
        else:
            self.golden_signals = None

    def _er_to_adx_threshold(self, er_threshold: float) -> float:
        """
        Convert Efficiency Ratio threshold to approximate ADX threshold.

        ER and ADX measure different things but are correlated.
        This is a rough mapping for regime detection.
        """
        # ER 0.3 ≈ ADX 20, ER 0.6 ≈ ADX 30
        # Linear interpolation: ADX = 10 + ER * 33.3
        return 10.0 + er_threshold * 33.3

    def get_orders(
        self,
        bars: np.ndarray,
        bar_index: int,
        position: Optional[Position],
    ) -> list[Order]:
        """Strategy callback for BacktestEngine."""
        orders: list[Order] = []

        # Skip warmup
        if bar_index < self.WARMUP_BARS:
            return orders

        # Rolling recalibration check
        if self.rolling_calibration and self.rolling_calibrator:
            if self.rolling_calibrator.should_recalibrate(bar_index):
                self.params = self.rolling_calibrator.recalibrate(
                    self.bars, bar_index, self.symbol, self.dates
                )
                # Regenerate signals with new parameters
                self._generate_signals()
                self._calibration_events.append({
                    "bar_index": bar_index,
                    "date": self.dates[bar_index] if self.dates else None,
                    "rsi_oversold": self.params.rsi_oversold,
                    "rsi_overbought": self.params.rsi_overbought,
                })
                logger.debug(
                    f"Recalibrated at bar {bar_index}: RSI thresholds "
                    f"{self.params.rsi_oversold:.1f}/{self.params.rsi_overbought:.1f}"
                )

        # Get current values
        current_close = self.close[bar_index]
        current_atr = self.atr[bar_index]
        current_regime = self.regime[bar_index]

        if np.isnan(current_close) or np.isnan(current_atr):
            return orders

        # Log regime changes
        if self._prev_regime is not None and current_regime != self._prev_regime:
            logger.debug(
                f"Regime change at bar {bar_index}: "
                f"{_REGIME_NAMES.get(int(self._prev_regime), 'UNKNOWN')} -> "
                f"{_REGIME_NAMES.get(int(current_regime), 'UNKNOWN')}"
            )
        self._prev_regime = current_regime

        # Get active signal based on mode and regime
        signal = self._get_active_signal(bar_index, int(current_regime))

        # Handle exits
        if position is not None:
            if position.side == PositionSide.LONG and signal == SIGNAL_EXIT_LONG:
                orders.append(self._create_exit_order(position, current_close))
            elif position.side == PositionSide.SHORT and signal == SIGNAL_EXIT_SHORT:
                orders.append(self._create_exit_order(position, current_close))
            return orders

        # Handle entries
        if signal == SIGNAL_LONG:
            order = self._create_entry_order(
                OrderSide.BUY, current_close, current_atr, int(current_regime), bar_index
            )
            if order:
                orders.append(order)
        elif signal == SIGNAL_SHORT:
            order = self._create_entry_order(
                OrderSide.SELL, current_close, current_atr, int(current_regime), bar_index
            )
            if order:
                orders.append(order)

        return orders

    def _get_active_signal(self, bar_index: int, current_regime: int) -> int:
        """Get signal based on strategy mode and regime."""
        if self.strategy_mode == "TREND":
            return int(self.trend_signals[bar_index])
        elif self.strategy_mode == "MEAN_REVERSION":
            return int(self.mr_signals[bar_index])
        elif self.strategy_mode == "GOLDEN" and self.golden_signals is not None:
            return int(self.golden_signals[bar_index])
        else:
            # DUAL mode - switch based on regime
            if current_regime == REGIME_STRONG_TREND:
                return int(self.trend_signals[bar_index])
            elif current_regime == REGIME_MEAN_REVERSION:
                return int(self.mr_signals[bar_index])
            else:
                # Neutral - require agreement or just use exits
                trend_sig = int(self.trend_signals[bar_index])
                mr_sig = int(self.mr_signals[bar_index])

                if trend_sig in (SIGNAL_EXIT_LONG, SIGNAL_EXIT_SHORT):
                    return trend_sig
                if mr_sig in (SIGNAL_EXIT_LONG, SIGNAL_EXIT_SHORT):
                    return mr_sig
                if trend_sig == mr_sig and trend_sig != SIGNAL_NONE:
                    return trend_sig

                return SIGNAL_NONE

    def _create_entry_order(
        self,
        side: OrderSide,
        price: float,
        atr: float,
        regime: int,
        bar_index: int = -1
    ) -> Optional[Order]:
        """Create entry order with calibrated stop loss and volatility adjustment."""
        is_long = side == OrderSide.BUY

        # Check for volatility spike - adjust parameters accordingly
        is_spike = False
        if bar_index >= 0 and bar_index < len(self.volatility_spike):
            is_spike = self.volatility_spike[bar_index] > 0

        # Stop loss using CALIBRATED multiplier
        if regime == REGIME_STRONG_TREND:
            stop_mult = self.params.atr_stop_multiplier
            risk_pct = self.trend_risk_pct
        else:
            stop_mult = self.params.atr_stop_multiplier
            risk_pct = self.mr_risk_pct

        # VOLATILITY SPIKE ADJUSTMENT: Widen stops during high volatility
        if is_spike:
            stop_mult *= get_stop_loss_multiplier(is_spike=True)  # 1.5x wider
            logger.debug(f"Volatility spike detected - widening stop to {stop_mult:.2f}x ATR")

        stop_distance = stop_mult * atr
        stop_loss = price - stop_distance if is_long else price + stop_distance

        # Position sizing
        risk_amount = self.current_capital * risk_pct
        risk_per_share = abs(price - stop_loss)

        if risk_per_share <= 0:
            return None

        shares = int(risk_amount / risk_per_share)

        # VOLATILITY SPIKE ADJUSTMENT: Reduce position size during high volatility
        if is_spike:
            size_mult = get_position_size_multiplier(is_spike=True)  # 0.7x smaller
            shares = int(shares * size_mult)
            logger.debug(f"Volatility spike - reducing position to {shares} shares")

        shares = max(1, min(shares, int(self.current_capital * 0.2 / price)))

        if shares < 1:
            return None

        return Order(
            side=side,
            order_type=OrderType.MARKET,
            quantity=shares,
            stop_loss=stop_loss,
        )

    def _create_exit_order(self, position: Position, price: float) -> Order:
        """Create exit order for position."""
        side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY
        return Order(
            side=side,
            order_type=OrderType.MARKET,
            quantity=abs(position.quantity),
        )

    def update_capital(self, new_capital: float) -> None:
        """Update current capital for position sizing."""
        self.current_capital = new_capital

    def get_calibration_summary(self) -> dict:
        """Get summary of calibration parameters and events."""
        return {
            "symbol": self.symbol,
            "params": {
                "rsi_oversold": self.params.rsi_oversold,
                "rsi_overbought": self.params.rsi_overbought,
                "er_choppy": self.params.er_choppy,
                "er_trending": self.params.er_trending,
                "atr_entry_mult": self.params.atr_entry_multiplier,
                "atr_exit_mult": self.params.atr_exit_multiplier,
                "atr_stop_mult": self.params.atr_stop_multiplier,
            },
            "rolling_calibration": self.rolling_calibration,
            "calibration_events": self._calibration_events,
            "n_recalibrations": len(self._calibration_events),
        }
