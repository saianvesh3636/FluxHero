"""
Adaptive Trend Strategy - No Magic Numbers, Regime-Aware

This strategy improves on the basic KAMA breakout by:
1. Using calibrated parameters instead of hardcoded values
2. Adding regime check (skip trades in MEAN_REVERSION regime)
3. Adding ADX filter (only enter when trend strength is sufficient)
4. Rolling recalibration to adapt to changing market conditions

Usage:
    from backend.strategy.adaptive_trend_strategy import AdaptiveTrendStrategy
    
    strategy = AdaptiveTrendStrategy(bars, symbol="SPY")
    # Use with BacktestEngine
"""

import logging
from typing import Optional
import numpy as np

from backend.backtesting.engine import Order, OrderSide, OrderType, Position, PositionSide
from backend.computation.adaptive_ema import calculate_kama
from backend.computation.indicators import calculate_atr
from backend.computation.volatility import calculate_atr_ma
from backend.strategy.regime_detector import (
    detect_regime,
    REGIME_MEAN_REVERSION,
    REGIME_NEUTRAL,
    REGIME_STRONG_TREND,
)
from backend.calibration import PercentileCalibrator, CalibratedParameters
from backend.calibration.rolling_calibrator import RollingCalibrator

logger = logging.getLogger(__name__)

# Signal constants
SIGNAL_NONE = 0
SIGNAL_LONG = 1
SIGNAL_SHORT = -1
SIGNAL_EXIT_LONG = 2
SIGNAL_EXIT_SHORT = -2


class AdaptiveTrendStrategy:
    """
    Adaptive trend-following strategy with regime filtering.
    
    Key improvements over basic KAMA breakout:
    - All thresholds calibrated from data (no magic numbers)
    - Regime check: skip entries in MEAN_REVERSION
    - ADX filter: require minimum trend strength for entry
    - Rolling recalibration for adaptation
    
    Entry: Price crosses above KAMA + (calibrated_entry_mult × ATR)
           AND regime != MEAN_REVERSION
           AND ADX > adx_threshold
           
    Exit: Price crosses below KAMA - (calibrated_exit_mult × ATR)
          OR regime changes to MEAN_REVERSION
    """
    
    WARMUP_BARS = 60
    
    def __init__(
        self,
        bars: np.ndarray,
        symbol: str = "SPY",
        initial_capital: float = 100000.0,
        risk_pct: float = 0.01,
        # Regime filtering
        skip_mean_reversion: bool = True,
        require_trend_for_entry: bool = False,  # Only enter in STRONG_TREND
        # ADX filtering
        use_adx_filter: bool = True,
        adx_entry_threshold: float = 20.0,  # Minimum ADX to enter
        # Calibration
        enable_rolling_calibration: bool = True,
        recalibrate_every: int = 21,
        lookback_bars: int = 252,
        # Override calibrated params (for testing)
        calibrated_params: Optional[CalibratedParameters] = None,
        dates: Optional[list[str]] = None,
    ):
        """
        Initialize adaptive trend strategy.
        
        Parameters
        ----------
        bars : np.ndarray
            OHLCV data (N, 5)
        symbol : str
            Asset symbol
        initial_capital : float
            Starting capital for position sizing
        risk_pct : float
            Risk per trade as decimal (0.01 = 1%)
        skip_mean_reversion : bool
            If True, don't enter new positions in MEAN_REVERSION regime
        require_trend_for_entry : bool
            If True, only enter in STRONG_TREND regime (more conservative)
        use_adx_filter : bool
            If True, require ADX > threshold for entry
        adx_entry_threshold : float
            Minimum ADX value for entry (default: 20)
        enable_rolling_calibration : bool
            If True, recalibrate parameters periodically
        recalibrate_every : int
            Recalibrate every N bars
        lookback_bars : int
            Bars to use for calibration
        calibrated_params : CalibratedParameters, optional
            Pre-calibrated parameters (skip auto-calibration)
        dates : list[str], optional
            Date strings for each bar
        """
        self.symbol = symbol.upper()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_pct = risk_pct
        self.dates = dates
        self.bars = bars
        
        # Regime filtering settings
        self.skip_mean_reversion = skip_mean_reversion
        self.require_trend_for_entry = require_trend_for_entry
        
        # ADX filtering settings
        self.use_adx_filter = use_adx_filter
        self.adx_entry_threshold = adx_entry_threshold
        
        # Initialize calibration
        if calibrated_params is not None:
            self.params = calibrated_params
            self.rolling_calibrator = None
        elif enable_rolling_calibration:
            self.rolling_calibrator = RollingCalibrator(
                lookback_bars=lookback_bars,
                recalibrate_every=recalibrate_every,
                min_bars_for_calibration=100,
            )
            # Initial calibration
            if len(bars) > self.WARMUP_BARS:
                self.params = self.rolling_calibrator.recalibrate(
                    bars, self.WARMUP_BARS, symbol, dates, reason="initial"
                )
            else:
                # Fall back to default params
                self.params = self._get_default_params()
        else:
            # One-time calibration
            calibrator = PercentileCalibrator()
            self.params = calibrator.calibrate(symbol, bars, dates)
            self.rolling_calibrator = None
        
        # Pre-compute indicators
        self._compute_indicators()
        
        # State tracking
        self._prev_regime = None
        self._trade_log = []
        
        logger.info(
            f"AdaptiveTrendStrategy initialized: symbol={symbol}, "
            f"skip_mr={skip_mean_reversion}, adx_filter={use_adx_filter}, "
            f"adx_threshold={adx_entry_threshold}"
        )
        logger.info(
            f"Calibrated params: entry_mult={self.params.atr_entry_multiplier:.2f}, "
            f"exit_mult={self.params.atr_exit_multiplier:.2f}, "
            f"stop_mult={self.params.atr_stop_multiplier:.2f}"
        )
    
    def _get_default_params(self) -> CalibratedParameters:
        """Get default parameters when calibration isn't possible."""
        from datetime import datetime
        return CalibratedParameters(
            symbol=self.symbol,
            calibration_date=datetime.now().strftime("%Y-%m-%d"),
            lookback_bars=0,
            data_start_date="unknown",
            data_end_date="unknown",
            rsi_oversold=30.0,
            rsi_overbought=70.0,
            rsi_extreme_oversold=20.0,
            rsi_extreme_overbought=80.0,
            rsi_neutral_low=40.0,
            rsi_neutral_high=60.0,
            er_choppy=0.25,
            er_trending=0.5,
            er_strong_trend=0.7,
            vol_low=0.7,
            vol_normal_low=0.85,
            vol_normal_high=1.15,
            vol_high=1.3,
            vol_extreme=2.0,
            alpha_slow_regime=0.1,
            alpha_fast_regime=0.5,
            atr_entry_multiplier=0.5,
            atr_exit_multiplier=1.0,
            atr_stop_multiplier=2.5,
        )
    
    def _compute_indicators(self) -> None:
        """Pre-compute all indicators."""
        high = np.ascontiguousarray(self.bars[:, 1], dtype=np.float64)
        low = np.ascontiguousarray(self.bars[:, 2], dtype=np.float64)
        close = np.ascontiguousarray(self.bars[:, 3], dtype=np.float64)
        
        self.close = close
        self.high = high
        self.low = low
        
        # Core indicators
        self.kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
        self.atr = calculate_atr(high, low, close, period=14)
        self.atr_ma = calculate_atr_ma(self.atr, period=50)
        
        # Regime detection with calibrated vol thresholds
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
            vol_high_threshold=self.params.vol_high,
            vol_low_threshold=self.params.vol_low,
        )
        
        self.regime = regime_data["trend_regime_confirmed"]
        self.adx = regime_data["adx"]
        self.r_squared = regime_data["r_squared"]
        self.volatility_regime = regime_data["volatility_regime"]
        
        # Pre-compute entry/exit bands using CALIBRATED multipliers
        self.upper_entry = self.kama + (self.params.atr_entry_multiplier * self.atr)
        self.lower_exit = self.kama - (self.params.atr_exit_multiplier * self.atr)
        
        # Generate signals
        self._generate_signals()
    
    def _generate_signals(self) -> None:
        """Generate trend signals with regime and ADX filtering."""
        n = len(self.close)
        self.signals = np.zeros(n, dtype=np.int32)
        self.signal_reasons = [""] * n
        
        in_long = False
        
        for i in range(1, n):
            # Skip if indicators not ready
            if (np.isnan(self.close[i]) or np.isnan(self.kama[i]) or 
                np.isnan(self.atr[i]) or np.isnan(self.regime[i])):
                continue
            if np.isnan(self.close[i-1]) or np.isnan(self.kama[i-1]):
                continue
            
            current_regime = int(self.regime[i])
            current_adx = self.adx[i] if not np.isnan(self.adx[i]) else 0
            
            # EXIT LOGIC (always check, regardless of regime)
            if in_long:
                # Exit if price crosses below exit band
                if self.close[i] < self.lower_exit[i] and self.close[i-1] >= self.lower_exit[i-1]:
                    self.signals[i] = SIGNAL_EXIT_LONG
                    self.signal_reasons[i] = f"Price below exit band (KAMA - {self.params.atr_exit_multiplier:.1f}×ATR)"
                    in_long = False
                    continue
                
                # Exit if regime changes to MEAN_REVERSION (optional)
                if self.skip_mean_reversion and current_regime == REGIME_MEAN_REVERSION:
                    self.signals[i] = SIGNAL_EXIT_LONG
                    self.signal_reasons[i] = "Regime changed to MEAN_REVERSION"
                    in_long = False
                    continue
            
            # ENTRY LOGIC
            if not in_long:
                # Check regime filter
                if self.skip_mean_reversion and current_regime == REGIME_MEAN_REVERSION:
                    continue  # Skip entry in mean-reversion regime
                
                if self.require_trend_for_entry and current_regime != REGIME_STRONG_TREND:
                    continue  # Only enter in strong trend
                
                # Check ADX filter
                if self.use_adx_filter and current_adx < self.adx_entry_threshold:
                    continue  # ADX too low, no clear trend
                
                # Check breakout condition
                if self.close[i] > self.upper_entry[i] and self.close[i-1] <= self.upper_entry[i-1]:
                    self.signals[i] = SIGNAL_LONG
                    regime_name = {0: "MEAN_REV", 1: "NEUTRAL", 2: "TREND"}.get(current_regime, "?")
                    self.signal_reasons[i] = (
                        f"Breakout above KAMA + {self.params.atr_entry_multiplier:.1f}×ATR | "
                        f"Regime: {regime_name} | ADX: {current_adx:.1f}"
                    )
                    in_long = True
    
    def get_orders(
        self,
        bars: np.ndarray,
        bar_index: int,
        position: Optional[Position],
    ) -> list[Order]:
        """Strategy callback for BacktestEngine."""
        orders = []
        
        # Skip warmup
        if bar_index < self.WARMUP_BARS:
            return orders
        
        # Rolling recalibration
        if self.rolling_calibrator and self.rolling_calibrator.should_recalibrate(bar_index):
            self.params = self.rolling_calibrator.recalibrate(
                self.bars, bar_index, self.symbol, self.dates, reason="scheduled"
            )
            # Recompute indicators with new params
            self._compute_indicators()
            logger.debug(f"Recalibrated at bar {bar_index}")
        
        # Get pre-computed signal
        signal = self.signals[bar_index]
        
        # Handle exits
        if position is not None:
            if position.side == PositionSide.LONG and signal == SIGNAL_EXIT_LONG:
                orders.append(Order(
                    bar_index=bar_index,
                    symbol=self.symbol,
                    side=OrderSide.SELL,
                    shares=position.shares,
                    order_type=OrderType.MARKET,
                ))
                self._log_trade("EXIT", bar_index, self.close[bar_index], self.signal_reasons[bar_index])
            return orders
        
        # Handle entries
        if signal == SIGNAL_LONG:
            order = self._create_entry_order(bar_index)
            if order:
                orders.append(order)
                self._log_trade("ENTRY", bar_index, self.close[bar_index], self.signal_reasons[bar_index])
        
        return orders
    
    def _create_entry_order(self, bar_index: int) -> Optional[Order]:
        """Create entry order with calibrated stop loss."""
        price = self.close[bar_index]
        atr = self.atr[bar_index]
        
        if np.isnan(atr) or atr <= 0:
            return None
        
        # Stop loss using CALIBRATED multiplier
        stop_distance = self.params.atr_stop_multiplier * atr
        stop_loss = price - stop_distance
        
        # Position sizing based on risk
        risk_amount = self.current_capital * self.risk_pct
        risk_per_share = abs(price - stop_loss)
        
        if risk_per_share <= 0:
            return None
        
        shares = int(risk_amount / risk_per_share)
        
        # Cap at 20% of capital
        max_shares = int(self.current_capital * 0.20 / price)
        shares = max(1, min(shares, max_shares))
        
        if shares < 1:
            return None
        
        return Order(
            bar_index=bar_index,
            symbol=self.symbol,
            side=OrderSide.BUY,
            shares=shares,
            order_type=OrderType.MARKET,
        )
    
    def _log_trade(self, action: str, bar_index: int, price: float, reason: str) -> None:
        """Log trade for analysis."""
        date = self.dates[bar_index] if self.dates and bar_index < len(self.dates) else str(bar_index)
        self._trade_log.append({
            "action": action,
            "bar_index": bar_index,
            "date": date,
            "price": price,
            "reason": reason,
        })
    
    def update_capital(self, new_capital: float) -> None:
        """Update current capital for position sizing."""
        self.current_capital = new_capital
    
    def get_trade_log(self) -> list[dict]:
        """Get log of all trades with reasons."""
        return self._trade_log
    
    def get_strategy_summary(self) -> dict:
        """Get summary of strategy configuration and signals."""
        valid_regime = self.regime[~np.isnan(self.regime)]
        total = len(valid_regime) if len(valid_regime) > 0 else 1
        
        return {
            "symbol": self.symbol,
            "config": {
                "skip_mean_reversion": self.skip_mean_reversion,
                "require_trend_for_entry": self.require_trend_for_entry,
                "use_adx_filter": self.use_adx_filter,
                "adx_entry_threshold": self.adx_entry_threshold,
            },
            "calibrated_params": {
                "atr_entry_mult": self.params.atr_entry_multiplier,
                "atr_exit_mult": self.params.atr_exit_multiplier,
                "atr_stop_mult": self.params.atr_stop_multiplier,
            },
            "regime_distribution": {
                "trend_pct": np.sum(valid_regime == REGIME_STRONG_TREND) / total * 100,
                "neutral_pct": np.sum(valid_regime == REGIME_NEUTRAL) / total * 100,
                "mean_rev_pct": np.sum(valid_regime == REGIME_MEAN_REVERSION) / total * 100,
            },
            "signals": {
                "long_entries": int(np.sum(self.signals == SIGNAL_LONG)),
                "long_exits": int(np.sum(self.signals == SIGNAL_EXIT_LONG)),
            },
        }
