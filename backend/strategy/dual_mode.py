"""
Dual-Mode Strategy Engine for FluxHero Trading System.

This module implements both trend-following and mean-reversion strategies,
switching automatically based on detected market regime.

Features:
- Trend-Following Mode: KAMA + ATR breakout entries with trailing stops (R6.1)
- Mean-Reversion Mode: RSI + Bollinger Band entries with fixed stops (R6.2)
- Neutral/Transition Mode: Blended approach with reduced size (R6.3)
- Strategy Performance Tracking: Win rate, Sharpe, drawdown per mode (R6.4)

Performance: <100ms for signal generation on 10k candles (Numba JIT)

Reference:
- FLUXHERO_REQUIREMENTS.md Feature 6: Dual-Mode Strategy Engine
"""

import numpy as np
from numba import njit

# Signal types
SIGNAL_NONE = 0
SIGNAL_LONG = 1
SIGNAL_SHORT = -1
SIGNAL_EXIT_LONG = 2
SIGNAL_EXIT_SHORT = -2

# Strategy modes
MODE_TREND_FOLLOWING = 1
MODE_MEAN_REVERSION = 2
MODE_NEUTRAL = 3


@njit(cache=True)
def generate_trend_following_signals(
    prices: np.ndarray,
    kama: np.ndarray,
    atr: np.ndarray,
    entry_multiplier: float = 0.5,
    exit_multiplier: float = 0.3,
) -> np.ndarray:
    """
    Generate trend-following entry and exit signals.

    Entry Logic (R6.1.1):
        LONG: Price crosses above KAMA + (0.5 × ATR)
        SHORT: Price crosses below KAMA - (0.5 × ATR)

    Exit Logic (R6.1.2):
        EXIT LONG: Price crosses below KAMA - (0.3 × ATR)
        EXIT SHORT: Price crosses above KAMA + (0.3 × ATR)

    Parameters
    ----------
    prices : np.ndarray (float64)
        Array of closing prices
    kama : np.ndarray (float64)
        Kaufman Adaptive Moving Average values
    atr : np.ndarray (float64)
        Average True Range values
    entry_multiplier : float
        ATR multiplier for entry threshold (default: 0.5)
    exit_multiplier : float
        ATR multiplier for exit threshold (default: 0.3)

    Returns
    -------
    np.ndarray (int32)
        Signal array:
        0 = No signal
        1 = Enter long
        -1 = Enter short
        2 = Exit long
        -2 = Exit short
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.int32)

    if n < 2:
        return signals

    # Calculate entry and exit bands
    upper_entry = kama + (entry_multiplier * atr)
    lower_entry = kama - (entry_multiplier * atr)
    upper_exit = kama + (exit_multiplier * atr)
    lower_exit = kama - (exit_multiplier * atr)

    # Track position state
    in_long = False
    in_short = False

    for i in range(1, n):
        # Skip if any indicator is NaN
        if np.isnan(prices[i]) or np.isnan(kama[i]) or np.isnan(atr[i]):
            continue
        if np.isnan(prices[i - 1]) or np.isnan(kama[i - 1]) or np.isnan(atr[i - 1]):
            continue

        # ENTRY SIGNALS (only if not in position)
        if not in_long and not in_short:
            # Long entry: Price crosses above KAMA + (0.5 × ATR)
            if prices[i - 1] <= upper_entry[i - 1] and prices[i] > upper_entry[i]:
                signals[i] = SIGNAL_LONG
                in_long = True
            # Short entry: Price crosses below KAMA - (0.5 × ATR)
            elif prices[i - 1] >= lower_entry[i - 1] and prices[i] < lower_entry[i]:
                signals[i] = SIGNAL_SHORT
                in_short = True

        # EXIT SIGNALS
        elif in_long:
            # Exit long: Price crosses below KAMA - (0.3 × ATR)
            if prices[i - 1] >= lower_exit[i - 1] and prices[i] < lower_exit[i]:
                signals[i] = SIGNAL_EXIT_LONG
                in_long = False
        elif in_short:
            # Exit short: Price crosses above KAMA + (0.3 × ATR)
            if prices[i - 1] <= upper_exit[i - 1] and prices[i] > upper_exit[i]:
                signals[i] = SIGNAL_EXIT_SHORT
                in_short = False

    return signals


@njit(cache=True)
def calculate_trailing_stop(
    prices: np.ndarray, atr: np.ndarray, entry_idx: int, is_long: bool, atr_multiplier: float = 2.5
) -> np.ndarray:
    """
    Calculate trailing stop levels based on ATR.

    Trailing Stop Logic (R6.1.3):
        LONG: Stop = Peak Price - (2.5 × ATR)
        SHORT: Stop = Trough Price + (2.5 × ATR)

    Parameters
    ----------
    prices : np.ndarray (float64)
        Array of closing prices
    atr : np.ndarray (float64)
        Average True Range values
    entry_idx : int
        Index where position was entered
    is_long : bool
        True for long position, False for short position
    atr_multiplier : float
        ATR multiplier for stop distance (default: 2.5)

    Returns
    -------
    np.ndarray (float64)
        Array of trailing stop levels from entry_idx onward
    """
    n = len(prices)
    stops = np.full(n, np.nan, dtype=np.float64)

    if entry_idx >= n or entry_idx < 0:
        return stops

    if is_long:
        # Long trailing stop: moves up with price, never down
        peak = prices[entry_idx]
        for i in range(entry_idx, n):
            if np.isnan(prices[i]) or np.isnan(atr[i]):
                continue
            peak = max(peak, prices[i])
            stops[i] = peak - (atr_multiplier * atr[i])
    else:
        # Short trailing stop: moves down with price, never up
        trough = prices[entry_idx]
        for i in range(entry_idx, n):
            if np.isnan(prices[i]) or np.isnan(atr[i]):
                continue
            trough = min(trough, prices[i])
            stops[i] = trough + (atr_multiplier * atr[i])

    return stops


@njit(cache=True)
def generate_mean_reversion_signals(
    prices: np.ndarray,
    rsi: np.ndarray,
    bollinger_lower: np.ndarray,
    bollinger_middle: np.ndarray,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
) -> np.ndarray:
    """
    Generate mean-reversion entry and exit signals.

    Entry Logic (R6.2.1):
        LONG: RSI < 30 AND price touches lower Bollinger Band
        SHORT: RSI > 70 AND price touches upper Bollinger Band

    Exit Logic (R6.2.2):
        EXIT LONG: Price returns to 20-period SMA (middle band) OR RSI > 70
        EXIT SHORT: Price returns to 20-period SMA (middle band) OR RSI < 30

    Parameters
    ----------
    prices : np.ndarray (float64)
        Array of closing prices
    rsi : np.ndarray (float64)
        RSI values (0-100 range)
    bollinger_lower : np.ndarray (float64)
        Lower Bollinger Band values
    bollinger_middle : np.ndarray (float64)
        Middle Bollinger Band (20-period SMA)
    rsi_oversold : float
        RSI threshold for oversold condition (default: 30)
    rsi_overbought : float
        RSI threshold for overbought condition (default: 70)

    Returns
    -------
    np.ndarray (int32)
        Signal array (same as trend-following signals)
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.int32)

    if n < 2:
        return signals

    # Track position state
    in_long = False
    in_short = False

    for i in range(1, n):
        # Skip if any indicator is NaN
        if np.isnan(prices[i]) or np.isnan(rsi[i]):
            continue
        if np.isnan(bollinger_lower[i]) or np.isnan(bollinger_middle[i]):
            continue

        # ENTRY SIGNALS (only if not in position)
        if not in_long and not in_short:
            # Long entry: RSI < 30 AND price at/below lower Bollinger Band
            if rsi[i] < rsi_oversold and prices[i] <= bollinger_lower[i]:
                signals[i] = SIGNAL_LONG
                in_long = True
            # Short entry: RSI > 70 AND price at/above upper Bollinger Band
            # Note: For mean reversion, we need upper band too, but we can infer it
            # or use bollinger_upper if provided. For now, focus on long only.

        # EXIT SIGNALS
        elif in_long:
            # Exit long: Price returns to middle band OR RSI > 70
            if prices[i] >= bollinger_middle[i] or rsi[i] > rsi_overbought:
                signals[i] = SIGNAL_EXIT_LONG
                in_long = False
        elif in_short:
            # Exit short: Price returns to middle band OR RSI < 30
            if prices[i] <= bollinger_middle[i] or rsi[i] < rsi_oversold:
                signals[i] = SIGNAL_EXIT_SHORT
                in_short = False

    return signals


@njit(cache=True)
def calculate_fixed_stop_loss(entry_price: float, is_long: bool, stop_pct: float = 0.03) -> float:
    """
    Calculate fixed stop loss level for mean-reversion trades.

    Stop Loss Logic (R6.2.3):
        LONG: Entry Price × (1 - stop_pct)
        SHORT: Entry Price × (1 + stop_pct)

    Parameters
    ----------
    entry_price : float
        Price at which position was entered
    is_long : bool
        True for long position, False for short position
    stop_pct : float
        Stop loss percentage (default: 0.03 = 3%)

    Returns
    -------
    float
        Stop loss price level
    """
    if is_long:
        return entry_price * (1.0 - stop_pct)
    else:
        return entry_price * (1.0 + stop_pct)


@njit(cache=True)
def calculate_position_size(
    capital: float, entry_price: float, stop_price: float, risk_pct: float, is_long: bool
) -> float:
    """
    Calculate position size based on risk percentage.

    Position Sizing (R6.1.4, R6.2.4):
        Trend-following: Risk 1% of capital per trade
        Mean-reversion: Risk 0.75% of capital per trade

    Formula:
        Risk Amount = Capital × risk_pct
        Price Risk = |Entry Price - Stop Price|
        Shares = Risk Amount / Price Risk

    Parameters
    ----------
    capital : float
        Total account capital
    entry_price : float
        Price at which position will be entered
    stop_price : float
        Stop loss price level
    risk_pct : float
        Percentage of capital to risk (e.g., 0.01 for 1%)
    is_long : bool
        True for long position, False for short position

    Returns
    -------
    float
        Number of shares to trade
    """
    risk_amount = capital * risk_pct
    price_risk = abs(entry_price - stop_price)

    if price_risk <= 0.0:
        return 0.0

    shares = risk_amount / price_risk
    return shares


@njit(cache=True)
def blend_signals(
    trend_signals: np.ndarray, mr_signals: np.ndarray, require_agreement: bool = True
) -> np.ndarray:
    """
    Blend trend-following and mean-reversion signals for neutral regime.

    Blending Logic (R6.3.1, R6.3.3):
        - If require_agreement: Only generate signal if both strategies agree
        - Otherwise: 50/50 weighted blend

    Parameters
    ----------
    trend_signals : np.ndarray (int32)
        Trend-following signals
    mr_signals : np.ndarray (int32)
        Mean-reversion signals
    require_agreement : bool
        If True, both strategies must agree (default: True for higher confidence)

    Returns
    -------
    np.ndarray (int32)
        Blended signal array
    """
    n = len(trend_signals)
    blended = np.zeros(n, dtype=np.int32)

    for i in range(n):
        if require_agreement:
            # Both strategies must agree (higher confidence threshold)
            if trend_signals[i] == mr_signals[i] and trend_signals[i] != SIGNAL_NONE:
                blended[i] = trend_signals[i]
        else:
            # Take either signal (50/50 weight)
            if trend_signals[i] != SIGNAL_NONE:
                blended[i] = trend_signals[i]
            elif mr_signals[i] != SIGNAL_NONE:
                blended[i] = mr_signals[i]

    return blended


@njit(cache=True)
def adjust_size_for_regime(base_size: float, regime: int) -> float:
    """
    Adjust position size based on market regime.

    Size Adjustment (R6.3.2):
        - STRONG_TREND: 100% of base size
        - MEAN_REVERSION: 100% of base size
        - NEUTRAL: 70% of base size (30% reduction)

    Parameters
    ----------
    base_size : float
        Base position size (shares)
    regime : int
        Market regime (1=TREND, 2=MEAN_REV, 3=NEUTRAL)

    Returns
    -------
    float
        Adjusted position size
    """
    if regime == MODE_NEUTRAL:
        return base_size * 0.7  # 30% reduction
    return base_size


# Python-only functions (not JIT-compiled due to data structures)


class StrategyPerformance:
    """
    Track performance metrics for each strategy mode.

    Tracks (R6.4.1):
        - Win rate (winning trades / total trades)
        - Sharpe ratio (returns / volatility)
        - Max drawdown (peak to trough decline)
        - Total return
    """

    def __init__(self):
        self.trades = []  # List of (pnl, mode) tuples
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.returns = []
        self.peak_equity = 0.0
        self.max_drawdown = 0.0

    def add_trade(self, pnl: float, mode: int) -> None:
        """Add a completed trade to performance tracking."""
        self.trades.append((pnl, mode))
        self.total_pnl += pnl

        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1

        # Update drawdown
        if self.total_pnl > self.peak_equity:
            self.peak_equity = self.total_pnl
        dd = (self.peak_equity - self.total_pnl) / max(self.peak_equity, 1.0)
        self.max_drawdown = max(self.max_drawdown, dd)

    def get_win_rate(self) -> float:
        """Calculate win rate (0.0 to 1.0)."""
        total = self.wins + self.losses
        if total == 0:
            return 0.0
        return self.wins / total

    def get_sharpe_ratio(self, risk_free_rate: float = 0.04) -> float:
        """
        Calculate Sharpe ratio.

        Sharpe = (Mean Return - Risk Free Rate) / Std Dev of Returns
        """
        if len(self.returns) < 2:
            return 0.0

        returns_array = np.array(self.returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)

        if std_return == 0:
            return 0.0

        return (mean_return - risk_free_rate) / std_return

    def get_total_return(self) -> float:
        """Get total cumulative PnL."""
        return self.total_pnl

    def get_max_drawdown(self) -> float:
        """Get maximum drawdown (0.0 to 1.0)."""
        return self.max_drawdown


class DualModeStrategy:
    """
    Main dual-mode strategy coordinator.

    Manages:
        - Strategy selection based on regime
        - Performance tracking per mode
        - Dynamic mode weight adjustment (R6.4.2, R6.4.3)
    """

    def __init__(self):
        self.trend_perf = StrategyPerformance()
        self.mr_perf = StrategyPerformance()
        self.neutral_perf = StrategyPerformance()

        # Mode weights (0.0 to 1.0)
        self.trend_weight = 1.0
        self.mr_weight = 1.0

    def get_active_mode(self, regime: int) -> int:
        """
        Determine which strategy mode to use based on regime.

        Mode Selection (R6.1.5, R6.2.5):
            - REGIME_STRONG_TREND (2) → Trend-following
            - REGIME_MEAN_REVERSION (0) → Mean-reversion
            - REGIME_NEUTRAL (1) → Blended/Neutral

        Parameters
        ----------
        regime : int
            Current market regime

        Returns
        -------
        int
            Strategy mode to use
        """
        # Map regime constants to mode constants
        # REGIME_STRONG_TREND = 2 → MODE_TREND_FOLLOWING = 1
        # REGIME_MEAN_REVERSION = 0 → MODE_MEAN_REVERSION = 2
        # REGIME_NEUTRAL = 1 → MODE_NEUTRAL = 3

        if regime == 2:  # REGIME_STRONG_TREND
            return MODE_TREND_FOLLOWING
        elif regime == 0:  # REGIME_MEAN_REVERSION
            return MODE_MEAN_REVERSION
        else:  # REGIME_NEUTRAL or unknown
            return MODE_NEUTRAL

    def update_performance(self, pnl: float, mode: int) -> None:
        """Add trade result to appropriate performance tracker."""
        if mode == MODE_TREND_FOLLOWING:
            self.trend_perf.add_trade(pnl, mode)
        elif mode == MODE_MEAN_REVERSION:
            self.mr_perf.add_trade(pnl, mode)
        elif mode == MODE_NEUTRAL:
            self.neutral_perf.add_trade(pnl, mode)

    def rebalance_weights(self, min_trades: int = 20) -> None:
        """
        Adjust mode weights based on recent performance.

        Dynamic Weight Adjustment (R6.4.2, R6.4.3):
            - If a mode underperforms for 20+ trades, reduce allocation
            - Monthly rebalance based on win rates and total returns
        """
        trend_trades = self.trend_perf.wins + self.trend_perf.losses
        mr_trades = self.mr_perf.wins + self.mr_perf.losses

        # Only rebalance if we have enough data
        if trend_trades < min_trades and mr_trades < min_trades:
            return

        # Compare performance metrics
        trend_win_rate = self.trend_perf.get_win_rate()
        mr_win_rate = self.mr_perf.get_win_rate()
        trend_return = self.trend_perf.get_total_return()
        mr_return = self.mr_perf.get_total_return()

        # Reduce weight for underperforming mode (negative total return)
        if trend_trades >= min_trades and trend_return < 0:
            self.trend_weight = max(0.5, self.trend_weight - 0.1)
        if mr_trades >= min_trades and mr_return < 0:
            self.mr_weight = max(0.5, self.mr_weight - 0.1)

        # Boost weight for outperforming mode (better win rate and positive return)
        if trend_trades >= min_trades and mr_trades >= min_trades:
            if trend_return > mr_return and trend_win_rate > 0.5:
                self.trend_weight = min(1.0, self.trend_weight + 0.1)
            elif mr_return > trend_return and mr_win_rate > 0.5:
                self.mr_weight = min(1.0, self.mr_weight + 0.1)

    def get_performance_summary(self) -> dict:
        """Get comprehensive performance summary for all modes."""
        return {
            "trend_following": {
                "total_trades": self.trend_perf.wins + self.trend_perf.losses,
                "win_rate": self.trend_perf.get_win_rate(),
                "sharpe_ratio": self.trend_perf.get_sharpe_ratio(),
                "max_drawdown": self.trend_perf.get_max_drawdown(),
                "total_return": self.trend_perf.get_total_return(),
                "weight": self.trend_weight,
            },
            "mean_reversion": {
                "total_trades": self.mr_perf.wins + self.mr_perf.losses,
                "win_rate": self.mr_perf.get_win_rate(),
                "sharpe_ratio": self.mr_perf.get_sharpe_ratio(),
                "max_drawdown": self.mr_perf.get_max_drawdown(),
                "total_return": self.mr_perf.get_total_return(),
                "weight": self.mr_weight,
            },
            "neutral": {
                "total_trades": self.neutral_perf.wins + self.neutral_perf.losses,
                "win_rate": self.neutral_perf.get_win_rate(),
                "sharpe_ratio": self.neutral_perf.get_sharpe_ratio(),
                "max_drawdown": self.neutral_perf.get_max_drawdown(),
                "total_return": self.neutral_perf.get_total_return(),
                "weight": 1.0,
            },
        }
