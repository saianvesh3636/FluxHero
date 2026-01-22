"""
Signal Generator with Explanation Logging for FluxHero Trading System.

This module generates trading signals with comprehensive explanations,
capturing the reasoning behind each trade decision for transparency and analysis.

Features:
- Signal generation with complete context (volatility, regime, indicators)
- Trade reason formatting with detailed explanations
- Integration with dual-mode strategy engine
- Storage-ready signal explanations for database logging

Reference:
- FLUXHERO_REQUIREMENTS.md Feature 12.3: Signal Explainer
"""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class SignalType(IntEnum):
    """Signal type enumeration."""

    NONE = 0
    LONG = 1
    SHORT = -1
    EXIT_LONG = 2
    EXIT_SHORT = -2


class RegimeType(IntEnum):
    """Market regime enumeration."""

    MEAN_REVERSION = 0
    NEUTRAL = 1
    STRONG_TREND = 2


class VolatilityState(IntEnum):
    """Volatility state enumeration."""

    LOW = 0
    NORMAL = 1
    HIGH = 2


class StrategyMode(IntEnum):
    """Strategy mode enumeration."""

    MEAN_REVERSION = 1
    TREND_FOLLOWING = 2
    NEUTRAL = 3


@dataclass
class SignalExplanation:
    """
    Complete explanation for a trading signal.

    Attributes
    ----------
    symbol : str
        Trading symbol (e.g., "SPY")
    signal_type : SignalType
        Type of signal (LONG, SHORT, EXIT_LONG, EXIT_SHORT, NONE)
    price : float
        Price at which signal was generated
    timestamp : float
        Unix timestamp of signal
    strategy_mode : StrategyMode
        Active strategy mode (TREND_FOLLOWING, MEAN_REVERSION, NEUTRAL)
    regime : RegimeType
        Detected market regime (STRONG_TREND, MEAN_REVERSION, NEUTRAL)
    volatility_state : VolatilityState
        Current volatility state (LOW, NORMAL, HIGH)

    Indicator values:
    atr : float
        Average True Range value
    kama : float
        Kaufman Adaptive Moving Average value
    rsi : float or None
        Relative Strength Index (if applicable)
    adx : float or None
        Average Directional Index (if available)
    r_squared : float or None
        Linear regression R² (if available)

    Risk parameters:
    risk_amount : float
        Dollar risk on this trade
    risk_percent : float
        Percentage of account risked
    stop_loss : float
        Stop loss price
    position_size : int
        Number of shares/contracts

    Signal details:
    entry_trigger : str
        What triggered the entry (e.g., "KAMA crossover", "RSI oversold")
    noise_filtered : bool
        Whether noise filter was applied
    volume_validated : bool
        Whether volume validation passed
    """

    symbol: str
    signal_type: SignalType
    price: float
    timestamp: float
    strategy_mode: StrategyMode
    regime: RegimeType
    volatility_state: VolatilityState

    # Indicator values
    atr: float
    kama: float
    rsi: float | None = None
    adx: float | None = None
    r_squared: float | None = None

    # Risk parameters
    risk_amount: float = 0.0
    risk_percent: float = 0.0
    stop_loss: float = 0.0
    position_size: int = 0

    # Signal details
    entry_trigger: str = ""
    noise_filtered: bool = True
    volume_validated: bool = True

    def format_signal_reason(self) -> str:
        """
        Format a human-readable explanation of the signal.

        Returns
        -------
        str
            Formatted multi-line explanation following R12.3.1 format:
            "BUY SPY @ $420.50
            Reason: Volatility (ATR=3.2, High) + KAMA crossover (Price > KAMA+0.5×ATR)
            Regime: STRONG_TREND (ADX=32, R²=0.81)
            Risk: $1,000 (1% account), Stop: $415.00"
        """
        # Header line: Action, symbol, price
        if self.signal_type == SignalType.NONE:
            return "NO SIGNAL"

        action = self._get_action_name()
        header = f"{action} {self.symbol} @ ${self.price:.2f}"

        # Reason line: Volatility state + Entry trigger
        vol_state_name = self._get_volatility_state_name()
        reason = f"Reason: Volatility (ATR={self.atr:.2f}, {vol_state_name}) + {self.entry_trigger}"

        # Regime line: Regime type with supporting indicators
        regime_name = self._get_regime_name()
        regime_details = self._format_regime_details()
        regime_line = f"Regime: {regime_name} {regime_details}"

        # Risk line: Dollar risk, percent risk, stop loss
        risk_line = ""
        if self.signal_type in (SignalType.LONG, SignalType.SHORT):
            risk_line = (
                f"Risk: ${self.risk_amount:.0f} ({self.risk_percent:.2f}% account), "
                f"Stop: ${self.stop_loss:.2f}"
            )

        # Combine all lines
        lines = [header, reason, regime_line]
        if risk_line:
            lines.append(risk_line)

        return "\n".join(lines)

    def format_compact_reason(self) -> str:
        """
        Format a compact single-line explanation for tooltips.

        Returns
        -------
        str
            Single-line explanation for UI display
        """
        if self.signal_type == SignalType.NONE:
            return "No signal"

        action = self._get_action_name()
        vol_state = self._get_volatility_state_name()
        regime_name = self._get_regime_name()

        return (
            f"{action} @ ${self.price:.2f} | "
            f"{self.entry_trigger} | "
            f"{regime_name} (ATR={self.atr:.2f}, {vol_state}) | "
            f"Risk: ${self.risk_amount:.0f} ({self.risk_percent:.2f}%)"
        )

    def to_dict(self) -> dict:
        """
        Convert signal explanation to dictionary for storage.

        Returns
        -------
        dict
            Dictionary representation suitable for database storage
        """
        return {
            "symbol": self.symbol,
            "signal_type": int(self.signal_type),
            "price": self.price,
            "timestamp": self.timestamp,
            "strategy_mode": int(self.strategy_mode),
            "regime": int(self.regime),
            "volatility_state": int(self.volatility_state),
            "atr": self.atr,
            "kama": self.kama,
            "rsi": self.rsi,
            "adx": self.adx,
            "r_squared": self.r_squared,
            "risk_amount": self.risk_amount,
            "risk_percent": self.risk_percent,
            "stop_loss": self.stop_loss,
            "position_size": self.position_size,
            "entry_trigger": self.entry_trigger,
            "noise_filtered": self.noise_filtered,
            "volume_validated": self.volume_validated,
            "formatted_reason": self.format_signal_reason(),
            "compact_reason": self.format_compact_reason(),
        }

    def _get_action_name(self) -> str:
        """Get human-readable action name."""
        if self.signal_type == SignalType.LONG:
            return "BUY"
        elif self.signal_type == SignalType.SHORT:
            return "SELL SHORT"
        elif self.signal_type == SignalType.EXIT_LONG:
            return "SELL"
        elif self.signal_type == SignalType.EXIT_SHORT:
            return "COVER SHORT"
        return "NO ACTION"

    def _get_volatility_state_name(self) -> str:
        """Get human-readable volatility state name."""
        if self.volatility_state == VolatilityState.LOW:
            return "Low"
        elif self.volatility_state == VolatilityState.HIGH:
            return "High"
        return "Normal"

    def _get_regime_name(self) -> str:
        """Get human-readable regime name."""
        if self.regime == RegimeType.STRONG_TREND:
            return "STRONG_TREND"
        elif self.regime == RegimeType.MEAN_REVERSION:
            return "MEAN_REVERSION"
        return "NEUTRAL"

    def _format_regime_details(self) -> str:
        """Format regime details with available indicators."""
        details = []
        if self.adx is not None:
            details.append(f"ADX={self.adx:.1f}")
        if self.r_squared is not None:
            details.append(f"R²={self.r_squared:.2f}")
        if self.rsi is not None:
            details.append(f"RSI={self.rsi:.1f}")

        if details:
            return f"({', '.join(details)})"
        return ""


class SignalGenerator:
    """
    Generate trading signals with comprehensive explanations.

    This class integrates with the dual-mode strategy engine and other
    FluxHero components to produce fully-explained trading signals.
    """

    def __init__(self, account_balance: float = 100000.0):
        """
        Initialize signal generator.

        Parameters
        ----------
        account_balance : float
            Current account balance for risk calculations (default: $100,000)
        """
        self.account_balance = account_balance

    def generate_signal_with_explanation(
        self,
        symbol: str,
        timestamp: float,
        price: float,
        signal_type: int,
        strategy_mode: int,
        regime: int,
        volatility_state: int,
        atr: float,
        kama: float,
        entry_trigger: str,
        risk_percent: float = 0.01,
        stop_loss: float = 0.0,
        rsi: float | None = None,
        adx: float | None = None,
        r_squared: float | None = None,
        noise_filtered: bool = True,
        volume_validated: bool = True,
    ) -> SignalExplanation:
        """
        Generate a signal with complete explanation.

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g., "SPY")
        timestamp : float
            Unix timestamp of signal
        price : float
            Current price
        signal_type : int
            Signal type (LONG=1, SHORT=-1, EXIT_LONG=2, EXIT_SHORT=-2, NONE=0)
        strategy_mode : int
            Active strategy mode
        regime : int
            Market regime
        volatility_state : int
            Volatility state
        atr : float
            Average True Range
        kama : float
            KAMA value
        entry_trigger : str
            What triggered the signal (e.g., "KAMA crossover", "RSI oversold")
        risk_percent : float
            Percentage of account to risk (default: 1%)
        stop_loss : float
            Stop loss price
        rsi : float or None
            RSI value (optional)
        adx : float or None
            ADX value (optional)
        r_squared : float or None
            R² value (optional)
        noise_filtered : bool
            Whether noise filter passed
        volume_validated : bool
            Whether volume validation passed

        Returns
        -------
        SignalExplanation
            Complete signal explanation with all context
        """
        # Calculate risk parameters
        risk_amount = 0.0
        position_size = 0

        if signal_type in (SignalType.LONG, SignalType.SHORT) and stop_loss > 0:
            risk_amount = self.account_balance * risk_percent
            price_risk = abs(price - stop_loss)
            if price_risk > 0:
                position_size = int(risk_amount / price_risk)

        return SignalExplanation(
            symbol=symbol,
            signal_type=SignalType(signal_type),
            price=price,
            timestamp=timestamp,
            strategy_mode=StrategyMode(strategy_mode),
            regime=RegimeType(regime),
            volatility_state=VolatilityState(volatility_state),
            atr=atr,
            kama=kama,
            rsi=rsi,
            adx=adx,
            r_squared=r_squared,
            risk_amount=risk_amount,
            risk_percent=risk_percent,
            stop_loss=stop_loss,
            position_size=position_size,
            entry_trigger=entry_trigger,
            noise_filtered=noise_filtered,
            volume_validated=volume_validated,
        )

    def batch_generate_explanations(
        self,
        symbol: str,
        timestamps: np.ndarray,
        prices: np.ndarray,
        signals: np.ndarray,
        strategy_modes: np.ndarray,
        regimes: np.ndarray,
        volatility_states: np.ndarray,
        atr_values: np.ndarray,
        kama_values: np.ndarray,
        entry_triggers: list,
        stop_losses: np.ndarray,
        risk_percent: float = 0.01,
        rsi_values: np.ndarray | None = None,
        adx_values: np.ndarray | None = None,
        r_squared_values: np.ndarray | None = None,
    ) -> list:
        """
        Generate explanations for a batch of signals.

        Parameters
        ----------
        symbol : str
            Trading symbol
        timestamps : np.ndarray
            Array of timestamps
        prices : np.ndarray
            Array of prices
        signals : np.ndarray
            Array of signal types
        strategy_modes : np.ndarray
            Array of strategy modes
        regimes : np.ndarray
            Array of regime states
        volatility_states : np.ndarray
            Array of volatility states
        atr_values : np.ndarray
            Array of ATR values
        kama_values : np.ndarray
            Array of KAMA values
        entry_triggers : list
            List of entry trigger descriptions
        stop_losses : np.ndarray
            Array of stop loss prices
        risk_percent : float
            Risk percentage per trade
        rsi_values : np.ndarray or None
            Array of RSI values (optional)
        adx_values : np.ndarray or None
            Array of ADX values (optional)
        r_squared_values : np.ndarray or None
            Array of R² values (optional)

        Returns
        -------
        list
            List of SignalExplanation objects for non-NONE signals
        """
        explanations = []
        n = len(signals)

        for i in range(n):
            # Skip NONE signals
            if signals[i] == SignalType.NONE:
                continue

            # Extract optional values
            rsi = rsi_values[i] if rsi_values is not None else None
            adx = adx_values[i] if adx_values is not None else None
            r_sq = r_squared_values[i] if r_squared_values is not None else None
            trigger = entry_triggers[i] if i < len(entry_triggers) else "Signal detected"

            explanation = self.generate_signal_with_explanation(
                symbol=symbol,
                timestamp=timestamps[i],
                price=prices[i],
                signal_type=int(signals[i]),
                strategy_mode=int(strategy_modes[i]),
                regime=int(regimes[i]),
                volatility_state=int(volatility_states[i]),
                atr=atr_values[i],
                kama=kama_values[i],
                entry_trigger=trigger,
                risk_percent=risk_percent,
                stop_loss=stop_losses[i],
                rsi=rsi,
                adx=adx,
                r_squared=r_sq,
            )

            explanations.append(explanation)

        return explanations

    def update_account_balance(self, new_balance: float):
        """
        Update account balance for risk calculations.

        Parameters
        ----------
        new_balance : float
            New account balance
        """
        self.account_balance = new_balance
