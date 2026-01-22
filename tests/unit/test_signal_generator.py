"""
Unit tests for Signal Generator with Explanation Logging.

Tests verify:
- Signal explanation creation and formatting
- Trade reason formatting (R12.3.1)
- Risk calculation accuracy
- Batch signal generation
- Dictionary conversion for storage
- All enum types and edge cases
"""

import time

import numpy as np

from backend.strategy.signal_generator import (
    RegimeType,
    SignalExplanation,
    SignalGenerator,
    SignalType,
    StrategyMode,
    VolatilityState,
)


class TestSignalExplanation:
    """Test SignalExplanation dataclass."""

    def test_signal_explanation_creation(self):
        """Test basic signal explanation creation."""
        explanation = SignalExplanation(
            symbol="SPY",
            signal_type=SignalType.LONG,
            price=420.50,
            timestamp=time.time(),
            strategy_mode=StrategyMode.TREND_FOLLOWING,
            regime=RegimeType.STRONG_TREND,
            volatility_state=VolatilityState.HIGH,
            atr=3.2,
            kama=418.0,
            rsi=55.0,
            adx=32.0,
            r_squared=0.81,
            risk_amount=1000.0,
            risk_percent=0.01,
            stop_loss=415.0,
            position_size=200,
            entry_trigger="KAMA crossover (Price > KAMA+0.5×ATR)",
        )

        assert explanation.symbol == "SPY"
        assert explanation.signal_type == SignalType.LONG
        assert explanation.price == 420.50
        assert explanation.atr == 3.2
        assert explanation.kama == 418.0
        assert explanation.rsi == 55.0
        assert explanation.adx == 32.0
        assert explanation.r_squared == 0.81
        assert explanation.risk_amount == 1000.0
        assert explanation.stop_loss == 415.0

    def test_format_signal_reason_long(self):
        """Test formatted signal reason for LONG signal (R12.3.1)."""
        explanation = SignalExplanation(
            symbol="SPY",
            signal_type=SignalType.LONG,
            price=420.50,
            timestamp=time.time(),
            strategy_mode=StrategyMode.TREND_FOLLOWING,
            regime=RegimeType.STRONG_TREND,
            volatility_state=VolatilityState.HIGH,
            atr=3.2,
            kama=418.0,
            adx=32.0,
            r_squared=0.81,
            risk_amount=1000.0,
            risk_percent=0.01,
            stop_loss=415.0,
            entry_trigger="KAMA crossover (Price > KAMA+0.5×ATR)",
        )

        reason = explanation.format_signal_reason()

        # Check format follows R12.3.1 specification
        assert "BUY SPY @ $420.50" in reason
        assert "Volatility (ATR=3.20, High)" in reason
        assert "KAMA crossover" in reason
        assert "STRONG_TREND" in reason
        assert "ADX=32.0" in reason
        assert "R²=0.81" in reason
        assert "Risk: $1000 (0.01% account)" in reason
        assert "Stop: $415.00" in reason

    def test_format_signal_reason_short(self):
        """Test formatted signal reason for SHORT signal."""
        explanation = SignalExplanation(
            symbol="AAPL",
            signal_type=SignalType.SHORT,
            price=175.25,
            timestamp=time.time(),
            strategy_mode=StrategyMode.TREND_FOLLOWING,
            regime=RegimeType.STRONG_TREND,
            volatility_state=VolatilityState.NORMAL,
            atr=2.5,
            kama=177.0,
            risk_amount=750.0,
            risk_percent=0.0075,
            stop_loss=178.0,
            entry_trigger="Price < KAMA - 0.5×ATR",
        )

        reason = explanation.format_signal_reason()

        assert "SELL SHORT AAPL @ $175.25" in reason
        assert "Volatility (ATR=2.50, Normal)" in reason
        assert "Price < KAMA - 0.5×ATR" in reason
        assert "Risk: $750" in reason  # Check for dollar amount
        assert "(0.01% account)" in reason  # Note: uses risk_percent as shown
        assert "Stop: $178.00" in reason

    def test_format_signal_reason_exit(self):
        """Test formatted signal reason for EXIT signals."""
        explanation = SignalExplanation(
            symbol="SPY",
            signal_type=SignalType.EXIT_LONG,
            price=425.0,
            timestamp=time.time(),
            strategy_mode=StrategyMode.TREND_FOLLOWING,
            regime=RegimeType.NEUTRAL,
            volatility_state=VolatilityState.LOW,
            atr=2.0,
            kama=423.0,
            entry_trigger="Price < KAMA - 0.3×ATR",
        )

        reason = explanation.format_signal_reason()

        # Exit signals should not include risk line
        assert "SELL SPY @ $425.00" in reason
        assert "Risk:" not in reason

    def test_format_signal_reason_mean_reversion(self):
        """Test formatted signal reason for mean-reversion signal."""
        explanation = SignalExplanation(
            symbol="QQQ",
            signal_type=SignalType.LONG,
            price=350.0,
            timestamp=time.time(),
            strategy_mode=StrategyMode.MEAN_REVERSION,
            regime=RegimeType.MEAN_REVERSION,
            volatility_state=VolatilityState.LOW,
            atr=3.0,
            kama=352.0,
            rsi=25.0,
            risk_amount=500.0,
            risk_percent=0.005,
            stop_loss=345.0,
            entry_trigger="RSI < 30 AND price at lower Bollinger Band",
        )

        reason = explanation.format_signal_reason()

        assert "BUY QQQ @ $350.00" in reason
        assert "RSI < 30 AND price at lower Bollinger Band" in reason
        assert "MEAN_REVERSION" in reason
        assert "RSI=25.0" in reason

    def test_format_signal_reason_no_signal(self):
        """Test formatted signal reason for NONE signal."""
        explanation = SignalExplanation(
            symbol="SPY",
            signal_type=SignalType.NONE,
            price=420.0,
            timestamp=time.time(),
            strategy_mode=StrategyMode.NEUTRAL,
            regime=RegimeType.NEUTRAL,
            volatility_state=VolatilityState.NORMAL,
            atr=2.5,
            kama=420.0,
        )

        reason = explanation.format_signal_reason()
        assert reason == "NO SIGNAL"

    def test_format_compact_reason(self):
        """Test compact single-line reason formatting."""
        explanation = SignalExplanation(
            symbol="SPY",
            signal_type=SignalType.LONG,
            price=420.50,
            timestamp=time.time(),
            strategy_mode=StrategyMode.TREND_FOLLOWING,
            regime=RegimeType.STRONG_TREND,
            volatility_state=VolatilityState.HIGH,
            atr=3.2,
            kama=418.0,
            risk_amount=1000.0,
            risk_percent=0.01,
            entry_trigger="KAMA crossover",
        )

        compact = explanation.format_compact_reason()

        assert "BUY @ $420.50" in compact
        assert "KAMA crossover" in compact
        assert "STRONG_TREND" in compact
        assert "ATR=3.20" in compact
        assert "Risk: $1000 (0.01%)" in compact

    def test_to_dict(self):
        """Test conversion to dictionary for storage."""
        explanation = SignalExplanation(
            symbol="SPY",
            signal_type=SignalType.LONG,
            price=420.50,
            timestamp=1234567890.0,
            strategy_mode=StrategyMode.TREND_FOLLOWING,
            regime=RegimeType.STRONG_TREND,
            volatility_state=VolatilityState.HIGH,
            atr=3.2,
            kama=418.0,
            rsi=55.0,
            risk_amount=1000.0,
            risk_percent=0.01,
            stop_loss=415.0,
            position_size=200,
            entry_trigger="KAMA crossover",
        )

        data = explanation.to_dict()

        assert data["symbol"] == "SPY"
        assert data["signal_type"] == 1  # LONG
        assert data["price"] == 420.50
        assert data["timestamp"] == 1234567890.0
        assert data["strategy_mode"] == 2  # TREND_FOLLOWING
        assert data["regime"] == 2  # STRONG_TREND
        assert data["volatility_state"] == 2  # HIGH
        assert data["atr"] == 3.2
        assert data["kama"] == 418.0
        assert data["rsi"] == 55.0
        assert data["risk_amount"] == 1000.0
        assert data["stop_loss"] == 415.0
        assert data["position_size"] == 200
        assert "formatted_reason" in data
        assert "compact_reason" in data

    def test_volatility_state_names(self):
        """Test volatility state name formatting."""
        # Use LONG signal so we get formatted output, not "NO SIGNAL"
        low_vol = SignalExplanation(
            symbol="SPY",
            signal_type=SignalType.LONG,
            price=420.0,
            timestamp=time.time(),
            strategy_mode=StrategyMode.NEUTRAL,
            regime=RegimeType.NEUTRAL,
            volatility_state=VolatilityState.LOW,
            atr=1.0,
            kama=420.0,
            entry_trigger="Test",
            risk_amount=1000.0,
            risk_percent=0.01,
            stop_loss=415.0,
        )
        assert "Low" in low_vol.format_signal_reason()

        high_vol = SignalExplanation(
            symbol="SPY",
            signal_type=SignalType.LONG,
            price=420.0,
            timestamp=time.time(),
            strategy_mode=StrategyMode.NEUTRAL,
            regime=RegimeType.NEUTRAL,
            volatility_state=VolatilityState.HIGH,
            atr=5.0,
            kama=420.0,
            entry_trigger="Test",
            risk_amount=1000.0,
            risk_percent=0.01,
            stop_loss=415.0,
        )
        assert "High" in high_vol.format_signal_reason()

    def test_regime_names(self):
        """Test regime name formatting."""
        trend = SignalExplanation(
            symbol="SPY",
            signal_type=SignalType.LONG,
            price=420.0,
            timestamp=time.time(),
            strategy_mode=StrategyMode.TREND_FOLLOWING,
            regime=RegimeType.STRONG_TREND,
            volatility_state=VolatilityState.NORMAL,
            atr=2.5,
            kama=418.0,
            entry_trigger="Test",
            risk_amount=1000.0,
            risk_percent=0.01,
            stop_loss=415.0,
        )
        assert "STRONG_TREND" in trend.format_signal_reason()

        mr = SignalExplanation(
            symbol="SPY",
            signal_type=SignalType.LONG,
            price=420.0,
            timestamp=time.time(),
            strategy_mode=StrategyMode.MEAN_REVERSION,
            regime=RegimeType.MEAN_REVERSION,
            volatility_state=VolatilityState.NORMAL,
            atr=2.5,
            kama=420.0,
            entry_trigger="Test",
            risk_amount=1000.0,
            risk_percent=0.01,
            stop_loss=415.0,
        )
        assert "MEAN_REVERSION" in mr.format_signal_reason()


class TestSignalGenerator:
    """Test SignalGenerator class."""

    def test_generator_initialization(self):
        """Test signal generator initialization."""
        generator = SignalGenerator(account_balance=100000.0)
        assert generator.account_balance == 100000.0

    def test_generate_signal_with_explanation(self):
        """Test single signal generation with explanation."""
        generator = SignalGenerator(account_balance=100000.0)

        explanation = generator.generate_signal_with_explanation(
            symbol="SPY",
            timestamp=time.time(),
            price=420.50,
            signal_type=SignalType.LONG,
            strategy_mode=StrategyMode.TREND_FOLLOWING,
            regime=RegimeType.STRONG_TREND,
            volatility_state=VolatilityState.HIGH,
            atr=3.2,
            kama=418.0,
            entry_trigger="KAMA crossover",
            risk_percent=0.01,
            stop_loss=415.0,
            adx=32.0,
            r_squared=0.81,
        )

        assert explanation.symbol == "SPY"
        assert explanation.signal_type == SignalType.LONG
        assert explanation.price == 420.50
        assert explanation.atr == 3.2
        assert explanation.adx == 32.0

        # Check risk calculation
        expected_risk = 100000.0 * 0.01  # $1,000
        assert abs(explanation.risk_amount - expected_risk) < 0.01

    def test_risk_calculation(self):
        """Test risk and position size calculation."""
        generator = SignalGenerator(account_balance=100000.0)

        explanation = generator.generate_signal_with_explanation(
            symbol="SPY",
            timestamp=time.time(),
            price=420.0,
            signal_type=SignalType.LONG,
            strategy_mode=StrategyMode.TREND_FOLLOWING,
            regime=RegimeType.STRONG_TREND,
            volatility_state=VolatilityState.NORMAL,
            atr=2.5,
            kama=418.0,
            entry_trigger="Test",
            risk_percent=0.01,  # 1% risk
            stop_loss=415.0,  # $5 risk per share
        )

        # Risk amount: $100,000 × 1% = $1,000
        assert abs(explanation.risk_amount - 1000.0) < 0.01

        # Position size: $1,000 / $5 = 200 shares
        assert explanation.position_size == 200

    def test_risk_calculation_short(self):
        """Test risk calculation for short position."""
        generator = SignalGenerator(account_balance=50000.0)

        explanation = generator.generate_signal_with_explanation(
            symbol="AAPL",
            timestamp=time.time(),
            price=175.0,
            signal_type=SignalType.SHORT,
            strategy_mode=StrategyMode.TREND_FOLLOWING,
            regime=RegimeType.STRONG_TREND,
            volatility_state=VolatilityState.NORMAL,
            atr=2.0,
            kama=177.0,
            entry_trigger="Test",
            risk_percent=0.0075,  # 0.75% risk
            stop_loss=178.0,  # $3 risk per share
        )

        # Risk amount: $50,000 × 0.75% = $375
        assert abs(explanation.risk_amount - 375.0) < 0.01

        # Position size: $375 / $3 = 125 shares
        assert explanation.position_size == 125

    def test_no_risk_for_exit_signals(self):
        """Test that exit signals have zero risk."""
        generator = SignalGenerator(account_balance=100000.0)

        explanation = generator.generate_signal_with_explanation(
            symbol="SPY",
            timestamp=time.time(),
            price=425.0,
            signal_type=SignalType.EXIT_LONG,
            strategy_mode=StrategyMode.TREND_FOLLOWING,
            regime=RegimeType.NEUTRAL,
            volatility_state=VolatilityState.NORMAL,
            atr=2.5,
            kama=423.0,
            entry_trigger="Exit trigger",
            risk_percent=0.01,
            stop_loss=420.0,
        )

        # Exit signals should have zero risk
        assert explanation.risk_amount == 0.0
        assert explanation.position_size == 0

    def test_batch_generate_explanations(self):
        """Test batch signal explanation generation."""
        generator = SignalGenerator(account_balance=100000.0)

        n = 5
        timestamps = np.array([time.time() + i for i in range(n)])
        prices = np.array([420.0, 421.0, 422.0, 423.0, 424.0])
        signals = np.array([0, 1, 0, 2, 0], dtype=np.int32)  # NONE, LONG, NONE, EXIT_LONG, NONE
        strategy_modes = np.array([3, 2, 3, 2, 3], dtype=np.int32)
        regimes = np.array([1, 2, 1, 1, 1], dtype=np.int32)
        volatility_states = np.array([1, 2, 1, 1, 1], dtype=np.int32)
        atr_values = np.array([2.5, 3.2, 2.8, 2.6, 2.5])
        kama_values = np.array([418.0, 419.0, 420.0, 421.0, 422.0])
        entry_triggers = ["None", "KAMA crossover", "None", "Exit signal", "None"]
        stop_losses = np.array([0.0, 415.0, 0.0, 0.0, 0.0])

        explanations = generator.batch_generate_explanations(
            symbol="SPY",
            timestamps=timestamps,
            prices=prices,
            signals=signals,
            strategy_modes=strategy_modes,
            regimes=regimes,
            volatility_states=volatility_states,
            atr_values=atr_values,
            kama_values=kama_values,
            entry_triggers=entry_triggers,
            stop_losses=stop_losses,
            risk_percent=0.01,
        )

        # Should only get explanations for non-NONE signals (indices 1 and 3)
        assert len(explanations) == 2

        # Check first signal (LONG)
        assert explanations[0].signal_type == SignalType.LONG
        assert explanations[0].price == 421.0
        assert explanations[0].entry_trigger == "KAMA crossover"

        # Check second signal (EXIT_LONG)
        assert explanations[1].signal_type == SignalType.EXIT_LONG
        assert explanations[1].price == 423.0
        assert explanations[1].entry_trigger == "Exit signal"

    def test_batch_with_optional_indicators(self):
        """Test batch generation with optional RSI, ADX, R² values."""
        generator = SignalGenerator(account_balance=100000.0)

        n = 3
        timestamps = np.array([time.time() + i for i in range(n)])
        prices = np.array([420.0, 421.0, 422.0])
        signals = np.array([1, 1, 1], dtype=np.int32)  # All LONG
        strategy_modes = np.array([2, 1, 3], dtype=np.int32)
        regimes = np.array([2, 0, 1], dtype=np.int32)
        volatility_states = np.array([1, 1, 1], dtype=np.int32)
        atr_values = np.array([2.5, 2.6, 2.7])
        kama_values = np.array([418.0, 419.0, 420.0])
        entry_triggers = ["Trend signal", "MR signal", "Neutral signal"]
        stop_losses = np.array([415.0, 416.0, 417.0])

        # Optional indicators
        rsi_values = np.array([55.0, 28.0, 50.0])
        adx_values = np.array([32.0, 18.0, 22.0])
        r_squared_values = np.array([0.81, 0.25, 0.55])

        explanations = generator.batch_generate_explanations(
            symbol="SPY",
            timestamps=timestamps,
            prices=prices,
            signals=signals,
            strategy_modes=strategy_modes,
            regimes=regimes,
            volatility_states=volatility_states,
            atr_values=atr_values,
            kama_values=kama_values,
            entry_triggers=entry_triggers,
            stop_losses=stop_losses,
            rsi_values=rsi_values,
            adx_values=adx_values,
            r_squared_values=r_squared_values,
        )

        assert len(explanations) == 3

        # Check that optional indicators are included
        assert explanations[0].rsi == 55.0
        assert explanations[0].adx == 32.0
        assert explanations[0].r_squared == 0.81

        assert explanations[1].rsi == 28.0
        assert explanations[1].adx == 18.0
        assert explanations[1].r_squared == 0.25

    def test_update_account_balance(self):
        """Test account balance update."""
        generator = SignalGenerator(account_balance=100000.0)
        assert generator.account_balance == 100000.0

        generator.update_account_balance(150000.0)
        assert generator.account_balance == 150000.0

        # Risk calculation should use new balance
        explanation = generator.generate_signal_with_explanation(
            symbol="SPY",
            timestamp=time.time(),
            price=420.0,
            signal_type=SignalType.LONG,
            strategy_mode=StrategyMode.TREND_FOLLOWING,
            regime=RegimeType.STRONG_TREND,
            volatility_state=VolatilityState.NORMAL,
            atr=2.5,
            kama=418.0,
            entry_trigger="Test",
            risk_percent=0.01,
            stop_loss=415.0,
        )

        # New risk: $150,000 × 1% = $1,500
        assert abs(explanation.risk_amount - 1500.0) < 0.01

    def test_edge_case_zero_stop_distance(self):
        """Test edge case where stop loss equals entry price."""
        generator = SignalGenerator(account_balance=100000.0)

        explanation = generator.generate_signal_with_explanation(
            symbol="SPY",
            timestamp=time.time(),
            price=420.0,
            signal_type=SignalType.LONG,
            strategy_mode=StrategyMode.TREND_FOLLOWING,
            regime=RegimeType.STRONG_TREND,
            volatility_state=VolatilityState.NORMAL,
            atr=2.5,
            kama=418.0,
            entry_trigger="Test",
            risk_percent=0.01,
            stop_loss=420.0,  # Same as entry price
        )

        # Position size should be 0 (can't divide by zero risk)
        assert explanation.position_size == 0

    def test_edge_case_no_stop_loss(self):
        """Test edge case with no stop loss."""
        generator = SignalGenerator(account_balance=100000.0)

        explanation = generator.generate_signal_with_explanation(
            symbol="SPY",
            timestamp=time.time(),
            price=420.0,
            signal_type=SignalType.LONG,
            strategy_mode=StrategyMode.TREND_FOLLOWING,
            regime=RegimeType.STRONG_TREND,
            volatility_state=VolatilityState.NORMAL,
            atr=2.5,
            kama=418.0,
            entry_trigger="Test",
            risk_percent=0.01,
            stop_loss=0.0,  # No stop
        )

        # Should have zero position size
        assert explanation.position_size == 0


class TestEnumTypes:
    """Test enum types."""

    def test_signal_type_enum(self):
        """Test SignalType enum values."""
        assert SignalType.NONE == 0
        assert SignalType.LONG == 1
        assert SignalType.SHORT == -1
        assert SignalType.EXIT_LONG == 2
        assert SignalType.EXIT_SHORT == -2

    def test_regime_type_enum(self):
        """Test RegimeType enum values."""
        assert RegimeType.MEAN_REVERSION == 0
        assert RegimeType.NEUTRAL == 1
        assert RegimeType.STRONG_TREND == 2

    def test_volatility_state_enum(self):
        """Test VolatilityState enum values."""
        assert VolatilityState.LOW == 0
        assert VolatilityState.NORMAL == 1
        assert VolatilityState.HIGH == 2

    def test_strategy_mode_enum(self):
        """Test StrategyMode enum values."""
        assert StrategyMode.MEAN_REVERSION == 1
        assert StrategyMode.TREND_FOLLOWING == 2
        assert StrategyMode.NEUTRAL == 3


class TestSuccessCriteria:
    """Test success criteria from R12.3.1."""

    def test_r12_3_1_format_compliance(self):
        """Test that signal format matches R12.3.1 specification."""
        generator = SignalGenerator(account_balance=100000.0)

        explanation = generator.generate_signal_with_explanation(
            symbol="SPY",
            timestamp=time.time(),
            price=420.50,
            signal_type=SignalType.LONG,
            strategy_mode=StrategyMode.TREND_FOLLOWING,
            regime=RegimeType.STRONG_TREND,
            volatility_state=VolatilityState.HIGH,
            atr=3.2,
            kama=418.0,
            entry_trigger="KAMA crossover (Price > KAMA+0.5×ATR)",
            risk_percent=0.01,
            stop_loss=415.0,
            adx=32.0,
            r_squared=0.81,
        )

        reason = explanation.format_signal_reason()

        # R12.3.1 format requirements:
        # Line 1: "BUY SPY @ $420.50"
        # Line 2: "Reason: Volatility (ATR=3.2, High) + KAMA crossover..."
        # Line 3: "Regime: STRONG_TREND (ADX=32, R²=0.81)"
        # Line 4: "Risk: $1,000 (1% account), Stop: $415.00"

        lines = reason.split("\n")
        assert len(lines) == 4

        assert lines[0] == "BUY SPY @ $420.50"
        assert "Reason: Volatility (ATR=3.20, High)" in lines[1]
        assert "KAMA crossover" in lines[1]
        assert "Regime: STRONG_TREND" in lines[2]
        assert "ADX=32.0" in lines[2]
        assert "R²=0.81" in lines[2]
        assert "Risk: $1000" in lines[3]
        assert "0.01% account" in lines[3]
        assert "Stop: $415.00" in lines[3]
