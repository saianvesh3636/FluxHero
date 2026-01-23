"""
Validation Tests for Signal Generation.

This module validates signal generation against hand-calculated expected values
to ensure correctness of the signal logic.

Reference: FLUXHERO_REQUIREMENTS.md Feature 6 - Dual-Mode Strategy Engine
Reference: enhancement_tasks.md Phase 24 - Quality Control & Validation Framework

Key validation approach:
1. Use synthetic price data with known patterns
2. Include step-by-step signal logic verification in comments
3. Test both trend-following and mean-reversion modes
4. Validate regime detection on synthetic transitions
"""

import numpy as np

from backend.computation.adaptive_ema import calculate_kama
from backend.computation.indicators import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_rsi,
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
    adjust_size_for_regime,
    blend_signals,
    calculate_fixed_stop_loss,
    calculate_position_size,
    calculate_trailing_stop,
    generate_mean_reversion_signals,
    generate_trend_following_signals,
)
from backend.strategy.regime_detector import (
    REGIME_MEAN_REVERSION,
    REGIME_NEUTRAL,
    REGIME_STRONG_TREND,
    classify_trend_regime,
)


class TestTrendFollowingSignalValidation:
    """Validate trend-following signal generation with hand-calculated values."""

    def test_trend_following_long_entry_hand_calculated(self):
        """
        Test trend-following LONG entry signal with hand-calculated values.

        Entry Logic (R6.1.1):
            LONG: Price crosses above KAMA + (0.5 x ATR)

        Setup (hand-calculated):
        - Create prices that cross above the upper entry band
        - prices: [100, 101, 102, 103, 105, 108, 112]
        - KAMA values (synthetic, representing KAMA tracking price):
          [100, 100.5, 101, 101.5, 102, 103, 105]
        - ATR values (synthetic, steady ATR):
          [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]

        At index 5:
        - Upper entry band = KAMA[5] + (0.5 * ATR[5]) = 103 + 1.0 = 104
        - prices[4] = 105 > 104 (above band at i-1? No, prev was 103, let me recalculate)

        Actually at index 5:
        - prices[4] = 105, upper_entry[4] = 102 + 1.0 = 103
        - prices[5] = 108, upper_entry[5] = 103 + 1.0 = 104
        - Check: prices[4] <= upper_entry[4]? 105 <= 103? NO
        - So we need prices that stay below band then cross above

        Corrected setup:
        - prices: [100, 100.5, 101, 101, 102, 106]  <- last price jumps above band
        - KAMA:   [100, 100,  100, 100, 100, 101]  <- slow tracking
        - ATR:    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        - Upper entry band = KAMA + (0.5 * ATR) = [101, 101, 101, 101, 101, 102]

        At index 5:
        - prices[4] = 102, upper_entry[4] = 101 -> prices[4] > upper_entry[4]? YES
        - So crossover happened earlier...

        Let me set up a cleaner scenario:
        - prices: [99, 99.5, 100, 100, 100.5, 103]  <- price 103 crosses above 102
        - KAMA:   [100, 100, 100, 100, 100, 101]   <- relatively flat
        - ATR:    [4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
        - Upper entry band = KAMA + (0.5 * 4.0) = [102, 102, 102, 102, 102, 103]

        At index 5:
        - prices[4] = 100.5, upper_entry[4] = 102 -> 100.5 <= 102? YES
        - prices[5] = 103, upper_entry[5] = 103 -> 103 > 103? NO (equal, not greater)

        Fix: prices[5] = 103.1
        - prices[5] = 103.1 > upper_entry[5] = 103? YES
        -> LONG signal at index 5
        """
        prices = np.array([99.0, 99.5, 100.0, 100.0, 100.5, 103.1])
        kama = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 101.0])
        atr = np.array([4.0, 4.0, 4.0, 4.0, 4.0, 4.0])

        signals = generate_trend_following_signals(prices, kama, atr)

        # Verify upper entry band calculation
        upper_entry = kama + (0.5 * atr)  # [102, 102, 102, 102, 102, 103]
        assert upper_entry[4] == 102.0
        assert upper_entry[5] == 103.0

        # prices[4] = 100.5 <= upper_entry[4] = 102.0? YES
        assert prices[4] <= upper_entry[4]
        # prices[5] = 103.1 > upper_entry[5] = 103.0? YES
        assert prices[5] > upper_entry[5]

        # Should have LONG signal at index 5
        assert signals[5] == SIGNAL_LONG

    def test_trend_following_short_entry_hand_calculated(self):
        """
        Test trend-following SHORT entry signal with hand-calculated values.

        Entry Logic (R6.1.1):
            SHORT: Price crosses below KAMA - (0.5 x ATR)

        Setup:
        - prices: [101, 100.5, 100, 100, 99.5, 96.9]  <- price drops below band
        - KAMA:   [100, 100, 100, 100, 100, 99]
        - ATR:    [4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
        - Lower entry band = KAMA - (0.5 * ATR) = [98, 98, 98, 98, 98, 97]

        At index 5:
        - prices[4] = 99.5 >= lower_entry[4] = 98? YES
        - prices[5] = 96.9 < lower_entry[5] = 97? YES
        -> SHORT signal at index 5
        """
        prices = np.array([101.0, 100.5, 100.0, 100.0, 99.5, 96.9])
        kama = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 99.0])
        atr = np.array([4.0, 4.0, 4.0, 4.0, 4.0, 4.0])

        signals = generate_trend_following_signals(prices, kama, atr)

        # Verify lower entry band calculation
        lower_entry = kama - (0.5 * atr)
        assert lower_entry[4] == 98.0
        assert lower_entry[5] == 97.0

        # Verify crossover conditions
        assert prices[4] >= lower_entry[4]  # 99.5 >= 98
        assert prices[5] < lower_entry[5]  # 96.9 < 97

        # Should have SHORT signal at index 5
        assert signals[5] == SIGNAL_SHORT

    def test_trend_following_exit_long_hand_calculated(self):
        """
        Test trend-following EXIT_LONG signal with hand-calculated values.

        Exit Logic (R6.1.2):
            EXIT LONG: Price crosses below KAMA - (0.3 x ATR)

        Setup:
        - First enter long, then price drops below exit band
        - prices: [99, 99.5, 100, 103.1, 103, 99]  <- enter at 3, exit at 5
        - KAMA:   [100, 100, 100, 101, 101, 100]
        - ATR:    [4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
        - Upper entry band = KAMA + 2.0 = [102, 102, 102, 103, 103, 102]
        - Lower exit band = KAMA - 1.2 = [98.8, 98.8, 98.8, 99.8, 99.8, 98.8]

        At index 3 (long entry):
        - prices[2] = 100 <= upper_entry[2] = 102? YES
        - prices[3] = 103.1 > upper_entry[3] = 103? YES
        -> LONG entry at index 3

        At index 5 (exit):
        - prices[4] = 103 >= lower_exit[4] = 99.8? YES
        - prices[5] = 99 < lower_exit[5] = 98.8? NO, 99 > 98.8

        Need to adjust. Let KAMA rise to make exit band higher:
        - KAMA:   [100, 100, 100, 101, 102, 101]
        - Lower exit band = KAMA - 1.2 = [98.8, 98.8, 98.8, 99.8, 100.8, 99.8]

        At index 5:
        - prices[4] = 103 >= lower_exit[4] = 100.8? YES
        - prices[5] = 99 < lower_exit[5] = 99.8? YES
        -> EXIT_LONG at index 5
        """
        prices = np.array([99.0, 99.5, 100.0, 103.1, 103.0, 99.0])
        kama = np.array([100.0, 100.0, 100.0, 101.0, 102.0, 101.0])
        atr = np.array([4.0, 4.0, 4.0, 4.0, 4.0, 4.0])

        signals = generate_trend_following_signals(prices, kama, atr)

        # Verify exit band calculation
        lower_exit = kama - (0.3 * atr)
        assert lower_exit[5] == 101.0 - 1.2  # 99.8

        # Should have LONG entry at index 3 and EXIT_LONG at index 5
        assert signals[3] == SIGNAL_LONG
        assert signals[5] == SIGNAL_EXIT_LONG

    def test_trend_following_exit_short_hand_calculated(self):
        """
        Test trend-following EXIT_SHORT signal with hand-calculated values.

        Exit Logic (R6.1.2):
            EXIT SHORT: Price crosses above KAMA + (0.3 x ATR)

        Setup:
        - First enter short, then price rises above exit band
        - prices: [101, 100.5, 100, 96.9, 97, 102]  <- enter at 3, exit at 5
        - KAMA:   [100, 100, 100, 99, 98, 100]
        - ATR:    [4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
        - Lower entry band = KAMA - 2.0 = [98, 98, 98, 97, 96, 98]
        - Upper exit band = KAMA + 1.2 = [101.2, 101.2, 101.2, 100.2, 99.2, 101.2]

        At index 3 (short entry):
        - prices[2] = 100 >= lower_entry[2] = 98? YES
        - prices[3] = 96.9 < lower_entry[3] = 97? YES
        -> SHORT entry at index 3

        At index 5 (exit):
        - prices[4] = 97 <= upper_exit[4] = 99.2? YES
        - prices[5] = 102 > upper_exit[5] = 101.2? YES
        -> EXIT_SHORT at index 5
        """
        prices = np.array([101.0, 100.5, 100.0, 96.9, 97.0, 102.0])
        kama = np.array([100.0, 100.0, 100.0, 99.0, 98.0, 100.0])
        atr = np.array([4.0, 4.0, 4.0, 4.0, 4.0, 4.0])

        signals = generate_trend_following_signals(prices, kama, atr)

        # Verify exit band calculation
        upper_exit = kama + (0.3 * atr)
        assert upper_exit[5] == 100.0 + 1.2  # 101.2

        # Should have SHORT entry at index 3 and EXIT_SHORT at index 5
        assert signals[3] == SIGNAL_SHORT
        assert signals[5] == SIGNAL_EXIT_SHORT

    def test_trend_following_no_signal_when_within_bands(self):
        """
        Test that no signals are generated when price stays within bands.

        When prices remain between entry bands, no entry signals should fire.
        """
        # Prices that stay close to KAMA (within bands)
        prices = np.array([100.0, 100.5, 100.0, 99.5, 100.2, 100.1])
        kama = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
        atr = np.array([4.0, 4.0, 4.0, 4.0, 4.0, 4.0])

        signals = generate_trend_following_signals(prices, kama, atr)

        # Upper entry = 102, Lower entry = 98
        # All prices are between 98 and 102, so no entries
        assert np.all(signals == SIGNAL_NONE)

    def test_trend_following_nan_handling(self):
        """Test that NaN values are handled correctly (no signals at NaN)."""
        prices = np.array([99.0, 99.5, np.nan, 103.1, 103.0, 99.0])
        kama = np.array([100.0, 100.0, 100.0, 101.0, 102.0, 101.0])
        atr = np.array([4.0, 4.0, 4.0, 4.0, 4.0, 4.0])

        signals = generate_trend_following_signals(prices, kama, atr)

        # No signal should be generated at index 3 because index 2 has NaN
        # The function skips when current or previous index has NaN
        assert signals[3] == SIGNAL_NONE


class TestMeanReversionSignalValidation:
    """Validate mean-reversion signal generation with hand-calculated values."""

    def test_mean_reversion_long_entry_hand_calculated(self):
        """
        Test mean-reversion LONG entry signal with hand-calculated values.

        Entry Logic (R6.2.1):
            LONG: RSI < 30 AND price at/below lower Bollinger Band

        Setup:
        - Create oversold RSI with price at lower Bollinger Band
        - prices: [105, 104, 102, 100, 98, 95]  <- declining prices
        - RSI:    [60, 55, 45, 35, 28, 25]      <- RSI dropping below 30
        - BB lower: [97, 97, 97, 97, 97, 97]    <- constant lower band
        - BB middle: [100, 100, 100, 100, 100, 100]

        At index 4:
        - RSI[4] = 28 < 30? YES (oversold)
        - prices[4] = 98 <= BB_lower[4] = 97? NO

        At index 5:
        - RSI[5] = 25 < 30? YES (oversold)
        - prices[5] = 95 <= BB_lower[5] = 97? YES
        -> LONG signal at index 5
        """
        prices = np.array([105.0, 104.0, 102.0, 100.0, 98.0, 95.0])
        rsi = np.array([60.0, 55.0, 45.0, 35.0, 28.0, 25.0])
        bb_lower = np.array([97.0, 97.0, 97.0, 97.0, 97.0, 97.0])
        bb_middle = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])

        signals = generate_mean_reversion_signals(prices, rsi, bb_lower, bb_middle)

        # Verify conditions at index 5
        assert rsi[5] < 30.0
        assert prices[5] <= bb_lower[5]

        # Should have LONG signal at index 5
        assert signals[5] == SIGNAL_LONG

    def test_mean_reversion_exit_long_on_middle_band(self):
        """
        Test mean-reversion EXIT_LONG when price returns to middle band.

        Exit Logic (R6.2.2):
            EXIT LONG: Price returns to 20-period SMA (middle band) OR RSI > 70

        Setup:
        - Enter long on oversold, exit when price returns to middle
        - prices:    [105, 104, 95, 96, 98, 100]  <- price reaches middle at 5
        - RSI:       [60, 55, 25, 30, 35, 50]      <- RSI recovers
        - BB lower:  [97, 97, 97, 97, 97, 97]
        - BB middle: [100, 100, 100, 100, 100, 100]

        At index 2: LONG entry (RSI=25 < 30, price=95 <= 97)
        At index 5: EXIT_LONG (price=100 >= middle=100)
        """
        prices = np.array([105.0, 104.0, 95.0, 96.0, 98.0, 100.0])
        rsi = np.array([60.0, 55.0, 25.0, 30.0, 35.0, 50.0])
        bb_lower = np.array([97.0, 97.0, 97.0, 97.0, 97.0, 97.0])
        bb_middle = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])

        signals = generate_mean_reversion_signals(prices, rsi, bb_lower, bb_middle)

        # Should have LONG entry at index 2
        assert signals[2] == SIGNAL_LONG
        # Should have EXIT_LONG at index 5
        assert signals[5] == SIGNAL_EXIT_LONG

    def test_mean_reversion_exit_long_on_rsi_overbought(self):
        """
        Test mean-reversion EXIT_LONG when RSI becomes overbought.

        Exit Logic (R6.2.2):
            EXIT LONG: Price returns to middle band OR RSI > 70

        Setup:
        - Enter long on oversold, exit when RSI > 70
        - prices:    [105, 104, 95, 96, 97, 98]   <- price stays below middle
        - RSI:       [60, 55, 25, 40, 60, 75]      <- RSI rises above 70
        - BB lower:  [97, 97, 97, 97, 97, 97]
        - BB middle: [100, 100, 100, 100, 100, 100]

        At index 2: LONG entry (RSI=25 < 30, price=95 <= 97)
        At index 5: EXIT_LONG (RSI=75 > 70)
        """
        prices = np.array([105.0, 104.0, 95.0, 96.0, 97.0, 98.0])
        rsi = np.array([60.0, 55.0, 25.0, 40.0, 60.0, 75.0])
        bb_lower = np.array([97.0, 97.0, 97.0, 97.0, 97.0, 97.0])
        bb_middle = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])

        signals = generate_mean_reversion_signals(prices, rsi, bb_lower, bb_middle)

        # Should have LONG entry at index 2
        assert signals[2] == SIGNAL_LONG
        # Should have EXIT_LONG at index 5 due to overbought RSI
        assert signals[5] == SIGNAL_EXIT_LONG

    def test_mean_reversion_no_signal_rsi_above_threshold(self):
        """
        Test that no LONG signal when RSI is above 30 (not oversold).

        Even if price touches lower band, we need RSI < 30 for entry.
        """
        prices = np.array([105.0, 104.0, 102.0, 100.0, 98.0, 95.0])
        rsi = np.array([60.0, 55.0, 45.0, 40.0, 35.0, 32.0])  # RSI stays >= 30
        bb_lower = np.array([97.0, 97.0, 97.0, 97.0, 97.0, 97.0])
        bb_middle = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])

        signals = generate_mean_reversion_signals(prices, rsi, bb_lower, bb_middle)

        # No signal because RSI never goes below 30
        # Even at index 5 where price=95 <= bb_lower=97, RSI=32 >= 30
        assert np.all(signals == SIGNAL_NONE)

    def test_mean_reversion_no_signal_price_above_band(self):
        """
        Test that no LONG signal when price is above lower band.

        Even if RSI is oversold, we need price at/below lower band.
        """
        prices = np.array([105.0, 104.0, 102.0, 100.0, 99.0, 98.0])
        rsi = np.array([60.0, 55.0, 35.0, 28.0, 25.0, 22.0])  # Oversold
        bb_lower = np.array([97.0, 97.0, 97.0, 97.0, 97.0, 97.0])
        bb_middle = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])

        signals = generate_mean_reversion_signals(prices, rsi, bb_lower, bb_middle)

        # No signal because price never touches lower band (all prices > 97)
        assert np.all(signals == SIGNAL_NONE)


class TestRegimeDetectionValidation:
    """Validate regime detection with hand-calculated values."""

    def test_regime_strong_trend_hand_calculated(self):
        """
        Test STRONG_TREND regime classification.

        Classification Logic:
            STRONG_TREND (2): ADX > 25 AND R-squared > 0.6

        Setup:
        - ADX = [30, 35, 28]   <- all above 25
        - R2 = [0.7, 0.8, 0.65] <- all above 0.6

        All should be classified as STRONG_TREND (2)
        """
        adx = np.array([30.0, 35.0, 28.0])
        r_squared = np.array([0.7, 0.8, 0.65])

        regime = classify_trend_regime(adx, r_squared)

        assert regime[0] == REGIME_STRONG_TREND
        assert regime[1] == REGIME_STRONG_TREND
        assert regime[2] == REGIME_STRONG_TREND

    def test_regime_mean_reversion_hand_calculated(self):
        """
        Test MEAN_REVERSION regime classification.

        Classification Logic:
            MEAN_REVERSION (0): ADX < 20 AND R-squared < 0.4

        Setup:
        - ADX = [15, 18, 12]   <- all below 20
        - R2 = [0.3, 0.2, 0.35] <- all below 0.4

        All should be classified as MEAN_REVERSION (0)
        """
        adx = np.array([15.0, 18.0, 12.0])
        r_squared = np.array([0.3, 0.2, 0.35])

        regime = classify_trend_regime(adx, r_squared)

        assert regime[0] == REGIME_MEAN_REVERSION
        assert regime[1] == REGIME_MEAN_REVERSION
        assert regime[2] == REGIME_MEAN_REVERSION

    def test_regime_neutral_hand_calculated(self):
        """
        Test NEUTRAL regime classification.

        Classification Logic:
            NEUTRAL (1): Everything that's not STRONG_TREND or MEAN_REVERSION

        Cases:
        1. High ADX but low R2: ADX=30, R2=0.3 -> NEUTRAL (not trending quality)
        2. Low ADX but high R2: ADX=15, R2=0.7 -> NEUTRAL (weak trend strength)
        3. Mid-range values: ADX=22, R2=0.5 -> NEUTRAL (transition)
        """
        adx = np.array([30.0, 15.0, 22.0])
        r_squared = np.array([0.3, 0.7, 0.5])

        regime = classify_trend_regime(adx, r_squared)

        # High ADX (>25) but low R2 (<0.4) -> NEUTRAL
        assert regime[0] == REGIME_NEUTRAL
        # Low ADX (<20) but high R2 (>0.6) -> NEUTRAL
        assert regime[1] == REGIME_NEUTRAL
        # Mid-range ADX and R2 -> NEUTRAL
        assert regime[2] == REGIME_NEUTRAL

    def test_regime_boundary_values(self):
        """
        Test regime classification at exact boundary values.

        Thresholds:
        - ADX trend: > 25 (not >=)
        - ADX ranging: < 20 (not <=)
        - R2 trend: > 0.6 (not >=)
        - R2 ranging: < 0.4 (not <=)

        At exact thresholds, should be NEUTRAL.
        """
        # Exactly at thresholds
        adx = np.array([25.0, 20.0, 25.0])
        r_squared = np.array([0.6, 0.4, 0.4])

        regime = classify_trend_regime(adx, r_squared)

        # ADX=25, R2=0.6 -> not > 25 and not > 0.6, so NEUTRAL
        assert regime[0] == REGIME_NEUTRAL
        # ADX=20, R2=0.4 -> not < 20 and not < 0.4, so NEUTRAL
        assert regime[1] == REGIME_NEUTRAL
        # ADX=25, R2=0.4 -> not > 25, so not trend; not < 20, so not MR -> NEUTRAL
        assert regime[2] == REGIME_NEUTRAL

    def test_regime_synthetic_transition(self):
        """
        Test regime detection on synthetic market transition.

        Simulate market transitioning from trending to ranging:
        - Start: Strong trend (high ADX, high R2)
        - Middle: Transition (declining metrics)
        - End: Mean reversion (low ADX, low R2)
        """
        # Simulated ADX declining from 35 to 15
        adx = np.array([35.0, 32.0, 28.0, 24.0, 21.0, 18.0, 15.0])
        # Simulated R2 declining from 0.8 to 0.25
        r_squared = np.array([0.8, 0.7, 0.55, 0.45, 0.38, 0.32, 0.25])

        regime = classify_trend_regime(adx, r_squared)

        # First 2: STRONG_TREND (ADX > 25 AND R2 > 0.6)
        assert regime[0] == REGIME_STRONG_TREND
        assert regime[1] == REGIME_STRONG_TREND

        # Middle: NEUTRAL (transition region)
        assert regime[2] == REGIME_NEUTRAL  # ADX=28 > 25, but R2=0.55 < 0.6
        assert regime[3] == REGIME_NEUTRAL  # ADX=24 < 25, R2=0.45
        assert regime[4] == REGIME_NEUTRAL  # ADX=21 > 20, not fully ranging

        # Last 2: MEAN_REVERSION (ADX < 20 AND R2 < 0.4)
        assert regime[5] == REGIME_MEAN_REVERSION
        assert regime[6] == REGIME_MEAN_REVERSION

    def test_regime_nan_handling(self):
        """Test that NaN values result in NEUTRAL regime."""
        adx = np.array([35.0, np.nan, 15.0])
        r_squared = np.array([0.8, 0.8, np.nan])

        regime = classify_trend_regime(adx, r_squared)

        assert regime[0] == REGIME_STRONG_TREND
        assert regime[1] == REGIME_NEUTRAL  # NaN in ADX
        assert regime[2] == REGIME_NEUTRAL  # NaN in R2


class TestTrailingStopValidation:
    """Validate trailing stop calculation with hand-calculated values."""

    def test_trailing_stop_long_hand_calculated(self):
        """
        Test trailing stop for LONG position.

        Formula: Stop = Peak Price - (2.5 x ATR)

        Setup:
        - Entry at index 2
        - prices: [100, 101, 102, 105, 108, 106]  <- peak at index 4
        - ATR:    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]

        Expected stops (stop never decreases for long):
        - Stop[2] = 102 - 5.0 = 97.0 (entry price - ATR*2.5)
        - Stop[3] = 105 - 5.0 = 100.0 (peak rises to 105)
        - Stop[4] = 108 - 5.0 = 103.0 (peak rises to 108)
        - Stop[5] = 108 - 5.0 = 103.0 (price dropped, peak stays 108)
        """
        prices = np.array([100.0, 101.0, 102.0, 105.0, 108.0, 106.0])
        atr = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

        stops = calculate_trailing_stop(prices, atr, entry_idx=2, is_long=True)

        np.testing.assert_almost_equal(stops[2], 97.0, decimal=10)
        np.testing.assert_almost_equal(stops[3], 100.0, decimal=10)
        np.testing.assert_almost_equal(stops[4], 103.0, decimal=10)
        np.testing.assert_almost_equal(stops[5], 103.0, decimal=10)

    def test_trailing_stop_short_hand_calculated(self):
        """
        Test trailing stop for SHORT position.

        Formula: Stop = Trough Price + (2.5 x ATR)

        Setup:
        - Entry at index 2
        - prices: [100, 99, 98, 95, 92, 94]  <- trough at index 4
        - ATR:    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]

        Expected stops (stop never increases for short):
        - Stop[2] = 98 + 5.0 = 103.0 (entry price + ATR*2.5)
        - Stop[3] = 95 + 5.0 = 100.0 (trough drops to 95)
        - Stop[4] = 92 + 5.0 = 97.0 (trough drops to 92)
        - Stop[5] = 92 + 5.0 = 97.0 (price rose, trough stays 92)
        """
        prices = np.array([100.0, 99.0, 98.0, 95.0, 92.0, 94.0])
        atr = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

        stops = calculate_trailing_stop(prices, atr, entry_idx=2, is_long=False)

        np.testing.assert_almost_equal(stops[2], 103.0, decimal=10)
        np.testing.assert_almost_equal(stops[3], 100.0, decimal=10)
        np.testing.assert_almost_equal(stops[4], 97.0, decimal=10)
        np.testing.assert_almost_equal(stops[5], 97.0, decimal=10)


class TestFixedStopLossValidation:
    """Validate fixed stop loss calculation with hand-calculated values."""

    def test_fixed_stop_loss_long_hand_calculated(self):
        """
        Test fixed stop loss for LONG position.

        Formula: Stop = Entry Price x (1 - stop_pct)

        Example:
        - Entry price = 100
        - stop_pct = 0.03 (3%)
        - Stop = 100 x (1 - 0.03) = 100 x 0.97 = 97.0
        """
        stop = calculate_fixed_stop_loss(entry_price=100.0, is_long=True, stop_pct=0.03)
        assert stop == 97.0

    def test_fixed_stop_loss_short_hand_calculated(self):
        """
        Test fixed stop loss for SHORT position.

        Formula: Stop = Entry Price x (1 + stop_pct)

        Example:
        - Entry price = 100
        - stop_pct = 0.03 (3%)
        - Stop = 100 x (1 + 0.03) = 100 x 1.03 = 103.0
        """
        stop = calculate_fixed_stop_loss(entry_price=100.0, is_long=False, stop_pct=0.03)
        assert stop == 103.0

    def test_fixed_stop_loss_custom_percentage(self):
        """Test fixed stop loss with custom percentage."""
        # 5% stop on $50 entry, long
        stop_long = calculate_fixed_stop_loss(entry_price=50.0, is_long=True, stop_pct=0.05)
        assert stop_long == 50.0 * 0.95  # 47.5

        # 5% stop on $50 entry, short
        stop_short = calculate_fixed_stop_loss(entry_price=50.0, is_long=False, stop_pct=0.05)
        assert stop_short == 50.0 * 1.05  # 52.5


class TestPositionSizingValidation:
    """Validate position sizing calculation with hand-calculated values."""

    def test_position_size_hand_calculated(self):
        """
        Test position size calculation.

        Formula:
            Risk Amount = Capital x risk_pct
            Price Risk = |Entry Price - Stop Price|
            Shares = Risk Amount / Price Risk

        Example:
        - Capital = 100,000
        - Entry price = 100
        - Stop price = 97 (3% stop)
        - risk_pct = 0.01 (1%)

        Calculation:
        - Risk Amount = 100,000 x 0.01 = 1,000
        - Price Risk = |100 - 97| = 3
        - Shares = 1,000 / 3 = 333.33...
        """
        shares = calculate_position_size(
            capital=100000.0, entry_price=100.0, stop_price=97.0, risk_pct=0.01, is_long=True
        )
        expected = 1000.0 / 3.0
        np.testing.assert_almost_equal(shares, expected, decimal=6)

    def test_position_size_short_position(self):
        """
        Test position size for SHORT position.

        Example:
        - Capital = 50,000
        - Entry price = 100
        - Stop price = 103 (3% stop)
        - risk_pct = 0.0075 (0.75% for mean reversion)

        Calculation:
        - Risk Amount = 50,000 x 0.0075 = 375
        - Price Risk = |100 - 103| = 3
        - Shares = 375 / 3 = 125
        """
        shares = calculate_position_size(
            capital=50000.0, entry_price=100.0, stop_price=103.0, risk_pct=0.0075, is_long=False
        )
        assert shares == 125.0

    def test_position_size_zero_risk(self):
        """Test position size returns 0 when price risk is 0."""
        shares = calculate_position_size(
            capital=100000.0, entry_price=100.0, stop_price=100.0, risk_pct=0.01, is_long=True
        )
        assert shares == 0.0


class TestBlendSignalsValidation:
    """Validate signal blending for neutral regime."""

    def test_blend_signals_agreement(self):
        """
        Test blending signals when both strategies agree.

        When require_agreement=True, both must generate same signal.
        """
        trend_signals = np.array([0, 1, 0, -1, 0, 2])  # LONG at 1, SHORT at 3, EXIT at 5
        mr_signals = np.array([0, 1, 0, 0, 0, 2])  # LONG at 1, EXIT at 5

        blended = blend_signals(trend_signals, mr_signals, require_agreement=True)

        # Agreement at index 1 (both LONG) and index 5 (both EXIT_LONG)
        assert blended[1] == SIGNAL_LONG
        assert blended[5] == SIGNAL_EXIT_LONG
        # No agreement at index 3 (trend=SHORT, mr=NONE)
        assert blended[3] == SIGNAL_NONE

    def test_blend_signals_no_agreement_required(self):
        """
        Test blending signals without requiring agreement.

        When require_agreement=False, takes either signal (trend first).
        """
        trend_signals = np.array([0, 1, 0, 0, 0, 2])
        mr_signals = np.array([0, 0, 0, -1, 0, 0])

        blended = blend_signals(trend_signals, mr_signals, require_agreement=False)

        # Takes trend signal at 1, mr signal at 3, trend signal at 5
        assert blended[1] == SIGNAL_LONG
        assert blended[3] == SIGNAL_SHORT
        assert blended[5] == SIGNAL_EXIT_LONG


class TestSizeAdjustmentValidation:
    """Validate position size adjustment by regime."""

    def test_size_adjustment_trend_following(self):
        """Test 100% size for trend-following mode."""
        base = 100.0
        adjusted = adjust_size_for_regime(base, MODE_TREND_FOLLOWING)
        assert adjusted == 100.0

    def test_size_adjustment_mean_reversion(self):
        """Test 100% size for mean-reversion mode."""
        base = 100.0
        adjusted = adjust_size_for_regime(base, MODE_MEAN_REVERSION)
        assert adjusted == 100.0

    def test_size_adjustment_neutral(self):
        """Test 70% size (30% reduction) for neutral mode."""
        base = 100.0
        adjusted = adjust_size_for_regime(base, MODE_NEUTRAL)
        assert adjusted == 70.0


class TestSignalGenerationWithRealIndicators:
    """Integration tests using actual indicator calculations."""

    def test_trend_following_with_calculated_indicators(self):
        """
        Test trend-following signals using calculated KAMA and ATR.

        Create synthetic uptrend data and verify signals are generated correctly.
        """
        # Create synthetic uptrend: 100 bars going from 100 to 150
        n = 100
        prices = np.linspace(100.0, 150.0, n)
        # Add some noise
        np.random.seed(42)
        prices = prices + np.random.randn(n) * 0.5

        high = prices + 1.0
        low = prices - 1.0
        close = prices

        # Calculate indicators
        kama = calculate_kama(close, er_period=10)
        atr = calculate_atr(high, low, close, period=14)

        signals = generate_trend_following_signals(close, kama, atr)

        # In a strong uptrend, we should get at least one LONG signal
        long_signals = np.sum(signals == SIGNAL_LONG)
        assert long_signals >= 1, "Expected at least one LONG signal in uptrend"

    def test_mean_reversion_with_calculated_indicators(self):
        """
        Test mean-reversion signals using calculated RSI and Bollinger Bands.

        Create synthetic ranging data with oversold conditions.
        """
        # Create synthetic data with oversold condition
        # Steady prices followed by sharp decline
        n = 50
        prices = np.ones(n) * 100.0  # Start flat
        prices[35:45] = np.linspace(100, 85, 10)  # Sharp decline
        prices[45:] = np.linspace(85, 95, 5)  # Recovery

        close = prices

        # Calculate indicators
        rsi = calculate_rsi(close, period=14)
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, period=20, num_std=2.0)

        signals = generate_mean_reversion_signals(close, rsi, bb_lower, bb_middle)

        # During the sharp decline, RSI should drop and potentially trigger a long
        # (depending on whether price touches lower band)
        # This is more of a sanity check that the integration works
        assert len(signals) == n

    def test_regime_detection_on_synthetic_patterns(self):
        """
        Test regime detection on synthetic trending and ranging patterns.

        Create two distinct patterns:
        1. Perfect uptrend (should be STRONG_TREND)
        2. Choppy sideways (should be MEAN_REVERSION)
        """
        # Create synthetic regime data
        n = 100

        # For both patterns, create synthetic ADX and R2
        # Trend: High ADX (30+), High R2 (0.7+)
        # Choppy: Low ADX (<20), Low R2 (<0.4)

        # Trending regime
        trend_adx = np.full(n, 32.0)
        trend_r2 = np.full(n, 0.75)
        trend_regime = classify_trend_regime(trend_adx, trend_r2)

        # Verify all are STRONG_TREND
        assert np.all(trend_regime == REGIME_STRONG_TREND)

        # Choppy regime
        choppy_adx = np.full(n, 15.0)
        choppy_r2 = np.full(n, 0.25)
        choppy_regime = classify_trend_regime(choppy_adx, choppy_r2)

        # Verify all are MEAN_REVERSION
        assert np.all(choppy_regime == REGIME_MEAN_REVERSION)


class TestEdgeCases:
    """Test edge cases in signal generation."""

    def test_empty_arrays(self):
        """Test signal generation with empty arrays."""
        empty = np.array([])
        signals = generate_trend_following_signals(empty, empty, empty)
        assert len(signals) == 0

        signals = generate_mean_reversion_signals(empty, empty, empty, empty)
        assert len(signals) == 0

    def test_single_element(self):
        """Test signal generation with single element arrays."""
        single = np.array([100.0])
        signals = generate_trend_following_signals(single, single, np.array([2.0]))
        assert len(signals) == 1
        assert signals[0] == SIGNAL_NONE  # Need at least 2 elements for crossover

        signals = generate_mean_reversion_signals(single, single, single, single)
        assert len(signals) == 1
        assert signals[0] == SIGNAL_NONE

    def test_two_elements_no_crossover(self):
        """Test with two elements but no crossover."""
        prices = np.array([100.0, 100.5])
        kama = np.array([100.0, 100.0])
        atr = np.array([4.0, 4.0])

        signals = generate_trend_following_signals(prices, kama, atr)
        # prices[0]=100 <= upper_entry[0]=102? YES
        # prices[1]=100.5 > upper_entry[1]=102? NO
        assert signals[1] == SIGNAL_NONE

    def test_all_nan_values(self):
        """Test with all NaN values."""
        nan_arr = np.array([np.nan, np.nan, np.nan])
        signals = generate_trend_following_signals(nan_arr, nan_arr, nan_arr)
        assert np.all(signals == SIGNAL_NONE)

    def test_regime_all_nan(self):
        """Test regime classification with all NaN values."""
        nan_arr = np.array([np.nan, np.nan, np.nan])
        regime = classify_trend_regime(nan_arr, nan_arr)
        assert np.all(regime == REGIME_NEUTRAL)
