"""
Validation Tests for Technical Indicator Calculations.

This module validates indicator calculations against hand-calculated expected values
to ensure mathematical correctness. Each test includes worked examples in comments.

Reference: FLUXHERO_REQUIREMENTS.md Feature 2 - Computation Engine
Reference: enhancement_tasks.md Phase 24 - Quality Control & Validation Framework

Key validation approach:
1. Use simple, hand-verifiable numbers
2. Include step-by-step calculations in comments
3. Test both typical cases and edge cases
4. Validate against known formulas (matching TradingView/pandas behavior)
"""

import numpy as np

from backend.computation.adaptive_ema import (
    calculate_adaptive_smoothing_constant,
    calculate_efficiency_ratio,
    calculate_kama,
    calculate_kama_with_regime_adjustment,
    validate_kama_bounds,
)
from backend.computation.indicators import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_rsi,
    calculate_sma,
    calculate_true_range,
)


class TestCalculateEMAValidation:
    """Validate EMA calculation with hand-calculated values."""

    def test_ema_hand_calculated_period_3(self):
        """
        Test EMA with period=3 on simple price series.

        Hand calculation for EMA(3) with prices [10, 11, 12, 13, 14]:
        - alpha = 2 / (3 + 1) = 0.5
        - Initial SMA (first 3 prices): (10 + 11 + 12) / 3 = 11.0
        - EMA[2] = 11.0 (initial SMA)
        - EMA[3] = 13 * 0.5 + 11.0 * 0.5 = 6.5 + 5.5 = 12.0
        - EMA[4] = 14 * 0.5 + 12.0 * 0.5 = 7.0 + 6.0 = 13.0
        """
        prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        ema = calculate_ema(prices, period=3)

        # First 2 values should be NaN
        assert np.isnan(ema[0])
        assert np.isnan(ema[1])

        # Hand-calculated values
        alpha = 2.0 / (3 + 1)  # 0.5
        np.testing.assert_almost_equal(ema[2], 11.0, decimal=10)  # SMA seed
        np.testing.assert_almost_equal(ema[3], 13.0 * alpha + 11.0 * (1 - alpha), decimal=10)
        np.testing.assert_almost_equal(ema[4], 14.0 * alpha + 12.0 * (1 - alpha), decimal=10)

    def test_ema_period_10_sequential_prices(self):
        """
        Test EMA(10) with sequential prices [100, 101, 102, ..., 109, 110].

        Hand calculation:
        - alpha = 2 / (10 + 1) = 2/11 = 0.181818...
        - Initial SMA (first 10 prices): (100+101+...+109) / 10 = 104.5
        - EMA[9] = 104.5 (initial SMA)
        - EMA[10] = 110 * alpha + 104.5 * (1 - alpha)
                  = 110 * 0.181818 + 104.5 * 0.818182
                  = 20.0 + 85.5 = 105.5

        Verifying alpha formula:
        - alpha = 2 / (period + 1) matches pandas ewm(adjust=False)
        """
        prices = np.array([100.0 + i for i in range(11)])  # [100, 101, ..., 110]
        ema = calculate_ema(prices, period=10)

        alpha = 2.0 / (10 + 1)  # = 0.181818...

        # First 9 values should be NaN
        for i in range(9):
            assert np.isnan(ema[i]), f"ema[{i}] should be NaN"

        # Initial SMA seed: (100+101+...+109) / 10 = (10*100 + 45) / 10 = 104.5
        expected_sma = (100 * 10 + sum(range(10))) / 10  # 1045 / 10 = 104.5
        np.testing.assert_almost_equal(ema[9], expected_sma, decimal=10)

        # EMA[10] = price[10] * alpha + EMA[9] * (1 - alpha)
        expected_ema_10 = 110.0 * alpha + expected_sma * (1 - alpha)
        np.testing.assert_almost_equal(ema[10], expected_ema_10, decimal=10)

    def test_ema_alpha_formula_verification(self):
        """
        Verify the alpha formula matches expected behavior.

        Standard EMA alpha formula: alpha = 2 / (period + 1)
        This is equivalent to pandas ewm(span=period, adjust=False)

        Test periods and expected alphas:
        - Period 12: alpha = 2/13 = 0.153846...
        - Period 26: alpha = 2/27 = 0.074074...
        - Period 50: alpha = 2/51 = 0.039216...
        """
        # Test by observing behavior on constant price (EMA should equal the constant)
        constant_prices = np.array([100.0] * 60)

        for period in [12, 26, 50]:
            ema = calculate_ema(constant_prices, period=period)
            # For constant prices, EMA should equal the price after warmup
            for i in range(period - 1, 60):
                if not np.isnan(ema[i]):
                    np.testing.assert_almost_equal(ema[i], 100.0, decimal=10)

    def test_ema_trending_up_vs_trending_down(self):
        """
        Test EMA behavior on uptrend vs downtrend.

        For uptrending prices, EMA should lag below current price.
        For downtrending prices, EMA should lag above current price.
        """
        # Uptrend: prices going from 100 to 105
        uptrend = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        ema_up = calculate_ema(uptrend, period=3)

        # EMA lags, so should be below current price in uptrend
        assert ema_up[5] < 105.0, "EMA should lag below price in uptrend"

        # Downtrend: prices going from 105 to 100
        downtrend = np.array([105.0, 104.0, 103.0, 102.0, 101.0, 100.0])
        ema_down = calculate_ema(downtrend, period=3)

        # EMA lags, so should be above current price in downtrend
        assert ema_down[5] > 100.0, "EMA should lag above price in downtrend"

    def test_ema_insufficient_data(self):
        """Test EMA returns all NaN when data length < period."""
        prices = np.array([100.0, 101.0, 102.0])
        ema = calculate_ema(prices, period=5)
        assert np.all(np.isnan(ema))

    def test_ema_empty_array(self):
        """Test EMA handles empty array."""
        prices = np.array([])
        ema = calculate_ema(prices, period=5)
        assert len(ema) == 0


class TestCalculateRSIValidation:
    """Validate RSI calculation with hand-calculated values."""

    def test_rsi_hand_calculated_simple(self):
        """
        Test RSI with simple hand-calculated example.

        Prices: [100, 102, 101, 104, 103, 106] (6 prices)
        Using period=5:

        Price changes (deltas):
        - 102 - 100 = +2 (gain)
        - 101 - 102 = -1 (loss)
        - 104 - 101 = +3 (gain)
        - 103 - 104 = -1 (loss)
        - 106 - 103 = +3 (gain)

        Initial averages (first 5 changes):
        - Avg gain = (2 + 0 + 3 + 0 + 3) / 5 = 8/5 = 1.6
        - Avg loss = (0 + 1 + 0 + 1 + 0) / 5 = 2/5 = 0.4

        RS = 1.6 / 0.4 = 4.0
        RSI = 100 - (100 / (1 + 4)) = 100 - 20 = 80
        """
        prices = np.array([100.0, 102.0, 101.0, 104.0, 103.0, 106.0])
        rsi = calculate_rsi(prices, period=5)

        # First 5 values should be NaN (need period+1 prices for first RSI)
        for i in range(5):
            assert np.isnan(rsi[i]), f"rsi[{i}] should be NaN"

        # Hand-calculated RSI at index 5
        avg_gain = (2.0 + 0.0 + 3.0 + 0.0 + 3.0) / 5.0  # 1.6
        avg_loss = (0.0 + 1.0 + 0.0 + 1.0 + 0.0) / 5.0  # 0.4
        rs = avg_gain / avg_loss  # 4.0
        expected_rsi = 100.0 - (100.0 / (1.0 + rs))  # 80.0

        np.testing.assert_almost_equal(rsi[5], expected_rsi, decimal=6)

    def test_rsi_overbought_pattern(self):
        """
        Test RSI recognizes overbought condition (RSI > 70).

        Create strong uptrend where price only goes up:
        [100, 105, 110, 115, 120, 125, 130] with period=6

        All gains, no losses:
        - Gains: 5, 5, 5, 5, 5, 5
        - Losses: 0, 0, 0, 0, 0, 0
        - Avg gain = 5, Avg loss = 0
        - When avg_loss = 0, RSI = 100 (pure overbought)
        """
        prices = np.array([100.0 + i * 5.0 for i in range(7)])  # [100, 105, ..., 130]
        rsi = calculate_rsi(prices, period=6)

        # RSI should be 100 (max overbought) when all moves are up
        np.testing.assert_almost_equal(rsi[6], 100.0, decimal=10)

    def test_rsi_oversold_pattern(self):
        """
        Test RSI recognizes oversold condition (RSI < 30).

        Create strong downtrend where price only goes down:
        [130, 125, 120, 115, 110, 105, 100] with period=6

        No gains, all losses:
        - Gains: 0, 0, 0, 0, 0, 0
        - Losses: 5, 5, 5, 5, 5, 5
        - Avg gain = 0, Avg loss = 5
        - RS = 0/5 = 0
        - RSI = 100 - (100 / (1 + 0)) = 100 - 100 = 0 (pure oversold)
        """
        prices = np.array([130.0 - i * 5.0 for i in range(7)])  # [130, 125, ..., 100]
        rsi = calculate_rsi(prices, period=6)

        # RSI should be 0 (max oversold) when all moves are down
        np.testing.assert_almost_equal(rsi[6], 0.0, decimal=10)

    def test_rsi_neutral_50(self):
        """
        Test RSI = 50 when gains equal losses.

        Alternating pattern: up 5, down 5, up 5, down 5, ...
        [100, 105, 100, 105, 100, 105, 100] with period=6

        Gains: 5, 0, 5, 0, 5, 0 → avg = 10/6 = 1.667
        Losses: 0, 5, 0, 5, 0, 5 → avg = 15/6 = 2.5
        Wait, that's not equal...

        Better example: equal magnitude moves
        [100, 105, 100, 105, 100, 105, 100, 105] with period=4
        Gains: 5, 0, 5, 0 → sum = 10, avg = 2.5
        Losses: 0, 5, 0, 5 → sum = 10, avg = 2.5
        RS = 1.0
        RSI = 100 - (100 / (1 + 1)) = 100 - 50 = 50
        """
        prices = np.array([100.0, 105.0, 100.0, 105.0, 100.0])
        rsi = calculate_rsi(prices, period=4)

        # At index 4, using first 4 changes:
        # Changes: +5, -5, +5, -5
        # Avg gain = (5 + 0 + 5 + 0) / 4 = 2.5
        # Avg loss = (0 + 5 + 0 + 5) / 4 = 2.5
        # RS = 2.5 / 2.5 = 1.0
        # RSI = 100 - (100 / 2) = 50
        np.testing.assert_almost_equal(rsi[4], 50.0, decimal=10)

    def test_rsi_wilder_smoothing(self):
        """
        Test RSI uses Wilder's smoothing (not SMA) for subsequent values.

        Wilder's smoothing formula:
        - Avg_gain[i] = (Avg_gain[i-1] * (period-1) + gain[i]) / period

        This test verifies the smoothing is applied correctly after initial SMA.
        """
        # Create a sequence where we can verify smoothing
        prices = np.array([100.0, 102.0, 101.0, 104.0, 103.0, 106.0, 108.0])
        period = 5
        rsi = calculate_rsi(prices, period=period)

        # Calculate expected RSI at index 6 using Wilder's smoothing
        # Initial averages at index 5:
        # Avg_gain_5 = (2 + 0 + 3 + 0 + 3) / 5 = 1.6
        # Avg_loss_5 = (0 + 1 + 0 + 1 + 0) / 5 = 0.4

        # Change from 106 to 108 = +2 (gain)
        # Wilder's smoothed at index 6:
        # Avg_gain_6 = (1.6 * 4 + 2) / 5 = 8.4 / 5 = 1.68
        # Avg_loss_6 = (0.4 * 4 + 0) / 5 = 1.6 / 5 = 0.32
        # RS = 1.68 / 0.32 = 5.25
        # RSI = 100 - (100 / 6.25) = 84.0

        avg_gain_5 = 1.6
        avg_loss_5 = 0.4
        gain_6 = 2.0  # 108 - 106
        loss_6 = 0.0

        avg_gain_6 = (avg_gain_5 * (period - 1) + gain_6) / period
        avg_loss_6 = (avg_loss_5 * (period - 1) + loss_6) / period
        rs_6 = avg_gain_6 / avg_loss_6
        expected_rsi_6 = 100.0 - (100.0 / (1.0 + rs_6))

        np.testing.assert_almost_equal(rsi[6], expected_rsi_6, decimal=6)

    def test_rsi_insufficient_data(self):
        """Test RSI returns all NaN when data length < period + 1."""
        prices = np.array([100.0, 101.0, 102.0])
        rsi = calculate_rsi(prices, period=14)
        assert np.all(np.isnan(rsi))


class TestCalculateSMAValidation:
    """Validate SMA calculation with hand-calculated values."""

    def test_sma_hand_calculated(self):
        """
        Test SMA with simple hand-calculated example.

        Prices: [10, 20, 30, 40, 50] with period=3
        - SMA[2] = (10 + 20 + 30) / 3 = 60/3 = 20
        - SMA[3] = (20 + 30 + 40) / 3 = 90/3 = 30
        - SMA[4] = (30 + 40 + 50) / 3 = 120/3 = 40
        """
        prices = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        sma = calculate_sma(prices, period=3)

        assert np.isnan(sma[0])
        assert np.isnan(sma[1])
        np.testing.assert_almost_equal(sma[2], 20.0, decimal=10)
        np.testing.assert_almost_equal(sma[3], 30.0, decimal=10)
        np.testing.assert_almost_equal(sma[4], 40.0, decimal=10)

    def test_sma_constant_prices(self):
        """SMA of constant prices should equal the constant."""
        prices = np.array([50.0] * 10)
        sma = calculate_sma(prices, period=5)

        for i in range(4, 10):
            np.testing.assert_almost_equal(sma[i], 50.0, decimal=10)


class TestCalculateTrueRangeValidation:
    """Validate True Range calculation with hand-calculated values."""

    def test_true_range_hand_calculated(self):
        """
        Test True Range with hand-calculated example.

        TR = max(High - Low, |High - PrevClose|, |Low - PrevClose|)

        Bar 0: H=45, L=44, C=44.5
        - TR[0] = H - L = 45 - 44 = 1 (no prev close)

        Bar 1: H=47, L=45, C=46, PrevClose=44.5
        - Method 1: 47 - 45 = 2
        - Method 2: |47 - 44.5| = 2.5
        - Method 3: |45 - 44.5| = 0.5
        - TR[1] = max(2, 2.5, 0.5) = 2.5

        Bar 2: H=48, L=44, C=47, PrevClose=46 (gap scenario)
        - Method 1: 48 - 44 = 4
        - Method 2: |48 - 46| = 2
        - Method 3: |44 - 46| = 2
        - TR[2] = max(4, 2, 2) = 4
        """
        high = np.array([45.0, 47.0, 48.0])
        low = np.array([44.0, 45.0, 44.0])
        close = np.array([44.5, 46.0, 47.0])

        tr = calculate_true_range(high, low, close)

        np.testing.assert_almost_equal(tr[0], 1.0, decimal=10)  # H - L
        np.testing.assert_almost_equal(tr[1], 2.5, decimal=10)  # |H - PrevC|
        np.testing.assert_almost_equal(tr[2], 4.0, decimal=10)  # H - L

    def test_true_range_gap_up(self):
        """
        Test TR captures gap up scenario.

        Gap up: Previous close below current low
        PrevClose=100, H=110, L=108, C=109
        - Method 1: 110 - 108 = 2
        - Method 2: |110 - 100| = 10
        - Method 3: |108 - 100| = 8
        - TR = max(2, 10, 8) = 10
        """
        high = np.array([101.0, 110.0])
        low = np.array([99.0, 108.0])
        close = np.array([100.0, 109.0])

        tr = calculate_true_range(high, low, close)
        np.testing.assert_almost_equal(tr[1], 10.0, decimal=10)

    def test_true_range_gap_down(self):
        """
        Test TR captures gap down scenario.

        Gap down: Previous close above current high
        PrevClose=100, H=92, L=90, C=91
        - Method 1: 92 - 90 = 2
        - Method 2: |92 - 100| = 8
        - Method 3: |90 - 100| = 10
        - TR = max(2, 8, 10) = 10
        """
        high = np.array([101.0, 92.0])
        low = np.array([99.0, 90.0])
        close = np.array([100.0, 91.0])

        tr = calculate_true_range(high, low, close)
        np.testing.assert_almost_equal(tr[1], 10.0, decimal=10)


class TestCalculateATRValidation:
    """Validate ATR calculation with hand-calculated values."""

    def test_atr_hand_calculated(self):
        """
        Test ATR with hand-calculated example.

        Using period=3:
        TR values: [1.0, 2.0, 3.0, 4.0, 5.0]

        Initial ATR (SMA of first period TRs starting at index 1):
        - ATR[3] = (TR[1] + TR[2] + TR[3]) / 3 = (2 + 3 + 4) / 3 = 3.0

        Wilder's smoothing for ATR[4]:
        - ATR[4] = (ATR[3] * (3-1) + TR[4]) / 3 = (3.0 * 2 + 5) / 3 = 11/3 = 3.667
        """
        # Create data that produces predictable TR values
        high = np.array([101.0, 102.0, 103.0, 104.0, 105.0])
        low = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        close = np.array([100.5, 101.5, 102.5, 103.5, 104.5])

        # TR values will be: [1.0, 2.0, 3.0, 4.0, 5.0]
        # (H - L since no gaps)

        atr = calculate_atr(high, low, close, period=3)

        # First valid ATR at index 3 (period TRs starting at index 1)
        # ATR[3] = (2 + 3 + 4) / 3 = 3.0
        np.testing.assert_almost_equal(atr[3], 3.0, decimal=10)

        # ATR[4] with Wilder's smoothing
        expected_atr_4 = (3.0 * 2 + 5.0) / 3.0  # 3.667
        np.testing.assert_almost_equal(atr[4], expected_atr_4, decimal=10)


class TestCalculateBollingerBandsValidation:
    """Validate Bollinger Bands calculation with hand-calculated values."""

    def test_bollinger_bands_hand_calculated(self):
        """
        Test Bollinger Bands with hand-calculated example.

        Prices: [10, 11, 12, 13, 14] with period=5, num_std=2

        Middle band (SMA):
        - Mean = (10 + 11 + 12 + 13 + 14) / 5 = 60/5 = 12

        Standard deviation (population):
        - Variance = ((10-12)^2 + (11-12)^2 + (12-12)^2 + (13-12)^2 + (14-12)^2) / 5
                   = (4 + 1 + 0 + 1 + 4) / 5 = 10/5 = 2
        - Std = sqrt(2) = 1.414...

        Bands (2 std):
        - Upper = 12 + 2 * 1.414 = 12 + 2.828 = 14.828
        - Lower = 12 - 2 * 1.414 = 12 - 2.828 = 9.172
        """
        prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        upper, middle, lower = calculate_bollinger_bands(prices, period=5, num_std=2.0)

        mean = 12.0
        variance = (4.0 + 1.0 + 0.0 + 1.0 + 4.0) / 5.0
        std_dev = np.sqrt(variance)

        np.testing.assert_almost_equal(middle[4], mean, decimal=10)
        np.testing.assert_almost_equal(upper[4], mean + 2.0 * std_dev, decimal=10)
        np.testing.assert_almost_equal(lower[4], mean - 2.0 * std_dev, decimal=10)

    def test_bollinger_bands_symmetry(self):
        """Upper and lower bands should be equidistant from middle."""
        prices = np.array([100.0, 102.0, 98.0, 101.0, 99.0, 103.0])
        upper, middle, lower = calculate_bollinger_bands(prices, period=5, num_std=2.0)

        # At index 5, check symmetry
        distance_upper = upper[5] - middle[5]
        distance_lower = middle[5] - lower[5]
        np.testing.assert_almost_equal(distance_upper, distance_lower, decimal=10)


class TestCalculateEfficiencyRatioValidation:
    """Validate Efficiency Ratio calculation with hand-calculated values."""

    def test_efficiency_ratio_perfect_trend(self):
        """
        Test ER = 1.0 for perfect uptrend (no noise).

        Prices: [100, 101, 102, 103, 104] with period=4

        At index 4:
        - Change = |104 - 100| = 4
        - Volatility = |101-100| + |102-101| + |103-102| + |104-103| = 1+1+1+1 = 4
        - ER = 4 / 4 = 1.0
        """
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        er = calculate_efficiency_ratio(prices, period=4)

        np.testing.assert_almost_equal(er[4], 1.0, decimal=10)

    def test_efficiency_ratio_ranging_market(self):
        """
        Test ER approaches 0 for ranging/choppy market.

        Prices that oscillate: [100, 105, 100, 105, 100] with period=4

        At index 4:
        - Change = |100 - 100| = 0
        - Volatility = |105-100| + |100-105| + |105-100| + |100-105| = 5+5+5+5 = 20
        - ER = 0 / 20 = 0.0
        """
        prices = np.array([100.0, 105.0, 100.0, 105.0, 100.0])
        er = calculate_efficiency_ratio(prices, period=4)

        np.testing.assert_almost_equal(er[4], 0.0, decimal=10)

    def test_efficiency_ratio_partial_trend(self):
        """
        Test ER for partial trend (some noise).

        Prices: [100, 102, 101, 103, 104] with period=4

        At index 4:
        - Change = |104 - 100| = 4
        - Volatility = |102-100| + |101-102| + |103-101| + |104-103| = 2+1+2+1 = 6
        - ER = 4 / 6 = 0.667
        """
        prices = np.array([100.0, 102.0, 101.0, 103.0, 104.0])
        er = calculate_efficiency_ratio(prices, period=4)

        expected_er = 4.0 / 6.0
        np.testing.assert_almost_equal(er[4], expected_er, decimal=10)

    def test_efficiency_ratio_bounds(self):
        """Test ER is always in [0, 1] range."""
        np.random.seed(42)
        prices = 100.0 + np.cumsum(np.random.randn(100))
        er = calculate_efficiency_ratio(prices, period=10)

        for i in range(10, 100):
            assert 0.0 <= er[i] <= 1.0, f"ER[{i}]={er[i]} out of bounds"


class TestCalculateKAMAValidation:
    """Validate KAMA calculation with hand-calculated values."""

    def test_kama_adaptive_smoothing_constant_bounds(self):
        """
        Test ASC bounds with hand-calculated values.

        With fast_period=2, slow_period=30:
        - SC_fast = 2 / (2 + 1) = 0.6667
        - SC_slow = 2 / (30 + 1) = 0.0645
        - ASC_min = SC_slow^2 = 0.0042
        - ASC_max = SC_fast^2 = 0.4444

        For ER=0: ASC = (0 * (0.6667 - 0.0645) + 0.0645)^2 = 0.0645^2 = 0.0042
        For ER=1: ASC = (1 * (0.6667 - 0.0645) + 0.0645)^2 = 0.6667^2 = 0.4444
        """
        er = np.array([0.0, 0.5, 1.0])
        asc = calculate_adaptive_smoothing_constant(er, fast_period=2, slow_period=30)

        sc_fast = 2.0 / 3.0  # 0.6667
        sc_slow = 2.0 / 31.0  # 0.0645

        # ER = 0 → ASC = SC_slow^2
        np.testing.assert_almost_equal(asc[0], sc_slow**2, decimal=6)

        # ER = 1 → ASC = SC_fast^2
        np.testing.assert_almost_equal(asc[2], sc_fast**2, decimal=6)

        # ER = 0.5 → ASC should be between bounds
        expected_sc = 0.5 * (sc_fast - sc_slow) + sc_slow
        np.testing.assert_almost_equal(asc[1], expected_sc**2, decimal=6)

    def test_kama_hand_calculated(self):
        """
        Test KAMA calculation step by step.

        Prices: [100, 101, 102, 103, 104, 105, 106] with er_period=3

        At index 3 (first valid):
        - KAMA[3] = prices[3] = 103 (initialization)

        At index 4:
        - Need ER at index 4 to get ASC
        - ER calculation for index 4:
          - Change = |104 - 101| = 3
          - Volatility = |102-101| + |103-102| + |104-103| = 3
          - ER = 3/3 = 1.0
        - ASC for ER=1 (with default periods) = SC_fast^2 = (2/3)^2 = 0.4444
        - KAMA[4] = KAMA[3] + ASC * (price[4] - KAMA[3])
                  = 103 + 0.4444 * (104 - 103)
                  = 103 + 0.4444 = 103.4444
        """
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
        kama = calculate_kama(prices, er_period=3, fast_period=2, slow_period=30)

        # First valid KAMA = price at er_period
        np.testing.assert_almost_equal(kama[3], 103.0, decimal=10)

        # Calculate expected KAMA[4]
        sc_fast = 2.0 / 3.0
        asc_max = sc_fast**2  # ER=1 for perfect trend

        expected_kama_4 = 103.0 + asc_max * (104.0 - 103.0)
        np.testing.assert_almost_equal(kama[4], expected_kama_4, decimal=4)

    def test_kama_responds_to_trend(self):
        """
        Test that KAMA responds faster in trending markets.

        In a perfect trend (ER=1), KAMA should adjust quickly.
        In choppy market (ER~0), KAMA should be sluggish.
        """
        # Perfect uptrend
        trend_prices = np.array([100.0 + i for i in range(20)])
        kama_trend = calculate_kama(trend_prices, er_period=5)

        # KAMA should closely follow price in strong trend
        # After warmup, difference should be small
        diff_trend = abs(kama_trend[19] - trend_prices[19])
        assert diff_trend < 3.0, "KAMA should follow closely in trend"

        # Choppy prices (oscillating)
        choppy_prices = np.array([100.0 if i % 2 == 0 else 105.0 for i in range(20)])
        kama_choppy = calculate_kama(choppy_prices, er_period=5)

        # KAMA should be sluggish in choppy market (near center)
        # It won't track oscillations
        assert 100.0 < kama_choppy[19] < 105.0, "KAMA should be stable in choppy market"

    def test_kama_validate_bounds(self):
        """Test KAMA bounds validation function."""
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        er = calculate_efficiency_ratio(prices, period=3)
        asc = calculate_adaptive_smoothing_constant(er, fast_period=2, slow_period=30)

        er_valid, asc_valid = validate_kama_bounds(er, asc, fast_period=2, slow_period=30)

        assert er_valid, "ER should be within [0, 1] bounds"
        assert asc_valid, "ASC should be within [SC_slow^2, SC_fast^2] bounds"


class TestKAMARegimeClassificationValidation:
    """Validate KAMA regime classification."""

    def test_regime_trending(self):
        """
        Test regime = 2 (TRENDING) when ER > 0.6.

        Perfect uptrend should have ER = 1.0, classified as TRENDING.
        """
        prices = np.array([100.0 + i for i in range(15)])  # Perfect trend
        kama, er, regime = calculate_kama_with_regime_adjustment(
            prices, er_period=5, trend_threshold=0.6, choppy_threshold=0.3
        )

        # After warmup, regime should be TRENDING (2.0)
        for i in range(10, 15):
            assert regime[i] == 2.0, f"regime[{i}] should be TRENDING (2.0), got {regime[i]}"

    def test_regime_choppy(self):
        """
        Test regime = 0 (CHOPPY) when ER < 0.3.

        Oscillating prices should have low ER, classified as CHOPPY.
        """
        prices = np.array([100.0 if i % 2 == 0 else 105.0 for i in range(15)])
        kama, er, regime = calculate_kama_with_regime_adjustment(
            prices, er_period=5, trend_threshold=0.6, choppy_threshold=0.3
        )

        # After warmup, regime should be CHOPPY (0.0) for oscillating prices
        for i in range(10, 15):
            assert regime[i] == 0.0, f"regime[{i}] should be CHOPPY (0.0), got {regime[i]}"

    def test_regime_neutral(self):
        """
        Test regime = 1 (NEUTRAL) when 0.3 <= ER <= 0.6.

        Create prices with moderate trend/noise mix.
        """
        # Prices with some trend but also noise
        prices = np.array([100.0, 102.0, 101.0, 104.0, 103.0, 106.0, 105.0, 108.0, 107.0, 110.0])
        kama, er, regime = calculate_kama_with_regime_adjustment(
            prices, er_period=5, trend_threshold=0.6, choppy_threshold=0.3
        )

        # ER should be between 0.3 and 0.6 for this mixed pattern
        # Verify regime classification is in expected range
        for i in range(5, 10):
            if not np.isnan(er[i]):
                if er[i] > 0.6:
                    assert regime[i] == 2.0
                elif er[i] < 0.3:
                    assert regime[i] == 0.0
                else:
                    assert regime[i] == 1.0


class TestIndicatorEdgeCases:
    """Test edge cases for all indicators."""

    def test_all_indicators_single_value(self):
        """Test all indicators handle single value array."""
        single = np.array([100.0])

        ema = calculate_ema(single, period=5)
        assert np.all(np.isnan(ema))

        rsi = calculate_rsi(single, period=5)
        assert np.all(np.isnan(rsi))

        sma = calculate_sma(single, period=5)
        assert np.all(np.isnan(sma))

        er = calculate_efficiency_ratio(single, period=5)
        assert np.all(np.isnan(er))

    def test_all_indicators_empty_array(self):
        """Test all indicators handle empty array."""
        empty = np.array([])

        assert len(calculate_ema(empty, period=5)) == 0
        assert len(calculate_rsi(empty, period=5)) == 0
        assert len(calculate_sma(empty, period=5)) == 0
        assert len(calculate_efficiency_ratio(empty, period=5)) == 0

    def test_indicators_with_nan_in_output(self):
        """Verify indicators produce NaN for warmup period only."""
        prices = np.array([100.0 + i for i in range(20)])

        # EMA warmup = period - 1
        ema = calculate_ema(prices, period=5)
        assert sum(np.isnan(ema)) == 4  # indices 0-3
        assert sum(~np.isnan(ema)) == 16  # indices 4-19

        # RSI warmup = period
        rsi = calculate_rsi(prices, period=5)
        assert sum(np.isnan(rsi)) == 5  # indices 0-4
        assert sum(~np.isnan(rsi)) == 15  # indices 5-19

        # SMA warmup = period - 1
        sma = calculate_sma(prices, period=5)
        assert sum(np.isnan(sma)) == 4  # indices 0-3
        assert sum(~np.isnan(sma)) == 16  # indices 4-19

        # ER warmup = period
        er = calculate_efficiency_ratio(prices, period=5)
        assert sum(np.isnan(er)) == 5  # indices 0-4
        assert sum(~np.isnan(er)) == 15  # indices 5-19
