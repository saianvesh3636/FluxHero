"""
Adaptive EMA (KAMA-Based) Module

Implements Kaufman's Adaptive Moving Average (KAMA) which adjusts its smoothing
based on market efficiency. KAMA responds faster in trending markets and slower
in choppy/ranging markets.

Key Components:
- Efficiency Ratio (ER): Measures trend strength (0 = noise, 1 = perfect trend)
- Adaptive Smoothing Constant (ASC): Adjusts EMA alpha based on ER
- KAMA: The final adaptive moving average

References:
- FLUXHERO_REQUIREMENTS.md: Feature 2
- Perry Kaufman's "Trading Systems and Methods"
"""

import numpy as np
from numba import njit


@njit(cache=True)
def calculate_efficiency_ratio(
    prices: np.ndarray,
    period: int = 10
) -> np.ndarray:
    """
    Calculate Efficiency Ratio (ER) - measures trend strength.

    Formula:
        ER = |Change| / Sum(|Price[i] - Price[i-1]|)

    Where:
        Change = Price[today] - Price[period ago]
        Denominator = Sum of absolute price changes over period

    ER Interpretation:
        ER = 1.0 → Perfect trend (straight line, no noise)
        ER = 0.0 → Pure noise (random walk, no direction)
        ER > 0.6 → Strong trending market
        ER < 0.3 → Choppy/ranging market

    Parameters
    ----------
    prices : np.ndarray (float64)
        Array of prices (close prices typically)
    period : int
        Lookback period for ER calculation (default: 10)

    Returns
    -------
    np.ndarray (float64)
        Array of efficiency ratios (same length as prices)
        First 'period' values are NaN

    Examples
    --------
    >>> prices = np.array([100., 101., 102., 103., 104.])
    >>> er = calculate_efficiency_ratio(prices, period=4)
    >>> # Perfect uptrend should have ER close to 1.0
    """
    n = len(prices)
    er = np.full(n, np.nan, dtype=np.float64)

    if n < period + 1:
        return er

    for i in range(period, n):
        # Calculate net change over period
        change = abs(prices[i] - prices[i - period])

        # Calculate sum of absolute price changes (volatility)
        volatility = 0.0
        for j in range(i - period + 1, i + 1):
            volatility += abs(prices[j] - prices[j - 1])

        # Avoid division by zero
        if volatility > 0.0:
            er[i] = change / volatility
        else:
            # If no volatility, price is constant - treat as perfect trend
            er[i] = 1.0

    return er


@njit(cache=True)
def calculate_adaptive_smoothing_constant(
    efficiency_ratio: np.ndarray,
    fast_period: int = 2,
    slow_period: int = 30
) -> np.ndarray:
    """
    Calculate Adaptive Smoothing Constant (ASC) based on Efficiency Ratio.

    Formula:
        SC_fast = 2 / (fast_period + 1)
        SC_slow = 2 / (slow_period + 1)
        ASC = [ER × (SC_fast - SC_slow) + SC_slow]²

    The squaring amplifies the effect and ensures smooth transitions.

    Parameters
    ----------
    efficiency_ratio : np.ndarray (float64)
        Array of efficiency ratios (from calculate_efficiency_ratio)
    fast_period : int
        Fast EMA period (default: 2, gives SC = 0.6667)
    slow_period : int
        Slow EMA period (default: 30, gives SC = 0.0645)

    Returns
    -------
    np.ndarray (float64)
        Array of adaptive smoothing constants
        Values are bounded between SC_slow² and SC_fast²

    Examples
    --------
    >>> er = np.array([0.0, 0.5, 1.0])
    >>> asc = calculate_adaptive_smoothing_constant(er)
    >>> # asc[0] should be close to SC_slow²
    >>> # asc[2] should be close to SC_fast²
    """
    # Calculate smoothing constants
    sc_fast = 2.0 / (fast_period + 1)
    sc_slow = 2.0 / (slow_period + 1)

    n = len(efficiency_ratio)
    asc = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        if not np.isnan(efficiency_ratio[i]):
            # Linear interpolation between slow and fast SC based on ER
            sc = efficiency_ratio[i] * (sc_fast - sc_slow) + sc_slow
            # Square to amplify effect (Kaufman's formula)
            asc[i] = sc * sc

    return asc


@njit(cache=True)
def calculate_kama(
    prices: np.ndarray,
    er_period: int = 10,
    fast_period: int = 2,
    slow_period: int = 30
) -> np.ndarray:
    """
    Calculate Kaufman's Adaptive Moving Average (KAMA).

    KAMA combines Efficiency Ratio and Adaptive Smoothing to create a
    moving average that:
    - Follows price closely in trending markets (high ER)
    - Smooths out noise in choppy markets (low ER)

    Formula:
        KAMA[today] = KAMA[yesterday] + ASC × (Price[today] - KAMA[yesterday])

    Where ASC is calculated from ER using calculate_adaptive_smoothing_constant.

    Initialization:
        KAMA[first valid bar] = Price[first valid bar]

    Parameters
    ----------
    prices : np.ndarray (float64)
        Array of prices (close prices typically)
    er_period : int
        Lookback period for Efficiency Ratio (default: 10)
    fast_period : int
        Fast EMA period for ASC calculation (default: 2)
    slow_period : int
        Slow EMA period for ASC calculation (default: 30)

    Returns
    -------
    np.ndarray (float64)
        Array of KAMA values (same length as prices)
        First 'er_period' values are NaN

    Examples
    --------
    >>> prices = np.array([100., 102., 104., 103., 105., 107.])
    >>> kama = calculate_kama(prices, er_period=3)
    >>> # KAMA should adapt to the trending behavior
    """
    n = len(prices)
    kama = np.full(n, np.nan, dtype=np.float64)

    if n < er_period + 1:
        return kama

    # Calculate ER and ASC
    er = calculate_efficiency_ratio(prices, er_period)
    asc = calculate_adaptive_smoothing_constant(er, fast_period, slow_period)

    # Initialize KAMA with first valid price
    kama[er_period] = prices[er_period]

    # Calculate KAMA iteratively
    for i in range(er_period + 1, n):
        if not np.isnan(asc[i]):
            kama[i] = kama[i - 1] + asc[i] * (prices[i] - kama[i - 1])

    return kama


@njit(cache=True)
def calculate_kama_with_regime_adjustment(
    prices: np.ndarray,
    er_period: int = 10,
    fast_period: int = 2,
    slow_period: int = 30,
    trend_threshold: float = 0.6,
    choppy_threshold: float = 0.3
) -> tuple:
    """
    Calculate KAMA with regime-aware adjustments.

    Returns both KAMA values and regime classifications.

    Regime Classification:
        - TRENDING (2): ER > trend_threshold (0.6)
        - CHOPPY (0): ER < choppy_threshold (0.3)
        - NEUTRAL (1): Between thresholds

    In trending markets, bias toward faster SC.
    In choppy markets, bias toward slower SC.

    Parameters
    ----------
    prices : np.ndarray (float64)
        Array of prices (close prices)
    er_period : int
        Lookback period for Efficiency Ratio (default: 10)
    fast_period : int
        Fast EMA period for ASC calculation (default: 2)
    slow_period : int
        Slow EMA period for ASC calculation (default: 30)
    trend_threshold : float
        ER threshold for trending regime (default: 0.6)
    choppy_threshold : float
        ER threshold for choppy regime (default: 0.3)

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (kama, efficiency_ratio, regime)
        - kama: KAMA values
        - efficiency_ratio: ER values for analysis
        - regime: Regime classification (0=choppy, 1=neutral, 2=trending)

    Examples
    --------
    >>> prices = np.array([100., 101., 102., 103., 104., 105.])
    >>> kama, er, regime = calculate_kama_with_regime_adjustment(prices)
    >>> # Strong uptrend should show regime = 2 (TRENDING)
    """
    n = len(prices)

    # Calculate base KAMA and ER
    er = calculate_efficiency_ratio(prices, er_period)
    kama = calculate_kama(prices, er_period, fast_period, slow_period)

    # Classify regime
    regime = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        if not np.isnan(er[i]):
            if er[i] > trend_threshold:
                regime[i] = 2.0  # TRENDING
            elif er[i] < choppy_threshold:
                regime[i] = 0.0  # CHOPPY
            else:
                regime[i] = 1.0  # NEUTRAL

    return kama, er, regime


@njit(cache=True)
def validate_kama_bounds(
    efficiency_ratio: np.ndarray,
    adaptive_smoothing_constant: np.ndarray,
    fast_period: int = 2,
    slow_period: int = 30
) -> tuple:
    """
    Validate that ER and ASC are within expected mathematical bounds.

    Validation Rules:
        1. ER must be between 0 and 1 (inclusive)
        2. ASC must be between SC_slow² and SC_fast² (inclusive)

    Parameters
    ----------
    efficiency_ratio : np.ndarray (float64)
        Array of efficiency ratios
    adaptive_smoothing_constant : np.ndarray (float64)
        Array of adaptive smoothing constants
    fast_period : int
        Fast EMA period (default: 2)
    slow_period : int
        Slow EMA period (default: 30)

    Returns
    -------
    tuple[bool, bool]
        (er_valid, asc_valid)
        - er_valid: True if all non-NaN ER values in [0, 1]
        - asc_valid: True if all non-NaN ASC values in [SC_slow², SC_fast²]

    Examples
    --------
    >>> er = np.array([0.0, 0.5, 1.0])
    >>> asc = calculate_adaptive_smoothing_constant(er)
    >>> er_valid, asc_valid = validate_kama_bounds(er, asc)
    >>> assert er_valid and asc_valid
    """
    # Calculate expected bounds
    sc_fast = 2.0 / (fast_period + 1)
    sc_slow = 2.0 / (slow_period + 1)
    asc_min = sc_slow * sc_slow
    asc_max = sc_fast * sc_fast

    # Validate ER
    er_valid = True
    for i in range(len(efficiency_ratio)):
        if not np.isnan(efficiency_ratio[i]):
            if efficiency_ratio[i] < 0.0 or efficiency_ratio[i] > 1.0:
                er_valid = False
                break

    # Validate ASC
    asc_valid = True
    for i in range(len(adaptive_smoothing_constant)):
        if not np.isnan(adaptive_smoothing_constant[i]):
            # Allow small numerical tolerance
            if (adaptive_smoothing_constant[i] < asc_min - 1e-10 or
                adaptive_smoothing_constant[i] > asc_max + 1e-10):
                asc_valid = False
                break

    return er_valid, asc_valid
