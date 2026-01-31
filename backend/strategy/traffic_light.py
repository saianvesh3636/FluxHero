"""
Traffic Light Signal Aggregation for Regime Anticipation.

This module aggregates multiple anticipation indicators into a simple
traffic light signal (GREEN/YELLOW/RED) for regime change warnings.

IMPORTANT: This is DIAGNOSTIC ONLY - it does not modify position sizing.
The traffic light is used for:
- Logging alongside trades for analysis
- Tracking in backtest results for correlation studies
- Validating if anticipation signals precede regime changes

Future enhancement: Can be made active (affect sizing) once validated.
"""

import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from numba import njit


class TrafficLight(IntEnum):
    """Traffic light states for regime anticipation."""
    GREEN = 0   # All clear, trade normally
    YELLOW = 1  # Caution, potential regime change coming
    RED = 2     # High probability of regime change imminent


@dataclass
class TrafficLightResult:
    """Result of traffic light calculation for a single bar.

    Attributes:
        signal: TrafficLight state (GREEN, YELLOW, RED)
        score: Aggregate anticipation score (0-1)
        components: Individual indicator contributions
        message: Human-readable interpretation
    """
    signal: TrafficLight
    score: float
    components: dict[str, float]
    message: str


@njit(cache=True)
def calculate_traffic_light_scores(
    squeeze_on: np.ndarray,
    squeeze_intensity: np.ndarray,
    bullish_divergence: np.ndarray,
    bearish_divergence: np.ndarray,
    volume_exhaustion: np.ndarray,
    er_acceleration: np.ndarray,
    adx_acceleration: np.ndarray,
    squeeze_weight: float = 0.30,
    divergence_weight: float = 0.25,
    exhaustion_weight: float = 0.20,
    acceleration_weight: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate traffic light scores from anticipation indicators.

    Aggregates multiple indicators into a single 0-1 score, then
    maps to traffic light state based on thresholds.

    Parameters
    ----------
    squeeze_on : np.ndarray
        Squeeze active indicator (0 or 1)
    squeeze_intensity : np.ndarray
        Squeeze tightness (0-1)
    bullish_divergence : np.ndarray
        Bullish RSI divergence score (0-1)
    bearish_divergence : np.ndarray
        Bearish RSI divergence score (0-1)
    volume_exhaustion : np.ndarray
        Trend exhaustion score (0-1)
    er_acceleration : np.ndarray
        Efficiency ratio acceleration
    adx_acceleration : np.ndarray
        ADX acceleration
    squeeze_weight : float
        Weight for squeeze component (default: 0.30)
    divergence_weight : float
        Weight for divergence component (default: 0.25)
    exhaustion_weight : float
        Weight for volume exhaustion component (default: 0.20)
    acceleration_weight : float
        Weight for regime acceleration component (default: 0.25)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (scores, signals)
        - scores: Aggregate anticipation score (0-1)
        - signals: Traffic light state (0=GREEN, 1=YELLOW, 2=RED)
    """
    n = len(squeeze_on)
    scores = np.zeros(n, dtype=np.float64)
    signals = np.zeros(n, dtype=np.int32)

    # Thresholds for traffic light states
    YELLOW_THRESHOLD = 0.3
    RED_THRESHOLD = 0.6

    for i in range(n):
        # Squeeze component: active squeeze with high intensity
        squeeze_score = squeeze_on[i] * squeeze_intensity[i]

        # Divergence component: max of bullish and bearish
        div_score = max(bullish_divergence[i], bearish_divergence[i])

        # Exhaustion component: direct score
        exhaust_score = volume_exhaustion[i]

        # Acceleration component: significant change in regime indicators
        # Normalize acceleration to 0-1 range (typical values are small)
        er_acc_norm = min(1.0, abs(er_acceleration[i]) * 50.0)  # Scale factor
        adx_acc_norm = min(1.0, abs(adx_acceleration[i]) * 10.0)  # Scale factor
        accel_score = max(er_acc_norm, adx_acc_norm)

        # Weighted aggregate score
        total_score = (
            squeeze_weight * squeeze_score +
            divergence_weight * div_score +
            exhaustion_weight * exhaust_score +
            acceleration_weight * accel_score
        )

        scores[i] = min(1.0, total_score)

        # Map to traffic light
        if scores[i] >= RED_THRESHOLD:
            signals[i] = 2  # RED
        elif scores[i] >= YELLOW_THRESHOLD:
            signals[i] = 1  # YELLOW
        else:
            signals[i] = 0  # GREEN

    return scores, signals


def calculate_traffic_light(
    squeeze_on: np.ndarray,
    squeeze_intensity: np.ndarray,
    bullish_divergence: np.ndarray,
    bearish_divergence: np.ndarray,
    volume_exhaustion: np.ndarray,
    er_acceleration: np.ndarray,
    adx_acceleration: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate traffic light signals from anticipation indicators.

    This is the main entry point for traffic light calculation.

    Parameters
    ----------
    squeeze_on : np.ndarray
        Squeeze active indicator (0 or 1)
    squeeze_intensity : np.ndarray
        Squeeze tightness (0-1)
    bullish_divergence : np.ndarray
        Bullish RSI divergence score (0-1)
    bearish_divergence : np.ndarray
        Bearish RSI divergence score (0-1)
    volume_exhaustion : np.ndarray
        Trend exhaustion score (0-1)
    er_acceleration : np.ndarray
        Efficiency ratio acceleration
    adx_acceleration : np.ndarray
        ADX acceleration

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (scores, signals)
        - scores: Aggregate anticipation score (0-1)
        - signals: Traffic light state (0=GREEN, 1=YELLOW, 2=RED)
    """
    return calculate_traffic_light_scores(
        squeeze_on,
        squeeze_intensity,
        bullish_divergence,
        bearish_divergence,
        volume_exhaustion,
        er_acceleration,
        adx_acceleration,
    )


def get_traffic_light_at_bar(
    bar_index: int,
    squeeze_on: np.ndarray,
    squeeze_intensity: np.ndarray,
    bullish_divergence: np.ndarray,
    bearish_divergence: np.ndarray,
    volume_exhaustion: np.ndarray,
    er_acceleration: np.ndarray,
    adx_acceleration: np.ndarray,
) -> TrafficLightResult:
    """
    Get detailed traffic light result for a specific bar.

    Useful for logging and analysis of individual trade entries.

    Parameters
    ----------
    bar_index : int
        The bar index to analyze
    [other params] : np.ndarray
        Anticipation indicator arrays

    Returns
    -------
    TrafficLightResult
        Detailed result with signal, score, components, and message
    """
    scores, signals = calculate_traffic_light(
        squeeze_on,
        squeeze_intensity,
        bullish_divergence,
        bearish_divergence,
        volume_exhaustion,
        er_acceleration,
        adx_acceleration,
    )

    signal = TrafficLight(signals[bar_index])
    score = scores[bar_index]

    # Calculate individual components for analysis
    squeeze_contrib = squeeze_on[bar_index] * squeeze_intensity[bar_index]
    div_contrib = max(bullish_divergence[bar_index], bearish_divergence[bar_index])
    exhaust_contrib = volume_exhaustion[bar_index]
    er_acc_contrib = min(1.0, abs(er_acceleration[bar_index]) * 50.0)
    adx_acc_contrib = min(1.0, abs(adx_acceleration[bar_index]) * 10.0)

    components = {
        "squeeze": squeeze_contrib,
        "divergence": div_contrib,
        "exhaustion": exhaust_contrib,
        "er_acceleration": er_acc_contrib,
        "adx_acceleration": adx_acc_contrib,
    }

    # Generate human-readable message
    if signal == TrafficLight.GREEN:
        message = "All clear - no significant regime change signals"
    elif signal == TrafficLight.YELLOW:
        warnings = []
        if squeeze_contrib > 0.2:
            warnings.append("volatility squeeze forming")
        if div_contrib > 0.2:
            div_type = "bullish" if bullish_divergence[bar_index] > bearish_divergence[bar_index] else "bearish"
            warnings.append(f"{div_type} divergence detected")
        if exhaust_contrib > 0.2:
            warnings.append("volume exhaustion")
        if er_acc_contrib > 0.2 or adx_acc_contrib > 0.2:
            warnings.append("regime indicators accelerating")
        message = f"Caution - {', '.join(warnings) if warnings else 'moderate signals'}"
    else:  # RED
        alerts = []
        if squeeze_contrib > 0.4:
            alerts.append("tight squeeze")
        if div_contrib > 0.4:
            div_type = "bullish" if bullish_divergence[bar_index] > bearish_divergence[bar_index] else "bearish"
            alerts.append(f"strong {div_type} divergence")
        if exhaust_contrib > 0.4:
            alerts.append("significant exhaustion")
        if er_acc_contrib > 0.4 or adx_acc_contrib > 0.4:
            alerts.append("rapid regime change")
        message = f"Warning - {', '.join(alerts) if alerts else 'high probability regime change'}"

    return TrafficLightResult(
        signal=signal,
        score=score,
        components=components,
        message=message,
    )


def format_traffic_light_summary(
    signals: np.ndarray,
    scores: np.ndarray,
) -> str:
    """
    Format a summary of traffic light states over a period.

    Useful for backtest reports.

    Parameters
    ----------
    signals : np.ndarray
        Traffic light signals array
    scores : np.ndarray
        Aggregate scores array

    Returns
    -------
    str
        Formatted summary string
    """
    n = len(signals)
    green_count = np.sum(signals == TrafficLight.GREEN)
    yellow_count = np.sum(signals == TrafficLight.YELLOW)
    red_count = np.sum(signals == TrafficLight.RED)

    avg_score = np.mean(scores)
    max_score = np.max(scores)

    return (
        f"Traffic Light Summary ({n} bars):\n"
        f"  GREEN:  {green_count:4d} ({green_count/n*100:5.1f}%)\n"
        f"  YELLOW: {yellow_count:4d} ({yellow_count/n*100:5.1f}%)\n"
        f"  RED:    {red_count:4d} ({red_count/n*100:5.1f}%)\n"
        f"  Avg Score: {avg_score:.3f}\n"
        f"  Max Score: {max_score:.3f}"
    )


@njit(cache=True)
def adjust_size_for_traffic_light(
    base_size: float,
    signal: int,
    green_mult: float = 1.0,
    yellow_mult: float = 0.5,
    red_mult: float = 0.0,
) -> float:
    """
    Adjust position size based on traffic light signal.

    NOTE: This function exists for FUTURE USE when traffic light
    is validated and made active. Currently not used in live trading.

    Parameters
    ----------
    base_size : float
        Base position size (shares)
    signal : int
        Traffic light signal (0=GREEN, 1=YELLOW, 2=RED)
    green_mult : float
        Size multiplier for GREEN (default: 1.0 = 100%)
    yellow_mult : float
        Size multiplier for YELLOW (default: 0.5 = 50%)
    red_mult : float
        Size multiplier for RED (default: 0.0 = no entry)

    Returns
    -------
    float
        Adjusted position size
    """
    if signal == 0:  # GREEN
        return base_size * green_mult
    elif signal == 1:  # YELLOW
        return base_size * yellow_mult
    else:  # RED
        return base_size * red_mult
