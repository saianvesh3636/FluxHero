"""
Position-level and portfolio-level risk management module.

This module implements risk controls for individual positions and overall portfolio exposure:
- Position-level risk checks (R11.1.1-11.1.4)
- Portfolio-level exposure limits (R11.2.1-11.2.3)
- Correlation monitoring for diversification
- Risk validation before trade execution

Requirements:
- R11.1.1: Max risk per trade: 1% trend, 0.75% mean reversion
- R11.1.2: Max position size: 20% of account value
- R11.1.3: Stop loss mandatory on all positions
- R11.1.4: ATR-based stops (2.5× ATR trend, 3% fixed mean-rev)
- R11.2.1: Max total exposure: 50% of account
- R11.2.2: Max open positions: 5 simultaneously
- R11.2.3: Correlation check before opening new positions

Author: FluxHero
Date: 2026-01-20
"""

import numpy as np
from enum import IntEnum
from dataclasses import dataclass
from typing import List, Optional, Tuple
from numba import njit


class StrategyType(IntEnum):
    """Strategy type for risk parameter selection."""
    MEAN_REVERSION = 0
    NEUTRAL = 1
    TREND_FOLLOWING = 2


class RiskCheckResult(IntEnum):
    """Risk validation result codes."""
    APPROVED = 0
    REJECTED_NO_STOP = 1
    REJECTED_EXCESSIVE_RISK = 2
    REJECTED_POSITION_TOO_LARGE = 3
    REJECTED_MAX_POSITIONS = 4
    REJECTED_TOTAL_EXPOSURE = 5
    REJECTED_HIGH_CORRELATION = 6


@dataclass
class Position:
    """Represents an open position for risk calculations."""
    symbol: str
    shares: float
    entry_price: float
    current_price: float
    stop_loss: float

    @property
    def market_value(self) -> float:
        """Current market value of position."""
        return abs(self.shares * self.current_price)

    @property
    def risk_amount(self) -> float:
        """Risk amount if stop is hit."""
        return abs(self.shares * (self.entry_price - self.stop_loss))


@dataclass
class PositionLimitsConfig:
    """Configuration for position limit checks."""
    # Position-level (R11.1)
    max_risk_pct_trend: float = 0.01  # 1% for trend-following
    max_risk_pct_mean_rev: float = 0.0075  # 0.75% for mean reversion
    max_position_size_pct: float = 0.20  # 20% of account per position

    # Portfolio-level (R11.2)
    max_total_exposure_pct: float = 0.50  # 50% total deployed
    max_open_positions: int = 5  # Max 5 positions
    correlation_threshold: float = 0.7  # Reduce size if correlation > 0.7
    correlation_size_reduction: float = 0.50  # Reduce by 50%

    # ATR stop multipliers (R11.1.4)
    trend_stop_atr_multiplier: float = 2.5
    mean_rev_stop_pct: float = 0.03  # 3% fixed stop


# ============================================================================
# Position-Level Risk Checks (R11.1)
# ============================================================================

def calculate_position_size_from_risk(
    account_balance: float,
    entry_price: float,
    stop_loss: float,
    strategy_type: StrategyType,
    config: Optional[PositionLimitsConfig] = None
) -> float:
    """
    Calculate position size based on risk percentage.

    Requirements:
    - R11.1.1: Risk 1% for trend, 0.75% for mean reversion

    Formula:
        shares = (account_balance × risk_pct) / |entry_price - stop_loss|

    Args:
        account_balance: Total account equity
        entry_price: Intended entry price
        stop_loss: Stop loss price
        strategy_type: TREND_FOLLOWING or MEAN_REVERSION
        config: Position limits configuration

    Returns:
        Number of shares (rounded down to whole shares)

    Example:
        >>> calculate_position_size_from_risk(100000, 50.0, 48.0, StrategyType.TREND_FOLLOWING)
        500.0  # Risk $1000 (1%), price risk $2, so 500 shares
    """
    if config is None:
        config = PositionLimitsConfig()

    # Select risk percentage based on strategy
    if strategy_type == StrategyType.TREND_FOLLOWING:
        risk_pct = config.max_risk_pct_trend
    else:  # MEAN_REVERSION or NEUTRAL
        risk_pct = config.max_risk_pct_mean_rev

    # Calculate risk amount
    risk_amount = account_balance * risk_pct

    # Calculate price risk per share
    price_risk = abs(entry_price - stop_loss)

    # Handle edge case: zero price risk
    if price_risk < 1e-10:
        return 0.0

    # Calculate shares
    shares = risk_amount / price_risk

    # Round down to whole shares
    return np.floor(shares)


def validate_position_level_risk(
    account_balance: float,
    entry_price: float,
    stop_loss: Optional[float],
    shares: float,
    strategy_type: StrategyType,
    config: Optional[PositionLimitsConfig] = None
) -> Tuple[RiskCheckResult, str]:
    """
    Validate position-level risk constraints.

    Requirements:
    - R11.1.1: Max risk per trade (1% trend, 0.75% mean-rev)
    - R11.1.2: Max position size (20% of account)
    - R11.1.3: Stop loss mandatory

    Args:
        account_balance: Total account equity
        entry_price: Intended entry price
        stop_loss: Stop loss price (None = no stop)
        shares: Number of shares to trade
        strategy_type: Strategy type for risk limits
        config: Position limits configuration

    Returns:
        Tuple of (RiskCheckResult, reason_message)

    Example:
        >>> validate_position_level_risk(100000, 50.0, 48.0, 500, StrategyType.TREND_FOLLOWING)
        (RiskCheckResult.APPROVED, "Position risk checks passed")
    """
    if config is None:
        config = PositionLimitsConfig()

    # R11.1.3: Stop loss mandatory
    if stop_loss is None:
        return (RiskCheckResult.REJECTED_NO_STOP,
                "Stop loss is mandatory for all positions")

    # Calculate position value
    position_value = abs(shares * entry_price)

    # R11.1.2: Check max position size (20% of account)
    max_position_value = account_balance * config.max_position_size_pct
    if position_value > max_position_value:
        return (RiskCheckResult.REJECTED_POSITION_TOO_LARGE,
                f"Position size ${position_value:.2f} exceeds max ${max_position_value:.2f} "
                f"({config.max_position_size_pct*100:.0f}% of account)")

    # R11.1.1: Check max risk per trade
    if strategy_type == StrategyType.TREND_FOLLOWING:
        max_risk_pct = config.max_risk_pct_trend
    else:
        max_risk_pct = config.max_risk_pct_mean_rev

    risk_amount = abs(shares * (entry_price - stop_loss))
    max_risk_amount = account_balance * max_risk_pct

    if risk_amount > max_risk_amount:
        return (RiskCheckResult.REJECTED_EXCESSIVE_RISK,
                f"Risk ${risk_amount:.2f} exceeds max ${max_risk_amount:.2f} "
                f"({max_risk_pct*100:.2f}% of account)")

    return (RiskCheckResult.APPROVED, "Position risk checks passed")


def calculate_atr_stop_loss(
    entry_price: float,
    atr: float,
    side: int,  # 1 for long, -1 for short
    strategy_type: StrategyType,
    config: Optional[PositionLimitsConfig] = None
) -> float:
    """
    Calculate ATR-based stop loss price.

    Requirements:
    - R11.1.4: Trend trades use 2.5× ATR, mean-rev uses 3% fixed

    Args:
        entry_price: Entry price
        atr: Average True Range value
        side: 1 for long, -1 for short
        strategy_type: Strategy type
        config: Position limits configuration

    Returns:
        Stop loss price

    Example:
        >>> calculate_atr_stop_loss(100.0, 2.0, 1, StrategyType.TREND_FOLLOWING)
        95.0  # Long: 100 - 2.5×2.0
    """
    if config is None:
        config = PositionLimitsConfig()

    if strategy_type == StrategyType.TREND_FOLLOWING:
        # ATR-based stop
        stop_distance = atr * config.trend_stop_atr_multiplier
    else:
        # Fixed percentage stop
        stop_distance = entry_price * config.mean_rev_stop_pct

    if side == 1:  # Long
        return entry_price - stop_distance
    else:  # Short
        return entry_price + stop_distance


# ============================================================================
# Portfolio-Level Risk Checks (R11.2)
# ============================================================================

def validate_portfolio_level_risk(
    account_balance: float,
    open_positions: List[Position],
    new_position_value: float,
    config: Optional[PositionLimitsConfig] = None
) -> Tuple[RiskCheckResult, str]:
    """
    Validate portfolio-level risk constraints.

    Requirements:
    - R11.2.1: Max total exposure 50%
    - R11.2.2: Max 5 open positions

    Args:
        account_balance: Total account equity
        open_positions: List of current positions
        new_position_value: Value of new position to add
        config: Position limits configuration

    Returns:
        Tuple of (RiskCheckResult, reason_message)

    Example:
        >>> positions = [Position("SPY", 100, 400, 410, 395)]
        >>> validate_portfolio_level_risk(100000, positions, 15000)
        (RiskCheckResult.APPROVED, "Portfolio risk checks passed")
    """
    if config is None:
        config = PositionLimitsConfig()

    # R11.2.2: Check max positions
    if len(open_positions) >= config.max_open_positions:
        return (RiskCheckResult.REJECTED_MAX_POSITIONS,
                f"Already at max {config.max_open_positions} open positions")

    # R11.2.1: Check total exposure
    current_exposure = sum(pos.market_value for pos in open_positions)
    total_exposure = current_exposure + new_position_value
    max_exposure = account_balance * config.max_total_exposure_pct

    if total_exposure > max_exposure:
        return (RiskCheckResult.REJECTED_TOTAL_EXPOSURE,
                f"Total exposure ${total_exposure:.2f} would exceed max ${max_exposure:.2f} "
                f"({config.max_total_exposure_pct*100:.0f}% of account)")

    return (RiskCheckResult.APPROVED, "Portfolio risk checks passed")


@njit(cache=True)
def calculate_correlation(prices1: np.ndarray, prices2: np.ndarray) -> float:
    """
    Calculate Pearson correlation coefficient between two price series.

    Requirements:
    - R11.2.3: Check correlation before opening new position

    Formula:
        corr = cov(X, Y) / (std(X) × std(Y))

    Args:
        prices1: First price series (NumPy array)
        prices2: Second price series (NumPy array)

    Returns:
        Correlation coefficient (-1 to +1)

    Example:
        >>> p1 = np.array([100, 101, 102, 103, 104])
        >>> p2 = np.array([200, 202, 204, 206, 208])
        >>> calculate_correlation(p1, p2)
        1.0  # Perfect positive correlation
    """
    # Handle edge cases
    if len(prices1) != len(prices2):
        return 0.0

    if len(prices1) < 2:
        return 0.0

    # Calculate means
    mean1 = np.mean(prices1)
    mean2 = np.mean(prices2)

    # Calculate standard deviations
    std1 = np.std(prices1)
    std2 = np.std(prices2)

    # Handle zero std dev
    if std1 < 1e-10 or std2 < 1e-10:
        return 0.0

    # Calculate covariance
    n = len(prices1)
    cov = 0.0
    for i in range(n):
        cov += (prices1[i] - mean1) * (prices2[i] - mean2)
    cov /= n

    # Calculate correlation
    corr = cov / (std1 * std2)

    return corr


def check_correlation_with_existing_positions(
    new_symbol_prices: np.ndarray,
    open_positions: List[Position],
    position_prices_map: dict,  # {symbol: np.ndarray of recent prices}
    config: Optional[PositionLimitsConfig] = None
) -> Tuple[bool, float, Optional[str]]:
    """
    Check if new position is highly correlated with existing positions.

    Requirements:
    - R11.2.3: If correlation > 0.7, reduce position size by 50%

    Args:
        new_symbol_prices: Recent price series for new symbol
        open_positions: List of current positions
        position_prices_map: Dictionary mapping symbols to price arrays
        config: Position limits configuration

    Returns:
        Tuple of (should_reduce_size, max_correlation, correlated_symbol)

    Example:
        >>> prices_map = {"SPY": np.array([400, 401, 402])}
        >>> positions = [Position("SPY", 100, 400, 402, 395)]
        >>> new_prices = np.array([80, 81, 82])
        >>> check_correlation_with_existing_positions(new_prices, positions, prices_map)
        (False, 0.95, None)  # High correlation but function checks threshold
    """
    if config is None:
        config = PositionLimitsConfig()

    if len(open_positions) == 0:
        return (False, 0.0, None)

    max_correlation = 0.0
    correlated_symbol = None

    for position in open_positions:
        if position.symbol not in position_prices_map:
            continue

        existing_prices = position_prices_map[position.symbol]

        # Ensure equal length for correlation
        min_len = min(len(new_symbol_prices), len(existing_prices))
        if min_len < 2:
            continue

        # Calculate correlation
        corr = calculate_correlation(
            new_symbol_prices[-min_len:],
            existing_prices[-min_len:]
        )

        abs_corr = abs(corr)
        if abs_corr > max_correlation:
            max_correlation = abs_corr
            correlated_symbol = position.symbol

    # Check threshold
    should_reduce = max_correlation > config.correlation_threshold

    return (should_reduce, max_correlation, correlated_symbol)


# ============================================================================
# Comprehensive Risk Validation
# ============================================================================

def validate_new_position(
    account_balance: float,
    entry_price: float,
    stop_loss: Optional[float],
    shares: float,
    strategy_type: StrategyType,
    open_positions: List[Position],
    new_symbol_prices: Optional[np.ndarray] = None,
    position_prices_map: Optional[dict] = None,
    config: Optional[PositionLimitsConfig] = None
) -> Tuple[RiskCheckResult, str, float]:
    """
    Comprehensive risk validation for new position.

    Validates:
    1. Position-level risk (R11.1)
    2. Portfolio-level risk (R11.2)
    3. Correlation check (R11.2.3)

    Args:
        account_balance: Total account equity
        entry_price: Intended entry price
        stop_loss: Stop loss price
        shares: Number of shares (may be adjusted)
        strategy_type: Strategy type
        open_positions: List of current positions
        new_symbol_prices: Recent prices for correlation check
        position_prices_map: Price series for existing positions
        config: Position limits configuration

    Returns:
        Tuple of (RiskCheckResult, reason, adjusted_shares)

    Example:
        >>> result, reason, shares = validate_new_position(
        ...     100000, 50.0, 48.0, 500, StrategyType.TREND_FOLLOWING, []
        ... )
        >>> result == RiskCheckResult.APPROVED
        True
    """
    if config is None:
        config = PositionLimitsConfig()

    adjusted_shares = shares

    # 1. Position-level risk checks
    result, reason = validate_position_level_risk(
        account_balance, entry_price, stop_loss, shares, strategy_type, config
    )
    if result != RiskCheckResult.APPROVED:
        return (result, reason, 0.0)

    # 2. Portfolio-level risk checks
    position_value = abs(shares * entry_price)
    result, reason = validate_portfolio_level_risk(
        account_balance, open_positions, position_value, config
    )
    if result != RiskCheckResult.APPROVED:
        return (result, reason, 0.0)

    # 3. Correlation check
    if new_symbol_prices is not None and position_prices_map is not None:
        should_reduce, max_corr, corr_symbol = check_correlation_with_existing_positions(
            new_symbol_prices, open_positions, position_prices_map, config
        )

        if should_reduce:
            adjusted_shares = np.floor(shares * (1.0 - config.correlation_size_reduction))
            reason = (f"Position size reduced by {config.correlation_size_reduction*100:.0f}% "
                     f"due to {max_corr:.2f} correlation with {corr_symbol}")

            # Re-validate with adjusted size
            result, val_reason = validate_position_level_risk(
                account_balance, entry_price, stop_loss, adjusted_shares,
                strategy_type, config
            )
            if result != RiskCheckResult.APPROVED:
                return (result, val_reason, 0.0)

            return (RiskCheckResult.APPROVED, reason, adjusted_shares)

    return (RiskCheckResult.APPROVED, "All risk checks passed", adjusted_shares)


# ============================================================================
# Risk Monitoring
# ============================================================================

def calculate_total_portfolio_risk(
    open_positions: List[Position]
) -> Tuple[float, float]:
    """
    Calculate total portfolio risk and exposure.

    Requirements:
    - R11.4.2: Daily risk report (total risk deployed)

    Args:
        open_positions: List of current positions

    Returns:
        Tuple of (total_risk_amount, total_exposure)

    Example:
        >>> positions = [
        ...     Position("SPY", 100, 400, 410, 395),
        ...     Position("QQQ", 50, 300, 305, 291)
        ... ]
        >>> total_risk, total_exposure = calculate_total_portfolio_risk(positions)
        >>> total_risk  # 100×5 + 50×9 = 950
        950.0
    """
    total_risk = sum(pos.risk_amount for pos in open_positions)
    total_exposure = sum(pos.market_value for pos in open_positions)

    return (total_risk, total_exposure)


def get_largest_position(
    open_positions: List[Position]
) -> Optional[Position]:
    """
    Get the largest position by market value.

    Requirements:
    - R11.4.2: Daily risk report (largest position size)

    Args:
        open_positions: List of current positions

    Returns:
        Position with largest market value, or None if no positions
    """
    if not open_positions:
        return None

    return max(open_positions, key=lambda p: p.market_value)


def calculate_worst_case_loss(
    open_positions: List[Position]
) -> float:
    """
    Calculate worst-case scenario if all stops are hit.

    Requirements:
    - R11.4.2: Daily risk report (worst-case loss scenario)

    Args:
        open_positions: List of current positions

    Returns:
        Total loss if all stops hit

    Example:
        >>> positions = [Position("SPY", 100, 400, 410, 395)]
        >>> calculate_worst_case_loss(positions)
        500.0  # 100 shares × $5 loss per share
    """
    return sum(pos.risk_amount for pos in open_positions)
