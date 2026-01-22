"""
Drawdown circuit breaker and kill switch implementation.

This module implements automatic risk controls based on drawdown levels:
- Tracks equity peak and current drawdown
- Reduces position sizes at 15% drawdown (R11.3.2)
- Closes all positions and disables trading at 20% drawdown (R11.3.3)
- Provides real-time risk monitoring (R11.4.1)
- Generates daily risk reports (R11.4.2)

Requirements:
- R11.3.1: Track drawdown from equity peak
- R11.3.2: At 15% drawdown: reduce sizes 50%, tighten stops, alert
- R11.3.3: At 20% drawdown: close all, disable trading, require manual review
- R11.4.1: Real-time risk monitoring (drawdown, exposure, correlation)
- R11.4.2: Daily risk report generation

Author: FluxHero
Date: 2026-01-20
"""

from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum

import numpy as np


class DrawdownLevel(IntEnum):
    """Drawdown severity levels for circuit breakers."""

    NORMAL = 0  # Below 15% drawdown
    WARNING = 1  # 15-20% drawdown (reduce sizes)
    CRITICAL = 2  # 20%+ drawdown (stop trading)


class TradingStatus(IntEnum):
    """Trading system status."""

    ACTIVE = 0  # Normal trading
    REDUCED = 1  # Reduced size mode (15% DD)
    DISABLED = 2  # Trading disabled (20% DD)


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

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        return self.shares * (self.current_price - self.entry_price)


@dataclass
class DrawdownCircuitBreakerConfig:
    """Configuration for drawdown circuit breakers."""

    # Drawdown thresholds (R11.3)
    warning_drawdown_pct: float = 0.15  # 15% drawdown warning
    critical_drawdown_pct: float = 0.20  # 20% drawdown critical

    # Actions at warning level (R11.3.2)
    warning_size_reduction: float = 0.50  # Reduce sizes by 50%
    warning_stop_multiplier: float = 2.0  # Tighten stops to 2.0× ATR
    normal_stop_multiplier: float = 2.5  # Normal stops at 2.5× ATR

    # Manual review required at critical level (R11.3.3)
    require_manual_review: bool = True


@dataclass
class RiskMetrics:
    """Real-time risk metrics for monitoring."""

    # Drawdown tracking (R11.3.1, R11.4.1)
    equity_peak: float
    current_equity: float
    current_drawdown: float
    current_drawdown_pct: float
    drawdown_level: DrawdownLevel

    # Portfolio exposure (R11.4.1)
    total_exposure: float
    exposure_pct: float
    num_positions: int

    # Risk metrics (R11.4.2)
    total_risk_deployed: float
    worst_case_loss: float
    largest_position_symbol: str | None
    largest_position_value: float

    # Correlation (R11.4.1)
    correlation_matrix: dict[tuple[str, str], float] | None = None


@dataclass
class DailyRiskReport:
    """Daily risk report summary."""

    date: datetime
    account_balance: float
    equity_peak: float
    current_drawdown_pct: float
    num_positions: int
    total_exposure: float
    exposure_pct: float
    total_risk_deployed: float
    worst_case_loss: float
    largest_position: str | None
    largest_position_value: float
    trading_status: TradingStatus
    alerts: list[str]


# ============================================================================
# Drawdown Tracking (R11.3.1)
# ============================================================================


class EquityTracker:
    """
    Tracks equity peak and current drawdown.

    Requirements:
    - R11.3.1: Track drawdown from equity peak
    """

    def __init__(self, initial_equity: float):
        """
        Initialize equity tracker.

        Args:
            initial_equity: Starting account equity
        """
        self.equity_peak = initial_equity
        self.current_equity = initial_equity
        self.equity_history: list[tuple[datetime, float]] = []

    def update_equity(self, new_equity: float, timestamp: datetime | None = None) -> float:
        """
        Update current equity and track peak.

        Args:
            new_equity: Current account equity
            timestamp: Timestamp of update (defaults to now)

        Returns:
            Current drawdown percentage

        Example:
            >>> tracker = EquityTracker(100000)
            >>> tracker.update_equity(95000)
            0.05  # 5% drawdown
        """
        if timestamp is None:
            timestamp = datetime.now()

        self.current_equity = new_equity
        self.equity_history.append((timestamp, new_equity))

        # Update peak if equity reaches new high
        if new_equity > self.equity_peak:
            self.equity_peak = new_equity

        return self.calculate_drawdown_pct()

    def calculate_drawdown(self) -> float:
        """
        Calculate current drawdown in dollars.

        Returns:
            Drawdown amount (peak - current)
        """
        return self.equity_peak - self.current_equity

    def calculate_drawdown_pct(self) -> float:
        """
        Calculate current drawdown percentage.

        Formula:
            drawdown_pct = (peak - current) / peak

        Returns:
            Drawdown percentage (0.0 to 1.0)

        Example:
            >>> tracker = EquityTracker(100000)
            >>> tracker.current_equity = 85000
            >>> tracker.calculate_drawdown_pct()
            0.15  # 15% drawdown
        """
        if self.equity_peak == 0:
            return 0.0

        return (self.equity_peak - self.current_equity) / self.equity_peak

    def reset_peak(self, new_peak: float):
        """
        Manually reset equity peak (use with caution).

        Args:
            new_peak: New equity peak value
        """
        self.equity_peak = new_peak


# ============================================================================
# Circuit Breaker Logic (R11.3.2, R11.3.3)
# ============================================================================


class DrawdownCircuitBreaker:
    """
    Implements drawdown-based circuit breakers.

    Requirements:
    - R11.3.2: At 15% DD: reduce sizes, tighten stops, alert
    - R11.3.3: At 20% DD: close all, disable trading
    """

    def __init__(self, config: DrawdownCircuitBreakerConfig | None = None):
        """
        Initialize circuit breaker.

        Args:
            config: Circuit breaker configuration
        """
        if config is None:
            config = DrawdownCircuitBreakerConfig()

        self.config = config
        self.trading_status = TradingStatus.ACTIVE
        self.manual_review_required = False
        self.alerts: list[tuple[datetime, str]] = []

    def check_drawdown_level(self, drawdown_pct: float) -> DrawdownLevel:
        """
        Determine drawdown severity level.

        Args:
            drawdown_pct: Current drawdown percentage (0.0 to 1.0)

        Returns:
            DrawdownLevel (NORMAL, WARNING, or CRITICAL)

        Example:
            >>> breaker = DrawdownCircuitBreaker()
            >>> breaker.check_drawdown_level(0.16)
            DrawdownLevel.WARNING
        """
        if drawdown_pct >= self.config.critical_drawdown_pct:
            return DrawdownLevel.CRITICAL
        elif drawdown_pct >= self.config.warning_drawdown_pct:
            return DrawdownLevel.WARNING
        else:
            return DrawdownLevel.NORMAL

    def update_trading_status(
        self, drawdown_pct: float, timestamp: datetime | None = None
    ) -> tuple[TradingStatus, list[str]]:
        """
        Update trading status based on drawdown.

        Requirements:
        - R11.3.2: At 15% DD: switch to REDUCED mode
        - R11.3.3: At 20% DD: switch to DISABLED mode

        Args:
            drawdown_pct: Current drawdown percentage
            timestamp: Timestamp of update

        Returns:
            Tuple of (TradingStatus, list of new alerts)

        Example:
            >>> breaker = DrawdownCircuitBreaker()
            >>> status, alerts = breaker.update_trading_status(0.16)
            >>> status == TradingStatus.REDUCED
            True
        """
        if timestamp is None:
            timestamp = datetime.now()

        level = self.check_drawdown_level(drawdown_pct)
        new_alerts = []

        # R11.3.3: Critical drawdown (20%+)
        if level == DrawdownLevel.CRITICAL:
            if self.trading_status != TradingStatus.DISABLED:
                self.trading_status = TradingStatus.DISABLED
                self.manual_review_required = self.config.require_manual_review

                alert = (
                    f"CRITICAL: 20% drawdown reached ({drawdown_pct * 100:.1f}%). "
                    "All trading disabled. Manual review required."
                )
                new_alerts.append(alert)
                self.alerts.append((timestamp, alert))

        # R11.3.2: Warning drawdown (15-20%)
        elif level == DrawdownLevel.WARNING:
            if self.trading_status == TradingStatus.ACTIVE:
                self.trading_status = TradingStatus.REDUCED

                alert = (
                    f"WARNING: 15% drawdown reached ({drawdown_pct * 100:.1f}%). "
                    "Position sizes reduced 50%, stops tightened to 2.0× ATR."
                )
                new_alerts.append(alert)
                self.alerts.append((timestamp, alert))

        # Normal operation
        else:
            if self.trading_status == TradingStatus.REDUCED:
                # Recovery from warning level
                alert = (
                    f"INFO: Drawdown recovered to {drawdown_pct * 100:.1f}%. "
                    "Normal trading resumed."
                )
                new_alerts.append(alert)
                self.alerts.append((timestamp, alert))

            self.trading_status = TradingStatus.ACTIVE

        return (self.trading_status, new_alerts)

    def get_position_size_multiplier(self) -> float:
        """
        Get position size multiplier based on trading status.

        Requirements:
        - R11.3.2: Reduce sizes by 50% at warning level

        Returns:
            Multiplier (1.0 normal, 0.5 reduced, 0.0 disabled)

        Example:
            >>> breaker = DrawdownCircuitBreaker()
            >>> breaker.trading_status = TradingStatus.REDUCED
            >>> breaker.get_position_size_multiplier()
            0.5
        """
        if self.trading_status == TradingStatus.DISABLED:
            return 0.0
        elif self.trading_status == TradingStatus.REDUCED:
            return 1.0 - self.config.warning_size_reduction
        else:
            return 1.0

    def get_stop_loss_multiplier(self) -> float:
        """
        Get stop loss ATR multiplier based on trading status.

        Requirements:
        - R11.3.2: Tighten stops to 2.0× ATR at warning level

        Returns:
            ATR multiplier (2.5 normal, 2.0 reduced)

        Example:
            >>> breaker = DrawdownCircuitBreaker()
            >>> breaker.trading_status = TradingStatus.REDUCED
            >>> breaker.get_stop_loss_multiplier()
            2.0
        """
        if self.trading_status == TradingStatus.REDUCED:
            return self.config.warning_stop_multiplier
        else:
            return self.config.normal_stop_multiplier

    def can_open_new_position(self) -> tuple[bool, str]:
        """
        Check if new positions can be opened.

        Requirements:
        - R11.3.3: Disable trading at critical level

        Returns:
            Tuple of (can_trade, reason)

        Example:
            >>> breaker = DrawdownCircuitBreaker()
            >>> breaker.trading_status = TradingStatus.DISABLED
            >>> can_trade, reason = breaker.can_open_new_position()
            >>> can_trade
            False
        """
        if self.trading_status == TradingStatus.DISABLED:
            return (False, "Trading disabled due to 20% drawdown. Manual review required.")
        elif self.manual_review_required:
            return (False, "Manual review required before resuming trading.")
        else:
            return (True, "Trading allowed")

    def should_close_all_positions(self) -> bool:
        """
        Check if all positions should be closed.

        Requirements:
        - R11.3.3: Close all positions at 20% drawdown

        Returns:
            True if positions should be closed
        """
        return self.trading_status == TradingStatus.DISABLED

    def acknowledge_manual_review(self):
        """
        Acknowledge manual review and allow trading to resume.

        Requirements:
        - R11.3.3: Require manual review before resuming
        """
        self.manual_review_required = False
        self.trading_status = TradingStatus.ACTIVE

    def get_recent_alerts(self, n: int = 10) -> list[tuple[datetime, str]]:
        """
        Get recent alerts.

        Args:
            n: Number of recent alerts to retrieve

        Returns:
            List of (timestamp, alert_message) tuples
        """
        return self.alerts[-n:]


# ============================================================================
# Real-Time Risk Monitoring (R11.4.1)
# ============================================================================


def calculate_risk_metrics(
    account_balance: float,
    equity_peak: float,
    current_equity: float,
    open_positions: list[Position],
    config: DrawdownCircuitBreakerConfig | None = None,
) -> RiskMetrics:
    """
    Calculate comprehensive risk metrics for monitoring.

    Requirements:
    - R11.4.1: Real-time display of drawdown, exposure, risk per position

    Args:
        account_balance: Current account balance
        equity_peak: Historical equity peak
        current_equity: Current equity value
        open_positions: List of open positions
        config: Circuit breaker configuration

    Returns:
        RiskMetrics object with all metrics

    Example:
        >>> positions = [Position("SPY", 100, 400, 410, 395)]
        >>> metrics = calculate_risk_metrics(100000, 105000, 100000, positions)
        >>> metrics.current_drawdown_pct
        0.047619...  # ~4.76% drawdown
    """
    if config is None:
        config = DrawdownCircuitBreakerConfig()

    # Calculate drawdown
    drawdown = equity_peak - current_equity
    drawdown_pct = drawdown / equity_peak if equity_peak > 0 else 0.0

    # Determine drawdown level
    if drawdown_pct >= config.critical_drawdown_pct:
        level = DrawdownLevel.CRITICAL
    elif drawdown_pct >= config.warning_drawdown_pct:
        level = DrawdownLevel.WARNING
    else:
        level = DrawdownLevel.NORMAL

    # Calculate exposure
    total_exposure = sum(pos.market_value for pos in open_positions)
    exposure_pct = total_exposure / account_balance if account_balance > 0 else 0.0

    # Calculate risk metrics
    total_risk = sum(pos.risk_amount for pos in open_positions)
    worst_case = total_risk  # Same as total risk (all stops hit)

    # Find largest position
    largest_symbol = None
    largest_value = 0.0
    if open_positions:
        largest_pos = max(open_positions, key=lambda p: p.market_value)
        largest_symbol = largest_pos.symbol
        largest_value = largest_pos.market_value

    return RiskMetrics(
        equity_peak=equity_peak,
        current_equity=current_equity,
        current_drawdown=drawdown,
        current_drawdown_pct=drawdown_pct,
        drawdown_level=level,
        total_exposure=total_exposure,
        exposure_pct=exposure_pct,
        num_positions=len(open_positions),
        total_risk_deployed=total_risk,
        worst_case_loss=worst_case,
        largest_position_symbol=largest_symbol,
        largest_position_value=largest_value,
    )


def calculate_correlation_matrix(
    position_prices_map: dict[str, np.ndarray],
) -> dict[tuple[str, str], float]:
    """
    Calculate correlation matrix for all open positions.

    Requirements:
    - R11.4.1: Display correlation matrix

    Args:
        position_prices_map: Dictionary mapping symbols to price arrays

    Returns:
        Dictionary mapping (symbol1, symbol2) tuples to correlation values

    Example:
        >>> prices = {"SPY": np.array([100, 101, 102]), "QQQ": np.array([200, 202, 204])}
        >>> matrix = calculate_correlation_matrix(prices)
        >>> matrix[("SPY", "QQQ")]
        1.0  # Perfect correlation
    """
    symbols = list(position_prices_map.keys())
    matrix = {}

    for i, sym1 in enumerate(symbols):
        for sym2 in symbols[i:]:
            if sym1 == sym2:
                matrix[(sym1, sym2)] = 1.0
            else:
                prices1 = position_prices_map[sym1]
                prices2 = position_prices_map[sym2]

                # Ensure equal length
                min_len = min(len(prices1), len(prices2))
                if min_len < 2:
                    matrix[(sym1, sym2)] = 0.0
                    matrix[(sym2, sym1)] = 0.0
                    continue

                # Calculate correlation using NumPy
                corr = np.corrcoef(prices1[-min_len:], prices2[-min_len:])[0, 1]

                # Handle NaN from constant prices
                if np.isnan(corr):
                    corr = 0.0

                matrix[(sym1, sym2)] = corr
                matrix[(sym2, sym1)] = corr

    return matrix


# ============================================================================
# Daily Risk Report (R11.4.2)
# ============================================================================


def generate_daily_risk_report(
    account_balance: float,
    equity_tracker: EquityTracker,
    open_positions: list[Position],
    circuit_breaker: DrawdownCircuitBreaker,
    timestamp: datetime | None = None,
) -> DailyRiskReport:
    """
    Generate comprehensive daily risk report.

    Requirements:
    - R11.4.2: Total risk deployed, largest position, worst-case scenario

    Args:
        account_balance: Current account balance
        equity_tracker: EquityTracker instance
        open_positions: List of open positions
        circuit_breaker: DrawdownCircuitBreaker instance
        timestamp: Report timestamp (defaults to now)

    Returns:
        DailyRiskReport object

    Example:
        >>> tracker = EquityTracker(100000)
        >>> breaker = DrawdownCircuitBreaker()
        >>> positions = [Position("SPY", 100, 400, 410, 395)]
        >>> report = generate_daily_risk_report(100000, tracker, positions, breaker)
        >>> report.num_positions
        1
    """
    if timestamp is None:
        timestamp = datetime.now()

    # Calculate metrics
    metrics = calculate_risk_metrics(
        account_balance, equity_tracker.equity_peak, equity_tracker.current_equity, open_positions
    )

    # Get recent alerts
    recent_alerts = [msg for _, msg in circuit_breaker.get_recent_alerts(5)]

    return DailyRiskReport(
        date=timestamp,
        account_balance=account_balance,
        equity_peak=metrics.equity_peak,
        current_drawdown_pct=metrics.current_drawdown_pct,
        num_positions=metrics.num_positions,
        total_exposure=metrics.total_exposure,
        exposure_pct=metrics.exposure_pct,
        total_risk_deployed=metrics.total_risk_deployed,
        worst_case_loss=metrics.worst_case_loss,
        largest_position=metrics.largest_position_symbol,
        largest_position_value=metrics.largest_position_value,
        trading_status=circuit_breaker.trading_status,
        alerts=recent_alerts,
    )


def format_daily_risk_report(report: DailyRiskReport) -> str:
    """
    Format daily risk report as human-readable string.

    Args:
        report: DailyRiskReport object

    Returns:
        Formatted report string

    Example:
        >>> tracker = EquityTracker(100000)
        >>> breaker = DrawdownCircuitBreaker()
        >>> report = generate_daily_risk_report(100000, tracker, [], breaker)
        >>> text = format_daily_risk_report(report)
        >>> "Daily Risk Report" in text
        True
    """
    lines = [
        "=" * 60,
        f"Daily Risk Report - {report.date.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "Account Status:",
        f"  Balance:           ${report.account_balance:,.2f}",
        f"  Equity Peak:       ${report.equity_peak:,.2f}",
        f"  Current Drawdown:  {report.current_drawdown_pct * 100:.2f}%",
        f"  Trading Status:    {report.trading_status.name}",
        "",
        "Portfolio Exposure:",
        f"  Open Positions:    {report.num_positions}",
        f"  Total Exposure:    ${report.total_exposure:,.2f} ({report.exposure_pct * 100:.1f}%)",
        "",
        "Risk Metrics:",
        f"  Total Risk:        ${report.total_risk_deployed:,.2f}",
        f"  Worst-Case Loss:   ${report.worst_case_loss:,.2f}",
    ]

    if report.largest_position:
        lines.append(
            f"  Largest Position:  {report.largest_position} "
            f"(${report.largest_position_value:,.2f})"
        )
    else:
        lines.append("  Largest Position:  None")

    if report.alerts:
        lines.append("")
        lines.append("Recent Alerts:")
        for alert in report.alerts:
            lines.append(f"  - {alert}")

    lines.append("=" * 60)

    return "\n".join(lines)
