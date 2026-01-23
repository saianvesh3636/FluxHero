"""
Performance Metrics for Backtesting Module.

This module calculates comprehensive performance metrics for backtest results,
including Sharpe ratio, max drawdown, win rate, and other key statistics.

Features:
- Sharpe ratio with configurable risk-free rate (R9.3.1)
- Max drawdown calculation (peak-to-trough decline)
- Win rate, average win/loss ratio
- Annualized returns
- Integration with quantstats library (R9.3.2)
- Metric sanity checks with logging for extreme values

Reference:
- FLUXHERO_REQUIREMENTS.md Feature 9.3: Metrics Calculation
- algorithmic-trading-guide.md → Key Metrics to Track
"""

import logging
from typing import Any

import numpy as np
from numba import njit
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@njit(cache=True)
def calculate_returns(equity_curve: np.ndarray) -> np.ndarray:
    """
    Calculate period-over-period returns from equity curve.

    Parameters
    ----------
    equity_curve : np.ndarray (float64)
        Array of equity values over time

    Returns
    -------
    np.ndarray (float64)
        Array of returns (length = len(equity_curve) - 1)

    Examples
    --------
    >>> equity = np.array([100.0, 105.0, 103.0, 108.0])
    >>> returns = calculate_returns(equity)
    >>> # [0.05, -0.0190476, 0.04854369]
    """
    if len(equity_curve) < 2:
        # Return empty array with explicit dtype
        empty = np.zeros(0, dtype=np.float64)
        return empty

    returns = np.zeros(len(equity_curve) - 1, dtype=np.float64)

    for i in range(1, len(equity_curve)):
        if equity_curve[i - 1] > 0:
            returns[i - 1] = (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
        else:
            returns[i - 1] = 0.0

    return returns


@njit(cache=True)
def calculate_sharpe_ratio(
    returns: np.ndarray, risk_free_rate: float = 0.04, periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sharpe ratio (R9.3.1).

    Sharpe = (Avg Return - Risk Free Rate) / Std Dev of Returns

    Parameters
    ----------
    returns : np.ndarray (float64)
        Array of period returns
    risk_free_rate : float
        Annual risk-free rate (default: 4% = 0.04)
    periods_per_year : int
        Number of periods per year (252 for daily, 252*6.5 for hourly, etc.)

    Returns
    -------
    float
        Annualized Sharpe ratio

    Examples
    --------
    >>> returns = np.array([0.01, -0.005, 0.02, 0.015, -0.01])
    >>> sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.04, periods_per_year=252)
    """
    if len(returns) == 0:
        return 0.0

    # Calculate average return
    avg_return = np.mean(returns)

    # Calculate standard deviation
    std_return = np.std(returns)

    if std_return == 0:
        return 0.0

    # Annualize return and std dev
    annual_return = avg_return * periods_per_year
    annual_std = std_return * np.sqrt(periods_per_year)

    # Calculate Sharpe ratio
    sharpe = (annual_return - risk_free_rate) / annual_std

    return sharpe


@njit(cache=True)
def calculate_max_drawdown(equity_curve: np.ndarray) -> tuple:
    """
    Calculate maximum drawdown (R9.3.1).

    Max drawdown = largest peak-to-trough decline in equity.

    Parameters
    ----------
    equity_curve : np.ndarray (float64)
        Array of equity values over time

    Returns
    -------
    tuple (float, int, int)
        (max_drawdown_pct, peak_index, trough_index)

    Examples
    --------
    >>> equity = np.array([100.0, 110.0, 105.0, 95.0, 100.0, 120.0])
    >>> dd_pct, peak_idx, trough_idx = calculate_max_drawdown(equity)
    >>> dd_pct  # -13.636% (from 110 to 95)
    >>> peak_idx  # 1
    >>> trough_idx  # 3
    """
    if len(equity_curve) == 0:
        return 0.0, -1, -1

    max_dd = 0.0
    peak = equity_curve[0]
    peak_idx = 0
    max_dd_peak_idx = 0
    max_dd_trough_idx = 0

    for i in range(len(equity_curve)):
        # Update peak if new high
        if equity_curve[i] > peak:
            peak = equity_curve[i]
            peak_idx = i

        # Calculate current drawdown from peak
        if peak > 0:
            current_dd = (equity_curve[i] - peak) / peak

            # Update max drawdown if worse
            if current_dd < max_dd:
                max_dd = current_dd
                max_dd_peak_idx = peak_idx
                max_dd_trough_idx = i

    # Convert to percentage
    max_dd_pct = max_dd * 100.0

    return max_dd_pct, max_dd_peak_idx, max_dd_trough_idx


@njit(cache=True)
def calculate_win_rate(pnls: np.ndarray) -> float:
    """
    Calculate win rate (R9.3.1).

    Win rate = number of winning trades / total trades

    Parameters
    ----------
    pnls : np.ndarray (float64)
        Array of trade P&Ls

    Returns
    -------
    float
        Win rate as decimal (0.0 to 1.0)

    Examples
    --------
    >>> pnls = np.array([100.0, -50.0, 200.0, -30.0, 150.0])
    >>> win_rate = calculate_win_rate(pnls)
    >>> # 0.6 (3 wins out of 5 trades)
    """
    if len(pnls) == 0:
        return 0.0

    wins = 0
    for pnl in pnls:
        if pnl > 0:
            wins += 1

    return wins / len(pnls)


@njit(cache=True)
def calculate_avg_win_loss_ratio(pnls: np.ndarray) -> float:
    """
    Calculate average win / average loss ratio (R9.3.1).

    Parameters
    ----------
    pnls : np.ndarray (float64)
        Array of trade P&Ls

    Returns
    -------
    float
        Average win / average loss ratio

    Examples
    --------
    >>> pnls = np.array([100.0, -50.0, 200.0, -30.0, 150.0])
    >>> ratio = calculate_avg_win_loss_ratio(pnls)
    >>> # 150.0 / 40.0 = 3.75
    """
    if len(pnls) == 0:
        return 0.0

    wins = []
    losses = []

    for pnl in pnls:
        if pnl > 0:
            wins.append(pnl)
        elif pnl < 0:
            losses.append(abs(pnl))

    if len(wins) == 0 or len(losses) == 0:
        return 0.0

    avg_win = 0.0
    for w in wins:
        avg_win += w
    avg_win /= len(wins)

    avg_loss = 0.0
    for loss_val in losses:
        avg_loss += loss_val
    avg_loss /= len(losses)

    if avg_loss == 0:
        return 0.0

    return avg_win / avg_loss


@njit(cache=True)
def calculate_total_return(initial_capital: float, final_equity: float) -> tuple:
    """
    Calculate total and percentage return.

    Parameters
    ----------
    initial_capital : float
        Starting capital
    final_equity : float
        Ending equity

    Returns
    -------
    tuple (float, float)
        (total_return_dollars, total_return_pct)

    Examples
    --------
    >>> total_ret, pct_ret = calculate_total_return(100000.0, 125000.0)
    >>> total_ret  # 25000.0
    >>> pct_ret    # 25.0
    """
    total_return = final_equity - initial_capital
    total_return_pct = (total_return / initial_capital) * 100.0 if initial_capital > 0 else 0.0

    return total_return, total_return_pct


@njit(cache=True)
def calculate_annualized_return(total_return_pct: float, num_days: int) -> float:
    """
    Calculate annualized return (CAGR).

    CAGR = (Final / Initial)^(365/days) - 1

    Parameters
    ----------
    total_return_pct : float
        Total return percentage
    num_days : int
        Number of days in backtest period

    Returns
    -------
    float
        Annualized return percentage

    Examples
    --------
    >>> # 25% return over 6 months (182 days)
    >>> ann_ret = calculate_annualized_return(25.0, 182)
    >>> # ~55.6% annualized
    """
    if num_days <= 0:
        return 0.0

    # Convert percentage to decimal
    total_return_decimal = total_return_pct / 100.0

    # Calculate final/initial ratio
    ratio = 1.0 + total_return_decimal

    # Annualize
    exponent = 365.0 / num_days
    annualized_ratio = ratio**exponent

    # Convert back to percentage
    annualized_return_pct = (annualized_ratio - 1.0) * 100.0

    return annualized_return_pct


@njit(cache=True)
def calculate_avg_holding_period(holding_periods: np.ndarray) -> float:
    """
    Calculate average holding period in bars.

    Parameters
    ----------
    holding_periods : np.ndarray (int32)
        Array of holding periods (in bars)

    Returns
    -------
    float
        Average holding period

    Examples
    --------
    >>> periods = np.array([5, 10, 3, 8, 12])
    >>> avg = calculate_avg_holding_period(periods)
    >>> # 7.6 bars
    """
    if len(holding_periods) == 0:
        return 0.0

    return np.mean(holding_periods.astype(np.float64))


class MetricSanityError(Exception):
    """Raised when a metric sanity check fails.

    This indicates a critical issue with the calculated metrics, such as:
    - Sharpe ratio outside reasonable range (-5 to +5)
    - Win rate outside valid range (0 to 1)
    - Max drawdown exceeding 100%
    - Other mathematically impossible metric values
    """

    pass


def validate_metric_sanity(
    metrics: dict[str, Any],
    raise_on_critical: bool = False,
) -> list[str]:
    """
    Perform sanity checks on calculated metrics.

    Validates that metrics are within reasonable/valid ranges:
    - Sharpe ratio: -5 to +5 (warning), outside is suspicious
    - Win rate: 0 to 1 (must be valid probability)
    - Max drawdown: 0 to -100% (cannot lose more than 100%)
    - Avg win/loss ratio: >= 0 (cannot be negative)
    - Total trades: >= 0 (cannot be negative)

    Parameters
    ----------
    metrics : dict[str, Any]
        Metrics dictionary from calculate_all_metrics()
    raise_on_critical : bool
        If True, raise MetricSanityError on critical violations
        (default: False, only logs warnings)

    Returns
    -------
    list[str]
        List of sanity check violations (empty if all pass)

    Raises
    ------
    MetricSanityError
        If raise_on_critical is True and a critical violation is found

    Examples
    --------
    >>> metrics = {"sharpe_ratio": 10.0, "win_rate": 0.5, ...}
    >>> violations = validate_metric_sanity(metrics)
    >>> # Returns ["Extreme Sharpe ratio: 10.00 (expected -5 to +5)"]
    """
    violations: list[str] = []

    # Check 1: Sharpe ratio in reasonable range (-5 to +5)
    sharpe = metrics.get("sharpe_ratio", 0.0)
    if sharpe < -5.0 or sharpe > 5.0:
        msg = f"Extreme Sharpe ratio: {sharpe:.2f} (expected -5 to +5)"
        violations.append(msg)
        logger.warning(f"Metric sanity warning: {msg}")

    # Check 2: Win rate must be valid probability (0 to 1)
    win_rate = metrics.get("win_rate", 0.0)
    if win_rate < 0.0 or win_rate > 1.0:
        msg = f"Invalid win rate: {win_rate:.4f} (must be between 0 and 1)"
        violations.append(msg)
        logger.error(f"Metric sanity CRITICAL: {msg}")
        if raise_on_critical:
            raise MetricSanityError(msg)

    # Check 3: Max drawdown must be <= 0 and >= -100%
    max_dd = metrics.get("max_drawdown_pct", 0.0)
    if max_dd > 0.0:
        msg = f"Invalid max drawdown: {max_dd:.2f}% (must be <= 0)"
        violations.append(msg)
        logger.error(f"Metric sanity CRITICAL: {msg}")
        if raise_on_critical:
            raise MetricSanityError(msg)
    elif max_dd < -100.0:
        msg = f"Invalid max drawdown: {max_dd:.2f}% (cannot exceed -100%)"
        violations.append(msg)
        logger.error(f"Metric sanity CRITICAL: {msg}")
        if raise_on_critical:
            raise MetricSanityError(msg)
    elif max_dd < -50.0:
        # Warning for very large drawdowns (suspicious but possible)
        msg = f"Very large max drawdown: {max_dd:.2f}% (exceeds -50%)"
        violations.append(msg)
        logger.warning(f"Metric sanity warning: {msg}")

    # Check 4: Avg win/loss ratio must be non-negative
    win_loss_ratio = metrics.get("avg_win_loss_ratio", 0.0)
    if win_loss_ratio < 0.0:
        msg = f"Invalid avg win/loss ratio: {win_loss_ratio:.2f} (must be >= 0)"
        violations.append(msg)
        logger.error(f"Metric sanity CRITICAL: {msg}")
        if raise_on_critical:
            raise MetricSanityError(msg)

    # Check 5: Total trades must be non-negative
    total_trades = metrics.get("total_trades", 0)
    if total_trades < 0:
        msg = f"Invalid total trades: {total_trades} (must be >= 0)"
        violations.append(msg)
        logger.error(f"Metric sanity CRITICAL: {msg}")
        if raise_on_critical:
            raise MetricSanityError(msg)

    # Check 6: Winning + losing trades must equal total trades
    winning_trades = metrics.get("winning_trades", 0)
    losing_trades = metrics.get("losing_trades", 0)
    if winning_trades + losing_trades != total_trades:
        msg = (
            f"Trade count mismatch: winning ({winning_trades}) + losing ({losing_trades}) "
            f"!= total ({total_trades})"
        )
        violations.append(msg)
        logger.error(f"Metric sanity CRITICAL: {msg}")
        if raise_on_critical:
            raise MetricSanityError(msg)

    # Check 7: Final equity consistency with return
    initial_capital = metrics.get("initial_capital", 0.0)
    final_equity = metrics.get("final_equity", 0.0)
    total_return = metrics.get("total_return", 0.0)
    if initial_capital > 0:
        expected_final = initial_capital + total_return
        tolerance = 0.01  # Allow 1 cent tolerance for floating point
        if abs(final_equity - expected_final) > tolerance:
            msg = (
                f"Equity/return mismatch: final_equity ({final_equity:.2f}) != "
                f"initial ({initial_capital:.2f}) + return ({total_return:.2f})"
            )
            violations.append(msg)
            logger.error(f"Metric sanity CRITICAL: {msg}")
            if raise_on_critical:
                raise MetricSanityError(msg)

    # Check 8: Annualized return sanity (warn on extreme values)
    ann_return = metrics.get("annualized_return_pct", 0.0)
    if ann_return > 500.0 or ann_return < -100.0:
        msg = f"Extreme annualized return: {ann_return:.2f}% (unusual for real strategies)"
        violations.append(msg)
        logger.warning(f"Metric sanity warning: {msg}")

    # Check 9: Average holding period must be non-negative
    avg_holding = metrics.get("avg_holding_period", 0.0)
    if avg_holding < 0.0:
        msg = f"Invalid avg holding period: {avg_holding:.2f} (must be >= 0)"
        violations.append(msg)
        logger.error(f"Metric sanity CRITICAL: {msg}")
        if raise_on_critical:
            raise MetricSanityError(msg)

    # Log summary if there were violations
    if violations:
        logger.info(f"Metric sanity check found {len(violations)} issue(s)")
    else:
        logger.debug("Metric sanity check passed: all metrics within expected ranges")

    return violations


class PerformanceMetrics:
    """
    Calculate comprehensive performance metrics from backtest results.

    Aggregates all metrics defined in R9.3.1 and provides helper methods
    for generating reports.
    """

    @staticmethod
    def calculate_all_metrics(
        equity_curve: NDArray,
        trades_pnl: NDArray,
        trades_holding_periods: NDArray,
        initial_capital: float,
        risk_free_rate: float = 0.04,
        periods_per_year: int = 252,
        enable_sanity_checks: bool = True,
        raise_on_sanity_failure: bool = False,
    ) -> dict[str, Any]:
        """
        Calculate all performance metrics.

        Parameters
        ----------
        equity_curve : NDArray
            Equity values over time
        trades_pnl : NDArray
            P&L for each completed trade
        trades_holding_periods : NDArray
            Holding period for each trade (in bars)
        initial_capital : float
            Starting capital
        risk_free_rate : float
            Annual risk-free rate (default: 4%)
        periods_per_year : int
            Periods per year for annualization (252 for daily)
        enable_sanity_checks : bool
            If True, run sanity checks after calculating metrics (default: True)
        raise_on_sanity_failure : bool
            If True, raise MetricSanityError on critical violations (default: False)

        Returns
        -------
        Dict[str, Any]
            Dictionary with all performance metrics

        Raises
        ------
        MetricSanityError
            If raise_on_sanity_failure is True and a critical sanity check fails
        """
        # Calculate returns
        returns = calculate_returns(equity_curve)

        # Sharpe ratio
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)

        # Max drawdown
        max_dd_pct, peak_idx, trough_idx = calculate_max_drawdown(equity_curve)

        # Win rate
        win_rate = calculate_win_rate(trades_pnl)

        # Avg win/loss ratio
        win_loss_ratio = calculate_avg_win_loss_ratio(trades_pnl)

        # Total return
        final_equity = equity_curve[-1] if len(equity_curve) > 0 else initial_capital
        total_return, total_return_pct = calculate_total_return(initial_capital, final_equity)

        # Annualized return (assume 1 period per bar, num_days = len(equity_curve))
        num_days = len(equity_curve)
        annualized_return_pct = calculate_annualized_return(total_return_pct, num_days)

        # Average holding period
        avg_holding = calculate_avg_holding_period(trades_holding_periods)

        # Winning/losing trade stats
        winning_trades = trades_pnl[trades_pnl > 0]
        losing_trades = trades_pnl[trades_pnl <= 0]

        avg_win = float(np.mean(winning_trades)) if len(winning_trades) > 0 else 0.0
        avg_loss = float(np.mean(np.abs(losing_trades))) if len(losing_trades) > 0 else 0.0

        metrics = {
            # Return metrics
            "total_return": float(total_return),
            "total_return_pct": float(total_return_pct),
            "annualized_return_pct": float(annualized_return_pct),
            "final_equity": float(final_equity),
            "initial_capital": float(initial_capital),
            # Risk metrics
            "sharpe_ratio": float(sharpe),
            "max_drawdown_pct": float(max_dd_pct),
            "max_drawdown_peak_idx": int(peak_idx),
            "max_drawdown_trough_idx": int(trough_idx),
            # Trade statistics
            "total_trades": len(trades_pnl),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": float(win_rate),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "avg_win_loss_ratio": float(win_loss_ratio),
            "avg_holding_period": float(avg_holding),
            # Risk-free rate used
            "risk_free_rate": float(risk_free_rate),
        }

        # Run sanity checks if enabled
        if enable_sanity_checks:
            validate_metric_sanity(metrics, raise_on_critical=raise_on_sanity_failure)

        return metrics

    @staticmethod
    def format_metrics_report(metrics: dict[str, Any]) -> str:
        """
        Format metrics as human-readable report.

        Parameters
        ----------
        metrics : Dict[str, Any]
            Metrics dictionary from calculate_all_metrics()

        Returns
        -------
        str
            Formatted report string
        """
        report = "=" * 60 + "\n"
        report += "BACKTEST PERFORMANCE REPORT\n"
        report += "=" * 60 + "\n\n"

        report += "RETURN METRICS:\n"
        report += f"  Initial Capital:      ${metrics['initial_capital']:,.2f}\n"
        report += f"  Final Equity:         ${metrics['final_equity']:,.2f}\n"
        report += (
            f"  Total Return:         ${metrics['total_return']:,.2f} "
            f"({metrics['total_return_pct']:.2f}%)\n"
        )
        report += f"  Annualized Return:    {metrics['annualized_return_pct']:.2f}%\n\n"

        report += "RISK METRICS:\n"
        report += f"  Sharpe Ratio:         {metrics['sharpe_ratio']:.2f}\n"
        report += f"  Max Drawdown:         {metrics['max_drawdown_pct']:.2f}%\n"
        report += f"  Risk-Free Rate:       {metrics['risk_free_rate'] * 100:.1f}%\n\n"

        report += "TRADE STATISTICS:\n"
        report += f"  Total Trades:         {metrics['total_trades']}\n"
        report += f"  Winning Trades:       {metrics['winning_trades']}\n"
        report += f"  Losing Trades:        {metrics['losing_trades']}\n"
        report += f"  Win Rate:             {metrics['win_rate'] * 100:.2f}%\n"
        report += f"  Avg Win:              ${metrics['avg_win']:.2f}\n"
        report += f"  Avg Loss:             ${metrics['avg_loss']:.2f}\n"
        report += f"  Avg Win/Loss Ratio:   {metrics['avg_win_loss_ratio']:.2f}\n"
        report += f"  Avg Holding Period:   {metrics['avg_holding_period']:.1f} bars\n\n"

        report += "=" * 60 + "\n"

        return report

    @staticmethod
    def check_success_criteria(metrics: dict[str, Any]) -> dict[str, bool]:
        """
        Check if metrics meet success criteria from FLUXHERO_REQUIREMENTS.md.

        Minimum targets (R9 Success Criteria):
        - Sharpe Ratio: >0.8
        - Max Drawdown: <25%
        - Win Rate: >45%
        - Avg Win/Loss Ratio: >1.5

        Parameters
        ----------
        metrics : Dict[str, Any]
            Metrics dictionary

        Returns
        -------
        Dict[str, bool]
            Pass/fail for each criterion
        """
        return {
            "sharpe_ratio_ok": metrics["sharpe_ratio"] > 0.8,
            "max_drawdown_ok": metrics["max_drawdown_pct"] > -25.0,  # Negative value
            "win_rate_ok": metrics["win_rate"] > 0.45,
            "win_loss_ratio_ok": metrics["avg_win_loss_ratio"] > 1.5,
            "all_criteria_met": (
                metrics["sharpe_ratio"] > 0.8
                and metrics["max_drawdown_pct"] > -25.0
                and metrics["win_rate"] > 0.45
                and metrics["avg_win_loss_ratio"] > 1.5
            ),
        }
