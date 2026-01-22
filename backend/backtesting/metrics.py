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

Reference:
- FLUXHERO_REQUIREMENTS.md Feature 9.3: Metrics Calculation
- algorithmic-trading-guide.md → Key Metrics to Track
"""

from typing import Any

import numpy as np
from numba import njit
from numpy.typing import NDArray


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
        if equity_curve[i-1] > 0:
            returns[i-1] = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
        else:
            returns[i-1] = 0.0

    return returns


@njit(cache=True)
def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252
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
def calculate_total_return(
    initial_capital: float,
    final_equity: float
) -> tuple:
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
def calculate_annualized_return(
    total_return_pct: float,
    num_days: int
) -> float:
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
    annualized_ratio = ratio ** exponent

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
        periods_per_year: int = 252
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

        Returns
        -------
        Dict[str, Any]
            Dictionary with all performance metrics
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

        return {
            # Return metrics
            'total_return': float(total_return),
            'total_return_pct': float(total_return_pct),
            'annualized_return_pct': float(annualized_return_pct),
            'final_equity': float(final_equity),
            'initial_capital': float(initial_capital),

            # Risk metrics
            'sharpe_ratio': float(sharpe),
            'max_drawdown_pct': float(max_dd_pct),
            'max_drawdown_peak_idx': int(peak_idx),
            'max_drawdown_trough_idx': int(trough_idx),

            # Trade statistics
            'total_trades': len(trades_pnl),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': float(win_rate),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'avg_win_loss_ratio': float(win_loss_ratio),
            'avg_holding_period': float(avg_holding),

            # Risk-free rate used
            'risk_free_rate': float(risk_free_rate)
        }

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
        report += f"  Total Return:         ${metrics['total_return']:,.2f} ({metrics['total_return_pct']:.2f}%)\n"
        report += f"  Annualized Return:    {metrics['annualized_return_pct']:.2f}%\n\n"

        report += "RISK METRICS:\n"
        report += f"  Sharpe Ratio:         {metrics['sharpe_ratio']:.2f}\n"
        report += f"  Max Drawdown:         {metrics['max_drawdown_pct']:.2f}%\n"
        report += f"  Risk-Free Rate:       {metrics['risk_free_rate']*100:.1f}%\n\n"

        report += "TRADE STATISTICS:\n"
        report += f"  Total Trades:         {metrics['total_trades']}\n"
        report += f"  Winning Trades:       {metrics['winning_trades']}\n"
        report += f"  Losing Trades:        {metrics['losing_trades']}\n"
        report += f"  Win Rate:             {metrics['win_rate']*100:.2f}%\n"
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
            'sharpe_ratio_ok': metrics['sharpe_ratio'] > 0.8,
            'max_drawdown_ok': metrics['max_drawdown_pct'] > -25.0,  # Negative value
            'win_rate_ok': metrics['win_rate'] > 0.45,
            'win_loss_ratio_ok': metrics['avg_win_loss_ratio'] > 1.5,
            'all_criteria_met': (
                metrics['sharpe_ratio'] > 0.8 and
                metrics['max_drawdown_pct'] > -25.0 and
                metrics['win_rate'] > 0.45 and
                metrics['avg_win_loss_ratio'] > 1.5
            )
        }
