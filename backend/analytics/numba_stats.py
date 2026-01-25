"""
Numba JIT-Compiled Advanced Performance Metrics

This module provides high-performance implementations of advanced financial metrics
using Numba's @njit compilation for near-C++ speeds.

Tier 1 Metrics (Numba-optimized):
- Sortino Ratio: Downside-only risk-adjusted return
- Calmar Ratio: CAGR / Max Drawdown
- Profit Factor: Gross profits / Gross losses
- Value at Risk (VaR): Maximum expected loss at confidence level
- CVaR / Expected Shortfall: Expected loss beyond VaR
- Kelly Criterion: Optimal position sizing
- Recovery Factor: Total return / Max drawdown
- Ulcer Index: Drawdown depth and duration (RMS)
- Alpha/Beta: Jensen's alpha and beta vs benchmark
- Consecutive Wins/Losses: Streak analysis
- Skewness/Kurtosis: Distribution shape metrics
- Tail Ratio: Right/left tail comparison
- Information Ratio: Active return / tracking error
- R-Squared: Correlation with benchmark

Performance targets (10,000 data points):
- Individual metric: <5ms
- Full Tier 1 suite: <50ms

Reference: /Users/anvesh/.claude/plans/swirling-tumbling-cloud.md
"""

import numpy as np
from numba import njit


@njit(cache=True)
def calculate_sortino_ratio(
    returns: np.ndarray,
    target_return: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Sortino ratio (downside-only risk adjustment).

    Unlike Sharpe ratio which penalizes all volatility, Sortino only considers
    downside deviation (negative returns), making it more appropriate for
    asymmetric return distributions.

    Formula:
        Downside Deviation = sqrt(mean(min(R - target, 0)^2))
        Sortino = (Annualized Return - Target) / Annualized Downside Deviation

    Args:
        returns: Array of period returns (e.g., daily returns as decimals)
        target_return: Minimum acceptable return (default: 0)
        periods_per_year: Annualization factor (252 for daily, 12 for monthly)

    Returns:
        Annualized Sortino ratio (higher is better, >2 is excellent)

    Example:
        >>> returns = np.array([0.01, -0.005, 0.02, 0.015, -0.01, 0.008])
        >>> sortino = calculate_sortino_ratio(returns, target_return=0.0)
    """
    if len(returns) < 2:
        return 0.0

    # Calculate excess returns over target
    target_per_period = target_return / periods_per_year
    excess_returns = returns - target_per_period

    # Calculate downside deviation (only negative deviations)
    downside_sum = 0.0
    n = len(returns)

    for r in excess_returns:
        if r < 0:
            downside_sum += r * r

    downside_deviation = np.sqrt(downside_sum / n)

    if downside_deviation == 0:
        # No downside volatility - return high value if positive returns
        mean_return = np.mean(returns)
        if mean_return > 0:
            return 10.0  # Cap at 10 to avoid infinity
        return 0.0

    # Annualize
    mean_excess = np.mean(excess_returns) * periods_per_year
    downside_annual = downside_deviation * np.sqrt(periods_per_year)

    return mean_excess / downside_annual


@njit(cache=True)
def calculate_calmar_ratio(
    returns: np.ndarray,
    equity_curve: np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Calmar ratio (CAGR / Max Drawdown).

    Measures risk-adjusted return relative to the worst peak-to-trough decline.
    Useful for evaluating strategies where drawdown management is critical.

    Formula:
        CAGR = (Final / Initial)^(periods_per_year / n) - 1
        Max Drawdown = max(peak - trough) / peak
        Calmar = CAGR / |Max Drawdown|

    Args:
        returns: Array of period returns
        equity_curve: Array of equity values over time
        periods_per_year: Annualization factor

    Returns:
        Calmar ratio (higher is better, >3 is excellent)

    Example:
        >>> equity = np.array([100000, 105000, 103000, 110000, 108000, 115000])
        >>> returns = np.diff(equity) / equity[:-1]
        >>> calmar = calculate_calmar_ratio(returns, equity)
    """
    if len(equity_curve) < 2 or len(returns) < 1:
        return 0.0

    # Calculate CAGR
    initial = equity_curve[0]
    final = equity_curve[-1]

    if initial <= 0:
        return 0.0

    n_periods = len(returns)
    total_return = final / initial

    if total_return <= 0:
        return 0.0

    cagr = (total_return ** (periods_per_year / n_periods)) - 1.0

    # Calculate max drawdown
    peak = equity_curve[0]
    max_dd = 0.0

    for equity in equity_curve:
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd

    if max_dd == 0:
        # No drawdown - return high value if positive CAGR
        if cagr > 0:
            return 10.0
        return 0.0

    return cagr / max_dd


@njit(cache=True)
def calculate_profit_factor(pnls: np.ndarray) -> float:
    """
    Calculate profit factor (gross profits / gross losses).

    Measures how much profit is generated per unit of loss.
    A value > 1 indicates a profitable strategy.

    Formula:
        Profit Factor = Sum(Winning Trades) / |Sum(Losing Trades)|

    Args:
        pnls: Array of trade P&Ls (positive = profit, negative = loss)

    Returns:
        Profit factor (>1 is profitable, >2 is good, >3 is excellent)

    Example:
        >>> pnls = np.array([100, -50, 200, -75, 150, -30])
        >>> pf = calculate_profit_factor(pnls)
        >>> # (100 + 200 + 150) / (50 + 75 + 30) = 450 / 155 = 2.90
    """
    if len(pnls) == 0:
        return 0.0

    gross_profit = 0.0
    gross_loss = 0.0

    for pnl in pnls:
        if pnl > 0:
            gross_profit += pnl
        elif pnl < 0:
            gross_loss += abs(pnl)

    if gross_loss == 0:
        if gross_profit > 0:
            return 10.0  # Cap to avoid infinity
        return 0.0

    return gross_profit / gross_loss


@njit(cache=True)
def calculate_value_at_risk(
    returns: np.ndarray,
    confidence_level: float = 0.95,
) -> float:
    """
    Calculate historical Value at Risk (VaR).

    VaR represents the maximum expected loss over a given time period
    at a specified confidence level.

    Formula:
        VaR = Percentile(returns, 1 - confidence_level)

    Args:
        returns: Array of period returns
        confidence_level: Confidence level (default: 0.95 = 95%)

    Returns:
        VaR as a negative decimal (e.g., -0.023 = -2.3% expected max loss)

    Example:
        >>> returns = np.array([0.01, -0.02, 0.015, -0.03, 0.005, -0.01])
        >>> var_95 = calculate_value_at_risk(returns, 0.95)
    """
    if len(returns) < 2:
        return 0.0

    # Sort returns to find percentile
    sorted_returns = np.sort(returns)
    n = len(sorted_returns)

    # Calculate index for the percentile
    percentile = 1.0 - confidence_level
    idx = int(np.floor(percentile * n))

    if idx >= n:
        idx = n - 1
    if idx < 0:
        idx = 0

    return sorted_returns[idx]


@njit(cache=True)
def calculate_cvar(
    returns: np.ndarray,
    confidence_level: float = 0.95,
) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.

    CVaR represents the expected loss given that the loss exceeds VaR.
    It's a more conservative risk measure than VaR.

    Formula:
        CVaR = mean(returns where returns <= VaR)

    Args:
        returns: Array of period returns
        confidence_level: Confidence level (default: 0.95)

    Returns:
        CVaR as a negative decimal (always <= VaR)

    Example:
        >>> returns = np.array([0.01, -0.02, 0.015, -0.03, 0.005, -0.01, -0.025])
        >>> cvar_95 = calculate_cvar(returns, 0.95)
    """
    if len(returns) < 2:
        return 0.0

    var = calculate_value_at_risk(returns, confidence_level)

    # Calculate mean of returns below VaR
    tail_sum = 0.0
    tail_count = 0

    for r in returns:
        if r <= var:
            tail_sum += r
            tail_count += 1

    if tail_count == 0:
        return var

    return tail_sum / tail_count


@njit(cache=True)
def calculate_kelly_criterion(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
) -> float:
    """
    Calculate Kelly Criterion for optimal position sizing.

    The Kelly Criterion determines the optimal fraction of capital to risk
    on each trade to maximize long-term growth rate.

    Formula:
        Kelly % = W - [(1 - W) / R]
        Where:
            W = Win rate (probability of winning)
            R = Win/Loss ratio (average win / average loss)

    Args:
        win_rate: Probability of winning (0 to 1)
        avg_win: Average winning trade amount
        avg_loss: Average losing trade amount (positive value)

    Returns:
        Kelly fraction (0 to 1, represents % of capital to risk)
        Note: Many practitioners use half-Kelly for safety

    Example:
        >>> kelly = calculate_kelly_criterion(0.55, 150, 100)
        >>> # Kelly = 0.55 - (0.45 / 1.5) = 0.55 - 0.30 = 0.25 (25%)
    """
    if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0

    win_loss_ratio = avg_win / avg_loss
    kelly = win_rate - ((1.0 - win_rate) / win_loss_ratio)

    # Clamp to valid range
    if kelly < 0:
        return 0.0
    if kelly > 1:
        return 1.0

    return kelly


@njit(cache=True)
def calculate_recovery_factor(
    total_return: float,
    max_drawdown: float,
) -> float:
    """
    Calculate Recovery Factor (Total Return / Max Drawdown).

    Measures how much profit the strategy generated relative to its
    worst drawdown. Higher values indicate better risk efficiency.

    Formula:
        Recovery Factor = Total Return / |Max Drawdown|

    Args:
        total_return: Total return as decimal (e.g., 0.25 = 25%)
        max_drawdown: Max drawdown as decimal (e.g., -0.15 = -15%)

    Returns:
        Recovery factor (>3 is good, >5 is excellent)

    Example:
        >>> rf = calculate_recovery_factor(0.30, -0.10)
        >>> # 0.30 / 0.10 = 3.0
    """
    if max_drawdown >= 0:
        # No drawdown
        if total_return > 0:
            return 10.0
        return 0.0

    return total_return / abs(max_drawdown)


@njit(cache=True)
def calculate_ulcer_index(equity_curve: np.ndarray) -> float:
    """
    Calculate Ulcer Index (drawdown depth and duration).

    The Ulcer Index measures the depth and duration of drawdowns,
    giving more weight to longer and deeper drawdowns. Lower is better.

    Formula:
        Ulcer Index = sqrt(mean(Drawdown%^2))

    Args:
        equity_curve: Array of equity values over time

    Returns:
        Ulcer Index (lower is better, <5 is good)

    Example:
        >>> equity = np.array([100, 105, 102, 98, 103, 108, 105])
        >>> ui = calculate_ulcer_index(equity)
    """
    if len(equity_curve) < 2:
        return 0.0

    peak = equity_curve[0]
    sum_squared_dd = 0.0
    n = len(equity_curve)

    for equity in equity_curve:
        if equity > peak:
            peak = equity

        if peak > 0:
            dd_pct = ((peak - equity) / peak) * 100.0  # As percentage
            sum_squared_dd += dd_pct * dd_pct

    return np.sqrt(sum_squared_dd / n)


@njit(cache=True)
def calculate_alpha_beta(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
) -> tuple:
    """
    Calculate Jensen's Alpha and Beta vs benchmark.

    Alpha measures excess return over what would be expected given the
    strategy's beta exposure to the benchmark.
    Beta measures the strategy's sensitivity to benchmark movements.

    Formula:
        Beta = Cov(strategy, benchmark) / Var(benchmark)
        Alpha = mean(strategy) - Beta * mean(benchmark)

    Args:
        strategy_returns: Array of strategy returns
        benchmark_returns: Array of benchmark returns (same length)

    Returns:
        Tuple of (alpha, beta)
        - Alpha: Excess return (annualized)
        - Beta: Market sensitivity (1.0 = moves with market)

    Example:
        >>> strategy = np.array([0.01, -0.005, 0.02, 0.015, -0.01])
        >>> benchmark = np.array([0.008, -0.003, 0.015, 0.01, -0.008])
        >>> alpha, beta = calculate_alpha_beta(strategy, benchmark)
    """
    n = len(strategy_returns)
    if n < 2 or len(benchmark_returns) != n:
        return 0.0, 1.0

    # Calculate means
    mean_strategy = np.mean(strategy_returns)
    mean_benchmark = np.mean(benchmark_returns)

    # Calculate variance and covariance
    var_benchmark = 0.0
    cov = 0.0

    for i in range(n):
        diff_benchmark = benchmark_returns[i] - mean_benchmark
        diff_strategy = strategy_returns[i] - mean_strategy
        var_benchmark += diff_benchmark * diff_benchmark
        cov += diff_strategy * diff_benchmark

    var_benchmark /= n
    cov /= n

    if var_benchmark == 0:
        return mean_strategy * 252, 1.0  # Annualized alpha, default beta

    beta = cov / var_benchmark
    alpha = mean_strategy - (beta * mean_benchmark)

    # Annualize alpha (assuming daily data)
    alpha_annual = alpha * 252

    return alpha_annual, beta


@njit(cache=True)
def calculate_consecutive_wins_losses(pnls: np.ndarray) -> tuple:
    """
    Calculate maximum consecutive wins and losses.

    Useful for understanding streak patterns and psychological
    demands of a strategy.

    Args:
        pnls: Array of trade P&Ls

    Returns:
        Tuple of (max_consecutive_wins, max_consecutive_losses)

    Example:
        >>> pnls = np.array([100, 50, -30, -20, -10, 80, 60, 40])
        >>> wins, losses = calculate_consecutive_wins_losses(pnls)
        >>> # wins = 3 (last three), losses = 3 (middle three)
    """
    if len(pnls) == 0:
        return 0, 0

    max_wins = 0
    max_losses = 0
    current_wins = 0
    current_losses = 0

    for pnl in pnls:
        if pnl > 0:
            current_wins += 1
            current_losses = 0
            if current_wins > max_wins:
                max_wins = current_wins
        elif pnl < 0:
            current_losses += 1
            current_wins = 0
            if current_losses > max_losses:
                max_losses = current_losses
        else:
            # pnl == 0, reset both
            current_wins = 0
            current_losses = 0

    return max_wins, max_losses


@njit(cache=True)
def calculate_skewness(returns: np.ndarray) -> float:
    """
    Calculate skewness of return distribution.

    Skewness measures the asymmetry of the distribution.
    - Positive skew: Right tail is longer (more extreme gains)
    - Negative skew: Left tail is longer (more extreme losses)

    Formula:
        Skewness = E[(X - mean)^3] / std^3

    Args:
        returns: Array of returns

    Returns:
        Skewness coefficient
        - 0: Symmetric distribution
        - > 0: Positive skew (favorable)
        - < 0: Negative skew (unfavorable)

    Example:
        >>> returns = np.array([0.01, -0.005, 0.02, 0.015, -0.01, 0.03])
        >>> skew = calculate_skewness(returns)
    """
    n = len(returns)
    if n < 3:
        return 0.0

    mean = np.mean(returns)
    std = np.std(returns)

    if std == 0:
        return 0.0

    skew_sum = 0.0
    for r in returns:
        skew_sum += ((r - mean) / std) ** 3

    return skew_sum / n


@njit(cache=True)
def calculate_kurtosis(returns: np.ndarray) -> float:
    """
    Calculate excess kurtosis of return distribution.

    Kurtosis measures the "tailedness" of the distribution.
    Excess kurtosis compares to a normal distribution (kurtosis = 3).

    Formula:
        Excess Kurtosis = E[(X - mean)^4] / std^4 - 3

    Args:
        returns: Array of returns

    Returns:
        Excess kurtosis
        - 0: Normal distribution
        - > 0: Fat tails (leptokurtic) - more extreme events
        - < 0: Thin tails (platykurtic) - fewer extreme events

    Example:
        >>> returns = np.array([0.01, -0.005, 0.02, 0.015, -0.01, -0.08])
        >>> kurt = calculate_kurtosis(returns)
    """
    n = len(returns)
    if n < 4:
        return 0.0

    mean = np.mean(returns)
    std = np.std(returns)

    if std == 0:
        return 0.0

    kurt_sum = 0.0
    for r in returns:
        kurt_sum += ((r - mean) / std) ** 4

    # Return excess kurtosis (subtract 3 for comparison to normal)
    return (kurt_sum / n) - 3.0


@njit(cache=True)
def calculate_tail_ratio(
    returns: np.ndarray,
    percentile: float = 0.05,
) -> float:
    """
    Calculate tail ratio (right tail / left tail).

    Compares the magnitude of extreme gains to extreme losses.
    A ratio > 1 indicates larger upside potential than downside risk.

    Formula:
        Tail Ratio = |Percentile(95%)| / |Percentile(5%)|

    Args:
        returns: Array of returns
        percentile: Tail percentile (default: 0.05 = 5th/95th)

    Returns:
        Tail ratio (>1 is favorable)

    Example:
        >>> returns = np.array([0.01, -0.02, 0.03, -0.015, 0.02, -0.025, 0.04])
        >>> tail = calculate_tail_ratio(returns)
    """
    n = len(returns)
    if n < 20:  # Need enough data for meaningful tail analysis
        return 1.0

    sorted_returns = np.sort(returns)

    # Calculate indices for percentiles
    lower_idx = int(np.floor(percentile * n))
    upper_idx = int(np.floor((1.0 - percentile) * n))

    if lower_idx >= n:
        lower_idx = n - 1
    if upper_idx >= n:
        upper_idx = n - 1

    left_tail = abs(sorted_returns[lower_idx])
    right_tail = abs(sorted_returns[upper_idx])

    if left_tail == 0:
        if right_tail > 0:
            return 10.0
        return 1.0

    return right_tail / left_tail


@njit(cache=True)
def calculate_information_ratio(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Information Ratio (active return / tracking error).

    Measures the consistency of outperformance vs a benchmark.
    Higher values indicate more consistent alpha generation.

    Formula:
        Active Return = mean(strategy - benchmark)
        Tracking Error = std(strategy - benchmark)
        IR = Active Return / Tracking Error (annualized)

    Args:
        strategy_returns: Array of strategy returns
        benchmark_returns: Array of benchmark returns
        periods_per_year: Annualization factor

    Returns:
        Information ratio (>0.5 is good, >1.0 is excellent)

    Example:
        >>> strategy = np.array([0.01, -0.005, 0.02, 0.015, -0.01])
        >>> benchmark = np.array([0.008, -0.003, 0.015, 0.01, -0.008])
        >>> ir = calculate_information_ratio(strategy, benchmark)
    """
    n = len(strategy_returns)
    if n < 2 or len(benchmark_returns) != n:
        return 0.0

    # Calculate active returns
    active_returns = np.zeros(n, dtype=np.float64)
    for i in range(n):
        active_returns[i] = strategy_returns[i] - benchmark_returns[i]

    mean_active = np.mean(active_returns)
    std_active = np.std(active_returns)

    if std_active == 0:
        if mean_active > 0:
            return 10.0
        return 0.0

    # Annualize
    annual_active = mean_active * periods_per_year
    annual_tracking_error = std_active * np.sqrt(periods_per_year)

    return annual_active / annual_tracking_error


@njit(cache=True)
def calculate_r_squared(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
) -> float:
    """
    Calculate R-squared (coefficient of determination) vs benchmark.

    Measures how much of the strategy's variance is explained by
    benchmark movements. Lower R² indicates more independent alpha.

    Formula:
        R² = Correlation(strategy, benchmark)²

    Args:
        strategy_returns: Array of strategy returns
        benchmark_returns: Array of benchmark returns

    Returns:
        R-squared (0 to 1)
        - 1.0: Perfectly correlated with benchmark
        - 0.0: No correlation (pure alpha)

    Example:
        >>> strategy = np.array([0.01, -0.005, 0.02, 0.015, -0.01])
        >>> benchmark = np.array([0.008, -0.003, 0.015, 0.01, -0.008])
        >>> r2 = calculate_r_squared(strategy, benchmark)
    """
    n = len(strategy_returns)
    if n < 2 or len(benchmark_returns) != n:
        return 0.0

    mean_s = np.mean(strategy_returns)
    mean_b = np.mean(benchmark_returns)

    # Calculate correlation
    cov = 0.0
    var_s = 0.0
    var_b = 0.0

    for i in range(n):
        diff_s = strategy_returns[i] - mean_s
        diff_b = benchmark_returns[i] - mean_b
        cov += diff_s * diff_b
        var_s += diff_s * diff_s
        var_b += diff_b * diff_b

    if var_s == 0 or var_b == 0:
        return 0.0

    correlation = cov / np.sqrt(var_s * var_b)
    return correlation * correlation


# =============================================================================
# Aggregation Functions
# =============================================================================


def calculate_tier1_metrics(
    returns: np.ndarray,
    equity_curve: np.ndarray,
    pnls: np.ndarray,
    benchmark_returns: np.ndarray | None = None,
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252,
) -> dict:
    """
    Calculate all Tier 1 metrics in a single call.

    This is not JIT-compiled as it returns a dict, but it calls
    all the JIT-compiled functions for performance.

    Args:
        returns: Array of period returns
        equity_curve: Array of equity values
        pnls: Array of trade P&Ls
        benchmark_returns: Optional benchmark returns for comparison
        risk_free_rate: Annual risk-free rate
        periods_per_year: Annualization factor

    Returns:
        Dictionary with all Tier 1 metrics
    """
    # Ensure arrays are float64 for Numba
    returns = np.ascontiguousarray(returns, dtype=np.float64)
    equity_curve = np.ascontiguousarray(equity_curve, dtype=np.float64)
    pnls = np.ascontiguousarray(pnls, dtype=np.float64)

    # Basic metrics
    sortino = calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
    calmar = calculate_calmar_ratio(returns, equity_curve, periods_per_year)
    profit_factor = calculate_profit_factor(pnls)
    var_95 = calculate_value_at_risk(returns, 0.95)
    cvar_95 = calculate_cvar(returns, 0.95)

    # Calculate Kelly inputs
    win_count = np.sum(pnls > 0)
    loss_count = np.sum(pnls < 0)
    total_trades = len(pnls)

    if total_trades > 0:
        win_rate = win_count / total_trades
        avg_win = np.mean(pnls[pnls > 0]) if win_count > 0 else 0.0
        avg_loss = abs(np.mean(pnls[pnls < 0])) if loss_count > 0 else 0.0
    else:
        win_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0

    kelly = calculate_kelly_criterion(win_rate, avg_win, avg_loss)

    # Drawdown metrics
    initial_equity = equity_curve[0] if len(equity_curve) > 0 else 0.0
    final_equity = equity_curve[-1] if len(equity_curve) > 0 else 0.0
    total_return = (final_equity - initial_equity) / initial_equity if initial_equity > 0 else 0.0

    # Max drawdown
    peak = equity_curve[0] if len(equity_curve) > 0 else 0.0
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd

    recovery = calculate_recovery_factor(total_return, -max_dd if max_dd > 0 else 0.0)
    ulcer = calculate_ulcer_index(equity_curve)

    # Distribution metrics
    consecutive_wins, consecutive_losses = calculate_consecutive_wins_losses(pnls)
    skewness = calculate_skewness(returns)
    kurtosis = calculate_kurtosis(returns)
    tail_ratio = calculate_tail_ratio(returns)

    metrics = {
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "profit_factor": profit_factor,
        "value_at_risk_95": var_95,
        "cvar_95": cvar_95,
        "kelly_criterion": kelly,
        "recovery_factor": recovery,
        "ulcer_index": ulcer,
        "max_consecutive_wins": consecutive_wins,
        "max_consecutive_losses": consecutive_losses,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "tail_ratio": tail_ratio,
    }

    # Benchmark metrics (if provided)
    if benchmark_returns is not None:
        benchmark_returns = np.ascontiguousarray(benchmark_returns, dtype=np.float64)
        if len(benchmark_returns) == len(returns):
            alpha, beta = calculate_alpha_beta(returns, benchmark_returns)
            info_ratio = calculate_information_ratio(returns, benchmark_returns, periods_per_year)
            r_squared = calculate_r_squared(returns, benchmark_returns)

            metrics["alpha"] = alpha
            metrics["beta"] = beta
            metrics["information_ratio"] = info_ratio
            metrics["r_squared"] = r_squared
    else:
        metrics["alpha"] = 0.0
        metrics["beta"] = 1.0
        metrics["information_ratio"] = 0.0
        metrics["r_squared"] = 0.0

    return metrics
