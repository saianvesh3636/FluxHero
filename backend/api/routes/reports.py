"""
API Routes for Report Generation and Enhanced Metrics

Provides endpoints for:
- HTML tearsheet generation (QuantStats) from real data
- Enhanced metrics (60+ metrics with Numba optimization)
- Report download and management
- Individual trade review

Data Sources:
- Backtest results (stored in SQLite backtest_results table)
- Live/Paper trading history (stored in live_trades/paper_trades tables)

Reference: /Users/anvesh/.claude/plans/swirling-tumbling-cloud.md
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.analytics import (
    QuantStatsAdapter,
    TearsheetGenerator,
    get_generator,
)
from backend.storage.sqlite_store import TradingMode, TradeStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/reports", tags=["reports"])

# =============================================================================
# Constants
# =============================================================================

# Standard trading days per year for annualization calculations
# This is a financial industry standard (252 trading days = ~365 - weekends - holidays)
TRADING_DAYS_PER_YEAR = 252

# Report expiration time in hours (reports are temporary files)
REPORT_EXPIRATION_HOURS = 24


# =============================================================================
# Helper to get app_state (imported at runtime to avoid circular imports)
# =============================================================================


def get_sqlite_store():
    """Get SQLite store from app_state (lazy import to avoid circular dependency)."""
    from backend.api.server import app_state

    if not app_state.sqlite_store:
        raise HTTPException(status_code=503, detail="Database not initialized")
    return app_state.sqlite_store


# =============================================================================
# Request/Response Models
# =============================================================================


class ReportRequest(BaseModel):
    """Request model for report generation."""

    backtest_run_id: str | None = Field(
        default=None, description="Backtest run ID to generate report from"
    )
    mode: str | None = Field(
        default=None, description="Trading mode: 'live' or 'paper'"
    )
    days: int | None = Field(
        default=30, description="Number of days for trading history (when using mode)"
    )
    trade_id: int | None = Field(
        default=None, description="Single trade ID to analyze"
    )
    benchmark: str = Field(default="SPY", description="Benchmark symbol for comparison")
    title: str | None = Field(default=None, description="Custom report title")


class ReportResponse(BaseModel):
    """Response model for report generation."""

    report_id: str
    download_url: str
    generated_at: str
    expires_at: str
    title: str
    source: str = Field(description="Data source: backtest, live, paper, or trade")
    symbol: str | None = None
    strategy: str | None = None
    date_range: str | None = None


class ReportListItem(BaseModel):
    """Report metadata for listing."""

    report_id: str
    filename: str
    created_at: str
    size_bytes: int
    download_url: str


class ReportListResponse(BaseModel):
    """Response for list reports endpoint."""

    reports: list[ReportListItem]
    total_count: int


class EnhancedMetricsResponse(BaseModel):
    """Response with all 60+ metrics."""

    # Tier 1 - High Priority (Numba-optimized)
    sortino_ratio: float = Field(description="Sortino ratio (downside deviation)")
    calmar_ratio: float = Field(description="Calmar ratio (CAGR/MaxDD)")
    profit_factor: float = Field(description="Gross profits / Gross losses")
    value_at_risk_95: float = Field(description="95% VaR")
    cvar_95: float = Field(description="95% CVaR / Expected Shortfall")
    alpha: float = Field(description="Jensen's alpha vs benchmark")
    beta: float = Field(description="Beta vs benchmark")
    kelly_criterion: float = Field(description="Optimal position size fraction")
    recovery_factor: float = Field(description="Total return / Max drawdown")
    ulcer_index: float = Field(description="Drawdown depth and duration")

    # Tier 2 - Useful
    max_consecutive_wins: int = Field(description="Longest winning streak")
    max_consecutive_losses: int = Field(description="Longest losing streak")
    skewness: float = Field(description="Return distribution skewness")
    kurtosis: float = Field(description="Return distribution kurtosis")
    tail_ratio: float = Field(description="Right tail / Left tail")
    information_ratio: float = Field(description="Active return / Tracking error")
    r_squared: float = Field(description="Correlation with benchmark")

    # Standard metrics (compatible with existing)
    sharpe_ratio: float = Field(description="Sharpe ratio")
    max_drawdown_pct: float = Field(description="Maximum drawdown percentage")
    win_rate: float = Field(description="Win rate (0-1)")
    avg_win_loss_ratio: float = Field(description="Average win / Average loss")
    total_return_pct: float = Field(description="Total return percentage")
    annualized_return_pct: float = Field(description="Annualized return percentage")

    # Metadata
    periods_analyzed: int = Field(description="Number of periods in analysis")
    benchmark_symbol: str = Field(description="Benchmark used for comparison")
    data_source: str = Field(description="Where data came from")


class TradeReviewResponse(BaseModel):
    """Response model for individual trade review."""

    trade_id: int
    symbol: str
    side: str  # "LONG" or "SHORT"
    status: str  # "OPEN", "CLOSED", "CANCELLED"

    # Entry details
    entry_price: float
    entry_time: str
    shares: int

    # Exit details (if closed)
    exit_price: float | None
    exit_time: str | None

    # P&L
    realized_pnl: float | None
    return_pct: float | None
    holding_period_days: float | None

    # Strategy info
    strategy: str
    regime: str | None
    signal_reason: str | None
    signal_explanation: str | None

    # Risk management
    stop_loss: float
    take_profit: float | None


# =============================================================================
# Helper Functions
# =============================================================================


def calculate_equity_from_trades(
    trades: list, initial_capital: float = 100000.0
) -> tuple[np.ndarray, list[str]]:
    """
    Calculate equity curve from completed trades.

    Returns:
        Tuple of (equity_array, timestamp_list)
    """
    if not trades:
        return np.array([initial_capital]), [datetime.now().isoformat()]

    # Sort trades by exit time (for closed trades) or entry time
    sorted_trades = sorted(
        trades,
        key=lambda t: t.exit_time if t.exit_time else t.entry_time
    )

    equity = [initial_capital]
    timestamps = [sorted_trades[0].entry_time if sorted_trades else datetime.now().isoformat()]

    current_equity = initial_capital
    for trade in sorted_trades:
        if trade.realized_pnl is not None:
            current_equity += trade.realized_pnl
            equity.append(current_equity)
            timestamps.append(trade.exit_time or trade.entry_time)

    return np.array(equity, dtype=np.float64), timestamps


def calculate_returns_from_equity(equity: np.ndarray) -> np.ndarray:
    """Calculate returns from equity curve."""
    if len(equity) < 2:
        return np.array([0.0])

    returns = np.diff(equity) / equity[:-1]
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    return returns


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/generate", response_model=ReportResponse)
async def generate_report(
    request: ReportRequest,
    background_tasks: BackgroundTasks,
) -> ReportResponse:
    """
    Generate HTML tearsheet report from real data.

    Data sources (in priority order):
    1. backtest_run_id - Fetch from stored backtest results
    2. mode + days - Fetch from live/paper trading history
    3. trade_id - Analyze single trade

    Returns:
        ReportResponse with download URL
    """
    try:
        generator = get_generator()
        store = get_sqlite_store()

        returns = None
        timestamps = None
        title = request.title
        source = "unknown"
        symbol = None
        strategy = None
        date_range = None

        # Option 1: Generate from backtest results
        if request.backtest_run_id:
            logger.info(f"Generating report from backtest: {request.backtest_run_id}")

            result = await store.get_backtest_result(request.backtest_run_id)
            if not result:
                raise HTTPException(
                    status_code=404,
                    detail=f"Backtest '{request.backtest_run_id}' not found"
                )

            # Parse stored data
            equity_curve = json.loads(result.equity_curve_json) if result.equity_curve_json else []

            if not equity_curve or len(equity_curve) < 2:
                raise HTTPException(
                    status_code=400,
                    detail="Backtest has insufficient equity data for report"
                )

            equity = np.array(equity_curve, dtype=np.float64)
            returns = calculate_returns_from_equity(equity)

            source = "backtest"
            symbol = result.symbol
            strategy = result.strategy_mode
            date_range = f"{result.start_date} to {result.end_date}"
            title = title or f"Backtest Report - {symbol} ({strategy})"

        # Option 2: Generate from trading history
        elif request.mode:
            mode = TradingMode.LIVE if request.mode.lower() == "live" else TradingMode.PAPER
            days = request.days or 30

            logger.info(f"Generating report from {mode.value} trading, last {days} days")

            # Fetch trades
            trades = await store.get_recent_trades_for_mode(mode, limit=1000)

            # Filter by date range
            cutoff = datetime.now() - timedelta(days=days)
            cutoff_str = cutoff.isoformat()
            trades = [t for t in trades if t.entry_time >= cutoff_str]

            if not trades:
                raise HTTPException(
                    status_code=400,
                    detail=f"No trades found in {mode.value} mode for the last {days} days"
                )

            # Calculate equity curve from trades
            equity, eq_timestamps = calculate_equity_from_trades(trades)
            returns = calculate_returns_from_equity(equity)
            # Align timestamps with returns array:
            # equity has N points, returns has N-1 (calculated via np.diff)
            # Each return[i] corresponds to the period ending at timestamp[i+1]
            # So we skip the first timestamp to match returns length
            timestamps = eq_timestamps[1:] if len(eq_timestamps) > len(returns) else eq_timestamps

            # Get P&L array for trade statistics
            pnls = np.array(
                [t.realized_pnl for t in trades if t.realized_pnl is not None],
                dtype=np.float64
            )

            source = mode.value
            symbols = list(set(t.symbol for t in trades))
            symbol = symbols[0] if len(symbols) == 1 else f"{len(symbols)} symbols"
            date_range = f"Last {days} days"
            title = title or f"{mode.value.title()} Trading Report - {date_range}"

        # Option 3: Fallback to sample data (for testing)
        else:
            logger.warning("No data source specified, using sample data")
            np.random.seed(42)
            n_periods = TRADING_DAYS_PER_YEAR
            returns = np.random.normal(0.0003, 0.012, n_periods)
            source = "sample"
            title = title or "Sample Report (No Real Data)"

        # Validate we have enough data
        if returns is None or len(returns) < 5:
            raise HTTPException(
                status_code=400,
                detail="Insufficient data for report generation (need at least 5 periods)"
            )

        # Generate report
        report_path = generator.generate_tearsheet(
            returns=returns,
            timestamps=timestamps,
            benchmark_symbol=request.benchmark,
            title=title,
        )

        # Schedule cleanup in background
        background_tasks.add_task(generator.cleanup_old_reports, REPORT_EXPIRATION_HOURS)

        # Build response
        report_id = report_path.stem
        generated_at = datetime.now()
        expires_at = generated_at + timedelta(hours=REPORT_EXPIRATION_HOURS)

        return ReportResponse(
            report_id=report_id,
            download_url=f"/api/reports/download/{report_id}",
            generated_at=generated_at.isoformat(),
            expires_at=expires_at.isoformat(),
            title=title,
            source=source,
            symbol=symbol,
            strategy=strategy,
            date_range=date_range,
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error generating report: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@router.get("/download/{report_id}")
async def download_report(report_id: str) -> FileResponse:
    """Download generated report."""
    generator = get_generator()
    report_path = generator.get_report_path(report_id)

    if not report_path or not report_path.exists():
        raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found")

    return FileResponse(
        path=str(report_path),
        filename=f"fluxhero_report_{report_id}.html",
        media_type="text/html",
    )


@router.get("/list", response_model=ReportListResponse)
async def list_reports() -> ReportListResponse:
    """List all available reports."""
    generator = get_generator()
    reports = generator.list_reports()

    return ReportListResponse(
        reports=[ReportListItem(**r) for r in reports],
        total_count=len(reports),
    )


@router.delete("/{report_id}")
async def delete_report(report_id: str) -> dict[str, Any]:
    """Delete a specific report."""
    generator = get_generator()

    if generator.delete_report(report_id):
        return {"status": "deleted", "report_id": report_id}
    else:
        raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found")


@router.get("/metrics", response_model=EnhancedMetricsResponse)
async def get_enhanced_metrics(
    mode: str = Query(default="paper", description="Trading mode: 'live' or 'paper'"),
    benchmark: str = Query(default="SPY", description="Benchmark symbol"),
    backtest_run_id: str | None = Query(default=None, description="Backtest run ID"),
    days: int = Query(default=30, description="Days of history to analyze"),
) -> EnhancedMetricsResponse:
    """
    Get enhanced metrics (60+ metrics) from real data.

    Data sources (in priority order):
    1. backtest_run_id - Analyze stored backtest
    2. mode + days - Analyze trading history
    """
    try:
        store = get_sqlite_store()
        data_source = "unknown"

        # Initialize variables to avoid scope issues
        equity: np.ndarray | None = None
        pnls: np.ndarray = np.array([], dtype=np.float64)

        # Option 1: From backtest
        if backtest_run_id:
            result = await store.get_backtest_result(backtest_run_id)
            if not result:
                raise HTTPException(status_code=404, detail="Backtest not found")

            equity_curve = json.loads(result.equity_curve_json) if result.equity_curve_json else []
            trades_data = json.loads(result.trades_json) if result.trades_json else []

            equity = np.array(equity_curve, dtype=np.float64)
            returns = calculate_returns_from_equity(equity)
            pnls = np.array([t.get("pnl", 0) for t in trades_data], dtype=np.float64)
            data_source = f"backtest:{backtest_run_id[:8]}"

        # Option 2: From trading history
        else:
            trading_mode = TradingMode.LIVE if mode.lower() == "live" else TradingMode.PAPER
            trades = await store.get_recent_trades_for_mode(trading_mode, limit=1000)

            # Filter by date
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            trades = [t for t in trades if t.entry_time >= cutoff]

            if not trades:
                # Return zeros if no trades
                return EnhancedMetricsResponse(
                    sortino_ratio=0.0, calmar_ratio=0.0, profit_factor=0.0,
                    value_at_risk_95=0.0, cvar_95=0.0, alpha=0.0, beta=1.0,
                    kelly_criterion=0.0, recovery_factor=0.0, ulcer_index=0.0,
                    max_consecutive_wins=0, max_consecutive_losses=0,
                    skewness=0.0, kurtosis=0.0, tail_ratio=1.0,
                    information_ratio=0.0, r_squared=0.0,
                    sharpe_ratio=0.0, max_drawdown_pct=0.0, win_rate=0.0,
                    avg_win_loss_ratio=0.0, total_return_pct=0.0,
                    annualized_return_pct=0.0,
                    periods_analyzed=0, benchmark_symbol=benchmark,
                    data_source=f"{mode}:no_trades"
                )

            equity, _ = calculate_equity_from_trades(trades)
            returns = calculate_returns_from_equity(equity)
            pnls = np.array(
                [t.realized_pnl for t in trades if t.realized_pnl is not None],
                dtype=np.float64
            )
            data_source = f"{mode}:{days}d"

        # Create adapter and calculate metrics
        adapter = QuantStatsAdapter(
            returns=returns,
            equity_curve=equity,
            pnls=pnls if len(pnls) > 0 else None,
            risk_free_rate=0.04,
        )

        tier1 = adapter.get_tier1_metrics()

        # Calculate standard metrics
        initial = equity[0] if equity is not None and len(equity) > 0 else 100000.0
        final = equity[-1] if equity is not None and len(equity) > 0 else initial
        total_return_pct = ((final - initial) / initial) * 100
        n_periods = len(returns)
        annualized = ((1 + total_return_pct / 100) ** (TRADING_DAYS_PER_YEAR / max(n_periods, 1)) - 1) * 100

        # Sharpe ratio (annualized)
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(TRADING_DAYS_PER_YEAR) if np.std(returns) > 0 else 0

        # Win rate
        wins = pnls[pnls > 0] if len(pnls) > 0 else np.array([])
        losses = pnls[pnls < 0] if len(pnls) > 0 else np.array([])
        win_rate = len(wins) / len(pnls) if len(pnls) > 0 else 0
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0
        avg_win_loss = avg_win / avg_loss if avg_loss > 0 else 0

        # Max drawdown
        max_dd = 0.0
        if equity is not None and len(equity) > 0:
            peak = equity[0]
            for eq in equity:
                if eq > peak:
                    peak = eq
                if peak > 0:
                    dd = (peak - eq) / peak
                    if dd > max_dd:
                        max_dd = dd

        return EnhancedMetricsResponse(
            sortino_ratio=tier1["sortino_ratio"],
            calmar_ratio=tier1["calmar_ratio"],
            profit_factor=tier1["profit_factor"],
            value_at_risk_95=tier1["value_at_risk_95"],
            cvar_95=tier1["cvar_95"],
            alpha=tier1["alpha"],
            beta=tier1["beta"],
            kelly_criterion=tier1["kelly_criterion"],
            recovery_factor=tier1["recovery_factor"],
            ulcer_index=tier1["ulcer_index"],
            max_consecutive_wins=tier1["max_consecutive_wins"],
            max_consecutive_losses=tier1["max_consecutive_losses"],
            skewness=tier1["skewness"],
            kurtosis=tier1["kurtosis"],
            tail_ratio=tier1["tail_ratio"],
            information_ratio=tier1["information_ratio"],
            r_squared=tier1["r_squared"],
            sharpe_ratio=sharpe,
            max_drawdown_pct=-max_dd * 100,
            win_rate=win_rate,
            avg_win_loss_ratio=avg_win_loss,
            total_return_pct=total_return_pct,
            annualized_return_pct=annualized,
            periods_analyzed=n_periods,
            benchmark_symbol=benchmark,
            data_source=data_source,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating enhanced metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics calculation failed: {str(e)}")


@router.get("/trade/{trade_id}/review", response_model=TradeReviewResponse)
async def review_trade(
    trade_id: int,
    mode: str = Query(default="paper", description="Trading mode"),
) -> TradeReviewResponse:
    """
    Get detailed review of a single trade.

    Returns complete trade information including:
    - Entry/exit details
    - P&L and return
    - Strategy and signal information
    - Risk management levels
    """
    try:
        store = get_sqlite_store()
        trading_mode = TradingMode.LIVE if mode.lower() == "live" else TradingMode.PAPER

        # Get the trade using the public API method
        trade = await store.get_trade_for_mode(trade_id, trading_mode)

        if not trade:
            raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")

        # Calculate derived fields
        side = "LONG" if trade.side == 1 else "SHORT"
        status_map = {0: "OPEN", 1: "CLOSED", 2: "CANCELLED"}
        status = status_map.get(trade.status, "UNKNOWN")

        return_pct = None
        holding_days = None

        if trade.exit_price and trade.entry_price:
            if trade.side == 1:  # LONG
                return_pct = ((trade.exit_price - trade.entry_price) / trade.entry_price) * 100
            else:  # SHORT
                return_pct = ((trade.entry_price - trade.exit_price) / trade.entry_price) * 100

        if trade.exit_time and trade.entry_time:
            try:
                entry_dt = datetime.fromisoformat(trade.entry_time.replace("Z", "+00:00"))
                exit_dt = datetime.fromisoformat(trade.exit_time.replace("Z", "+00:00"))
                holding_days = (exit_dt - entry_dt).total_seconds() / 86400
            except Exception:
                pass

        return TradeReviewResponse(
            trade_id=trade.id,
            symbol=trade.symbol,
            side=side,
            status=status,
            entry_price=trade.entry_price,
            entry_time=trade.entry_time,
            shares=trade.shares,
            exit_price=trade.exit_price,
            exit_time=trade.exit_time,
            realized_pnl=trade.realized_pnl,
            return_pct=return_pct,
            holding_period_days=holding_days,
            strategy=trade.strategy,
            regime=trade.regime if trade.regime else None,
            signal_reason=trade.signal_reason if trade.signal_reason else None,
            signal_explanation=trade.signal_explanation,
            stop_loss=trade.stop_loss,
            take_profit=trade.take_profit,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reviewing trade {trade_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Trade review failed: {str(e)}")


@router.post("/cleanup")
async def cleanup_reports(
    max_age_hours: int = Query(default=24, description="Maximum age in hours"),
) -> dict[str, Any]:
    """Cleanup old reports."""
    generator = get_generator()
    deleted = generator.cleanup_old_reports(max_age_hours)

    return {
        "status": "completed",
        "deleted_count": deleted,
        "max_age_hours": max_age_hours,
    }
