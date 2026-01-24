"""
FastAPI Server for FluxHero - Backend API Layer

This module implements the REST API and WebSocket endpoints for the FluxHero
trading system frontend.

Requirements implemented:
- Phase 13: Backend API Layer
  - FastAPI setup with CORS configuration
  - REST endpoints for positions, trades, account info, system status
  - Backtest execution endpoint
  - WebSocket endpoint for live price updates

Endpoints:
- GET /api/positions: Retrieve current open positions
- GET /api/trades: Retrieve trade history with pagination
- GET /api/account: Get account equity and P&L
- GET /api/status: System health check (heartbeat)
- POST /api/backtest: Execute backtest with configuration
- WebSocket /ws/prices: Live price updates stream
"""

import asyncio
import logging
import os

# Import storage modules
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.api.auth import validate_websocket_auth
from backend.api.rate_limit import RateLimitMiddleware
from backend.backtesting.engine import (
    BacktestConfig,
    BacktestEngine,
)
from backend.backtesting.metrics import PerformanceMetrics
from backend.core.config import get_settings
from backend.storage.sqlite_store import (
    SQLiteStore,
    TradeStatus,
    TradingMode,
    ModeState,
    BacktestResult,
)

if TYPE_CHECKING:
    from backend.execution.brokers.paper_broker import PaperBroker

# ============================================================================
# Logger Configuration (using loguru)
# ============================================================================


# Intercept standard logging and route to loguru
class InterceptHandler(logging.Handler):
    def emit(self, record):
        level = (
            logger.level(record.levelname).name
            if record.levelname in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
            else record.levelno
        )
        logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())


# Setup: intercept uvicorn/fastapi logs
logging.basicConfig(handlers=[InterceptHandler()], level=logging.DEBUG, force=True)
for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
    logging.getLogger(name).handlers = [InterceptHandler()]


# ============================================================================
# Pydantic Models (Request/Response schemas)
# ============================================================================


class PositionResponse(BaseModel):
    """Response model for a position"""

    id: int | None
    symbol: str
    side: int  # 1 = LONG, -1 = SHORT
    shares: int
    entry_price: float
    current_price: float
    unrealized_pnl: float
    stop_loss: float
    take_profit: float | None = None
    entry_time: str
    updated_at: str


class TradeResponse(BaseModel):
    """Response model for a trade"""

    id: int | None
    symbol: str
    side: int
    entry_price: float
    entry_time: str
    exit_price: float | None = None
    exit_time: str | None = None
    shares: int
    stop_loss: float
    take_profit: float | None = None
    realized_pnl: float | None = None
    status: int
    strategy: str
    regime: str
    signal_reason: str


class TradeHistoryResponse(BaseModel):
    """Paginated trade history response"""

    trades: list[TradeResponse]
    total_count: int
    page: int
    page_size: int
    total_pages: int


class AccountInfoResponse(BaseModel):
    """Account information response"""

    equity: float
    cash: float
    buying_power: float
    total_pnl: float
    daily_pnl: float
    num_positions: int


class SystemStatusResponse(BaseModel):
    """System health status response"""

    status: str  # "ACTIVE", "DELAYED", "OFFLINE"
    uptime_seconds: float
    last_update: str
    websocket_connected: bool
    data_feed_active: bool
    message: str = ""


# ============================================================================
# Mode Management Models
# ============================================================================


class ModeStateResponse(BaseModel):
    """Response model for trading mode state"""

    active_mode: str  # "live" or "paper"
    last_mode_change: str | None
    paper_balance: float
    paper_realized_pnl: float
    is_live_broker_configured: bool


class SwitchModeRequest(BaseModel):
    """Request to switch trading mode"""

    mode: str = Field(..., description="Target mode: 'live' or 'paper'")
    confirm_live: bool = Field(default=False, description="Must be True to switch to live")


class PlaceOrderRequest(BaseModel):
    """Request to place an order"""

    symbol: str
    qty: int
    side: str = Field(..., description="'buy' or 'sell'")
    order_type: str = Field(default="market", description="'market' or 'limit'")
    limit_price: float | None = None


class PlaceOrderResponse(BaseModel):
    """Response for placed order"""

    order_id: str
    symbol: str
    qty: int
    side: str
    status: str
    filled_price: float | None
    mode: str


class BacktestResultSummaryResponse(BaseModel):
    """Summary response for backtest results list"""

    id: int | None
    run_id: str
    symbol: str
    strategy_mode: str
    start_date: str
    end_date: str
    total_return_pct: float | None
    sharpe_ratio: float | None
    max_drawdown_pct: float | None
    win_rate: float | None
    num_trades: int | None
    created_at: str


class BacktestResultDetailResponse(BaseModel):
    """Detailed response for a single backtest result"""

    id: int | None
    run_id: str
    symbol: str
    strategy_mode: str
    start_date: str
    end_date: str
    initial_capital: float
    final_equity: float
    total_return_pct: float | None
    sharpe_ratio: float | None
    max_drawdown_pct: float | None
    win_rate: float | None
    num_trades: int | None
    equity_curve_json: str | None
    trades_json: str | None
    config_json: str | None
    created_at: str


class BacktestRequest(BaseModel):
    """Backtest configuration request"""

    symbol: str = Field(..., description="Symbol to backtest (e.g., SPY)")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    initial_capital: float = Field(default=10000.0, description="Starting capital")
    commission_per_share: float = Field(default=0.005, description="Commission per share")
    slippage_pct: float = Field(default=0.0001, description="Slippage percentage (0.01%)")
    # Strategy parameters (optional, use defaults if not provided)
    strategy_mode: str = Field(default="DUAL", description="TREND, MEAN_REVERSION, or DUAL")


class BacktestResultResponse(BaseModel):
    """Backtest results response"""

    symbol: str
    start_date: str
    end_date: str
    initial_capital: float
    final_equity: float
    total_return: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    win_rate: float
    num_trades: int
    avg_win_loss_ratio: float
    success_criteria_met: bool
    equity_curve: list[float]
    timestamps: list[str]


class PriceUpdate(BaseModel):
    """WebSocket price update message"""

    symbol: str
    price: float
    timestamp: str
    volume: int | None = None


class SymbolValidationRequest(BaseModel):
    """Request to validate a stock symbol"""

    symbol: str = Field(..., description="Stock ticker symbol to validate (e.g., AAPL)")


class SymbolValidationResponse(BaseModel):
    """Response from symbol validation"""

    symbol: str
    name: str
    exchange: str | None = None
    currency: str | None = None
    type: str | None = None
    is_valid: bool


class SymbolSearchResponse(BaseModel):
    """Response from symbol search"""

    query: str
    results: list[SymbolValidationResponse]


class WalkForwardRequest(BaseModel):
    """Walk-forward backtest configuration request.

    Walk-forward testing divides historical data into train/test windows
    to validate strategy robustness out-of-sample.

    Reference: FLUXHERO_REQUIREMENTS.md R9.4
    """

    symbol: str = Field(..., description="Symbol to backtest (e.g., SPY)")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    initial_capital: float = Field(default=10000.0, description="Starting capital")
    commission_per_share: float = Field(default=0.005, description="Commission per share")
    slippage_pct: float = Field(default=0.0001, description="Slippage percentage (0.01%)")
    train_bars: int = Field(default=63, description="Training period bars (~3 months)")
    test_bars: int = Field(default=21, description="Test period bars (~1 month)")
    strategy_mode: str = Field(default="DUAL", description="TREND, MEAN_REVERSION, or DUAL")
    pass_threshold: float = Field(
        default=0.6,
        description="Pass rate threshold (strategy passes if >threshold profitable windows)",
    )


class WalkForwardWindowMetrics(BaseModel):
    """Metrics for a single walk-forward test window."""

    window_id: int
    train_start_date: str | None = None
    train_end_date: str | None = None
    test_start_date: str | None = None
    test_end_date: str | None = None
    initial_equity: float
    final_equity: float
    return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    num_trades: int
    is_profitable: bool


class WalkForwardResponse(BaseModel):
    """Walk-forward backtest results response.

    Contains aggregate metrics and per-window details to evaluate
    strategy robustness across multiple out-of-sample periods.

    Reference: FLUXHERO_REQUIREMENTS.md R9.4
    """

    symbol: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return_pct: float

    # Walk-forward specific metrics
    total_windows: int
    profitable_windows: int
    pass_rate: float
    passes_walk_forward_test: bool
    pass_threshold: float

    # Aggregate metrics across all windows
    aggregate_sharpe: float
    aggregate_max_drawdown_pct: float
    aggregate_win_rate: float
    total_trades: int

    # Per-window results
    window_results: list[WalkForwardWindowMetrics]

    # Combined equity curve from all test periods
    combined_equity_curve: list[float]
    timestamps: list[str]

    # Configuration used
    train_bars: int
    test_bars: int


# ============================================================================
# Trade Analytics Response Models (Phase G)
# ============================================================================


class CandleData(BaseModel):
    """OHLCV candle data for charts"""

    time: int  # Unix timestamp in seconds
    open: float
    high: float
    low: float
    close: float
    volume: int = 0


class IndicatorData(BaseModel):
    """Indicator values at each candle"""

    time: int
    kama: float | None = None
    atr_upper: float | None = None
    atr_lower: float | None = None


class TradeChartDataResponse(BaseModel):
    """Response for trade chart data with candles and indicators"""

    trade: TradeResponse
    candles: list[CandleData]
    indicators: list[IndicatorData]
    entry_index: int
    exit_index: int | None


class DailyTradeBreakdown(BaseModel):
    """Trades grouped by date with aggregated metrics"""

    date: str  # YYYY-MM-DD
    trades: list[TradeResponse]
    trade_count: int
    realized_pnl: float
    win_count: int
    loss_count: int
    daily_return_pct: float


class TotalsSummary(BaseModel):
    """Summary totals across all trades"""

    closed_count: int
    open_count: int
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    total_return_pct: float
    win_rate: float


class DailySummaryResponse(BaseModel):
    """Response for daily trade summary with grouping"""

    daily_groups: list[DailyTradeBreakdown]
    totals: TotalsSummary
    open_positions: list[PositionResponse]


class EquityCurvePoint(BaseModel):
    """Single point on equity curve"""

    date: str  # YYYY-MM-DD
    equity: float
    benchmark_equity: float
    daily_pnl: float
    cumulative_pnl: float
    cumulative_return_pct: float
    benchmark_return_pct: float


class RiskMetrics(BaseModel):
    """Risk-adjusted performance metrics"""

    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float


class DailyBreakdown(BaseModel):
    """Daily P&L breakdown for analysis"""

    date: str
    pnl: float
    return_pct: float
    trade_count: int
    cumulative_pnl: float


class LiveAnalysisResponse(BaseModel):
    """Response for live trading analysis dashboard"""

    equity_curve: list[EquityCurvePoint]
    risk_metrics: RiskMetrics
    daily_breakdown: list[DailyBreakdown]
    initial_capital: float
    current_equity: float
    benchmark_symbol: str
    trading_days: int


# ============================================================================
# Global State Management
# ============================================================================


class AppState:
    """Global application state"""

    def __init__(self):
        self.sqlite_store: SQLiteStore | None = None
        self.websocket_clients: list[WebSocket] = []
        self.start_time: datetime = datetime.now()
        self.last_update: datetime = datetime.now()
        self.data_feed_active: bool = False
        # Metrics tracking
        self.request_latencies: list[float] = []
        self.request_count: int = 0
        self.request_count_by_path: dict[str, int] = {}
        # Test data cache (for development) - maps symbol to list of candles
        self.test_data: dict[str, list[dict]] = {}

    def update_timestamp(self):
        """Update last activity timestamp"""
        self.last_update = datetime.now()

    def record_request_latency(self, path: str, latency_ms: float):
        """Record request latency for metrics"""
        self.request_latencies.append(latency_ms)
        self.request_count += 1
        self.request_count_by_path[path] = self.request_count_by_path.get(path, 0) + 1
        # Keep only last 1000 latencies to prevent memory growth
        if len(self.request_latencies) > 1000:
            self.request_latencies = self.request_latencies[-1000:]


# Global state instance
app_state = AppState()


# ============================================================================
# Lifespan Context Manager (Startup/Shutdown)
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.

    Startup:
    - Initialize SQLite store
    - Initialize data connections

    Shutdown:
    - Close SQLite store
    - Close WebSocket connections
    """
    # Startup
    logger.info("Starting FluxHero API server")

    # Initialize SQLite store
    db_path = Path(__file__).parent.parent.parent.parent / "data" / "system.db"
    app_state.sqlite_store = SQLiteStore(db_path=str(db_path))
    await app_state.sqlite_store.initialize()
    logger.info("SQLite store initialized", extra={"db_path": str(db_path)})

    # Load test data (only if not in production)
    env = os.getenv("ENV", "development")
    if env != "production":
        # Load data for multiple symbols
        symbols = ["SPY", "AAPL", "MSFT"]
        test_data_dir = Path(__file__).parent.parent / "test_data"

        for symbol in symbols:
            try:
                csv_path = test_data_dir / f"{symbol.lower()}_daily.csv"
                if csv_path.exists():
                    # Read CSV - SPY has a different format with extra header row
                    if symbol == "SPY":
                        df = pd.read_csv(csv_path, skiprows=[1])
                        df = df.rename(columns={"Price": "Date"})
                    else:
                        # AAPL and MSFT have standard format
                        df = pd.read_csv(csv_path)
                        df = df.rename(columns={"Price": "Date"})

                    # Common column renaming
                    df = df.rename(
                        columns={
                            "Close": "close",
                            "High": "high",
                            "Low": "low",
                            "Open": "open",
                            "Volume": "volume",
                        }
                    )

                    # Convert to list of dicts
                    symbol_data = []
                    for _, row in df.iterrows():
                        try:
                            symbol_data.append(
                                {
                                    "timestamp": str(row["Date"]),
                                    "open": float(row["open"]),
                                    "high": float(row["high"]),
                                    "low": float(row["low"]),
                                    "close": float(row["close"]),
                                    "volume": int(row["volume"]),
                                }
                            )
                        except (ValueError, KeyError):
                            continue  # Skip invalid rows

                    app_state.test_data[symbol] = symbol_data
                    logger.info(f"Loaded {len(symbol_data)} rows of {symbol} test data")
                else:
                    logger.warning(f"{symbol} test data file not found: {csv_path}")
            except Exception as e:
                logger.error(f"Failed to load {symbol} test data: {e}")

    # Mark data feed as inactive (will be activated when WebSocket connects)
    app_state.data_feed_active = False
    app_state.start_time = datetime.now()

    logger.info("FluxHero API server ready")

    yield

    # Shutdown
    logger.info("Shutting down FluxHero API server")

    # Close SQLite store
    if app_state.sqlite_store:
        await app_state.sqlite_store.close()
        logger.info("SQLite store closed")

    # Close all WebSocket connections
    for client in app_state.websocket_clients:
        await client.close()
    logger.info(
        "WebSocket connections closed",
        extra={"client_count": len(app_state.websocket_clients)},
    )

    logger.info("FluxHero API server stopped")


# ============================================================================
# FastAPI App Initialization
# ============================================================================

settings = get_settings()

app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# Rate Limiting Configuration
# Limit: 100 requests per 60 seconds per IP
# Exclude health check and root endpoints from rate limiting
app.add_middleware(
    RateLimitMiddleware,
    max_requests=100,
    window_seconds=60,
    exclude_paths=["/health", "/", "/docs", "/openapi.json", "/redoc"],
)


# ============================================================================
# Request/Response Logging Middleware
# ============================================================================

# Sensitive fields to mask in request body logging
SENSITIVE_FIELDS = {"password", "token", "api_key", "apikey", "secret", "credential", "auth"}

# Maximum length for logged request bodies
MAX_BODY_LOG_LENGTH = 500


def _mask_sensitive_data(data: dict | list | str) -> dict | list | str:
    """
    Recursively mask sensitive fields in request data.

    Sensitive fields (password, token, api_key, etc.) are replaced with '[REDACTED]'.

    Args:
        data: Request body data (dict, list, or string)

    Returns:
        Data with sensitive fields masked
    """
    if isinstance(data, dict):
        masked = {}
        for key, value in data.items():
            if key.lower() in SENSITIVE_FIELDS:
                masked[key] = "[REDACTED]"
            elif isinstance(value, (dict, list)):
                masked[key] = _mask_sensitive_data(value)
            else:
                masked[key] = value
        return masked
    elif isinstance(data, list):
        return [_mask_sensitive_data(item) for item in data]
    return data


def _truncate_body(body: str) -> str:
    """
    Truncate request body if it exceeds MAX_BODY_LOG_LENGTH.

    Args:
        body: Request body string

    Returns:
        Truncated body with indicator if truncated
    """
    if len(body) > MAX_BODY_LOG_LENGTH:
        return body[:MAX_BODY_LOG_LENGTH] + f"... [truncated, total {len(body)} chars]"
    return body


def _should_log_request_bodies() -> bool:
    """
    Check if request body logging is enabled.

    Request body logging is only enabled when:
    1. LOG_REQUEST_BODIES env var is set to 'true' or '1'
    2. ENV is not 'production'

    Returns:
        True if request body logging is enabled
    """
    env = os.getenv("ENV", "development")
    if env == "production":
        return False
    log_bodies = os.getenv("LOG_REQUEST_BODIES", "false").lower()
    return log_bodies in ("true", "1", "yes")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware to log all HTTP requests and responses.

    Logs:
    - Request method, path, client IP
    - Response status code, processing time
    - Request body (optional, development only, with sensitive field masking)
    """
    # Generate request ID for tracking
    request_id = f"{int(time.time() * 1000)}-{id(request)}"

    # Prepare extra logging data
    log_extra = {
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "query_params": str(request.query_params),
        "client_ip": request.client.host if request.client else "unknown",
    }

    # Optionally log request body (development only)
    if _should_log_request_bodies() and request.method in ("POST", "PUT", "PATCH"):
        try:
            # Read body - need to cache it since body can only be read once
            body_bytes = await request.body()
            if body_bytes:
                import json

                try:
                    body_data = json.loads(body_bytes)
                    masked_body = _mask_sensitive_data(body_data)
                    body_str = json.dumps(masked_body)
                except json.JSONDecodeError:
                    # Not JSON, log as string
                    body_str = body_bytes.decode("utf-8", errors="replace")

                log_extra["request_body"] = _truncate_body(body_str)

            # Create a new request with the cached body for downstream processing
            # This is necessary because the body stream can only be read once
            async def receive():
                return {"type": "http.request", "body": body_bytes}

            request = Request(request.scope, receive)
        except Exception as e:
            logger.debug(f"Could not read request body for logging: {e}")

    # Log incoming request
    start_time = time.time()
    logger.info("Incoming request", extra=log_extra)

    # Process request
    try:
        response: Response = await call_next(request)
        process_time = time.time() - start_time

        # Log response
        logger.info(
            "Request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "process_time_ms": round(process_time * 1000, 2),
            },
        )

        # Record metrics
        app_state.record_request_latency(request.url.path, round(process_time * 1000, 2))

        # Add request ID and process time to response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))

        return response

    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            "Request failed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "process_time_ms": round(process_time * 1000, 2),
                "error": str(e),
            },
            exc_info=True,
        )
        raise


# ============================================================================
# REST API Endpoints
# ============================================================================


@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "name": "FluxHero API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "positions": "/api/positions",
            "trades": "/api/trades",
            "account": "/api/account",
            "status": "/api/status",
            "backtest": "/api/backtest",
            "websocket": "/ws/prices",
        },
    }


@app.get("/api/positions", response_model=list[PositionResponse])
async def get_positions():
    """
    Get all current open positions.

    Returns:
        List[PositionResponse]: List of open positions with unrealized P&L
    """
    if not app_state.sqlite_store:
        raise HTTPException(status_code=503, detail="Database not initialized")

    # Fetch open positions from database
    positions = await app_state.sqlite_store.get_open_positions()

    # Convert to response models
    response = []
    for pos in positions:
        response.append(
            PositionResponse(
                id=pos.id,
                symbol=pos.symbol,
                side=pos.side,
                shares=pos.shares,
                entry_price=pos.entry_price,
                current_price=pos.current_price,
                unrealized_pnl=pos.unrealized_pnl,
                stop_loss=pos.stop_loss,
                take_profit=pos.take_profit,
                entry_time=pos.entry_time,
                updated_at=pos.updated_at,
            )
        )

    app_state.update_timestamp()
    return response


@app.get("/api/trades", response_model=TradeHistoryResponse)
async def get_trades(
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(default=20, ge=1, le=100, description="Trades per page"),
    status: str | None = Query(
        default=None, description="Filter by status (OPEN, CLOSED, CANCELLED)"
    ),
):
    """
    Get trade history with pagination.

    Args:
        page: Page number (1-indexed)
        page_size: Number of trades per page (max 100)
        status: Optional filter by trade status

    Returns:
        TradeHistoryResponse: Paginated trade history
    """
    if not app_state.sqlite_store:
        raise HTTPException(status_code=503, detail="Database not initialized")

    # Fetch recent trades (limit to 1000 for pagination)
    all_trades = await app_state.sqlite_store.get_recent_trades(limit=1000)

    # Apply status filter if provided
    if status:
        status_map = {
            "OPEN": TradeStatus.OPEN,
            "CLOSED": TradeStatus.CLOSED,
            "CANCELLED": TradeStatus.CANCELLED,
        }
        if status.upper() not in status_map:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

        status_value = status_map[status.upper()]
        all_trades = [t for t in all_trades if t.status == status_value]

    # Pagination
    total_count = len(all_trades)
    total_pages = (total_count + page_size - 1) // page_size  # Ceiling division
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_trades = all_trades[start_idx:end_idx]

    # Convert to response models
    trades_response = []
    for trade in page_trades:
        trades_response.append(
            TradeResponse(
                id=trade.id,
                symbol=trade.symbol,
                side=trade.side,
                entry_price=trade.entry_price,
                entry_time=trade.entry_time,
                exit_price=trade.exit_price,
                exit_time=trade.exit_time,
                shares=trade.shares,
                stop_loss=trade.stop_loss,
                take_profit=trade.take_profit,
                realized_pnl=trade.realized_pnl,
                status=trade.status,
                strategy=trade.strategy,
                regime=trade.regime,
                signal_reason=trade.signal_reason,
            )
        )

    app_state.update_timestamp()

    return TradeHistoryResponse(
        trades=trades_response,
        total_count=total_count,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@app.get("/api/account", response_model=AccountInfoResponse)
async def get_account_info():
    """
    Get account information (equity, P&L, positions).

    Returns:
        AccountInfoResponse: Account equity and P&L data
    """
    if not app_state.sqlite_store:
        raise HTTPException(status_code=503, detail="Database not initialized")

    # Get current positions
    positions = await app_state.sqlite_store.get_open_positions()

    # Calculate total unrealized P&L
    total_unrealized_pnl = sum(pos.unrealized_pnl for pos in positions)

    # Get closed trades to calculate realized P&L
    recent_trades = await app_state.sqlite_store.get_recent_trades(limit=1000)
    closed_trades = [
        t for t in recent_trades if t.status == TradeStatus.CLOSED and t.realized_pnl is not None
    ]
    total_realized_pnl = sum(t.realized_pnl for t in closed_trades)

    # Calculate daily P&L (trades closed today)
    today = datetime.now().date()
    daily_pnl = 0.0
    for trade in closed_trades:
        if trade.exit_time:
            exit_date = datetime.fromisoformat(trade.exit_time).date()
            if exit_date == today:
                daily_pnl += trade.realized_pnl

    # Get initial capital from settings (default: $10,000)
    initial_capital_setting = await app_state.sqlite_store.get_setting(
        "initial_capital", default="10000.0"
    )
    initial_capital = float(initial_capital_setting)

    # Calculate equity: initial capital + realized P&L + unrealized P&L
    equity = initial_capital + total_realized_pnl + total_unrealized_pnl

    # Cash = equity - value of open positions
    position_value = sum(pos.shares * pos.current_price for pos in positions)
    cash = equity - position_value

    # Buying power (simplified: 2x cash for margin accounts, 1x for cash accounts)
    buying_power = cash * 2.0  # Assume margin account

    app_state.update_timestamp()

    return AccountInfoResponse(
        equity=equity,
        cash=cash,
        buying_power=buying_power,
        total_pnl=total_realized_pnl + total_unrealized_pnl,
        daily_pnl=daily_pnl,
        num_positions=len(positions),
    )


@app.get("/api/status", response_model=SystemStatusResponse)
async def get_system_status():
    """
    Get system health status (heartbeat).

    Returns:
        SystemStatusResponse: System health and connectivity status
    """
    # Calculate uptime
    uptime_seconds = (datetime.now() - app_state.start_time).total_seconds()

    # Determine status based on last update time
    time_since_update = (datetime.now() - app_state.last_update).total_seconds()

    if time_since_update < 60:
        status = "ACTIVE"
        message = "System operating normally"
    elif time_since_update < 300:  # 5 minutes
        status = "DELAYED"
        message = f"No updates for {int(time_since_update)} seconds"
    else:
        status = "OFFLINE"
        message = f"System inactive for {int(time_since_update / 60)} minutes"

    # Check WebSocket clients
    websocket_connected = len(app_state.websocket_clients) > 0

    return SystemStatusResponse(
        status=status,
        uptime_seconds=uptime_seconds,
        last_update=app_state.last_update.isoformat(),
        websocket_connected=websocket_connected,
        data_feed_active=app_state.data_feed_active,
        message=message,
    )


@app.post("/api/backtest", response_model=BacktestResultResponse)
async def run_backtest(config: BacktestRequest):
    """
    Execute a backtest with the provided configuration.

    Fetches real market data from Yahoo Finance and runs the dual-mode
    strategy (trend-following + mean-reversion with automatic regime detection).

    Args:
        config: Backtest configuration (symbol, dates, capital, strategy params)

    Returns:
        BacktestResultResponse: Backtest performance metrics and equity curve
    """
    # Parse and validate dates
    try:
        start_dt = datetime.strptime(config.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(config.end_date, "%Y-%m-%d")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")

    num_days = (end_dt - start_dt).days
    if num_days <= 0:
        raise HTTPException(status_code=400, detail="End date must be after start date")

    # Fetch real data from data provider (Yahoo Finance by default)
    # No synthetic fallback - return proper errors for invalid symbols
    from backend.data.provider import (
        DataProviderError,
        DateRangeError,
        InsufficientDataError,
        SymbolNotFoundError,
        get_provider,
    )

    symbol = config.symbol.upper().strip()
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol cannot be empty")

    try:
        import asyncio

        provider = get_provider()

        # Run async provider method
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in an async context, use the provider directly
            data = await provider.fetch_historical_data(
                symbol=symbol,
                start_date=config.start_date,
                end_date=config.end_date,
                interval="1d",
            )
        else:
            # Fallback for sync context
            data = asyncio.run(
                provider.fetch_historical_data(
                    symbol=symbol,
                    start_date=config.start_date,
                    end_date=config.end_date,
                    interval="1d",
                )
            )

        bars = data.bars
        timestamps_float = data.timestamps
        timestamps_list = data.dates
        logger.info(f"Fetched {len(bars)} bars from {data.provider} for {symbol}")

    except SymbolNotFoundError as e:
        logger.warning(f"Symbol not found: {symbol} - {e}")
        raise HTTPException(
            status_code=404,
            detail=f"Symbol '{symbol}' not found. Please verify the ticker symbol is correct. "
            "Examples: AAPL (Apple), MSFT (Microsoft), SPY (S&P 500 ETF)",
        )

    except DateRangeError as e:
        logger.warning(f"Date range error for {symbol}: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except InsufficientDataError as e:
        logger.warning(f"Insufficient data for {symbol}: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient data for {symbol}: found {e.bars_found} trading days, "
            f"need at least {e.bars_required}. Try a longer date range.",
        )

    except DataProviderError as e:
        logger.error(f"Data provider error for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching data for {symbol}: {e}",
        )

    # Validate minimum data points for indicators
    min_bars_required = 70  # Need at least 60 for indicator warmup + some trading
    if len(bars) < min_bars_required:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient data: got {len(bars)} bars, need at least {min_bars_required}. "
            "Try a longer date range.",
        )

    # Create BacktestConfig
    bt_config = BacktestConfig(
        initial_capital=config.initial_capital,
        commission_per_share=config.commission_per_share,
        slippage_pct=config.slippage_pct,
    )

    # Initialize backtest engine
    engine = BacktestEngine(config=bt_config)

    # Initialize dual-mode strategy
    from backend.strategy.backtest_strategy import DualModeBacktestStrategy

    strategy = DualModeBacktestStrategy(
        bars=bars,
        initial_capital=config.initial_capital,
        strategy_mode=config.strategy_mode,
    )

    # Run backtest with dual-mode strategy
    state = engine.run(
        bars=bars,
        strategy_func=strategy.get_orders,
        symbol=config.symbol,
        timestamps=timestamps_float,
    )

    # Extract equity curve and trades from BacktestState
    equity_curve = np.array(state.equity_curve, dtype=np.float64)
    trades_pnl = np.array([t.pnl for t in state.trades], dtype=np.float64)
    trades_holding_periods = np.array([t.holding_bars for t in state.trades], dtype=np.int32)

    # Calculate performance metrics with correct signature
    metrics = PerformanceMetrics.calculate_all_metrics(
        equity_curve=equity_curve,
        trades_pnl=trades_pnl,
        trades_holding_periods=trades_holding_periods,
        initial_capital=config.initial_capital,
    )

    # Check success criteria
    criteria = PerformanceMetrics.check_success_criteria(metrics)
    success_criteria_met = criteria["all_criteria_met"]

    # Calculate max drawdown in dollars from the peak/trough indices
    # The metrics module returns max_drawdown_pct (percentage) but the API response
    # also needs max_drawdown (dollar value)
    peak_idx = metrics["max_drawdown_peak_idx"]
    trough_idx = metrics["max_drawdown_trough_idx"]
    if len(equity_curve) > 0 and peak_idx < len(equity_curve) and trough_idx < len(equity_curve):
        max_drawdown_dollars = equity_curve[peak_idx] - equity_curve[trough_idx]
    else:
        max_drawdown_dollars = 0.0

    app_state.update_timestamp()

    return BacktestResultResponse(
        symbol=config.symbol,
        start_date=config.start_date,
        end_date=config.end_date,
        initial_capital=config.initial_capital,
        final_equity=metrics["final_equity"],
        total_return=metrics["total_return"],
        total_return_pct=metrics["total_return_pct"],
        sharpe_ratio=metrics["sharpe_ratio"],
        max_drawdown=max_drawdown_dollars,
        max_drawdown_pct=abs(
            metrics["max_drawdown_pct"]
        ),  # Convert negative to positive for display
        win_rate=metrics["win_rate"],
        num_trades=metrics["total_trades"],
        avg_win_loss_ratio=metrics["avg_win_loss_ratio"],
        success_criteria_met=success_criteria_met,
        equity_curve=equity_curve.tolist(),
        timestamps=timestamps_list,
    )


@app.post("/api/backtest/walk-forward", response_model=WalkForwardResponse)
async def run_walk_forward_backtest(config: WalkForwardRequest):
    """
    Execute a walk-forward backtest with the provided configuration.

    Walk-forward testing divides historical data into consecutive train/test
    windows to validate strategy performance out-of-sample. The strategy is
    optionally re-optimized on each training window before being tested.

    Default configuration:
    - 3-month training period (63 trading days)
    - 1-month testing period (21 trading days)
    - Strategy passes if >60% of test windows are profitable

    Args:
        config: Walk-forward configuration (symbol, dates, capital, window sizes)

    Returns:
        WalkForwardResponse: Per-window metrics and aggregate results

    Reference: FLUXHERO_REQUIREMENTS.md R9.4
    """
    from backend.backtesting.walk_forward import InsufficientDataError as WFInsufficientDataError
    from backend.backtesting.walk_forward import aggregate_walk_forward_results
    from backend.backtesting.walk_forward import run_walk_forward_backtest as wf_run_backtest
    from backend.data.provider import (
        DataProviderError,
        DateRangeError,
        InsufficientDataError,
        SymbolNotFoundError,
        get_provider,
    )
    from backend.strategy.backtest_strategy import DualModeBacktestStrategy

    # Parse and validate dates
    try:
        start_dt = datetime.strptime(config.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(config.end_date, "%Y-%m-%d")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")

    num_days = (end_dt - start_dt).days
    if num_days <= 0:
        raise HTTPException(status_code=400, detail="End date must be after start date")

    # Validate window parameters
    if config.train_bars <= 0:
        raise HTTPException(status_code=400, detail="train_bars must be positive")
    if config.test_bars <= 0:
        raise HTTPException(status_code=400, detail="test_bars must be positive")
    if not 0.0 <= config.pass_threshold <= 1.0:
        raise HTTPException(status_code=400, detail="pass_threshold must be between 0 and 1")

    symbol = config.symbol.upper().strip()
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol cannot be empty")

    # Fetch real data from data provider
    try:
        provider = get_provider()
        data = await provider.fetch_historical_data(
            symbol=symbol,
            start_date=config.start_date,
            end_date=config.end_date,
            interval="1d",
        )
        bars = data.bars
        timestamps_float = data.timestamps
        timestamps_list = data.dates
        logger.info(f"Fetched {len(bars)} bars from {data.provider} for {symbol}")

    except SymbolNotFoundError as e:
        logger.warning(f"Symbol not found: {symbol} - {e}")
        raise HTTPException(
            status_code=404,
            detail=f"Symbol '{symbol}' not found. Please verify the ticker symbol is correct.",
        )

    except DateRangeError as e:
        logger.warning(f"Date range error for {symbol}: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except InsufficientDataError as e:
        logger.warning(f"Insufficient data for {symbol}: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient data for {symbol}: found {e.bars_found} trading days, "
            f"need at least {e.bars_required}. Try a longer date range.",
        )

    except DataProviderError as e:
        logger.error(f"Data provider error for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching data for {symbol}: {e}",
        )

    # Check minimum data for walk-forward testing
    min_bars_required = config.train_bars + (config.test_bars // 2)
    if len(bars) < min_bars_required:
        min_test = config.test_bars // 2
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient data for walk-forward: got {len(bars)} bars, "
            f"need at least {min_bars_required} (train={config.train_bars} + min_test={min_test}). "
            "Try a longer date range.",
        )

    # Create BacktestConfig
    bt_config = BacktestConfig(
        initial_capital=config.initial_capital,
        commission_per_share=config.commission_per_share,
        slippage_pct=config.slippage_pct,
    )

    # Create strategy factory for walk-forward
    def strategy_factory(
        window_bars: np.ndarray,
        capital: float,
        params: dict,
    ):
        """Factory function to create strategy instances for each window."""
        strategy = DualModeBacktestStrategy(
            bars=window_bars,
            initial_capital=capital,
            strategy_mode=params.get("strategy_mode", config.strategy_mode),
        )
        return strategy.get_orders

    # Run walk-forward backtest
    try:
        wf_result = wf_run_backtest(
            bars=bars,
            strategy_factory=strategy_factory,
            config=bt_config,
            train_bars=config.train_bars,
            test_bars=config.test_bars,
            timestamps=timestamps_float,
            symbol=symbol,
            initial_params={"strategy_mode": config.strategy_mode},
        )
    except WFInsufficientDataError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient data for walk-forward testing: {e}",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Aggregate results
    aggregate = aggregate_walk_forward_results(wf_result, pass_threshold=config.pass_threshold)

    # Convert window results to response format
    window_metrics: list[WalkForwardWindowMetrics] = []
    for wr in wf_result.window_results:
        # Calculate return percentage for this window
        if wr.initial_equity > 0:
            return_pct = ((wr.final_equity - wr.initial_equity) / wr.initial_equity) * 100
        else:
            return_pct = 0.0

        window_metrics.append(
            WalkForwardWindowMetrics(
                window_id=wr.window.window_id,
                train_start_date=(
                    wr.window.train_start_date.strftime("%Y-%m-%d")
                    if wr.window.train_start_date
                    else None
                ),
                train_end_date=(
                    wr.window.train_end_date.strftime("%Y-%m-%d")
                    if wr.window.train_end_date
                    else None
                ),
                test_start_date=(
                    wr.window.test_start_date.strftime("%Y-%m-%d")
                    if wr.window.test_start_date
                    else None
                ),
                test_end_date=(
                    wr.window.test_end_date.strftime("%Y-%m-%d")
                    if wr.window.test_end_date
                    else None
                ),
                initial_equity=wr.initial_equity,
                final_equity=wr.final_equity,
                return_pct=return_pct,
                sharpe_ratio=wr.metrics.get("sharpe_ratio", 0.0),
                max_drawdown_pct=abs(wr.metrics.get("max_drawdown_pct", 0.0)),
                win_rate=wr.metrics.get("win_rate", 0.0),
                num_trades=wr.metrics.get("total_trades", 0),
                is_profitable=wr.is_profitable,
            )
        )

    # Build timestamps for combined equity curve
    # For simplicity, generate sequential timestamps based on test window dates
    combined_timestamps: list[str] = []
    ts_idx = 0
    for wr in wf_result.window_results:
        test_start_idx = wr.window.test_start_idx
        window_ts_count = len(wr.equity_curve)
        # First window includes all points, subsequent windows skip first
        if ts_idx == 0:
            for i in range(window_ts_count):
                bar_idx = test_start_idx + i
                if bar_idx < len(timestamps_list):
                    combined_timestamps.append(timestamps_list[bar_idx])
                else:
                    combined_timestamps.append("")
        else:
            # Skip first point (junction point)
            for i in range(1, window_ts_count):
                bar_idx = test_start_idx + i
                if bar_idx < len(timestamps_list):
                    combined_timestamps.append(timestamps_list[bar_idx])
                else:
                    combined_timestamps.append("")
        ts_idx += 1

    app_state.update_timestamp()

    return WalkForwardResponse(
        symbol=symbol,
        start_date=config.start_date,
        end_date=config.end_date,
        initial_capital=config.initial_capital,
        final_capital=aggregate.final_capital,
        total_return_pct=aggregate.total_return_pct,
        total_windows=aggregate.total_windows,
        profitable_windows=aggregate.total_profitable_windows,
        pass_rate=aggregate.pass_rate,
        passes_walk_forward_test=aggregate.passes_walk_forward_test,
        pass_threshold=config.pass_threshold,
        aggregate_sharpe=aggregate.aggregate_sharpe,
        aggregate_max_drawdown_pct=abs(aggregate.aggregate_max_drawdown_pct),
        aggregate_win_rate=aggregate.aggregate_win_rate,
        total_trades=aggregate.total_trades,
        window_results=window_metrics,
        combined_equity_curve=aggregate.combined_equity_curve,
        timestamps=combined_timestamps,
        train_bars=config.train_bars,
        test_bars=config.test_bars,
    )


# ============================================================================
# Symbol Validation Endpoints
# ============================================================================


@app.post("/api/symbol/validate", response_model=SymbolValidationResponse)
async def validate_symbol(request: SymbolValidationRequest):
    """
    Validate a stock symbol and get its metadata.

    Returns symbol info if valid, raises 404 if symbol not found.
    Uses the configured data provider (Yahoo Finance by default).

    Args:
        request: SymbolValidationRequest with symbol to validate

    Returns:
        SymbolValidationResponse with symbol metadata

    Raises:
        404: Symbol not found
        500: Provider error
    """
    from backend.data.provider import (
        DataProviderError,
        SymbolNotFoundError,
        get_provider,
    )

    symbol = request.symbol.upper().strip()

    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol cannot be empty")

    try:
        provider = get_provider()
        symbol_info = await provider.validate_symbol(symbol)

        return SymbolValidationResponse(
            symbol=symbol_info.symbol,
            name=symbol_info.name,
            exchange=symbol_info.exchange,
            currency=symbol_info.currency,
            type=symbol_info.type,
            is_valid=symbol_info.is_valid,
        )

    except SymbolNotFoundError as e:
        logger.warning(f"Symbol not found: {symbol} - {e}")
        raise HTTPException(
            status_code=404,
            detail=f"Symbol '{symbol}' not found. Please check the ticker symbol is correct.",
        )

    except DataProviderError as e:
        logger.error(f"Provider error validating {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error validating symbol: {e}",
        )


@app.get("/api/symbol/search", response_model=SymbolSearchResponse)
async def search_symbols(
    q: str = Query(..., min_length=1, description="Search query (symbol or company name)"),
    limit: int = Query(default=10, ge=1, le=50, description="Maximum results"),
):
    """
    Search for stock symbols matching a query.

    Currently supports exact symbol matching.
    Future: Add fuzzy company name search via external API.

    Args:
        q: Search query
        limit: Maximum results to return (1-50)

    Returns:
        SymbolSearchResponse with matching symbols
    """
    from backend.data.provider import DataProviderError, get_provider

    query = q.upper().strip()

    try:
        provider = get_provider()
        results = await provider.search_symbols(query, limit)

        return SymbolSearchResponse(
            query=query,
            results=[
                SymbolValidationResponse(
                    symbol=r.symbol,
                    name=r.name,
                    exchange=r.exchange,
                    currency=r.currency,
                    type=r.type,
                    is_valid=r.is_valid,
                )
                for r in results
            ],
        )

    except DataProviderError as e:
        logger.error(f"Provider error searching '{query}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error searching symbols: {e}",
        )


# ============================================================================
# WebSocket Endpoint
# ============================================================================


@app.websocket("/ws/prices")
async def websocket_prices(websocket: WebSocket):
    """
    WebSocket endpoint for live price updates.

    Clients connect to this endpoint to receive real-time price updates.
    Messages are sent in JSON format with symbol, price, timestamp.

    Authentication:
        Requires valid bearer token in Authorization header.
        Connections without valid auth are rejected with code 4001.
    """
    # Validate authentication before accepting connection
    if not validate_websocket_auth(websocket.headers):
        logger.warning(
            "WebSocket connection rejected: invalid authentication",
            extra={"client": websocket.client},
        )
        await websocket.close(code=4001, reason="Authentication failed")
        return

    await websocket.accept()
    app_state.websocket_clients.append(websocket)
    app_state.data_feed_active = True

    logger.info(
        "WebSocket client connected", extra={"total_clients": len(app_state.websocket_clients)}
    )

    try:
        # Send initial connection message
        await websocket.send_json(
            {
                "type": "connection",
                "status": "connected",
                "message": "WebSocket connection established",
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Replay CSV data if available, otherwise send synthetic data
        if app_state.test_data:
            # Replay mode: iterate through CSV data for all symbols
            # Create index trackers for each symbol
            symbol_indices = {symbol: 0 for symbol in app_state.test_data.keys()}

            logger.info(
                "WebSocket: Starting CSV replay mode",
                extra={
                    "symbols": list(app_state.test_data.keys()),
                    "rows_per_symbol": {k: len(v) for k, v in app_state.test_data.items()},
                },
            )

            while True:
                # Send updates for all symbols
                for symbol, data in app_state.test_data.items():
                    if not data:
                        continue

                    # Get current candle for this symbol
                    data_index = symbol_indices[symbol]
                    candle = data[data_index]

                    # Send OHLCV update to client
                    await websocket.send_json(
                        {
                            "type": "price_update",
                            "symbol": symbol,
                            "timestamp": candle["timestamp"],
                            "open": candle["open"],
                            "high": candle["high"],
                            "low": candle["low"],
                            "close": candle["close"],
                            "volume": candle["volume"],
                            "replay_index": data_index,
                            "total_rows": len(data),
                        }
                    )
                    app_state.update_timestamp()

                    # Move to next row for this symbol, loop back to start when done
                    symbol_indices[symbol] = (data_index + 1) % len(data)

                # Send updates every 2 seconds (simulating live feed)
                # Divide by number of symbols to maintain reasonable update frequency
                await asyncio.sleep(2.0)

        else:
            # Fallback: synthetic price updates if CSV data not available
            logger.warning("WebSocket: CSV data not available, using synthetic prices")
            synthetic_symbols = ["SPY", "AAPL", "MSFT"]
            while True:
                # Generate synthetic price updates for all symbols
                for symbol in synthetic_symbols:
                    # Set price range based on typical stock prices
                    if symbol == "SPY":
                        price_range = (400.0, 450.0)
                    elif symbol == "AAPL":
                        price_range = (150.0, 250.0)
                    else:  # MSFT
                        price_range = (300.0, 450.0)

                    price_update = PriceUpdate(
                        symbol=symbol,
                        price=round(np.random.uniform(*price_range), 2),
                        timestamp=datetime.now().isoformat(),
                        volume=np.random.randint(100000, 1000000),
                    )

                    # Send update to client
                    await websocket.send_json(price_update.model_dump())
                    app_state.update_timestamp()

                # Wait 5 seconds before next batch of updates
                await asyncio.sleep(5.0)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error("WebSocket error", extra={"error": str(e)}, exc_info=True)
    finally:
        # Remove client from list
        if websocket in app_state.websocket_clients:
            app_state.websocket_clients.remove(websocket)

        # Mark data feed as inactive if no clients
        if len(app_state.websocket_clients) == 0:
            app_state.data_feed_active = False

        logger.info(
            "WebSocket client removed", extra={"total_clients": len(app_state.websocket_clients)}
        )


# ============================================================================
# Health Check Endpoint
# ============================================================================


@app.get("/metrics")
async def metrics():
    """
    Prometheus-compatible metrics endpoint.

    Exposes:
    - Order counts (total trades from database)
    - Request latency percentiles (p50, p90, p95, p99)
    - Current drawdown percentage
    - Request counts by endpoint
    - System uptime

    Returns metrics in Prometheus text format.
    """
    lines = []

    # Helper to format Prometheus metrics
    def add_metric(name: str, value: float, help_text: str, metric_type: str = "gauge"):
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {metric_type}")
        lines.append(f"{name} {value}")
        lines.append("")

    # System uptime
    uptime_seconds = (datetime.now() - app_state.start_time).total_seconds()
    add_metric("fluxhero_uptime_seconds", uptime_seconds, "System uptime in seconds", "counter")

    # Request metrics
    add_metric(
        "fluxhero_requests_total",
        app_state.request_count,
        "Total number of HTTP requests",
        "counter",
    )

    # Request latency percentiles
    if app_state.request_latencies:
        latencies = np.array(app_state.request_latencies)
        p50 = float(np.percentile(latencies, 50))
        p90 = float(np.percentile(latencies, 90))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))

        add_metric(
            "fluxhero_request_latency_p50_ms",
            p50,
            "Request latency 50th percentile in milliseconds",
        )
        add_metric(
            "fluxhero_request_latency_p90_ms",
            p90,
            "Request latency 90th percentile in milliseconds",
        )
        add_metric(
            "fluxhero_request_latency_p95_ms",
            p95,
            "Request latency 95th percentile in milliseconds",
        )
        add_metric(
            "fluxhero_request_latency_p99_ms",
            p99,
            "Request latency 99th percentile in milliseconds",
        )

    # Order/Trade counts from database
    if app_state.sqlite_store:
        try:
            # Get recent trades to calculate metrics (limit to last 1000 for performance)
            all_trades = await app_state.sqlite_store.get_recent_trades(limit=1000)
            total_trades = len(all_trades)
            add_metric(
                "fluxhero_orders_total", total_trades, "Total number of orders/trades", "counter"
            )

            # Calculate current drawdown from trades
            if all_trades:
                # Calculate equity curve from trade P&L
                equity = 100000.0  # Assume initial capital
                peak_equity = equity
                current_drawdown_pct = 0.0

                for trade in all_trades:
                    if trade.realized_pnl is not None:
                        equity += trade.realized_pnl
                        if equity > peak_equity:
                            peak_equity = equity
                        elif peak_equity > 0:
                            current_drawdown_pct = ((peak_equity - equity) / peak_equity) * 100.0

                add_metric(
                    "fluxhero_drawdown_percent", current_drawdown_pct, "Current drawdown percentage"
                )
                add_metric("fluxhero_equity", equity, "Current equity value")

            # Win rate
            completed_trades = [t for t in all_trades if t.realized_pnl is not None]
            if completed_trades:
                winning_trades = len([t for t in completed_trades if t.realized_pnl > 0])
                win_rate = (winning_trades / len(completed_trades)) * 100.0
                add_metric("fluxhero_win_rate_percent", win_rate, "Win rate percentage")
        except Exception as e:
            logger.error(f"Error calculating trade metrics: {e}")

    # Request counts by endpoint
    for path, count in app_state.request_count_by_path.items():
        # Sanitize path for Prometheus label
        safe_path = path.replace("/", "_").replace("-", "_").strip("_")
        if safe_path:
            lines.append("# HELP fluxhero_requests_by_path_total Requests by endpoint path")
            lines.append("# TYPE fluxhero_requests_by_path_total counter")
            lines.append(f'fluxhero_requests_by_path_total{{path="{path}"}} {count}')
            lines.append("")

    # WebSocket connections
    add_metric(
        "fluxhero_websocket_connections",
        len(app_state.websocket_clients),
        "Active WebSocket connections",
    )

    # Data feed status
    add_metric(
        "fluxhero_data_feed_active",
        1.0 if app_state.data_feed_active else 0.0,
        "Data feed active status (1=active, 0=inactive)",
    )

    # Return as plain text with Prometheus content type
    return Response(content="\n".join(lines), media_type="text/plain; version=0.0.4")


@app.get("/api/test/candles")
async def get_test_candles(
    symbol: str = Query(
        default="SPY",
        description="Symbol to fetch (SPY, AAPL, or MSFT)",
    ),
):
    """
    TEST ENDPOINT: Get historical candle data for development.

    This endpoint serves static data from CSV files for frontend development.
    Supports SPY, AAPL, and MSFT symbols.
    It is disabled in production environments (ENV=production).

    Args:
        symbol: The symbol to fetch (SPY, AAPL, or MSFT)

    Returns:
        List of candle data with timestamp, open, high, low, close, volume

    Raises:
        HTTPException: If endpoint is disabled or data is not available
    """
    # Check if endpoint is enabled (disabled in production)
    env = os.getenv("ENV", "development")
    if env == "production":
        raise HTTPException(
            status_code=403,
            detail="Test endpoints are disabled in production",
        )

    # Validate symbol
    symbol_upper = symbol.upper()
    supported_symbols = ["SPY", "AAPL", "MSFT"]
    if symbol_upper not in supported_symbols:
        raise HTTPException(
            status_code=400,
            detail=f"Symbol {symbol} not supported. Available: {', '.join(supported_symbols)}",
        )

    # Check if data is loaded
    if symbol_upper not in app_state.test_data or not app_state.test_data[symbol_upper]:
        raise HTTPException(
            status_code=503,
            detail=f"Test data for {symbol_upper} not available. Check server logs for details.",
        )

    return app_state.test_data[symbol_upper]


# ============================================================================
# Mode Management Endpoints
# ============================================================================


def _is_live_broker_configured() -> bool:
    """Check if live broker credentials are configured."""
    settings = get_settings()
    return bool(settings.alpaca_api_key and settings.alpaca_api_secret)


@app.get("/api/mode", response_model=ModeStateResponse)
async def get_mode_state():
    """
    Get the current trading mode state.

    Returns:
    - active_mode: 'live' or 'paper'
    - last_mode_change: timestamp of last mode change
    - paper_balance: current paper trading balance
    - paper_realized_pnl: total realized P&L in paper mode
    - is_live_broker_configured: whether live broker is available
    """
    if not app_state.sqlite_store:
        raise HTTPException(status_code=503, detail="Database not initialized")

    mode_state = await app_state.sqlite_store.get_mode_state()

    return ModeStateResponse(
        active_mode=mode_state.active_mode,
        last_mode_change=mode_state.last_mode_change,
        paper_balance=mode_state.paper_balance,
        paper_realized_pnl=mode_state.paper_realized_pnl,
        is_live_broker_configured=_is_live_broker_configured(),
    )


@app.post("/api/mode", response_model=ModeStateResponse)
async def switch_mode(request: SwitchModeRequest):
    """
    Switch trading mode between live and paper.

    Requirements:
    - To switch to live mode, confirm_live must be True
    - Live broker must be configured to switch to live mode

    Raises:
    - 400: Invalid mode or confirmation not provided
    - 503: Database not initialized
    """
    if not app_state.sqlite_store:
        raise HTTPException(status_code=503, detail="Database not initialized")

    # Validate mode
    if request.mode not in ("live", "paper"):
        raise HTTPException(status_code=400, detail="Mode must be 'live' or 'paper'")

    target_mode = TradingMode.LIVE if request.mode == "live" else TradingMode.PAPER

    # Safety check for live mode
    if target_mode == TradingMode.LIVE:
        if not request.confirm_live:
            raise HTTPException(
                status_code=400,
                detail="Live mode requires explicit confirmation (confirm_live=true)",
            )
        if not _is_live_broker_configured():
            raise HTTPException(
                status_code=400,
                detail="Live broker is not configured. Add broker credentials first.",
            )

    # Switch mode
    await app_state.sqlite_store.set_active_mode(target_mode)
    logger.info(f"Trading mode switched to {target_mode.value}")

    # Return updated state
    mode_state = await app_state.sqlite_store.get_mode_state()

    return ModeStateResponse(
        active_mode=mode_state.active_mode,
        last_mode_change=mode_state.last_mode_change,
        paper_balance=mode_state.paper_balance,
        paper_realized_pnl=mode_state.paper_realized_pnl,
        is_live_broker_configured=_is_live_broker_configured(),
    )


# ============================================================================
# Mode-Specific Endpoints: Positions
# ============================================================================


@app.get("/api/live/positions", response_model=list[PositionResponse])
async def get_live_positions():
    """Get all positions from live trading."""
    if not app_state.sqlite_store:
        raise HTTPException(status_code=503, detail="Database not initialized")

    positions = await app_state.sqlite_store.get_positions_for_mode(TradingMode.LIVE)

    return [
        PositionResponse(
            id=pos.id,
            symbol=pos.symbol,
            side=pos.side,
            shares=pos.shares,
            entry_price=pos.entry_price,
            current_price=pos.current_price,
            unrealized_pnl=pos.unrealized_pnl,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
            entry_time=pos.entry_time,
            updated_at=pos.updated_at,
        )
        for pos in positions
    ]


@app.get("/api/paper/positions", response_model=list[PositionResponse])
async def get_paper_positions():
    """Get all positions from paper trading."""
    if not app_state.sqlite_store:
        raise HTTPException(status_code=503, detail="Database not initialized")

    positions = await app_state.sqlite_store.get_positions_for_mode(TradingMode.PAPER)

    return [
        PositionResponse(
            id=pos.id,
            symbol=pos.symbol,
            side=pos.side,
            shares=pos.shares,
            entry_price=pos.entry_price,
            current_price=pos.current_price,
            unrealized_pnl=pos.unrealized_pnl,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
            entry_time=pos.entry_time,
            updated_at=pos.updated_at,
        )
        for pos in positions
    ]


# ============================================================================
# Mode-Specific Endpoints: Trades
# ============================================================================


@app.get("/api/live/trades", response_model=TradeHistoryResponse)
async def get_live_trades(
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
):
    """Get trade history from live trading with pagination."""
    if not app_state.sqlite_store:
        raise HTTPException(status_code=503, detail="Database not initialized")

    trades, total_count = await app_state.sqlite_store.get_trades_paginated_for_mode(
        TradingMode.LIVE, page, page_size
    )
    total_pages = (total_count + page_size - 1) // page_size

    return TradeHistoryResponse(
        trades=[
            TradeResponse(
                id=t.id,
                symbol=t.symbol,
                side=t.side,
                entry_price=t.entry_price,
                exit_price=t.exit_price,
                entry_time=t.entry_time,
                exit_time=t.exit_time,
                shares=t.shares,
                realized_pnl=t.realized_pnl,
                status=t.status,
                strategy=t.strategy or "",
                regime=t.regime or "",
                signal_reason=t.signal_reason or "",
                stop_loss=t.stop_loss,
                take_profit=t.take_profit,
            )
            for t in trades
        ],
        total_count=total_count,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@app.get("/api/paper/trades", response_model=TradeHistoryResponse)
async def get_paper_trades(
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
):
    """Get trade history from paper trading with pagination."""
    if not app_state.sqlite_store:
        raise HTTPException(status_code=503, detail="Database not initialized")

    trades, total_count = await app_state.sqlite_store.get_trades_paginated_for_mode(
        TradingMode.PAPER, page, page_size
    )
    total_pages = (total_count + page_size - 1) // page_size

    return TradeHistoryResponse(
        trades=[
            TradeResponse(
                id=t.id,
                symbol=t.symbol,
                side=t.side,
                entry_price=t.entry_price,
                exit_price=t.exit_price,
                entry_time=t.entry_time,
                exit_time=t.exit_time,
                shares=t.shares,
                realized_pnl=t.realized_pnl,
                status=t.status,
                strategy=t.strategy or "",
                regime=t.regime or "",
                signal_reason=t.signal_reason or "",
                stop_loss=t.stop_loss,
                take_profit=t.take_profit,
            )
            for t in trades
        ],
        total_count=total_count,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


# ============================================================================
# Mode-Specific Endpoints: Account
# ============================================================================


@app.get("/api/live/account", response_model=AccountInfoResponse)
async def get_live_account():
    """Get account info for live trading."""
    if not app_state.sqlite_store:
        raise HTTPException(status_code=503, detail="Database not initialized")

    # Get positions and trades for live mode
    positions = await app_state.sqlite_store.get_positions_for_mode(TradingMode.LIVE)
    trades, _ = await app_state.sqlite_store.get_trades_paginated_for_mode(
        TradingMode.LIVE, page=1, page_size=1000
    )

    # Calculate metrics
    total_unrealized = sum(p.unrealized_pnl for p in positions)
    total_realized = sum(t.realized_pnl or 0 for t in trades if t.status == TradeStatus.CLOSED)

    # For live account, we'd normally get this from the broker
    # For now, use a placeholder - in production this would query the broker
    initial_equity = 100000.0  # Placeholder
    current_equity = initial_equity + total_realized + total_unrealized

    return AccountInfoResponse(
        equity=current_equity,
        cash=current_equity - sum(p.current_price * p.shares for p in positions),
        buying_power=current_equity * 2,  # Assume 2x margin
        total_pnl=total_realized + total_unrealized,
        daily_pnl=total_unrealized,  # Simplified - would need proper daily tracking
        num_positions=len(positions),
    )


@app.get("/api/paper/account", response_model=AccountInfoResponse)
async def get_paper_account_info():
    """Get account info for paper trading."""
    if not app_state.sqlite_store:
        raise HTTPException(status_code=503, detail="Database not initialized")

    # Get mode state for paper balance
    mode_state = await app_state.sqlite_store.get_mode_state()

    # Get positions and calculate unrealized P&L
    positions = await app_state.sqlite_store.get_positions_for_mode(TradingMode.PAPER)
    total_unrealized = sum(p.unrealized_pnl for p in positions)
    position_value = sum(p.current_price * p.shares for p in positions)

    current_equity = mode_state.paper_balance + total_unrealized

    return AccountInfoResponse(
        equity=current_equity,
        cash=mode_state.paper_balance - position_value,
        buying_power=(mode_state.paper_balance - position_value) * 2,
        total_pnl=mode_state.paper_realized_pnl + total_unrealized,
        daily_pnl=total_unrealized,
        num_positions=len(positions),
    )


# ============================================================================
# Mode-Specific Endpoints: Order Placement
# ============================================================================


@app.post("/api/paper/orders", response_model=PlaceOrderResponse)
async def place_paper_order(request: PlaceOrderRequest):
    """
    Place an order in paper trading mode.

    Executes via PaperBroker for simulated trading.

    Args:
        request: Order details (symbol, qty, side, order_type, limit_price)

    Returns:
        PlaceOrderResponse with order execution details
    """
    if not app_state.sqlite_store:
        raise HTTPException(status_code=503, detail="Database not initialized")

    try:
        from backend.execution.broker_base import OrderSide, OrderType

        # Get paper broker
        paper_broker = await _get_paper_broker()

        # Parse side
        side = OrderSide.BUY if request.side.lower() == "buy" else OrderSide.SELL

        # Parse order type
        order_type = OrderType.MARKET
        if request.order_type.lower() == "limit":
            order_type = OrderType.LIMIT

        # Place the order
        order = await paper_broker.place_order(
            symbol=request.symbol.upper(),
            qty=request.qty,
            side=side,
            order_type=order_type,
            limit_price=request.limit_price,
        )

        # Return response
        return PlaceOrderResponse(
            order_id=order.order_id,
            symbol=order.symbol,
            qty=order.qty,
            side=request.side.lower(),
            status=order.status.name.lower(),
            filled_price=order.filled_price,
            mode="paper",
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to place paper order: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to place order: {str(e)}")


@app.post("/api/live/orders", response_model=PlaceOrderResponse)
async def place_live_order(
    request: PlaceOrderRequest,
    x_confirm_live_trade: str | None = Header(default=None, alias="X-Confirm-Live-Trade"),
):
    """
    Place an order in live trading mode.

    CAUTION: This places real orders with real money.
    Requires X-Confirm-Live-Trade: true header for safety.

    Args:
        request: Order details (symbol, qty, side, order_type, limit_price)
        x_confirm_live_trade: Safety header - must be "true" to execute

    Returns:
        PlaceOrderResponse with order execution details
    """
    if not app_state.sqlite_store:
        raise HTTPException(status_code=503, detail="Database not initialized")

    # Safety check - require explicit confirmation header
    if x_confirm_live_trade != "true":
        raise HTTPException(
            status_code=400,
            detail="Live trading requires X-Confirm-Live-Trade: true header"
        )

    # Check if live broker is configured
    if not app_state.broker:
        raise HTTPException(
            status_code=503,
            detail="Live broker not configured. Please configure broker in settings."
        )

    try:
        from backend.execution.broker_base import OrderSide, OrderType

        # Parse side
        side = OrderSide.BUY if request.side.lower() == "buy" else OrderSide.SELL

        # Parse order type
        order_type = OrderType.MARKET
        if request.order_type.lower() == "limit":
            order_type = OrderType.LIMIT

        # Place the order via live broker
        order = await app_state.broker.place_order(
            symbol=request.symbol.upper(),
            qty=request.qty,
            side=side,
            order_type=order_type,
            limit_price=request.limit_price,
        )

        # Return response
        return PlaceOrderResponse(
            order_id=order.order_id,
            symbol=order.symbol,
            qty=order.qty,
            side=request.side.lower(),
            status=order.status.name.lower(),
            filled_price=order.filled_price,
            mode="live",
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to place live order: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to place order: {str(e)}")


# ============================================================================
# Backtest Results Endpoints
# ============================================================================


@app.get("/api/backtest/results", response_model=list[BacktestResultSummaryResponse])
async def get_backtest_results(
    limit: int = Query(default=50, ge=1, le=200, description="Max results to return"),
):
    """Get list of backtest results (for viewing in paper mode)."""
    if not app_state.sqlite_store:
        raise HTTPException(status_code=503, detail="Database not initialized")

    results = await app_state.sqlite_store.get_backtest_results(limit=limit)

    return [
        BacktestResultSummaryResponse(
            id=r.id,
            run_id=r.run_id,
            symbol=r.symbol,
            strategy_mode=r.strategy_mode,
            start_date=r.start_date,
            end_date=r.end_date,
            total_return_pct=r.total_return_pct,
            sharpe_ratio=r.sharpe_ratio,
            max_drawdown_pct=r.max_drawdown_pct,
            win_rate=r.win_rate,
            num_trades=r.num_trades,
            created_at=r.created_at,
        )
        for r in results
    ]


@app.get("/api/backtest/results/{run_id}", response_model=BacktestResultDetailResponse)
async def get_backtest_result_detail(run_id: str):
    """Get detailed backtest result by run_id."""
    if not app_state.sqlite_store:
        raise HTTPException(status_code=503, detail="Database not initialized")

    result = await app_state.sqlite_store.get_backtest_result(run_id)

    if not result:
        raise HTTPException(status_code=404, detail=f"Backtest result not found: {run_id}")

    return BacktestResultDetailResponse(
        id=result.id,
        run_id=result.run_id,
        symbol=result.symbol,
        strategy_mode=result.strategy_mode,
        start_date=result.start_date,
        end_date=result.end_date,
        initial_capital=result.initial_capital,
        final_equity=result.final_equity,
        total_return_pct=result.total_return_pct,
        sharpe_ratio=result.sharpe_ratio,
        max_drawdown_pct=result.max_drawdown_pct,
        win_rate=result.win_rate,
        num_trades=result.num_trades,
        equity_curve_json=result.equity_curve_json,
        trades_json=result.trades_json,
        config_json=result.config_json,
        created_at=result.created_at,
    )


# ============================================================================
# Data Management Endpoint
# ============================================================================


@app.post("/api/data/clear")
async def clear_all_trading_data(
    confirm: bool = Query(default=False, description="Must be true to confirm data deletion"),
):
    """
    Clear all trading data for fresh start.
    This removes all trades, positions, and backtest results.
    Resets paper balance to default.

    WARNING: This action is irreversible!
    """
    if not app_state.sqlite_store:
        raise HTTPException(status_code=503, detail="Database not initialized")

    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Must confirm data deletion with confirm=true",
        )

    await app_state.sqlite_store.clear_all_trading_data()
    logger.warning("All trading data cleared by user request")

    return {"status": "success", "message": "All trading data has been cleared"}


@app.get("/health")
async def health_check():
    """
    Health check endpoint with basic system metrics.

    Returns:
    - Status: healthy/unhealthy
    - Timestamp: Current server time
    - Uptime: Seconds since server start
    - Database: Connection status
    - Active connections: WebSocket client count
    """
    uptime_seconds = (datetime.now() - app_state.start_time).total_seconds()

    # Check database connectivity
    db_healthy = False
    if app_state.sqlite_store:
        try:
            # Simple connectivity test
            await app_state.sqlite_store.get_recent_trades(limit=1)
            db_healthy = True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")

    # Determine overall health status
    overall_status = "healthy" if db_healthy else "degraded"

    return {
        "status": overall_status,
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": round(uptime_seconds, 2),
        "database_connected": db_healthy,
        "websocket_connections": len(app_state.websocket_clients),
        "data_feed_active": app_state.data_feed_active,
        "total_requests": app_state.request_count,
    }


# ============================================================================
# Broker Management Endpoints
# ============================================================================


class BrokerConfigRequest(BaseModel):
    """Request model for adding a broker configuration."""

    broker_type: str = Field(..., description="Type of broker (e.g., 'alpaca')")
    name: str = Field(
        ..., min_length=1, max_length=100, description="Display name for broker config"
    )
    api_key: str = Field(..., min_length=1, description="Broker API key")
    api_secret: str = Field(..., min_length=1, description="Broker API secret")
    base_url: str | None = Field(
        default=None,
        description="Optional API base URL (defaults to paper trading URL)",
    )


class BrokerConfigResponse(BaseModel):
    """Response model for broker configuration (without secrets)."""

    id: str
    broker_type: str
    name: str
    api_key_masked: str
    base_url: str
    is_connected: bool = False
    created_at: str
    updated_at: str


class BrokerListResponse(BaseModel):
    """Response model for listing brokers."""

    brokers: list[BrokerConfigResponse]
    total: int


class BrokerHealthResponse(BaseModel):
    """Response model for broker health check."""

    id: str
    name: str
    broker_type: str
    is_connected: bool
    is_authenticated: bool
    latency_ms: float | None = None
    last_heartbeat: str | None = None
    error_message: str | None = None


# ============================================================================
# Paper Trading Pydantic Models
# ============================================================================


class PaperPositionResponse(BaseModel):
    """Response model for a paper trading position."""

    symbol: str
    qty: int
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    cost_basis: float


class PaperAccountResponse(BaseModel):
    """Response model for paper trading account information."""

    account_id: str
    balance: float
    buying_power: float
    equity: float
    cash: float
    positions_value: float
    realized_pnl: float
    unrealized_pnl: float
    positions: list[PaperPositionResponse]


class PaperTradeResponse(BaseModel):
    """Response model for a paper trade."""

    trade_id: str
    order_id: str
    symbol: str
    side: str  # "BUY" or "SELL"
    qty: int
    price: float
    slippage: float
    timestamp: str
    realized_pnl: float


class PaperTradeHistoryResponse(BaseModel):
    """Response model for paper trade history."""

    trades: list[PaperTradeResponse]
    total_count: int


class PaperResetResponse(BaseModel):
    """Response model for paper account reset."""

    message: str
    account_id: str
    initial_balance: float
    timestamp: str


# Storage key prefix for broker configs
BROKER_CONFIG_PREFIX = "broker_config:"


def _get_broker_storage_key(broker_id: str) -> str:
    """Get the storage key for a broker config."""
    return f"{BROKER_CONFIG_PREFIX}{broker_id}"


def _generate_broker_id() -> str:
    """Generate a unique broker ID."""
    import uuid

    return str(uuid.uuid4())[:8]


async def _get_all_broker_configs(store: SQLiteStore) -> list[dict]:
    """
    Get all broker configurations from storage.

    Returns:
        List of broker config dictionaries
    """
    import json

    all_settings = await store.get_all_settings()
    configs = []

    for key, value in all_settings.items():
        if key.startswith(BROKER_CONFIG_PREFIX):
            try:
                config = json.loads(value)
                configs.append(config)
            except json.JSONDecodeError:
                logger.warning(f"Invalid broker config JSON for key: {key}")

    return configs


async def _get_broker_config(store: SQLiteStore, broker_id: str) -> dict | None:
    """
    Get a specific broker configuration by ID.

    Args:
        store: SQLite store instance
        broker_id: Broker configuration ID

    Returns:
        Broker config dict or None if not found
    """
    import json

    key = _get_broker_storage_key(broker_id)
    value = await store.get_setting(key)

    if value is None:
        return None

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        logger.warning(f"Invalid broker config JSON for ID: {broker_id}")
        return None


async def _save_broker_config(store: SQLiteStore, config: dict) -> None:
    """
    Save a broker configuration to storage.

    Args:
        store: SQLite store instance
        config: Broker config dictionary
    """
    import json

    key = _get_broker_storage_key(config["id"])
    await store.set_setting(key, json.dumps(config), f"Broker config: {config['name']}")


async def _delete_broker_config(store: SQLiteStore, broker_id: str) -> bool:
    """
    Delete a broker configuration from storage.

    Args:
        store: SQLite store instance
        broker_id: Broker configuration ID

    Returns:
        True if deleted, False if not found
    """
    key = _get_broker_storage_key(broker_id)
    existing = await store.get_setting(key)

    if existing is None:
        return False

    # Delete by setting to empty value and then using raw SQL
    # Since SQLiteStore doesn't have delete_setting, we'll use set_setting with a marker
    # Actually, let's add a proper delete - for now we'll mark as deleted
    # Better approach: just store empty JSON which we filter out
    conn = store._get_connection()
    conn.execute("DELETE FROM settings WHERE key = ?", (key,))
    conn.commit()
    return True


@app.get("/api/brokers", response_model=BrokerListResponse)
async def list_brokers():
    """
    List all configured brokers.

    Returns configured brokers with masked credentials.
    Does not include full API keys/secrets for security.

    Returns:
        BrokerListResponse with list of broker configurations
    """
    from backend.execution.broker_credentials import mask_credential

    if not app_state.sqlite_store:
        raise HTTPException(status_code=503, detail="Database not initialized")

    configs = await _get_all_broker_configs(app_state.sqlite_store)

    brokers = []
    for config in configs:
        brokers.append(
            BrokerConfigResponse(
                id=config["id"],
                broker_type=config["broker_type"],
                name=config["name"],
                api_key_masked=mask_credential(config.get("api_key_masked", "****")),
                base_url=config.get("base_url", ""),
                is_connected=False,  # Will be updated by health check
                created_at=config.get("created_at", ""),
                updated_at=config.get("updated_at", ""),
            )
        )

    app_state.update_timestamp()

    return BrokerListResponse(
        brokers=brokers,
        total=len(brokers),
    )


@app.post("/api/brokers", response_model=BrokerConfigResponse, status_code=201)
async def add_broker(config: BrokerConfigRequest):
    """
    Add a new broker configuration.

    Validates the broker config using Pydantic models and stores
    credentials encrypted with AES-256-GCM.

    Args:
        config: Broker configuration with credentials

    Returns:
        BrokerConfigResponse with the created broker config (credentials masked)

    Raises:
        400: Invalid broker type or config
        503: Database not initialized
    """
    from backend.execution.broker_credentials import encrypt_credential, mask_credential
    from backend.execution.broker_factory import BROKER_CONFIG_MODELS

    if not app_state.sqlite_store:
        raise HTTPException(status_code=503, detail="Database not initialized")

    # Validate broker type
    if config.broker_type not in BROKER_CONFIG_MODELS:
        supported = list(BROKER_CONFIG_MODELS.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Unknown broker type: '{config.broker_type}'. Supported types: {supported}",
        )

    # Validate config using Pydantic model
    config_model = BROKER_CONFIG_MODELS[config.broker_type]
    try:
        # Build config dict for validation
        validation_config = {
            "api_key": config.api_key,
            "api_secret": config.api_secret,
        }
        if config.base_url:
            validation_config["base_url"] = config.base_url

        validated = config_model(**validation_config)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid broker configuration: {e}",
        )

    # Generate unique ID
    broker_id = _generate_broker_id()
    now = datetime.now().isoformat()

    # Encrypt credentials
    try:
        encrypted_api_key = encrypt_credential(config.api_key)
        encrypted_api_secret = encrypt_credential(config.api_secret)
    except Exception as e:
        logger.error(f"Failed to encrypt credentials: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to encrypt credentials",
        )

    # Build stored config (credentials encrypted, original masked for display)
    stored_config = {
        "id": broker_id,
        "broker_type": config.broker_type,
        "name": config.name,
        "api_key_encrypted": encrypted_api_key,
        "api_secret_encrypted": encrypted_api_secret,
        "api_key_masked": mask_credential(config.api_key),
        "base_url": config.base_url or validated.base_url,
        "created_at": now,
        "updated_at": now,
    }

    # Save to storage
    await _save_broker_config(app_state.sqlite_store, stored_config)

    logger.info(
        "Broker configuration added",
        extra={"broker_id": broker_id, "broker_type": config.broker_type, "name": config.name},
    )

    app_state.update_timestamp()

    return BrokerConfigResponse(
        id=broker_id,
        broker_type=config.broker_type,
        name=config.name,
        api_key_masked=stored_config["api_key_masked"],
        base_url=stored_config["base_url"],
        is_connected=False,
        created_at=now,
        updated_at=now,
    )


@app.delete("/api/brokers/{broker_id}", status_code=204)
async def delete_broker(broker_id: str):
    """
    Delete a broker configuration.

    Removes the broker configuration from storage. This does not
    affect any active broker connections.

    Args:
        broker_id: Unique broker configuration ID

    Raises:
        404: Broker not found
        503: Database not initialized
    """
    if not app_state.sqlite_store:
        raise HTTPException(status_code=503, detail="Database not initialized")

    # Check if broker exists
    config = await _get_broker_config(app_state.sqlite_store, broker_id)
    if config is None:
        raise HTTPException(status_code=404, detail=f"Broker not found: {broker_id}")

    # Delete from storage
    deleted = await _delete_broker_config(app_state.sqlite_store, broker_id)

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Broker not found: {broker_id}")

    logger.info(
        "Broker configuration deleted",
        extra={"broker_id": broker_id, "name": config.get("name", "unknown")},
    )

    app_state.update_timestamp()

    # Return 204 No Content (handled by status_code=204)
    return Response(status_code=204)


@app.get("/api/brokers/{broker_id}/health", response_model=BrokerHealthResponse)
async def check_broker_health(broker_id: str):
    """
    Check the health of a broker connection.

    Attempts to connect to the broker and verify authentication.
    Returns connection status, latency, and any error messages.

    Args:
        broker_id: Unique broker configuration ID

    Returns:
        BrokerHealthResponse with connection health details

    Raises:
        404: Broker not found
        503: Database not initialized
    """
    from backend.execution.broker_credentials import decrypt_credential
    from backend.execution.broker_factory import BrokerFactory, BrokerFactoryError

    if not app_state.sqlite_store:
        raise HTTPException(status_code=503, detail="Database not initialized")

    # Get broker config
    config = await _get_broker_config(app_state.sqlite_store, broker_id)
    if config is None:
        raise HTTPException(status_code=404, detail=f"Broker not found: {broker_id}")

    # Decrypt credentials
    try:
        api_key = decrypt_credential(config["api_key_encrypted"])
        api_secret = decrypt_credential(config["api_secret_encrypted"])
    except Exception as e:
        logger.error(f"Failed to decrypt credentials for broker {broker_id}: {e}")
        return BrokerHealthResponse(
            id=broker_id,
            name=config.get("name", "unknown"),
            broker_type=config.get("broker_type", "unknown"),
            is_connected=False,
            is_authenticated=False,
            error_message="Failed to decrypt credentials",
        )

    # Create broker instance (don't cache for health checks)
    try:
        factory = BrokerFactory()
        broker = factory.create_broker(
            broker_type=config["broker_type"],
            config={
                "api_key": api_key,
                "api_secret": api_secret,
                "base_url": config.get("base_url"),
            },
            use_cache=False,  # Don't cache health check instances
        )
    except BrokerFactoryError as e:
        logger.error(f"Failed to create broker instance: {e}")
        return BrokerHealthResponse(
            id=broker_id,
            name=config.get("name", "unknown"),
            broker_type=config.get("broker_type", "unknown"),
            is_connected=False,
            is_authenticated=False,
            error_message=str(e),
        )

    # Perform health check
    try:
        health = await broker.health_check()

        app_state.update_timestamp()

        return BrokerHealthResponse(
            id=broker_id,
            name=config.get("name", "unknown"),
            broker_type=config.get("broker_type", "unknown"),
            is_connected=health.is_connected,
            is_authenticated=health.is_authenticated,
            latency_ms=health.latency_ms,
            last_heartbeat=(
                datetime.fromtimestamp(health.last_heartbeat).isoformat()
                if health.last_heartbeat
                else None
            ),
            error_message=health.error_message,
        )
    except Exception as e:
        logger.error(f"Health check failed for broker {broker_id}: {e}")
        return BrokerHealthResponse(
            id=broker_id,
            name=config.get("name", "unknown"),
            broker_type=config.get("broker_type", "unknown"),
            is_connected=False,
            is_authenticated=False,
            error_message=f"Health check failed: {str(e)}",
        )
    finally:
        # Clean up broker connection
        try:
            await broker.disconnect()
        except Exception:
            pass


# ============================================================================
# Paper Trading Endpoints
# ============================================================================

# Global paper broker instance (lazy initialized)
_paper_broker: "PaperBroker | None" = None


async def _get_paper_broker() -> "PaperBroker":
    """
    Get or create the paper broker instance.

    Returns:
        Initialized PaperBroker instance

    Raises:
        HTTPException: If paper broker fails to initialize
    """
    global _paper_broker

    if _paper_broker is not None and _paper_broker._connected:
        return _paper_broker

    from backend.core.config import get_settings
    from backend.execution.brokers.paper_broker import PaperBroker

    settings = get_settings()

    # Use database path from app_state if available, otherwise default
    db_path = "data/system.db"
    if app_state.sqlite_store is not None:
        db_path = app_state.sqlite_store.db_path

    _paper_broker = PaperBroker(
        initial_balance=settings.paper_initial_balance,
        db_path=db_path,
        slippage_bps=settings.paper_slippage_bps,
        use_price_provider=False,  # Disable for API testing, can enable with config
        mock_price=settings.paper_mock_price,
        price_cache_ttl=settings.paper_price_cache_ttl,
    )

    connected = await _paper_broker.connect()
    if not connected:
        raise HTTPException(status_code=503, detail="Failed to initialize paper broker")

    return _paper_broker


# Note: /api/paper/account is defined in Mode-Specific Endpoints section
# using AccountInfoResponse for consistency with /api/live/account


@app.post("/api/paper/reset", response_model=PaperResetResponse)
async def reset_paper_account():
    """
    Reset paper trading account to initial state.

    Clears all positions, orders, and trade history, then restores
    the account to the initial $100,000 balance.

    Returns:
        PaperResetResponse: Confirmation with reset details
    """
    try:
        paper_broker = await _get_paper_broker()

        # Reset the account
        await paper_broker.reset_account()

        app_state.update_timestamp()

        return PaperResetResponse(
            message="Paper account reset successfully",
            account_id="PAPER-001",
            initial_balance=paper_broker.initial_balance,
            timestamp=datetime.now().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reset paper account: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset paper account: {str(e)}")


@app.get("/api/paper/trades", response_model=PaperTradeHistoryResponse)
async def get_paper_trades():
    """
    Get paper trading trade history.

    Returns all executed paper trades with fill details, slippage
    information, and realized P&L. Uses a format compatible with
    live broker trade history for UI consistency.

    Returns:
        PaperTradeHistoryResponse: List of paper trades
    """
    try:
        paper_broker = await _get_paper_broker()

        # Get trades
        trades = await paper_broker.get_trades()

        # Build trade responses
        trade_responses = [
            PaperTradeResponse(
                trade_id=trade.trade_id,
                order_id=trade.order_id,
                symbol=trade.symbol,
                side=trade.side.name,  # Convert OrderSide enum to string
                qty=trade.qty,
                price=trade.price,
                slippage=trade.slippage,
                timestamp=datetime.fromtimestamp(trade.timestamp).isoformat(),
                realized_pnl=trade.realized_pnl,
            )
            for trade in trades
        ]

        app_state.update_timestamp()

        return PaperTradeHistoryResponse(
            trades=trade_responses,
            total_count=len(trade_responses),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get paper trades: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get paper trades: {str(e)}")


# ============================================================================
# Trade Analytics Endpoints (Phase G)
# ============================================================================


@app.get("/api/trades/{trade_id}/chart-data", response_model=TradeChartDataResponse)
async def get_trade_chart_data(trade_id: int):
    """
    Get chart data for a specific trade.

    Returns the trade details along with candles before/after the trade
    and indicator overlays (KAMA, ATR bands).

    Args:
        trade_id: The trade ID to fetch chart data for

    Returns:
        TradeChartDataResponse with candles, indicators, and trade info
    """
    if not app_state.sqlite_store:
        raise HTTPException(status_code=503, detail="Database not initialized")

    # Fetch the trade
    all_trades = await app_state.sqlite_store.get_recent_trades(limit=1000)
    trade = next((t for t in all_trades if t.id == trade_id), None)

    if not trade:
        raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")

    # Convert trade to response model
    trade_response = TradeResponse(
        id=trade.id,
        symbol=trade.symbol,
        side=trade.side,
        entry_price=trade.entry_price,
        entry_time=trade.entry_time,
        exit_price=trade.exit_price,
        exit_time=trade.exit_time,
        shares=trade.shares,
        stop_loss=trade.stop_loss,
        take_profit=trade.take_profit,
        realized_pnl=trade.realized_pnl,
        status=trade.status,
        strategy=trade.strategy,
        regime=trade.regime,
        signal_reason=trade.signal_reason,
    )

    # Parse entry/exit times
    from datetime import timedelta

    try:
        entry_dt = datetime.fromisoformat(trade.entry_time.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        entry_dt = datetime.now() - timedelta(days=7)

    exit_dt = None
    if trade.exit_time:
        try:
            exit_dt = datetime.fromisoformat(trade.exit_time.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            pass

    # Calculate date range: 50 candles before entry, 20 after exit (or current)
    candles_before = 50
    candles_after = 20
    start_date = (entry_dt - timedelta(days=candles_before + 10)).strftime("%Y-%m-%d")
    end_date = (
        (exit_dt or datetime.now()) + timedelta(days=candles_after + 10)
    ).strftime("%Y-%m-%d")

    # Fetch historical data
    candles: list[CandleData] = []
    indicators: list[IndicatorData] = []

    try:
        from backend.data.yahoo_provider import YahooDataProvider

        provider = YahooDataProvider()
        hist_data = await provider.fetch_historical_data(
            trade.symbol, start_date, end_date, interval="1d"
        )

        if hist_data and hist_data.bars:
            import numpy as np
            from backend.computation.adaptive_ema import compute_kama
            from backend.computation.volatility import compute_atr

            closes = np.array([bar.close for bar in hist_data.bars], dtype=np.float64)
            highs = np.array([bar.high for bar in hist_data.bars], dtype=np.float64)
            lows = np.array([bar.low for bar in hist_data.bars], dtype=np.float64)

            # Compute indicators
            kama_values = compute_kama(closes, period=10, fast_span=2, slow_span=30)
            atr_values = compute_atr(highs, lows, closes, period=14)

            for i, bar in enumerate(hist_data.bars):
                timestamp = int(
                    datetime.strptime(hist_data.dates[i], "%Y-%m-%d").timestamp()
                )
                candles.append(
                    CandleData(
                        time=timestamp,
                        open=bar.open,
                        high=bar.high,
                        low=bar.low,
                        close=bar.close,
                        volume=bar.volume,
                    )
                )

                kama = float(kama_values[i]) if not np.isnan(kama_values[i]) else None
                atr = float(atr_values[i]) if not np.isnan(atr_values[i]) else None

                indicators.append(
                    IndicatorData(
                        time=timestamp,
                        kama=kama,
                        atr_upper=kama + (atr * 2.5) if kama and atr else None,
                        atr_lower=kama - (atr * 2.5) if kama and atr else None,
                    )
                )

    except Exception as e:
        logger.warning(f"Failed to fetch chart data for {trade.symbol}: {e}")

    # Find entry/exit indices in the candle array
    entry_ts = int(entry_dt.timestamp())
    exit_ts = int(exit_dt.timestamp()) if exit_dt else None

    entry_index = 0
    exit_index = None

    for i, candle in enumerate(candles):
        if candle.time <= entry_ts:
            entry_index = i
        if exit_ts and candle.time <= exit_ts:
            exit_index = i

    app_state.update_timestamp()

    return TradeChartDataResponse(
        trade=trade_response,
        candles=candles,
        indicators=indicators,
        entry_index=entry_index,
        exit_index=exit_index,
    )


async def _get_daily_summary_for_mode(
    mode: TradingMode,
    days: int = 30
) -> DailySummaryResponse:
    """
    Internal helper to get daily summary for a specific trading mode.

    Args:
        mode: TradingMode.LIVE or TradingMode.PAPER
        days: Number of days of history to include

    Returns:
        DailySummaryResponse with daily groups and totals
    """
    if not app_state.sqlite_store:
        raise HTTPException(status_code=503, detail="Database not initialized")

    # Fetch trades and positions for the specified mode
    trades_result = await app_state.sqlite_store.get_trades_paginated_for_mode(
        mode, page=1, page_size=1000
    )
    all_trades = trades_result[0]  # (trades, total_count)
    positions = await app_state.sqlite_store.get_positions_for_mode(mode)

    # Filter to closed trades within date range
    cutoff_date = datetime.now() - timedelta(days=days)
    closed_trades = []
    open_trades = []

    for trade in all_trades:
        if trade.status == TradeStatus.CLOSED:
            try:
                exit_dt = datetime.fromisoformat(
                    trade.exit_time.replace("Z", "+00:00")
                ) if trade.exit_time else None
                if exit_dt and exit_dt >= cutoff_date:
                    closed_trades.append(trade)
            except (ValueError, AttributeError):
                pass
        elif trade.status == TradeStatus.OPEN:
            open_trades.append(trade)

    # Group by date
    from collections import defaultdict

    daily_map: dict[str, list] = defaultdict(list)

    for trade in closed_trades:
        try:
            exit_dt = datetime.fromisoformat(trade.exit_time.replace("Z", "+00:00"))
            date_key = exit_dt.strftime("%Y-%m-%d")
            daily_map[date_key].append(trade)
        except (ValueError, AttributeError):
            pass

    # Build daily groups
    daily_groups: list[DailyTradeBreakdown] = []
    total_realized = 0.0
    total_wins = 0
    total_losses = 0

    for date_key in sorted(daily_map.keys(), reverse=True):
        trades_on_day = daily_map[date_key]
        day_pnl = sum(t.realized_pnl or 0 for t in trades_on_day)
        wins = sum(1 for t in trades_on_day if (t.realized_pnl or 0) > 0)
        losses = sum(1 for t in trades_on_day if (t.realized_pnl or 0) < 0)

        total_realized += day_pnl
        total_wins += wins
        total_losses += losses

        trade_responses = [
            TradeResponse(
                id=t.id,
                symbol=t.symbol,
                side=t.side,
                entry_price=t.entry_price,
                entry_time=t.entry_time,
                exit_price=t.exit_price,
                exit_time=t.exit_time,
                shares=t.shares,
                stop_loss=t.stop_loss,
                take_profit=t.take_profit,
                realized_pnl=t.realized_pnl,
                status=t.status,
                strategy=t.strategy,
                regime=t.regime,
                signal_reason=t.signal_reason,
            )
            for t in trades_on_day
        ]

        daily_groups.append(
            DailyTradeBreakdown(
                date=date_key,
                trades=trade_responses,
                trade_count=len(trades_on_day),
                realized_pnl=day_pnl,
                win_count=wins,
                loss_count=losses,
                daily_return_pct=0.0,  # Would need account equity to calculate
            )
        )

    # Calculate unrealized P&L from positions
    unrealized_pnl = sum(p.unrealized_pnl for p in positions)

    # Build position responses
    position_responses = [
        PositionResponse(
            id=p.id,
            symbol=p.symbol,
            side=p.side,
            shares=p.shares,
            entry_price=p.entry_price,
            current_price=p.current_price,
            unrealized_pnl=p.unrealized_pnl,
            stop_loss=p.stop_loss,
            take_profit=p.take_profit,
            entry_time=p.entry_time,
            updated_at=p.updated_at,
        )
        for p in positions
    ]

    # Calculate totals
    total_count = total_wins + total_losses
    win_rate = (total_wins / total_count * 100) if total_count > 0 else 0.0

    totals = TotalsSummary(
        closed_count=len(closed_trades),
        open_count=len(open_trades),
        realized_pnl=total_realized,
        unrealized_pnl=unrealized_pnl,
        total_pnl=total_realized + unrealized_pnl,
        total_return_pct=0.0,  # Would need initial capital to calculate
        win_rate=win_rate,
    )

    app_state.update_timestamp()

    return DailySummaryResponse(
        daily_groups=daily_groups,
        totals=totals,
        open_positions=position_responses,
    )


@app.get("/api/live/daily-summary", response_model=DailySummaryResponse)
async def get_live_daily_summary(
    days: int = Query(default=30, ge=1, le=365, description="Number of days to include")
):
    """
    Get LIVE trades grouped by date with aggregated daily metrics.

    Returns daily trade groups with P&L, win/loss counts, and totals.
    Also includes current open positions for live trading.

    Args:
        days: Number of days of history to include (default 30)

    Returns:
        DailySummaryResponse with daily groups and totals
    """
    return await _get_daily_summary_for_mode(TradingMode.LIVE, days)


@app.get("/api/paper/daily-summary", response_model=DailySummaryResponse)
async def get_paper_daily_summary(
    days: int = Query(default=30, ge=1, le=365, description="Number of days to include")
):
    """
    Get PAPER trades grouped by date with aggregated daily metrics.

    Returns daily trade groups with P&L, win/loss counts, and totals.
    Also includes current open positions for paper trading.

    Args:
        days: Number of days of history to include (default 30)

    Returns:
        DailySummaryResponse with daily groups and totals
    """
    return await _get_daily_summary_for_mode(TradingMode.PAPER, days)


async def _get_analysis_for_mode(
    mode: TradingMode,
    benchmark: str = "VTI"
) -> LiveAnalysisResponse:
    """
    Internal helper to get trading analysis for a specific mode.

    Args:
        mode: TradingMode.LIVE or TradingMode.PAPER
        benchmark: Symbol to use as benchmark

    Returns:
        LiveAnalysisResponse with comprehensive analysis data
    """
    if not app_state.sqlite_store:
        raise HTTPException(status_code=503, detail="Database not initialized")

    # Fetch trades and positions for the specified mode
    trades_result = await app_state.sqlite_store.get_trades_paginated_for_mode(
        mode, page=1, page_size=1000
    )
    all_trades = trades_result[0]
    positions = await app_state.sqlite_store.get_positions_for_mode(mode)

    # Get closed trades sorted by exit time
    closed_trades = [t for t in all_trades if t.status == TradeStatus.CLOSED]
    closed_trades.sort(
        key=lambda t: t.exit_time or "", reverse=False
    )

    # Assume initial capital (could be stored in config)
    initial_capital = 10000.0

    # Build equity curve from trades
    equity_curve: list[EquityCurvePoint] = []
    daily_breakdown: list[DailyBreakdown] = []

    if closed_trades:
        from collections import defaultdict

        # Group trades by date
        daily_pnls: dict[str, float] = defaultdict(float)
        daily_counts: dict[str, int] = defaultdict(int)

        for trade in closed_trades:
            try:
                exit_dt = datetime.fromisoformat(trade.exit_time.replace("Z", "+00:00"))
                date_key = exit_dt.strftime("%Y-%m-%d")
                daily_pnls[date_key] += trade.realized_pnl or 0
                daily_counts[date_key] += 1
            except (ValueError, AttributeError):
                pass

        # Build equity curve
        sorted_dates = sorted(daily_pnls.keys())
        cumulative_pnl = 0.0

        for date_key in sorted_dates:
            cumulative_pnl += daily_pnls[date_key]
            equity = initial_capital + cumulative_pnl
            return_pct = (cumulative_pnl / initial_capital) * 100

            equity_curve.append(
                EquityCurvePoint(
                    date=date_key,
                    equity=equity,
                    benchmark_equity=initial_capital,  # Placeholder
                    daily_pnl=daily_pnls[date_key],
                    cumulative_pnl=cumulative_pnl,
                    cumulative_return_pct=return_pct,
                    benchmark_return_pct=0.0,  # Placeholder
                )
            )

            daily_breakdown.append(
                DailyBreakdown(
                    date=date_key,
                    pnl=daily_pnls[date_key],
                    return_pct=(daily_pnls[date_key] / initial_capital) * 100,
                    trade_count=daily_counts[date_key],
                    cumulative_pnl=cumulative_pnl,
                )
            )

    # Fetch benchmark data for comparison
    try:
        from backend.data.yahoo_provider import YahooDataProvider

        if equity_curve:
            provider = YahooDataProvider()
            start_date = equity_curve[0].date
            end_date = equity_curve[-1].date

            benchmark_data = await provider.fetch_historical_data(
                benchmark, start_date, end_date, interval="1d"
            )

            if benchmark_data and benchmark_data.bars:
                initial_benchmark = benchmark_data.bars[0].close
                benchmark_map = {
                    benchmark_data.dates[i]: bar.close
                    for i, bar in enumerate(benchmark_data.bars)
                }

                for point in equity_curve:
                    if point.date in benchmark_map:
                        bench_price = benchmark_map[point.date]
                        bench_return = (
                            (bench_price - initial_benchmark) / initial_benchmark
                        ) * 100
                        point.benchmark_return_pct = bench_return
                        point.benchmark_equity = initial_capital * (1 + bench_return / 100)
    except Exception as e:
        logger.warning(f"Failed to fetch benchmark data: {e}")

    # Calculate risk metrics
    import numpy as np

    pnls = [t.realized_pnl or 0 for t in closed_trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    # Simple risk calculations
    sharpe = 0.0
    sortino = 0.0
    calmar = 0.0
    max_dd = 0.0
    max_dd_pct = 0.0

    if pnls:
        returns = np.array(pnls) / initial_capital
        avg_return = np.mean(returns)
        std_return = np.std(returns) if len(returns) > 1 else 0.001

        sharpe = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0.0

        # Sortino (downside deviation)
        negative_returns = returns[returns < 0]
        downside_std = np.std(negative_returns) if len(negative_returns) > 1 else 0.001
        sortino = (avg_return / downside_std) * np.sqrt(252) if downside_std > 0 else 0.0

        # Max drawdown
        equity_values = [initial_capital]
        for pnl in pnls:
            equity_values.append(equity_values[-1] + pnl)

        peak = equity_values[0]
        for eq in equity_values:
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = (dd / peak) * 100 if peak > 0 else 0

        # Calmar
        total_return = (equity_values[-1] - initial_capital) / initial_capital
        calmar = (total_return / (max_dd_pct / 100)) if max_dd_pct > 0 else 0.0

    win_rate = (len(wins) / len(pnls) * 100) if pnls else 0.0
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = abs(np.mean(losses)) if losses else 0.0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else 0.0

    risk_metrics = RiskMetrics(
        sharpe_ratio=round(sharpe, 2),
        sortino_ratio=round(sortino, 2),
        calmar_ratio=round(calmar, 2),
        max_drawdown=max_dd,
        max_drawdown_pct=round(max_dd_pct, 2),
        win_rate=round(win_rate, 1),
        profit_factor=round(profit_factor, 2),
        avg_win=round(avg_win, 2),
        avg_loss=round(avg_loss, 2),
    )

    # Current equity
    current_equity = initial_capital + sum(pnls) + sum(p.unrealized_pnl for p in positions)

    app_state.update_timestamp()

    return LiveAnalysisResponse(
        equity_curve=equity_curve,
        risk_metrics=risk_metrics,
        daily_breakdown=daily_breakdown,
        initial_capital=initial_capital,
        current_equity=current_equity,
        benchmark_symbol=benchmark,
        trading_days=len(equity_curve),
    )


@app.get("/api/live/analysis", response_model=LiveAnalysisResponse)
async def get_live_analysis(
    benchmark: str = Query(default="VTI", description="Benchmark symbol for comparison")
):
    """
    Get LIVE trading analysis with performance metrics and benchmark comparison.

    Returns equity curve, risk metrics, and daily breakdown with benchmark overlay.

    Args:
        benchmark: Symbol to use as benchmark (default VTI)

    Returns:
        LiveAnalysisResponse with comprehensive analysis data
    """
    return await _get_analysis_for_mode(TradingMode.LIVE, benchmark)


@app.get("/api/paper/analysis", response_model=LiveAnalysisResponse)
async def get_paper_analysis(
    benchmark: str = Query(default="VTI", description="Benchmark symbol for comparison")
):
    """
    Get PAPER trading analysis with performance metrics and benchmark comparison.

    Returns equity curve, risk metrics, and daily breakdown with benchmark overlay.

    Args:
        benchmark: Symbol to use as benchmark (default VTI)

    Returns:
        LiveAnalysisResponse with comprehensive analysis data
    """
    return await _get_analysis_for_mode(TradingMode.PAPER, benchmark)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("=" * 60)
    logger.info("FluxHero API Server")
    logger.info("=" * 60)
    logger.info("Starting server on http://localhost:8000")
    logger.info("API docs available at http://localhost:8000/docs")
    logger.info("=" * 60)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
