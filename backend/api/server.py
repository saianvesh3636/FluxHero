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

# Import storage modules
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.api.auth import validate_websocket_auth
from backend.api.rate_limit import RateLimitMiddleware
from backend.backtesting.engine import BacktestConfig, BacktestEngine
from backend.backtesting.metrics import PerformanceMetrics
from backend.core.config import get_settings
from backend.storage.sqlite_store import (
    SQLiteStore,
    TradeStatus,
)

# ============================================================================
# Logger Configuration
# ============================================================================

logger = logging.getLogger(__name__)


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
    logger.info("WebSocket connections closed", extra={"client_count": len(app_state.websocket_clients)})

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

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware to log all HTTP requests and responses.

    Logs:
    - Request method, path, client IP
    - Response status code, processing time
    - Request/response body size
    """
    # Generate request ID for tracking
    request_id = f"{int(time.time() * 1000)}-{id(request)}"

    # Log incoming request
    start_time = time.time()
    logger.info(
        "Incoming request",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query_params": str(request.query_params),
            "client_ip": request.client.host if request.client else "unknown",
        }
    )

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
            }
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
            exc_info=True
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
        }
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
        response.append(PositionResponse(
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
        ))

    app_state.update_timestamp()
    return response


@app.get("/api/trades", response_model=TradeHistoryResponse)
async def get_trades(
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(default=20, ge=1, le=100, description="Trades per page"),
    status: str | None = Query(default=None, description="Filter by status (OPEN, CLOSED, CANCELLED)"),
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
        status_map = {"OPEN": TradeStatus.OPEN, "CLOSED": TradeStatus.CLOSED, "CANCELLED": TradeStatus.CANCELLED}
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
        trades_response.append(TradeResponse(
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
        ))

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
    closed_trades = [t for t in recent_trades if t.status == TradeStatus.CLOSED and t.realized_pnl is not None]
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
    initial_capital_setting = await app_state.sqlite_store.get_setting("initial_capital", default="10000.0")
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

    Args:
        config: Backtest configuration (symbol, dates, capital, strategy params)

    Returns:
        BacktestResultResponse: Backtest performance metrics and equity curve
    """
    # Create synthetic data for demonstration (in production, fetch real data)
    # This is a placeholder - in real implementation, fetch from API or cache

    # Parse dates
    try:
        start_dt = datetime.strptime(config.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(config.end_date, "%Y-%m-%d")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")

    # Calculate number of trading days
    num_days = (end_dt - start_dt).days
    if num_days <= 0:
        raise HTTPException(status_code=400, detail="End date must be after start date")

    # Generate synthetic OHLCV data for demonstration
    # In production, use backend.data.fetcher to fetch real data
    np.random.seed(42)
    num_bars = num_days  # Daily bars

    # Simulate trending price data
    returns = np.random.normal(0.0005, 0.02, num_bars)  # 0.05% daily return, 2% volatility
    prices = 100.0 * np.exp(np.cumsum(returns))

    opens = prices * (1 + np.random.uniform(-0.01, 0.01, num_bars))
    highs = np.maximum(opens, prices) * (1 + np.random.uniform(0, 0.02, num_bars))
    lows = np.minimum(opens, prices) * (1 - np.random.uniform(0, 0.02, num_bars))
    closes = prices
    volumes = np.random.randint(1000000, 10000000, num_bars)

    # Generate timestamps
    timestamps = [start_dt.strftime("%Y-%m-%d")]
    for i in range(1, num_bars):
        timestamps.append((start_dt + np.timedelta64(i, 'D')).strftime("%Y-%m-%d"))

    # Create BacktestConfig
    bt_config = BacktestConfig(
        initial_capital=config.initial_capital,
        commission_per_share=config.commission_per_share,
        slippage_pct=config.slippage_pct,
    )

    # Initialize backtest engine
    engine = BacktestEngine(config=bt_config)

    # Generate simple buy-and-hold signals for demonstration
    # In production, use strategy engine to generate signals
    signals = np.zeros(num_bars, dtype=np.int32)
    signals[0] = 1  # Buy on first bar
    signals[-1] = -1  # Sell on last bar

    # Run backtest
    results = engine.run(
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
        volumes=volumes,
        signals=signals,
        stop_losses=None,
        take_profits=None,
    )

    # Calculate performance metrics
    metrics = PerformanceMetrics.calculate_all_metrics(
        equity_curve=results["equity_curve"],
        trades=results["trades"],
        initial_capital=config.initial_capital,
        num_bars=num_bars,
    )

    # Check success criteria
    success_criteria_met = PerformanceMetrics.check_success_criteria(
        sharpe_ratio=metrics["sharpe_ratio"],
        max_drawdown_pct=metrics["max_drawdown_pct"],
        win_rate=metrics["win_rate"],
    )

    app_state.update_timestamp()

    return BacktestResultResponse(
        symbol=config.symbol,
        start_date=config.start_date,
        end_date=config.end_date,
        initial_capital=config.initial_capital,
        final_equity=results["equity_curve"][-1],
        total_return=metrics["total_return"],
        total_return_pct=metrics["total_return_pct"],
        sharpe_ratio=metrics["sharpe_ratio"],
        max_drawdown=metrics["max_drawdown"],
        max_drawdown_pct=metrics["max_drawdown_pct"],
        win_rate=metrics["win_rate"],
        num_trades=len(results["trades"]),
        avg_win_loss_ratio=metrics["avg_win_loss_ratio"],
        success_criteria_met=success_criteria_met,
        equity_curve=results["equity_curve"].tolist(),
        timestamps=timestamps,
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
            extra={"client": websocket.client}
        )
        await websocket.close(code=4001, reason="Authentication failed")
        return

    await websocket.accept()
    app_state.websocket_clients.append(websocket)
    app_state.data_feed_active = True

    logger.info("WebSocket client connected", extra={"total_clients": len(app_state.websocket_clients)})

    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "message": "WebSocket connection established",
            "timestamp": datetime.now().isoformat(),
        })

        # Simulate price updates (in production, connect to real data feed)
        # For demonstration, send synthetic price updates every 5 seconds
        while True:
            # Generate synthetic price update
            price_update = PriceUpdate(
                symbol="SPY",
                price=round(np.random.uniform(400.0, 450.0), 2),
                timestamp=datetime.now().isoformat(),
                volume=np.random.randint(100000, 1000000),
            )

            # Send update to client
            await websocket.send_json(price_update.model_dump())
            app_state.update_timestamp()

            # Wait 5 seconds before next update
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

        logger.info("WebSocket client removed", extra={"total_clients": len(app_state.websocket_clients)})


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
    add_metric("fluxhero_requests_total", app_state.request_count, "Total number of HTTP requests", "counter")

    # Request latency percentiles
    if app_state.request_latencies:
        latencies = np.array(app_state.request_latencies)
        p50 = float(np.percentile(latencies, 50))
        p90 = float(np.percentile(latencies, 90))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))

        add_metric("fluxhero_request_latency_p50_ms", p50, "Request latency 50th percentile in milliseconds")
        add_metric("fluxhero_request_latency_p90_ms", p90, "Request latency 90th percentile in milliseconds")
        add_metric("fluxhero_request_latency_p95_ms", p95, "Request latency 95th percentile in milliseconds")
        add_metric("fluxhero_request_latency_p99_ms", p99, "Request latency 99th percentile in milliseconds")

    # Order/Trade counts from database
    if app_state.sqlite_store:
        try:
            # Get recent trades to calculate metrics (limit to last 1000 for performance)
            all_trades = await app_state.sqlite_store.get_recent_trades(limit=1000)
            total_trades = len(all_trades)
            add_metric("fluxhero_orders_total", total_trades, "Total number of orders/trades", "counter")

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

                add_metric("fluxhero_drawdown_percent", current_drawdown_pct, "Current drawdown percentage")
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
            lines.append('# HELP fluxhero_requests_by_path_total Requests by endpoint path')
            lines.append('# TYPE fluxhero_requests_by_path_total counter')
            lines.append(f'fluxhero_requests_by_path_total{{path="{path}"}} {count}')
            lines.append("")

    # WebSocket connections
    add_metric("fluxhero_websocket_connections", len(app_state.websocket_clients), "Active WebSocket connections")

    # Data feed status
    add_metric("fluxhero_data_feed_active", 1.0 if app_state.data_feed_active else 0.0, "Data feed active status (1=active, 0=inactive)")

    # Return as plain text with Prometheus content type
    return Response(content="\n".join(lines), media_type="text/plain; version=0.0.4")


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
