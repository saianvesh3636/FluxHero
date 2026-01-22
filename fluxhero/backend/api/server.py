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
from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

# Import storage modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.storage.sqlite_store import (
    SQLiteStore,
    TradeStatus,
)
from backend.backtesting.engine import BacktestEngine, BacktestConfig
from backend.backtesting.metrics import PerformanceMetrics


# ============================================================================
# Pydantic Models (Request/Response schemas)
# ============================================================================

class PositionResponse(BaseModel):
    """Response model for a position"""
    id: Optional[int]
    symbol: str
    side: int  # 1 = LONG, -1 = SHORT
    shares: int
    entry_price: float
    current_price: float
    unrealized_pnl: float
    stop_loss: float
    take_profit: Optional[float] = None
    entry_time: str
    updated_at: str


class TradeResponse(BaseModel):
    """Response model for a trade"""
    id: Optional[int]
    symbol: str
    side: int
    entry_price: float
    entry_time: str
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    shares: int
    stop_loss: float
    take_profit: Optional[float] = None
    realized_pnl: Optional[float] = None
    status: int
    strategy: str
    regime: str
    signal_reason: str


class TradeHistoryResponse(BaseModel):
    """Paginated trade history response"""
    trades: List[TradeResponse]
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
    equity_curve: List[float]
    timestamps: List[str]


class PriceUpdate(BaseModel):
    """WebSocket price update message"""
    symbol: str
    price: float
    timestamp: str
    volume: Optional[int] = None


# ============================================================================
# Global State Management
# ============================================================================

class AppState:
    """Global application state"""
    def __init__(self):
        self.sqlite_store: Optional[SQLiteStore] = None
        self.websocket_clients: List[WebSocket] = []
        self.start_time: datetime = datetime.now()
        self.last_update: datetime = datetime.now()
        self.data_feed_active: bool = False

    def update_timestamp(self):
        """Update last activity timestamp"""
        self.last_update = datetime.now()


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
    print("🚀 Starting FluxHero API server...")

    # Initialize SQLite store
    db_path = Path(__file__).parent.parent.parent.parent / "data" / "system.db"
    app_state.sqlite_store = SQLiteStore(db_path=str(db_path))
    await app_state.sqlite_store.initialize()
    print(f"✓ SQLite store initialized: {db_path}")

    # Mark data feed as inactive (will be activated when WebSocket connects)
    app_state.data_feed_active = False
    app_state.start_time = datetime.now()

    print("✓ FluxHero API server ready")

    yield

    # Shutdown
    print("🛑 Shutting down FluxHero API server...")

    # Close SQLite store
    if app_state.sqlite_store:
        await app_state.sqlite_store.close()
        print("✓ SQLite store closed")

    # Close all WebSocket connections
    for client in app_state.websocket_clients:
        await client.close()
    print("✓ WebSocket connections closed")

    print("✓ FluxHero API server stopped")


# ============================================================================
# FastAPI App Initialization
# ============================================================================

app = FastAPI(
    title="FluxHero API",
    description="REST API for FluxHero adaptive quant trading system",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "http://localhost:3001",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


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


@app.get("/api/positions", response_model=List[PositionResponse])
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
    status: Optional[str] = Query(default=None, description="Filter by status (OPEN, CLOSED, CANCELLED)"),
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
    """
    await websocket.accept()
    app_state.websocket_clients.append(websocket)
    app_state.data_feed_active = True

    print(f"✓ WebSocket client connected. Total clients: {len(app_state.websocket_clients)}")

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
        print("✓ WebSocket client disconnected")
    except Exception as e:
        print(f"✗ WebSocket error: {e}")
    finally:
        # Remove client from list
        if websocket in app_state.websocket_clients:
            app_state.websocket_clients.remove(websocket)

        # Mark data feed as inactive if no clients
        if len(app_state.websocket_clients) == 0:
            app_state.data_feed_active = False

        print(f"✓ Client removed. Total clients: {len(app_state.websocket_clients)}")


# ============================================================================
# Health Check Endpoint
# ============================================================================

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("FluxHero API Server")
    print("=" * 60)
    print("Starting server on http://localhost:8000")
    print("API docs available at http://localhost:8000/docs")
    print("=" * 60)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
