# FluxHero API Documentation

Version: 1.0.0

## Overview

The FluxHero API provides REST endpoints and WebSocket connections for interacting with the adaptive quant trading system. The API supports position management, trade history, account information, system monitoring, and backtesting capabilities.

**Base URL**: `http://localhost:8000`

**Interactive Documentation**: `http://localhost:8000/docs` (Swagger UI)

---

## Table of Contents

1. [Authentication](#authentication)
2. [REST Endpoints](#rest-endpoints)
   - [GET /](#get-)
   - [GET /api/positions](#get-apipositions)
   - [GET /api/trades](#get-apitrades)
   - [GET /api/account](#get-apiaccount)
   - [GET /api/status](#get-apistatus)
   - [POST /api/backtest](#post-apibacktest)
   - [GET /health](#get-health)
3. [WebSocket Endpoints](#websocket-endpoints)
   - [WS /ws/prices](#ws-wsprices)
4. [Data Models](#data-models)
5. [Error Handling](#error-handling)
6. [Rate Limits](#rate-limits)

---

## Authentication

Currently, the FluxHero API does not require authentication. This is suitable for local development and single-user deployments. For production deployments with multiple users, implement OAuth2 or API key authentication.

**CORS Configuration**: The API allows requests from:
- `http://localhost:3000` (Next.js dev server)
- `http://localhost:3001`
- `http://127.0.0.1:3000`

---

## REST Endpoints

### GET /

Get API information and available endpoints.

**Request:**
```bash
GET /
```

**Response:** `200 OK`
```json
{
  "name": "FluxHero API",
  "version": "1.0.0",
  "status": "active",
  "endpoints": {
    "positions": "/api/positions",
    "trades": "/api/trades",
    "account": "/api/account",
    "status": "/api/status",
    "backtest": "/api/backtest",
    "websocket": "/ws/prices"
  }
}
```

---

### GET /api/positions

Retrieve all currently open positions with unrealized P&L.

**Request:**
```bash
GET /api/positions
```

**Response:** `200 OK`
```json
[
  {
    "id": 1,
    "symbol": "SPY",
    "side": 1,
    "shares": 100,
    "entry_price": 450.25,
    "current_price": 452.80,
    "unrealized_pnl": 255.00,
    "stop_loss": 447.50,
    "take_profit": 455.00,
    "entry_time": "2024-01-15T09:30:00",
    "updated_at": "2024-01-15T14:30:00"
  }
]
```

**Field Descriptions:**
- `id`: Position ID (integer, nullable)
- `symbol`: Ticker symbol (string)
- `side`: Position direction (1 = LONG, -1 = SHORT)
- `shares`: Number of shares (integer)
- `entry_price`: Entry price per share (float)
- `current_price`: Current market price (float)
- `unrealized_pnl`: Unrealized profit/loss (float)
- `stop_loss`: Stop loss price (float)
- `take_profit`: Take profit price (float, nullable)
- `entry_time`: Entry timestamp (ISO 8601 string)
- `updated_at`: Last update timestamp (ISO 8601 string)

**Error Responses:**
- `503 Service Unavailable`: Database not initialized

---

### GET /api/trades

Retrieve trade history with pagination and optional filtering.

**Request:**
```bash
GET /api/trades?page=1&page_size=20&status=CLOSED
```

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `page` | integer | No | 1 | Page number (1-indexed, min: 1) |
| `page_size` | integer | No | 20 | Trades per page (min: 1, max: 100) |
| `status` | string | No | null | Filter by status: `OPEN`, `CLOSED`, `CANCELLED` |

**Response:** `200 OK`
```json
{
  "trades": [
    {
      "id": 42,
      "symbol": "SPY",
      "side": 1,
      "entry_price": 445.20,
      "entry_time": "2024-01-14T10:15:00",
      "exit_price": 448.50,
      "exit_time": "2024-01-14T15:30:00",
      "shares": 50,
      "stop_loss": 442.00,
      "take_profit": 450.00,
      "realized_pnl": 165.00,
      "status": 2,
      "strategy": "TREND_FOLLOWING",
      "regime": "STRONG_TREND",
      "signal_reason": "Price crossed above KAMA + 0.5*ATR. Volatility: NORMAL. ADX: 28.5"
    }
  ],
  "total_count": 150,
  "page": 1,
  "page_size": 20,
  "total_pages": 8
}
```

**Field Descriptions:**
- `trades`: Array of trade objects
- `total_count`: Total number of trades matching filter (integer)
- `page`: Current page number (integer)
- `page_size`: Trades per page (integer)
- `total_pages`: Total number of pages (integer)

**Trade Object Fields:**
- `id`: Trade ID (integer, nullable)
- `symbol`: Ticker symbol (string)
- `side`: Trade direction (1 = LONG, -1 = SHORT)
- `entry_price`: Entry price per share (float)
- `entry_time`: Entry timestamp (ISO 8601 string)
- `exit_price`: Exit price per share (float, nullable)
- `exit_time`: Exit timestamp (ISO 8601 string, nullable)
- `shares`: Number of shares (integer)
- `stop_loss`: Stop loss price (float)
- `take_profit`: Take profit price (float, nullable)
- `realized_pnl`: Realized profit/loss (float, nullable)
- `status`: Trade status code (0 = OPEN, 1 = CANCELLED, 2 = CLOSED)
- `strategy`: Strategy used (string: `TREND_FOLLOWING`, `MEAN_REVERSION`, `DUAL`)
- `regime`: Market regime at entry (string: `STRONG_TREND`, `MEAN_REVERSION`, `NEUTRAL`)
- `signal_reason`: Human-readable explanation of trade signal (string)

**Error Responses:**
- `400 Bad Request`: Invalid status filter
- `503 Service Unavailable`: Database not initialized

---

### GET /api/account

Get account information including equity, P&L, and position count.

**Request:**
```bash
GET /api/account
```

**Response:** `200 OK`
```json
{
  "equity": 10525.50,
  "cash": 5200.00,
  "buying_power": 10400.00,
  "total_pnl": 525.50,
  "daily_pnl": 125.75,
  "num_positions": 3
}
```

**Field Descriptions:**
- `equity`: Total account value (initial capital + realized P&L + unrealized P&L)
- `cash`: Available cash (equity - value of open positions)
- `buying_power`: Leverage-adjusted buying power (2x cash for margin accounts)
- `total_pnl`: Total profit/loss (realized + unrealized)
- `daily_pnl`: Profit/loss from trades closed today
- `num_positions`: Number of open positions

**Calculation Details:**
```
equity = initial_capital + total_realized_pnl + total_unrealized_pnl
cash = equity - sum(position.shares * position.current_price)
buying_power = cash * 2.0  (assumes margin account)
total_pnl = total_realized_pnl + total_unrealized_pnl
daily_pnl = sum(realized_pnl for trades closed today)
```

**Error Responses:**
- `503 Service Unavailable`: Database not initialized

---

### GET /api/status

Get system health status and connectivity information (heartbeat endpoint).

**Request:**
```bash
GET /api/status
```

**Response:** `200 OK`
```json
{
  "status": "ACTIVE",
  "uptime_seconds": 3600.5,
  "last_update": "2024-01-15T14:30:00.123456",
  "websocket_connected": true,
  "data_feed_active": true,
  "message": "System operating normally"
}
```

**Field Descriptions:**
- `status`: System status (string: `ACTIVE`, `DELAYED`, `OFFLINE`)
- `uptime_seconds`: Time since server started (float)
- `last_update`: Timestamp of last activity (ISO 8601 string)
- `websocket_connected`: Whether WebSocket clients are connected (boolean)
- `data_feed_active`: Whether data feed is active (boolean)
- `message`: Human-readable status message (string)

**Status Determination:**
- `ACTIVE`: Last update within 60 seconds
- `DELAYED`: Last update between 60-300 seconds ago
- `OFFLINE`: No updates for >300 seconds

**Use Cases:**
- Frontend heartbeat monitoring (poll every 5-10 seconds)
- Health check for load balancers
- System monitoring dashboards

---

### POST /api/backtest

Execute a backtest with the provided configuration.

**Request:**
```bash
POST /api/backtest
Content-Type: application/json

{
  "symbol": "SPY",
  "start_date": "2023-01-01",
  "end_date": "2023-12-31",
  "initial_capital": 10000.0,
  "commission_per_share": 0.005,
  "slippage_pct": 0.0001,
  "strategy_mode": "DUAL"
}
```

**Request Body Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `symbol` | string | Yes | - | Ticker symbol (e.g., "SPY") |
| `start_date` | string | Yes | - | Start date (YYYY-MM-DD format) |
| `end_date` | string | Yes | - | End date (YYYY-MM-DD format) |
| `initial_capital` | float | No | 10000.0 | Starting capital ($) |
| `commission_per_share` | float | No | 0.005 | Commission per share ($) |
| `slippage_pct` | float | No | 0.0001 | Slippage percentage (0.01% = 0.0001) |
| `strategy_mode` | string | No | "DUAL" | Strategy mode: `TREND`, `MEAN_REVERSION`, `DUAL` |

**Response:** `200 OK`
```json
{
  "symbol": "SPY",
  "start_date": "2023-01-01",
  "end_date": "2023-12-31",
  "initial_capital": 10000.0,
  "final_equity": 11250.50,
  "total_return": 1250.50,
  "total_return_pct": 12.51,
  "sharpe_ratio": 1.25,
  "max_drawdown": 850.00,
  "max_drawdown_pct": 8.5,
  "win_rate": 0.52,
  "num_trades": 45,
  "avg_win_loss_ratio": 1.8,
  "success_criteria_met": true,
  "equity_curve": [10000.0, 10050.0, 10025.0, ...],
  "timestamps": ["2023-01-01", "2023-01-02", "2023-01-03", ...]
}
```

**Response Field Descriptions:**
- `symbol`: Backtested symbol
- `start_date`: Backtest start date
- `end_date`: Backtest end date
- `initial_capital`: Starting capital
- `final_equity`: Ending equity value
- `total_return`: Absolute return ($)
- `total_return_pct`: Percentage return (%)
- `sharpe_ratio`: Risk-adjusted return metric (target: >0.8)
- `max_drawdown`: Maximum equity drawdown ($)
- `max_drawdown_pct`: Maximum equity drawdown (%, target: <25%)
- `win_rate`: Fraction of winning trades (target: >45%)
- `num_trades`: Total number of trades executed
- `avg_win_loss_ratio`: Average win size / average loss size
- `success_criteria_met`: Whether backtest meets success criteria (boolean)
- `equity_curve`: Array of equity values over time
- `timestamps`: Array of timestamps corresponding to equity curve

**Success Criteria:**
- Sharpe Ratio > 0.8
- Max Drawdown < 25%
- Win Rate > 45%

**Error Responses:**
- `400 Bad Request`: Invalid date format or end date before start date
- `503 Service Unavailable`: Backtest engine unavailable

**Notes:**
- Current implementation uses synthetic data for demonstration
- Production version should fetch real historical data from data provider
- Backtests run synchronously; large date ranges may take several seconds

---

### GET /health

Simple health check endpoint for monitoring and load balancers.

**Request:**
```bash
GET /health
```

**Response:** `200 OK`
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T14:30:00.123456"
}
```

**Use Cases:**
- Kubernetes liveness/readiness probes
- Load balancer health checks
- Uptime monitoring services

---

## WebSocket Endpoints

### WS /ws/prices

Real-time price update stream via WebSocket.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/prices');

ws.onopen = () => {
  console.log('Connected to price feed');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Price update:', data);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Disconnected from price feed');
};
```

**Initial Connection Message:**
```json
{
  "type": "connection",
  "status": "connected",
  "message": "WebSocket connection established",
  "timestamp": "2024-01-15T14:30:00.123456"
}
```

**Price Update Messages:**
```json
{
  "symbol": "SPY",
  "price": 452.80,
  "timestamp": "2024-01-15T14:30:05.123456",
  "volume": 250000
}
```

**Message Fields:**
- `symbol`: Ticker symbol (string)
- `price`: Current price (float, rounded to 2 decimals)
- `timestamp`: Update timestamp (ISO 8601 string)
- `volume`: Volume for this update (integer, nullable)

**Update Frequency:**
- Current implementation: Every 5 seconds (demonstration)
- Production implementation: Real-time as market data arrives

**Connection Management:**
- Server tracks all connected clients
- Auto-cleanup on disconnect
- Data feed marked active when clients connected
- Heartbeat monitoring recommended on client side

**Example Python Client:**
```python
import asyncio
import websockets
import json

async def subscribe_prices():
    uri = "ws://localhost:8000/ws/prices"
    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            print(f"Price update: {data}")

asyncio.run(subscribe_prices())
```

**Example React Client:**
```typescript
import { useEffect, useState } from 'react';

function usePriceFeed() {
  const [price, setPrice] = useState(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/prices');

    ws.onopen = () => setConnected(true);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.price) {
        setPrice(data);
      }
    };

    ws.onclose = () => setConnected(false);

    return () => ws.close();
  }, []);

  return { price, connected };
}
```

---

## Data Models

### Position

```typescript
interface Position {
  id: number | null;
  symbol: string;
  side: number;  // 1 = LONG, -1 = SHORT
  shares: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  stop_loss: number;
  take_profit: number | null;
  entry_time: string;  // ISO 8601
  updated_at: string;  // ISO 8601
}
```

### Trade

```typescript
interface Trade {
  id: number | null;
  symbol: string;
  side: number;  // 1 = LONG, -1 = SHORT
  entry_price: number;
  entry_time: string;  // ISO 8601
  exit_price: number | null;
  exit_time: string | null;  // ISO 8601
  shares: number;
  stop_loss: number;
  take_profit: number | null;
  realized_pnl: number | null;
  status: number;  // 0 = OPEN, 1 = CANCELLED, 2 = CLOSED
  strategy: string;  // "TREND_FOLLOWING" | "MEAN_REVERSION" | "DUAL"
  regime: string;  // "STRONG_TREND" | "MEAN_REVERSION" | "NEUTRAL"
  signal_reason: string;
}
```

### Account Info

```typescript
interface AccountInfo {
  equity: number;
  cash: number;
  buying_power: number;
  total_pnl: number;
  daily_pnl: number;
  num_positions: number;
}
```

### System Status

```typescript
interface SystemStatus {
  status: "ACTIVE" | "DELAYED" | "OFFLINE";
  uptime_seconds: number;
  last_update: string;  // ISO 8601
  websocket_connected: boolean;
  data_feed_active: boolean;
  message: string;
}
```

### Backtest Request

```typescript
interface BacktestRequest {
  symbol: string;
  start_date: string;  // YYYY-MM-DD
  end_date: string;    // YYYY-MM-DD
  initial_capital?: number;  // default: 10000.0
  commission_per_share?: number;  // default: 0.005
  slippage_pct?: number;  // default: 0.0001
  strategy_mode?: "TREND" | "MEAN_REVERSION" | "DUAL";  // default: "DUAL"
}
```

### Backtest Result

```typescript
interface BacktestResult {
  symbol: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  final_equity: number;
  total_return: number;
  total_return_pct: number;
  sharpe_ratio: number;
  max_drawdown: number;
  max_drawdown_pct: number;
  win_rate: number;
  num_trades: number;
  avg_win_loss_ratio: number;
  success_criteria_met: boolean;
  equity_curve: number[];
  timestamps: string[];
}
```

### Price Update

```typescript
interface PriceUpdate {
  symbol: string;
  price: number;
  timestamp: string;  // ISO 8601
  volume?: number | null;
}
```

---

## Error Handling

### Standard Error Response

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common HTTP Status Codes

| Code | Description | Common Causes |
|------|-------------|---------------|
| 200 | OK | Request successful |
| 400 | Bad Request | Invalid parameters, date format errors |
| 503 | Service Unavailable | Database not initialized, service down |
| 500 | Internal Server Error | Unexpected server error |

### Error Examples

**Invalid Date Format:**
```json
{
  "detail": "Invalid date format: time data '2023-13-01' does not match format '%Y-%m-%d'"
}
```

**Invalid Status Filter:**
```json
{
  "detail": "Invalid status: INVALID_STATUS"
}
```

**Database Not Initialized:**
```json
{
  "detail": "Database not initialized"
}
```

---

## Rate Limits

Currently, the FluxHero API does not enforce rate limits. For production deployments, consider implementing:

- **REST endpoints**: 100 requests/minute per IP
- **WebSocket connections**: 5 concurrent connections per IP
- **Backtest endpoint**: 10 requests/hour (backtests are computationally expensive)

Recommended rate limiting middleware: `slowapi` (FastAPI rate limiter)

---

## Best Practices

### Polling vs WebSocket

**Use REST polling for:**
- Account info (poll every 30-60 seconds)
- System status (poll every 10 seconds)
- Trade history (user-initiated or every 5 minutes)

**Use WebSocket for:**
- Real-time price updates
- Live position P&L updates

### Error Handling

Always implement retry logic with exponential backoff:

```python
import asyncio

async def fetch_with_retry(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)  # 1s, 2s, 4s
```

### Pagination

For large datasets, always use pagination:

```python
# Fetch all trades
all_trades = []
page = 1
while True:
    response = await client.get(f"/api/trades?page={page}&page_size=100")
    data = response.json()
    all_trades.extend(data["trades"])

    if page >= data["total_pages"]:
        break
    page += 1
```

### WebSocket Reconnection

Implement automatic reconnection for WebSocket:

```javascript
function connectWebSocket() {
  const ws = new WebSocket('ws://localhost:8000/ws/prices');

  ws.onclose = () => {
    console.log('Disconnected, reconnecting in 5s...');
    setTimeout(connectWebSocket, 5000);
  };

  return ws;
}
```

---

## Testing

### Using cURL

**Get positions:**
```bash
curl http://localhost:8000/api/positions
```

**Get trade history:**
```bash
curl "http://localhost:8000/api/trades?page=1&page_size=10&status=CLOSED"
```

**Run backtest:**
```bash
curl -X POST http://localhost:8000/api/backtest \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "SPY",
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "initial_capital": 10000
  }'
```

### Using Python httpx

```python
import httpx
import asyncio

async def test_api():
    async with httpx.AsyncClient() as client:
        # Get positions
        response = await client.get("http://localhost:8000/api/positions")
        print(response.json())

        # Get account info
        response = await client.get("http://localhost:8000/api/account")
        print(response.json())

        # Run backtest
        response = await client.post(
            "http://localhost:8000/api/backtest",
            json={
                "symbol": "SPY",
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
            }
        )
        print(response.json())

asyncio.run(test_api())
```

### Interactive Documentation

FastAPI provides auto-generated interactive API documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

Use these interfaces to test endpoints directly from your browser.

---

## Deployment Notes

### Starting the Server

```bash
# Development mode
cd fluxhero/backend/api
python server.py

# Production mode with Gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker server:app --bind 0.0.0.0:8000

# Production mode with Uvicorn
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4
```

### Environment Variables

```bash
# Optional configuration
export FLUXHERO_DB_PATH="/path/to/system.db"
export FLUXHERO_CORS_ORIGINS="http://localhost:3000,https://yourdomain.com"
export FLUXHERO_LOG_LEVEL="info"
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY fluxhero /app/fluxhero
EXPOSE 8000

CMD ["uv", "run", "uvicorn", "fluxhero.backend.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Support

For issues, questions, or feature requests, please refer to the main FluxHero documentation or create an issue in the project repository.

**Version**: 1.0.0
**Last Updated**: January 2024
**Maintained By**: FluxHero Team
