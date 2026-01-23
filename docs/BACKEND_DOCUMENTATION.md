# FluxHero Backend Documentation

A comprehensive guide to the FluxHero quantitative trading platform backend architecture, features, and usage.

---

## Table of Contents

1. [Overview](#overview)
2. [Technology Stack](#technology-stack)
3. [Project Structure](#project-structure)
4. [API Endpoints](#api-endpoints)
5. [WebSocket Functionality](#websocket-functionality)
6. [Trading Strategies](#trading-strategies)
7. [Regime Detection](#regime-detection)
8. [Signal Generation](#signal-generation)
9. [Backtesting Engine](#backtesting-engine)
10. [Risk Management](#risk-management)
11. [Data Storage](#data-storage)
12. [Technical Indicators](#technical-indicators)
13. [Configuration](#configuration)
14. [Authentication & Security](#authentication--security)
15. [Running the Backend](#running-the-backend)
16. [Key File Reference](#key-file-reference)

---

## Overview

FluxHero is a production-grade quantitative trading platform designed for algorithmic trading with:

- **Dual-mode strategy engine** (Trend-Following + Mean Reversion)
- **Real-time regime detection** using ADX and R² metrics
- **Sophisticated risk management** with circuit breakers
- **High-performance computation** via Numba JIT compilation
- **WebSocket live price streaming** with CSV replay for development
- **Comprehensive backtesting** with realistic execution simulation

---

## Technology Stack

### Core Dependencies

| Category | Libraries | Version |
|----------|-----------|---------|
| **Web Framework** | FastAPI, Uvicorn | 0.109.0+, 0.27.0+ |
| **Performance** | Numba, NumPy, Pandas | 0.59.0+, 1.26.0+, 2.2.0+ |
| **Async I/O** | httpx, websockets, aiofiles | 0.27.0+, 12.0+, 23.2.0+ |
| **Data Storage** | PyArrow, SQLite | 15.0.0+, stdlib |
| **Quantitative** | QuantStats, SciPy, scikit-learn, yfinance | 0.0.62+, 1.12.0+, 1.4.0+, 0.2.36+ |
| **Validation** | Pydantic, pydantic-settings | 2.6.0+, 2.1.0+ |
| **Testing** | pytest, pytest-asyncio, pytest-cov | 8.0.0+, 0.23.0+, 4.1.0+ |
| **Code Quality** | Ruff, mypy | 0.2.0+, 1.8.0+ |

### Key Characteristics

- **JIT-compiled indicators**: <100ms for 10k candles
- **Async SQLite writes**: <5ms non-blocking
- **Backtest performance**: <10 seconds for 1 year of minute data (100k+ candles)

---

## Project Structure

```
backend/
├── api/
│   ├── server.py          # FastAPI app, REST endpoints, WebSocket
│   ├── auth.py            # Token validation and authentication
│   └── rate_limit.py      # Rate limiting middleware
├── core/
│   ├── config.py          # Pydantic settings from environment
│   └── logging_config.py  # Structured logging setup
├── strategy/
│   ├── dual_mode.py       # Trend-following + mean reversion strategies
│   ├── regime_detector.py # ADX/R² regime classification
│   ├── signal_generator.py # Signal creation with explanations
│   ├── backtest_strategy.py # Strategy adapter for backtesting API
│   └── noise_filter.py    # Signal quality validation
├── backtesting/
│   ├── engine.py          # Backtesting execution engine
│   └── metrics.py         # Performance metrics calculation
├── execution/
│   ├── order_manager.py   # Order tracking and chase logic
│   ├── broker_interface.py # Broker API abstraction
│   └── position_sizer.py  # Risk-based position sizing
├── computation/
│   └── indicators.py      # EMA, RSI, ATR calculations (Numba)
├── storage/
│   └── sqlite_store.py    # Trade/position persistence
├── risk/
│   ├── kill_switch.py     # Circuit breaker and drawdown monitoring
│   └── position_limits.py # Portfolio exposure limits
├── data/
│   ├── fetcher.py         # Market data REST/WebSocket client (Alpaca)
│   └── yahoo_fetcher.py   # Yahoo Finance historical data fetcher
└── maintenance/
    └── daily_ops.py       # Daily rollover and cleanup
```

---

## API Endpoints

### REST Endpoints

| Endpoint | Method | Description | Response |
|----------|--------|-------------|----------|
| `/` | GET | API info and endpoint discovery | JSON with available endpoints |
| `/api/positions` | GET | All open positions with unrealized P&L | `List[PositionResponse]` |
| `/api/trades` | GET | Paginated trade history with filtering | `TradeHistoryResponse` |
| `/api/account` | GET | Account equity, cash, buying power, daily P&L | `AccountInfoResponse` |
| `/api/status` | GET | System health check (heartbeat) | `SystemStatusResponse` |
| `/api/backtest` | POST | Execute backtest with configuration | `BacktestResultResponse` |
| `/api/test/candles` | GET | Test endpoint for CSV candle data | List of candles (dev only) |
| `/health` | GET | Health check with database connectivity | Health status JSON |
| `/metrics` | GET | Prometheus-compatible metrics | Prometheus text format |

### Response Models

#### PositionResponse
```python
{
    "id": "pos_123",
    "symbol": "SPY",
    "side": 1,           # 1 = LONG, -1 = SHORT
    "shares": 100,
    "entry_price": 450.00,
    "current_price": 452.50,
    "unrealized_pnl": 250.00,
    "stop_loss": 445.00,
    "take_profit": 460.00,
    "entry_time": "2024-01-22T10:30:00Z"
}
```

#### TradeHistoryResponse
```python
{
    "trades": [...],
    "total": 150,
    "page": 1,
    "page_size": 20,
    "total_pages": 8
}
```

#### AccountInfoResponse
```python
{
    "equity": 105000.00,
    "cash": 50000.00,
    "buying_power": 100000.00,
    "daily_pnl": 1250.00,
    "daily_pnl_pct": 1.20
}
```

#### BacktestResultResponse
```python
{
    "total_return": 0.15,
    "sharpe_ratio": 1.45,
    "max_drawdown": 0.08,
    "win_rate": 0.58,
    "total_trades": 142,
    "equity_curve": [...],
    "trades": [...]
}
```

---

## WebSocket Functionality

### Endpoint: `/ws/prices`

Real-time price streaming with authentication and CSV replay support.

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/prices');
ws.onopen = () => {
    ws.send(JSON.stringify({
        type: 'auth',
        token: 'Bearer YOUR_AUTH_TOKEN'
    }));
};
```

### Message Format

```json
{
    "type": "price_update",
    "symbol": "SPY",
    "timestamp": "2024-01-22",
    "open": 450.00,
    "high": 451.50,
    "low": 449.80,
    "close": 450.25,
    "volume": 1000000,
    "replay_index": 42,
    "total_rows": 504
}
```

### Features

- **CSV Replay Mode**: Loads test data (SPY, AAPL, MSFT daily candles) and replays iteratively
- **Synthetic Fallback**: Generates synthetic prices if CSV unavailable
- **Update Frequency**: 2 seconds per symbol (CSV), 5 seconds (synthetic)
- **Multi-Symbol Support**: Streams SPY, AAPL, MSFT concurrently
- **Auto-Reconnect**: Client management with automatic cleanup on disconnect

---

## Trading Strategies

### Dual-Mode Strategy Engine

**File**: `backend/strategy/dual_mode.py`

The system runs two complementary strategies that adapt to market conditions:

### Trend-Following Mode

| Parameter | Value |
|-----------|-------|
| **Entry Condition** | ADX > 25 AND R² > 0.6 |
| **Long Entry** | Price > KAMA + 0.5 × ATR |
| **Short Entry** | Price < KAMA - 0.5 × ATR |
| **Exit (Long)** | Price < KAMA - 0.3 × ATR |
| **Exit (Short)** | Price > KAMA + 0.3 × ATR |
| **Stop Loss** | ATR-based trailing stop (2.5× multiplier) |
| **Risk Per Trade** | 1.0% of account |

### Mean Reversion Mode

| Parameter | Value |
|-----------|-------|
| **Entry Condition** | ADX < 20 AND R² < 0.4 |
| **Long Entry** | RSI < 30 AND Price at Lower Bollinger Band |
| **Short Entry** | RSI > 70 AND Price at Upper Bollinger Band |
| **Exit** | Price returns to 20-SMA or RSI flips |
| **Stop Loss** | Fixed 3% stop |
| **Risk Per Trade** | 0.75% of account |

### Neutral (Blended) Mode

| Parameter | Value |
|-----------|-------|
| **Condition** | Neither trend nor mean-reversion conditions met |
| **Signal Rule** | Both strategies must agree |
| **Weighting** | 50/50 split |
| **Risk Adjustment** | 0.7× base risk |

### Key Functions

```python
# Trend-following signals (Numba JIT)
generate_trend_following_signals(prices, kama, atr) -> signals

# Mean reversion signals (Numba JIT)
generate_mean_reversion_signals(prices, rsi, bb_upper, bb_lower, sma) -> signals

# Trailing stop calculation
calculate_trailing_stop(entry_price, current_price, atr, side) -> stop_price

# Position sizing based on risk
calculate_position_size(account_value, risk_pct, entry, stop) -> shares

# Blend signals when both strategies agree
blend_signals(trend_signal, mr_signal, trend_weight, mr_weight) -> signal
```

### Performance Tracking

The strategy tracks per-mode metrics:
- Win rate
- Sharpe ratio
- Max drawdown
- Total return

After 20+ trades per mode, weights are dynamically rebalanced based on performance (range: 0.5 to 1.0).

---

## Regime Detection

### Regime Classification System

**File**: `backend/strategy/regime_detector.py`

### Regime Types

| Regime | Code | Conditions |
|--------|------|------------|
| **STRONG_TREND** | 2 | ADX > 25 AND R² > 0.6 |
| **NEUTRAL** | 1 | Transition states |
| **MEAN_REVERSION** | 0 | ADX < 20 AND R² < 0.4 |

### Volatility Classification

| State | Code | Condition |
|-------|------|-----------|
| **HIGH** | 2 | ATR > 1.5 × ATR_MA (elevated) |
| **NORMAL** | 1 | Between thresholds |
| **LOW** | 0 | ATR < 0.7 × ATR_MA (calm) |

### Key Algorithms

#### ADX Calculation (Average Directional Index)
```python
def calculate_adx(high, low, close, period=14):
    """
    Returns ADX value 0-100 scale
    - ADX > 25: Strong trend
    - ADX < 20: Weak trend / ranging market
    """
```

#### R² Linear Regression
```python
def calculate_linear_regression(prices, period=20):
    """
    Returns R² coefficient (0 to 1)
    - R² > 0.7: Strong linear trend
    - R² < 0.4: Mean-reverting behavior
    """
```

### Regime Persistence

Requires **3-bar confirmation** before switching regimes to avoid whipsaws.

### Multi-Asset Correlation

```python
def calculate_correlation_matrix(returns_dict):
    """
    Returns correlation matrix for portfolio assets
    Used for correlation-based position sizing
    """
```

---

## Signal Generation

### SignalExplanation System

**File**: `backend/strategy/signal_generator.py`

Every signal includes a detailed explanation for transparency and debugging.

### SignalExplanation Dataclass

```python
@dataclass
class SignalExplanation:
    signal_type: str          # "BUY", "SELL", "HOLD"
    symbol: str
    price: float
    timestamp: datetime

    # Strategy context
    strategy_mode: str        # "TREND", "MEAN_REVERSION", "BLENDED"
    regime: str               # "STRONG_TREND", "NEUTRAL", "MEAN_REVERSION"
    volatility_state: str     # "HIGH", "NORMAL", "LOW"

    # Indicator values
    atr: float
    kama: float
    rsi: float
    adx: float
    r_squared: float

    # Risk parameters
    risk_amount: float        # Dollar amount at risk
    risk_pct: float           # Percentage of account
    stop_loss: float
    position_size: int

    # Trigger explanation
    entry_trigger: str        # e.g., "KAMA crossover"
    noise_filter_passed: bool
    volume_validated: bool
```

### Output Formats

#### Multi-Line Format
```
BUY SPY @ $420.50
Reason: Volatility (ATR=3.2, High) + KAMA crossover (Price > KAMA+0.5×ATR)
Regime: STRONG_TREND (ADX=32, R²=0.81)
Risk: $1,000 (1% account), Stop: $415.00
```

#### Compact Format (UI Tooltips)
```
TREND BUY: KAMA↑ | ADX=32 | Risk=1%
```

#### JSON Format (Database Storage)
```python
signal.to_dict()  # Returns JSON-serializable dictionary
```

---

## Backtesting Engine

### Configuration

**File**: `backend/backtesting/engine.py`

```python
@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission_per_share: float = 0.005   # Alpaca-like
    slippage_pct: float = 0.0001          # 0.01% slippage
    impact_threshold: float = 0.1         # 10% of avg volume
    impact_penalty_pct: float = 0.0005    # Extra 0.05% for large orders
    risk_free_rate: float = 0.04          # 4% for Sharpe calculation
```

### Execution Flow

1. **Fill pending orders** (next-bar fill logic)
2. **Check stop loss / take profit**
3. **Update position value and equity**
4. **Generate signals from strategy**
5. **Track equity curve**

### Order Types

```python
@dataclass
class Order:
    symbol: str
    side: int              # 1 = BUY, -1 = SELL
    quantity: int
    order_type: str        # "MARKET" or "LIMIT"
    limit_price: float     # Optional for limit orders
    commission: float      # Calculated on fill
    slippage: float        # Applied on fill
```

### Position Tracking

```python
@dataclass
class Position:
    symbol: str
    side: int              # 1 = LONG, -1 = SHORT
    shares: int
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
```

### Trade Recording

```python
@dataclass
class Trade:
    symbol: str
    side: int
    entry_price: float
    exit_price: float
    shares: int
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    commission: float
    slippage: float
```

### Performance Metrics

**File**: `backend/backtesting/metrics.py`

| Metric | Formula |
|--------|---------|
| **Sharpe Ratio** | (mean_return - risk_free_rate) / std_dev (annualized) |
| **Max Drawdown** | Peak-to-trough decline percentage |
| **Win Rate** | Winning trades / total trades |
| **Avg Win/Loss** | Mean profit per win / mean loss per loss |
| **Total Return** | (Final equity - Initial) / Initial |

### Yahoo Finance Data Integration

**File**: `backend/data/yahoo_fetcher.py`

The backtesting system can fetch real historical data from Yahoo Finance:

```python
from backend.data.yahoo_fetcher import YahooFinanceFetcher

fetcher = YahooFinanceFetcher()
data = fetcher.fetch_historical_data(
    symbol="SPY",
    start_date="2024-01-01",
    end_date="2024-12-31",
    interval="1d"
)

# Returns:
# - data['bars']: numpy array (N, 5) with [open, high, low, close, volume]
# - data['timestamps']: numpy array of Unix timestamps
# - data['dates']: list of date strings for display
```

### Strategy Adapter for Backtesting

**File**: `backend/strategy/backtest_strategy.py`

The `DualModeBacktestStrategy` class provides a pre-computed strategy adapter:

```python
from backend.strategy.backtest_strategy import DualModeBacktestStrategy

strategy = DualModeBacktestStrategy(
    bars=bars,                      # OHLCV numpy array
    initial_capital=100000.0,       # Starting capital
    strategy_mode="DUAL",           # "TREND", "MEAN_REVERSION", or "DUAL"
    trend_risk_pct=0.01,            # 1% risk for trend trades
    mr_risk_pct=0.0075,             # 0.75% risk for mean-reversion trades
)

# Use with BacktestEngine
state = engine.run(
    bars=bars,
    strategy_func=strategy.get_orders,
    symbol="SPY",
    timestamps=timestamps,
)
```

**Strategy Modes:**

| Mode | Description |
|------|-------------|
| `TREND` | Only use trend-following signals (KAMA crossover) |
| `MEAN_REVERSION` | Only use mean-reversion signals (RSI + Bollinger Bands) |
| `DUAL` | Automatically switch based on regime detection (default) |

### Running a Backtest via API

```bash
curl -X POST http://localhost:8000/api/backtest \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "SPY",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "initial_capital": 100000,
    "strategy_mode": "DUAL"
  }'
```

**Response includes:**
- `total_return_pct`, `sharpe_ratio`, `max_drawdown_pct`, `win_rate`
- `num_trades`, `avg_win_loss_ratio`
- `equity_curve` array for charting
- `success_criteria_met` (Sharpe > 0.8, DD < 25%, Win Rate > 45%)

---

## Risk Management

### Kill Switch / Circuit Breaker

**File**: `backend/risk/kill_switch.py`

### Drawdown Levels

| Level | Drawdown | Actions |
|-------|----------|---------|
| **NORMAL** | < 15% | Normal trading |
| **WARNING** | 15-20% | Reduce sizes 50%, tighten stops |
| **CRITICAL** | > 20% | Close all positions, disable trading |

### Trading Status

```python
class TradingStatus(Enum):
    ACTIVE = 0      # Normal trading
    REDUCED = 1     # Reduced size mode (15% DD)
    DISABLED = 2    # Trading disabled (20% DD)
```

### Key Functions

```python
# Track equity and drawdown
class EquityTracker:
    def update(self, current_equity: float) -> DrawdownLevel
    def get_current_drawdown(self) -> float
    def get_peak_equity(self) -> float

# Check drawdown severity
def check_drawdown_level(drawdown: float) -> DrawdownLevel

# Get position size multiplier based on status
def get_position_size_multiplier(status: TradingStatus) -> float
# Returns: 1.0 (normal), 0.5 (reduced), 0.0 (disabled)

# Get stop loss multiplier
def get_stop_loss_multiplier(status: TradingStatus) -> float
# Returns: 2.5 (normal), 2.0 (reduced/tighter)

# Check if new positions allowed
def can_open_new_position(status: TradingStatus) -> bool
```

### Risk Metrics Monitoring

```python
@dataclass
class RiskMetrics:
    current_drawdown: float
    peak_equity: float
    current_equity: float
    total_exposure_pct: float
    num_positions: int
    correlation_alert: bool
    trading_status: TradingStatus
```

### Position Limits

| Limit | Default | Description |
|-------|---------|-------------|
| Max Position Size | 20% | Single position as % of account |
| Max Total Exposure | 50% | Portfolio-level max exposure |
| Max Open Positions | 5 | Maximum concurrent positions |
| Correlation Threshold | 0.7 | High correlation alert level |
| Correlation Size Reduction | 50% | Size reduction for correlated assets |

---

## Data Storage

### SQLite Store

**File**: `backend/storage/sqlite_store.py`

### Schema

#### Trades Table
```sql
CREATE TABLE trades (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    side INTEGER NOT NULL,        -- 1=LONG, -1=SHORT
    entry_price REAL NOT NULL,
    entry_time TEXT NOT NULL,
    exit_price REAL,
    exit_time TEXT,
    shares INTEGER NOT NULL,
    stop_loss REAL,
    take_profit REAL,
    realized_pnl REAL,
    status TEXT NOT NULL,         -- OPEN, CLOSED, CANCELLED
    strategy TEXT,                -- TREND, MEAN_REVERSION
    regime TEXT,                  -- STRONG_TREND, MEAN_REVERSION, NEUTRAL
    signal_reason TEXT,           -- Human-readable
    signal_explanation TEXT,      -- JSON
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

#### Positions Table
```sql
CREATE TABLE positions (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    side INTEGER NOT NULL,
    shares INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    current_price REAL,
    unrealized_pnl REAL,
    stop_loss REAL,
    take_profit REAL,
    entry_time TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

#### Settings Table
```sql
CREATE TABLE settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    description TEXT,
    updated_at TEXT NOT NULL
);
```

### Features

- **Async writes**: Non-blocking via asyncio.Queue
- **30-day retention**: Daily rollover for trade history
- **Real-time updates**: Supports WebSocket data logging
- **Target size**: <100 MB after 1 year

---

## Technical Indicators

### Implemented Indicators

**File**: `backend/computation/indicators.py`

All indicators are Numba JIT-compiled for performance.

#### Exponential Moving Average (EMA)
```python
@njit
def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Exponential Moving Average
    - More responsive to recent prices
    - Used for KAMA baseline
    """
```

#### Relative Strength Index (RSI)
```python
@njit
def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    RSI oscillator (0-100 scale)
    - RSI < 30: Oversold (mean-reversion buy)
    - RSI > 70: Overbought (mean-reversion sell)
    """
```

#### Average True Range (ATR)
```python
@njit
def calculate_atr(high: np.ndarray, low: np.ndarray,
                  close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Volatility measure using Wilder's smoothing
    - Used for stop loss calculation
    - Used for position sizing
    """
```

### Noise Filter

**File**: `backend/strategy/noise_filter.py`

```python
def calculate_spread_to_volatility_ratio(spread: float, atr: float) -> float:
    """
    Rejects signals when spread/ATR ratio is too high
    Indicates noisy/illiquid conditions
    """

def validate_volume(current_volume: float, avg_volume: float,
                   threshold: float = 0.5) -> bool:
    """
    Confirms signal with volume validation
    Requires current volume > threshold × average
    """
```

---

## Configuration

### Environment Variables

**File**: `backend/core/config.py`

All settings use the `FLUXHERO_` prefix.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `FLUXHERO_AUTH_SECRET` | str | dev-secret | API authentication secret |
| `FLUXHERO_API_TITLE` | str | FluxHero API | OpenAPI title |
| `FLUXHERO_API_VERSION` | str | v1.0.0 | API version |
| `FLUXHERO_CORS_ORIGINS` | list | localhost:3000,3001 | CORS whitelist |
| `FLUXHERO_ALPACA_API_URL` | str | paper-api.alpaca.markets | Broker endpoint |
| `FLUXHERO_ALPACA_API_KEY` | str | (empty) | Broker API key |
| `FLUXHERO_ALPACA_API_SECRET` | str | (empty) | Broker API secret |
| `FLUXHERO_MAX_RISK_PCT_TREND` | float | 0.01 | Trend trade risk (1%) |
| `FLUXHERO_MAX_RISK_PCT_MEAN_REV` | float | 0.0075 | Mean-rev trade risk (0.75%) |
| `FLUXHERO_MAX_POSITION_SIZE_PCT` | float | 0.20 | Max position size (20%) |
| `FLUXHERO_MAX_TOTAL_EXPOSURE_PCT` | float | 0.50 | Max portfolio exposure (50%) |
| `FLUXHERO_MAX_OPEN_POSITIONS` | int | 5 | Max concurrent positions |
| `FLUXHERO_CORRELATION_THRESHOLD` | float | 0.7 | Correlation alert level |
| `FLUXHERO_CORRELATION_SIZE_REDUCTION` | float | 0.50 | Size reduction for correlation |
| `FLUXHERO_TREND_STOP_ATR_MULTIPLIER` | float | 2.5 | ATR multiplier for stops |
| `FLUXHERO_MEAN_REV_STOP_PCT` | float | 0.03 | Fixed stop for mean-rev (3%) |
| `FLUXHERO_CACHE_DIR` | str | data/cache | Cache directory |
| `FLUXHERO_DEFAULT_TIMEFRAME` | str | 1h | Default data timeframe |
| `FLUXHERO_INITIAL_CANDLES` | int | 500 | Initial candles to fetch |

### Example .env File

```env
# Authentication
FLUXHERO_AUTH_SECRET=your-production-secret-here

# Broker Configuration
FLUXHERO_ALPACA_API_URL=https://paper-api.alpaca.markets
FLUXHERO_ALPACA_API_KEY=your-api-key
FLUXHERO_ALPACA_API_SECRET=your-api-secret

# Risk Management
FLUXHERO_MAX_RISK_PCT_TREND=0.01
FLUXHERO_MAX_RISK_PCT_MEAN_REV=0.0075
FLUXHERO_MAX_POSITION_SIZE_PCT=0.20
FLUXHERO_MAX_TOTAL_EXPOSURE_PCT=0.50
FLUXHERO_MAX_OPEN_POSITIONS=5

# CORS (comma-separated)
FLUXHERO_CORS_ORIGINS=http://localhost:3000,http://localhost:3001
```

---

## Authentication & Security

### Token Authentication

**File**: `backend/api/auth.py`

```python
def validate_token(token: str, secret: str) -> bool:
    """
    Validates Bearer token using constant-time comparison
    Resistant to timing attacks via secrets.compare_digest()
    """

def validate_websocket_auth(headers: dict) -> bool:
    """
    Validates WebSocket connection authentication
    Extracts token from Authorization header
    """
```

### Rate Limiting

**File**: `backend/api/rate_limit.py`

| Parameter | Default |
|-----------|---------|
| Max Requests | 100 |
| Time Window | 60 seconds |
| Per | IP Address |
| Excluded Paths | /health, /metrics |

Returns HTTP 429 with `Retry-After` header when exceeded.

### Security Features

1. **Constant-time comparison**: Timing attack resistant token validation
2. **CORS whitelist**: Origin validation
3. **Rate limiting**: Sliding window per-IP
4. **Environment secrets**: Loaded from .env with production warnings
5. **Pydantic validation**: Schema enforcement on all inputs
6. **Structured logging**: JSON logs with exception tracking

---

## Running the Backend

### Development

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env

# Edit .env with your settings
nano .env

# Run development server
uvicorn backend.api.server:app --reload --host 0.0.0.0 --port 8000
```

### Production

```bash
# Run with Gunicorn + Uvicorn workers
gunicorn backend.api.server:app \
    -w 4 \
    -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=backend

# Run in parallel
pytest -n auto

# Run specific test file
pytest backend/tests/test_strategy.py
```

### API Documentation

Once running, access:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

---

## Key File Reference

| Component | File | Key Lines |
|-----------|------|-----------|
| API Server | `backend/api/server.py` | 336-360 (app init), 459-829 (endpoints), 836-968 (WebSocket) |
| Config | `backend/core/config.py` | 28-235 (Settings class) |
| Dual-Mode Strategy | `backend/strategy/dual_mode.py` | 35-123 (trend), 181-260 (mean-rev), 409-599 (performance) |
| Backtest Strategy | `backend/strategy/backtest_strategy.py` | Full file (strategy adapter for API) |
| Regime Detector | `backend/strategy/regime_detector.py` | 177-264 (ADX), 362-421 (classification), 631-702 (pipeline) |
| Signal Generator | `backend/strategy/signal_generator.py` | 57-277 (SignalExplanation), 279-500 (SignalGenerator) |
| Backtesting Engine | `backend/backtesting/engine.py` | 60-77 (config), 196-327 (run method) |
| Performance Metrics | `backend/backtesting/metrics.py` | 26-150+ (Sharpe, drawdown, returns) |
| SQLite Store | `backend/storage/sqlite_store.py` | 102-193 (schema, async ops) |
| Kill Switch | `backend/risk/kill_switch.py` | 138-445 (circuit breaker), 453-643 (monitoring) |
| Authentication | `backend/api/auth.py` | 39-128 (token validation) |
| Rate Limiting | `backend/api/rate_limit.py` | 30-233 (RateLimiter, middleware) |
| Logging | `backend/core/logging_config.py` | 27-259 (formatters, setup) |
| Indicators | `backend/computation/indicators.py` | 21-150+ (EMA, RSI, ATR) |
| Noise Filter | `backend/strategy/noise_filter.py` | 28-80 (spread-to-vol ratio) |
| Data Fetcher (Alpaca) | `backend/data/fetcher.py` | 1-100+ (async REST, WebSocket) |
| Yahoo Finance Fetcher | `backend/data/yahoo_fetcher.py` | Full file (historical data from Yahoo Finance) |
| Order Manager | `backend/execution/order_manager.py` | 26-100+ (ManagedOrder, chase logic) |

---

## Usage Examples

### Running a Backtest via API

The backtest endpoint fetches real data from Yahoo Finance and runs the dual-mode strategy:

```bash
curl -X POST http://localhost:8000/api/backtest \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "SPY",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "initial_capital": 100000,
    "strategy_mode": "DUAL"
  }'
```

**Strategy Mode Options:**
- `"DUAL"` - Automatically switches between trend-following and mean-reversion based on regime
- `"TREND"` - Only uses trend-following signals (KAMA crossover + ATR)
- `"MEAN_REVERSION"` - Only uses mean-reversion signals (RSI + Bollinger Bands)

**Response Example:**
```json
{
  "symbol": "SPY",
  "start_date": "2024-01-01",
  "end_date": "2024-12-31",
  "initial_capital": 100000,
  "final_equity": 112500,
  "total_return": 12500,
  "total_return_pct": 12.5,
  "sharpe_ratio": 1.45,
  "max_drawdown": 5200,
  "max_drawdown_pct": 5.2,
  "win_rate": 0.58,
  "num_trades": 24,
  "avg_win_loss_ratio": 1.8,
  "success_criteria_met": true,
  "equity_curve": [100000, 100250, ...],
  "timestamps": ["2024-01-02", "2024-01-03", ...]
}
```

### Connecting to WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/prices');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'auth',
    token: 'Bearer your-token'
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'price_update') {
    console.log(`${data.symbol}: ${data.close}`);
  }
};
```

### Getting Account Info

```bash
curl http://localhost:8000/api/account \
  -H "Authorization: Bearer your-token"
```

### Fetching Trade History

```bash
curl "http://localhost:8000/api/trades?page=1&page_size=20&status=CLOSED" \
  -H "Authorization: Bearer your-token"
```

---

## Architecture Highlights

1. **Performance-First Design**: Numba JIT compilation for all numerical operations
2. **Async I/O**: Non-blocking database writes and WebSocket handling
3. **Dual Strategy Adaptation**: Automatic regime detection and strategy switching
4. **Comprehensive Risk Management**: Circuit breakers, position limits, correlation monitoring
5. **Production-Ready**: Structured logging, Prometheus metrics, health checks
6. **Developer-Friendly**: CSV replay mode, detailed signal explanations, extensive configuration

---

*Documentation generated for FluxHero Backend v1.0.0*
