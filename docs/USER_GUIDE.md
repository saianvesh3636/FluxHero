# FluxHero User Guide

**Version**: 1.0.0
**Last Updated**: 2026-01-21

## Table of Contents

1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Running the System](#running-the-system)
6. [Using the Dashboard](#using-the-dashboard)
7. [Monitoring](#monitoring)
8. [Common Operations](#common-operations)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

---

## Introduction

FluxHero is an adaptive retail quant trading system designed for solo developers. It combines high-performance Python computation (via Numba JIT), adaptive trading strategies, and a modern React dashboard for monitoring and analysis.

**Key Features**:
- Dual-mode strategy engine (trend-following + mean-reversion)
- Adaptive indicators (KAMA, volatility-adjusted smoothing)
- Market regime detection (trending vs ranging)
- Comprehensive risk management with circuit breakers
- Real-time monitoring dashboard
- Professional backtesting with realistic slippage/commission modeling

**Target User**: Solo retail developer trading US equities/ETFs

---

## System Requirements

### Hardware
- **CPU**: Modern multi-core processor (4+ cores recommended)
- **RAM**: 4GB minimum, 8GB+ recommended
- **Storage**: 2GB for system + additional space for market data cache
- **Network**: Stable internet connection for market data APIs

### Software
- **Operating System**: Linux, macOS, or Windows 10+
- **Python**: 3.10 or higher
- **Node.js**: 18.0 or higher
- **npm**: 9.0 or higher

### Market Data Provider
- Alpaca Markets account (free tier available)
- API keys (KEY_ID and SECRET_KEY)

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd project
```

### 2. Backend Setup

#### Create Python Virtual Environment

```bash
# Navigate to the project root
cd project

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

#### Install Backend Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies**:
- `numba`: JIT compilation for high-performance computations
- `fastapi`: REST API framework
- `httpx`: Async HTTP client for data fetching
- `quantstats`: Backtesting performance metrics
- `pandas`, `numpy`: Data manipulation
- `pyarrow`: Parquet storage for market data cache

#### Verify Installation

```bash
python -c "import numba; print(f'Numba {numba.__version__} installed successfully')"
```

### 3. Frontend Setup

#### Install Frontend Dependencies

```bash
cd fluxhero/frontend
npm install
cd ../..
```

**Key Dependencies**:
- `next`: React framework
- `typescript`: Type-safe development
- `lightweight-charts`: TradingView-style charts

### 4. Create Required Directories

```bash
mkdir -p data/cache data/archive logs config
```

---

## Configuration

### 1. Environment Variables

Create a `.env` file in the project root:

```bash
# API Keys (Alpaca Markets)
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Paper trading
# For live trading: https://api.alpaca.markets

# Backend Configuration
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000

# Frontend Configuration
FRONTEND_PORT=3000

# Risk Management
MAX_DAILY_LOSS_PERCENT=3.0      # Kill switch at 3% daily loss
MAX_POSITION_SIZE_PERCENT=20.0  # Max 20% per position
MAX_TOTAL_EXPOSURE_PERCENT=50.0 # Max 50% total deployed capital

# Data Settings
CANDLE_BUFFER_SIZE=500           # Keep last 500 candles in memory
CACHE_EXPIRY_HOURS=24            # Refresh cached data after 24 hours

# Strategy Parameters
TREND_KAMA_FAST=2                # Fast EMA period for KAMA
TREND_KAMA_SLOW=30               # Slow EMA period for KAMA
VOLATILITY_ATR_PERIOD=14         # ATR calculation period
REGIME_ADX_PERIOD=14             # ADX period for trend detection

# Logging
LOG_LEVEL=INFO                   # DEBUG, INFO, WARNING, ERROR
LOG_FILE=logs/fluxhero.log
```

**Security Notes**:
- **NEVER** commit `.env` to version control
- Use paper trading URL (`https://paper-api.alpaca.markets`) for testing
- Store API keys securely (use password manager)

### 2. Trading Strategy Configuration

Default strategy parameters are defined in the backend code but can be overridden via environment variables or API calls.

**Trend-Following Mode**:
- Entry: Price > KAMA + 0.5 × ATR
- Exit: Trailing stop at 2.5 × ATR from peak
- Risk: 1% of portfolio per trade

**Mean-Reversion Mode**:
- Entry: RSI < 30 AND price at lower Bollinger Band
- Exit: Return to 20-SMA OR RSI > 70
- Risk: 0.75% of portfolio per trade

**Regime Detection Thresholds**:
- Strong Trend: ADX > 25 AND R² > 0.7
- Ranging: ADX < 20 AND R² < 0.3
- Neutral: Between the above

---

## Running the System

### 1. Start the Backend API

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Start FastAPI server
cd fluxhero
uvicorn backend.api.server:app --host 0.0.0.0 --port 8000 --reload
```

**Expected Output**:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**Startup Sequence**:
1. Loads configuration from `.env`
2. Initializes SQLite database (creates tables if needed)
3. Validates Parquet cache (checks if data is <24 hours old)
4. Starts REST API server
5. Waits for WebSocket connections

**API Documentation**: Visit `http://localhost:8000/docs` for interactive Swagger UI

### 2. Start the Frontend Dashboard

Open a new terminal:

```bash
cd fluxhero/frontend
npm run dev
```

**Expected Output**:
```
   ▲ Next.js 16.0.0
   - Local:        http://localhost:3000
   - Ready in 1.2s
```

**Access Dashboard**: Open `http://localhost:3000` in your browser

### 3. Verify System Health

#### Check Backend Health
```bash
curl http://localhost:8000/health
```

**Expected Response**:
```json
{
  "status": "healthy",
  "timestamp": "2026-01-21T10:30:00Z"
}
```

#### Check System Status
```bash
curl http://localhost:8000/api/status
```

**Expected Response**:
```json
{
  "system_status": "active",
  "last_data_update": "2026-01-21T10:29:55Z",
  "websocket_connected": true,
  "active_positions": 0,
  "daily_pnl": 0.0
}
```

---

## Using the Dashboard

### Tab A: Live Trading

**Purpose**: Monitor open positions and system status in real-time

**Features**:
- **Positions Table**: Shows all open positions with entry price, current price, P&L, and position size
- **System Heartbeat**: Indicator showing connection status
  - 🟢 Active: System receiving data normally
  - 🟡 Delayed: No data for 30-60 seconds
  - 🔴 Offline: No data for >60 seconds
- **Quick Stats**: Daily P&L, current drawdown %, total exposure %
- **Auto-refresh**: Updates every 5 seconds

**Example Use**:
1. Open Live Trading tab
2. Check heartbeat indicator (should be 🟢)
3. Review open positions
4. Monitor daily P&L and exposure

### Tab B: Analytics

**Purpose**: Visualize price action, indicators, and signals

**Features**:
- **Candlestick Chart**: TradingView-style price chart with OHLC data
- **KAMA Overlay**: Adaptive moving average line
- **ATR Bands**: Volatility bands (KAMA ± ATR)
- **Signal Annotations**: Buy/sell arrows on chart
- **Indicator Panel**: Real-time values for ATR, RSI, ADX, regime state
- **Performance Metrics**: Total return %, Sharpe ratio, win rate, max drawdown

**Example Use**:
1. Select symbol from dropdown
2. Adjust timeframe (1m, 5m, 1h, 1d)
3. Analyze indicator values and regime state
4. Review historical signals on chart

### Tab C: Trade History

**Purpose**: Review past trades and export data

**Features**:
- **Trade Log Table**: All executed trades with timestamp, symbol, side, price, P&L
- **Pagination**: 20 trades per page
- **CSV Export**: Download trade history for external analysis
- **Signal Explanations**: Hover over trades to see why the signal was generated

**Example Use**:
1. Open Trade History tab
2. Review recent trades
3. Click on trade for detailed signal explanation
4. Export to CSV for further analysis in Excel/Python

### Tab D: Backtesting

**Purpose**: Test strategies on historical data

**Features**:
- **Configuration Form**:
  - Date range picker (start/end dates)
  - Symbol selector (SPY, QQQ, AAPL, etc.)
  - Parameter sliders (KAMA periods, ATR multipliers, risk %)
- **Run Backtest Button**: Executes backtest with loading spinner
- **Results Display**: Quantstats tearsheet with performance metrics
- **Export Options**: Download PDF report or HTML tearsheet

**Example Use**:
1. Select symbol (e.g., SPY)
2. Set date range (e.g., 2024-01-01 to 2024-12-31)
3. Adjust parameters if needed
4. Click "Run Backtest"
5. Review results (Sharpe ratio, max drawdown, win rate)
6. Export PDF report

---

## Monitoring

### 1. Real-Time Monitoring

**System Health Checks**:
- **Heartbeat Monitor**: Backend checks for data gaps >60 seconds
- **WebSocket Connection**: Auto-reconnect with exponential backoff (max 5 retries)
- **API Rate Limiting**: Respects broker limits (200 req/min for Alpaca)

**Key Metrics to Watch**:
- Daily P&L (should stay within risk limits)
- Current drawdown % (alert if >15%, kill switch at 20%)
- Total exposure % (should be ≤50%)
- Position count (max 5 concurrent positions)

### 2. Log Monitoring

**Backend Logs**: `logs/fluxhero.log`

```bash
# Watch logs in real-time
tail -f logs/fluxhero.log

# Search for errors
grep ERROR logs/fluxhero.log

# Check recent signals
grep "Signal generated" logs/fluxhero.log | tail -10
```

**Log Levels**:
- `DEBUG`: Detailed diagnostic information
- `INFO`: General informational messages (signals, trades, state changes)
- `WARNING`: Potential issues (high correlation, near limits)
- `ERROR`: Errors that require attention (API failures, order rejections)

**Important Log Patterns**:
```
INFO:     Signal generated: BUY SPY @ 450.25 (regime=STRONG_TREND, er=0.65)
INFO:     Order placed: BUY 100 SPY @ market (order_id=abc123)
WARNING:  Correlation high: SPY vs QQQ = 0.85, reducing position size by 50%
ERROR:    API request failed: Rate limit exceeded, retrying in 2s
```

### 3. Database Monitoring

**Check Trades**:
```bash
sqlite3 data/fluxhero.db "SELECT * FROM trades ORDER BY timestamp DESC LIMIT 10;"
```

**Check Positions**:
```bash
sqlite3 data/fluxhero.db "SELECT * FROM positions WHERE status='open';"
```

**Check Daily Performance**:
```bash
sqlite3 data/fluxhero.db "SELECT date, SUM(pnl) as daily_pnl FROM trades WHERE date=date('now') GROUP BY date;"
```

---

## Common Operations

### 1. Starting the System Daily

**Recommended Startup Sequence** (9:00 AM EST):

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Run daily reboot script (fetches 500 candles, reconnects WebSocket)
python -m fluxhero.backend.maintenance.daily_reboot

# 3. Start backend
uvicorn fluxhero.backend.api.server:app --host 0.0.0.0 --port 8000

# 4. Start frontend (in separate terminal)
cd fluxhero/frontend && npm run dev
```

**Why 9:00 AM EST?**
- Market opens at 9:30 AM EST
- Gives 30 minutes to fetch data, validate systems, and prepare

### 2. Running a Backtest

**Via Dashboard**:
1. Open Backtesting tab
2. Configure parameters
3. Click "Run Backtest"

**Via API**:
```bash
curl -X POST http://localhost:8000/api/backtest \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "SPY",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "initial_capital": 100000,
    "kama_fast": 2,
    "kama_slow": 30,
    "risk_per_trade": 0.01
  }'
```

**Via Python**:
```python
from fluxhero.backend.backtesting.engine import BacktestEngine

engine = BacktestEngine(
    symbol="SPY",
    start_date="2024-01-01",
    end_date="2024-12-31",
    initial_capital=100000
)
results = engine.run()
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

### 3. Exporting Data

**Export Trade History**:
```bash
# Export to CSV
curl http://localhost:8000/api/trades?format=csv > trades.csv

# Export to JSON
curl http://localhost:8000/api/trades > trades.json
```

**Export Performance Report**:
- Use the Backtesting tab "Export PDF" button
- Or call the API: `GET /api/backtest/{backtest_id}/report`

### 4. Archiving Old Data

**Daily Rollover** (automated):
- Keeps last 30 days in SQLite
- Archives older trades to `data/archive/`

**Manual Archive**:
```bash
python -m fluxhero.backend.storage.archive --days 30
```

### 5. Stopping the System

**Graceful Shutdown**:
1. Close all open positions (via dashboard or API)
2. Wait for pending orders to fill or cancel
3. Stop frontend: `Ctrl+C` in frontend terminal
4. Stop backend: `Ctrl+C` in backend terminal

**Emergency Shutdown** (if positions need to remain open):
1. Note open positions from dashboard
2. Stop frontend and backend
3. Positions remain on broker side (can manage via broker's interface)

---

## Troubleshooting

### Issue 1: Backend Won't Start

**Symptom**: `uvicorn` command fails or exits immediately

**Possible Causes**:
1. **Port already in use**
   - Check: `lsof -i :8000`
   - Fix: Kill the process or use a different port

2. **Missing dependencies**
   - Check: `pip list | grep fastapi`
   - Fix: `pip install -r requirements.txt`

3. **Python version too old**
   - Check: `python --version`
   - Fix: Install Python 3.10+

4. **Database corruption**
   - Check: `sqlite3 data/fluxhero.db ".schema"`
   - Fix: Delete database file (will recreate on next start)

### Issue 2: WebSocket Disconnects

**Symptom**: Heartbeat indicator turns 🟡 or 🔴

**Possible Causes**:
1. **Network interruption**
   - Check internet connection
   - WebSocket will auto-reconnect (up to 5 retries)

2. **Broker API issues**
   - Check Alpaca status page
   - Wait for broker to restore service

3. **Rate limit exceeded**
   - Check logs for "Rate limit" errors
   - System will automatically backoff

**Manual Reconnect**:
```bash
# Restart backend
# WebSocket reconnects on startup
```

### Issue 3: No Data in Dashboard

**Symptom**: Charts are empty, no positions shown

**Possible Causes**:
1. **Backend not running**
   - Check: `curl http://localhost:8000/health`
   - Fix: Start backend

2. **CORS error**
   - Check browser console (F12)
   - Fix: Ensure backend allows `http://localhost:3000`

3. **API keys invalid**
   - Check `.env` file for correct keys
   - Test: `curl -H "APCA-API-KEY-ID: your_key" https://paper-api.alpaca.markets/v2/account`

4. **No market data**
   - Check: `curl http://localhost:8000/api/status`
   - Market may be closed (weekends, holidays)

### Issue 4: Backtest Fails

**Symptom**: Backtest returns error or unexpected results

**Possible Causes**:
1. **Insufficient data**
   - Check date range (must have market data)
   - Alpaca free tier has data limits

2. **Invalid parameters**
   - KAMA fast must be < KAMA slow
   - Risk per trade must be 0 < risk ≤ 0.05 (5%)

3. **Symbol not found**
   - Ensure symbol exists and is tradable
   - Check for correct ticker (SPY not S&P500)

### Issue 5: Circuit Breaker Triggered

**Symptom**: "Kill switch activated" message, all positions closed

**Cause**: Daily loss exceeded threshold (default 3%)

**Actions**:
1. **Review trades**: Check what went wrong
2. **Analyze logs**: Look for pattern of losses
3. **Adjust strategy**: Consider parameter changes
4. **Reset for next day**: System will auto-reset at midnight

**Manual Reset** (use with caution):
```bash
curl -X POST http://localhost:8000/api/risk/reset-kill-switch
```

### Issue 6: High Slippage in Backtests

**Symptom**: Live results differ significantly from backtest

**Possible Causes**:
1. **Illiquid symbol**
   - Backtest assumes 0.01% slippage
   - Real slippage may be higher for low-volume stocks

2. **Large position size**
   - Orders >10% of average volume incur extra slippage
   - Reduce position size

3. **Market gaps**
   - Price gap filter rejects trades if gap >2%
   - Backtests may not model this accurately

**Fix**: Use more liquid symbols (SPY, QQQ) for testing

---

## Best Practices

### 1. Risk Management

**Position Sizing**:
- Never exceed 20% of capital per position
- Keep total exposure ≤50%
- Use 1% risk for trend-following, 0.75% for mean-reversion

**Diversification**:
- Limit to 5 concurrent positions
- Avoid correlated assets (>0.7 correlation)
- Mix trend and mean-reversion trades

**Loss Limits**:
- Daily loss limit: 3% (kill switch)
- Drawdown alert: 15% (reduce size by 50%)
- Max drawdown: 20% (full stop)

### 2. Strategy Optimization

**Backtesting**:
- Test on at least 1 year of data
- Use walk-forward testing (3-month train, 1-month test)
- Target Sharpe ratio >0.8, max drawdown <25%, win rate >45%

**Parameter Tuning**:
- Don't over-optimize (use round numbers)
- Test multiple symbols (SPY, QQQ, IWM)
- Validate on out-of-sample data

**Regime Awareness**:
- Check regime before trading
- Avoid mean-reversion in strong trends
- Avoid trend-following in choppy markets

### 3. System Maintenance

**Daily Tasks**:
- Run daily reboot script (9:00 AM EST)
- Check system status before market open
- Review overnight positions

**Weekly Tasks**:
- Review trade performance (win rate, Sharpe ratio)
- Check log files for errors
- Archive old data

**Monthly Tasks**:
- Update dependencies (`pip install --upgrade -r requirements.txt`)
- Review and adjust strategy parameters
- Analyze monthly performance report

### 4. Data Management

**Cache Management**:
- Cache expires after 24 hours (automatic refresh)
- Manually refresh: Delete files in `data/cache/`
- Archive grows over time (compress/delete old files)

**Database Maintenance**:
- SQLite auto-manages indexes
- Vacuum database monthly: `sqlite3 data/fluxhero.db "VACUUM;"`
- Backup before major changes: `cp data/fluxhero.db data/fluxhero_backup.db`

### 5. Security

**API Keys**:
- Use paper trading keys for testing
- Rotate keys periodically
- Never share keys or commit to Git

**Network Security**:
- Run backend on localhost (not public internet)
- Use VPN if accessing remotely
- Keep dependencies updated for security patches

**Access Control**:
- Limit who can access the dashboard
- Use strong passwords for broker account
- Enable 2FA on broker account

### 6. Performance Tuning

**Backend**:
- Numba functions are cached (first run is slow, subsequent runs are fast)
- Increase `CANDLE_BUFFER_SIZE` if you need more history (uses more RAM)
- Use `LOG_LEVEL=WARNING` in production to reduce I/O

**Frontend**:
- Charts render up to 1000 candles efficiently
- Reduce auto-refresh interval if system is slow
- Close unused browser tabs

**Database**:
- Archive old trades to keep database small
- Index on `timestamp`, `symbol`, `status` columns (already created)

---

## Support and Resources

### Documentation
- **API Reference**: `docs/API_DOCUMENTATION.md`
- **Risk Management Rules**: `docs/RISK_MANAGEMENT.md`
- **Deployment Guide**: `docs/DEPLOYMENT_GUIDE.md`
- **Maintenance Guide**: `docs/MAINTENANCE_GUIDE.md`

### Logs and Debugging
- Backend logs: `logs/fluxhero.log`
- Frontend logs: Browser console (F12)
- Database queries: `sqlite3 data/fluxhero.db`

### Testing
- Run unit tests: `pytest tests/`
- Run integration tests: `pytest tests/integration/`
- Performance benchmarks: `pytest tests/performance/`

### Community
- GitHub Issues: Report bugs and request features
- Trading Strategy Ideas: See `algorithmic-trading-guide.md`
- Quant Trading Resources: See `quant_trading_guide.md`

---

## Quick Reference

### Common Commands

```bash
# Start backend
uvicorn fluxhero.backend.api.server:app --host 0.0.0.0 --port 8000

# Start frontend
cd fluxhero/frontend && npm run dev

# Check system status
curl http://localhost:8000/api/status

# Run backtest
curl -X POST http://localhost:8000/api/backtest -d '{"symbol":"SPY",...}'

# View logs
tail -f logs/fluxhero.log

# Export trades
curl http://localhost:8000/api/trades?format=csv > trades.csv

# Archive old data
python -m fluxhero.backend.storage.archive --days 30
```

### Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `ALPACA_API_KEY` | - | Alpaca API key (required) |
| `ALPACA_SECRET_KEY` | - | Alpaca secret key (required) |
| `BACKEND_PORT` | 8000 | Backend server port |
| `FRONTEND_PORT` | 3000 | Frontend dev server port |
| `MAX_DAILY_LOSS_PERCENT` | 3.0 | Kill switch threshold |
| `MAX_POSITION_SIZE_PERCENT` | 20.0 | Max position size |
| `CANDLE_BUFFER_SIZE` | 500 | In-memory candle buffer |
| `LOG_LEVEL` | INFO | Logging verbosity |

### Key Files and Directories

| Path | Purpose |
|------|---------|
| `.env` | Environment variables and secrets |
| `data/fluxhero.db` | SQLite database (trades, positions) |
| `data/cache/` | Parquet cached market data |
| `logs/fluxhero.log` | Application logs |
| `fluxhero/backend/api/server.py` | FastAPI server entry point |
| `fluxhero/frontend/app/page.tsx` | Frontend entry point |

---

**End of User Guide**
