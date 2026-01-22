# FluxHero - Adaptive Retail Quant Trading System

**Version**: 1.0.0
**License**: MIT
**Status**: Production Ready

---

## Overview

FluxHero is an adaptive quantitative trading system designed for solo retail developers. It combines high-performance Python computation (via Numba JIT), adaptive trading strategies, and a modern React dashboard for real-time monitoring and analysis.

### Key Features

- **Dual-Mode Strategy Engine**: Automatically switches between trend-following and mean-reversion strategies based on market regime detection
- **Adaptive Indicators**: KAMA (Kaufman Adaptive Moving Average), volatility-adjusted smoothing, and regime-aware signal generation
- **Market Regime Detection**: Real-time classification of trending vs ranging markets using ADX and linear regression
- **Comprehensive Risk Management**: Position sizing, portfolio limits, correlation checks, and circuit breakers
- **High-Performance Computation**: Numba JIT-compiled indicators for <100ms processing of 10k candles
- **Professional Backtesting**: Realistic slippage/commission modeling, walk-forward testing, and quantstats integration
- **Real-Time Dashboard**: React + Next.js frontend with live P&L tracking, charts, and performance analytics
- **Market Microstructure Filters**: Spread-to-volatility validation, volume checks, and illiquid period handling

### Target Audience

Solo retail developers trading US equities/ETFs who want a production-ready, customizable trading system with institutional-grade risk management.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running the System](#running-the-system)
5. [Testing](#testing)
6. [Project Structure](#project-structure)
7. [Usage](#usage)
8. [Troubleshooting](#troubleshooting)
9. [Documentation](#documentation)

---

## System Requirements

### Hardware

- **CPU**: Modern multi-core processor (4+ cores recommended)
- **RAM**: 4GB minimum, 8GB+ recommended for optimal performance
- **Storage**: 2GB for system + additional space for market data cache (recommend 10GB+)
- **Network**: Stable internet connection for market data APIs and WebSocket feeds

### Software

- **Operating System**: Linux, macOS, or Windows 10+
- **Python**: 3.10 or higher (3.11+ recommended)
- **Node.js**: 18.0 or higher
- **npm**: 9.0 or higher
- **Git**: For version control and updates

### API Access

- **Market Data Provider**: Alpaca, Interactive Brokers, or compatible API
- **API Keys**: Required for live trading and data feeds (see Configuration section)

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/fluxhero.git
cd fluxhero
```

### 2. Backend Setup (Python)

#### Create Virtual Environment

```bash
# Navigate to the fluxhero directory
cd fluxhero

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

#### Install Python Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation
python -c "import numba; import numpy; import pandas; print('Backend dependencies installed successfully')"
```

**Core Dependencies**:
- `numba>=0.58.0` - JIT compilation for performance
- `numpy>=1.24.0` - Numerical computation
- `pandas>=2.0.0` - Data manipulation
- `pyarrow>=14.0.0` - Parquet storage
- `httpx>=0.25.0` - Async HTTP client
- `websockets>=12.0` - WebSocket feed handling
- `fastapi>=0.104.0` - REST API framework
- `uvicorn[standard]>=0.24.0` - ASGI server
- `quantstats>=0.0.62` - Backtesting analytics
- `pytest>=7.4.0` - Testing framework
- `ruff>=0.1.0` - Code linting

### 3. Frontend Setup (React + Next.js)

#### Navigate to Frontend Directory

```bash
# From project root
cd frontend
```

#### Install Node.js Dependencies

```bash
# Install all packages
npm install

# Verify installation
npm run build
```

**Core Dependencies**:
- `next@^16.1.4` - React framework
- `react@^19.2.3` - UI library
- `react-dom@^19.2.3` - React DOM renderer
- `lightweight-charts@^5.1.0` - TradingView-style charts
- `typescript@^5.9.3` - Type safety

### 4. Create Required Directories

```bash
# From project root
mkdir -p data/cache data/logs config logs
```

### 5. Verify Installation

```bash
# Run backend tests
cd fluxhero
source venv/bin/activate  # if not already activated
pytest tests/ -v

# Run frontend tests
cd ../frontend
npm test

# Check linting
cd ../fluxhero
ruff check .
```

---

## Configuration

### 1. Environment Variables

Create a `.env` file in the project root:

```bash
# Copy template (if available)
cp .env.example .env

# Edit with your settings
nano .env
```

**Required Variables**:

```env
# API Configuration
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Use paper trading for testing

# Database
DATABASE_PATH=./data/fluxhero.db

# Cache
CACHE_DIR=./data/cache

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/fluxhero.log

# Risk Management
MAX_POSITION_SIZE=0.20  # 20% per position
MAX_TOTAL_DEPLOYMENT=0.50  # 50% total
DAILY_LOSS_LIMIT=0.03  # 3% daily loss limit

# Backend API
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 2. Trading Configuration

Edit `config/trading_config.yaml`:

```yaml
# Strategy Parameters
strategy:
  mode: dual  # dual, trend_only, mean_reversion_only
  trend_weight: 0.5
  reversion_weight: 0.5

# Risk Management
risk:
  max_position_risk: 0.01  # 1% risk per trade
  max_portfolio_risk: 0.05  # 5% total portfolio risk
  position_size_limit: 0.20  # 20% per position
  total_deployment_limit: 0.50  # 50% max deployed
  correlation_threshold: 0.7  # Reduce size if correlation > 0.7

# Circuit Breakers
circuit_breakers:
  drawdown_warning: 0.15  # 15% DD - reduce position sizes by 50%
  drawdown_halt: 0.20  # 20% DD - stop trading
  daily_loss_limit: 0.03  # 3% daily loss - close all positions

# Indicator Parameters
indicators:
  kama_fast: 2
  kama_slow: 30
  atr_period: 14
  rsi_period: 14
  adx_period: 14

# Market Microstructure
microstructure:
  max_spread_volatility_ratio: 0.05  # Reject if SV_Ratio > 5%
  min_volume_multiplier: 0.5  # Require volume > 0.5× avg
  breakout_volume_multiplier: 1.5  # Require volume > 1.5× avg for breakouts

# Backtesting
backtesting:
  slippage: 0.0001  # 0.01% slippage
  commission: 0.005  # $0.005 per share
  impact_threshold: 0.10  # Extra slippage if order > 10% avg volume
  risk_free_rate: 0.04  # 4% for Sharpe calculation
```

---

## Running the System

### Development Mode

#### 1. Start the Backend

```bash
# From fluxhero directory
cd fluxhero
source venv/bin/activate

# Run backend server
uvicorn backend.api.server:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at `http://localhost:8000`

API documentation: `http://localhost:8000/docs`

#### 2. Start the Frontend

```bash
# From frontend directory (new terminal)
cd frontend
npm run dev
```

Frontend will be available at `http://localhost:3000`

### Production Mode

#### 1. Build Frontend

```bash
cd frontend
npm run build
```

#### 2. Start Production Servers

```bash
# Backend (use a process manager like systemd or PM2)
cd fluxhero
source venv/bin/activate
uvicorn backend.api.server:app --host 0.0.0.0 --port 8000 --workers 4

# Frontend
cd frontend
npm run start
```

### Running Backtests

```bash
# From fluxhero directory
cd fluxhero
source venv/bin/activate

# Run backtest script
python -m backend.backtesting.run_backtest \
  --symbol SPY \
  --start 2023-01-01 \
  --end 2024-01-01 \
  --initial-capital 100000 \
  --output ./data/backtest_results.html

# View results in browser
open ./data/backtest_results.html
```

---

## Testing

### Backend Tests

```bash
cd fluxhero
source venv/bin/activate

# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_indicators.py -v

# Run with coverage
pytest tests/ --cov=backend --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Frontend Tests

```bash
cd frontend

# Run all tests
npm test

# Run with coverage
npm run test:coverage

# Run in watch mode (for development)
npm run test:watch
```

### Integration Tests

```bash
# From project root
pytest tests/integration/ -v
```

### Linting

```bash
# Backend (Python)
cd fluxhero
ruff check .
ruff format .

# Frontend (TypeScript)
cd frontend
npm run lint
```

---

## Project Structure

```
project/
├── fluxhero/                    # Backend application
│   ├── backend/
│   │   ├── computation/         # Numba JIT indicators
│   │   │   ├── indicators.py    # EMA, RSI, ATR
│   │   │   ├── adaptive_ema.py  # KAMA implementation
│   │   │   └── volatility.py    # Volatility calculations
│   │   ├── strategy/            # Trading strategies
│   │   │   ├── dual_mode.py     # Trend + Mean-reversion
│   │   │   ├── regime_detector.py  # ADX, R² regime detection
│   │   │   ├── noise_filter.py  # Microstructure filters
│   │   │   └── signal_generator.py  # Signal generation
│   │   ├── data/                # Data fetching
│   │   │   ├── fetcher.py       # Async API wrapper
│   │   │   └── websocket_feed.py  # Live data feeds
│   │   ├── storage/             # State management
│   │   │   ├── sqlite_store.py  # Trade/position storage
│   │   │   ├── parquet_store.py # Market data cache
│   │   │   └── buffer.py        # In-memory candle buffer
│   │   ├── backtesting/         # Backtesting engine
│   │   │   ├── engine.py        # Backtest orchestrator
│   │   │   ├── fills.py         # Fill simulation
│   │   │   └── metrics.py       # Performance analytics
│   │   ├── execution/           # Order execution
│   │   │   ├── broker_interface.py  # Broker abstraction
│   │   │   ├── order_manager.py     # Order lifecycle
│   │   │   └── position_sizer.py    # Position sizing
│   │   ├── risk/                # Risk management
│   │   │   ├── position_limits.py   # Position-level limits
│   │   │   └── kill_switch.py       # Circuit breakers
│   │   ├── api/                 # REST API
│   │   │   └── server.py        # FastAPI endpoints
│   │   └── maintenance/         # System operations
│   │       └── daily_reboot.py  # Daily maintenance
│   ├── tests/                   # Backend tests
│   │   ├── test_indicators.py
│   │   ├── test_strategy.py
│   │   ├── test_backtesting.py
│   │   └── integration/
│   ├── requirements.txt         # Python dependencies
│   └── venv/                    # Virtual environment
│
├── frontend/                    # React + Next.js frontend
│   ├── pages/                   # Next.js pages
│   │   ├── live.tsx             # Live trading tab
│   │   ├── analytics.tsx        # Charts & indicators
│   │   ├── history.tsx          # Trade history
│   │   └── backtest.tsx         # Backtesting interface
│   ├── components/              # React components
│   │   ├── PositionTable.tsx
│   │   ├── CandlestickChart.tsx
│   │   ├── IndicatorPanel.tsx
│   │   └── MetricsDisplay.tsx
│   ├── lib/                     # Utilities
│   │   ├── api.ts               # API client
│   │   └── websocket.ts         # WebSocket handler
│   ├── package.json             # Node.js dependencies
│   └── node_modules/            # Node packages
│
├── config/                      # Configuration files
│   ├── trading_config.yaml
│   └── data_sources.yaml
│
├── data/                        # Data storage
│   ├── cache/                   # Parquet cache
│   ├── logs/                    # Application logs
│   └── fluxhero.db              # SQLite database
│
├── docs/                        # Documentation
│   ├── API_DOCUMENTATION.md
│   ├── USER_GUIDE.md
│   └── RISK_MANAGEMENT.md
│
├── tests/                       # Integration tests
│   └── integration/
│
├── .env                         # Environment variables (create from .env.example)
├── .gitignore                   # Git ignore rules
├── FLUXHERO_REQUIREMENTS.md     # Feature specifications
├── FLUXHERO_TASKS.md            # Implementation task list
└── README.md                    # This file
```

---

## Usage

### Dashboard Overview

Access the dashboard at `http://localhost:3000` after starting both backend and frontend.

#### Tab A: Live Trading

- **Open Positions Table**: Real-time P&L, entry price, current price, unrealized profit/loss
- **System Heartbeat**: 🟢 Active / 🟡 Delayed / 🔴 Offline
- **Quick Stats**: Daily P&L, current drawdown %, total exposure %
- **Auto-refresh**: Updates every 5 seconds

#### Tab B: Analytics

- **Candlestick Chart**: TradingView-style chart with KAMA overlay and ATR bands
- **Signal Annotations**: Buy/sell arrows with hover explanations
- **Indicator Panel**: Real-time ATR, RSI, ADX, regime state
- **Performance Metrics**: Total return %, Sharpe ratio, win rate, max drawdown

#### Tab C: Trade History

- **Trade Log Table**: All executed trades with entry/exit details
- **Pagination**: 20 trades per page
- **CSV Export**: Download trade history for analysis
- **Trade Details**: Hover tooltips with signal explanations

#### Tab D: Backtesting

- **Configuration Form**: Date range, symbol selector, parameter sliders
- **Run Backtest**: Execute backtest with loading spinner
- **Results Display**: Quantstats tearsheet with detailed metrics
- **Export**: Download results as HTML or PDF

### API Endpoints

Full API documentation available at `http://localhost:8000/docs`

**Key Endpoints**:
- `GET /api/positions` - Current open positions
- `GET /api/trades` - Trade history (with pagination)
- `GET /api/account` - Account info and buying power
- `GET /api/status` - System health and heartbeat
- `POST /api/backtest` - Execute backtest
- `WS /ws/prices` - Live price updates (WebSocket)

### Common Operations

#### Start Daily Trading Session

```bash
# 1. Activate virtual environment
cd fluxhero
source venv/bin/activate

# 2. Run daily maintenance (fetches 500 candles, validates cache)
python -m backend.maintenance.daily_reboot

# 3. Start backend
uvicorn backend.api.server:app --host 0.0.0.0 --port 8000

# 4. Monitor logs
tail -f ../logs/fluxhero.log
```

#### Monitor System Health

```bash
# Check system status
curl http://localhost:8000/api/status

# Check current positions
curl http://localhost:8000/api/positions

# View recent trades
curl http://localhost:8000/api/trades?limit=10
```

#### Emergency Stop

```bash
# Stop all trading immediately (circuit breaker)
curl -X POST http://localhost:8000/api/emergency_stop

# Or manually close all positions in the dashboard
```

---

## Troubleshooting

### Common Issues

#### Backend Won't Start

**Error**: `ModuleNotFoundError: No module named 'numba'`

**Solution**:
```bash
cd fluxhero
source venv/bin/activate
pip install -r requirements.txt
```

#### Frontend Build Fails

**Error**: `Cannot find module 'next'`

**Solution**:
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

#### WebSocket Connection Fails

**Error**: Connection refused on `/ws/prices`

**Solution**:
1. Check backend is running: `curl http://localhost:8000/api/status`
2. Check CORS settings in `backend/api/server.py`
3. Verify `NEXT_PUBLIC_API_URL` in frontend `.env`

#### Numba Compilation Errors

**Error**: `LoweringError` during indicator calculation

**Solution**:
1. Check Numba version: `pip show numba` (should be >=0.58.0)
2. Update LLVM: `pip install --upgrade llvmlite`
3. Clear cache: `rm -rf ~/.cache/numba_cache`

#### Database Locked

**Error**: `sqlite3.OperationalError: database is locked`

**Solution**:
```bash
# Check for other processes
lsof data/fluxhero.db

# Kill blocking processes or restart backend
```

### Performance Issues

#### Slow Indicator Calculations

- Check Numba compilation: First run is slow (compilation), subsequent runs should be fast
- Verify `@njit(cache=True)` is enabled
- Monitor with: `python -m backend.computation.benchmark`

#### High Memory Usage

- Reduce candle buffer size in `config/trading_config.yaml`
- Enable Parquet compression: `compression='snappy'`
- Limit in-memory data retention to 500 candles

### Logging

Enable debug logging:

```bash
# In .env
LOG_LEVEL=DEBUG

# View logs
tail -f logs/fluxhero.log

# Filter for errors
grep ERROR logs/fluxhero.log
```

---

## Documentation

### Available Guides

- **[API Documentation](docs/API_DOCUMENTATION.md)**: REST API endpoint specifications
- **[User Guide](docs/USER_GUIDE.md)**: Detailed usage instructions
- **[Risk Management](docs/RISK_MANAGEMENT.md)**: Circuit breaker behavior and risk rules
- **[Deployment Guide](docs/DEPLOYMENT_GUIDE.md)**: VPS and cloud hosting instructions (coming soon)
- **[Maintenance Guide](docs/MAINTENANCE_GUIDE.md)**: Dependency updates and backups (coming soon)

### Additional Resources

- **[FLUXHERO_REQUIREMENTS.md](FLUXHERO_REQUIREMENTS.md)**: Detailed feature specifications with formulas
- **[quant_trading_guide.md](quant_trading_guide.md)**: Technical indicator reference
- **[algorithmic-trading-guide.md](algorithmic-trading-guide.md)**: Trading concepts and best practices

---

## License

MIT License - See [LICENSE](LICENSE) file for details

---

## Disclaimer

**IMPORTANT**: This software is for educational and research purposes. Trading involves substantial risk of loss. Past performance does not guarantee future results. Always test thoroughly with paper trading before deploying real capital. The authors are not responsible for any financial losses incurred using this software.

---

**Built with**: Python (Numba, FastAPI, NumPy), React (Next.js, TypeScript), SQLite, Parquet
**Performance**: <100ms for 10k candle processing
**Risk Management**: Multi-layer circuit breakers and position limits
**Status**: Production-ready, actively maintained

---

*Last Updated: 2026-01-21*
