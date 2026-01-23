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

---

## Quick Start

```bash
# Install all dependencies
make install

# Start both backend and frontend
make dev

# Run tests
make test

# Stop all services
make stop
```

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running the System](#running-the-system)
5. [Make Commands](#make-commands)
6. [Testing](#testing)
7. [Project Structure](#project-structure)
8. [Usage](#usage)
9. [Troubleshooting](#troubleshooting)
10. [Documentation](#documentation)

---

## System Requirements

### Software

- **Python**: 3.10 or higher (3.12+ recommended)
- **Node.js**: 18.0 or higher (22+ recommended)
- **npm**: 9.0 or higher
- **Git**: For version control

### Hardware

- **CPU**: Modern multi-core processor (4+ cores recommended)
- **RAM**: 4GB minimum, 8GB+ recommended
- **Storage**: 2GB for system + additional space for market data cache

---

## Installation

### Option 1: Using Make (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/fluxhero.git
cd fluxhero

# Install all dependencies (backend + frontend)
make install
```

### Option 2: Manual Installation

#### Backend Setup (Python with uv)

```bash
# Install uv (if not already installed)
# On macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
# On Windows:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Create virtual environment and install all dependencies
uv sync

# Activate virtual environment (for running commands manually)
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Verify installation
python -c "import numba; import numpy; import pandas; print('Backend ready')"
```

#### Frontend Setup (Node.js)

```bash
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install

# Verify installation
npm run build
```

### Create Required Directories

```bash
mkdir -p data/cache data/logs config logs data/archive
```

---

## Configuration

### Environment Variables

Create a `.env` file in the project root (copy from `.env.example`):

```env
# Authentication
FLUXHERO_AUTH_SECRET=your-secret-key-here

# Alpaca API (for live/paper trading)
FLUXHERO_ALPACA_API_KEY=your_api_key
FLUXHERO_ALPACA_API_SECRET=your_api_secret
FLUXHERO_ALPACA_API_URL=https://paper-api.alpaca.markets

# CORS (for frontend)
FLUXHERO_CORS_ORIGINS=["http://localhost:3000","http://localhost:3001"]

# Risk Management
FLUXHERO_MAX_RISK_PCT_TREND=0.01
FLUXHERO_MAX_RISK_PCT_MEAN_REV=0.0075
FLUXHERO_MAX_POSITION_SIZE_PCT=0.20
FLUXHERO_MAX_TOTAL_EXPOSURE_PCT=0.50

# Data Storage
FLUXHERO_CACHE_DIR=data/cache
FLUXHERO_LOG_FILE=logs/daily_reboot.log
```

---

## Running the System

### Development Mode (Recommended)

```bash
# Start both backend and frontend
make dev

# Or start separately:
make dev-backend    # Backend on http://localhost:8000
make dev-frontend   # Frontend on http://localhost:3000
```

### Manual Start

```bash
# Terminal 1: Backend
source .venv/bin/activate
uvicorn backend.api.server:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev
```

### Access Points

- **Frontend Dashboard**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Metrics**: http://localhost:8000/metrics

---

## Make Commands

### Development

| Command | Description |
|---------|-------------|
| `make dev` | Start both backend and frontend |
| `make dev-backend` | Start backend only (port 8000) |
| `make dev-frontend` | Start frontend only (port 3000) |
| `make stop` | Stop all running services |
| `make logs` | Show backend logs |

### Testing

| Command | Description |
|---------|-------------|
| `make test` | Run all tests (parallel) |
| `make test-unit` | Run unit tests only |
| `make test-integration` | Run integration tests only |
| `make test-serial` | Run tests serially |
| `make test-coverage` | Run tests with coverage report |

### Code Quality

| Command | Description |
|---------|-------------|
| `make lint` | Run ruff linter |
| `make format` | Auto-format code with ruff |
| `make typecheck` | Run mypy type checking |
| `make check` | Run all quality checks |

### Setup & Maintenance

| Command | Description |
|---------|-------------|
| `make install` | Install all dependencies |
| `make install-backend` | Install Python dependencies |
| `make install-frontend` | Install Node.js dependencies |
| `make clean` | Remove generated files and caches |
| `make daily-reboot` | Run daily maintenance script |
| `make archive-trades` | Archive old trades to Parquet |
| `make seed-data` | Seed database with test positions |

---

## Testing

### Backend Tests (Python)

```bash
# Run all tests (parallel by default)
make test

# Run with coverage
make test-coverage

# Run specific test file
source .venv/bin/activate
pytest tests/unit/test_indicators.py -v

# Run validation tests (hand-calculated expected values)
pytest tests/validation/ -v

# Run regression tests (golden results + benchmarks)
pytest tests/regression/ -v
```

#### Test Suites

| Suite | Location | Description |
|-------|----------|-------------|
| Unit | `tests/unit/` | Component-level tests |
| Integration | `tests/integration/` | Cross-component tests |
| Validation | `tests/validation/` | Tests with hand-calculated expected values for metrics, indicators, signals |
| Regression | `tests/regression/` | Golden result tests (SPY baseline) and benchmark comparisons |
| Performance | `tests/performance/` | Performance benchmarks |

### Frontend Tests (Jest)

```bash
cd frontend

# Run all tests
npm test

# Run with coverage
npm run test:coverage

# Run in watch mode
npm run test:watch
```

### E2E Tests (Playwright)

```bash
cd frontend

# Run all E2E tests
npm run test:e2e

# Run in UI mode (interactive)
npx playwright test --ui

# Run visual regression tests
npx playwright test e2e/visual-regression.spec.ts

# Update visual baselines
npm run test:e2e:update-snapshots
```

---

## Project Structure

```
project/
├── backend/                     # Python backend
│   ├── api/                     # REST API (FastAPI)
│   │   ├── server.py            # Main API server
│   │   ├── auth.py              # Authentication
│   │   └── rate_limit.py        # Rate limiting
│   ├── computation/             # Numba JIT indicators
│   │   ├── indicators.py        # EMA, RSI, ATR
│   │   ├── adaptive_ema.py      # KAMA implementation
│   │   └── volatility.py        # Volatility calculations
│   ├── strategy/                # Trading strategies
│   │   ├── dual_mode.py         # Trend + Mean-reversion
│   │   ├── regime_detector.py   # Market regime detection
│   │   ├── noise_filter.py      # Microstructure filters
│   │   └── signal_generator.py  # Signal generation
│   ├── storage/                 # Data persistence
│   │   ├── sqlite_store.py      # Trade/position storage
│   │   ├── parquet_store.py     # Market data cache
│   │   └── candle_buffer.py     # In-memory buffer
│   ├── backtesting/             # Backtesting engine
│   │   ├── engine.py            # Backtest orchestrator
│   │   ├── walk_forward.py      # Walk-forward testing
│   │   ├── fills.py             # Fill simulation
│   │   └── metrics.py           # Performance analytics
│   ├── execution/               # Order execution
│   │   ├── broker_interface.py  # Broker abstraction
│   │   ├── order_manager.py     # Order lifecycle
│   │   └── position_sizer.py    # Position sizing
│   ├── risk/                    # Risk management
│   │   ├── position_limits.py   # Position-level limits
│   │   └── kill_switch.py       # Circuit breakers
│   ├── core/                    # Core utilities
│   │   ├── config.py            # Centralized configuration
│   │   └── logging_config.py    # Logging setup
│   ├── data/                    # Data fetching
│   │   └── fetcher.py           # Async API wrapper
│   ├── maintenance/             # System operations
│   │   └── daily_reboot.py      # Daily maintenance
│   └── test_data/               # Test data files
│       ├── spy_daily.csv        # SPY historical data
│       ├── aapl_daily.csv       # AAPL historical data
│       └── msft_daily.csv       # MSFT historical data
│
├── frontend/                    # React + Next.js frontend
│   ├── app/                     # Next.js App Router
│   │   ├── page.tsx             # Home page
│   │   ├── live/                # Live trading page
│   │   ├── analytics/           # Analytics page
│   │   ├── backtest/            # Backtesting page
│   │   ├── walk-forward/        # Walk-forward testing page
│   │   ├── history/             # Trade history page
│   │   ├── signals/             # Signal analysis page
│   │   └── layout.tsx           # Root layout
│   ├── components/              # React components
│   ├── contexts/                # React contexts
│   ├── hooks/                   # Custom hooks
│   ├── utils/                   # Utilities (API client)
│   ├── e2e/                     # Playwright E2E tests
│   └── package.json             # Node.js dependencies
│
├── tests/                       # Backend tests
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   ├── validation/              # Hand-calculated validation tests
│   ├── regression/              # Golden results + benchmark tests
│   ├── e2e/                     # End-to-end tests
│   └── performance/             # Performance tests
│
├── docs/                        # Documentation
│   ├── API_DOCUMENTATION.md
│   ├── USER_GUIDE.md
│   ├── RISK_MANAGEMENT.md
│   └── DEPLOYMENT_GUIDE.md
│
├── scripts/                     # Utility scripts
│   ├── seed_test_data.py        # Test data seeding
│   └── run_spy_backtest.py      # SPY backtest runner
│
├── data/                        # Data storage
│   ├── cache/                   # Parquet cache
│   ├── archive/                 # Archived trades
│   └── logs/                    # Application logs
│
├── .venv/                       # Python virtual environment
├── Makefile                     # Development commands
├── pyproject.toml               # Python project config & dependencies
├── uv.lock                      # Locked Python dependencies (uv)
├── .env                         # Environment variables
├── .gitignore                   # Git ignore rules
├── FLUXHERO_REQUIREMENTS.md     # Feature specifications
├── FLUXHERO_TASKS.md            # Implementation tasks
├── TASKS.md                     # Current task tracking
├── PROJECT_AUDIT.md             # Audit report
└── README.md                    # This file
```

---

## Usage

### Dashboard Overview

Access the dashboard at http://localhost:3000 after starting the system.

#### Live Trading Tab
- Open positions with real-time P&L
- System heartbeat indicator
- Daily P&L, drawdown %, exposure %

#### Analytics Tab
- Candlestick chart with KAMA overlay
- Real-time indicators (ATR, RSI, ADX)
- Performance metrics

#### Trade History Tab
- Complete trade log with pagination
- CSV export functionality
- Signal explanations on hover

#### Backtesting Tab
- Configure backtest parameters
- Run simulations
- View quantstats tearsheet

#### Walk-Forward Testing Tab
- Configure train/test window sizes (default: 63/21 bars)
- Set pass rate threshold (default: 60%)
- View per-window results with PASS/FAIL status
- Combined equity curve visualization
- Export results to CSV

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | System health |
| `/api/positions` | GET | Current positions |
| `/api/trades` | GET | Trade history |
| `/api/account` | GET | Account info |
| `/api/backtest` | POST | Run backtest |
| `/api/backtest/walk-forward` | POST | Run walk-forward test |
| `/ws/prices` | WS | Live price updates |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |

---

## Troubleshooting

### Backend Won't Start

```bash
# Reinstall dependencies and recreate virtual environment
uv sync

# Or if venv already exists, just activate it
source .venv/bin/activate
```

### Frontend Build Fails

```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### E2E Tests Fail

```bash
# Install Playwright browsers
cd frontend
npx playwright install

# Update visual baselines if UI changed
npm run test:e2e:update-snapshots
```

### Port Already in Use

```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Kill process on port 3000
lsof -ti:3000 | xargs kill -9
```

---

## Documentation

- [API Documentation](docs/API_DOCUMENTATION.md)
- [User Guide](docs/USER_GUIDE.md)
- [Risk Management](docs/RISK_MANAGEMENT.md)
- [Deployment Guide](docs/DEPLOYMENT_GUIDE.md)
- [System Assumptions](docs/ASSUMPTIONS.md) - Commission, slippage, fill model documentation
- [Feature Requirements](FLUXHERO_REQUIREMENTS.md)
- [Implementation Tasks](FLUXHERO_TASKS.md)

---

## License

MIT License - See [LICENSE](LICENSE) file for details.

---

## Disclaimer

**IMPORTANT**: This software is for educational and research purposes. Trading involves substantial risk of loss. Always test thoroughly with paper trading before deploying real capital.

---

*Last Updated: 2026-01-23*
