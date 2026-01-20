# FluxHero Project Structure

This directory contains the FluxHero adaptive retail quant trading system implementation.

## Directory Structure

```
fluxhero/
├── backend/                    # Backend Python codebase
│   ├── computation/            # JIT-optimized indicator calculations (Numba)
│   ├── strategy/               # Trading strategies and signal generation
│   ├── storage/                # State management (SQLite, Parquet)
│   ├── data/                   # API wrappers for data fetching
│   ├── backtesting/            # Backtesting engine and metrics
│   ├── execution/              # Order execution and broker interfaces
│   ├── risk/                   # Risk management and position limits
│   └── api/                    # FastAPI REST/WebSocket endpoints
│
├── frontend/                   # React + Next.js frontend
│   ├── pages/                  # Next.js page components
│   ├── components/             # Reusable React components
│   ├── utils/                  # Frontend utility functions
│   └── styles/                 # CSS and styling files
│
../tests/                       # Test suites (at project root)
│   ├── unit/                   # Unit tests for individual components
│   ├── integration/            # Integration tests for system workflows
│   └── e2e/                    # End-to-end tests
│
../data/                        # Data storage (at project root)
│   ├── cache/                  # Parquet cached market data
│   └── archive/                # Archived historical data
│
../logs/                        # Application logs (at project root)
│
../config/                      # Configuration files (at project root)
```

## Module Responsibilities

### Backend Modules

- **computation/**: Numba-optimized (@njit) calculations for EMA, RSI, ATR, KAMA, and other indicators
- **strategy/**: Regime detection, dual-mode strategies, noise filtering, signal generation
- **storage/**: Lightweight SQLite for trades/positions, Parquet for market data caching
- **data/**: Async API wrappers (httpx, WebSocket) with retry logic and rate limiting
- **backtesting/**: Backtest engine, fill simulation, slippage models, performance metrics
- **execution/**: Order management, position sizing, broker interfaces, kill-switch logic
- **risk/**: Position limits, correlation checks, drawdown monitoring, circuit breakers
- **api/**: FastAPI server with REST endpoints and WebSocket for live data

### Frontend Modules

- **pages/**: Next.js pages for Live Trading, Analytics, Trade History, Backtesting tabs
- **components/**: Reusable UI components (charts, tables, indicators, forms)
- **utils/**: API client, WebSocket handlers, data formatters
- **styles/**: Global styles, component styles, dark mode themes

### Test Structure

- **unit/**: Test individual functions (indicators, calculations, utilities)
- **integration/**: Test multi-component workflows (data fetch → signal → execution)
- **e2e/**: Test full system flows (startup → trade → storage → display)

## Development Workflow

1. Each feature is developed in its own git branch
2. Unit tests are required for all new functionality
3. Integration tests validate cross-module interactions
4. Performance benchmarks ensure <100ms for 10k candles
5. Code is linted and formatted before commits

## Key Technologies

- **Backend**: Python 3.10+, Numba, FastAPI, httpx, websockets, SQLite, PyArrow
- **Frontend**: React 18+, Next.js 13+, TypeScript, TradingView Lightweight Charts
- **Testing**: pytest, pytest-asyncio, React Testing Library
- **Performance**: Numba JIT compilation, async I/O, connection pooling
