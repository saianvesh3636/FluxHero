# Changelog

All notable changes to FluxHero are documented in this file.

---

## [2026-01-23] - Quality Control & Walk-Forward Release

### Added

#### Phase 24: Quality Control & Validation Framework
- **Metric Validation Suite** (`tests/validation/test_metric_calculations.py`)
  - 44 tests with hand-calculated expected values for Sharpe, drawdown, win rate
  - Includes worked examples in comments

- **Indicator Validation Suite** (`tests/validation/test_indicator_calculations.py`)
  - 34 tests for EMA, RSI, SMA, ATR, Bollinger Bands, KAMA
  - Verified against manual calculations

- **Signal Validation Suite** (`tests/validation/test_signal_generation.py`)
  - 38 tests for trend-following and mean-reversion signals
  - Regime detection validation

- **Data Validation on Load** (`backend/data/yahoo_provider.py`)
  - Checks for NaN in OHLCV, negative prices, volume=0, high<low errors
  - Alerts on gaps >5 days

- **Bar Integrity Checks** (`backend/backtesting/engine.py`)
  - Validates OHLC relationships (high >= low, etc.)
  - Timestamps monotonically increasing

- **Golden Test Suite** (`tests/regression/test_golden_results.py`)
  - Regression tests vs SPY 2020-2024 baseline
  - Alerts on >1% deviation from expected metrics

- **Benchmark Comparison Tests** (`tests/regression/test_benchmark_comparison.py`)
  - Compares strategy returns vs buy-and-hold
  - Warns if significantly underperforming

- **Assumptions Document** (`docs/ASSUMPTIONS.md`)
  - Documents commission model ($0.005/share)
  - Slippage model (0.01% + 0.05% impact)
  - Fill assumptions (next-bar open)
  - Position sizing risk model

- **Sanity Check Assertions** (`backend/backtesting/engine.py`)
  - Runtime assertions: equity never negative, valid position sizes
  - Trade entry < exit timestamps

- **Metric Sanity Checks** (`backend/backtesting/metrics.py`)
  - Sharpe ratio range [-5, +5]
  - Win rate [0, 1]
  - Max drawdown <= 100%

#### Phase 18: Walk-Forward Testing
- **Walk-Forward Module** (`backend/backtesting/walk_forward.py`)
  - `WalkForwardWindow` dataclass with train/test indices
  - `generate_walk_forward_windows()` - 63-day train / 21-day test default
  - `run_walk_forward_backtest()` - orchestrator for rolling windows
  - `aggregate_walk_forward_results()` - combines equity curves
  - `passes_walk_forward_test()` - requires >60% profitable windows

- **Walk-Forward API Endpoint** (`backend/api/server.py`)
  - `POST /api/backtest/walk-forward`
  - Returns per-window metrics and pass/fail status

- **Walk-Forward Frontend Page** (`frontend/app/walk-forward/page.tsx`)
  - Configuration form (train/test bars, threshold)
  - Per-window results table with PASS/FAIL badges
  - Combined equity curve chart
  - CSV export

- **Walk-Forward Tests**
  - 71 unit tests (`tests/unit/test_walk_forward.py`)
  - 13 integration tests (`tests/integration/test_walk_forward_backtest.py`)

#### Phase 19: Logging Enhancements
- **Request Body Logging** (`backend/api/server.py`)
  - `LOG_REQUEST_BODIES=true` env var (dev only)
  - Masks sensitive fields (password, token, api_key)
  - Truncates bodies >500 chars

- **Backtest Operation Logging** (`backend/backtesting/engine.py`)
  - Logs start with config summary
  - Progress every 10% of bars
  - Final metrics summary with duration in ms

- **Strategy Decision Logging** (`backend/strategy/backtest_strategy.py`)
  - DEBUG level logging for signal generation
  - Logs regime changes, entry/exit decisions

### Changed
- **Package Manager**: Migrated from pip to uv for faster dependency resolution
- **README.md**: Updated with new features, test suites, and documentation links

### Fixed
- All parallel test execution issues resolved
- Frontend-backend integration working correctly

---

## [2026-01-22] - Frontend Redesign & Parallel Testing

### Added
- **Frontend Redesign** - All 6 pages with new design system
  - Home, Live, Analytics, Backtest, History, Signals pages
  - Dark mode only, flat design, no animations
  - Tailwind CSS v4 integration

- **Parallel Test Execution**
  - pytest-xdist integration for multi-worker testing
  - All 1347+ tests pass with `pytest -n auto`

- **E2E Tests** (Playwright)
  - Visual regression tests
  - WebSocket connection tests
  - Error state tests

### Documentation
- `docs/PARALLEL_TEST_DATABASE_ANALYSIS.md`
- `docs/PARALLEL_TEST_PORT_ANALYSIS.md`
- `PARALLEL_TEST_FINDINGS.md`

---

## Documentation Index

| Document | Purpose |
|----------|---------|
| `README.md` | Project overview, quick start, make commands |
| `FLUXHERO_REQUIREMENTS.md` | Detailed feature specifications with formulas |
| `FLUXHERO_TASKS.md` | Master implementation task list (118 tasks) |
| `docs/API_DOCUMENTATION.md` | REST API and WebSocket reference |
| `docs/USER_GUIDE.md` | End-user guide |
| `docs/ASSUMPTIONS.md` | Trading system assumptions |
| `docs/RISK_MANAGEMENT.md` | Risk management system details |
| `docs/DEPLOYMENT_GUIDE.md` | Production deployment instructions |

---

## Frontend Pages

| Page | URL | Backend Endpoints |
|------|-----|-------------------|
| Home | `/` | `/api/status`, `/api/account` |
| Live | `/live` | `/api/positions`, `/ws/prices` |
| Analytics | `/analytics` | `/api/status`, chart data |
| Backtest | `/backtest` | `POST /api/backtest` |
| Walk-Forward | `/walk-forward` | `POST /api/backtest/walk-forward` |
| History | `/history` | `/api/trades` |
| Signals | `/signals` | `/api/signals` |

---

## Test Suites

| Suite | Location | Count | Description |
|-------|----------|-------|-------------|
| Unit | `tests/unit/` | 700+ | Component-level tests |
| Integration | `tests/integration/` | 50+ | Cross-component tests |
| Validation | `tests/validation/` | 116 | Hand-calculated expected values |
| Regression | `tests/regression/` | 21 | Golden results + benchmarks |
| E2E | `frontend/e2e/` | 20+ | Playwright browser tests |

---

*See `.context/history.md` for detailed development history.*
