# FluxHero Context Index

## Current Status
**Phase:** 24 - Quality Control & Validation (Complete)
**Recent:** Walk-forward testing, validation suites, logging enhancements
**Progress:** ~95% complete (see CHANGELOG.md for details)

## Quick Navigation

### Load On Demand (use @ prefix)
| Resource | Path | When to Load |
|----------|------|--------------|
| Current task spec | `@FLUXHERO_REQUIREMENTS.md` | For feature specs |
| Task list | `@FLUXHERO_TASKS.md` | To see all tasks |
| Change log | `@CHANGELOG.md` | For recent changes |
| Architecture | `@.context/discovery/architecture.md` | When exploring codebase |
| Requirements | `@TRADE_ANALYTICS_REQUIREMENTS.md` |
| Assumptions | `@docs/ASSUMPTIONS.md` | For trading system assumptions |

### Available Modules

| Module | Status | Location | Key Functions |
|--------|--------|----------|---------------|
| Computation | Done | `backend/computation/` | calculate_ema, calculate_rsi, calculate_kama |
| Strategy | Done | `backend/strategy/` | detect_regime, generate_signals, apply_noise_filter |
| Storage | Done | `backend/storage/` | SQLiteStore, ParquetStore, CandleBuffer |
| Data | Done | `backend/data/` | AsyncAPIClient, WebSocketFeed, validate_ohlcv_data |
| Backtesting | Done | `backend/backtesting/` | BacktestEngine, walk_forward, calculate_metrics |
| Execution | Done | `backend/execution/` | BrokerBase, OrderManager, PositionSizer |
| Risk | Done | `backend/risk/` | PositionLimits, KillSwitch |
| API | Done | `backend/api/` | FastAPI server, backtest endpoints, walk-forward |
| Frontend | Done | `frontend/` | 7 pages: Home, Live, Analytics, Backtest, Walk-Forward, History, Signals |

## Rules (Always Apply)
- Use Numba @njit for performance-critical code
- Type hints for all functions
- Unit tests with benchmarks for each feature
- Async for all I/O operations (httpx, asyncio)
- Search codebase before loading documentation

## Context Management
1. **Load only what you need** - Don't load multiple detail files at once
2. **Search before loading** - Use Grep/Read to find specific code
3. **Reference existing code** - Look at broker_interface.py for patterns
4. **When stuck** - Search codebase first, load docs only if needed

## State Files
- Current state: `.context/state/current.yaml`
- Checksums: `.context/state/checksums.yaml`

## Test Suites

| Suite | Location | Description |
|-------|----------|-------------|
| Unit | `tests/unit/` | Component-level tests |
| Integration | `tests/integration/` | Cross-component tests |
| Validation | `tests/validation/` | Hand-calculated expected values |
| Regression | `tests/regression/` | Golden results + benchmarks |
| E2E | `frontend/e2e/` | Playwright browser tests |

## Authority Hierarchy (if conflicts)
1. FLUXHERO_REQUIREMENTS.md (highest - what to build)
2. FLUXHERO_TASKS.md (how to build it)
3. Existing code (current implementation)
4. discovery/architecture.md (summary - may be stale)

---
*Last updated: 2026-01-23*
