# FluxHero Context Index

## Current Status
**Phase:** 11 - Order Execution Engine
**Task:** Implement order_manager.py
**Progress:** 87/118 tasks (~74%)

## Quick Navigation

### Load On Demand (use @ prefix)
| Resource | Path | When to Load |
|----------|------|--------------|
| Current task spec | `@FLUXHERO_REQUIREMENTS.md` | For Feature 10 specs |
| Task list | `@FLUXHERO_TASKS.md` | To see all tasks |
| Architecture | `@.context/discovery/architecture.md` | When exploring codebase |

### Available Modules

| Module | Status | Location | Key Functions |
|--------|--------|----------|---------------|
| Computation | Done | `backend/computation/` | calculate_ema, calculate_rsi, calculate_kama |
| Strategy | Done | `backend/strategy/` | detect_regime, generate_signals, apply_noise_filter |
| Storage | Done | `backend/storage/` | SQLiteStore, ParquetStore, CandleBuffer |
| Data | Done | `backend/data/` | AsyncAPIClient, WebSocketFeed |
| Backtesting | Done | `backend/backtesting/` | BacktestEngine, calculate_metrics |
| Execution | **In Progress** | `backend/execution/` | BrokerBase (done), OrderManager (next) |
| Risk | Not Started | `backend/risk/` | - |
| API | Not Started | `backend/api/` | - |

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

## Authority Hierarchy (if conflicts)
1. FLUXHERO_REQUIREMENTS.md (highest - what to build)
2. FLUXHERO_TASKS.md (how to build it)
3. Existing code (current implementation)
4. discovery/architecture.md (summary - may be stale)

---
*Last updated: 2026-01-21*
