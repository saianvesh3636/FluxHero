```markdown
# FluxHero Audit Remediation Tasks

Fix critical issues, improve observability, and address architecture gaps identified in the 2026-01-21 audit.

**Execution Mode**: Sequential
**Source**: Audit from 2026-01-21

---

## Reference Documents

- PROJECT_AUDIT.md
- FLUXHERO_REQUIREMENTS.md
- docs/RISK_MANAGEMENT.md
- docs/API_DOCUMENTATION.md
- docs/DEPLOYMENT_GUIDE.md
- docs/MAINTENANCE_GUIDE.md

---

## Phase 1: Critical Storage Fixes

- [x] Replace bare `except Exception` in `backend/storage/parquet_store.py:344` with `except pyarrow.ArrowException`, add logging for failures
- [x] Replace bare `except Exception: pass` in `backend/storage/sqlite_store.py:243` `_write_worker` with `except sqlite3.Error`, log and handle explicitly
- [x] Implement `archive_old_trades()` in `backend/storage/sqlite_store.py:570` - export old records to Parquet, then delete from SQLite (per R7.1.3)
- [x] Add unit tests for exception handling in `parquet_store.py` and `sqlite_store.py`
- [x] Add integration test for `archive_old_trades()` verifying export and deletion

---

## Phase 2: Structured Logging

- [x] Add Python logging configuration module at `backend/core/logging_config.py` using standard library logging
- [x] Add structured logging to `backend/storage/parquet_store.py` (read/write operations, errors)
- [x] Add structured logging to `backend/storage/sqlite_store.py` (all database operations)
- [x] Replace all `print()` statements in `backend/api/server.py` with proper logging calls
- [x] Verify logging consistency with `backend/execution/order_manager.py` as reference pattern

---

## Phase 3: WebSocket Authentication

- [x] Create authentication middleware at `backend/api/auth.py` with token validation function
- [x] Update WebSocket endpoint in `backend/api/server.py:587-591` to validate token before `websocket.accept()`
- [ ] Add configuration for auth secret key (prepare for Phase 4 centralized config)
- [x] Add tests for WebSocket authentication (valid token, invalid token, missing token)

---

## Phase 4: Centralized Configuration

- [ ] Create `backend/core/config.py` using pydantic-settings with `.env` file support
- [ ] Move CORS origins from `backend/api/server.py:239-247` to config
- [ ] Move API URLs from `backend/daily_reboot.py:44` to config
- [ ] Move risk parameters from `backend/risk/position_limits.py:71-84` to config
- [ ] Create `.env.example` with all configurable values documented
- [ ] Update all modules to import from centralized config

---

## Phase 5: API Middleware & Observability

- [ ] Add request/response logging middleware to `backend/api/server.py`
- [ ] Add rate limiting middleware to API endpoints (adapt pattern from `backend/data/fetcher.py`)
- [ ] Create `/metrics` endpoint with Prometheus-compatible format (order counts, latency percentiles, drawdown %)
- [ ] Update `/health` endpoint to include basic system metrics
- [ ] Add tests for rate limiting and metrics endpoint

---

## Phase 6: Backtesting Evaluation

- [ ] Research third-party backtesting libraries (backtrader, vectorbt, zipline-reloaded) for compatibility
- [ ] Document evaluation criteria and findings in `docs/BACKTESTING_EVALUATION.md`
- [ ] If suitable library found, create integration plan as separate task list

---

## Notes

- Phase 1 is highest priority - silent failures in storage can cause data loss
- Phase 4 (config) should be done before production deployment
- Phase 6 is exploratory - do not implement backtesting SDK without validation
- All changes should maintain existing test coverage (49 test files)
```
