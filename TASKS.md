# FluxHero Tasks

> **Quick Summary**
> - **Total Tasks**: 20+ across 6 phases
> - **Focus**: Parallel test execution + Frontend-Backend integration
> - **Key Deliverable**: All tests passing + Working frontend displaying live data
> - **Source**: FRONTEND_REQUIREMENTS.md, REQUIREMENTS.md

**Execution Mode**: Sequential within phases, Phases 1-2 parallel to Phases 3-6
**Last Updated**: 2026-01-22

---

## Reference Documents

- FRONTEND_REQUIREMENTS.md
- REQUIREMENTS.md
- docs/PARALLEL_TEST_DATABASE_ANALYSIS.md
- docs/PARALLEL_TEST_PORT_ANALYSIS.md
- PARALLEL_TEST_FINDINGS.md
- docs/API_DOCUMENTATION.md

---

## Phase 1: Parallel Test Fixes (Completed)

- [x] Run `pytest -n auto` and capture list of failing tests
- [x] Fix tests with shared file conflicts using `tmp_path` fixture
- [x] Fix tests with port binding conflicts using dynamic port allocation
- [x] Fix database tests with isolation issues
- [x] Verify all tests pass with `pytest -n auto`

## Phase 2: Parallel Test Enhancements (Completed)

- [x] Configure pytest-xdist for parallel execution
- [x] Add `make test` target to Makefile
- [x] Document parallel vs sequential execution

---

## Phase 3: Frontend-Backend Diagnosis (Do First)

### Task 3.1: Verify Backend Server
- [x] Backend starts successfully on port 8000
- [x] `curl http://localhost:8000/api/status` returns valid response
- [x] `curl http://localhost:8000/api/positions` returns data
- [x] `curl http://localhost:8000/api/account` returns data
- [x] CORS headers present (`Access-Control-Allow-Origin`)

### Task 3.2: Verify Frontend Proxy
- [x] Check `frontend/next.config.ts` rewrites configuration
- [x] Verify `/api/*` routes proxy to `localhost:8000/api/*`
- [x] Check browser Network tab for request/response flow
- [x] No CORS errors in browser console

### Task 3.3: Verify API Client
- [x] Check `frontend/utils/api.ts` base URL configuration
- [x] Verify `apiClient.getPositions()` makes correct request
- [x] Verify `apiClient.getAccountInfo()` makes correct request
- [x] Verify `apiClient.getSystemStatus()` makes correct request

---

## Phase 4: Frontend Integration Fixes (MVP)

### Task 4.1: Fix Integration Issue
**Goal**: Frontend displays live data instead of static text

- [x] Identify root cause (backend not running / proxy misconfigured / code bug)
- [x] Fix the specific integration failure
- [x] Live page (`/live`) displays real position data
- [x] Live page displays real account data
- [x] No unhandled promise rejections or fetch failures

### Task 4.2: SPY Test Data Endpoint
**Goal**: Backend endpoint serving static SPY data for development

- [x] Download SPY daily OHLCV data (1 year) from Yahoo Finance
- [x] Save to `backend/test_data/spy_daily.csv`
- [x] Add `GET /api/test/candles?symbol=SPY` endpoint in `server.py`
- [x] Response format: `[{timestamp, open, high, low, close, volume}, ...]`
- [x] Gate endpoint: disabled when `ENV=production`
- [x] Cache CSV data in memory at startup

### Task 4.3: Frontend Error States
**Goal**: UI handles API errors gracefully

- [ ] Loading spinner shown while fetching data
- [ ] Error message displayed when API call fails
- [ ] "Backend offline" indicator when `/api/status` fails
- [ ] Retry button available on error states
- [ ] Verify `LoadingSpinner.tsx` is being used correctly
- [ ] Verify `try/catch` blocks set error state properly

### Task 4.4: Playwright E2E Tests
**Goal**: Automated tests verifying frontend-backend integration

- [ ] Install Playwright: `npm install -D @playwright/test`
- [ ] Create `frontend/e2e/` directory
- [ ] Test: Home page loads without errors
- [ ] Test: `/live` page displays position data (not placeholders)
- [ ] Test: `/backtest` form submits and shows results
- [ ] Test: Error state shown when backend is offline
- [ ] Tests run in headless mode (CI-compatible)
- [ ] Add `npm run test:e2e` script to package.json

---

## Phase 5: Should Have (Post-MVP)

### Task 5.1: WebSocket Connection Verification
- [ ] `WebSocketStatus` component shows "Connected" when backend running
- [ ] Price updates appear in real-time on analytics page
- [ ] Auto-reconnect works after connection drop

### Task 5.2: Test Data Seeding Script
- [ ] Create `scripts/seed_test_data.py`
- [ ] Script creates 5-10 sample positions with realistic data
- [ ] Positions have realistic P&L values
- [ ] Add `make seed-data` command to Makefile

---

## Phase 6: Nice to Have (Future)

- [ ] Simulated live price updates (WebSocket replays CSV data)
- [ ] Multiple test symbols (AAPL, MSFT)
- [ ] Visual regression tests with Playwright screenshots

---

## Out of Scope

- Production deployment or environment configuration
- Real broker API integration or testing
- Mobile responsiveness fixes
- Performance optimization of data streaming
- Authentication UI (WebSocket auth is backend-only)
- New features beyond fixing existing integration

---

## Technical Constraints

| Constraint | Details |
|------------|---------|
| Backend port | Must run on 8000 (frontend proxy hardcoded) |
| Test endpoint security | Must be disabled in production via env check |
| SPY data size | Daily data only (~250 rows/year) |
| Playwright CI | Tests must run headless without display server |

---

## Quick Commands

```bash
# Start both backend and frontend
make dev

# Start backend only
make dev-backend

# Start frontend only
make dev-frontend

# Stop all services
make stop

# Run all tests (parallel)
make test

# Run diagnosis
curl http://localhost:8000/api/status
curl http://localhost:8000/api/positions
curl http://localhost:8000/api/account

# Run E2E tests (after setup)
cd frontend && npm run test:e2e
```

---

## Acceptance Criteria

The task is complete when:
1. `make dev` starts both frontend and backend
2. Navigate to `http://localhost:3000/live`
3. Page shows actual position/account data (not "Loading..." or static text)
4. `npm run test:e2e` passes all tests
