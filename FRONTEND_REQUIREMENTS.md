# Frontend-Backend Integration Fix Requirements

> **Quick Summary**
> - **Building**: Debug and fix frontend-backend connectivity to display live data instead of static text
> - **For**: Developers/traders using FluxHero dashboard
> - **Core Features**: API connectivity fix, SPY test data endpoint, Playwright E2E tests
> - **Tech**: Next.js 16 + React 19 frontend, FastAPI backend, Playwright
> - **Scope**: Small (diagnosis + targeted fixes)

## Overview

The FluxHero frontend (Next.js 16 + React 19) displays static text instead of live data from the backend (FastAPI on port 8000). The architecture is already in place—API client, WebSocket hooks, proxy rewrites—but something is preventing data flow. This task involves diagnosing the specific integration failure, adding a test data endpoint for offline development, and writing Playwright tests to verify functionality.

## Must Have (MVP)

### Feature 1: Diagnose and Fix Integration Issue
- Description: Identify why frontend displays static text despite having proper API client and backend endpoints
- Acceptance Criteria:
  - [ ] Backend server starts successfully on port 8000
  - [ ] Frontend proxy rewrites (`/api/*` → `localhost:8000/api/*`) function correctly
  - [ ] `apiClient.getPositions()`, `getAccountInfo()`, `getSystemStatus()` return data
  - [ ] Live page (`/live`) displays real position and account data
  - [ ] No CORS errors in browser console
  - [ ] No unhandled promise rejections or fetch failures
- Technical Notes:
  - Check if backend is running: `curl http://localhost:8000/api/status`
  - Check Next.js rewrites in `frontend/next.config.ts`
  - Verify API client base URL in `frontend/utils/api.ts`
  - Check browser Network tab for failed requests

### Feature 2: SPY Test Data Endpoint
- Description: Add backend endpoint serving static SPY historical data for development/testing
- Acceptance Criteria:
  - [ ] SPY CSV file (OHLCV format) committed to repo at `backend/test_data/spy_daily.csv`
  - [ ] `GET /api/test/candles?symbol=SPY` returns JSON array of candle data
  - [ ] Response format: `[{timestamp, open, high, low, close, volume}, ...]`
  - [ ] Endpoint only available when `ENV != production` (gated by environment check)
  - [ ] CSV parsed once at startup, cached in memory
- Technical Notes:
  - Download from Yahoo Finance: daily SPY data for past 1 year
  - Use pandas to load CSV, convert to list of dicts
  - Add `@app.get("/api/test/candles")` route in `server.py`
  - Gate with: `if os.getenv("ENV") == "production": raise HTTPException(404)`

### Feature 3: Frontend Error States
- Description: Ensure UI handles API errors gracefully instead of showing nothing
- Acceptance Criteria:
  - [ ] Loading spinner shown while fetching data
  - [ ] Error message displayed when API call fails (not blank screen)
  - [ ] "Backend offline" indicator when `/api/status` returns error
  - [ ] Retry button available on error states
- Technical Notes:
  - Components already have `LoadingSpinner.tsx`—verify it's being used
  - Check that `try/catch` blocks in page components set error state

### Feature 4: Playwright E2E Tests
- Description: Automated tests verifying frontend displays backend data correctly
- Acceptance Criteria:
  - [ ] Playwright installed as dev dependency
  - [ ] Test: Home page loads without errors
  - [ ] Test: `/live` page displays position data (not static placeholders)
  - [ ] Test: `/backtest` form submits and shows results
  - [ ] Test: Error state shown when backend is offline
  - [ ] Tests run in CI-compatible headless mode
  - [ ] `npm run test:e2e` script added to package.json
- Technical Notes:
  - Install: `npm install -D @playwright/test`
  - Create `frontend/e2e/` directory for test files
  - Use `page.waitForSelector()` to verify dynamic content
  - Mock backend with Playwright's route interception for offline tests

## Should Have (Post-MVP)

### WebSocket Connection Verification
- Description: Verify WebSocket price streaming works end-to-end
- Acceptance Criteria:
  - [ ] `WebSocketStatus` component shows "Connected" when backend is running
  - [ ] Price updates appear in real-time on analytics page
  - [ ] Auto-reconnect works after connection drop
- Technical Notes: WebSocket hook already implements reconnection logic—verify it works

### Test Data Seeding Script
- Description: Script to populate backend with realistic test data
- Acceptance Criteria:
  - [ ] `scripts/seed_test_data.py` creates sample positions and trades
  - [ ] Running script populates SQLite database with 5-10 sample positions
  - [ ] Positions have realistic P&L values
- Technical Notes: Use existing `SQLiteStore` to insert data

## Nice to Have (Future)

- **Simulated live price updates**: WebSocket endpoint that replays CSV data as if live (polling is acceptable for MVP)
- **Visual regression tests**: Playwright screenshot comparisons (skip for MVP—too brittle)
- **Multiple test symbols**: Add AAPL, MSFT test data beyond SPY

## Out of Scope

- Production deployment or environment configuration
- Real broker API integration or testing
- Mobile responsiveness fixes
- Performance optimization of data streaming
- Authentication UI (WebSocket auth is backend-only)
- New features beyond fixing existing integration

## Technical Constraints

| Constraint | Details |
|------------|---------|
| Backend port | Must run on 8000 (frontend proxy hardcoded) |
| Test endpoint security | Must be disabled in production via env check |
| SPY data size | Daily data only (~250 rows/year)—no minute data |
| Playwright CI | Tests must run headless without display server |

## Implementation Order

1. **Verify backend runs** → `make run-backend` or `uvicorn backend.api.server:app`
2. **Check proxy works** → Browser Network tab, curl from frontend container
3. **Fix specific failure** → CORS? Wrong URL? Backend not started?
4. **Add test endpoint** → After core integration works
5. **Add Playwright tests** → After manual verification passes

## Diagnosis Checklist

Before implementing fixes, verify these in order:

```bash
# 1. Backend running?
curl http://localhost:8000/api/status

# 2. Backend returns data?
curl http://localhost:8000/api/positions
curl http://localhost:8000/api/account

# 3. Frontend proxy works?
# Start frontend, open browser devtools Network tab
# Look for requests to /api/* and check responses

# 4. CORS headers present?
curl -I http://localhost:8000/api/status
# Should see Access-Control-Allow-Origin header
```

## Open Questions

- **Root cause unknown**: Is the issue that the backend isn't running, the proxy isn't configured, or there's a code bug in the frontend? Diagnosis required before fixes.
- **Test data source**: Should we commit a static CSV or fetch from Yahoo Finance during setup? *Recommendation: Commit static CSV for reproducibility.*

## Acceptance Test

The task is complete when:
1. `npm run dev` (frontend) + `uvicorn backend.api.server:app` (backend)
2. Navigate to `http://localhost:3000/live`
3. Page shows actual position/account data (not "Loading..." or static text)
4. `npm run test:e2e` passes all tests

---

*Prerequisites: Run diagnosis checklist before implementing. The fix may be as simple as starting the backend server.*
