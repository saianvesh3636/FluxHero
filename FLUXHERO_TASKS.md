# FluxHero Implementation Tasks

This is the master task list for implementing FluxHero - an adaptive retail quant trading system.

**Execution Mode**: Sequential (one task at a time)
**Branch Strategy**: Create separate branch per task
**Target**: Full system implementation (Backend + Frontend)

---

## 📚 IMPORTANT: Context Documents (Read Before Each Task)

This task list provides **what** to build. The following documents explain **how** and **why**:

### Primary Reference Documents

1. **FLUXHERO_REQUIREMENTS.md** (MUST READ for each feature)
   - Detailed mathematical formulas (e.g., Efficiency Ratio, KAMA, ATR calculations)
   - Success criteria with specific metrics and targets
   - Performance benchmarks (e.g., <100ms for 10k candles)
   - Validation tests and edge cases
   - Architectural decisions and reasoning
   - **Usage**: Before implementing Feature X, read the corresponding "Feature X" section

2. **quant_trading_guide.md** (Technical reference)
   - EMA, RSI, MACD, ATR formulas and explanations
   - Alpha smoothing, volatility concepts
   - Adaptive indicators and regression techniques
   - When to use which indicators for different market conditions
   - **Usage**: Consult when implementing any technical indicator

3. **algorithmic-trading-guide.md** (Domain knowledge)
   - Risk management strategies (position sizing, drawdown limits)
   - Backtesting best practices (slippage, transaction costs)
   - Market microstructure (spreads, liquidity, order execution)
   - Realistic performance expectations
   - **Usage**: Reference for risk, execution, and backtesting tasks

### Workflow for Each Task

```
1. Read the task description below
2. Read the corresponding Feature section in FLUXHERO_REQUIREMENTS.md
3. Review relevant formulas in quant_trading_guide.md if needed
4. Consult algorithmic-trading-guide.md for domain-specific considerations
5. Implement with full context
6. Validate against success criteria in requirements
```

### Example

**Task**: "Implement adaptive EMA (KAMA) calculation"

**Before coding**:
- ✅ Read: FLUXHERO_REQUIREMENTS.md → "Feature 2: Adaptive EMA (KAMA-Based)"
  - Get: Efficiency Ratio formula, ASC calculation, regime thresholds
- ✅ Read: quant_trading_guide.md → "Section 9: Adaptive Scaling with Regression"
  - Get: Mathematical background on alpha adjustment
- ✅ Read: algorithmic-trading-guide.md → "Strategy Ideas"
  - Get: When KAMA works best (trending vs ranging markets)

**Result**: Complete implementation with proper formulas, edge cases, and validation tests.

---

## Phase 1: Project Setup & Core Infrastructure

- [x] Create project folder structure (fluxhero/backend/, fluxhero/frontend/, tests/, etc.)
- [x] Set up Python virtual environment and create requirements.txt with core dependencies (numba, numpy, pandas, httpx, websockets, fastapi, sqlite3, pyarrow)
- [x] Set up React + Next.js frontend project with TypeScript and configure API integration
- [x] Create .gitignore for Python and Node.js with data/ and .env exclusions

---

## Phase 2: Feature 1 - JIT Computation Engine

- [x] Implement backend/computation/indicators.py with Numba @njit decorator for basic indicator calculations
- [x] Create Numba-optimized EMA calculation function with type annotations (float64 arrays)
- [x] Create Numba-optimized RSI calculation function
- [x] Create Numba-optimized ATR (Average True Range) calculation function
- [x] Write unit tests for all indicators with performance benchmarks (target: 10k candles in <100ms)
- [x] Add @njit(cache=True) optimization to all JIT functions

---

## Phase 3: Feature 2 - Adaptive EMA (KAMA-Based)

**📖 READ FIRST**: FLUXHERO_REQUIREMENTS.md → "Feature 2: Adaptive EMA (KAMA-Based)" for formulas and success criteria

- [x] Implement backend/computation/adaptive_ema.py with Efficiency Ratio (ER) calculation (see R2.1.1 for formula: ER = |Change| / Sum|Price[i] - Price[i-1]|)
- [x] Implement adaptive smoothing constant (ASC) calculation using fast/slow smoothing constants (see R2.2.2: ASC = [ER × (SC_fast - SC_slow) + SC_slow]²)
- [x] Implement full KAMA (Kaufman Adaptive Moving Average) calculation with Numba optimization (see R2.2.3 for formula)
- [x] Add regime-aware adjustment logic (trending vs choppy markets based on ER thresholds: >0.6 trend, <0.3 choppy, see R2.3.1-2.3.3)
- [x] Write unit tests for KAMA with edge cases (perfect trend, pure noise, transitions) - see Success Criteria table in requirements
- [x] Create validation tests to ensure ER stays between 0-1 and ASC between SC_slow and SC_fast - see Mathematical Validation section

---

## Phase 4: Feature 3 - Volatility-Adaptive Smoothing

- [x] Implement backend/computation/volatility.py with 14-period ATR baseline calculation
- [x] Create volatility state classifier (low/normal/high based on ATR vs ATR_MA(50))
- [x] Implement dynamic period adjustment logic (shorten in high vol, lengthen in low vol)
- [x] Add multi-timeframe volatility checker (5-min vs 1-hour ATR comparison)
- [x] Implement volatility-alpha linkage using regression (α = f(ATR))
- [x] Write unit tests for volatility spike detection and period adjustments

---

## Phase 5: Feature 4 - Market Microstructure Noise Filter

- [x] Implement backend/strategy/noise_filter.py with spread-to-volatility ratio calculation
- [x] Create signal validation logic (reject if SV_Ratio > 0.05)
- [x] Add volume validation (require volume > 0.5× avg for signals, >1.5× for breakouts)
- [x] Implement time-of-day filter for illiquid hours (pre-market, lunch, after-hours)
- [x] Add special handling for illiquid periods (2× stricter SV_Ratio threshold)
- [x] Write unit tests for noise filter with various spread/volume scenarios

---

## Phase 6: Feature 5 - Regime Detection System

**📖 READ FIRST**: FLUXHERO_REQUIREMENTS.md → "Feature 5: Regime Detection System" for classification logic and thresholds

- [x] Implement backend/strategy/regime_detector.py with ADX (Average Directional Index) calculation (see R5.1.1: ADX >25 = trending, <20 = ranging)
- [x] Add linear regression slope and R² calculation for trend strength measurement (see R5.1.2: R² >0.7 = strong trend, <0.3 = no trend)
- [x] Create regime classification logic (STRONG_TREND / MEAN_REVERSION / NEUTRAL) - see R5.1.3 for combined ADX + R² logic
- [x] Implement volatility regime detection (HIGH_VOL / LOW_VOL based on ATR) - see R5.2.1 for thresholds (ATR > 1.5× ATR_MA)
- [x] Add regime persistence tracking with 3-bar confirmation to prevent whipsaws (see R5.2.3)
- [x] Add multi-asset correlation checker for portfolio strategies (see R5.3.1-3: correlation >0.8 = risk-on/risk-off)
- [x] Write unit tests for regime detection accuracy with historical market data - see Success Criteria table (>85% accuracy for trends)

---

## Phase 7: Feature 6 - Dual-Mode Strategy Engine

- [x] Implement backend/strategy/dual_mode.py with trend-following mode logic (entry: Price > KAMA + 0.5×ATR)
- [x] Add trend-following exit and trailing stop logic (2.5× ATR from peak)
- [x] Implement mean-reversion mode logic (entry: RSI < 30 AND price at lower Bollinger Band)
- [x] Add mean-reversion exit logic (return to 20-SMA OR RSI > 70)
- [x] Create neutral/transition mode with blended approach (50/50 weight, 30% size reduction)
- [x] Implement strategy performance tracking (win rate, Sharpe, drawdown per mode)
- [x] Add dynamic mode weight adjustment based on recent performance
- [x] Write unit tests for both strategy modes with backtested scenarios

---

## Phase 8: Feature 7 - Lightweight State Management

- [x] Create backend/storage/sqlite_store.py with trades, positions, and settings tables
- [x] Implement async write operations for trade logging (non-blocking)
- [x] Create daily rollover logic to archive old trades (keep last 30 days in SQLite)
- [x] Implement backend/storage/parquet_store.py for market data caching with snappy compression
- [x] Add cache validation logic (check if cached data <24 hours old on startup)
- [x] Create rolling 500-candle buffer in memory (discard older data)
- [x] Write unit tests for storage operations and cache hit/miss scenarios

---

## Phase 9: Feature 8 - Async API Wrapper

- [x] Implement backend/data/fetcher.py with httpx async REST client for OHLCV data
- [x] Add retry logic with exponential backoff (3 attempts: 1s, 2s, 4s delays)
- [x] Implement rate limiting to respect broker API limits (200 req/min for Alpaca)
- [x] Add connection pooling for HTTP efficiency
- [x] Create WebSocket live feed handler with auto-reconnect (max 5 retries)
- [x] Implement heartbeat monitor (alert if no data for >60 seconds)
- [x] Create data pipeline startup sequence (fetch 500 candles → open WebSocket → process ticks)
- [x] Write unit tests for API wrapper with mock responses and error scenarios

---

## Phase 10: Feature 9 - Backtesting Module

**📖 READ FIRST**: FLUXHERO_REQUIREMENTS.md → "Feature 9: Backtesting Module" for fill logic, slippage model, and success criteria
**📖 ALSO READ**: algorithmic-trading-guide.md → "Backtesting & Transaction Costs" for realistic modeling

- [x] Implement backend/backtesting/engine.py with backtest orchestrator
- [x] Create backend/backtesting/fills.py with next-bar fill logic (signal on bar N → fill at bar N+1 open, see R9.1.1 - simulates 60s delay)
- [x] Add slippage and commission model (0.01% slippage, $0.005/share commission, see R9.2.1-2.2.2)
- [x] Implement order size impact model (extra 0.05% slippage if >10% avg volume, see R9.2.3)
- [x] Create backend/backtesting/metrics.py with Sharpe ratio, max drawdown, win rate calculations (see R9.3.1 for formulas, risk-free rate = 4%)
- [x] Integrate quantstats library for detailed tearsheet and PDF export (see R9.3.3)
- [x] Implement walk-forward testing (3-month train, 1-month test, rolling windows, see R9.4.1-4.4 - must pass >60% of test periods)
- [x] Add validation tests (buy-and-hold SPY should match actual returns ±2%, see Validation Tests section)
- [x] Write comprehensive backtest unit tests with multiple scenarios - see Success Criteria table (Sharpe >0.8, Max DD <25%, Win Rate >45%)

---

## Phase 11: Feature 10 - Order Execution Engine

- [x] Create backend/execution/broker_interface.py with abstract broker methods (place_order, cancel_order, get_positions, etc.)
- [x] Implement PaperBroker class for testing (simulated fills)
- [x] Create backend/execution/order_manager.py with order heartbeat monitor (poll every 5s, cancel/rechase after 60s)
- [x] Implement backend/execution/position_sizer.py with 1% risk rule calculation
- [x] Add max position size limits (20% per position, 50% total deployment)
- [x] Implement safety kill-switch (close all positions if daily loss >3%)
- [x] Add order chasing logic (max 3 chase attempts, recalculate mid-price each time)
- [x] Write unit tests for order lifecycle and position sizing edge cases

---

## Phase 12: Feature 11 - Risk Management System

- [x] Implement backend/risk/position_limits.py with position-level risk checks (1% trend, 0.75% mean-rev)
- [x] Add portfolio-level exposure limits (max 50% deployed, max 5 positions)
- [x] Create correlation checker (reduce position size by 50% if correlation >0.7 with existing)
- [x] Write unit tests for risk limits with backtested worst-case scenarios
- [x] Implement backend/risk/kill_switch.py with drawdown circuit breakers (50% size cut at 15% DD, full stop at 20% DD)
- [x] Add real-time risk monitoring (current drawdown, exposure %, correlation matrix)
- [x] Create daily risk report generation (total risk deployed, worst-case loss scenario)

---

## Phase 13: Backend API Layer

- [x] Create backend/api/server.py with FastAPI setup and CORS configuration
- [x] Implement REST endpoints for positions (/api/positions GET)
- [x] Implement REST endpoints for trade history (/api/trades GET with pagination)
- [x] Implement REST endpoints for account info (/api/account GET)
- [x] Implement REST endpoints for system status (/api/status GET - heartbeat)
- [x] Implement REST endpoints for backtest execution (/api/backtest POST)
- [x] Add WebSocket endpoint for live price updates (/ws/prices)
- [x] Write API integration tests

---

## Phase 14: Feature 12 - Frontend Dashboard (React + Next.js)

### Tab A: Live Trading

- [x] Create frontend/pages/live.tsx with open positions table component
- [x] Add real-time P&L updates with color coding (green profit, red loss)
- [x] Implement system heartbeat indicator (🟢 Active / 🟡 Delayed / 🔴 Offline)
- [x] Add quick stats display (daily P&L, current drawdown %, total exposure %)
- [x] Set up 5-second auto-refresh for live data

### Tab B: Analytics

- [x] Create frontend/pages/analytics.tsx with TradingView-style candlestick chart using lightweight-charts library
- [x] Add KAMA line overlay and ATR bands to chart
- [x] Implement buy/sell signal annotations as arrows on chart
- [x] Create indicator panel showing real-time ATR, RSI, ADX, regime state
- [x] Add performance metrics display (total return %, Sharpe ratio, win rate, max drawdown)

### Tab C: Trade History

- [x] Create frontend/pages/history.tsx with trade log table component
- [x] Add pagination (20 trades per page)
- [x] Implement CSV export functionality
- [x] Add trade detail tooltips with signal explanations

### Tab D: Backtesting

- [x] Create frontend/pages/backtest.tsx with configuration form (date pickers, symbol selector, parameter sliders)
- [x] Implement "Run Backtest" button with loading spinner
- [x] Add results modal/page displaying backtest tearsheet
- [x] Integrate quantstats HTML export display

### UI/UX Polish

- [x] Implement dark mode toggle with persistent storage
- [x] Add responsive design for tablet support
- [x] Create loading states for all async operations
- [x] Add error boundaries for graceful error handling
- [x] Implement WebSocket connection for live price updates in frontend

---

## Phase 15: Signal Explainer & Logging

- [x] Create backend/strategy/signal_generator.py with signal explanation logging
- [x] Implement trade reason formatter (volatility state, regime, indicator values, risk calculation)
- [x] Add signal explanation storage to trades table
- [x] Create frontend tooltip component to display signal explanations on hover
- [x] Build signal explainer archive viewer for post-trade analysis

---

## Phase 16: Retail-Specific Optimizations

- [x] Add price gap filter to noise_filter.py (reject trades if |Open - Prev_Close| / Prev_Close > 0.02)
- [x] Implement "hourly close" optimization (check signal at minute 59, submit at minute 60)
- [ ] Create daily reboot script for 9:00 AM EST (fetch 500 candles, reconnect WebSocket)
- [ ] Add system operations documentation for maintenance tasks

---

## Phase 17: Testing & Validation

- [ ] Run full integration test (startup → data fetch → signal generation → simulated trade)
- [ ] Execute 1-year backtest on SPY with dual-mode strategy
- [ ] Validate backtest metrics meet minimum targets (Sharpe >0.8, Max DD <25%, Win Rate >45%)
- [ ] Test regime detection accuracy on historical trending/ranging periods
- [ ] Validate frontend displays correct real-time data from backend API
- [ ] Test all error scenarios (API failures, WebSocket disconnects, invalid data)
- [ ] Performance benchmark: ensure 10k candle indicator calculation completes in <500ms

---

## Phase 18: Documentation & Deployment

- [ ] Write API documentation (endpoint specs, request/response formats)
- [ ] Create user guide for running the system (setup, configuration, monitoring)
- [ ] Document risk management rules and circuit breaker behavior
- [ ] Add README with installation instructions for backend and frontend
- [ ] Create deployment guide for VPS/cloud hosting options
- [ ] Write maintenance guide (dependency updates, data backup, monitoring)

---

## Summary Stats

**Total Tasks**: 118 tasks across 18 phases
**Estimated Complexity**: Full-stack implementation with advanced quant features
**Key Technologies**: Python (Numba, FastAPI, httpx), React (Next.js, TypeScript), SQLite, Parquet

---

**Notes for Ralphy Execution**:
- Execute tasks sequentially to maintain dependencies
- Create separate git branch for each task
- Each task should include unit tests where applicable
- Follow FluxHero coding standards (Numba for performance-critical code, async operations, modular architecture)
