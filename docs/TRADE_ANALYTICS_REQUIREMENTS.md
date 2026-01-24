# Trade Analytics & Visualization Requirements

> **Quick Summary**
> - **Building**: Trade analytics dashboard with performance metrics and interactive visualizations
> - **For**: Traders using FluxHero for algorithmic trading
> - **Core Features**: P&L tracking, equity curve, trade visualizations, performance metrics, export
> - **Tech**: FastAPI, NumPy, SQLite (WAL), Parquet, TradingView Lightweight Charts
> - **Scope**: Medium (3 phases)

## Overview

A trade analytics module that aggregates execution data from the existing backtesting and execution systems, computes performance metrics, and presents them through interactive visualizations. This extends the current Analytics page with deeper trade-level insights, adds backend endpoints for metrics computation, and provides rich visual trade analysis with entry/exit markers and trade boxes.

**Current State**: FluxHero has analytics only in backtest results. Live trading lacks visual trade analysis.

**Target State**: Both live and backtest should have identical analytics capabilities with rich visual trade analysis.

---

## Charting Library Decision

### Current: TradingView Lightweight Charts v5.1.0
- **License**: Apache 2.0 (free, open-source)
- **Size**: ~45KB gzipped
- **Already integrated** in `frontend/app/analytics/page.tsx`

### Decision: Continue with Lightweight Charts + Custom Plugins
- Already integrated and working
- v5 added powerful plugin system for custom primitives
- Professional-grade financial charting
- No licensing costs

---

## Must Have (MVP - Phase 1)

### Feature 1: Trade Journal Storage

- **Description**: Persistent storage of all executed trades with metadata for analytics queries
- **Acceptance Criteria**:
  - [ ] Trades stored in SQLite with WAL mode enabled (prevents `database is locked` errors)
  - [ ] Schema: trade_id, symbol, strategy_id, side, quantity, entry_price, exit_price, entry_time, exit_time, realized_pnl, fees, stop_loss, take_profit
  - [ ] Partial fills recorded as separate trades with parent_trade_id reference
  - [ ] Trade timestamps use broker-provided time as canonical source
  - [ ] Empty state: new users see "No trades yet" message with guidance
- **Technical Notes**:
  - Use existing SQLiteStore patterns from `backend/storage/`
  - Enable WAL mode: `PRAGMA journal_mode=WAL`
  - Index on (strategy_id, entry_time) for common query patterns

### Feature 2: Realized P&L Tracking

- **Description**: Calculate and display realized profit/loss with daily/weekly/monthly breakdowns
- **Acceptance Criteria**:
  - [ ] Daily P&L calculated (using trader's configured timezone, default UTC)
  - [ ] Weekly and monthly rollups computed from daily data
  - [ ] P&L displayed in both absolute dollars and percentage terms
  - [ ] Zero-trade periods show as $0, not missing data
  - [ ] Division by zero handled (display "N/A" or 0%)
- **Technical Notes**:
  - Pre-aggregate daily summaries table for performance
  - Use NumPy vectorized operations (Numba unnecessary for this scale)

### Feature 3: Trade Box Visualization

- **Description**: Draw colored boxes on charts showing trade entry, exit, stop loss, and take profit zones
- **Visual Design**:
  ```
  ┌─────────────────────────────────────┐ ← Take Profit (green dashed)
  │      TRADE BOX (semi-transparent)   │
  │      Green = Profit, Red = Loss     │
  ├─────────────────────────────────────┤ ← Entry Price (solid white)
  └─────────────────────────────────────┘ ← Stop Loss (red dashed)
     ↑                               ↑
  Entry Time                    Exit Time
  ```
- **Acceptance Criteria**:
  - [ ] Trade boxes render correctly showing entry/exit/stop/target
  - [ ] Profit trades: `rgba(16, 185, 129, 0.15)` (green)
  - [ ] Loss trades: `rgba(239, 68, 68, 0.15)` (red)
  - [ ] Entry line: `#FFFFFF` (white solid)
  - [ ] Stop loss: `#EF4444` (red dashed)
  - [ ] Take profit: `#10B981` (green dashed)
- **Technical Notes**:
  - Create custom primitive: `TradePrimitive` implementing `ISeriesPrimitive`
  - File: `frontend/lib/charts/TradePrimitive.ts`

### Feature 4: Enhanced Trade Markers

- **Description**: Clear buy/sell markers with price labels on the chart
- **Acceptance Criteria**:
  - [ ] Buy marker: Green upward arrow below candle, label "BUY @ $XX.XX"
  - [ ] Sell marker: Red downward arrow above candle, label "SELL @ $XX.XX"
  - [ ] Entry marker: Blue circle at entry point
  - [ ] Exit marker: Orange circle at exit point
- **Technical Notes**:
  - Use `createSeriesMarkers()` API from lightweight-charts
  - Colors: Buy `#10B981`, Sell `#EF4444`, Entry `#3B82F6`, Exit `#F59E0B`

### Feature 5: Equity Curve Visualization

- **Description**: Interactive chart showing cumulative P&L over time with drawdown overlay
- **Acceptance Criteria**:
  - [ ] Line chart of cumulative equity over time
  - [ ] Drawdown percentage displayed as filled area below equity line
  - [ ] Interactive tooltips showing date, equity value, drawdown %
  - [ ] Date range selector (1W, 1M, 3M, YTD, 1Y, All)
  - [ ] Server-side downsampling for datasets >1000 points (LTTB algorithm)
  - [ ] Benchmark comparison overlay (algo vs buy-and-hold)
- **Technical Notes**:
  - API returns max 1000 points after downsampling

### Feature 6: Trade History Table

- **Description**: Searchable, filterable table of all trades
- **Acceptance Criteria**:
  - [ ] Columns: Date, Symbol, Side, Quantity, Entry, Exit, P&L, Duration
  - [ ] Sortable by any column
  - [ ] Filters: date range, symbol, strategy, win/loss
  - [ ] Pagination (50 trades per page default)
  - [ ] Date range filter validated server-side (max 2 years per query)
  - [ ] Empty filter results show "No trades match filters" message
- **Technical Notes**:
  - Rate limit: 10 requests/minute per user

### Feature 7: CSV Export

- **Description**: Export trade history and summary metrics to CSV
- **Acceptance Criteria**:
  - [ ] Export filtered trade history (respects current filters)
  - [ ] Export daily/monthly summary data
  - [ ] Include all relevant columns with headers
  - [ ] Filename: `trades_2024-01-01_2024-12-31.csv`
  - [ ] Rate limited: 5 exports/minute per user
- **Technical Notes**:
  - Stream large exports to avoid memory issues

### Feature 8: Basic Trade Statistics

- **Description**: Summary statistics displayed on dashboard
- **Acceptance Criteria**:
  - [ ] Win rate (# winning trades / total trades)
  - [ ] Average win amount and average loss amount
  - [ ] Profit factor (gross profit / gross loss)
  - [ ] Total number of trades
  - [ ] Max drawdown (with date)
  - [ ] Handle edge cases: profit factor = "No losses" when zero losses
  - [ ] Display "Insufficient data" when <10 trades
- **Technical Notes**:
  - Compute on filtered dataset, update when filters change

---

## Should Have (Phase 2)

### Feature 9: Trade Detail Chart View

- **Description**: Click on any trade to see a dedicated chart view
- **Acceptance Criteria**:
  - [ ] Zoom to trade timeframe (entry - 10 bars to exit + 10 bars)
  - [ ] Show candlestick chart with trade box, markers, KAMA overlay, ATR bands
  - [ ] Price levels panel: entry, stop loss, take profit, exit
  - [ ] Trade info overlay: Symbol, Side, Quantity, Entry/Exit times, P&L, Holding period, Strategy, Market regime
  - [ ] "← Back to Trades" button
  - [ ] Previous/Next trade navigation arrows
- **Technical Notes**:
  - File: `frontend/app/trades/[id]/page.tsx`

### Feature 10: Live P&L Analysis Page

- **Description**: Real-time P&L analysis page for live trading
- **Acceptance Criteria**:
  - [ ] Cumulative P&L chart: Algo P&L (green) vs Buy-and-Hold benchmark (orange)
  - [ ] Return % comparison chart with three lines: Algo %, Benchmark %, Difference %
  - [ ] Summary statistics panel: Initial Capital, Current Value, Trading Days, Risk-Free Rate, Max Drawdown, Annualized Return, Sharpe/Sortino/Calmar Ratios
  - [ ] Daily breakdown table with all metrics
  - [ ] Real-time updates every 5 seconds
- **Technical Notes**:
  - File: `frontend/app/live/analysis/page.tsx`

### Feature 11: Risk Ratios

- **Description**: Calculate Sharpe, Sortino, and Calmar ratios
- **Acceptance Criteria**:
  - [ ] Sharpe ratio using daily returns, annualized (252 trading days)
  - [ ] Sortino ratio (downside deviation only)
  - [ ] Calmar ratio (annualized return / max drawdown)
  - [ ] Handle zero standard deviation (display "N/A")
  - [ ] Max drawdown calculated as peak-to-trough since inception
  - [ ] Tooltip explanations for each ratio
- **Technical Notes**:
  - Use incremental/rolling calculation for large histories
  - NumPy vectorized operations sufficient

### Feature 12: Real-time WebSocket Updates

- **Description**: Live dashboard updates during active trading
- **Acceptance Criteria**:
  - [ ] New trades appear in table within 5 seconds of execution
  - [ ] P&L and statistics update on new trades
  - [ ] Connection status indicator (connected/reconnecting/disconnected)
  - [ ] Exponential backoff on disconnect (1s, 2s, 4s, 8s, max 30s)
  - [ ] "Data may be stale" warning after 30s disconnect
  - [ ] Auth token validated on connection and every 15 minutes
- **Technical Notes**:
  - Use existing WebSocketFeed infrastructure from `backend/data/`

### Feature 13: Daily Trade Grouping

- **Description**: Group trades by date with expandable rows
- **Visual Design**:
  ```
  ▶ 2026-01-21  R: +$43.76 | U: -$563.12 | Trades: 4 | Day: -1.85%
  ▼ 2026-01-16  R: +$320.65| U: +$0.00  | Trades: 17| Day: +1.14%
     ├─ AAPL  100  10:34:16  184.47  00:18:21  182.21  -$178.85
     └─ ...
  ```
- **Acceptance Criteria**:
  - [ ] Expandable rows (click to expand/collapse)
  - [ ] Date header shows: Realized P&L, Unrealized P&L, Trade count, Daily return %
  - [ ] Color-coded: Green for positive days, red for negative
- **Technical Notes**:
  - File: `frontend/components/trading/DailyTradeGroup.tsx`

### Feature 14: Summary Footer Bar

- **Description**: Persistent footer bar showing aggregate statistics
- **Acceptance Criteria**:
  - [ ] Fields: Closed count, Open count, Realized P&L, Unrealized P&L, Total P&L, Return %, vs BAH, Annualized return %
  - [ ] Real-time updates
- **Technical Notes**:
  - File: `frontend/components/trading/TradeSummaryFooter.tsx`

### Feature 15: Trade Exclusion/Adjustment

- **Description**: Ability to exclude or mark trades for analysis purposes
- **Acceptance Criteria**:
  - [ ] Mark trades as "excluded" (test trades, errors)
  - [ ] Excluded trades hidden from metrics by default
  - [ ] Toggle to show/hide excluded trades
  - [ ] Exclusion is soft-delete (data preserved)
  - [ ] Audit log of exclusions with reason and timestamp
- **Technical Notes**:
  - Add `excluded` boolean and `exclusion_reason` columns

---

## Nice to Have (Phase 3 - Future)

### Feature 16: Trade Reason Visualization

- **Description**: Visual explanation of why a trade was made
- **Acceptance Criteria**:
  - [ ] Signal reason card: Entry trigger text, Strategy mode badge, Market regime badge
  - [ ] Technical indicators at entry: RSI, ADX, KAMA slope, ATR, R²
  - [ ] Risk parameters: Position size, Risk $, Risk %, Stop/Target, R:R ratio
  - [ ] Validation checks: Noise filter, Volume, Spread, Gap detection (PASSED/FAILED)
- **Technical Notes**:
  - File: `frontend/components/trading/TradeReasonCard.tsx`

### Feature 17: Performance Attribution

- **Description**: Break down P&L by strategy and symbol
- **Acceptance Criteria**:
  - [ ] P&L grouped by strategy with drill-down
  - [ ] P&L grouped by symbol with drill-down
  - [ ] Time period attribution (morning vs afternoon, day of week)

### Feature 18: Benchmark Comparison

- **Description**: Compare performance against market benchmarks (SPY/VTI)
- **Acceptance Criteria**:
  - [ ] Compare equity curve vs SPY/QQQ
  - [ ] Calculate alpha and beta
  - [ ] Display relative performance metrics

### Feature 19: Trade Tagging

- **Description**: User-defined tags for trade categorization
- **Acceptance Criteria**:
  - [ ] Add custom tags to trades (e.g., "momentum", "mean-reversion")
  - [ ] Filter and group by tags
  - [ ] Bulk tag operations

### Feature 20: Alert Thresholds

- **Description**: Configurable alerts for risk metrics
- **Acceptance Criteria**:
  - [ ] Set drawdown threshold alerts
  - [ ] Set consecutive loss alerts
  - [ ] In-app notification when threshold breached

### Feature 21: HTML Report Export

- **Description**: Export P&L analysis as standalone HTML file
- **Acceptance Criteria**:
  - [ ] Self-contained HTML with embedded CSS
  - [ ] Charts rendered as static images or SVG
  - [ ] Filename: `FluxHero_Analysis_{symbol}_{date}.html`
- **Technical Notes**:
  - File: `frontend/lib/exportHtml.ts`

---

## Out of Scope

The following are explicitly **not** part of this project:

- **AI-powered trade recommendations** - This is analytics, not advisory
- **Social/sharing features** - Personal analytics, not social platform
- **Broker integration for live sync** - Trades come from existing execution system
- **Tax lot accounting (FIFO/LIFO)** - Tax software territory
- **Mobile-native app** - Web dashboard is responsive but no native mobile app
- **Multi-account consolidation** - Single account view per dashboard instance

---

## Technical Constraints

### Storage
- SQLite with WAL mode for trade journal (handles concurrent read/write)
- Parquet with daily partitions for time-series data (existing pattern)
- Pre-aggregate daily summaries for query performance

### Performance
- API responses <500ms for filtered queries
- Max 2-year date range per query
- Server-side downsampling for charts (LTTB algorithm, max 1000 points)
- NumPy for calculations (Numba unnecessary for this scale)

### Security
- Rate limit export endpoints: 5/minute
- Rate limit general API: 60/minute
- Validate all date ranges server-side
- Sanitize trade notes/metadata on render (XSS prevention)
- WebSocket auth token validation on connect + every 15 minutes

### Frontend
- TradingView Lightweight Charts for all financial charts (already integrated)
- Extend existing Analytics page (no new pages for MVP)
- Loading states for all async operations
- Error states with actionable messages

### Dependencies (existing in project)
- NumPy, pandas - calculations
- FastAPI, uvicorn - API endpoints
- pyarrow - Parquet storage
- sqlite3 (stdlib) - trade journal
- TradingView Lightweight Charts - charting (already integrated)

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analytics/trades` | GET | Paginated trade history with filters |
| `/api/analytics/trades/{id}` | GET | Single trade details |
| `/api/analytics/trades/{id}` | PATCH | Update trade (exclusion, notes) |
| `/api/analytics/summary` | GET | Summary statistics for date range |
| `/api/analytics/equity-curve` | GET | Equity curve data (downsampled) |
| `/api/analytics/daily-pnl` | GET | Daily P&L for date range |
| `/api/analytics/export/csv` | GET | Export trades as CSV |
| `/api/trades/{id}/chart-data` | GET | OHLCV data around specific trade |
| `/api/live/analysis` | GET | P&L analysis for live trading |
| `/api/live/daily-summary` | GET | Trades grouped by date |
| `/ws/analytics` | WS | Real-time trade updates |

---

## File Structure

```
frontend/
├── app/
│   ├── live/
│   │   ├── page.tsx              # Existing - add link to analysis
│   │   └── analysis/
│   │       └── page.tsx          # NEW: Live P&L Analysis
│   └── trades/
│       └── [id]/
│           └── page.tsx          # NEW: Trade Detail Chart View
├── components/
│   └── trading/
│       ├── DailyTradeGroup.tsx   # NEW: Daily grouping component
│       ├── TradeSummaryFooter.tsx# NEW: Summary footer bar
│       └── TradeReasonCard.tsx   # NEW: Trade reason visualization
├── lib/
│   ├── charts/
│   │   ├── TradePrimitive.ts     # NEW: Trade box primitive
│   │   ├── tradeMarkers.ts       # NEW: Enhanced markers
│   │   └── index.ts              # Chart utilities export
│   └── exportHtml.ts             # NEW: HTML export

backend/
├── analytics/                     # NEW: Analytics module
│   ├── __init__.py
│   ├── metrics.py                # Metric calculators
│   ├── aggregator.py             # Daily/weekly/monthly aggregation
│   └── exporters.py              # CSV/JSON export
├── api/
│   └── server.py                 # Add new endpoints
```

---

## Color Reference

| Purpose | Color Name | Hex Value |
|---------|------------|-----------|
| Profit | profit-500 | #10B981 |
| Loss | loss-500 | #EF4444 |
| Entry | blue-500 | #3B82F6 |
| Exit | warning-500 | #F59E0B |
| Neutral | text-500 | #8B8A96 |
| Background | panel-700 | #1C1C28 |
| Grid | panel-500 | #21222F |

---

## Open Questions

These require human decision before implementation:

1. **Unrealized P&L in MVP?**
   - PM suggests deferring (adds complexity)
   - Tech argues it's simple if position state exists
   - **Decision needed**: Check if execution system exposes position state. If yes, include in MVP.

2. **Timezone handling**
   - Daily P&L: whose timezone? Trader's local? Exchange? UTC?
   - **Recommendation**: Default to UTC, allow user preference setting in Phase 2

3. **Data retention policy**
   - How long to keep trade history?
   - **Recommendation**: No automatic deletion, user can manually archive

4. **Expected data volume**
   - Trades per day and total historical trades affects storage/query design
   - **Need from user**: Expected scale to validate SQLite is sufficient

---

## Definition of Done

A phase is complete when:
- [ ] All acceptance criteria pass
- [ ] Unit tests written with >80% coverage for new code
- [ ] Integration tests for API endpoints
- [ ] Features work identically for live and backtest data
- [ ] No P0/P1 bugs open
- [ ] Performance benchmarks met (<500ms API response)
- [ ] Code reviewed and merged to main

---

*Document version: 2.0*
*Last updated: 2026-01-23*
*Status: Ready for implementation*
