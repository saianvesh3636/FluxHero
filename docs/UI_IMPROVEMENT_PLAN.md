# FluxHero UI Improvement Plan

**Created**: 2026-01-24
**Status**: Planning - Ready for Implementation
**Document Version**: 2.0

---

## HOW TO USE THIS DOCUMENT

**When context clears, read this document first.** It contains:
1. Complete task list with all requirements
2. Research findings on charting libraries
3. File locations and what to modify
4. Design principles and constraints
5. Acceptance criteria for each phase

**Start Command**: `Read /docs/UI_IMPROVEMENT_PLAN.md` then proceed with tasks.

---

## Executive Summary

### Project Focus
**FluxHero** is an AUTO-TRADING platform focused on:
- Automated strategy execution
- Idea validation and assumption testing
- Performance analysis vs benchmarks

**NOT** a manual trading platform like Sentinel Trader. Manual trading is secondary (for testing only).

### Key Improvements
1. **Compact Trades Page** - Reduce wasted space, dense information display
2. **Order Entry Modal** - Simple buy/sell for testing (not primary focus)
3. **Homepage Optimization** - Remove redundancy, redirect to /trades
4. **Charting Upgrade** - Migrate from lightweight-charts to **Plotly.js**
5. **Footer Metrics** - Add missing VTBAM%, VTI, Annualized Return
6. **Light Mode Toggle** - Optional theme switching

### Design Principles
1. **Information density over visual polish** - More data visible, less wasted space
2. **Function over aesthetics** - Clean but not fancy
3. **Auto-trading focus** - Manual trading is secondary
4. **Reusability** - Use existing components (SymbolSearch, etc.)
5. **Minimal changes** - Don't over-engineer

---

## Current State Analysis

### Tech Stack
- **Frontend**: Next.js 14+ with App Router
- **Styling**: Tailwind CSS (dark theme)
- **Current Charts**: TradingView Lightweight Charts
- **State**: React hooks + Context

### What Works Well
- P&L Analysis page (`/live/analysis`) - good information density example
- `SymbolSearch` component - already reusable across 3 pages (analytics, backtest, walk-forward)
- Dark theme with color-coded P&L values (green profit, red loss)
- `TradeSummaryFooter` - horizontal layout is compact
- Backend API fully supports buy/sell orders (just missing frontend UI)

### What Needs Improvement
- Trades page uses too much vertical space (large padding, gaps)
- Date group headers span 2-3 lines (should be 1 line)
- Homepage duplicates nav links (redundant)
- No manual order entry UI (needed for testing)
- Missing footer metrics (VTBAM%, VTI, Annualized)
- Charts lack built-in indicators and good labels

### Existing Reusable Components
| Component | Location | Used In |
|-----------|----------|---------|
| `SymbolSearch` | `/frontend/components/trading/SymbolSearch.tsx` | analytics, backtest, walk-forward |
| `PLDisplay` | `/frontend/components/trading/PLDisplay.tsx` | Multiple pages |
| `Button` | `/frontend/components/ui/Button.tsx` | All pages |
| `Card` | `/frontend/components/ui/Card.tsx` | All pages |
| `Badge` | `/frontend/components/ui/Badge.tsx` | Status indicators |

### Backend API (Already Exists - No Changes Needed)
```
POST /api/paper/orders  - Place paper trading order
POST /api/live/orders   - Place live trading order
GET  /api/live/analysis - Analysis data with metrics
GET  /api/paper/analysis - Paper analysis data
GET  /api/chart         - OHLCV candle data
```

### Frontend API Methods (Already Exist in `/frontend/utils/api.ts`)
```typescript
apiClient.placePaperOrder(symbol, qty, side, orderType, limitPrice)
apiClient.placeLiveOrder(symbol, qty, side, orderType, limitPrice, confirmLiveTrade)
apiClient.placeOrder(mode, symbol, qty, side, orderType, limitPrice)
apiClient.sellPosition(mode, symbol, qty, confirmLiveTrade)
```

---

## Charting Research Findings

### Current: TradingView Lightweight Charts
- **Bundle**: ~45KB (very small)
- **Indicators**: NONE built-in (must calculate manually)
- **Labels/Annotations**: Limited
- **License**: Apache 2.0 (open source)
- **Used in**: analytics, backtest, walk-forward, live/analysis pages

### Decision: Migrate to Plotly.js

**Plotly.js** chosen because:
- Built-in candlestick and OHLC chart types
- Excellent labels and annotations
- Subplots for MACD, Volume, RSI below main chart
- Declarative React API
- Open source (MIT license)

### Plotly.js Details

**Installation**:
```bash
npm install react-plotly.js plotly.js
```

**Next.js Integration** (IMPORTANT - must disable SSR):
```typescript
// Use dynamic import for Next.js
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });
```

**Bundle Size**: ~2MB minified (partial bundles available for smaller size)

**Features**:
- Candlestick charts: `type: 'candlestick'`
- Line charts with labels
- Multiple subplots (MACD below main chart)
- Zoom, pan, hover tooltips built-in
- Annotations for entry/exit markers

**Candlestick Example**:
```typescript
const data = [{
  type: 'candlestick',
  x: dates,
  open: openPrices,
  high: highPrices,
  low: lowPrices,
  close: closePrices,
  increasing: { line: { color: '#22C55E' } },  // profit green
  decreasing: { line: { color: '#EF4444' } },  // loss red
}];
```

**Multi-line with Labels Example**:
```typescript
const data = [
  { type: 'scatter', mode: 'lines', name: 'Algo P&L', x: dates, y: algoPnl, line: { color: '#22C55E' } },
  { type: 'scatter', mode: 'lines', name: 'BAH VTI P&L', x: dates, y: bahPnl, line: { color: '#EF4444' } },
  { type: 'scatter', mode: 'lines', name: 'Diff %', x: dates, y: diff, line: { color: '#3B82F6' } },
];
```

### Comparison Table

| Feature | Lightweight Charts | Plotly.js | TradingView Charting Lib |
|---------|-------------------|-----------|-------------------------|
| Candlestick | Yes | Yes | Yes |
| MACD/EMA/RSI | Manual calc | Manual calc (subplots) | Built-in (100+) |
| Labels/Annotations | Limited | Excellent | Good |
| Bundle Size | ~45KB | ~2MB | Licensed |
| Open Source | Yes | Yes | No |
| React Support | Community | Official | Official |
| Next.js SSR | Works | Needs dynamic import | Works |

### References
- Plotly Candlestick: https://plotly.com/javascript/candlestick-charts/
- Plotly Financial Charts: https://plotly.com/javascript/financial-charts/
- react-plotly.js: https://github.com/plotly/react-plotly.js
- Next.js Integration: https://dev.to/composite/how-to-integrate-plotlyjs-on-nextjs-14-with-app-router-1loj

---

## Task List

### Phase 1: Trades Page Compacting (HIGH PRIORITY)

#### Task 1.1: Reduce Global Spacing in Trades Page
**File**: `/frontend/app/trades/page.tsx`

**Changes**:
- [ ] Reduce card padding from `p-5` (20px) to `p-3` (12px)
- [ ] Reduce gap between sections from `gap-8` to `gap-4`
- [ ] Reduce header margin from `mb-8` to `mb-4`
- [ ] Reduce stats grid gap

**Acceptance Criteria**:
- Page fits more content without scrolling
- Maintains readability

---

#### Task 1.2: Compact DailyTradeGroup Headers
**File**: `/frontend/components/trading/DailyTradeGroup.tsx`

**Current Format** (2-3 lines):
```
2026-01-21
R: +43.76 | U: -563.12 | Trades: 4 | Day: -1.85%
```

**Target Format** (1 line):
```
▶ 2026-01-21 [R +43.76 / U -563.12] - Trades: 4 - Day: -1.85%
```

**Changes**:
- [ ] Combine date and stats into single flex row
- [ ] Use smaller text (text-sm instead of text-base)
- [ ] Reduce header padding (py-2 instead of py-3)
- [ ] Use inline separators (pipes/dashes) instead of vertical stacking
- [ ] Keep expand/collapse arrow functional

**Acceptance Criteria**:
- Each date group header is exactly 1 line
- All stats visible: Date, Realized, Unrealized, Trade count, Daily %
- Expand/collapse still works

---

#### Task 1.3: Compact Trade Table Rows
**File**: `/frontend/components/trading/DailyTradeGroup.tsx`

**Changes**:
- [ ] Reduce row padding (py-1 instead of py-2)
- [ ] Use text-xs for data cells, text-sm for headers
- [ ] Right-align numeric columns
- [ ] Use `tabular-nums` font feature for consistent number widths
- [ ] Ensure columns match reference: SYMBOL, QTY, ENTRY DATE, ENTRY PRICE, EXIT DATE, EXIT PRICE, COMM, PROFIT, %, Action

**Acceptance Criteria**:
- More trades visible per screen (target: 2x current density)
- Numbers align properly in columns
- All data still readable

---

#### Task 1.4: Compact PositionsTable
**File**: `/frontend/components/trading/PositionsTable.tsx`

**Changes**:
- [ ] Reduce header padding
- [ ] Reduce row padding (py-1.5 instead of py-3)
- [ ] Use smaller text (text-sm)
- [ ] Keep Sell button but make it smaller (px-2 py-1)
- [ ] Add Commission column if not present

**Acceptance Criteria**:
- Positions table takes 50% less vertical space
- All columns still visible and functional
- Sell button still easily clickable

---

#### Task 1.5: Add Missing Footer Metrics
**File**: `/frontend/components/trading/TradeSummaryFooter.tsx`

**Current metrics**: Closed, Open, Realized, Unrealized, Total, Return%, vs B&H

**Missing metrics to add**:
- VTBAM% (vs Buy and Hold Market percentage)
- VTI value (benchmark comparison absolute value)
- Annualized Return %

**Changes**:
- [ ] Add VTBAM% metric display
- [ ] Add VTI value display
- [ ] Add Annualized Return % display
- [ ] Check backend API provides these values (should be in `/api/live/analysis`)
- [ ] Handle overflow on smaller screens (horizontal scroll or wrap to 2 rows)

**Target Footer Layout**:
```
Closed: 90 | Open: 2 | Realized: $1374.21 | Unrealized: -$563.12 | Total: $811.09 | Return: 2.89% | VTBAM: 1.27% | VTI: $341.47 | Ann: 73.79%
```

**Acceptance Criteria**:
- Footer shows all 10 metrics
- Matches reference screenshot layout
- Responsive on smaller screens

---

### Phase 2: Order Entry Modal (MEDIUM PRIORITY)

#### Task 2.1: Create OrderEntryModal Component
**File**: `/frontend/components/trading/OrderEntryModal.tsx` (NEW FILE)

**Purpose**: Simple modal for placing buy/sell orders (for testing purposes)

**UI Structure**:
```
┌─────────────────────────────────────┐
│ Place Order                    [X]  │
├─────────────────────────────────────┤
│ Mode:    ○ Paper    ● Live          │
│                                     │
│ Symbol:  [____________] [Validate]  │
│          (reuse SymbolSearch)       │
│                                     │
│ Side:    [  BUY  ] [  SELL  ]       │
│          (toggle buttons)           │
│                                     │
│ Quantity: [____________]            │
│                                     │
│ Type:    ○ Market    ○ Limit        │
│                                     │
│ Price:   [$_________] (if limit)    │
│                                     │
├─────────────────────────────────────┤
│ Estimated Cost: $XX,XXX.XX          │
├─────────────────────────────────────┤
│ [Cancel]              [Place Order] │
└─────────────────────────────────────┘
```

**Props Interface**:
```typescript
interface OrderEntryModalProps {
  isOpen: boolean;
  onClose: () => void;
  defaultMode: 'paper' | 'live';
  onOrderPlaced?: () => void;  // callback to refresh positions
}
```

**Features**:
- [ ] Mode toggle (Paper/Live) - defaults to Paper for safety
- [ ] Symbol input using existing `SymbolSearch` component
- [ ] Side toggle buttons (Buy = green, Sell = red)
- [ ] Quantity number input with validation
- [ ] Order type radio (Market/Limit)
- [ ] Limit price input (shown only if Limit selected)
- [ ] Estimated cost calculation (qty × current price)
- [ ] Confirmation dialog for Live orders ("Are you sure? This uses real money")
- [ ] Loading state during submission
- [ ] Success/Error toast notifications
- [ ] Close on ESC key and backdrop click

**API Integration**:
```typescript
// Paper order
await apiClient.placePaperOrder(symbol, qty, side, orderType, limitPrice);

// Live order (with confirmation)
await apiClient.placeLiveOrder(symbol, qty, side, orderType, limitPrice, true);
```

**Acceptance Criteria**:
- Modal opens from Trades page
- Can place paper orders successfully
- Live orders require explicit confirmation
- Errors displayed clearly
- Modal closes after successful order
- Positions list refreshes after order

---

#### Task 2.2: Add "New Order" Button to Trades Page
**File**: `/frontend/app/trades/page.tsx`

**Changes**:
- [ ] Add state: `const [isOrderModalOpen, setIsOrderModalOpen] = useState(false);`
- [ ] Add "New Order" button in page header (next to existing Analysis button)
- [ ] Import and render OrderEntryModal component
- [ ] Pass current mode (Live/Paper) as default to modal

**Button Placement**: In the page header row, right side

**Acceptance Criteria**:
- Button visible in Trades page header
- Clicking opens the OrderEntryModal
- Current trading mode is passed to modal

---

### Phase 3: Homepage Optimization (LOW PRIORITY)

#### Task 3.1: Redirect Homepage to Trades
**File**: `/frontend/app/page.tsx`

**Current Problem**: Homepage shows feature cards (Trading, Analytics, Backtest, Signals) that duplicate the navigation bar links.

**Solution**: Redirect to /trades directly

**Implementation**:
```typescript
// /frontend/app/page.tsx
import { redirect } from 'next/navigation';

export default function Home() {
  redirect('/trades');
}
```

**Alternative** (if redirect causes issues):
```typescript
'use client';
import { useEffect } from 'react';
import { useRouter } from 'next/navigation';

export default function Home() {
  const router = useRouter();
  useEffect(() => {
    router.replace('/trades');
  }, [router]);
  return null;
}
```

**Acceptance Criteria**:
- Visiting `/` takes user directly to `/trades`
- No flash of homepage content
- Browser back button works correctly

---

### Phase 4: Charting Migration to Plotly.js (MEDIUM-HIGH PRIORITY)

#### Task 4.1: Install Plotly.js Dependencies
**File**: `/frontend/package.json`

**Commands**:
```bash
cd frontend
npm install react-plotly.js plotly.js
npm install --save-dev @types/react-plotly.js
```

**Acceptance Criteria**:
- Packages installed without errors
- TypeScript types available

---

#### Task 4.2: Create Plotly Chart Wrapper Component
**File**: `/frontend/components/charts/PlotlyChart.tsx` (NEW FILE)

**Purpose**: Reusable wrapper for Plotly charts with Next.js SSR handling

**Implementation**:
```typescript
'use client';

import dynamic from 'next/dynamic';
import { PlotParams } from 'react-plotly.js';

// Dynamic import to disable SSR (Plotly requires browser)
const Plot = dynamic(() => import('react-plotly.js'), {
  ssr: false,
  loading: () => <div className="h-full w-full bg-panel-700 animate-pulse rounded" />
});

interface PlotlyChartProps extends Partial<PlotParams> {
  className?: string;
}

export function PlotlyChart({ data, layout, config, className = '' }: PlotlyChartProps) {
  const defaultLayout = {
    paper_bgcolor: '#1C1C28',  // panel-700
    plot_bgcolor: '#1C1C28',
    font: { color: '#CCCAD5' },  // text-400
    margin: { t: 30, r: 30, b: 40, l: 50 },
    ...layout,
  };

  const defaultConfig = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    ...config,
  };

  return (
    <div className={`w-full ${className}`}>
      <Plot
        data={data}
        layout={defaultLayout}
        config={defaultConfig}
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
}

export default PlotlyChart;
```

**Acceptance Criteria**:
- Component renders without SSR errors
- Shows loading state while Plotly loads
- Applies dark theme colors by default

---

#### Task 4.3: Create Candlestick Chart Component
**File**: `/frontend/components/charts/CandlestickChart.tsx` (NEW FILE)

**Purpose**: Candlestick chart for analytics page

**Props**:
```typescript
interface CandlestickChartProps {
  data: {
    dates: string[];
    open: number[];
    high: number[];
    low: number[];
    close: number[];
    volume?: number[];
  };
  indicators?: {
    name: string;
    values: number[];
    color: string;
  }[];
  height?: number;
  showVolume?: boolean;
}
```

**Features**:
- [ ] Candlestick OHLC display
- [ ] Optional volume subplot below
- [ ] Optional indicator overlays (EMA, KAMA)
- [ ] Profit candles green (#22C55E), loss candles red (#EF4444)
- [ ] Crosshair on hover
- [ ] Zoom and pan

**Acceptance Criteria**:
- Renders candlestick chart with OHLC data
- Green/red coloring matches app theme
- Supports indicator overlays

---

#### Task 4.4: Create P&L Comparison Chart Component
**File**: `/frontend/components/charts/PnLComparisonChart.tsx` (NEW FILE)

**Purpose**: Multi-line chart for P&L analysis with labeled series

**Props**:
```typescript
interface PnLComparisonChartProps {
  data: {
    dates: string[];
    algoPnl: number[];
    benchmarkPnl: number[];
    diff?: number[];
  };
  labels?: {
    algo: string;      // default: "Algo P&L"
    benchmark: string; // default: "BAH VTI P&L"
    diff: string;      // default: "Diff %"
  };
  height?: number;
}
```

**Features**:
- [ ] Multiple line series with different colors
- [ ] Legend with series names
- [ ] Labels at end of lines showing current values
- [ ] Hover tooltips with date and all values
- [ ] Time range selector (1W, 1M, 3M, 6M, 1Y) via rangeselector

**Acceptance Criteria**:
- Shows algo vs benchmark comparison clearly
- Labels visible at line endpoints
- Matches P&L Analysis screenshot style

---

#### Task 4.5: Create Equity Curve Chart Component
**File**: `/frontend/components/charts/EquityCurveChart.tsx` (NEW FILE)

**Purpose**: Equity curve for backtest and walk-forward pages

**Props**:
```typescript
interface EquityCurveChartProps {
  data: {
    dates: string[];
    equity: number[];
    baseline?: number[];  // initial capital line
  };
  initialCapital?: number;
  height?: number;
}
```

**Features**:
- [ ] Line chart showing equity over time
- [ ] Optional baseline (initial capital) reference line
- [ ] Fill area under curve
- [ ] Hover tooltip with date and equity value

**Acceptance Criteria**:
- Clean equity curve visualization
- Shows gains/losses clearly

---

#### Task 4.6: Migrate Analytics Page to Plotly
**File**: `/frontend/app/analytics/page.tsx`

**Changes**:
- [ ] Remove lightweight-charts imports
- [ ] Import new `CandlestickChart` component
- [ ] Replace chart rendering logic
- [ ] Ensure all existing functionality preserved (symbol search, interval selection)

**Acceptance Criteria**:
- Analytics page renders candlestick chart using Plotly
- Symbol search still works
- Interval switching still works
- No regression in functionality

---

#### Task 4.7: Migrate Backtest Page to Plotly
**File**: `/frontend/app/backtest/page.tsx`

**Changes**:
- [ ] Remove lightweight-charts imports
- [ ] Import new `EquityCurveChart` component
- [ ] Replace EquityCurveChart local component with new one

**Acceptance Criteria**:
- Backtest results show equity curve using Plotly
- All other backtest functionality preserved

---

#### Task 4.8: Migrate Walk-Forward Page to Plotly
**File**: `/frontend/app/walk-forward/page.tsx`

**Changes**:
- [ ] Remove lightweight-charts imports
- [ ] Import new `EquityCurveChart` component
- [ ] Replace chart rendering logic

**Acceptance Criteria**:
- Walk-forward results show equity curve using Plotly

---

#### Task 4.9: Migrate Live Analysis Page to Plotly
**File**: `/frontend/app/live/analysis/page.tsx`

**Changes**:
- [ ] Remove lightweight-charts imports
- [ ] Import new `PnLComparisonChart` component
- [ ] Replace both charts (Cumulative P&L and Return % vs Date)
- [ ] Ensure all metrics and data table still work

**Acceptance Criteria**:
- P&L comparison charts render with Plotly
- Labels visible on chart lines
- All Summary Statistics still display
- Daily breakdown table still works

---

#### Task 4.10: Remove Lightweight Charts Dependency
**File**: `/frontend/package.json`

**Commands** (after all migrations complete):
```bash
npm uninstall lightweight-charts
```

**Changes**:
- [ ] Remove from package.json
- [ ] Remove any remaining imports
- [ ] Delete mock file: `/frontend/__mocks__/lightweight-charts.js`
- [ ] Update jest.config.js if needed

**Acceptance Criteria**:
- lightweight-charts completely removed
- All tests still pass
- No build errors

---

### Phase 5: Light Mode Toggle (OPTIONAL/FUTURE)

#### Task 5.1: Add Theme Toggle
**Files**:
- `/frontend/contexts/ThemeContext.tsx` (NEW)
- `/frontend/tailwind.config.ts`
- `/frontend/components/layout/Navigation.tsx`

**Changes**:
- [ ] Create ThemeContext for dark/light mode state
- [ ] Add light mode color tokens to Tailwind config
- [ ] Add toggle button in Navigation header (sun/moon icon)
- [ ] Persist preference in localStorage
- [ ] Update Plotly charts to respect theme

**Priority**: Low - only if users request it

---

## File Reference

### Files to Modify

| File | Phase | Changes |
|------|-------|---------|
| `/frontend/app/trades/page.tsx` | 1, 2 | Reduce spacing, add New Order button |
| `/frontend/components/trading/DailyTradeGroup.tsx` | 1 | Compact headers and rows |
| `/frontend/components/trading/PositionsTable.tsx` | 1 | Compact rows |
| `/frontend/components/trading/TradeSummaryFooter.tsx` | 1 | Add missing metrics |
| `/frontend/app/page.tsx` | 3 | Redirect to /trades |
| `/frontend/app/analytics/page.tsx` | 4 | Migrate to Plotly |
| `/frontend/app/backtest/page.tsx` | 4 | Migrate to Plotly |
| `/frontend/app/walk-forward/page.tsx` | 4 | Migrate to Plotly |
| `/frontend/app/live/analysis/page.tsx` | 4 | Migrate to Plotly |
| `/frontend/package.json` | 4 | Add Plotly, remove lightweight-charts |

### Files to Create

| File | Phase | Purpose |
|------|-------|---------|
| `/frontend/components/trading/OrderEntryModal.tsx` | 2 | Buy/sell order form modal |
| `/frontend/components/charts/PlotlyChart.tsx` | 4 | Base Plotly wrapper |
| `/frontend/components/charts/CandlestickChart.tsx` | 4 | OHLC candlestick chart |
| `/frontend/components/charts/PnLComparisonChart.tsx` | 4 | Multi-line P&L chart |
| `/frontend/components/charts/EquityCurveChart.tsx` | 4 | Equity curve chart |
| `/frontend/components/charts/index.ts` | 4 | Chart component exports |

---

## Acceptance Criteria Summary

### Phase 1 Complete When:
- [ ] Trades page shows 50%+ more content without scrolling
- [ ] Date group headers are single-line
- [ ] Footer shows all 10 metrics including VTBAM%, VTI, Annualized
- [ ] All existing functionality preserved

### Phase 2 Complete When:
- [ ] Can place paper buy/sell orders from Trades page modal
- [ ] Can place live orders with confirmation dialog
- [ ] Orders appear in positions/history after placement
- [ ] Modal uses existing SymbolSearch component

### Phase 3 Complete When:
- [ ] Visiting `/` redirects to `/trades`
- [ ] No redundant homepage content shown

### Phase 4 Complete When:
- [ ] All 4 chart pages migrated to Plotly.js
- [ ] Candlestick charts work with indicators
- [ ] P&L comparison charts have labeled lines
- [ ] lightweight-charts dependency removed
- [ ] No regression in chart functionality

### Phase 5 Complete When (Optional):
- [ ] Light/dark mode toggle in navigation
- [ ] Theme preference persisted
- [ ] All components respect theme

---

## Implementation Order

**Recommended sequence**:

1. **Phase 1** (Trades compacting) - Quick wins, visible improvement
2. **Phase 4.1-4.2** (Plotly setup) - Install and create base wrapper
3. **Phase 4.3-4.5** (Chart components) - Build reusable chart components
4. **Phase 4.6-4.9** (Migrations) - Replace charts one page at a time
5. **Phase 2** (Order modal) - Add buy/sell capability
6. **Phase 3** (Homepage redirect) - Simple cleanup
7. **Phase 4.10** (Cleanup) - Remove old dependencies
8. **Phase 5** (Theme) - Optional enhancement

---

## Testing Checklist

After each phase, verify:
- [ ] No console errors
- [ ] All pages load without crashing
- [ ] Existing functionality still works
- [ ] Responsive layout on different screen sizes
- [ ] Dark theme colors correct

---

## Notes

### Out of Scope
- Visual aesthetics changes (fancy icons, shadows, gradients)
- Sentinel Trader feature parity (different problem space - manual vs auto trading)
- Complex order types (stop-loss, take-profit, brackets, OCO)
- Real-time streaming prices in order form
- Mobile app considerations

### Key Differences: FluxHero vs Sentinel Trader
| Aspect | FluxHero | Sentinel Trader |
|--------|----------|-----------------|
| Focus | Auto-trading | Manual trading |
| Orders | For testing | Primary feature |
| UI Priority | Information density | Visual polish |
| Position sizing | Strategy-driven | User-controlled |

### Reference Screenshots Described
1. **Live Trades screenshot**: Compact table with expandable date groups, footer with metrics
2. **P&L Analysis screenshot**: Dense layout with two stacked charts, summary sidebar, data table

---

## Quick Start for New Session

```
1. Read this file: /docs/UI_IMPROVEMENT_PLAN.md
2. Check current progress (look for [ ] vs [x] in task lists)
3. Continue with next uncompleted task
4. Test changes before marking complete
5. Update checkboxes as tasks complete
```

---

## Changelog

- **v2.0** (2026-01-24): Added Plotly.js charting migration, expanded task details
- **v1.0** (2026-01-24): Initial plan with TradingView Charting Library (replaced)
