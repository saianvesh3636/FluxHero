# FluxHero Frontend Redesign - Implementation Tasks

## Overview

This document outlines the implementation roadmap for redesigning the FluxHero trading dashboard frontend. Tasks are organized by phase with dependencies noted.

**Scope:** All 6 pages (Home, Live, Analytics, Backtest, History, Signals)

**Status:** COMPLETE - All phases implemented

---

## Phase 1: Foundation Setup

### 1.1 Tailwind CSS Configuration
- [x] Install Tailwind CSS v4 and dependencies
- [x] Create `app/globals.css` with CSS-based Tailwind v4 configuration
- [x] Configure custom colors (text, panel, profit, loss, accent) using @theme directive
- [x] Configure custom border-radius (22px default)
- [x] Configure custom spacing scale
- [x] Configure font families (Inter, JetBrains Mono)
- [x] Update `postcss.config.mjs`

### 1.2 Global Styles
- [x] Replace `globals.css` with Tailwind directives
- [x] Add CSS custom properties for design tokens
- [x] Configure base styles (body background, default text)
- [x] Add Inter and JetBrains Mono font imports via next/font
- [x] Remove all existing custom CSS classes
- [x] Add utility classes for common patterns (tabular-nums, truncate)

### 1.3 Install Dependencies
- [x] Install `clsx` for conditional classes
- [x] Install `tailwind-merge` for class merging
- [x] Create `lib/utils.ts` with `cn()` helper function
- [x] Configure Inter font from next/font/google
- [x] Configure JetBrains Mono font from next/font/google

**Deliverable:** Tailwind v4 configured with all design tokens, ready for component development.

---

## Phase 2: Base UI Components

### 2.1 Card Component
- [x] Create `components/ui/Card.tsx`
- [x] Implement base card with panel-600 background, 22px radius, 20px padding
- [x] Add variant: `elevated` (panel-400 background)
- [x] Add variant: `highlighted` (panel-100 background)
- [x] No hover effects (performance)
- [x] Export from `components/ui/index.ts`

### 2.2 Button Component
- [x] Create `components/ui/Button.tsx`
- [x] Implement variants: `primary`, `secondary`, `ghost`, `danger`
- [x] Implement sizes: `sm`, `md`, `lg`
- [x] Add loading state (static spinner icon, no animation)
- [x] Add disabled state styling
- [x] Add focus ring (2px accent-500)

### 2.3 Input Components
- [x] Create `components/ui/Input.tsx`
- [x] Style with panel-300 background, no border
- [x] Add focus ring styling
- [x] Create `components/ui/Select.tsx` dropdown
- [x] Create slider input helper in backtest page

### 2.4 Badge Component
- [x] Create `components/ui/Badge.tsx`
- [x] Implement status variants: `success`, `error`, `warning`, `info`, `neutral`
- [x] Add size variants: `sm`, `md`
- [x] Style with 20% opacity backgrounds + solid text

### 2.5 Table Component
- [x] Create `components/ui/Table.tsx`
- [x] Implement `Table`, `TableHeader`, `TableBody`, `TableRow`, `TableCell`, `TableHead`
- [x] Style header with panel-700 background
- [x] Add hover state for rows (panel-600)
- [x] Support right-alignment for numeric columns
- [x] Add `tabular-nums` to numeric cells

### 2.6 Status Indicator
- [x] Create `components/ui/StatusDot.tsx`
- [x] Implement color variants (connected, disconnected, connecting)
- [x] 8px dot size, no pulse animation
- [x] Optional text label

### 2.7 Skeleton Components
- [x] Create `components/ui/Skeleton.tsx`
- [x] Static skeleton (no shimmer animation)
- [x] Panel-500 background color
- [x] Variants for text, card, table row

**Deliverable:** Complete set of base UI components following design system.

---

## Phase 3: Layout Components

### 3.1 Navigation
- [x] Create `components/layout/Navigation.tsx`
- [x] Implement horizontal nav bar (56px height)
- [x] Panel-800 background
- [x] Logo/brand on left
- [x] Nav items with active state (accent-500 indicator)
- [x] System status indicator on right
- [x] Responsive: items overflow-x-auto on mobile

### 3.2 Page Container
- [x] Create `components/layout/PageContainer.tsx`
- [x] Max-width: 1280px, centered
- [x] Padding: 24px
- [x] Responsive padding adjustments

### 3.3 Page Header
- [x] Create `components/layout/PageHeader.tsx`
- [x] Title (text-3xl, font-semibold)
- [x] Optional subtitle
- [x] Optional action buttons area

### 3.4 Grid Layout
- [x] Create `components/layout/Grid.tsx`
- [x] Responsive grid with 20px gaps
- [x] Auto-fit columns based on breakpoints

### 3.5 Root Layout Update
- [x] Update `app/layout.tsx` with new providers
- [x] Apply panel-900 background to body
- [x] Include Navigation component
- [x] Remove old ThemeToggle (dark mode only)

**Deliverable:** Layout system ready for page implementation.

---

## Phase 4: Trading-Specific Components

### 4.1 Price Display
- [x] Create `components/trading/PriceDisplay.tsx`
- [x] Format with commas and decimals
- [x] Color coding for up/down changes
- [x] Tabular-nums styling
- [x] Optional currency symbol

### 4.2 P&L Display
- [x] Create `components/trading/PLDisplay.tsx`
- [x] Positive: profit-500 + "+" prefix
- [x] Negative: loss-500 + "-" prefix
- [x] Zero: text-500
- [x] Support dollar and percentage formats

### 4.3 Position Card
- [x] Create `components/trading/PositionCard.tsx`
- [x] Symbol, quantity, entry price
- [x] Current price with change indicator
- [x] P&L display (dollar and percent)
- [x] Side indicator (LONG/SHORT)

### 4.4 Account Summary
- [x] Create `components/trading/AccountSummary.tsx`
- [x] Equity, cash, buying power
- [x] Daily P&L, total P&L
- [x] Grid layout with metric cards

### 4.5 Positions Table
- [x] Create `components/trading/PositionsTable.tsx`
- [x] Use base Table components
- [x] Columns: Symbol, Side, Qty, Entry, Current, P&L, Actions
- [x] Sortable headers
- [x] Empty state

### 4.6 Trade History Row
- [x] Trade row functionality integrated in pages
- [x] Entry/exit times, prices
- [x] Symbol, side, shares
- [x] P&L with color coding
- [x] Strategy and regime badges

### 4.7 WebSocket Status
- [x] StatusDot component handles all status states
- [x] Connected/Disconnected/Connecting states
- [x] Integrated in Navigation component

**Deliverable:** Trading-specific components ready for page integration.

---

## Phase 5: Chart Components

### 5.1 Chart Library Architecture
- [x] Create `components/charts/` directory structure
- [x] Create `config/theme.ts` with CHART_COLORS mapped to Tailwind tokens
- [x] Create `config/constants.ts` with named dimensions (CHART_HEIGHT, LINE_WIDTH)
- [x] Create `types/index.ts` with TypeScript interfaces
- [x] Create `utils/dataTransformers.ts` for data conversion
- [x] Create `utils/formatters.ts` for price/volume formatting
- [x] Create `utils/colorUtils.ts` for color manipulation

### 5.2 Core Chart Components
- [x] Create `core/ChartContainer.tsx` base wrapper with loading state
- [x] Create `core/OHLCDataBox.tsx` for OHLC display on hover
- [x] Create `hooks/useChart.ts` for chart lifecycle management
- [x] Create `hooks/useSeries.ts` for series management

### 5.3 Composed Chart Components
- [x] Create `composed/CandlestickChart.tsx` with OHLC, volume, indicators
- [x] Create `composed/EquityCurveChart.tsx` with area fill, benchmark
- [x] Create `composed/LineChart.tsx` for simple line charts
- [x] Create `composed/PnLComparisonChart.tsx` with interactive legend
- [x] Create `composed/TradeDetailChart.tsx` with price lines and markers

### 5.4 Chart Features
- [x] OHLC data display on hover (OHLCDataBox)
- [x] Magnet crosshair mode for precise candle selection
- [x] Theme colors mapped to Tailwind design tokens
- [x] SSR-safe with dynamic imports
- [x] Smooth scrolling without jitter (separated data updates from chart creation)

**Deliverable:** Complete chart library using lightweight-charts v5.

---

## Phase 6: Page Redesigns

### 6.1 Home Page (`app/page.tsx`)
- [x] Apply PageContainer layout
- [x] Redesign system status section
- [x] Create feature cards with new Card component
- [x] Add quick navigation grid to other pages
- [x] Add backend connectivity status with retry

### 6.2 Live Trading Page (`app/live/page.tsx`)
- [x] Apply PageContainer layout
- [x] Implement AccountSummary component
- [x] Implement PositionsTable component
- [x] Add system status indicator in header
- [x] Style refresh indicator
- [x] Add empty state for no positions
- [x] Backend offline state with retry

### 6.3 Analytics Page (`app/analytics/page.tsx`)
- [x] Apply PageContainer layout
- [x] Integrate styled TradingChart
- [x] Add symbol/timeframe selectors
- [x] Implement IndicatorPanel
- [x] WebSocket connection status
- [x] Auto-refresh with interval

### 6.4 Backtest Page (`app/backtest/page.tsx`)
- [x] Apply PageContainer layout
- [x] Redesign configuration form
  - [x] Styled Input, Select components
  - [x] Form sections with Card backgrounds
- [x] Strategy parameters section with sliders
- [x] Risk management section
- [x] Results section with metrics
- [x] Trade log table with CSV export

### 6.5 History Page (`app/history/page.tsx`)
- [x] Migrate to App Router
- [x] Apply PageContainer layout
- [x] Implement trade log table with expandable rows
- [x] Pagination with Button components
- [x] Export button (CSV)

### 6.6 Signal Archive Page (`app/signals/page.tsx`)
- [x] Migrate to App Router
- [x] Apply PageContainer layout
- [x] Filter controls section (symbol, strategy, regime, date range)
- [x] Signal table with sortable columns
- [x] Search input styling
- [x] Detail view modal
- [x] CSV export

**Deliverable:** All pages redesigned with consistent styling.

---

## Phase 7: Polish & Cleanup

### 7.1 Remove Old Styles
- [x] Delete unused CSS from `globals.css`
- [x] Delete ThemeToggle component (dark only)
- [x] Delete LoadingSpinner component (replaced by Skeleton)
- [x] Delete WebSocketStatus component (replaced by StatusDot)
- [x] Remove light mode CSS variables

### 7.2 Component Exports
- [x] Create barrel exports (`index.ts`) for all component folders
- [x] Update imports across all pages
- [x] Remove duplicate component definitions

### 7.3 Error States
- [x] Update ErrorBoundary with new styling
- [x] Update ErrorFallback component
- [x] Style error messages consistently with Card component

### 7.4 Loading States
- [x] Skeleton component for static loading (no animation)
- [x] Consistent loading patterns across pages ("Loading...")

### 7.5 Route Updates
- [x] Update Navigation to point to `/signals` instead of `/signal-archive`
- [x] Remove old pages router files

**Deliverable:** Polished, consistent UI across entire application.

---

## Phase 8: Testing & Documentation

### 8.1 Unit Tests
- [x] Update `app/__tests__/page.test.tsx` for new design
- [x] Update `app/live/__tests__/page.test.tsx` for new design
- [x] Update `app/backtest/__tests__/page.test.tsx` for new design
- [x] All 46 tests passing

### 8.2 Test Updates for New Components
- [x] Update selectors for StatusDot instead of emoji indicators
- [x] Update text expectations (Loading..., Connected, Backend Offline)
- [x] Update currency format expectations (+$X,XXX.XX format)

### 8.3 Build Verification
- [x] Build successful with all 6 routes
- [x] No route conflicts between App Router and Pages Router
- [x] All pages rendering correctly

### 8.4 Documentation
- [x] Update FRONTEND_REQUIREMENTS.md with implementation status
- [x] Update FRONTEND_TASKS.md with completion status
- [x] Document design tokens in globals.css

**Deliverable:** Fully tested redesign ready for production.

---

## Task Dependencies

```
Phase 1 (Foundation)
    │
    ▼
Phase 2 (Base UI) ──────┐
    │                   │
    ▼                   ▼
Phase 3 (Layout)    Phase 4 (Trading)
    │                   │
    └───────┬───────────┘
            │
            ▼
      Phase 5 (Charts)
            │
            ▼
      Phase 6 (Pages)
            │
            ▼
      Phase 7 (Polish)
            │
            ▼
      Phase 8 (Testing)
```

---

## File Changes Summary

### New Files
```
components/ui/Card.tsx
components/ui/Button.tsx
components/ui/Input.tsx
components/ui/Select.tsx
components/ui/Badge.tsx
components/ui/Table.tsx
components/ui/StatusDot.tsx
components/ui/Skeleton.tsx
components/ui/index.ts
components/layout/Navigation.tsx
components/layout/PageContainer.tsx
components/layout/PageHeader.tsx
components/layout/Grid.tsx
components/layout/index.ts
components/trading/PriceDisplay.tsx
components/trading/PLDisplay.tsx
components/trading/PositionCard.tsx
components/trading/AccountSummary.tsx
components/trading/PositionsTable.tsx
components/trading/OrderEntryModal.tsx
components/trading/index.ts
components/charts/config/theme.ts
components/charts/config/constants.ts
components/charts/types/index.ts
components/charts/core/ChartContainer.tsx
components/charts/core/OHLCDataBox.tsx
components/charts/hooks/useChart.ts
components/charts/hooks/useSeries.ts
components/charts/hooks/useCrosshairOHLC.ts
components/charts/composed/CandlestickChart.tsx
components/charts/composed/EquityCurveChart.tsx
components/charts/composed/LineChart.tsx
components/charts/composed/PnLComparisonChart.tsx
components/charts/composed/TradeDetailChart.tsx
components/charts/utils/dataTransformers.ts
components/charts/utils/formatters.ts
components/charts/utils/colorUtils.ts
components/charts/index.ts
lib/utils.ts
app/history/page.tsx (migrated from pages/)
app/signals/page.tsx (migrated from pages/)
```

### Modified Files
```
app/globals.css (Tailwind v4 CSS-based config)
app/layout.tsx
app/page.tsx
app/live/page.tsx
app/analytics/page.tsx
app/backtest/page.tsx
components/charts/TradingChart.tsx (lightweight-charts v5)
components/ErrorBoundary.tsx
components/ErrorFallback.tsx
app/__tests__/page.test.tsx
app/live/__tests__/page.test.tsx
app/backtest/__tests__/page.test.tsx
postcss.config.mjs
```

### Deleted Files
```
components/ThemeToggle.tsx (dark mode only, removed)
components/LoadingSpinner.tsx (replaced by Skeleton)
components/WebSocketStatus.tsx (replaced by StatusDot)
pages/history.tsx (migrated to app/)
pages/signal-archive.tsx (migrated to app/)
```

---

## Quick Start Commands

```bash
# Install dependencies (already done)
npm install

# Start development
npm run dev

# Run tests
npm test

# Run build
npm run build
```

---

## Progress Tracking

| Phase | Status | Progress |
|-------|--------|----------|
| Phase 1: Foundation | Complete | 100% |
| Phase 2: Base UI | Complete | 100% |
| Phase 3: Layout | Complete | 100% |
| Phase 4: Trading | Complete | 100% |
| Phase 5: Charts | Complete | 100% |
| Phase 6: Pages | Complete | 100% |
| Phase 7: Polish | Complete | 100% |
| Phase 8: Testing | Complete | 100% |

**Overall Progress:** 100%

---

## Notes

- **No animations**: Performance-critical trading environment
- **Dark mode only**: Industry standard for trading platforms
- **Speed first**: Minimal re-renders, static skeletons
- **Tailwind CSS v4**: CSS-based configuration with @theme directive
- **lightweight-charts v5**: Updated API (addSeries, createSeriesMarkers)
- **Design reference**: [Felix Luebken Resume Page](https://github.com/felixluebken/resume_page)

## Implementation Summary

The frontend redesign has been fully implemented with:
- Clean, minimal UI with no shadows, borders, or animations
- Dark mode only design system
- All 6 pages redesigned (Home, Live, Analytics, Backtest, History, Signals)
- Backend connections preserved (apiClient, WebSocketContext)
- All 46 unit tests passing
- Build successful with all routes
