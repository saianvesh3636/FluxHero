# FluxHero Frontend Design System Requirements

## Overview

This document defines the design system requirements for the FluxHero trading dashboard frontend redesign. The goal is to create a clean, minimal, performance-focused UI that prioritizes data readability and speed.

**Design Philosophy:**
- Clean, flat design with no shadows or borders
- Dark-mode only (trading standard)
- Depth through color contrast, not visual effects
- No animations (performance-critical trading environment)
- Consistent spacing rhythm using atomic units
- Color-coded data categorization

**Tech Stack:**
- Next.js 16 (App Router)
- React 19
- TypeScript
- Tailwind CSS v4 (CSS-based configuration)
- lightweight-charts v5 (for trading charts)

---

## Implementation Status

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | Tailwind CSS v4 Setup | ✅ Complete |
| 2 | Base UI Components | ✅ Complete |
| 3 | Layout Components | ✅ Complete |
| 4 | Trading Components | ✅ Complete |
| 5 | Page Redesigns | ✅ Complete |
| 6 | Migration to App Router | ✅ Complete |
| 7 | Cleanup & Test Updates | ✅ Complete |

---

## 1. Color System

### 1.1 Text Colors (Light to Dark)

| Token | Hex | Usage |
|-------|-----|-------|
| `text-100` | `#716f7a` | Disabled, very subtle |
| `text-200` | `#8F8D98` | Muted labels |
| `text-300` | `#AEACB7` | Tertiary text, hints |
| `text-400` | `#CCCAD5` | Secondary labels, captions |
| `text-500` | `#eae8f3` | Body text |
| `text-600` | `#EFEEF6` | Emphasized body |
| `text-700` | `#F5F4F9` | Subheadings, values |
| `text-800` | `#FAF9FC` | Headings |
| `text-900` | `#ffffff` | Primary text, titles |

### 1.2 Panel/Background Colors (Dark to Light)

| Token | Hex | Usage |
|-------|-----|-------|
| `panel-900` | `#181621` | Main body background |
| `panel-800` | `#1A1924` | Navigation background |
| `panel-700` | `#1C1C28` | Chart backgrounds |
| `panel-600` | `#1E1F2B` | Card background (default) |
| `panel-500` | `#21222F` | Card dividers, borders |
| `panel-400` | `#232432` | Button backgrounds |
| `panel-300` | `#252735` | Input backgrounds |
| `panel-200` | `#272A39` | Hover states |
| `panel-100` | `#292D3C` | Highlighted panels |

### 1.3 Accent Colors (Trading Context)

**Profit/Positive (Green)**
| Token | Hex | Usage |
|-------|-----|-------|
| `profit-100` | `#4ADE80` | Light green accents |
| `profit-500` | `#22C55E` | Primary profit color |
| `profit-900` | `#16A34A` | Dark green accents |

**Loss/Negative (Red)**
| Token | Hex | Usage |
|-------|-----|-------|
| `loss-100` | `#F87171` | Light red accents |
| `loss-500` | `#EF4444` | Primary loss color |
| `loss-900` | `#DC2626` | Dark red accents |

**Primary Accent (Purple - Charts/Branding)**
| Token | Hex | Usage |
|-------|-----|-------|
| `accent-100` | `#C04DFE` | Light purple |
| `accent-500` | `#A549FC` | Primary accent (KAMA line, brand) |
| `accent-900` | `#8945FA` | Dark purple |

**Secondary Accent (Blue - Actions/Indicators)**
| Token | Hex | Usage |
|-------|-----|-------|
| `blue-100` | `#5790FC` | Light blue |
| `blue-500` | `#3E7AEE` | ATR bands, info badges |
| `blue-900` | `#2463E0` | Dark blue |

**Warning (Orange/Amber)**
| Token | Hex | Usage |
|-------|-----|-------|
| `warning-100` | `#FCD34D` | Light warning |
| `warning-500` | `#F59E0B` | Delayed status, warnings |
| `warning-900` | `#D97706` | Dark warning |

### 1.4 Status Colors

| Status | Color Token | Hex |
|--------|-------------|-----|
| Connected/Active | `profit-500` | `#22C55E` |
| Disconnected/Error | `loss-500` | `#EF4444` |
| Connecting/Delayed | `warning-500` | `#F59E0B` |
| Info/Neutral | `blue-500` | `#3E7AEE` |

---

## 2. Typography

### 2.1 Font Stack

```css
--font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
--font-mono: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
```

### 2.2 Font Sizes

| Token | Size | Line Height | Usage |
|-------|------|-------------|-------|
| `text-xs` | 12px | 16px | Captions, timestamps |
| `text-sm` | 14px | 20px | Labels, secondary text |
| `text-base` | 16px | 24px | Body text |
| `text-lg` | 18px | 28px | Subheadings |
| `text-xl` | 20px | 28px | Card titles |
| `text-2xl` | 24px | 32px | Section headings |
| `text-3xl` | 30px | 36px | Page titles |

### 2.3 Font Weights

| Token | Weight | Usage |
|-------|--------|-------|
| `font-normal` | 400 | Body text |
| `font-medium` | 500 | Labels, buttons |
| `font-semibold` | 600 | Headings, emphasis |
| `font-bold` | 700 | Strong emphasis |

### 2.4 Numeric Display

- Use `tabular-nums` for all trading data (prices, P&L, quantities)
- Use `font-mono` for order IDs, timestamps, and technical data
- Right-align numeric columns in tables

---

## 3. Spacing System

### 3.1 Base Unit

**Atomic unit: 4px** (Tailwind default)
**Primary unit: 20px (5 × 4px)** for major spacing

### 3.2 Spacing Scale

| Token | Value | Usage |
|-------|-------|-------|
| `gap-1` | 4px | Tight inline spacing |
| `gap-2` | 8px | Icon-text gaps |
| `gap-3` | 12px | Small component gaps |
| `gap-4` | 16px | Default gaps |
| `gap-5` | 20px | **Primary spacing unit** |
| `gap-6` | 24px | Section gaps |
| `gap-8` | 32px | Large section gaps |

### 3.3 Component Spacing

| Component | Padding | Gap |
|-----------|---------|-----|
| Cards | 20px (`p-5`) | - |
| Card Grid | - | 20px (`gap-5`) |
| Table Cells | 16px horizontal, 12px vertical | - |
| Buttons | 8px 16px (`py-2 px-4`) | 8px (icon) |
| Form Inputs | 12px 16px (`py-3 px-4`) | - |
| Nav Items | 8px 12px (`py-2 px-3`) | - |

---

## 4. Border Radius

| Token | Value | Usage |
|-------|-------|-------|
| `rounded` | 4px | Buttons, badges |
| `rounded-lg` | 8px | Inputs, small cards |
| `rounded-xl` | 12px | Metric cards |
| `rounded-2xl` | 22px | **Default cards/panels** |
| `rounded-full` | 9999px | Status dots, avatars |

**Standard:** Use `22px` (`rounded-2xl`) for all card/panel components.

---

## 5. Component Specifications

### 5.1 Card Component

```
Background: panel-600 (#1E1F2B)
Padding: 20px (p-5)
Border-radius: 22px (rounded-2xl)
Border: none
Shadow: none
Variant "highlighted": panel-500 background
Variant "noPadding": 0 padding (for tables)
```

### 5.2 Button Component

**Primary Button**
```
Background: accent-500 (#A549FC)
Hover: accent-900 (#8945FA)
Text: text-900 (#ffffff)
Padding: 8px 16px
Border-radius: 4px
Font-weight: 500
```

**Secondary Button**
```
Background: panel-400 (#232432)
Hover: panel-300 (#252735)
Text: text-700 (#F5F4F9)
Padding: 8px 16px
Border-radius: 4px
Font-weight: 500
```

**Danger Button**
```
Background: loss-500 (#EF4444)
Hover: loss-900 (#DC2626)
Text: text-900 (#ffffff)
```

**Ghost Button**
```
Background: transparent
Hover: panel-600 (#1E1F2B)
Text: text-400 (#CCCAD5)
```

### 5.3 Table Component

```
Header Background: panel-700 (#1C1C28)
Header Text: text-400 (#CCCAD5), uppercase, text-xs
Header Font-weight: 500
Row Background: transparent
Row Hover: panel-600 (#1E1F2B)
Cell Padding: py-3 px-4
Divider: none (clean look)
```

### 5.4 Input Component

```
Background: panel-300 (#252735)
Text: text-800 (#FAF9FC)
Placeholder: text-300 (#AEACB7)
Padding: 12px 16px (py-3 px-4)
Border-radius: 4px
Border: none
Focus: 2px ring accent-500
```

### 5.5 Select Component

```
Same as Input
Chevron: text-400
Options dropdown: panel-400 background
```

### 5.6 Badge Component

```
Padding: 4px 8px (px-2 py-0.5)
Border-radius: 4px
Font-size: text-xs
Font-weight: 500

Variants:
- success: bg-profit-500/20, text-profit-500
- error: bg-loss-500/20, text-loss-500
- warning: bg-warning-500/20, text-warning-500
- info: bg-blue-500/20, text-blue-500
- neutral: bg-panel-400, text-text-400
```

### 5.7 StatusDot Component

```
Size: 8px (sm), 12px (md), 16px (lg)
Border-radius: full
Animation: none

States:
- connected: bg-profit-500
- disconnected: bg-loss-500
- connecting: bg-warning-500
- error: bg-loss-500
```

### 5.8 Skeleton Component

```
Background: panel-500
Animation: none (static gray block)
Variants: text, title, rectangular
```

### 5.9 Navigation Component

```
Background: panel-800 (#1A1924)
Height: 56px (h-14)
Position: sticky top-0
z-index: 50

Active Item:
- Background: panel-600
- Text: text-900
- Left border: 2px accent-500

Inactive Item:
- Text: text-400
- Hover: text-700, bg-panel-700
```

---

## 6. Layout Components

### 6.1 AppShell

```
Structure: Navigation + Main content
Navigation: Sticky top, 56px height
Main: flex-1, overflow-y-auto
Background: panel-900
Min-height: 100vh
```

### 6.2 PageContainer

```
Max-width: 1280px (max-w-7xl)
Padding: 24px (p-6)
Margin: auto
```

### 6.3 PageHeader

```
Title: text-3xl font-bold text-text-900
Subtitle: text-text-400
Margin-bottom: 24px (mb-6)
Optional actions slot (right side)
```

### 6.4 Grid Components

**StatsGrid** - For metric cards
```
Columns: 1 (mobile) → 2 (sm) → 3 (md) → 4 (lg)
Gap: 20px (gap-5)
```

**CardGrid** - For feature cards
```
Columns: 1 (mobile) → 2 (md)
Gap: 20px (gap-5)
```

---

## 7. Trading Components

### 7.1 PriceDisplay

```
Price: text-text-900, font-mono tabular-nums
Change indicator: profit-500 (up) / loss-500 (down)
Arrow: ▲ / ▼ unicode characters
Size variants: sm, md, lg, xl
```

### 7.2 PLDisplay

```
Value: font-mono tabular-nums
Color: profit-500 (positive) / loss-500 (negative) / text-400 (zero)
Sign: Always show +/- prefix
Percent: Optional, shown in parentheses
```

### 7.3 PositionsTable

```
Columns: Symbol, Side, Qty, Entry, Current, P&L, P&L%
Side badge: success (long) / error (short)
P&L color-coded
Empty state: "No open positions" message
```

### 7.4 AccountSummary

```
Grid: 2 cols (mobile) → 3 cols (sm) → 5 cols (lg)
Items: Equity, Cash, Buying Power, Daily P&L, Exposure
P&L values color-coded
```

---

## 8. Chart Configuration

### 8.1 lightweight-charts v5 Setup

```typescript
const chart = createChart(container, {
  width: container.clientWidth,
  height: 500,
  layout: {
    background: { color: '#1C1C28' }, // panel-700
    textColor: '#CCCAD5', // text-400
  },
  grid: {
    vertLines: { color: '#21222F' }, // panel-500
    horzLines: { color: '#21222F' },
  },
  rightPriceScale: {
    borderColor: '#21222F',
  },
  timeScale: {
    borderColor: '#21222F',
    timeVisible: true,
  },
});
```

### 8.2 Candlestick Series

```typescript
chart.addSeries(CandlestickSeries, {
  upColor: '#22C55E', // profit-500
  downColor: '#EF4444', // loss-500
  borderVisible: false,
  wickUpColor: '#22C55E',
  wickDownColor: '#EF4444',
});
```

### 8.3 Line Series (Indicators)

```typescript
// KAMA Line
chart.addSeries(LineSeries, {
  color: '#A549FC', // accent-500
  lineWidth: 2,
  title: 'KAMA',
});

// ATR Bands
chart.addSeries(LineSeries, {
  color: '#3E7AEE', // blue-500
  lineWidth: 1,
  lineStyle: 2, // dashed
});
```

### 8.4 Signal Markers

```typescript
import { createSeriesMarkers } from 'lightweight-charts';

const markers = createSeriesMarkers(candlestickSeries, []);
markers.setMarkers([
  {
    time: timestamp,
    position: 'belowBar', // buy signal
    color: '#22C55E',
    shape: 'arrowUp',
    text: 'B',
  },
  {
    time: timestamp,
    position: 'aboveBar', // sell signal
    color: '#EF4444',
    shape: 'arrowDown',
    text: 'S',
  },
]);
```

---

## 9. Page Specifications

### 9.1 Home Page (`/`)

- System status card with StatusDot
- Feature cards grid (2 cols)
- Backend connection indicator
- Retry button on connection failure

### 9.2 Live Trading Page (`/live`)

- Quick stats grid (4 cols): Status, Daily P&L, Exposure, Total P&L
- Positions table with real-time updates
- Account summary section
- 5-second auto-refresh interval

### 9.3 Analytics Page (`/analytics`)

- Symbol/Timeframe selectors
- Trading chart with KAMA and ATR bands
- Technical indicators grid (ATR, RSI, ADX, Regime)
- Performance metrics section

### 9.4 Backtest Page (`/backtest`)

- Configuration form (symbol, dates, capital, parameters)
- Strategy parameter sliders
- Risk parameter sliders
- Results modal with metrics and trade log
- CSV export functionality

### 9.5 History Page (`/history`)

- Trade log table with expandable rows
- Pagination (20 per page)
- CSV export
- Trade details on expand

### 9.6 Signals Page (`/signals`)

- Filter controls (symbol, strategy, regime, date range)
- Search functionality
- Sortable columns
- Signal detail modal
- CSV export

---

## 10. File Structure

```
frontend/
├── app/
│   ├── globals.css              # Tailwind v4 @theme config
│   ├── layout.tsx               # Root layout with AppShell
│   ├── page.tsx                 # Home page
│   ├── analytics/
│   │   └── page.tsx
│   ├── backtest/
│   │   └── page.tsx
│   ├── history/
│   │   └── page.tsx
│   ├── live/
│   │   └── page.tsx
│   └── signals/
│       └── page.tsx
├── components/
│   ├── ui/                      # Base UI components
│   │   ├── Badge.tsx
│   │   ├── Button.tsx
│   │   ├── Card.tsx
│   │   ├── Input.tsx
│   │   ├── Select.tsx
│   │   ├── Skeleton.tsx
│   │   ├── StatusDot.tsx
│   │   ├── Table.tsx
│   │   └── index.ts
│   ├── layout/                  # Layout components
│   │   ├── AppShell.tsx
│   │   ├── Grid.tsx
│   │   ├── Navigation.tsx
│   │   ├── PageContainer.tsx
│   │   ├── PageHeader.tsx
│   │   └── index.ts
│   ├── trading/                 # Trading-specific components
│   │   ├── AccountSummary.tsx
│   │   ├── PLDisplay.tsx
│   │   ├── PositionCard.tsx
│   │   ├── PositionsTable.tsx
│   │   ├── PriceDisplay.tsx
│   │   ├── TradeRow.tsx
│   │   └── index.ts
│   ├── ErrorBoundary.tsx
│   └── [other error components]
├── contexts/
│   └── WebSocketContext.tsx     # Real-time price updates
├── lib/
│   └── utils.ts                 # cn(), formatCurrency(), formatPercent()
├── utils/
│   └── api.ts                   # API client
└── postcss.config.js            # @tailwindcss/postcss plugin
```

---

## 11. Tailwind CSS v4 Configuration

Tailwind v4 uses CSS-based configuration in `globals.css`:

```css
@import "tailwindcss";

@theme {
  /* Text colors */
  --color-text-100: #716f7a;
  --color-text-200: #8F8D98;
  --color-text-300: #AEACB7;
  --color-text-400: #CCCAD5;
  --color-text-500: #eae8f3;
  --color-text-600: #EFEEF6;
  --color-text-700: #F5F4F9;
  --color-text-800: #FAF9FC;
  --color-text-900: #ffffff;

  /* Panel colors */
  --color-panel-100: #292D3C;
  --color-panel-200: #272A39;
  --color-panel-300: #252735;
  --color-panel-400: #232432;
  --color-panel-500: #21222F;
  --color-panel-600: #1E1F2B;
  --color-panel-700: #1C1C28;
  --color-panel-800: #1A1924;
  --color-panel-900: #181621;

  /* Accent colors */
  --color-profit-100: #4ADE80;
  --color-profit-500: #22C55E;
  --color-profit-900: #16A34A;

  --color-loss-100: #F87171;
  --color-loss-500: #EF4444;
  --color-loss-900: #DC2626;

  --color-accent-100: #C04DFE;
  --color-accent-500: #A549FC;
  --color-accent-900: #8945FA;

  --color-blue-100: #5790FC;
  --color-blue-500: #3E7AEE;
  --color-blue-900: #2463E0;

  --color-warning-100: #FCD34D;
  --color-warning-500: #F59E0B;
  --color-warning-900: #D97706;

  /* Border radius */
  --radius-2xl: 22px;
}
```

PostCSS config (`postcss.config.js`):
```javascript
module.exports = {
  plugins: {
    '@tailwindcss/postcss': {},
  },
};
```

---

## 12. Utility Functions

### lib/utils.ts

```typescript
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

// Merge class names with Tailwind
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// Format number as currency
export function formatCurrency(value: number, showSign = false): string {
  const formatted = new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
  }).format(Math.abs(value));

  if (showSign && value !== 0) {
    return value > 0 ? `+${formatted}` : `-${formatted.replace('$', '')}`;
  }
  return value < 0 ? `-${formatted.replace('$', '')}` : formatted;
}

// Format number as percentage
export function formatPercent(value: number, showSign = false): string {
  const formatted = `${Math.abs(value).toFixed(2)}%`;
  if (showSign && value !== 0) {
    return value > 0 ? `+${formatted}` : `-${formatted}`;
  }
  return value < 0 ? `-${formatted}` : formatted;
}

// Get P&L color class
export function getPLColorClass(value: number): string {
  if (value > 0) return 'text-profit-500';
  if (value < 0) return 'text-loss-500';
  return 'text-text-400';
}
```

---

## 13. Backend Integration

### API Client (`utils/api.ts`)

All backend connections are maintained through `apiClient`:

| Method | Endpoint | Used By |
|--------|----------|---------|
| `getSystemStatus()` | `/api/status` | Home, Live |
| `getPositions()` | `/api/positions` | Live |
| `getAccountInfo()` | `/api/account` | Live |
| `getTrades(page, limit)` | `/api/trades` | History, Signals |

### WebSocket (`contexts/WebSocketContext.tsx`)

Real-time price updates for Analytics page:
- Endpoint: `/ws/prices`
- Methods: `subscribe([symbols])`, `getPrice(symbol)`
- Used by: Analytics chart

---

## 14. Accessibility

### Contrast Ratios
- Body text on backgrounds: minimum 4.5:1 ✅
- Large text (18px+): minimum 3:1 ✅
- Interactive elements: minimum 3:1 ✅

### Focus States
```css
focus:outline-none focus:ring-2 focus:ring-accent-500
```

### Touch Targets
- Minimum: 44px × 44px
- Buttons have minimum padding to meet this

---

## 15. Performance

### No Animations
- No CSS transitions on data components
- No hover animations on tables
- Static skeleton loading states
- Exception: Focus ring only

### Font Loading
- Google Fonts with `preconnect`
- System font fallbacks

---

## References

- Design inspiration: [Felix Luebken Resume Page](https://github.com/felixluebken/resume_page)
- Design principles: [Redwerk Frontend Design Guide](https://redwerk.com/blog/front-end-design-guides/)
- Chart library: [lightweight-charts v5](https://tradingview.github.io/lightweight-charts/)
- CSS framework: [Tailwind CSS v4](https://tailwindcss.com/)
