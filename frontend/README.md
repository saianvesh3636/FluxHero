# FluxHero Frontend

React + Next.js frontend for the FluxHero adaptive quant trading system.

## Technologies

- **Next.js 16**: React framework with App Router
- **React 19**: UI library
- **TypeScript**: Type-safe development
- **Tailwind CSS v4**: CSS-based configuration with design tokens
- **lightweight-charts v5**: TradingView charting library

## Project Structure

```
frontend/
├── app/                    # Next.js App Router pages
│   ├── layout.tsx          # Root layout with AppShell
│   ├── page.tsx            # Redirects to /trades
│   ├── globals.css         # Tailwind v4 @theme config
│   ├── analytics/          # Charts and indicators
│   ├── backtest/           # Backtesting interface
│   ├── history/            # Trade history
│   ├── live/               # Live trading dashboard
│   │   └── analysis/       # P&L comparison charts
│   ├── signals/            # Signal archive
│   ├── trades/             # Main trades page
│   └── walk-forward/       # Walk-forward testing
├── components/
│   ├── ui/                 # Base UI components (Button, Card, Badge, etc.)
│   ├── layout/             # Layout components (Navigation, PageContainer)
│   ├── trading/            # Trading components (PositionsTable, PLDisplay)
│   └── charts/             # Chart components (lightweight-charts based)
│       ├── config/         # Theme and constants
│       ├── core/           # ChartContainer, OHLCDataBox
│       ├── composed/       # CandlestickChart, EquityCurveChart, etc.
│       ├── hooks/          # useChart, useSeries
│       ├── utils/          # Data transformers, formatters
│       └── types/          # TypeScript interfaces
├── contexts/               # React contexts (TradingMode, WebSocket)
├── lib/                    # Utilities (cn, formatCurrency, formatPercent)
└── utils/                  # API client
```

## Getting Started

### Prerequisites

- Node.js 18+ installed
- Backend API running on `http://localhost:8000`

### Installation

```bash
npm install
```

### Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the application.

### Build for Production

```bash
npm run build
npm start
```

### Run Tests

```bash
npm test
```

## Pages

| Route | Description |
|-------|-------------|
| `/` | Redirects to /trades |
| `/trades` | Main trades page with positions and daily groups |
| `/analytics` | Symbol charts with KAMA and ATR indicators |
| `/backtest` | Strategy backtesting with equity curves |
| `/walk-forward` | Walk-forward optimization testing |
| `/live` | Live trading dashboard |
| `/live/analysis` | P&L comparison charts |
| `/history` | Trade history with pagination |
| `/signals` | Signal archive with filters |

## API Integration

The frontend communicates with the FluxHero backend API:

- REST API: Proxied through `/api/*` to `http://localhost:8000/api/*`
- WebSocket: Proxied through `/ws/*` to `http://localhost:8000/ws/*`

### Available API Methods

The `apiClient` utility (`utils/api.ts`) provides methods for:

- `getPositions()`: Fetch current open positions
- `getTrades(page, limit)`: Fetch trade history with pagination
- `getAccountInfo()`: Get account equity, cash, and P&L
- `getSystemStatus()`: Check system health and uptime
- `runBacktest(config)`: Execute backtests with custom parameters
- `placeOrder(mode, symbol, qty, side, orderType, limitPrice)`: Place orders
- `connectPriceWebSocket(onMessage, onError)`: Real-time price updates

## Chart Components

Charts use TradingView's lightweight-charts v5 library:

- **CandlestickChart**: OHLC with volume and indicator overlays
- **EquityCurveChart**: Line with area fill and benchmark comparison
- **LineChart**: Simple multi-series line charts
- **PnLComparisonChart**: Multi-series with interactive legend
- **TradeDetailChart**: Candlestick with price lines and markers

Features:
- OHLC data display on hover
- Magnet crosshair mode for precise candle selection
- Theme colors mapped to Tailwind design tokens
- SSR-safe with dynamic imports

## Design System

Dark-mode only trading UI with:
- Color-coded P&L (green profit, red loss)
- Tabular-nums for price alignment
- No animations (performance-critical)
- Consistent spacing using 4px base unit

See `FRONTEND_REQUIREMENTS.md` for full design token reference.
