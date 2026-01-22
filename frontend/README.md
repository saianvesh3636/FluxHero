# FluxHero Frontend

React + Next.js frontend for the FluxHero adaptive quant trading system.

## Technologies

- **Next.js 16**: React framework with App Router
- **TypeScript**: Type-safe development
- **React 19**: UI library
- **API Integration**: Pre-configured proxy to backend API

## Project Structure

```
frontend/
├── app/                    # Next.js App Router pages
│   ├── layout.tsx         # Root layout
│   ├── page.tsx           # Home page
│   └── globals.css        # Global styles
├── components/            # Reusable React components
├── pages/                 # Future page components
│   ├── live.tsx          # Live trading dashboard
│   ├── analytics.tsx     # Charts and indicators
│   ├── history.tsx       # Trade history
│   └── backtest.tsx      # Backtesting interface
├── utils/                # Utility functions
│   └── api.ts           # Backend API client
└── styles/              # Component styles
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

## API Integration

The frontend is configured to communicate with the FluxHero backend API:

- REST API: Proxied through `/api/*` to `http://localhost:8000/api/*`
- WebSocket: Proxied through `/ws/*` to `http://localhost:8000/ws/*`

### Available API Methods

The `apiClient` utility provides methods for:

- `getPositions()`: Fetch current open positions
- `getTrades(page, limit)`: Fetch trade history with pagination
- `getAccountInfo()`: Get account equity, cash, and P&L
- `getSystemStatus()`: Check system health and uptime
- `runBacktest(config)`: Execute backtests with custom parameters
- `connectPriceWebSocket(onMessage, onError)`: Real-time price updates

### Example Usage

```typescript
import { apiClient } from '@/utils/api';

// Fetch positions
const positions = await apiClient.getPositions();

// Connect to live prices
const ws = apiClient.connectPriceWebSocket(
  (data) => console.log('Price update:', data),
  (error) => console.error('WebSocket error:', error)
);
```

## Configuration

API endpoints can be customized in `next.config.ts`:

```typescript
async rewrites() {
  return [
    {
      source: '/api/:path*',
      destination: 'http://localhost:8000/api/:path*',
    },
  ];
}
```

## Future Implementation

The following pages are planned for Phase 14:

- **Live Trading**: Real-time positions, P&L, and system status
- **Analytics**: TradingView-style charts with KAMA, ATR, and regime indicators
- **Trade History**: Filterable trade log with CSV export
- **Backtesting**: Interactive backtest configuration and results viewer

## Notes

- Strict mode enabled for better development experience
- TypeScript strict mode enabled for type safety
- API proxy configured for seamless backend integration
- WebSocket support for real-time data streaming
