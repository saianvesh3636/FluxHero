# Live/Paper Trading Mode Implementation Plan

## Overview

Implement a unified trading UI with a toggle switch to switch between Live (production) and Paper (test) trading modes. Same UI for both modes, different data sources.

## User Decisions

1. **Data Separation**: Separate tables within same database (Option C)
2. **Test Mode**: Both paper trading AND backtest results viewing
3. **Toggle Behavior**: Reconnect to different broker/data source
4. **Paper Trading**: Fully functional with Sell buttons (execute paper trades)
5. **URL Structure**: Mode in URL (`/trades?mode=live` or `/trades?mode=paper`)
6. **Existing Data**: Clear all (no migration needed)

---

## Phase 1: Database Schema

### Files to Modify
- `backend/storage/sqlite_store.py`

### New Tables to Create

```sql
-- Live trades table
CREATE TABLE IF NOT EXISTS live_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    entry_time TEXT NOT NULL,
    exit_price REAL,
    exit_time TEXT,
    shares INTEGER NOT NULL,
    stop_loss REAL NOT NULL,
    take_profit REAL,
    realized_pnl REAL,
    status INTEGER NOT NULL DEFAULT 0,
    strategy TEXT NOT NULL,
    regime TEXT,
    signal_reason TEXT,
    signal_explanation TEXT,
    broker_order_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Paper trades table
CREATE TABLE IF NOT EXISTS paper_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    entry_time TEXT NOT NULL,
    exit_price REAL,
    exit_time TEXT,
    shares INTEGER NOT NULL,
    stop_loss REAL NOT NULL,
    take_profit REAL,
    realized_pnl REAL,
    status INTEGER NOT NULL DEFAULT 0,
    strategy TEXT NOT NULL,
    regime TEXT,
    signal_reason TEXT,
    signal_explanation TEXT,
    slippage_applied REAL DEFAULT 0.0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Live positions table
CREATE TABLE IF NOT EXISTS live_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL UNIQUE,
    side INTEGER NOT NULL,
    shares INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    current_price REAL NOT NULL,
    unrealized_pnl REAL NOT NULL,
    stop_loss REAL NOT NULL,
    take_profit REAL,
    entry_time TEXT NOT NULL,
    broker_position_id TEXT,
    updated_at TEXT NOT NULL
);

-- Paper positions table
CREATE TABLE IF NOT EXISTS paper_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL UNIQUE,
    side INTEGER NOT NULL,
    shares INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    current_price REAL NOT NULL,
    unrealized_pnl REAL NOT NULL,
    stop_loss REAL NOT NULL,
    take_profit REAL,
    entry_time TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Backtest results table (shared, view in Test mode)
CREATE TABLE IF NOT EXISTS backtest_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,
    symbol TEXT NOT NULL,
    strategy_mode TEXT NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    initial_capital REAL NOT NULL,
    final_equity REAL NOT NULL,
    total_return_pct REAL,
    sharpe_ratio REAL,
    max_drawdown_pct REAL,
    win_rate REAL,
    num_trades INTEGER,
    equity_curve_json TEXT,
    trades_json TEXT,
    config_json TEXT,
    created_at TEXT NOT NULL
);

-- Mode state table
CREATE TABLE IF NOT EXISTS mode_state (
    id INTEGER PRIMARY KEY DEFAULT 1,
    active_mode TEXT NOT NULL DEFAULT 'paper',
    last_mode_change TEXT,
    paper_balance REAL DEFAULT 100000.0,
    paper_realized_pnl REAL DEFAULT 0.0
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_live_trades_symbol ON live_trades(symbol);
CREATE INDEX IF NOT EXISTS idx_live_trades_status ON live_trades(status);
CREATE INDEX IF NOT EXISTS idx_live_trades_entry_time ON live_trades(entry_time);
CREATE INDEX IF NOT EXISTS idx_paper_trades_symbol ON paper_trades(symbol);
CREATE INDEX IF NOT EXISTS idx_paper_trades_status ON paper_trades(status);
CREATE INDEX IF NOT EXISTS idx_paper_trades_entry_time ON paper_trades(entry_time);
CREATE INDEX IF NOT EXISTS idx_backtest_results_symbol ON backtest_results(symbol);
```

### New Enum

```python
class TradingMode(str, Enum):
    LIVE = "live"
    PAPER = "paper"
```

### New Methods to Add to SQLiteStore

```python
# Mode-aware trade methods
async def add_trade_for_mode(self, trade: Trade, mode: TradingMode) -> int
async def get_recent_trades_for_mode(self, mode: TradingMode, limit: int = 50) -> list[Trade]
async def update_trade_for_mode(self, trade_id: int, mode: TradingMode, **kwargs) -> None

# Mode-aware position methods
async def upsert_position_for_mode(self, position: Position, mode: TradingMode) -> None
async def get_positions_for_mode(self, mode: TradingMode) -> list[Position]
async def delete_position_for_mode(self, symbol: str, mode: TradingMode) -> None

# Mode state methods
async def get_active_mode(self) -> TradingMode
async def set_active_mode(self, mode: TradingMode) -> None
async def get_paper_account_state(self) -> dict

# Backtest results methods
async def save_backtest_result(self, result: BacktestResult) -> str
async def get_backtest_results(self, limit: int = 50) -> list[BacktestResult]
async def get_backtest_result(self, run_id: str) -> BacktestResult | None

# Clear data (for fresh start)
async def clear_all_trading_data(self) -> None
```

---

## Phase 2: Backend API

### Files to Modify
- `backend/api/server.py`
- `backend/execution/broker_factory.py` (create TradingModeManager)

### New Pydantic Models

```python
class TradingMode(str, Enum):
    LIVE = "live"
    PAPER = "paper"

class ModeStateResponse(BaseModel):
    active_mode: TradingMode
    last_mode_change: str | None
    paper_balance: float
    paper_realized_pnl: float
    is_live_broker_configured: bool

class SwitchModeRequest(BaseModel):
    mode: TradingMode
    confirm_live: bool = False  # Must be True to switch to live

class PlaceOrderRequest(BaseModel):
    symbol: str
    qty: int
    side: Literal["buy", "sell"]
    order_type: Literal["market", "limit"] = "market"
    limit_price: float | None = None

class PlaceOrderResponse(BaseModel):
    order_id: str
    symbol: str
    qty: int
    side: str
    status: str
    filled_price: float | None
    mode: TradingMode
```

### New API Endpoints

```python
# Mode Management
GET  /api/mode                    -> ModeStateResponse
POST /api/mode                    -> ModeStateResponse (switch mode)

# Live Trading Endpoints
GET  /api/live/positions          -> list[PositionResponse]
GET  /api/live/trades             -> TradeHistoryResponse
GET  /api/live/account            -> AccountInfoResponse
GET  /api/live/daily-summary      -> DailySummaryResponse
GET  /api/live/analysis           -> LiveAnalysisResponse
POST /api/live/orders             -> PlaceOrderResponse

# Paper Trading Endpoints
GET  /api/paper/positions         -> list[PositionResponse]
GET  /api/paper/trades            -> TradeHistoryResponse
GET  /api/paper/account           -> AccountInfoResponse
GET  /api/paper/daily-summary     -> DailySummaryResponse
GET  /api/paper/analysis          -> LiveAnalysisResponse
POST /api/paper/orders            -> PlaceOrderResponse
POST /api/paper/reset             -> PaperResetResponse

# Backtest Results (part of Test mode)
GET  /api/backtest/results        -> list[BacktestResultSummary]
GET  /api/backtest/results/{id}   -> BacktestResultDetail
```

### TradingModeManager Class

Create new file: `backend/execution/trading_mode_manager.py`

```python
class TradingModeManager:
    """Manages active trading mode and broker connections."""

    def __init__(self, store: SQLiteStore, broker_factory: BrokerFactory):
        self._store = store
        self._factory = broker_factory
        self._live_broker: BrokerInterface | None = None
        self._paper_broker: BrokerInterface | None = None
        self._active_mode: TradingMode = TradingMode.PAPER

    async def initialize(self) -> None:
        """Load active mode from database."""
        self._active_mode = await self._store.get_active_mode()

    async def get_active_mode(self) -> TradingMode:
        return self._active_mode

    async def get_broker_for_mode(self, mode: TradingMode) -> BrokerInterface:
        """Get the broker for specified mode."""
        if mode == TradingMode.LIVE:
            if self._live_broker is None:
                self._live_broker = self._factory.create_broker("alpaca", {...})
            return self._live_broker
        else:
            if self._paper_broker is None:
                self._paper_broker = self._factory.create_broker("paper", {...})
            return self._paper_broker

    async def switch_mode(self, new_mode: TradingMode, confirm_live: bool = False) -> None:
        """Switch trading mode with safety checks."""
        if new_mode == TradingMode.LIVE and not confirm_live:
            raise ValueError("Live mode requires explicit confirmation")

        self._active_mode = new_mode
        await self._store.set_active_mode(new_mode)

    async def is_live_broker_configured(self) -> bool:
        """Check if live broker credentials are configured."""
        settings = get_settings()
        return bool(settings.alpaca_api_key and settings.alpaca_api_secret)
```

### Safety Checks for Live Trading

```python
@app.post("/api/live/orders")
async def place_live_order(order: PlaceOrderRequest, request: Request):
    # Safety check 1: Verify mode is actually live
    current_mode = await mode_manager.get_active_mode()
    if current_mode != TradingMode.LIVE:
        raise HTTPException(400, "Cannot place live order: not in live mode")

    # Safety check 2: Require confirmation header
    if request.headers.get("X-Confirm-Live-Trade") != "true":
        raise HTTPException(428, "Live trades require X-Confirm-Live-Trade header")

    # Execute order...
```

---

## Phase 3: Frontend Mode Integration

### Files to Modify
- `frontend/components/TradingModeToggle.tsx` (enhance existing)
- `frontend/utils/api.ts` (add mode-aware methods)
- `frontend/contexts/WebSocketContext.tsx` (mode awareness)

### Files to Create
- `frontend/contexts/TradingModeContext.tsx`
- `frontend/hooks/useTradingMode.ts`

### TradingModeContext

```typescript
// frontend/contexts/TradingModeContext.tsx

import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { useSearchParams, useRouter, usePathname } from 'next/navigation';
import { apiClient } from '../utils/api';

export type TradingMode = 'live' | 'paper';

interface ModeState {
  active_mode: TradingMode;
  last_mode_change: string | null;
  paper_balance: number;
  paper_realized_pnl: number;
  is_live_broker_configured: boolean;
}

interface TradingModeContextValue {
  mode: TradingMode;
  modeState: ModeState | null;
  isLive: boolean;
  isPaper: boolean;
  isLoading: boolean;
  switchMode: (newMode: TradingMode, confirmLive?: boolean) => Promise<void>;
  refreshModeState: () => Promise<void>;
}

const TradingModeContext = createContext<TradingModeContextValue | null>(null);

export function TradingModeProvider({ children }: { children: ReactNode }) {
  const [modeState, setModeState] = useState<ModeState | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const mode = modeState?.active_mode || 'paper';

  const refreshModeState = async () => {
    try {
      const state = await apiClient.getModeState();
      setModeState(state);
    } catch (error) {
      console.error('Failed to fetch mode state:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const switchMode = async (newMode: TradingMode, confirmLive = false) => {
    try {
      const result = await apiClient.switchMode(newMode, confirmLive);
      setModeState(result);

      // Update URL with new mode
      const params = new URLSearchParams(searchParams);
      params.set('mode', newMode);
      router.push(`${pathname}?${params.toString()}`);
    } catch (error) {
      throw error;
    }
  };

  useEffect(() => {
    refreshModeState();
  }, []);

  // Sync URL mode param with backend on navigation
  useEffect(() => {
    const urlMode = searchParams.get('mode') as TradingMode | null;
    if (urlMode && modeState && urlMode !== modeState.active_mode) {
      // URL and backend out of sync - backend is source of truth
      const params = new URLSearchParams(searchParams);
      params.set('mode', modeState.active_mode);
      router.replace(`${pathname}?${params.toString()}`);
    }
  }, [modeState, searchParams]);

  return (
    <TradingModeContext.Provider value={{
      mode,
      modeState,
      isLive: mode === 'live',
      isPaper: mode === 'paper',
      isLoading,
      switchMode,
      refreshModeState,
    }}>
      {children}
    </TradingModeContext.Provider>
  );
}

export function useTradingMode() {
  const context = useContext(TradingModeContext);
  if (!context) {
    throw new Error('useTradingMode must be used within TradingModeProvider');
  }
  return context;
}
```

### Enhanced TradingModeToggle Component

The existing component at `frontend/components/TradingModeToggle.tsx` needs:
1. Connect to backend via TradingModeContext (not just localStorage)
2. Add acknowledgment checkbox for live mode confirmation
3. Show broker connection status
4. Visual indicator matching reference image (dot + label + toggle)

### API Client Updates

```typescript
// Add to frontend/utils/api.ts

interface ModeStateResponse {
  active_mode: 'live' | 'paper';
  last_mode_change: string | null;
  paper_balance: number;
  paper_realized_pnl: number;
  is_live_broker_configured: boolean;
}

class ApiClient {
  // Mode management
  async getModeState(): Promise<ModeStateResponse> {
    return this.fetchJson<ModeStateResponse>(`${API_BASE_URL}/mode`);
  }

  async switchMode(mode: 'live' | 'paper', confirmLive: boolean = false): Promise<ModeStateResponse> {
    return this.fetchJson<ModeStateResponse>(`${API_BASE_URL}/mode`, {
      method: 'POST',
      body: JSON.stringify({ mode, confirm_live: confirmLive }),
    });
  }

  // Mode-specific data fetching
  async getPositionsForMode(mode: 'live' | 'paper'): Promise<Position[]> {
    const response = await this.fetchJson<PositionResponse[]>(`${API_BASE_URL}/${mode}/positions`);
    return response.map(transformPosition);
  }

  async getTradesForMode(mode: 'live' | 'paper', page = 1, pageSize = 20): Promise<TradeHistoryResponse> {
    return this.fetchJson<TradeHistoryResponse>(
      `${API_BASE_URL}/${mode}/trades?page=${page}&page_size=${pageSize}`
    );
  }

  async getAccountForMode(mode: 'live' | 'paper'): Promise<AccountInfo> {
    const response = await this.fetchJson<AccountInfoResponse>(`${API_BASE_URL}/${mode}/account`);
    return transformAccountInfo(response);
  }

  async getDailySummaryForMode(mode: 'live' | 'paper', days = 30): Promise<DailySummaryResponse> {
    return this.fetchJson<DailySummaryResponse>(`${API_BASE_URL}/${mode}/daily-summary?days=${days}`);
  }

  async placeOrder(mode: 'live' | 'paper', order: PlaceOrderRequest): Promise<PlaceOrderResponse> {
    const headers: HeadersInit = {};
    if (mode === 'live') {
      headers['X-Confirm-Live-Trade'] = 'true';
    }
    return this.fetchJson<PlaceOrderResponse>(`${API_BASE_URL}/${mode}/orders`, {
      method: 'POST',
      body: JSON.stringify(order),
      headers,
    });
  }

  // Backtest results
  async getBacktestResults(limit = 50): Promise<BacktestResultSummary[]> {
    return this.fetchJson<BacktestResultSummary[]>(`${API_BASE_URL}/backtest/results?limit=${limit}`);
  }
}
```

---

## Phase 4: Unified Trades Page

### Files to Create
- `frontend/app/trades/page.tsx` (main unified page)
- `frontend/app/trades/layout.tsx` (with TradingModeProvider)

### Files to Modify
- `frontend/components/trading/DailyTradeGroup.tsx` (add Sell button)
- `frontend/components/trading/PositionsTable.tsx` (add Sell action)
- `frontend/components/trading/TradeSummaryFooter.tsx` (add mode indicator)

### Unified Trades Page Structure

```
/trades?mode=paper (default)
/trades?mode=live

Page Layout:
┌─────────────────────────────────────────────────────────────┐
│ Navigation Bar                      [ModeToggle] [Status]   │
├─────────────────────────────────────────────────────────────┤
│ Live Trades          [Paper] [Live]      [Filter] [Analysis]│
│ Last updated: 3:45 PM                                       │
├─────────────────────────────────────────────────────────────┤
│ [Tab: Trading] [Tab: Backtest Results] (only in paper mode) │
├─────────────────────────────────────────────────────────────┤
│ ▶ 2026-01-23  R:+$43  U:-$563  Trades:4  Day:-1.85%        │
│ ▼ 2026-01-22  R:+$320  U:$0    Trades:17 Day:+1.14%        │
│   ├─ AAPL 100 $184.47 → $182.21  -$178.85  [Chart]         │
│   └─ NVDA 50  $890.12 → $895.40  +$264.00  [Chart]         │
├─────────────────────────────────────────────────────────────┤
│ Open Positions (2)                                          │
│ PANW  79  $184.47  $182.21  -$178.85  -1.23%  [Sell]       │
│ CRCL  190 $73.99   $71.97   -$384.28  -2.73%  [Sell]       │
├─────────────────────────────────────────────────────────────┤
│ [Footer: Closed:90 Open:2 Realized:$1374 Unrealized:-$563] │
└─────────────────────────────────────────────────────────────┘
```

### Backtest Results Tab (Paper Mode Only)

Shows historical backtest runs with:
- Symbol, date range, strategy
- Key metrics (Sharpe, Return, Drawdown)
- Click to view detailed equity curve

---

## Phase 5: Route Cleanup

### Navigation Updates

```typescript
// frontend/components/layout/Navigation.tsx

const navItems = [
  { label: 'Home', href: '/' },
  { label: 'Trades', href: '/trades' },  // Unified - replaces /live
  { label: 'Analytics', href: '/analytics' },
  { label: 'Backtest', href: '/backtest' },  // Run new backtests
  { label: 'Signals', href: '/signals' },
  { label: 'Settings', href: '/settings' },
];
```

### Redirects

Create `frontend/middleware.ts`:

```typescript
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export function middleware(request: NextRequest) {
  const url = request.nextUrl;

  // Redirect /live to /trades?mode=live
  if (url.pathname === '/live' || url.pathname.startsWith('/live/')) {
    const newPath = url.pathname.replace('/live', '/trades');
    return NextResponse.redirect(new URL(`${newPath}?mode=live`, request.url));
  }

  // Redirect /history to /trades (mode from current state)
  if (url.pathname === '/history') {
    return NextResponse.redirect(new URL('/trades', request.url));
  }

  return NextResponse.next();
}

export const config = {
  matcher: ['/live/:path*', '/history'],
};
```

### Pages to Remove/Deprecate
- `frontend/app/live/page.tsx` -> redirect to /trades?mode=live
- `frontend/app/live/analysis/page.tsx` -> redirect to /trades/analysis?mode=live
- `frontend/app/history/page.tsx` -> redirect to /trades

---

## Safety Features Summary

1. **Live Mode Confirmation**: Checkbox acknowledgment + confirm button
2. **Visual Indicators**:
   - Red pulsing dot when in Live mode
   - Warning banner at top of page
   - Mode badge in footer
3. **Backend Validation**: Mode checked before every order
4. **Header Requirement**: X-Confirm-Live-Trade header for live orders
5. **URL Sync**: Mode always visible in URL
6. **Cross-tab Sync**: localStorage event for multi-tab consistency

---

## Testing Checklist

- [ ] Database tables created correctly
- [ ] Mode switching works (paper -> live with confirmation)
- [ ] Paper trades execute and persist
- [ ] Live mode blocked without broker config
- [ ] Positions table shows Sell buttons
- [ ] Daily trade grouping displays correctly
- [ ] Backtest results tab appears in paper mode
- [ ] URL mode parameter syncs with backend
- [ ] Mode indicator visible in all states
- [ ] Redirects work from old routes
