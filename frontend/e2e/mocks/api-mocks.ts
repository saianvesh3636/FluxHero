import { Page } from '@playwright/test';

/**
 * Mock API responses for E2E tests
 * These mocks simulate backend responses so tests can run without a live backend
 * Field names match the interfaces in utils/api.ts
 */

// Position interface from utils/api.ts
export const mockPositions = [
  {
    symbol: 'SPY',
    quantity: 100,
    entry_price: 450.25,
    current_price: 455.80,
    pnl: 555.0,
    pnl_percent: 1.23,
  },
  {
    symbol: 'AAPL',
    quantity: 50,
    entry_price: 175.50,
    current_price: 178.25,
    pnl: 137.5,
    pnl_percent: 1.57,
  },
];

// AccountInfo interface from utils/api.ts
export const mockAccount = {
  equity: 100000.0,
  cash: 25000.0,
  buying_power: 50000.0,
  daily_pnl: 692.5,
  total_pnl: 5250.0,
};

// SystemStatus interface from utils/api.ts
export const mockStatus = {
  status: 'active' as const,
  last_update: new Date().toISOString(),
  uptime_seconds: 3600,
};

// Trade interface from utils/api.ts
export const mockTrades = [
  {
    id: 1,
    symbol: 'SPY',
    side: 'buy' as const,
    quantity: 100,
    price: 450.25,
    timestamp: new Date(Date.now() - 3600000).toISOString(),
    signal_explanation: 'Price crossed above KAMA with ATR confirmation',
  },
  {
    id: 2,
    symbol: 'AAPL',
    side: 'buy' as const,
    quantity: 50,
    price: 175.5,
    timestamp: new Date(Date.now() - 7200000).toISOString(),
    signal_explanation: 'RSI oversold with volume confirmation',
  },
];

// BacktestResult interface from utils/api.ts
export const mockBacktestResult = {
  sharpe_ratio: 1.45,
  max_drawdown: -0.085,
  total_return: 0.152,
  win_rate: 0.58,
  trades: mockTrades,
  tearsheet_url: '/backtest/tearsheet.html',
};

export const mockCandles = [
  {
    timestamp: Date.now() - 3600000,
    open: 450.0,
    high: 456.5,
    low: 449.2,
    close: 455.8,
    volume: 1250000,
  },
  {
    timestamp: Date.now() - 7200000,
    open: 448.5,
    high: 451.0,
    low: 447.0,
    close: 450.0,
    volume: 980000,
  },
];

export const mockIndicators = {
  symbol: 'SPY',
  timestamp: new Date().toISOString(),
  atr: 2.45,
  rsi: 55.2,
  adx: 28.5,
  kama: 453.2,
  regime: 'TRENDING',
  volatility_state: 'NORMAL',
};

// Mock Live Analysis data
export const mockLiveAnalysis = {
  equity_curve: [
    { date: '2025-01-10', equity: 10000, benchmark_equity: 10000, daily_pnl: 0, cumulative_pnl: 0, cumulative_return_pct: 0, benchmark_return_pct: 0 },
    { date: '2025-01-13', equity: 10150, benchmark_equity: 10050, daily_pnl: 150, cumulative_pnl: 150, cumulative_return_pct: 1.5, benchmark_return_pct: 0.5 },
    { date: '2025-01-14', equity: 10280, benchmark_equity: 10120, daily_pnl: 130, cumulative_pnl: 280, cumulative_return_pct: 2.8, benchmark_return_pct: 1.2 },
    { date: '2025-01-15', equity: 10180, benchmark_equity: 10080, daily_pnl: -100, cumulative_pnl: 180, cumulative_return_pct: 1.8, benchmark_return_pct: 0.8 },
    { date: '2025-01-16', equity: 10350, benchmark_equity: 10150, daily_pnl: 170, cumulative_pnl: 350, cumulative_return_pct: 3.5, benchmark_return_pct: 1.5 },
    { date: '2025-01-17', equity: 10520, benchmark_equity: 10200, daily_pnl: 170, cumulative_pnl: 520, cumulative_return_pct: 5.2, benchmark_return_pct: 2.0 },
    { date: '2025-01-21', equity: 10480, benchmark_equity: 10180, daily_pnl: -40, cumulative_pnl: 480, cumulative_return_pct: 4.8, benchmark_return_pct: 1.8 },
    { date: '2025-01-22', equity: 10650, benchmark_equity: 10250, daily_pnl: 170, cumulative_pnl: 650, cumulative_return_pct: 6.5, benchmark_return_pct: 2.5 },
  ],
  risk_metrics: {
    sharpe_ratio: 1.85,
    sortino_ratio: 2.42,
    calmar_ratio: 3.15,
    max_drawdown: -100,
    max_drawdown_pct: -0.97,
    win_rate: 0.72,
    profit_factor: 2.35,
    avg_win: 155,
    avg_loss: -70,
  },
  daily_breakdown: [
    { date: '2025-01-10', pnl: 0, return_pct: 0, trade_count: 0, cumulative_pnl: 0 },
    { date: '2025-01-13', pnl: 150, return_pct: 1.5, trade_count: 2, cumulative_pnl: 150 },
    { date: '2025-01-14', pnl: 130, return_pct: 1.28, trade_count: 1, cumulative_pnl: 280 },
    { date: '2025-01-15', pnl: -100, return_pct: -0.97, trade_count: 1, cumulative_pnl: 180 },
    { date: '2025-01-16', pnl: 170, return_pct: 1.67, trade_count: 2, cumulative_pnl: 350 },
    { date: '2025-01-17', pnl: 170, return_pct: 1.64, trade_count: 1, cumulative_pnl: 520 },
    { date: '2025-01-21', pnl: -40, return_pct: -0.38, trade_count: 1, cumulative_pnl: 480 },
    { date: '2025-01-22', pnl: 170, return_pct: 1.62, trade_count: 2, cumulative_pnl: 650 },
  ],
  initial_capital: 10000,
  current_equity: 10650,
  benchmark_symbol: 'VTI',
  trading_days: 8,
};

/**
 * Setup API mocks for a page
 * Call this in test.beforeEach to mock all API endpoints
 */
export async function setupApiMocks(page: Page) {
  // Mock /api/status endpoint
  await page.route('**/api/status', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockStatus),
    });
  });

  // Mock /api/positions endpoint
  await page.route('**/api/positions', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockPositions),
    });
  });

  // Mock /api/account endpoint
  await page.route('**/api/account', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockAccount),
    });
  });

  // Mock /api/trades endpoint
  await page.route('**/api/trades**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockTrades),
    });
  });

  // Mock /api/backtest endpoint
  await page.route('**/api/backtest', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockBacktestResult),
    });
  });

  // Mock /api/test/candles endpoint
  await page.route('**/api/test/candles**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockCandles),
    });
  });

  // Mock /api/indicators endpoint
  await page.route('**/api/indicators**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockIndicators),
    });
  });

  // Mock /api/live/analysis endpoint
  await page.route('**/api/live/analysis**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockLiveAnalysis),
    });
  });

  // Mock /api/paper/analysis endpoint (same mock data)
  await page.route('**/api/paper/analysis**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockLiveAnalysis),
    });
  });

  // Mock /health endpoint
  await page.route('**/health', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ status: 'healthy', timestamp: new Date().toISOString() }),
    });
  });

  // Mock /metrics endpoint
  await page.route('**/metrics', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'text/plain',
      body: `# HELP fluxhero_requests_total Total requests
# TYPE fluxhero_requests_total counter
fluxhero_requests_total 1234
`,
    });
  });
}

/**
 * Setup WebSocket mock (note: Playwright doesn't fully support WS mocking,
 * but we can intercept the initial HTTP upgrade request)
 */
export async function setupWebSocketMock(page: Page) {
  // Abort WebSocket connections to prevent them from hitting the real backend
  // The frontend should handle connection failures gracefully
  await page.route('**/ws/**', async (route) => {
    // Abort the WebSocket upgrade request - this prevents 403 errors from backend
    await route.abort('connectionfailed');
  });
}

/**
 * Setup all mocks (API + WebSocket)
 */
export async function setupAllMocks(page: Page) {
  // Set flag to disable WebSocket in the app
  await page.addInitScript(() => {
    (window as { __PLAYWRIGHT_TEST__?: boolean }).__PLAYWRIGHT_TEST__ = true;
  });

  await setupApiMocks(page);
  await setupWebSocketMock(page);
}
