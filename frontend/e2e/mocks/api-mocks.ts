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
  // For WebSocket, we can't fully mock the connection, but we can handle it gracefully
  // The frontend should handle connection failures
  await page.route('**/ws/**', async (route) => {
    // Let WebSocket connections through - they'll fail gracefully
    await route.continue();
  });
}

/**
 * Setup all mocks (API + WebSocket)
 */
export async function setupAllMocks(page: Page) {
  await setupApiMocks(page);
  await setupWebSocketMock(page);
}
