/**
 * API utility for communicating with the FluxHero backend
 * Base URL is configured through Next.js rewrites in next.config.ts
 *
 * All interfaces match the backend Pydantic models in backend/api/server.py
 */

const API_BASE_URL = '/api';

// Default development auth token (matches backend FLUXHERO_AUTH_SECRET default)
const DEV_AUTH_TOKEN = 'fluxhero-dev-secret-change-in-production';

// Auth token for API requests (set via setAuthToken)
// Defaults to dev token for local development
let authToken: string | null = process.env.NODE_ENV === 'development' ? DEV_AUTH_TOKEN : null;

/**
 * Set the authentication token for API requests
 * In production, this would come from a login flow
 */
export function setAuthToken(token: string | null) {
  authToken = token;
}

/**
 * Get the current auth token
 */
export function getAuthToken(): string | null {
  return authToken;
}

// ============================================================================
// Backend Response Interfaces (match backend/api/server.py Pydantic models)
// ============================================================================

/**
 * Position response from backend (PositionResponse model)
 * Raw format as returned by /api/positions
 */
export interface PositionResponse {
  id: number | null;
  symbol: string;
  side: number;           // 1 = LONG, -1 = SHORT
  shares: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  stop_loss: number;
  take_profit: number | null;
  entry_time: string;
  updated_at: string;
}

/**
 * Trade response from backend (TradeResponse model)
 * Raw format as returned in trade history
 */
export interface TradeResponse {
  id: number | null;
  symbol: string;
  side: number;           // 1 = LONG, -1 = SHORT
  entry_price: number;
  entry_time: string;
  exit_price: number | null;
  exit_time: string | null;
  shares: number;
  stop_loss: number;
  take_profit: number | null;
  realized_pnl: number | null;
  status: number;         // Trade status code
  strategy: string;
  regime: string;
  signal_reason: string;
}

/**
 * Paginated trade history response (TradeHistoryResponse model)
 */
export interface TradeHistoryResponse {
  trades: TradeResponse[];
  total_count: number;
  page: number;
  page_size: number;
  total_pages: number;
}

/**
 * Account info response (AccountInfoResponse model)
 */
export interface AccountInfoResponse {
  equity: number;
  cash: number;
  buying_power: number;
  total_pnl: number;
  daily_pnl: number;
  num_positions: number;
}

/**
 * System status response (SystemStatusResponse model)
 */
export interface SystemStatusResponse {
  status: 'ACTIVE' | 'DELAYED' | 'OFFLINE';
  uptime_seconds: number;
  last_update: string;
  websocket_connected: boolean;
  data_feed_active: boolean;
  message: string;
}

/**
 * Backtest request (BacktestRequest model)
 */
export interface BacktestRequest {
  symbol: string;
  start_date: string;      // YYYY-MM-DD
  end_date: string;        // YYYY-MM-DD
  initial_capital?: number;
  commission_per_share?: number;
  slippage_pct?: number;
  strategy_mode?: 'TREND' | 'MEAN_REVERSION' | 'DUAL';
}

/**
 * Backtest result response (BacktestResultResponse model)
 */
export interface BacktestResultResponse {
  symbol: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  final_equity: number;
  total_return: number;
  total_return_pct: number;
  sharpe_ratio: number;
  max_drawdown: number;
  max_drawdown_pct: number;
  win_rate: number;
  num_trades: number;
  avg_win_loss_ratio: number;
  success_criteria_met: boolean;
  equity_curve: number[];
  timestamps: string[];
}

/**
 * Walk-forward backtest request (WalkForwardRequest model)
 */
export interface WalkForwardRequest {
  symbol: string;
  start_date: string;      // YYYY-MM-DD
  end_date: string;        // YYYY-MM-DD
  initial_capital?: number;
  commission_per_share?: number;
  slippage_pct?: number;
  train_bars?: number;     // Training period bars (~3 months = 63)
  test_bars?: number;      // Test period bars (~1 month = 21)
  strategy_mode?: 'TREND' | 'MEAN_REVERSION' | 'DUAL';
  pass_threshold?: number; // Pass rate threshold (default 0.6)
}

/**
 * Metrics for a single walk-forward test window
 */
export interface WalkForwardWindowMetrics {
  window_id: number;
  train_start_date: string | null;
  train_end_date: string | null;
  test_start_date: string | null;
  test_end_date: string | null;
  initial_equity: number;
  final_equity: number;
  return_pct: number;
  sharpe_ratio: number;
  max_drawdown_pct: number;
  win_rate: number;
  num_trades: number;
  is_profitable: boolean;
}

/**
 * Walk-forward backtest response (WalkForwardResponse model)
 */
export interface WalkForwardResponse {
  symbol: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  final_capital: number;
  total_return_pct: number;

  // Walk-forward specific metrics
  total_windows: number;
  profitable_windows: number;
  pass_rate: number;
  passes_walk_forward_test: boolean;
  pass_threshold: number;

  // Aggregate metrics across all windows
  aggregate_sharpe: number;
  aggregate_max_drawdown_pct: number;
  aggregate_win_rate: number;
  total_trades: number;

  // Per-window results
  window_results: WalkForwardWindowMetrics[];

  // Combined equity curve and timestamps
  combined_equity_curve: number[];
  timestamps: string[];
  train_bars: number;
  test_bars: number;
}

/**
 * Health check response
 */
export interface HealthResponse {
  status: 'healthy' | 'degraded';
  timestamp: string;
  uptime_seconds: number;
  database_connected: boolean;
  websocket_connections: number;
  data_feed_active: boolean;
  total_requests: number;
}

// ============================================================================
// Mode Management Interfaces
// ============================================================================

/**
 * Mode state response (ModeStateResponse model)
 */
export interface ModeStateResponse {
  active_mode: 'live' | 'paper';
  last_mode_change: string | null;
  paper_balance: number;
  paper_realized_pnl: number;
  is_live_broker_configured: boolean;
}

/**
 * Request to place an order
 */
export interface PlaceOrderRequest {
  symbol: string;
  qty: number;
  side: 'buy' | 'sell';
  order_type?: 'market' | 'limit';
  limit_price?: number | null;
}

/**
 * Response from placing an order
 */
export interface PlaceOrderResponse {
  order_id: string;
  symbol: string;
  qty: number;
  side: string;
  status: string;
  filled_price: number | null;
  mode: 'live' | 'paper';
}

/**
 * Backtest result summary for list view
 */
export interface BacktestResultSummary {
  id: number | null;
  run_id: string;
  symbol: string;
  strategy_mode: string;
  start_date: string;
  end_date: string;
  total_return_pct: number | null;
  sharpe_ratio: number | null;
  max_drawdown_pct: number | null;
  win_rate: number | null;
  num_trades: number | null;
  created_at: string;
}

/**
 * Detailed backtest result
 */
export interface BacktestResultDetail extends BacktestResultSummary {
  initial_capital: number;
  final_equity: number;
  equity_curve_json: string | null;
  trades_json: string | null;
  config_json: string | null;
}

/**
 * Candle data for charts
 */
export interface CandleData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

/**
 * Symbol validation response (SymbolValidationResponse model)
 */
export interface SymbolValidationResponse {
  symbol: string;
  name: string;
  exchange: string | null;
  currency: string | null;
  type: string | null;
  is_valid: boolean;
}

/**
 * Symbol search response (SymbolSearchResponse model)
 */
export interface SymbolSearchResponse {
  query: string;
  results: SymbolValidationResponse[];
}

/**
 * Chart interval info from API
 */
export interface IntervalInfo {
  name: string;      // e.g., "1h", "4h", "1d"
  label: string;     // e.g., "1 Hour", "4 Hours", "1 Day"
  seconds: number;   // Duration in seconds
  max_days: number;  // Maximum history available
  native: boolean;   // True if provider supports natively
}

/**
 * Cache breakdown entry
 */
export interface CacheBreakdownEntry {
  symbol: string;
  interval: string;
  count: number;
  min_date: string;
  max_date: string;
}

/**
 * Cache statistics response
 */
export interface CacheStatsResponse {
  total_candles: number;
  unique_symbols: number;
  breakdown: CacheBreakdownEntry[];
}

/**
 * API Error with status code
 */
export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
    this.detail = detail;
    this.name = 'ApiError';
  }
}

// ============================================================================
// Frontend-friendly interfaces (transformed from backend responses)
// ============================================================================

/**
 * Position interface for frontend components
 * Transformed from PositionResponse for easier consumption
 */
export interface Position {
  id: number | null;
  symbol: string;
  side: 'long' | 'short';
  quantity: number;
  entry_price: number;
  current_price: number;
  pnl: number;
  pnl_percent: number;
  stop_loss: number;
  take_profit: number | null;
  entry_time: string;
  updated_at: string;
}

/**
 * Trade interface for frontend components
 * Transformed from TradeResponse for easier consumption
 */
export interface Trade {
  id: number | null;
  symbol: string;
  side: 'buy' | 'sell';
  entry_price: number;
  entry_time: string;
  exit_price: number | null;
  exit_time: string | null;
  shares: number;
  stop_loss: number;
  take_profit: number | null;
  realized_pnl: number | null;
  status: string;
  strategy: string;
  regime: string;
  signal_reason: string;
  // Computed fields
  return_pct: number | null;
}

/**
 * Account info for frontend components
 */
export interface AccountInfo {
  equity: number;
  cash: number;
  buying_power: number;
  daily_pnl: number;
  total_pnl: number;
  num_positions: number;
}

/**
 * System status for frontend components
 */
export interface SystemStatus {
  status: 'active' | 'delayed' | 'offline';
  last_update: string;
  uptime_seconds: number;
  websocket_connected: boolean;
  data_feed_active: boolean;
  message: string;
}

// ============================================================================
// Transform functions (backend response -> frontend format)
// ============================================================================

function transformPosition(pos: PositionResponse): Position {
  const marketValue = pos.current_price * pos.shares;
  const costBasis = pos.entry_price * pos.shares;
  const pnlPercent = costBasis !== 0 ? (pos.unrealized_pnl / costBasis) * 100 : 0;

  return {
    id: pos.id,
    symbol: pos.symbol,
    side: pos.side === 1 ? 'long' : 'short',
    quantity: pos.shares,
    entry_price: pos.entry_price,
    current_price: pos.current_price,
    pnl: pos.unrealized_pnl,
    pnl_percent: pnlPercent,
    stop_loss: pos.stop_loss,
    take_profit: pos.take_profit,
    entry_time: pos.entry_time,
    updated_at: pos.updated_at,
  };
}

function transformTrade(trade: TradeResponse): Trade {
  const returnPct = trade.entry_price && trade.shares && trade.realized_pnl
    ? (trade.realized_pnl / (trade.entry_price * trade.shares)) * 100
    : null;

  // Map status codes to readable strings
  const statusMap: Record<number, string> = {
    0: 'OPEN',
    1: 'CLOSED',
    2: 'CANCELLED',
  };

  return {
    id: trade.id,
    symbol: trade.symbol,
    side: trade.side === 1 ? 'buy' : 'sell',
    entry_price: trade.entry_price,
    entry_time: trade.entry_time,
    exit_price: trade.exit_price,
    exit_time: trade.exit_time,
    shares: trade.shares,
    stop_loss: trade.stop_loss,
    take_profit: trade.take_profit,
    realized_pnl: trade.realized_pnl,
    status: statusMap[trade.status] || 'UNKNOWN',
    strategy: trade.strategy,
    regime: trade.regime,
    signal_reason: trade.signal_reason,
    return_pct: returnPct,
  };
}

function transformSystemStatus(status: SystemStatusResponse): SystemStatus {
  return {
    status: status.status.toLowerCase() as 'active' | 'delayed' | 'offline',
    last_update: status.last_update,
    uptime_seconds: status.uptime_seconds,
    websocket_connected: status.websocket_connected,
    data_feed_active: status.data_feed_active,
    message: status.message,
  };
}

function transformAccountInfo(account: AccountInfoResponse): AccountInfo {
  return {
    equity: account.equity,
    cash: account.cash,
    buying_power: account.buying_power,
    daily_pnl: account.daily_pnl,
    total_pnl: account.total_pnl,
    num_positions: account.num_positions,
  };
}

// ============================================================================
// API Client
// ============================================================================

class ApiClient {
  private getHeaders(): HeadersInit {
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    };

    if (authToken) {
      headers['Authorization'] = `Bearer ${authToken}`;
    }

    return headers;
  }

  private async fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
    const response = await fetch(url, {
      ...options,
      headers: {
        ...this.getHeaders(),
        ...options?.headers,
      },
    });

    if (!response.ok) {
      let detail: string;
      try {
        const errorBody = await response.json();
        detail = errorBody.detail || errorBody.message || response.statusText;
      } catch {
        detail = await response.text().catch(() => response.statusText);
      }
      throw new ApiError(response.status, detail);
    }

    return response.json();
  }

  /**
   * Get all open positions
   * Transforms backend PositionResponse to frontend Position format
   */
  async getPositions(): Promise<Position[]> {
    const response = await this.fetchJson<PositionResponse[]>(`${API_BASE_URL}/positions`);
    return response.map(transformPosition);
  }

  /**
   * Get raw positions without transformation (for debugging)
   */
  async getPositionsRaw(): Promise<PositionResponse[]> {
    return this.fetchJson<PositionResponse[]>(`${API_BASE_URL}/positions`);
  }

  /**
   * Get paginated trade history
   * @param page - Page number (1-indexed)
   * @param pageSize - Number of trades per page (max 100)
   * @param status - Optional filter: 'OPEN', 'CLOSED', or 'CANCELLED'
   */
  async getTrades(
    page: number = 1,
    pageSize: number = 20,
    status?: 'OPEN' | 'CLOSED' | 'CANCELLED'
  ): Promise<{ trades: Trade[]; totalCount: number; totalPages: number }> {
    let url = `${API_BASE_URL}/trades?page=${page}&page_size=${pageSize}`;
    if (status) {
      url += `&status=${status}`;
    }

    const response = await this.fetchJson<TradeHistoryResponse>(url);

    return {
      trades: response.trades.map(transformTrade),
      totalCount: response.total_count,
      totalPages: response.total_pages,
    };
  }

  /**
   * Get raw trade history without transformation
   */
  async getTradesRaw(
    page: number = 1,
    pageSize: number = 20,
    status?: string
  ): Promise<TradeHistoryResponse> {
    let url = `${API_BASE_URL}/trades?page=${page}&page_size=${pageSize}`;
    if (status) {
      url += `&status=${status}`;
    }
    return this.fetchJson<TradeHistoryResponse>(url);
  }

  /**
   * Get account information
   */
  async getAccountInfo(): Promise<AccountInfo> {
    const response = await this.fetchJson<AccountInfoResponse>(`${API_BASE_URL}/account`);
    return transformAccountInfo(response);
  }

  /**
   * Get system status
   */
  async getSystemStatus(): Promise<SystemStatus> {
    const response = await this.fetchJson<SystemStatusResponse>(`${API_BASE_URL}/status`);
    return transformSystemStatus(response);
  }

  /**
   * Run a backtest with the given configuration
   */
  async runBacktest(config: BacktestRequest): Promise<BacktestResultResponse> {
    return this.fetchJson<BacktestResultResponse>(`${API_BASE_URL}/backtest`, {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  /**
   * Run a walk-forward backtest with the given configuration
   */
  async runWalkForwardBacktest(config: WalkForwardRequest): Promise<WalkForwardResponse> {
    return this.fetchJson<WalkForwardResponse>(`${API_BASE_URL}/backtest/walk-forward`, {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  /**
   * Get health check status
   */
  async getHealth(): Promise<HealthResponse> {
    return this.fetchJson<HealthResponse>('/health');
  }

  /**
   * Get test candle data (development only)
   * @param symbol - Symbol to fetch (SPY, AAPL, or MSFT)
   */
  async getTestCandles(symbol: string = 'SPY'): Promise<CandleData[]> {
    return this.fetchJson<CandleData[]>(`${API_BASE_URL}/test/candles?symbol=${symbol}`);
  }

  /**
   * Get chart data from Yahoo Finance
   * Returns OHLCV candle data for charting
   *
   * @param symbol - Stock symbol (e.g., SPY, AAPL, MSFT)
   * @param interval - Data interval: 1m, 5m, 15m, 1h, 4h, 1d
   * @param bars - Number of bars/candles to fetch (default: 300)
   * @param useCache - Whether to use cached data (default: true)
   */
  async getChartData(
    symbol: string = 'SPY',
    interval: string = '1d',
    bars: number = 300,
    useCache: boolean = true,
    maxDays?: number  // Optional: pass from /api/chart/intervals response
  ): Promise<ChartCandleData[]> {
    // Convert bars to days based on interval
    // Trading day = 6.5 hours = 390 minutes
    const barsPerDay: Record<string, number> = {
      '1m': 390,
      '5m': 78,
      '15m': 26,
      '30m': 13,
      '1h': 7,  // ~6.5 rounded up
      '4h': 2,  // ~1.6 rounded up
      '1d': 1,
      '1wk': 0.2,  // 1 bar per 5 days
    };

    const bpd = barsPerDay[interval] || 1;
    let days = Math.ceil(bars / bpd) + 5; // Add buffer for weekends/holidays

    // Cap days to provider's limit if specified (from /api/chart/intervals)
    if (maxDays !== undefined) {
      days = Math.min(days, maxDays);
    }

    const params = new URLSearchParams({
      symbol: symbol.toUpperCase(),
      interval,
      days: days.toString(),
      use_cache: useCache.toString(),
    });

    // Fetch and limit to requested bar count
    const data = await this.fetchJson<ChartCandleData[]>(`${API_BASE_URL}/chart?${params}`);

    // Return only the last N bars requested
    if (data.length > bars) {
      return data.slice(-bars);
    }
    return data;
  }

  /**
   * Get available chart intervals
   * Returns supported intervals with metadata (max history, labels, etc.)
   */
  async getChartIntervals(): Promise<IntervalInfo[]> {
    return this.fetchJson<IntervalInfo[]>(`${API_BASE_URL}/chart/intervals`);
  }

  /**
   * Get cache statistics
   * Returns info about cached candle data
   */
  async getCacheStats(): Promise<CacheStatsResponse> {
    return this.fetchJson<CacheStatsResponse>(`${API_BASE_URL}/cache/stats`);
  }

  /**
   * Clear candle cache
   * @param symbol - Optional symbol to clear (clears all if not specified)
   */
  async clearCache(symbol?: string): Promise<{ deleted: number; symbol: string }> {
    const url = symbol
      ? `${API_BASE_URL}/cache/clear?symbol=${symbol.toUpperCase()}`
      : `${API_BASE_URL}/cache/clear`;
    return this.fetchJson<{ deleted: number; symbol: string }>(url, {
      method: 'DELETE',
    });
  }

  /**
   * Check if a symbol/interval has cached data
   */
  async isCached(symbol: string, interval: string): Promise<boolean> {
    try {
      const stats = await this.getCacheStats();
      return stats.breakdown.some(
        (b) => b.symbol === symbol.toUpperCase() && b.interval === interval
      );
    } catch {
      return false;
    }
  }

  /**
   * Validate a stock symbol
   * Returns symbol info if valid, throws ApiError (404) if not found
   *
   * @param symbol - Stock ticker symbol (e.g., AAPL, SPY)
   * @throws ApiError with status 404 if symbol not found
   */
  async validateSymbol(symbol: string): Promise<SymbolValidationResponse> {
    return this.fetchJson<SymbolValidationResponse>(`${API_BASE_URL}/symbol/validate`, {
      method: 'POST',
      body: JSON.stringify({ symbol: symbol.toUpperCase() }),
    });
  }

  /**
   * Search for stock symbols
   * Returns matching symbols (currently exact match only)
   *
   * @param query - Search query (symbol or company name)
   * @param limit - Maximum results to return (1-50)
   */
  async searchSymbols(query: string, limit: number = 10): Promise<SymbolSearchResponse> {
    return this.fetchJson<SymbolSearchResponse>(
      `${API_BASE_URL}/symbol/search?q=${encodeURIComponent(query)}&limit=${limit}`
    );
  }

  // ============================================================================
  // Broker Management Methods
  // ============================================================================

  /**
   * Get all configured brokers
   */
  async getBrokers(): Promise<BrokerListResponse> {
    return this.fetchJson<BrokerListResponse>(`${API_BASE_URL}/brokers`);
  }

  /**
   * Add a new broker configuration
   */
  async addBroker(config: BrokerConfigRequest): Promise<BrokerConfigResponse> {
    return this.fetchJson<BrokerConfigResponse>(`${API_BASE_URL}/brokers`, {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  /**
   * Delete a broker configuration
   */
  async deleteBroker(brokerId: string): Promise<void> {
    await this.fetchJson<void>(`${API_BASE_URL}/brokers/${brokerId}`, {
      method: 'DELETE',
    });
  }

  /**
   * Check broker connection health
   */
  async getBrokerHealth(brokerId: string): Promise<BrokerHealthResponse> {
    return this.fetchJson<BrokerHealthResponse>(`${API_BASE_URL}/brokers/${brokerId}/health`);
  }

  // ============================================================================
  // Paper Trading Methods
  // ============================================================================

  /**
   * Get paper trading account information
   */
  async getPaperAccount(): Promise<PaperAccountResponse> {
    return this.fetchJson<PaperAccountResponse>(`${API_BASE_URL}/paper/account`);
  }

  /**
   * Reset paper trading account to initial state
   */
  async resetPaperAccount(): Promise<PaperResetResponse> {
    return this.fetchJson<PaperResetResponse>(`${API_BASE_URL}/paper/reset`, {
      method: 'POST',
    });
  }

  /**
   * Get paper trading trade history
   */
  async getPaperTrades(): Promise<PaperTradeHistoryResponse> {
    return this.fetchJson<PaperTradeHistoryResponse>(`${API_BASE_URL}/paper/trades`);
  }

  // ============================================================================
  // Trade Analytics Methods (Phase G)
  // ============================================================================

  /**
   * Get chart data for a specific trade
   * Returns candles, indicators, and trade details for visualization
   */
  async getTradeChartData(tradeId: number): Promise<TradeChartDataResponse> {
    return this.fetchJson<TradeChartDataResponse>(`${API_BASE_URL}/trades/${tradeId}/chart-data`);
  }

  /**
   * Get daily trade summary with grouping for a specific mode
   * @param mode - Trading mode ('live' or 'paper')
   * @param days - Number of days to include (default 30)
   */
  async getDailySummary(mode: 'live' | 'paper' = 'live', days: number = 30): Promise<DailySummaryResponse> {
    return this.fetchJson<DailySummaryResponse>(`${API_BASE_URL}/${mode}/daily-summary?days=${days}`);
  }

  /**
   * Get trading analysis with performance metrics for a specific mode
   * @param mode - Trading mode ('live' or 'paper')
   * @param benchmark - Benchmark symbol for comparison (default VTI)
   */
  async getAnalysis(mode: 'live' | 'paper' = 'live', benchmark: string = 'VTI'): Promise<LiveAnalysisResponse> {
    return this.fetchJson<LiveAnalysisResponse>(`${API_BASE_URL}/${mode}/analysis?benchmark=${benchmark}`);
  }

  /**
   * Get live trading analysis with performance metrics (alias for backward compatibility)
   * @param benchmark - Benchmark symbol for comparison (default VTI)
   */
  async getLiveAnalysis(benchmark: string = 'VTI'): Promise<LiveAnalysisResponse> {
    return this.getAnalysis('live', benchmark);
  }

  // ============================================================================
  // Mode Management Methods
  // ============================================================================

  /**
   * Get current trading mode state
   */
  async getModeState(): Promise<ModeStateResponse> {
    return this.fetchJson<ModeStateResponse>(`${API_BASE_URL}/mode`);
  }

  /**
   * Switch trading mode
   * @param mode - Target mode ('live' or 'paper')
   * @param confirmLive - Must be true to switch to live mode
   */
  async switchMode(mode: 'live' | 'paper', confirmLive: boolean = false): Promise<ModeStateResponse> {
    return this.fetchJson<ModeStateResponse>(`${API_BASE_URL}/mode`, {
      method: 'POST',
      body: JSON.stringify({ mode, confirm_live: confirmLive }),
    });
  }

  /**
   * Get positions for a specific mode
   */
  async getPositionsForMode(mode: 'live' | 'paper'): Promise<Position[]> {
    const response = await this.fetchJson<PositionResponse[]>(`${API_BASE_URL}/${mode}/positions`);
    return response.map(pos => ({
      id: pos.id,
      symbol: pos.symbol,
      side: pos.side > 0 ? 'long' : 'short',
      quantity: Math.abs(pos.shares || 0),
      entry_price: pos.entry_price,
      current_price: pos.current_price,
      pnl: pos.unrealized_pnl,
      pnl_percent: pos.entry_price > 0 && pos.shares > 0
        ? (pos.unrealized_pnl / (pos.entry_price * pos.shares)) * 100
        : 0,
      stop_loss: pos.stop_loss,
      take_profit: pos.take_profit,
      entry_time: pos.entry_time,
      updated_at: pos.updated_at,
    } as Position));
  }

  /**
   * Get trades for a specific mode with pagination
   */
  async getTradesForMode(
    mode: 'live' | 'paper',
    page: number = 1,
    pageSize: number = 20
  ): Promise<{ trades: Trade[]; totalCount: number; totalPages: number }> {
    const response = await this.fetchJson<TradeHistoryResponse>(
      `${API_BASE_URL}/${mode}/trades?page=${page}&page_size=${pageSize}`
    );
    return {
      trades: response.trades.map(t => ({
        id: t.id,
        symbol: t.symbol,
        side: t.side === 1 ? 'buy' : 'sell',
        entry_price: t.entry_price,
        exit_price: t.exit_price,
        entry_time: t.entry_time,
        exit_time: t.exit_time,
        shares: t.shares,
        realized_pnl: t.realized_pnl,
        status: t.status === 0 ? 'open' : 'closed',
        strategy: t.strategy,
        regime: t.regime,
        signal_reason: t.signal_reason,
        stop_loss: t.stop_loss,
        take_profit: t.take_profit,
      } as Trade)),
      totalCount: response.total_count,
      totalPages: response.total_pages,
    };
  }

  /**
   * Get account info for a specific mode
   */
  async getAccountForMode(mode: 'live' | 'paper'): Promise<AccountInfo> {
    const response = await this.fetchJson<AccountInfoResponse>(`${API_BASE_URL}/${mode}/account`);
    return {
      equity: response.equity,
      cash: response.cash,
      buying_power: response.buying_power,
      total_pnl: response.total_pnl,
      daily_pnl: response.daily_pnl,
      num_positions: response.num_positions,
    };
  }

  /**
   * Place an order in paper trading mode
   * @param order - Order details (symbol, qty, side, order_type, limit_price)
   */
  async placePaperOrder(order: PlaceOrderRequest): Promise<PlaceOrderResponse> {
    return this.fetchJson<PlaceOrderResponse>(`${API_BASE_URL}/paper/orders`, {
      method: 'POST',
      body: JSON.stringify(order),
    });
  }

  /**
   * Place an order in live trading mode
   * CAUTION: This places real orders with real money!
   * @param order - Order details (symbol, qty, side, order_type, limit_price)
   * @param confirmLiveTrade - Must be true to execute live trades
   */
  async placeLiveOrder(order: PlaceOrderRequest, confirmLiveTrade: boolean = false): Promise<PlaceOrderResponse> {
    const headers: Record<string, string> = {};
    if (confirmLiveTrade) {
      headers['X-Confirm-Live-Trade'] = 'true';
    }
    return this.fetchJson<PlaceOrderResponse>(`${API_BASE_URL}/live/orders`, {
      method: 'POST',
      body: JSON.stringify(order),
      headers,
    });
  }

  /**
   * Place an order for a specific mode
   * @param mode - Trading mode ('live' or 'paper')
   * @param order - Order details
   * @param confirmLiveTrade - Must be true for live mode
   */
  async placeOrder(
    mode: 'live' | 'paper',
    order: PlaceOrderRequest,
    confirmLiveTrade: boolean = false
  ): Promise<PlaceOrderResponse> {
    if (mode === 'live') {
      return this.placeLiveOrder(order, confirmLiveTrade);
    }
    return this.placePaperOrder(order);
  }

  /**
   * Sell a position (convenience method)
   * @param mode - Trading mode
   * @param symbol - Symbol to sell
   * @param qty - Quantity to sell
   * @param confirmLiveTrade - Must be true for live mode
   */
  async sellPosition(
    mode: 'live' | 'paper',
    symbol: string,
    qty: number,
    confirmLiveTrade: boolean = false
  ): Promise<PlaceOrderResponse> {
    return this.placeOrder(mode, {
      symbol,
      qty,
      side: 'sell',
      order_type: 'market',
    }, confirmLiveTrade);
  }

  /**
   * Get backtest results list
   */
  async getBacktestResults(limit: number = 50): Promise<BacktestResultSummary[]> {
    return this.fetchJson<BacktestResultSummary[]>(`${API_BASE_URL}/backtest/results?limit=${limit}`);
  }

  /**
   * Get detailed backtest result by run_id
   */
  async getBacktestResultDetail(runId: string): Promise<BacktestResultDetail> {
    return this.fetchJson<BacktestResultDetail>(`${API_BASE_URL}/backtest/results/${runId}`);
  }

  /**
   * Clear all trading data (for fresh start)
   */
  async clearAllTradingData(confirm: boolean = false): Promise<{ status: string; message: string }> {
    return this.fetchJson<{ status: string; message: string }>(
      `${API_BASE_URL}/data/clear?confirm=${confirm}`,
      { method: 'POST' }
    );
  }
}

export const apiClient = new ApiClient();

// Alias for backwards compatibility
export const api = apiClient;

// ============================================================================
// Legacy interfaces for backwards compatibility
// These map to the new interfaces but keep old field names where code depends on them
// ============================================================================

/** @deprecated Use BacktestRequest instead */
export interface BacktestConfig {
  symbol: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  parameters?: Record<string, number>;
}

/** @deprecated Use BacktestResultResponse instead */
export interface BacktestResult {
  sharpe_ratio: number;
  max_drawdown: number;
  total_return: number;
  win_rate: number;
  trades: Trade[];
  tearsheet_url?: string;
}

// ============================================================================
// Broker Management Interfaces
// ============================================================================

/**
 * Request model for adding a broker configuration
 */
export interface BrokerConfigRequest {
  broker_type: string;
  name: string;
  api_key: string;
  api_secret: string;
  base_url?: string;
}

/**
 * Response model for broker configuration (without secrets)
 */
export interface BrokerConfigResponse {
  id: string;
  broker_type: string;
  name: string;
  api_key_masked: string;
  base_url: string;
  is_connected: boolean;
  created_at: string;
  updated_at: string;
}

/**
 * Response model for listing brokers
 */
export interface BrokerListResponse {
  brokers: BrokerConfigResponse[];
  total: number;
}

/**
 * Response model for broker health check
 */
export interface BrokerHealthResponse {
  id: string;
  name: string;
  broker_type: string;
  is_connected: boolean;
  is_authenticated: boolean;
  latency_ms: number | null;
  last_heartbeat: string | null;
  error_message: string | null;
}

// ============================================================================
// Paper Trading Interfaces
// ============================================================================

/**
 * Paper trading position
 */
export interface PaperPosition {
  symbol: string;
  qty: number;
  entry_price: number;
  current_price: number;
  market_value: number;
  unrealized_pnl: number;
  cost_basis: number;
}

/**
 * Paper trading account response
 */
export interface PaperAccountResponse {
  account_id: string;
  balance: number;
  buying_power: number;
  equity: number;
  cash: number;
  positions_value: number;
  realized_pnl: number;
  unrealized_pnl: number;
  positions: PaperPosition[];
}

/**
 * Paper trade record
 */
export interface PaperTrade {
  trade_id: string;
  order_id: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  qty: number;
  price: number;
  slippage: number;
  timestamp: string;
  realized_pnl: number;
}

/**
 * Paper trade history response
 */
export interface PaperTradeHistoryResponse {
  trades: PaperTrade[];
  total_count: number;
}

/**
 * Paper account reset response
 */
export interface PaperResetResponse {
  message: string;
  account_id: string;
  initial_balance: number;
  timestamp: string;
}

// ============================================================================
// Trade Analytics Interfaces (Phase G)
// ============================================================================

/**
 * OHLCV candle data for charts
 */
export interface ChartCandleData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

/**
 * Indicator values at each candle
 */
export interface IndicatorData {
  time: number;
  kama: number | null;
  atr_upper: number | null;
  atr_lower: number | null;
}

/**
 * Trade chart data response
 */
export interface TradeChartDataResponse {
  trade: TradeResponse;
  candles: ChartCandleData[];
  indicators: IndicatorData[];
  entry_index: number;
  exit_index: number | null;
}

/**
 * Daily trade breakdown
 */
export interface DailyTradeBreakdown {
  date: string;
  trades: TradeResponse[];
  trade_count: number;
  realized_pnl: number;
  win_count: number;
  loss_count: number;
  daily_return_pct: number;
}

/**
 * Summary totals across all trades
 */
export interface TotalsSummary {
  closed_count: number;
  open_count: number;
  realized_pnl: number;
  unrealized_pnl: number;
  total_pnl: number;
  total_return_pct: number;
  win_rate: number;
}

/**
 * Daily summary response
 */
export interface DailySummaryResponse {
  daily_groups: DailyTradeBreakdown[];
  totals: TotalsSummary;
  open_positions: PositionResponse[];
}

/**
 * Equity curve point
 */
export interface EquityCurvePoint {
  date: string;
  equity: number;
  benchmark_equity: number;
  daily_pnl: number;
  cumulative_pnl: number;
  cumulative_return_pct: number;
  benchmark_return_pct: number;
}

/**
 * Risk-adjusted performance metrics
 */
export interface RiskMetrics {
  sharpe_ratio: number;
  sortino_ratio: number;
  calmar_ratio: number;
  max_drawdown: number;
  max_drawdown_pct: number;
  win_rate: number;
  profit_factor: number;
  avg_win: number;
  avg_loss: number;
}

/**
 * Daily P&L breakdown
 */
export interface DailyBreakdown {
  date: string;
  pnl: number;
  return_pct: number;
  trade_count: number;
  cumulative_pnl: number;
}

/**
 * Live analysis response
 */
export interface LiveAnalysisResponse {
  equity_curve: EquityCurvePoint[];
  risk_metrics: RiskMetrics;
  daily_breakdown: DailyBreakdown[];
  initial_capital: number;
  current_equity: number;
  benchmark_symbol: string;
  trading_days: number;
}
