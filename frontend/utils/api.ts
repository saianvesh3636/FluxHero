/**
 * API utility for communicating with the FluxHero backend
 * Base URL is configured through Next.js rewrites in next.config.ts
 */

const API_BASE_URL = '/api';

export interface Position {
  symbol: string;
  quantity: number;
  entry_price: number;
  current_price: number;
  pnl: number;
  pnl_percent: number;
}

export interface Trade {
  id: number;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  timestamp: string;
  signal_explanation?: string;
  // Extended trade fields for history page
  entry_time?: number;
  exit_time?: number;
  entry_price?: number;
  exit_price?: number;
  shares?: number;
  realized_pnl?: number;
  strategy?: string;
  regime?: string;
  stop_loss?: number;
  take_profit?: number;
  signal_reason?: string;
}

export interface AccountInfo {
  equity: number;
  cash: number;
  buying_power: number;
  daily_pnl: number;
  total_pnl: number;
}

export interface SystemStatus {
  status: 'active' | 'delayed' | 'offline';
  last_update: string;
  uptime_seconds: number;
}

export interface BacktestConfig {
  symbol: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  parameters?: Record<string, number>;
}

export interface BacktestResult {
  sharpe_ratio: number;
  max_drawdown: number;
  total_return: number;
  win_rate: number;
  trades: Trade[];
  tearsheet_url?: string;
}

class ApiClient {
  private async fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.statusText}`);
    }

    return response.json();
  }

  async getPositions(): Promise<Position[]> {
    return this.fetchJson<Position[]>(`${API_BASE_URL}/positions`);
  }

  async getTrades(page: number = 1, limit: number = 20): Promise<Trade[]> {
    return this.fetchJson<Trade[]>(
      `${API_BASE_URL}/trades?page=${page}&limit=${limit}`
    );
  }

  async getAccountInfo(): Promise<AccountInfo> {
    return this.fetchJson<AccountInfo>(`${API_BASE_URL}/account`);
  }

  async getSystemStatus(): Promise<SystemStatus> {
    return this.fetchJson<SystemStatus>(`${API_BASE_URL}/status`);
  }

  async runBacktest(config: BacktestConfig): Promise<BacktestResult> {
    return this.fetchJson<BacktestResult>(`${API_BASE_URL}/backtest`, {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  /**
   * Create a WebSocket connection for live price updates
   * @param onMessage Callback for incoming price updates
   * @returns WebSocket instance
   */
  connectPriceWebSocket(
    onMessage: (data: any) => void,
    onError?: (error: Event) => void
  ): WebSocket {
    const wsUrl = `ws://${window.location.host}/ws/prices`;
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      if (onError) {
        onError(error);
      }
    };

    ws.onclose = () => {
      console.log('WebSocket connection closed');
    };

    return ws;
  }
}

export const apiClient = new ApiClient();

// Alias for backwards compatibility
export const api = apiClient;
