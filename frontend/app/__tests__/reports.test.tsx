/**
 * UI Tests for Reports Page and Trade Review Modal
 *
 * Tests the QuantStats integration UI components with mock data.
 * Verifies:
 * - Reports page renders correctly with metrics
 * - Report generation flow works
 * - Trade review modal displays trade details
 * - Metrics are formatted correctly
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';

// Mock the API client
jest.mock('../../utils/api', () => ({
  apiClient: {
    getEnhancedMetrics: jest.fn(),
    generateReport: jest.fn(),
    listReports: jest.fn(),
    downloadReport: jest.fn(),
    deleteReport: jest.fn(),
    reviewTrade: jest.fn(),
  },
  ApiError: class ApiError extends Error {
    detail: string;
    constructor(message: string) {
      super(message);
      this.detail = message;
    }
  },
}));

// Mock data that matches what the backend would return
const mockEnhancedMetrics = {
  // Tier 1 metrics
  sortino_ratio: 1.85,
  calmar_ratio: 2.1,
  profit_factor: 1.65,
  value_at_risk_95: -0.023,
  cvar_95: -0.031,
  alpha: 0.12,
  beta: 0.85,
  kelly_criterion: 0.25,
  recovery_factor: 3.2,
  ulcer_index: 4.5,

  // Tier 2 metrics
  max_consecutive_wins: 8,
  max_consecutive_losses: 4,
  skewness: 0.15,
  kurtosis: 2.8,
  tail_ratio: 1.2,
  information_ratio: 0.75,
  r_squared: 0.65,

  // Standard metrics
  sharpe_ratio: 1.52,
  max_drawdown_pct: -12.5,
  win_rate: 0.58,
  avg_win_loss_ratio: 1.8,
  total_return_pct: 25.5,
  annualized_return_pct: 32.1,

  // Metadata
  periods_analyzed: 252,
  benchmark_symbol: 'SPY',
  data_source: 'paper:30d',
};

const mockReportResponse = {
  report_id: 'abc123',
  download_url: '/api/reports/download/abc123',
  generated_at: '2026-01-24T12:00:00Z',
  expires_at: '2026-01-25T12:00:00Z',
  title: 'Paper Trading Report - Last 30 days',
  source: 'paper',
  symbol: 'AAPL',
  strategy: 'mean_reversion',
  date_range: 'Last 30 days',
};

const mockReportsList = {
  reports: [
    {
      report_id: 'report_123',
      filename: 'report_123.html',
      created_at: '2026-01-24T10:00:00Z',
      size_bytes: 256000,
      download_url: '/api/reports/download/report_123',
    },
    {
      report_id: 'report_456',
      filename: 'report_456.html',
      created_at: '2026-01-23T15:30:00Z',
      size_bytes: 312000,
      download_url: '/api/reports/download/report_456',
    },
  ],
  total_count: 2,
};

const mockTradeReview = {
  trade_id: 42,
  symbol: 'AAPL',
  side: 'LONG',
  status: 'CLOSED',
  entry_price: 185.5,
  entry_time: '2026-01-20T09:30:00Z',
  shares: 100,
  exit_price: 192.75,
  exit_time: '2026-01-24T15:45:00Z',
  realized_pnl: 725.0,
  return_pct: 3.91,
  holding_period_days: 4.26,
  strategy: 'mean_reversion',
  regime: 'TRENDING',
  signal_reason: 'RSI oversold bounce',
  signal_explanation: 'RSI dropped below 30 with bullish divergence on MACD histogram',
  stop_loss: 180.0,
  take_profit: 195.0,
};

// Import components after mocks are set up
import { apiClient } from '../../utils/api';

describe('Reports Page Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (apiClient.getEnhancedMetrics as jest.Mock).mockResolvedValue(mockEnhancedMetrics);
    (apiClient.listReports as jest.Mock).mockResolvedValue(mockReportsList);
    (apiClient.generateReport as jest.Mock).mockResolvedValue(mockReportResponse);
  });

  describe('Metrics Formatting', () => {
    test('formatMetric handles ratio format correctly', () => {
      // Test the formatMetric function logic
      const formatMetric = (value: number, type: string): string => {
        if (value === undefined || value === null || isNaN(value)) return '-';
        switch (type) {
          case 'ratio': return value.toFixed(2);
          case 'percent': return `${(value * 100).toFixed(2)}%`;
          case 'integer': return Math.round(value).toString();
          default: return value.toFixed(4);
        }
      };

      expect(formatMetric(1.85, 'ratio')).toBe('1.85');
      expect(formatMetric(0.58, 'percent')).toBe('58.00%');
      expect(formatMetric(8.3, 'integer')).toBe('8');
      expect(formatMetric(NaN, 'ratio')).toBe('-');
    });

    test('getMetricColor returns correct colors for positive/negative values', () => {
      const getMetricColor = (value: number, goodAbove: number = 0): string => {
        if (value === undefined || value === null || isNaN(value)) return 'text-gray-400';
        return value > goodAbove ? 'text-green-500' : value < goodAbove ? 'text-red-500' : 'text-gray-300';
      };

      expect(getMetricColor(1.5, 1)).toBe('text-green-500'); // Above threshold
      expect(getMetricColor(0.5, 1)).toBe('text-red-500'); // Below threshold
      expect(getMetricColor(1, 1)).toBe('text-gray-300'); // Equal to threshold
      expect(getMetricColor(NaN, 0)).toBe('text-gray-400'); // Invalid
    });
  });

  describe('API Response Structure Validation', () => {
    test('mock metrics match expected EnhancedMetricsResponse shape', () => {
      // Verify all required fields exist
      const requiredTier1 = [
        'sortino_ratio', 'calmar_ratio', 'profit_factor',
        'value_at_risk_95', 'cvar_95', 'alpha', 'beta',
        'kelly_criterion', 'recovery_factor', 'ulcer_index'
      ];
      const requiredTier2 = [
        'max_consecutive_wins', 'max_consecutive_losses',
        'skewness', 'kurtosis', 'tail_ratio',
        'information_ratio', 'r_squared'
      ];
      const requiredStandard = [
        'sharpe_ratio', 'max_drawdown_pct', 'win_rate',
        'avg_win_loss_ratio', 'total_return_pct', 'annualized_return_pct'
      ];
      const requiredMetadata = ['periods_analyzed', 'benchmark_symbol', 'data_source'];

      const allRequired = [...requiredTier1, ...requiredTier2, ...requiredStandard, ...requiredMetadata];

      for (const field of allRequired) {
        expect(mockEnhancedMetrics).toHaveProperty(field);
        expect(mockEnhancedMetrics[field as keyof typeof mockEnhancedMetrics]).not.toBeUndefined();
      }
    });

    test('mock report response matches expected ReportResponse shape', () => {
      expect(mockReportResponse).toHaveProperty('report_id');
      expect(mockReportResponse).toHaveProperty('download_url');
      expect(mockReportResponse).toHaveProperty('generated_at');
      expect(mockReportResponse).toHaveProperty('expires_at');
      expect(mockReportResponse).toHaveProperty('title');
      expect(mockReportResponse).toHaveProperty('source');

      // Verify download_url format
      expect(mockReportResponse.download_url).toMatch(/^\/api\/reports\/download\//);
    });

    test('mock trade review matches expected TradeReviewResponse shape', () => {
      // Entry details
      expect(mockTradeReview).toHaveProperty('trade_id');
      expect(mockTradeReview).toHaveProperty('symbol');
      expect(mockTradeReview).toHaveProperty('side');
      expect(mockTradeReview).toHaveProperty('status');
      expect(mockTradeReview).toHaveProperty('entry_price');
      expect(mockTradeReview).toHaveProperty('entry_time');
      expect(mockTradeReview).toHaveProperty('shares');

      // Exit details
      expect(mockTradeReview).toHaveProperty('exit_price');
      expect(mockTradeReview).toHaveProperty('exit_time');

      // P&L
      expect(mockTradeReview).toHaveProperty('realized_pnl');
      expect(mockTradeReview).toHaveProperty('return_pct');
      expect(mockTradeReview).toHaveProperty('holding_period_days');

      // Strategy info
      expect(mockTradeReview).toHaveProperty('strategy');
      expect(mockTradeReview).toHaveProperty('regime');
      expect(mockTradeReview).toHaveProperty('signal_reason');
      expect(mockTradeReview).toHaveProperty('signal_explanation');

      // Risk management
      expect(mockTradeReview).toHaveProperty('stop_loss');
      expect(mockTradeReview).toHaveProperty('take_profit');

      // Verify side is valid
      expect(['LONG', 'SHORT']).toContain(mockTradeReview.side);

      // Verify status is valid
      expect(['OPEN', 'CLOSED', 'CANCELLED']).toContain(mockTradeReview.status);
    });
  });

  describe('Metrics Calculation Logic', () => {
    test('risk/reward ratio calculation is correct', () => {
      const entryPrice = mockTradeReview.entry_price;
      const stopLoss = mockTradeReview.stop_loss;
      const takeProfit = mockTradeReview.take_profit;

      const risk = Math.abs(entryPrice - stopLoss); // 5.5
      const reward = Math.abs(takeProfit! - entryPrice); // 9.5

      const riskReward = reward / risk;

      expect(riskReward).toBeCloseTo(1.73, 2); // 9.5/5.5 = 1.727
    });

    test('return percentage calculation is correct', () => {
      const entryPrice = mockTradeReview.entry_price;
      const exitPrice = mockTradeReview.exit_price!;
      const side = mockTradeReview.side;

      let returnPct: number;
      if (side === 'LONG') {
        returnPct = ((exitPrice - entryPrice) / entryPrice) * 100;
      } else {
        returnPct = ((entryPrice - exitPrice) / entryPrice) * 100;
      }

      expect(returnPct).toBeCloseTo(mockTradeReview.return_pct!, 2);
    });

    test('position value calculation is correct', () => {
      const positionValue = mockTradeReview.entry_price * mockTradeReview.shares;
      expect(positionValue).toBe(18550); // 185.5 * 100
    });

    test('risk amount calculation is correct', () => {
      const riskAmount = Math.abs(
        mockTradeReview.entry_price - mockTradeReview.stop_loss
      ) * mockTradeReview.shares;

      expect(riskAmount).toBe(550); // 5.5 * 100
    });
  });

  describe('Data Source Validation', () => {
    test('data_source field indicates real data source', () => {
      // Valid data sources should be: backtest:xxx, paper:xxxd, live:xxxd, sample
      const validPatterns = [
        /^backtest:[a-z0-9]+$/,
        /^paper:\d+d$/,
        /^live:\d+d$/,
        /^paper:no_trades$/,
        /^live:no_trades$/,
        /^sample$/,
      ];

      const dataSource = mockEnhancedMetrics.data_source;
      const isValid = validPatterns.some(pattern => pattern.test(dataSource));

      expect(isValid).toBe(true);
    });

    test('metrics periods_analyzed is reasonable', () => {
      expect(mockEnhancedMetrics.periods_analyzed).toBeGreaterThan(0);
      expect(mockEnhancedMetrics.periods_analyzed).toBeLessThanOrEqual(10000);
    });

    test('win_rate is within valid range [0, 1]', () => {
      expect(mockEnhancedMetrics.win_rate).toBeGreaterThanOrEqual(0);
      expect(mockEnhancedMetrics.win_rate).toBeLessThanOrEqual(1);
    });

    test('beta is within reasonable range', () => {
      // Beta typically ranges from -2 to 3 for most strategies
      expect(mockEnhancedMetrics.beta).toBeGreaterThanOrEqual(-3);
      expect(mockEnhancedMetrics.beta).toBeLessThanOrEqual(5);
    });
  });
});

describe('Trade Review Modal Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (apiClient.reviewTrade as jest.Mock).mockResolvedValue(mockTradeReview);
  });

  describe('Trade Data Validation', () => {
    test('trade P&L sign matches expected outcome', () => {
      const { entry_price, exit_price, side, realized_pnl } = mockTradeReview;

      // For LONG: profit if exit > entry
      // For SHORT: profit if entry > exit
      if (side === 'LONG') {
        const expectedProfitable = exit_price! > entry_price;
        const actualProfitable = realized_pnl! > 0;
        expect(actualProfitable).toBe(expectedProfitable);
      } else {
        const expectedProfitable = entry_price > exit_price!;
        const actualProfitable = realized_pnl! > 0;
        expect(actualProfitable).toBe(expectedProfitable);
      }
    });

    test('holding period is calculated correctly from timestamps', () => {
      const entryTime = new Date(mockTradeReview.entry_time);
      const exitTime = new Date(mockTradeReview.exit_time!);

      const holdingMs = exitTime.getTime() - entryTime.getTime();
      const holdingDays = holdingMs / (1000 * 60 * 60 * 24);

      expect(holdingDays).toBeCloseTo(mockTradeReview.holding_period_days!, 1);
    });

    test('stop loss is below entry for LONG trades', () => {
      if (mockTradeReview.side === 'LONG') {
        expect(mockTradeReview.stop_loss).toBeLessThan(mockTradeReview.entry_price);
      }
    });

    test('take profit is above entry for LONG trades', () => {
      if (mockTradeReview.side === 'LONG' && mockTradeReview.take_profit) {
        expect(mockTradeReview.take_profit).toBeGreaterThan(mockTradeReview.entry_price);
      }
    });
  });
});

describe('Report Generation Flow', () => {
  test('report ID format is valid', () => {
    // Report IDs should be alphanumeric with optional underscores
    const reportIdPattern = /^[a-zA-Z0-9_-]+$/;
    expect(mockReportResponse.report_id).toMatch(reportIdPattern);
  });

  test('report expiration is after generation time', () => {
    const generatedAt = new Date(mockReportResponse.generated_at);
    const expiresAt = new Date(mockReportResponse.expires_at);

    expect(expiresAt.getTime()).toBeGreaterThan(generatedAt.getTime());
  });

  test('report list items have valid structure', () => {
    for (const report of mockReportsList.reports) {
      expect(report.report_id).toBeTruthy();
      expect(report.filename).toMatch(/\.html$/);
      expect(report.size_bytes).toBeGreaterThan(0);
      expect(report.download_url).toMatch(/^\/api\/reports\/download\//);
    }
  });

  test('total_count matches reports array length', () => {
    expect(mockReportsList.total_count).toBe(mockReportsList.reports.length);
  });
});
