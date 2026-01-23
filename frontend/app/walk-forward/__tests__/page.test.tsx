/**
 * Unit tests for Walk-Forward Page
 *
 * Tests:
 * - Form rendering and configuration
 * - Error handling and display
 * - Retry functionality
 * - Loading states
 * - Results display with pass/fail status
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import WalkForwardPage from '../page';

// Mock fetch
global.fetch = jest.fn();

// Mock ResizeObserver for lightweight-charts
class ResizeObserverMock {
  observe() {}
  unobserve() {}
  disconnect() {}
}
global.ResizeObserver = ResizeObserverMock;

describe('Walk-Forward Page', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders page title and form', () => {
    render(<WalkForwardPage />);

    expect(screen.getByText('Walk-Forward Testing')).toBeInTheDocument();
    expect(
      screen.getByText('Validate strategy robustness with out-of-sample testing')
    ).toBeInTheDocument();
    expect(screen.getByText('Run Walk-Forward Test')).toBeInTheDocument();
  });

  test('renders walk-forward specific configuration inputs', () => {
    render(<WalkForwardPage />);

    expect(screen.getByText('Walk-Forward Parameters')).toBeInTheDocument();
    expect(screen.getByLabelText('Training Period (bars)')).toBeInTheDocument();
    expect(screen.getByLabelText('Test Period (bars)')).toBeInTheDocument();
    expect(screen.getByLabelText('Pass Threshold (%)')).toBeInTheDocument();
  });

  test('displays error message when walk-forward fails', async () => {
    (global.fetch as jest.Mock).mockRejectedValue(
      new Error('API request failed')
    );

    render(<WalkForwardPage />);

    const runButton = screen.getByText('Run Walk-Forward Test');
    fireEvent.click(runButton);

    await waitFor(() => {
      expect(screen.getByText('Error')).toBeInTheDocument();
      expect(screen.getByText('API request failed')).toBeInTheDocument();
    });
  });

  test('displays retry button when error occurs', async () => {
    (global.fetch as jest.Mock).mockRejectedValue(new Error('Network error'));

    render(<WalkForwardPage />);

    const runButton = screen.getByText('Run Walk-Forward Test');
    fireEvent.click(runButton);

    await waitFor(() => {
      expect(screen.getByText('Retry')).toBeInTheDocument();
    });
  });

  test('shows loading state when running walk-forward', async () => {
    (global.fetch as jest.Mock).mockImplementation(
      () => new Promise(() => {}) // Never resolves
    );

    render(<WalkForwardPage />);

    const runButton = screen.getByText('Run Walk-Forward Test');
    fireEvent.click(runButton);

    await waitFor(() => {
      expect(runButton).toBeDisabled();
    });
  });

  test('displays successful walk-forward results with PASS status', async () => {
    const mockResults = {
      symbol: 'SPY',
      start_date: '2022-01-01',
      end_date: '2024-01-01',
      initial_capital: 100000,
      final_capital: 115000,
      total_return_pct: 15,
      total_windows: 10,
      profitable_windows: 7,
      pass_rate: 0.7,
      passes_walk_forward_test: true,
      pass_threshold: 0.6,
      aggregate_sharpe: 1.2,
      aggregate_max_drawdown_pct: 12.5,
      aggregate_win_rate: 0.55,
      total_trades: 45,
      window_results: [
        {
          window_id: 0,
          train_start_date: '2022-01-01',
          train_end_date: '2022-03-31',
          test_start_date: '2022-04-01',
          test_end_date: '2022-04-30',
          initial_equity: 100000,
          final_equity: 102500,
          return_pct: 2.5,
          sharpe_ratio: 1.5,
          max_drawdown_pct: 3.5,
          win_rate: 0.6,
          num_trades: 5,
          is_profitable: true,
        },
      ],
      combined_equity_curve: [100000, 101000, 102500],
      timestamps: ['2022-04-01', '2022-04-15', '2022-04-30'],
      train_bars: 63,
      test_bars: 21,
    };

    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => mockResults,
    });

    render(<WalkForwardPage />);

    const runButton = screen.getByText('Run Walk-Forward Test');
    fireEvent.click(runButton);

    await waitFor(() => {
      expect(screen.getByText('Walk-Forward Results')).toBeInTheDocument();
      expect(screen.getByText('STRATEGY PASSED')).toBeInTheDocument();
      expect(screen.getByText(/Pass Rate: 70.0%/)).toBeInTheDocument();
      expect(screen.getByText(/7 \/ 10 profitable/)).toBeInTheDocument();
    });
  });

  test('displays failed walk-forward results with FAIL status', async () => {
    const mockResults = {
      symbol: 'SPY',
      start_date: '2022-01-01',
      end_date: '2024-01-01',
      initial_capital: 100000,
      final_capital: 95000,
      total_return_pct: -5,
      total_windows: 10,
      profitable_windows: 4,
      pass_rate: 0.4,
      passes_walk_forward_test: false,
      pass_threshold: 0.6,
      aggregate_sharpe: 0.3,
      aggregate_max_drawdown_pct: 18.5,
      aggregate_win_rate: 0.42,
      total_trades: 38,
      window_results: [
        {
          window_id: 0,
          train_start_date: '2022-01-01',
          train_end_date: '2022-03-31',
          test_start_date: '2022-04-01',
          test_end_date: '2022-04-30',
          initial_equity: 100000,
          final_equity: 98500,
          return_pct: -1.5,
          sharpe_ratio: -0.3,
          max_drawdown_pct: 5.5,
          win_rate: 0.4,
          num_trades: 4,
          is_profitable: false,
        },
      ],
      combined_equity_curve: [100000, 99000, 95000],
      timestamps: ['2022-04-01', '2022-04-15', '2022-04-30'],
      train_bars: 63,
      test_bars: 21,
    };

    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => mockResults,
    });

    render(<WalkForwardPage />);

    const runButton = screen.getByText('Run Walk-Forward Test');
    fireEvent.click(runButton);

    await waitFor(() => {
      expect(screen.getByText('Walk-Forward Results')).toBeInTheDocument();
      expect(screen.getByText('STRATEGY FAILED')).toBeInTheDocument();
      expect(screen.getByText(/Pass Rate: 40.0%/)).toBeInTheDocument();
      expect(screen.getByText(/4 \/ 10 profitable/)).toBeInTheDocument();
    });
  });

  test('handles API error with status code', async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: false,
      status: 400,
      json: async () => ({
        detail: 'Insufficient data for walk-forward testing',
      }),
    });

    render(<WalkForwardPage />);

    const runButton = screen.getByText('Run Walk-Forward Test');
    fireEvent.click(runButton);

    await waitFor(() => {
      expect(
        screen.getByText('Insufficient data for walk-forward testing')
      ).toBeInTheDocument();
    });
  });

  test('clears error when running new walk-forward', async () => {
    (global.fetch as jest.Mock).mockRejectedValueOnce(new Error('First error'));

    render(<WalkForwardPage />);

    const runButton = screen.getByText('Run Walk-Forward Test');
    fireEvent.click(runButton);

    await waitFor(() => {
      expect(screen.getByText('First error')).toBeInTheDocument();
    });

    (global.fetch as jest.Mock).mockRejectedValueOnce(
      new Error('Second error')
    );

    fireEvent.click(runButton);

    await waitFor(() => {
      expect(screen.queryByText('First error')).not.toBeInTheDocument();
      expect(screen.getByText('Second error')).toBeInTheDocument();
    });
  });

  test('retry button triggers walk-forward again', async () => {
    (global.fetch as jest.Mock).mockRejectedValueOnce(
      new Error('Network error')
    );

    render(<WalkForwardPage />);

    const runButton = screen.getByText('Run Walk-Forward Test');
    fireEvent.click(runButton);

    await waitFor(() => {
      expect(screen.getByText('Error')).toBeInTheDocument();
    });

    const mockResults = {
      symbol: 'SPY',
      start_date: '2022-01-01',
      end_date: '2024-01-01',
      initial_capital: 100000,
      final_capital: 110000,
      total_return_pct: 10,
      total_windows: 5,
      profitable_windows: 4,
      pass_rate: 0.8,
      passes_walk_forward_test: true,
      pass_threshold: 0.6,
      aggregate_sharpe: 1.0,
      aggregate_max_drawdown_pct: 10,
      aggregate_win_rate: 0.5,
      total_trades: 25,
      window_results: [],
      combined_equity_curve: [100000],
      timestamps: ['2022-01-01'],
      train_bars: 63,
      test_bars: 21,
    };

    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => mockResults,
    });

    const retryButton = screen.getByText('Retry');
    fireEvent.click(retryButton);

    await waitFor(() => {
      expect(screen.getByText('Walk-Forward Results')).toBeInTheDocument();
    });

    expect(global.fetch).toHaveBeenCalledTimes(2);
  });

  test('displays per-window results table', async () => {
    const mockResults = {
      symbol: 'SPY',
      start_date: '2022-01-01',
      end_date: '2024-01-01',
      initial_capital: 100000,
      final_capital: 115000,
      total_return_pct: 15,
      total_windows: 2,
      profitable_windows: 1,
      pass_rate: 0.5,
      passes_walk_forward_test: false,
      pass_threshold: 0.6,
      aggregate_sharpe: 0.8,
      aggregate_max_drawdown_pct: 14,
      aggregate_win_rate: 0.48,
      total_trades: 20,
      window_results: [
        {
          window_id: 0,
          train_start_date: '2022-01-01',
          train_end_date: '2022-03-31',
          test_start_date: '2022-04-01',
          test_end_date: '2022-04-30',
          initial_equity: 100000,
          final_equity: 105000,
          return_pct: 5.0,
          sharpe_ratio: 1.2,
          max_drawdown_pct: 4.0,
          win_rate: 0.55,
          num_trades: 10,
          is_profitable: true,
        },
        {
          window_id: 1,
          train_start_date: '2022-04-01',
          train_end_date: '2022-06-30',
          test_start_date: '2022-07-01',
          test_end_date: '2022-07-31',
          initial_equity: 105000,
          final_equity: 103000,
          return_pct: -1.9,
          sharpe_ratio: -0.5,
          max_drawdown_pct: 6.0,
          win_rate: 0.4,
          num_trades: 10,
          is_profitable: false,
        },
      ],
      combined_equity_curve: [100000, 105000, 103000],
      timestamps: ['2022-04-01', '2022-04-30', '2022-07-31'],
      train_bars: 63,
      test_bars: 21,
    };

    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => mockResults,
    });

    render(<WalkForwardPage />);

    const runButton = screen.getByText('Run Walk-Forward Test');
    fireEvent.click(runButton);

    await waitFor(() => {
      expect(screen.getByText('Per-Window Results')).toBeInTheDocument();
      // Check for PASS/FAIL badges in the table
      expect(screen.getByText('PASS')).toBeInTheDocument();
      expect(screen.getByText('FAIL')).toBeInTheDocument();
    });
  });

  test('displays info box with walk-forward explanation', () => {
    render(<WalkForwardPage />);

    expect(screen.getByText(/Walk-Forward Testing:/)).toBeInTheDocument();
    expect(
      screen.getByText(/Divides data into rolling train\/test windows/)
    ).toBeInTheDocument();
  });
});
