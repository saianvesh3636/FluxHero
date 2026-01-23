/**
 * Unit tests for Backtest Page
 *
 * Tests:
 * - Form rendering and configuration
 * - Error handling and display
 * - Retry functionality
 * - Loading states with LoadingButton
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import BacktestPage from '../page';

// Mock fetch
global.fetch = jest.fn();

describe('Backtest Page', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders page title and form', () => {
    render(<BacktestPage />);

    expect(screen.getByText('Backtesting Module')).toBeInTheDocument();
    expect(screen.getByText('Test your strategy against historical data')).toBeInTheDocument();
    expect(screen.getByText('Run Backtest')).toBeInTheDocument();
  });

  test('displays error message when backtest fails', async () => {
    (global.fetch as jest.Mock).mockRejectedValue(
      new Error('API request failed')
    );

    render(<BacktestPage />);

    const runButton = screen.getByText('Run Backtest');
    fireEvent.click(runButton);

    await waitFor(() => {
      expect(screen.getByText('Error')).toBeInTheDocument();
      expect(screen.getByText('API request failed')).toBeInTheDocument();
    });
  });

  test('displays retry button when error occurs', async () => {
    (global.fetch as jest.Mock).mockRejectedValue(
      new Error('Network error')
    );

    render(<BacktestPage />);

    const runButton = screen.getByText('Run Backtest');
    fireEvent.click(runButton);

    await waitFor(() => {
      expect(screen.getByText('Retry')).toBeInTheDocument();
    });
  });

  test('retry button triggers backtest again', async () => {
    // Mock initial failure
    (global.fetch as jest.Mock).mockRejectedValueOnce(
      new Error('Network error')
    );

    render(<BacktestPage />);

    const runButton = screen.getByText('Run Backtest');
    fireEvent.click(runButton);

    // Wait for error
    await waitFor(() => {
      expect(screen.getByText('Error')).toBeInTheDocument();
    });

    // Mock successful response for retry
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({
        totalReturn: 1000,
        totalReturnPct: 10,
        sharpeRatio: 1.5,
        maxDrawdown: 15,
        winRate: 60,
        totalTrades: 50,
        avgWin: 50,
        avgLoss: 30,
        profitFactor: 1.67,
        equity_curve: [],
        trade_log: [],
      }),
    });

    // Click retry
    const retryButton = screen.getByText('Retry');
    fireEvent.click(retryButton);

    // Wait for results
    await waitFor(() => {
      expect(screen.getByText('Backtest Results')).toBeInTheDocument();
    });

    // Verify fetch was called twice
    expect(global.fetch).toHaveBeenCalledTimes(2);
  });

  test('shows loading state when running backtest', async () => {
    (global.fetch as jest.Mock).mockImplementation(
      () => new Promise(() => {}) // Never resolves
    );

    render(<BacktestPage />);

    const runButton = screen.getByText('Run Backtest');
    fireEvent.click(runButton);

    // Button should show loading state
    await waitFor(() => {
      expect(runButton).toBeDisabled();
    });
  });

  test('handles API error with status code', async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: false,
      statusText: 'Internal Server Error',
    });

    render(<BacktestPage />);

    const runButton = screen.getByText('Run Backtest');
    fireEvent.click(runButton);

    await waitFor(() => {
      expect(screen.getByText(/Backtest failed: Internal Server Error/)).toBeInTheDocument();
    });
  });

  test('clears error when running new backtest', async () => {
    // Mock initial failure
    (global.fetch as jest.Mock).mockRejectedValueOnce(
      new Error('First error')
    );

    render(<BacktestPage />);

    const runButton = screen.getByText('Run Backtest');
    fireEvent.click(runButton);

    // Wait for error
    await waitFor(() => {
      expect(screen.getByText('First error')).toBeInTheDocument();
    });

    // Mock second failure with different error
    (global.fetch as jest.Mock).mockRejectedValueOnce(
      new Error('Second error')
    );

    // Click run again
    fireEvent.click(runButton);

    // Old error should be replaced
    await waitFor(() => {
      expect(screen.queryByText('First error')).not.toBeInTheDocument();
      expect(screen.getByText('Second error')).toBeInTheDocument();
    });
  });

  test('displays successful backtest results', async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({
        totalReturn: 5000,
        totalReturnPct: 50,
        sharpeRatio: 2.0,
        maxDrawdown: 10,
        winRate: 65,
        totalTrades: 100,
        avgWin: 100,
        avgLoss: 50,
        profitFactor: 2.5,
        equity_curve: [],
        trade_log: [
          {
            entry_date: '2024-01-01',
            exit_date: '2024-01-05',
            symbol: 'SPY',
            side: 'LONG',
            entry_price: 450,
            exit_price: 460,
            shares: 10,
            pnl: 100,
          },
        ],
      }),
    });

    render(<BacktestPage />);

    const runButton = screen.getByText('Run Backtest');
    fireEvent.click(runButton);

    await waitFor(() => {
      expect(screen.getByText('Backtest Results')).toBeInTheDocument();
      expect(screen.getByText('+$5,000.00')).toBeInTheDocument();
      expect(screen.getByText('(+50.00%)')).toBeInTheDocument();
    });
  });

  test('error is cleared when retry is clicked', async () => {
    // Mock initial failure
    (global.fetch as jest.Mock).mockRejectedValueOnce(
      new Error('Network error')
    );

    render(<BacktestPage />);

    const runButton = screen.getByText('Run Backtest');
    fireEvent.click(runButton);

    // Wait for error
    await waitFor(() => {
      expect(screen.getByText('Retry')).toBeInTheDocument();
      expect(screen.getByText('Network error')).toBeInTheDocument();
    });

    // Mock slow response for retry
    (global.fetch as jest.Mock).mockImplementation(
      () => new Promise(() => {}) // Never resolves
    );

    // Click retry
    const retryButton = screen.getByText('Retry');
    fireEvent.click(retryButton);

    // Error should be cleared when running
    await waitFor(() => {
      expect(screen.queryByText('Network error')).not.toBeInTheDocument();
    });
  });
});
