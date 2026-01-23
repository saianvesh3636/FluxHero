/**
 * Unit tests for Live Trading Page
 *
 * Tests:
 * - Loading state display
 * - Error states and handling
 * - Backend offline indicator
 * - Retry functionality
 * - Data fetching and display
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import LiveTradingPage from '../page';
import { apiClient } from '../../../utils/api';

// Mock the API client
jest.mock('../../../utils/api', () => ({
  apiClient: {
    getPositions: jest.fn(),
    getAccountInfo: jest.fn(),
    getSystemStatus: jest.fn(),
  },
}));

describe('Live Trading Page', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.runOnlyPendingTimers();
    jest.useRealTimers();
  });

  const mockPositions = [
    {
      symbol: 'SPY',
      quantity: 10,
      entry_price: 450.0,
      current_price: 455.0,
      pnl: 50.0,
      pnl_percent: 1.11,
    },
  ];

  const mockAccountInfo = {
    equity: 100000,
    cash: 50000,
    buying_power: 200000,
    daily_pnl: 500,
    total_pnl: 5000,
  };

  const mockSystemStatus = {
    status: 'active' as const,
    last_update: '2024-01-01T00:00:00Z',
    uptime_seconds: 3600,
  };

  test('displays loading state initially', () => {
    (apiClient.getPositions as jest.Mock).mockImplementation(
      () => new Promise(() => {}) // Never resolves
    );
    (apiClient.getAccountInfo as jest.Mock).mockImplementation(
      () => new Promise(() => {})
    );
    (apiClient.getSystemStatus as jest.Mock).mockImplementation(
      () => new Promise(() => {})
    );

    render(<LiveTradingPage />);

    expect(screen.getByText('Loading...')).toBeInTheDocument();
  });

  test('displays data when API calls succeed', async () => {
    (apiClient.getPositions as jest.Mock).mockResolvedValue(mockPositions);
    (apiClient.getAccountInfo as jest.Mock).mockResolvedValue(mockAccountInfo);
    (apiClient.getSystemStatus as jest.Mock).mockResolvedValue(mockSystemStatus);

    render(<LiveTradingPage />);

    await waitFor(() => {
      expect(screen.getByText('Live Trading')).toBeInTheDocument();
      expect(screen.getByText('SPY')).toBeInTheDocument();
      expect(screen.getByText(/Open Positions \(1\)/)).toBeInTheDocument();
    });
  });

  test('displays backend offline indicator when all API calls fail', async () => {
    (apiClient.getPositions as jest.Mock).mockRejectedValue(
      new Error('API request failed: Network error')
    );
    (apiClient.getAccountInfo as jest.Mock).mockRejectedValue(
      new Error('API request failed: Network error')
    );
    (apiClient.getSystemStatus as jest.Mock).mockRejectedValue(
      new Error('API request failed: Network error')
    );

    render(<LiveTradingPage />);

    await waitFor(() => {
      expect(screen.getByText('Backend Offline')).toBeInTheDocument();
      expect(screen.getByText(/API request failed: Network error/)).toBeInTheDocument();
    });
  });

  test('retry button appears when backend is offline', async () => {
    (apiClient.getPositions as jest.Mock).mockRejectedValue(
      new Error('Connection refused')
    );
    (apiClient.getAccountInfo as jest.Mock).mockRejectedValue(
      new Error('Connection refused')
    );
    (apiClient.getSystemStatus as jest.Mock).mockRejectedValue(
      new Error('Connection refused')
    );

    render(<LiveTradingPage />);

    await waitFor(() => {
      expect(screen.getByText('Retry Connection')).toBeInTheDocument();
    });
  });

  test('retry button refetches data when clicked', async () => {
    // Mock initial failure
    (apiClient.getPositions as jest.Mock).mockRejectedValueOnce(
      new Error('Network error')
    );
    (apiClient.getAccountInfo as jest.Mock).mockRejectedValueOnce(
      new Error('Network error')
    );
    (apiClient.getSystemStatus as jest.Mock).mockRejectedValueOnce(
      new Error('Network error')
    );

    render(<LiveTradingPage />);

    // Wait for error state
    await waitFor(() => {
      expect(screen.getByText('Backend Offline')).toBeInTheDocument();
    });

    // Mock successful retry
    (apiClient.getPositions as jest.Mock).mockResolvedValue(mockPositions);
    (apiClient.getAccountInfo as jest.Mock).mockResolvedValue(mockAccountInfo);
    (apiClient.getSystemStatus as jest.Mock).mockResolvedValue(mockSystemStatus);

    // Click retry
    const retryButton = screen.getByText('Retry Connection');
    fireEvent.click(retryButton);

    // Wait for success state
    await waitFor(() => {
      expect(screen.getByText('SPY')).toBeInTheDocument();
      expect(screen.queryByText('Backend Offline')).not.toBeInTheDocument();
    });

    // Verify APIs were called twice (initial + retry)
    expect(apiClient.getPositions).toHaveBeenCalledTimes(2);
  });

  test('displays no positions message when positions array is empty', async () => {
    (apiClient.getPositions as jest.Mock).mockResolvedValue([]);
    (apiClient.getAccountInfo as jest.Mock).mockResolvedValue(mockAccountInfo);
    (apiClient.getSystemStatus as jest.Mock).mockResolvedValue(mockSystemStatus);

    render(<LiveTradingPage />);

    await waitFor(() => {
      expect(screen.getByText('No open positions')).toBeInTheDocument();
    });
  });

  test('displays system status indicator correctly', async () => {
    (apiClient.getPositions as jest.Mock).mockResolvedValue([]);
    (apiClient.getAccountInfo as jest.Mock).mockResolvedValue(mockAccountInfo);
    (apiClient.getSystemStatus as jest.Mock).mockResolvedValue({
      status: 'active',
      last_update: '2024-01-01T00:00:00Z',
      uptime_seconds: 3600,
    });

    render(<LiveTradingPage />);

    await waitFor(() => {
      expect(screen.getByText('ACTIVE')).toBeInTheDocument();
    });
  });

  test('auto-refreshes data every 5 seconds', async () => {
    (apiClient.getPositions as jest.Mock).mockResolvedValue(mockPositions);
    (apiClient.getAccountInfo as jest.Mock).mockResolvedValue(mockAccountInfo);
    (apiClient.getSystemStatus as jest.Mock).mockResolvedValue(mockSystemStatus);

    render(<LiveTradingPage />);

    // Wait for initial load
    await waitFor(() => {
      expect(screen.getByText('SPY')).toBeInTheDocument();
    });

    // Verify initial call
    expect(apiClient.getPositions).toHaveBeenCalledTimes(1);

    // Fast-forward 5 seconds
    jest.advanceTimersByTime(5000);

    // Wait for refresh
    await waitFor(() => {
      expect(apiClient.getPositions).toHaveBeenCalledTimes(2);
    });
  });

  test('clears interval on unmount', async () => {
    (apiClient.getPositions as jest.Mock).mockResolvedValue(mockPositions);
    (apiClient.getAccountInfo as jest.Mock).mockResolvedValue(mockAccountInfo);
    (apiClient.getSystemStatus as jest.Mock).mockResolvedValue(mockSystemStatus);

    const { unmount } = render(<LiveTradingPage />);

    await waitFor(() => {
      expect(screen.getByText('SPY')).toBeInTheDocument();
    });

    // Unmount component
    unmount();

    // Fast-forward 10 seconds
    jest.advanceTimersByTime(10000);

    // API should not be called again after unmount
    expect(apiClient.getPositions).toHaveBeenCalledTimes(1);
  });
});
