/**
 * Unit tests for Home Page
 *
 * Tests:
 * - Backend status checking
 * - Backend offline indicator
 * - Retry functionality
 * - Loading states
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import Home from '../page';
import { apiClient } from '../../utils/api';

// Mock the API client
jest.mock('../../utils/api', () => ({
  apiClient: {
    getSystemStatus: jest.fn(),
  },
}));

describe('Home Page', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders page title and description', () => {
    (apiClient.getSystemStatus as jest.Mock).mockResolvedValue({
      status: 'active',
      last_update: '2024-01-01T00:00:00Z',
      uptime_seconds: 3600,
    });

    render(<Home />);

    expect(screen.getByText('FluxHero Trading System')).toBeInTheDocument();
    expect(screen.getByText('Adaptive retail quantitative trading platform')).toBeInTheDocument();
  });

  test('displays loading state initially', () => {
    (apiClient.getSystemStatus as jest.Mock).mockImplementation(
      () => new Promise(() => {}) // Never resolves
    );

    render(<Home />);

    expect(screen.getByText('Checking backend status...')).toBeInTheDocument();
  });

  test('displays backend online status when API is available', async () => {
    (apiClient.getSystemStatus as jest.Mock).mockResolvedValue({
      status: 'active',
      last_update: '2024-01-01T00:00:00Z',
      uptime_seconds: 3600,
    });

    render(<Home />);

    await waitFor(() => {
      expect(screen.getByText('Connected')).toBeInTheDocument();
      expect(screen.getByText('ACTIVE')).toBeInTheDocument();
      expect(screen.getByText('Backend API is connected and ready.')).toBeInTheDocument();
    });
  });

  test('displays backend offline indicator when API fails', async () => {
    (apiClient.getSystemStatus as jest.Mock).mockRejectedValue(
      new Error('Network error')
    );

    render(<Home />);

    await waitFor(() => {
      expect(screen.getByText('Backend Offline')).toBeInTheDocument();
      expect(screen.getByText(/Unable to connect to the backend server/)).toBeInTheDocument();
    });
  });

  test('retry button appears when backend is offline', async () => {
    (apiClient.getSystemStatus as jest.Mock).mockRejectedValue(
      new Error('Connection refused')
    );

    render(<Home />);

    await waitFor(() => {
      expect(screen.getByText('Retry Connection')).toBeInTheDocument();
    });
  });

  test('retry button calls API again when clicked', async () => {
    (apiClient.getSystemStatus as jest.Mock).mockRejectedValueOnce(
      new Error('Connection refused')
    );

    render(<Home />);

    // Wait for initial error state
    await waitFor(() => {
      expect(screen.getByText('Backend Offline')).toBeInTheDocument();
    });

    // Mock successful response for retry
    (apiClient.getSystemStatus as jest.Mock).mockResolvedValue({
      status: 'active',
      last_update: '2024-01-01T00:00:00Z',
      uptime_seconds: 3600,
    });

    // Click retry button
    const retryButton = screen.getByText('Retry Connection');
    fireEvent.click(retryButton);

    // Wait for success state
    await waitFor(() => {
      expect(screen.getByText('Backend API is connected and ready.')).toBeInTheDocument();
      expect(screen.queryByText('Backend Offline')).not.toBeInTheDocument();
    });

    // Verify API was called twice (initial + retry)
    expect(apiClient.getSystemStatus).toHaveBeenCalledTimes(2);
  });

  test('displays delayed status with warning badge', async () => {
    (apiClient.getSystemStatus as jest.Mock).mockResolvedValue({
      status: 'delayed',
      last_update: '2024-01-01T00:00:00Z',
      uptime_seconds: 3600,
    });

    render(<Home />);

    await waitFor(() => {
      expect(screen.getByText('DELAYED')).toBeInTheDocument();
    });
  });

  test('displays offline status with error badge', async () => {
    (apiClient.getSystemStatus as jest.Mock).mockResolvedValue({
      status: 'offline',
      last_update: '2024-01-01T00:00:00Z',
      uptime_seconds: 0,
    });

    render(<Home />);

    await waitFor(() => {
      expect(screen.getByText('OFFLINE')).toBeInTheDocument();
    });
  });
});
