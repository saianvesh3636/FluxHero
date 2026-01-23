/**
 * Unit tests for Settings Page
 *
 * Tests:
 * - Loading state display
 * - Empty broker list display
 * - Broker list rendering
 * - Add broker form display and submission
 * - Connection test functionality
 * - Delete broker functionality
 * - Form validation
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import SettingsPage from '../page';
import { apiClient } from '../../../utils/api';

// Mock the API client
jest.mock('../../../utils/api', () => ({
  apiClient: {
    getBrokers: jest.fn(),
    addBroker: jest.fn(),
    deleteBroker: jest.fn(),
    getBrokerHealth: jest.fn(),
  },
  ApiError: class ApiError extends Error {
    status: number;
    detail: string;
    constructor(status: number, detail: string) {
      super(detail);
      this.status = status;
      this.detail = detail;
    }
  },
}));

describe('Settings Page', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  const mockBroker = {
    id: 'abc123',
    broker_type: 'alpaca',
    name: 'My Alpaca Account',
    api_key_masked: 'PK***...***XY',
    base_url: 'https://paper-api.alpaca.markets',
    is_connected: false,
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
  };

  const mockHealthResponse = {
    id: 'abc123',
    name: 'My Alpaca Account',
    broker_type: 'alpaca',
    is_connected: true,
    is_authenticated: true,
    latency_ms: 45,
    last_heartbeat: '2024-01-01T00:00:00Z',
    error_message: null,
  };

  test('displays loading state initially', () => {
    (apiClient.getBrokers as jest.Mock).mockImplementation(
      () => new Promise(() => {}) // Never resolves
    );

    render(<SettingsPage />);

    expect(screen.getByText('Settings')).toBeInTheDocument();
    expect(screen.getByText('Configure broker connections')).toBeInTheDocument();
  });

  test('displays empty state when no brokers configured', async () => {
    (apiClient.getBrokers as jest.Mock).mockResolvedValue({
      brokers: [],
      total: 0,
    });

    render(<SettingsPage />);

    await waitFor(() => {
      expect(screen.getByText('No brokers configured')).toBeInTheDocument();
      expect(screen.getByText('Add Your First Broker')).toBeInTheDocument();
    });
  });

  test('displays broker list when brokers exist', async () => {
    (apiClient.getBrokers as jest.Mock).mockResolvedValue({
      brokers: [mockBroker],
      total: 1,
    });

    render(<SettingsPage />);

    await waitFor(() => {
      expect(screen.getByText('My Alpaca Account')).toBeInTheDocument();
      expect(screen.getByText('alpaca')).toBeInTheDocument();
      expect(screen.getByText(/PK\*\*\*\.\.\.\*\*\*XY/)).toBeInTheDocument();
      expect(screen.getByText('Test Connection')).toBeInTheDocument();
      expect(screen.getByText('Delete')).toBeInTheDocument();
    });
  });

  test('shows add broker form when Add Broker button is clicked', async () => {
    (apiClient.getBrokers as jest.Mock).mockResolvedValue({
      brokers: [],
      total: 0,
    });

    render(<SettingsPage />);

    await waitFor(() => {
      expect(screen.getByText('Add Your First Broker')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Add Your First Broker'));

    await waitFor(() => {
      // Check for the form title (h3 element)
      expect(screen.getByRole('heading', { name: 'Add Broker' })).toBeInTheDocument();
      expect(screen.getByLabelText('Display Name')).toBeInTheDocument();
      expect(screen.getByLabelText('API Key')).toBeInTheDocument();
      expect(screen.getByLabelText('API Secret')).toBeInTheDocument();
      expect(screen.getByText('Paper Trading')).toBeInTheDocument();
      expect(screen.getByText('Live Trading')).toBeInTheDocument();
    });
  });

  test('validates form fields before submission', async () => {
    (apiClient.getBrokers as jest.Mock).mockResolvedValue({
      brokers: [],
      total: 0,
    });

    render(<SettingsPage />);

    await waitFor(() => {
      expect(screen.getByText('Add Your First Broker')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Add Your First Broker'));

    // Submit form without filling required fields
    const submitButtons = screen.getAllByText('Add Broker');
    const formSubmitButton = submitButtons.find(
      (btn) => btn.tagName === 'BUTTON' && btn.getAttribute('type') === 'submit'
    );
    fireEvent.click(formSubmitButton!);

    await waitFor(() => {
      expect(screen.getByText('Name is required')).toBeInTheDocument();
    });
  });

  test('submits form successfully with valid data', async () => {
    (apiClient.getBrokers as jest.Mock)
      .mockResolvedValueOnce({ brokers: [], total: 0 })
      .mockResolvedValueOnce({ brokers: [mockBroker], total: 1 });

    (apiClient.addBroker as jest.Mock).mockResolvedValue(mockBroker);

    render(<SettingsPage />);

    await waitFor(() => {
      expect(screen.getByText('Add Your First Broker')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Add Your First Broker'));

    // Fill in form
    fireEvent.change(screen.getByLabelText('Display Name'), {
      target: { value: 'My Alpaca Account' },
    });
    fireEvent.change(screen.getByLabelText('API Key'), {
      target: { value: 'PKTEST123456' },
    });
    fireEvent.change(screen.getByLabelText('API Secret'), {
      target: { value: 'secretkey123' },
    });

    // Submit form
    const submitButtons = screen.getAllByText('Add Broker');
    const formSubmitButton = submitButtons.find(
      (btn) => btn.tagName === 'BUTTON' && btn.getAttribute('type') === 'submit'
    );
    fireEvent.click(formSubmitButton!);

    await waitFor(() => {
      expect(apiClient.addBroker).toHaveBeenCalledWith({
        broker_type: 'alpaca',
        name: 'My Alpaca Account',
        api_key: 'PKTEST123456',
        api_secret: 'secretkey123',
        base_url: 'https://paper-api.alpaca.markets',
      });
    });

    // Form should close and broker should appear in list
    await waitFor(() => {
      expect(screen.getByText('My Alpaca Account')).toBeInTheDocument();
    });
  });

  test('test connection button shows health status', async () => {
    (apiClient.getBrokers as jest.Mock).mockResolvedValue({
      brokers: [mockBroker],
      total: 1,
    });
    (apiClient.getBrokerHealth as jest.Mock).mockResolvedValue(mockHealthResponse);

    render(<SettingsPage />);

    await waitFor(() => {
      expect(screen.getByText('Test Connection')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Test Connection'));

    await waitFor(() => {
      expect(apiClient.getBrokerHealth).toHaveBeenCalledWith('abc123');
      expect(screen.getByText('45ms')).toBeInTheDocument();
    });
  });

  test('delete button shows confirmation dialog', async () => {
    (apiClient.getBrokers as jest.Mock).mockResolvedValue({
      brokers: [mockBroker],
      total: 1,
    });

    render(<SettingsPage />);

    await waitFor(() => {
      expect(screen.getByText('Delete')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Delete'));

    await waitFor(() => {
      expect(screen.getByText('Delete this broker?')).toBeInTheDocument();
      expect(screen.getByText('Confirm')).toBeInTheDocument();
      expect(screen.getByText('Cancel')).toBeInTheDocument();
    });
  });

  test('delete confirmation removes broker', async () => {
    (apiClient.getBrokers as jest.Mock)
      .mockResolvedValueOnce({ brokers: [mockBroker], total: 1 })
      .mockResolvedValueOnce({ brokers: [], total: 0 });
    (apiClient.deleteBroker as jest.Mock).mockResolvedValue(undefined);

    render(<SettingsPage />);

    await waitFor(() => {
      expect(screen.getByText('Delete')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Delete'));

    await waitFor(() => {
      expect(screen.getByText('Confirm')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Confirm'));

    await waitFor(() => {
      expect(apiClient.deleteBroker).toHaveBeenCalledWith('abc123');
      expect(screen.getByText('No brokers configured')).toBeInTheDocument();
    });
  });

  test('cancel delete hides confirmation dialog', async () => {
    (apiClient.getBrokers as jest.Mock).mockResolvedValue({
      brokers: [mockBroker],
      total: 1,
    });

    render(<SettingsPage />);

    await waitFor(() => {
      expect(screen.getByText('Delete')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Delete'));

    await waitFor(() => {
      expect(screen.getByText('Confirm')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Cancel'));

    await waitFor(() => {
      expect(screen.queryByText('Delete this broker?')).not.toBeInTheDocument();
      // Delete button should still be visible (broker not deleted)
      expect(screen.getByText('Delete')).toBeInTheDocument();
    });
  });

  test('environment toggle switches between paper and live', async () => {
    (apiClient.getBrokers as jest.Mock).mockResolvedValue({
      brokers: [],
      total: 0,
    });

    render(<SettingsPage />);

    await waitFor(() => {
      expect(screen.getByText('Add Your First Broker')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Add Your First Broker'));

    // Default should be paper trading
    expect(screen.getByText('Paper trading uses simulated funds')).toBeInTheDocument();

    // Click Live Trading
    fireEvent.click(screen.getByText('Live Trading'));

    await waitFor(() => {
      expect(screen.getByText('Warning: Live trading uses real money')).toBeInTheDocument();
    });

    // Click Paper Trading
    fireEvent.click(screen.getByText('Paper Trading'));

    await waitFor(() => {
      expect(screen.getByText('Paper trading uses simulated funds')).toBeInTheDocument();
    });
  });

  test('cancel button closes form without submitting', async () => {
    (apiClient.getBrokers as jest.Mock).mockResolvedValue({
      brokers: [],
      total: 0,
    });

    render(<SettingsPage />);

    await waitFor(() => {
      expect(screen.getByText('Add Your First Broker')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Add Your First Broker'));

    // Form should be visible
    expect(screen.getByLabelText('Display Name')).toBeInTheDocument();

    // Fill in some data
    fireEvent.change(screen.getByLabelText('Display Name'), {
      target: { value: 'Test Account' },
    });

    // Click Cancel
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));

    await waitFor(() => {
      // Form should be hidden
      expect(screen.queryByLabelText('Display Name')).not.toBeInTheDocument();
      // Add broker should not have been called
      expect(apiClient.addBroker).not.toHaveBeenCalled();
    });
  });

  test('displays connection error from health check', async () => {
    (apiClient.getBrokers as jest.Mock).mockResolvedValue({
      brokers: [mockBroker],
      total: 1,
    });
    (apiClient.getBrokerHealth as jest.Mock).mockResolvedValue({
      ...mockHealthResponse,
      is_connected: false,
      is_authenticated: false,
      error_message: 'Invalid API credentials',
    });

    render(<SettingsPage />);

    await waitFor(() => {
      expect(screen.getByText('Test Connection')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Test Connection'));

    await waitFor(() => {
      expect(screen.getByText('Invalid API credentials')).toBeInTheDocument();
    });
  });
});
