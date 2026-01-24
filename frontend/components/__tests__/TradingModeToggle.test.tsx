/**
 * Unit tests for TradingModeToggle component
 *
 * Tests:
 * - Component rendering with paper/live modes
 * - Mode switching behavior
 * - Confirmation dialog for live mode
 * - Acknowledgment checkbox
 * - Broker configuration status
 * - Visual indicators (colors, labels)
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { TradingModeToggle, TradingMode } from '../TradingModeToggle';

describe('TradingModeToggle Component', () => {
  const mockOnModeChange = jest.fn().mockResolvedValue(undefined);

  beforeEach(() => {
    jest.clearAllMocks();
  });

  // Test 1: Component renders with paper mode
  test('renders with paper mode', () => {
    render(<TradingModeToggle mode="paper" />);

    // Paper appears twice: as indicator label and as button
    expect(screen.getAllByText('Paper').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Live').length).toBeGreaterThan(0);

    // Paper button should be highlighted (active)
    const paperButton = screen.getAllByRole('button').find(
      (btn) => btn.textContent === 'Paper'
    );
    expect(paperButton).toHaveClass('bg-profit-500');
  });

  // Test 2: Component renders with live mode
  test('renders with live mode', () => {
    render(<TradingModeToggle mode="live" />);

    const liveButton = screen.getAllByRole('button').find(
      (btn) => btn.textContent === 'Live'
    );
    expect(liveButton).toHaveClass('bg-loss-500');
  });

  // Test 3: Switching to paper mode works directly (no confirmation)
  test('switching to paper mode works without confirmation', async () => {
    render(
      <TradingModeToggle
        mode="live"
        onModeChange={mockOnModeChange}
      />
    );

    const paperButton = screen.getAllByRole('button').find(
      (btn) => btn.textContent === 'Paper'
    );
    fireEvent.click(paperButton!);

    await waitFor(() => {
      expect(mockOnModeChange).toHaveBeenCalledWith('paper', false);
    });
  });

  // Test 4: Switching to live mode shows confirmation dialog
  test('switching to live mode shows confirmation dialog', () => {
    render(<TradingModeToggle mode="paper" />);

    const liveButton = screen.getAllByRole('button').find(
      (btn) => btn.textContent === 'Live'
    );
    fireEvent.click(liveButton!);

    // Confirmation dialog should appear
    expect(screen.getByText('Switch to Live Trading?')).toBeInTheDocument();
  });

  // Test 5: Confirmation dialog can be cancelled
  test('confirmation dialog can be cancelled', async () => {
    render(
      <TradingModeToggle
        mode="paper"
        onModeChange={mockOnModeChange}
      />
    );

    // Click live to open confirmation
    const liveButton = screen.getAllByRole('button').find(
      (btn) => btn.textContent === 'Live'
    );
    fireEvent.click(liveButton!);

    expect(screen.getByText('Switch to Live Trading?')).toBeInTheDocument();

    // Click cancel
    const cancelButton = screen.getByRole('button', { name: /cancel/i });
    fireEvent.click(cancelButton);

    // Dialog should be gone
    expect(screen.queryByText('Switch to Live Trading?')).not.toBeInTheDocument();
    expect(mockOnModeChange).not.toHaveBeenCalled();
  });

  // Test 6: Confirmation dialog requires acknowledgment
  test('confirmation dialog requires acknowledgment checkbox', () => {
    render(
      <TradingModeToggle
        mode="paper"
        isLiveBrokerConfigured={true}
        onModeChange={mockOnModeChange}
      />
    );

    // Click live to open confirmation
    const liveButton = screen.getAllByRole('button').find(
      (btn) => btn.textContent === 'Live'
    );
    fireEvent.click(liveButton!);

    // Switch to Live button should be disabled without checkbox
    const switchButton = screen.getByRole('button', { name: /switch to live/i });
    expect(switchButton).toBeDisabled();

    // Check the acknowledgment checkbox
    const checkbox = screen.getByRole('checkbox');
    fireEvent.click(checkbox);

    // Now the button should be enabled
    expect(switchButton).not.toBeDisabled();
  });

  // Test 7: Cannot switch to live without broker configured
  test('cannot switch to live mode without broker configured', () => {
    render(
      <TradingModeToggle
        mode="paper"
        isLiveBrokerConfigured={false}
        onModeChange={mockOnModeChange}
      />
    );

    // Click live to open confirmation
    const liveButton = screen.getAllByRole('button').find(
      (btn) => btn.textContent === 'Live'
    );
    fireEvent.click(liveButton!);

    // Should show broker not configured message
    expect(screen.getByText(/no broker configured/i)).toBeInTheDocument();

    // Check the acknowledgment checkbox
    const checkbox = screen.getByRole('checkbox');
    fireEvent.click(checkbox);

    // Switch to Live button should still be disabled
    const switchButton = screen.getByRole('button', { name: /switch to live/i });
    expect(switchButton).toBeDisabled();
  });

  // Test 8: Can switch to live with acknowledgment and broker configured
  test('can switch to live with acknowledgment and broker configured', async () => {
    render(
      <TradingModeToggle
        mode="paper"
        isLiveBrokerConfigured={true}
        onModeChange={mockOnModeChange}
      />
    );

    // Click live to open confirmation
    const liveButton = screen.getAllByRole('button').find(
      (btn) => btn.textContent === 'Live'
    );
    fireEvent.click(liveButton!);

    // Check the acknowledgment checkbox
    const checkbox = screen.getByRole('checkbox');
    fireEvent.click(checkbox);

    // Click Switch to Live
    const switchButton = screen.getByRole('button', { name: /switch to live/i });
    fireEvent.click(switchButton);

    await waitFor(() => {
      expect(mockOnModeChange).toHaveBeenCalledWith('live', true);
    });
  });

  // Test 9: Shows loading state
  test('shows loading skeleton when isLoading is true', () => {
    render(<TradingModeToggle mode="paper" isLoading={true} />);

    // Should show skeleton loader
    const skeleton = document.querySelector('.animate-pulse');
    expect(skeleton).toBeInTheDocument();

    // Should not show buttons
    expect(screen.queryByRole('button', { name: /paper/i })).not.toBeInTheDocument();
  });

  // Test 10: Shows broker configured status in dialog
  test('shows broker configured status in dialog', () => {
    render(
      <TradingModeToggle
        mode="paper"
        isLiveBrokerConfigured={true}
        onModeChange={mockOnModeChange}
      />
    );

    // Click live to open confirmation
    const liveButton = screen.getAllByRole('button').find(
      (btn) => btn.textContent === 'Live'
    );
    fireEvent.click(liveButton!);

    // Should show broker ready message
    expect(screen.getByText(/broker configured and ready/i)).toBeInTheDocument();
  });

  // Test 11: Mode indicator shows correct color for paper
  test('mode indicator shows green for paper mode', () => {
    render(<TradingModeToggle mode="paper" />);

    const indicator = document.querySelector('.bg-profit-500');
    expect(indicator).toBeInTheDocument();
  });

  // Test 12: Mode indicator shows correct color for live
  test('mode indicator shows red for live mode', () => {
    render(<TradingModeToggle mode="live" />);

    const indicator = document.querySelector('.bg-loss-500');
    expect(indicator).toBeInTheDocument();
  });

  // Test 13: Custom className is applied
  test('applies custom className', () => {
    const { container } = render(
      <TradingModeToggle mode="paper" className="custom-class" />
    );

    expect(container.firstChild).toHaveClass('custom-class');
  });

  // Test 14: Clicking same mode does nothing
  test('clicking current mode does nothing', () => {
    render(
      <TradingModeToggle
        mode="paper"
        onModeChange={mockOnModeChange}
      />
    );

    const paperButton = screen.getAllByRole('button').find(
      (btn) => btn.textContent === 'Paper'
    );
    fireEvent.click(paperButton!);

    expect(mockOnModeChange).not.toHaveBeenCalled();
  });
});
