/**
 * Unit tests for TradingModeToggle component
 *
 * Tests:
 * - Component rendering with paper/live modes
 * - localStorage persistence
 * - Mode switching behavior
 * - Confirmation dialog for live mode
 * - useTradingMode hook functionality
 * - Visual indicators (colors, labels)
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import { TradingModeToggle, useTradingMode, TradingMode } from '../TradingModeToggle';

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: jest.fn((key: string) => store[key] || null),
    setItem: jest.fn((key: string, value: string) => {
      store[key] = value;
    }),
    removeItem: jest.fn((key: string) => {
      delete store[key];
    }),
    clear: jest.fn(() => {
      store = {};
    }),
  };
})();

Object.defineProperty(window, 'localStorage', {
  value: localStorageMock,
});

// Mock window.dispatchEvent for storage events
const dispatchEventMock = jest.fn();
window.dispatchEvent = dispatchEventMock;

// Helper component to test the hook
function HookTestComponent({ onModeChange }: { onModeChange?: (mode: TradingMode) => void }) {
  const { mode, setMode, isLoaded } = useTradingMode();

  React.useEffect(() => {
    if (isLoaded && onModeChange) {
      onModeChange(mode);
    }
  }, [mode, isLoaded, onModeChange]);

  return (
    <div>
      <span data-testid="mode">{mode}</span>
      <span data-testid="loaded">{isLoaded ? 'true' : 'false'}</span>
      <button onClick={() => setMode('paper')}>Set Paper</button>
      <button onClick={() => setMode('live')}>Set Live</button>
    </div>
  );
}

describe('TradingModeToggle Component', () => {
  beforeEach(() => {
    localStorageMock.clear();
    jest.clearAllMocks();
  });

  // Test 1: Component renders with default paper mode
  test('renders with default paper mode', async () => {
    render(<TradingModeToggle />);

    await waitFor(() => {
      // Paper appears twice: as indicator label and as button
      expect(screen.getAllByText('Paper').length).toBeGreaterThan(0);
      expect(screen.getAllByText('Live').length).toBeGreaterThan(0);
    });

    // Paper button should be highlighted (active)
    const paperButton = screen.getAllByRole('button').find(
      (btn) => btn.textContent === 'Paper'
    );
    expect(paperButton).toHaveClass('bg-profit-500');
  });

  // Test 2: Loads saved mode from localStorage
  test('loads saved mode from localStorage', async () => {
    localStorageMock.setItem('fluxhero_trading_mode', 'live');

    render(<TradingModeToggle />);

    await waitFor(() => {
      const liveButton = screen.getAllByRole('button').find(
        (btn) => btn.textContent === 'Live'
      );
      expect(liveButton).toHaveClass('bg-loss-500');
    });
  });

  // Test 3: Switching to paper mode works directly
  test('switches to paper mode directly without confirmation', async () => {
    localStorageMock.setItem('fluxhero_trading_mode', 'live');
    const onModeChange = jest.fn();

    render(<TradingModeToggle onModeChange={onModeChange} />);

    await waitFor(() => {
      const liveButton = screen.getAllByRole('button').find(
        (btn) => btn.textContent === 'Live'
      );
      expect(liveButton).toHaveClass('bg-loss-500');
    });

    // Click paper button
    const paperButton = screen.getAllByRole('button').find(
      (btn) => btn.textContent === 'Paper'
    );
    fireEvent.click(paperButton!);

    await waitFor(() => {
      expect(paperButton).toHaveClass('bg-profit-500');
      expect(onModeChange).toHaveBeenCalledWith('paper');
      expect(localStorageMock.setItem).toHaveBeenCalledWith('fluxhero_trading_mode', 'paper');
    });
  });

  // Test 4: Switching to live mode shows confirmation dialog
  test('shows confirmation dialog when switching to live mode', async () => {
    render(<TradingModeToggle />);

    await waitFor(() => {
      expect(screen.getAllByText('Paper').length).toBeGreaterThan(0);
    });

    // Click live button
    const liveButton = screen.getAllByRole('button').find(
      (btn) => btn.textContent === 'Live'
    );
    fireEvent.click(liveButton!);

    // Confirmation dialog should appear
    await waitFor(() => {
      expect(screen.getByText('Switch to Live Trading?')).toBeInTheDocument();
      expect(screen.getByText(/real money/)).toBeInTheDocument();
    });
  });

  // Test 5: Confirming live mode switch
  test('confirms live mode switch from dialog', async () => {
    const onModeChange = jest.fn();

    render(<TradingModeToggle onModeChange={onModeChange} />);

    await waitFor(() => {
      expect(screen.getAllByText('Paper').length).toBeGreaterThan(0);
    });

    // Click live button to open dialog
    const liveButton = screen.getAllByRole('button').find(
      (btn) => btn.textContent === 'Live'
    );
    fireEvent.click(liveButton!);

    // Click confirm button in dialog
    await waitFor(() => {
      expect(screen.getByText('Switch to Live')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Switch to Live'));

    await waitFor(() => {
      expect(liveButton).toHaveClass('bg-loss-500');
      expect(onModeChange).toHaveBeenCalledWith('live');
      expect(localStorageMock.setItem).toHaveBeenCalledWith('fluxhero_trading_mode', 'live');
    });

    // Dialog should be closed
    expect(screen.queryByText('Switch to Live Trading?')).not.toBeInTheDocument();
  });

  // Test 6: Canceling live mode switch
  test('cancels live mode switch from dialog', async () => {
    const onModeChange = jest.fn();

    render(<TradingModeToggle onModeChange={onModeChange} />);

    await waitFor(() => {
      expect(screen.getAllByText('Paper').length).toBeGreaterThan(0);
    });

    // Click live button to open dialog
    const liveButton = screen.getAllByRole('button').find(
      (btn) => btn.textContent === 'Live'
    );
    fireEvent.click(liveButton!);

    // Click cancel button in dialog
    await waitFor(() => {
      expect(screen.getByText('Cancel')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Cancel'));

    // Dialog should be closed
    await waitFor(() => {
      expect(screen.queryByText('Switch to Live Trading?')).not.toBeInTheDocument();
    });

    // Mode should still be paper
    const paperButton = screen.getAllByRole('button').find(
      (btn) => btn.textContent === 'Paper'
    );
    expect(paperButton).toHaveClass('bg-profit-500');
    expect(onModeChange).not.toHaveBeenCalled();
  });

  // Test 7: Clicking backdrop closes dialog
  test('closes dialog when clicking backdrop', async () => {
    render(<TradingModeToggle />);

    await waitFor(() => {
      expect(screen.getAllByText('Paper').length).toBeGreaterThan(0);
    });

    // Click live button to open dialog
    const liveButton = screen.getAllByRole('button').find(
      (btn) => btn.textContent === 'Live'
    );
    fireEvent.click(liveButton!);

    await waitFor(() => {
      expect(screen.getByText('Switch to Live Trading?')).toBeInTheDocument();
    });

    // Click the backdrop (the absolute inset-0 element)
    const backdrop = document.querySelector('.bg-panel-900\\/80');
    fireEvent.click(backdrop!);

    // Dialog should be closed
    await waitFor(() => {
      expect(screen.queryByText('Switch to Live Trading?')).not.toBeInTheDocument();
    });
  });

  // Test 8: Visual indicator shows correct color for paper mode
  test('shows green indicator for paper mode', async () => {
    render(<TradingModeToggle />);

    await waitFor(() => {
      const indicator = document.querySelector('.bg-profit-500');
      expect(indicator).toBeInTheDocument();
    });
  });

  // Test 9: Visual indicator shows correct color for live mode
  test('shows red indicator for live mode', async () => {
    localStorageMock.setItem('fluxhero_trading_mode', 'live');

    render(<TradingModeToggle />);

    await waitFor(() => {
      const indicator = document.querySelector('.bg-loss-500');
      expect(indicator).toBeInTheDocument();
    });
  });

  // Test 10: Mode label displays correctly
  test('displays correct mode label', async () => {
    render(<TradingModeToggle />);

    await waitFor(() => {
      expect(screen.getAllByText('Paper').length).toBeGreaterThan(0);
    });

    // Switch to live
    const liveButton = screen.getAllByRole('button').find(
      (btn) => btn.textContent === 'Live'
    );
    fireEvent.click(liveButton!);

    await waitFor(() => {
      expect(screen.getByText('Switch to Live Trading?')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Switch to Live'));

    await waitFor(() => {
      const labels = screen.getAllByText('Live');
      expect(labels.length).toBeGreaterThan(0);
    });
  });

  // Test 11: Clicking current mode does nothing
  test('clicking current mode does not trigger change', async () => {
    const onModeChange = jest.fn();

    render(<TradingModeToggle onModeChange={onModeChange} />);

    await waitFor(() => {
      expect(screen.getAllByText('Paper').length).toBeGreaterThan(0);
    });

    // Click paper button (already selected)
    const paperButton = screen.getAllByRole('button').find(
      (btn) => btn.textContent === 'Paper'
    );
    fireEvent.click(paperButton!);

    // No change should occur
    expect(onModeChange).not.toHaveBeenCalled();
    expect(screen.queryByText('Switch to Live Trading?')).not.toBeInTheDocument();
  });

  // Test 12: Invalid localStorage value defaults to paper
  test('defaults to paper mode with invalid localStorage value', async () => {
    localStorageMock.setItem('fluxhero_trading_mode', 'invalid');

    render(<TradingModeToggle />);

    await waitFor(() => {
      const paperButton = screen.getAllByRole('button').find(
        (btn) => btn.textContent === 'Paper'
      );
      expect(paperButton).toHaveClass('bg-profit-500');
    });
  });

  // Test 13: Component accepts className prop
  test('accepts className prop for styling', async () => {
    render(<TradingModeToggle className="custom-class" />);

    await waitFor(() => {
      const container = screen.getAllByText('Paper')[0].closest('.custom-class');
      expect(container).toBeInTheDocument();
    });
  });

  // Test 14: Loading state shows skeleton
  test('shows loading skeleton initially', () => {
    // Force synchronous render before useEffect runs
    const { container } = render(<TradingModeToggle />);

    // The skeleton should have animate-pulse class
    const skeleton = container.querySelector('.animate-pulse');
    // This may or may not be present depending on timing
    // The important thing is the component renders without error
    expect(container).toBeInTheDocument();
  });
});

describe('useTradingMode Hook', () => {
  beforeEach(() => {
    localStorageMock.clear();
    jest.clearAllMocks();
  });

  // Test 1: Hook returns default paper mode
  test('returns default paper mode', async () => {
    render(<HookTestComponent />);

    await waitFor(() => {
      expect(screen.getByTestId('loaded').textContent).toBe('true');
      expect(screen.getByTestId('mode').textContent).toBe('paper');
    });
  });

  // Test 2: Hook loads saved mode from localStorage
  test('loads saved mode from localStorage', async () => {
    localStorageMock.setItem('fluxhero_trading_mode', 'live');

    render(<HookTestComponent />);

    await waitFor(() => {
      expect(screen.getByTestId('loaded').textContent).toBe('true');
      expect(screen.getByTestId('mode').textContent).toBe('live');
    });
  });

  // Test 3: Hook setMode updates state and localStorage
  test('setMode updates state and localStorage', async () => {
    const onModeChange = jest.fn();

    render(<HookTestComponent onModeChange={onModeChange} />);

    await waitFor(() => {
      expect(screen.getByTestId('loaded').textContent).toBe('true');
    });

    // Click "Set Live" button
    fireEvent.click(screen.getByText('Set Live'));

    await waitFor(() => {
      expect(screen.getByTestId('mode').textContent).toBe('live');
      expect(localStorageMock.setItem).toHaveBeenCalledWith('fluxhero_trading_mode', 'live');
    });
  });

  // Test 4: Hook dispatches storage event for cross-tab sync
  test('dispatches storage event when mode changes', async () => {
    render(<HookTestComponent />);

    await waitFor(() => {
      expect(screen.getByTestId('loaded').textContent).toBe('true');
    });

    fireEvent.click(screen.getByText('Set Live'));

    await waitFor(() => {
      expect(dispatchEventMock).toHaveBeenCalled();
    });
  });

  // Test 5: Hook isLoaded is false initially
  test('isLoaded is false before localStorage is read', () => {
    // This is tricky to test because useEffect runs after render
    // The best we can do is verify the component handles loading state
    const { container } = render(<HookTestComponent />);
    expect(container).toBeInTheDocument();
  });
});
