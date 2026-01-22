/**
 * Unit tests for SignalTooltip component
 *
 * Tests:
 * - Component rendering with various signal explanations
 * - JSON parsing of signal_reason field
 * - Hover behavior (show/hide tooltip)
 * - Display formatting (currency, percentages, decimals)
 * - Market context display (regime, volatility, strategy)
 * - Indicator values display
 * - Risk parameters display
 * - Validation checks display
 * - Edge cases (no data, partial data, plain text)
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { SignalTooltip, SignalExplanation } from '../SignalTooltip';

describe('SignalTooltip Component', () => {
  // Test 1: Component renders children without signal reason
  test('renders children without tooltip when no signal reason provided', () => {
    render(
      <SignalTooltip>
        <button>Trade Entry</button>
      </SignalTooltip>
    );

    expect(screen.getByText('Trade Entry')).toBeInTheDocument();
    expect(screen.queryByText('Signal Explanation')).not.toBeInTheDocument();
  });

  // Test 2: Component renders children with signal reason
  test('renders children and shows tooltip on hover', async () => {
    const signalReason = JSON.stringify({
      entry_trigger: 'Price crossed above KAMA + 0.5×ATR',
      regime: 'STRONG_TREND',
    });

    render(
      <SignalTooltip signalReason={signalReason}>
        <button>Trade Entry</button>
      </SignalTooltip>
    );

    const button = screen.getByText('Trade Entry');
    expect(button).toBeInTheDocument();

    // Tooltip should not be visible initially
    expect(screen.queryByText('Signal Explanation')).not.toBeInTheDocument();

    // Hover over the button
    fireEvent.mouseEnter(button.parentElement!);

    // Tooltip should be visible
    await waitFor(() => {
      expect(screen.getByText('Signal Explanation')).toBeInTheDocument();
      expect(screen.getByText('Price crossed above KAMA + 0.5×ATR')).toBeInTheDocument();
    });

    // Mouse leave
    fireEvent.mouseLeave(button.parentElement!);

    // Tooltip should be hidden
    await waitFor(() => {
      expect(screen.queryByText('Signal Explanation')).not.toBeInTheDocument();
    });
  });

  // Test 3: Parses JSON signal reason correctly
  test('parses JSON signal reason correctly', async () => {
    const explanation: SignalExplanation = {
      entry_trigger: 'RSI < 30 AND price at lower BB',
      regime: 'MEAN_REVERSION',
      volatility_state: 'NORMAL',
      strategy_mode: 'MEAN_REVERSION',
      atr: 2.5,
      rsi: 28.3,
      risk_amount: 100,
      position_size: 50,
    };

    render(
      <SignalTooltip signalReason={JSON.stringify(explanation)}>
        <span>Entry</span>
      </SignalTooltip>
    );

    fireEvent.mouseEnter(screen.getByText('Entry').parentElement!);

    await waitFor(() => {
      expect(screen.getByText('RSI < 30 AND price at lower BB')).toBeInTheDocument();
      expect(screen.getByText('↔️ Mean Reversion')).toBeInTheDocument();
      expect(screen.getByText('🟡 Normal')).toBeInTheDocument();
      expect(screen.getByText('Mean Reversion')).toBeInTheDocument();
    });
  });

  // Test 4: Handles plain text signal reason
  test('handles plain text signal reason (non-JSON)', async () => {
    const signalReason = 'Manual trade entry';

    render(
      <SignalTooltip signalReason={signalReason}>
        <span>Entry</span>
      </SignalTooltip>
    );

    fireEvent.mouseEnter(screen.getByText('Entry').parentElement!);

    await waitFor(() => {
      expect(screen.getByText('Manual trade entry')).toBeInTheDocument();
    });
  });

  // Test 5: Displays market context correctly
  test('displays market context with all states', async () => {
    const explanation: SignalExplanation = {
      regime: '2', // STRONG_TREND
      volatility_state: '2', // HIGH
      strategy_mode: '2', // TREND_FOLLOWING
    };

    render(
      <SignalTooltip signalReason={JSON.stringify(explanation)}>
        <span>Entry</span>
      </SignalTooltip>
    );

    fireEvent.mouseEnter(screen.getByText('Entry').parentElement!);

    await waitFor(() => {
      expect(screen.getByText('📈 Strong Trend')).toBeInTheDocument();
      expect(screen.getByText('🔴 High')).toBeInTheDocument();
      expect(screen.getByText('Trend Following')).toBeInTheDocument();
    });
  });

  // Test 6: Displays indicator values correctly
  test('displays indicator values with correct formatting', async () => {
    const explanation: SignalExplanation = {
      kama: 150.25,
      atr: 3.456,
      rsi: 65.7,
      adx: 28.9,
      r_squared: 0.856,
    };

    render(
      <SignalTooltip signalReason={JSON.stringify(explanation)}>
        <span>Entry</span>
      </SignalTooltip>
    );

    fireEvent.mouseEnter(screen.getByText('Entry').parentElement!);

    await waitFor(() => {
      expect(screen.getByText('150.25')).toBeInTheDocument(); // KAMA
      expect(screen.getByText('3.46')).toBeInTheDocument(); // ATR (2 decimals)
      expect(screen.getByText('65.7')).toBeInTheDocument(); // RSI (1 decimal)
      expect(screen.getByText('28.9')).toBeInTheDocument(); // ADX (1 decimal)
      expect(screen.getByText('0.856')).toBeInTheDocument(); // R² (3 decimals)
    });
  });

  // Test 7: Displays risk parameters correctly
  test('displays risk parameters with correct formatting', async () => {
    const explanation: SignalExplanation = {
      position_size: 75,
      risk_amount: 150.50,
      risk_percent: 0.01, // 1%
      stop_loss: 145.75,
    };

    render(
      <SignalTooltip signalReason={JSON.stringify(explanation)}>
        <span>Entry</span>
      </SignalTooltip>
    );

    fireEvent.mouseEnter(screen.getByText('Entry').parentElement!);

    await waitFor(() => {
      expect(screen.getByText('75 shares')).toBeInTheDocument();
      expect(screen.getByText('$150.50')).toBeInTheDocument(); // risk_amount
      expect(screen.getByText('1.00%')).toBeInTheDocument(); // risk_percent
      expect(screen.getByText('$145.75')).toBeInTheDocument(); // stop_loss
    });
  });

  // Test 8: Displays validation checks correctly
  test('displays validation checks with pass/fail indicators', async () => {
    const explanation: SignalExplanation = {
      noise_filtered: true,
      volume_valid: true,
      spread_valid: false,
    };

    render(
      <SignalTooltip signalReason={JSON.stringify(explanation)}>
        <span>Entry</span>
      </SignalTooltip>
    );

    fireEvent.mouseEnter(screen.getByText('Entry').parentElement!);

    await waitFor(() => {
      expect(screen.getByText('✅ Passed')).toBeInTheDocument(); // noise_filtered
      expect(screen.getByText('✅ Valid')).toBeInTheDocument(); // volume_valid
      expect(screen.getByText('❌ Invalid')).toBeInTheDocument(); // spread_valid
    });
  });

  // Test 9: Handles partial data (some fields missing)
  test('handles partial signal explanation data', async () => {
    const explanation: SignalExplanation = {
      entry_trigger: 'Breakout signal',
      atr: 2.0,
      // Missing: regime, volatility_state, other indicators, risk params
    };

    render(
      <SignalTooltip signalReason={JSON.stringify(explanation)}>
        <span>Entry</span>
      </SignalTooltip>
    );

    fireEvent.mouseEnter(screen.getByText('Entry').parentElement!);

    await waitFor(() => {
      expect(screen.getByText('Breakout signal')).toBeInTheDocument();
      expect(screen.getByText('2.00')).toBeInTheDocument();
      // Should not display sections for missing data
      expect(screen.queryByText('Market Context')).not.toBeInTheDocument();
      expect(screen.queryByText('Risk Management')).not.toBeInTheDocument();
    });
  });

  // Test 10: Handles empty JSON object
  test('handles empty JSON object', async () => {
    render(
      <SignalTooltip signalReason="{}">
        <span>Entry</span>
      </SignalTooltip>
    );

    fireEvent.mouseEnter(screen.getByText('Entry').parentElement!);

    await waitFor(() => {
      expect(screen.getByText('Signal Explanation')).toBeInTheDocument();
      // Should display header but no sections
    });
  });

  // Test 11: Tooltip placement variations
  test('supports different tooltip placements', () => {
    const signalReason = JSON.stringify({ entry_trigger: 'Test' });

    const { rerender } = render(
      <SignalTooltip signalReason={signalReason} placement="top">
        <span>Entry</span>
      </SignalTooltip>
    );

    expect(screen.getByText('Entry').parentElement).toHaveClass('placement-top');

    rerender(
      <SignalTooltip signalReason={signalReason} placement="bottom">
        <span>Entry</span>
      </SignalTooltip>
    );

    expect(screen.getByText('Entry').parentElement).toHaveClass('placement-bottom');

    rerender(
      <SignalTooltip signalReason={signalReason} placement="left">
        <span>Entry</span>
      </SignalTooltip>
    );

    expect(screen.getByText('Entry').parentElement).toHaveClass('placement-left');

    rerender(
      <SignalTooltip signalReason={signalReason} placement="right">
        <span>Entry</span>
      </SignalTooltip>
    );

    expect(screen.getByText('Entry').parentElement).toHaveClass('placement-right');
  });

  // Test 12: Comprehensive signal explanation (all fields)
  test('displays comprehensive signal explanation with all fields', async () => {
    const explanation: SignalExplanation = {
      symbol: 'SPY',
      signal_type: 'LONG',
      price: 450.25,
      timestamp: 1704067200,
      strategy_mode: 'TREND_FOLLOWING',
      regime: 'STRONG_TREND',
      volatility_state: 'NORMAL',
      atr: 3.5,
      kama: 448.0,
      rsi: 55.0,
      adx: 30.0,
      r_squared: 0.75,
      risk_amount: 200.0,
      risk_percent: 0.01,
      stop_loss: 440.0,
      position_size: 100,
      entry_trigger: 'Price crossed above KAMA + 0.5×ATR (450.25 > 449.75)',
      noise_filtered: true,
      volume_valid: true,
      spread_valid: true,
    };

    render(
      <SignalTooltip signalReason={JSON.stringify(explanation)}>
        <span>SPY Entry</span>
      </SignalTooltip>
    );

    fireEvent.mouseEnter(screen.getByText('SPY Entry').parentElement!);

    await waitFor(() => {
      // Check all sections are present
      expect(screen.getByText('Signal Explanation')).toBeInTheDocument();
      expect(screen.getByText('Entry Trigger')).toBeInTheDocument();
      expect(screen.getByText('Market Context')).toBeInTheDocument();
      expect(screen.getByText('Indicator Values')).toBeInTheDocument();
      expect(screen.getByText('Risk Management')).toBeInTheDocument();
      expect(screen.getByText('Validation Checks')).toBeInTheDocument();
    });
  });

  // Test 13: Currency formatting edge cases
  test('handles currency formatting edge cases', async () => {
    const explanation: SignalExplanation = {
      risk_amount: 0, // Zero values should not be displayed (falsy check)
      stop_loss: 1000.999, // Should round to 1001.00
    };

    render(
      <SignalTooltip signalReason={JSON.stringify(explanation)}>
        <span>Entry</span>
      </SignalTooltip>
    );

    fireEvent.mouseEnter(screen.getByText('Entry').parentElement!);

    await waitFor(() => {
      // risk_amount of 0 should not be displayed (falsy check in component)
      expect(screen.queryByText('Risk Amount:')).not.toBeInTheDocument();
      // stop_loss should be formatted and rounded correctly
      expect(screen.getByText('$1,001.00')).toBeInTheDocument();
    });
  });

  // Test 14: Percentage formatting edge cases
  test('handles percentage formatting edge cases', async () => {
    const explanation: SignalExplanation = {
      risk_percent: 0.0001, // 0.01%
    };

    render(
      <SignalTooltip signalReason={JSON.stringify(explanation)}>
        <span>Entry</span>
      </SignalTooltip>
    );

    fireEvent.mouseEnter(screen.getByText('Entry').parentElement!);

    await waitFor(() => {
      expect(screen.getByText('0.01%')).toBeInTheDocument();
    });
  });

  // Test 15: Decimal formatting with various precision
  test('handles decimal formatting with correct precision', async () => {
    const explanation: SignalExplanation = {
      kama: 100.123456, // 2 decimals
      rsi: 50.987654, // 1 decimal
      r_squared: 0.123456789, // 3 decimals
    };

    render(
      <SignalTooltip signalReason={JSON.stringify(explanation)}>
        <span>Entry</span>
      </SignalTooltip>
    );

    fireEvent.mouseEnter(screen.getByText('Entry').parentElement!);

    await waitFor(() => {
      expect(screen.getByText('100.12')).toBeInTheDocument();
      expect(screen.getByText('51.0')).toBeInTheDocument();
      expect(screen.getByText('0.123')).toBeInTheDocument();
    });
  });

  // Test 16: Regime display with numeric codes
  test('converts numeric regime codes to display names', async () => {
    const test0 = { regime: '0' }; // Mean Reversion
    const test1 = { regime: '1' }; // Neutral
    const test2 = { regime: '2' }; // Strong Trend

    const { rerender } = render(
      <SignalTooltip signalReason={JSON.stringify(test0)}>
        <span>Entry</span>
      </SignalTooltip>
    );

    fireEvent.mouseEnter(screen.getByText('Entry').parentElement!);
    await waitFor(() => expect(screen.getByText('↔️ Mean Reversion')).toBeInTheDocument());

    fireEvent.mouseLeave(screen.getByText('Entry').parentElement!);

    rerender(
      <SignalTooltip signalReason={JSON.stringify(test1)}>
        <span>Entry</span>
      </SignalTooltip>
    );

    fireEvent.mouseEnter(screen.getByText('Entry').parentElement!);
    await waitFor(() => expect(screen.getByText('➖ Neutral')).toBeInTheDocument());

    fireEvent.mouseLeave(screen.getByText('Entry').parentElement!);

    rerender(
      <SignalTooltip signalReason={JSON.stringify(test2)}>
        <span>Entry</span>
      </SignalTooltip>
    );

    fireEvent.mouseEnter(screen.getByText('Entry').parentElement!);
    await waitFor(() => expect(screen.getByText('📈 Strong Trend')).toBeInTheDocument());
  });

  // Test 17: Volatility display with numeric codes
  test('converts numeric volatility codes to display names', async () => {
    const test0 = { volatility_state: '0' }; // Low
    const test1 = { volatility_state: '1' }; // Normal
    const test2 = { volatility_state: '2' }; // High

    const { rerender } = render(
      <SignalTooltip signalReason={JSON.stringify(test0)}>
        <span>Entry</span>
      </SignalTooltip>
    );

    fireEvent.mouseEnter(screen.getByText('Entry').parentElement!);
    await waitFor(() => expect(screen.getByText('🟢 Low')).toBeInTheDocument());

    fireEvent.mouseLeave(screen.getByText('Entry').parentElement!);

    rerender(
      <SignalTooltip signalReason={JSON.stringify(test1)}>
        <span>Entry</span>
      </SignalTooltip>
    );

    fireEvent.mouseEnter(screen.getByText('Entry').parentElement!);
    await waitFor(() => expect(screen.getByText('🟡 Normal')).toBeInTheDocument());

    fireEvent.mouseLeave(screen.getByText('Entry').parentElement!);

    rerender(
      <SignalTooltip signalReason={JSON.stringify(test2)}>
        <span>Entry</span>
      </SignalTooltip>
    );

    fireEvent.mouseEnter(screen.getByText('Entry').parentElement!);
    await waitFor(() => expect(screen.getByText('🔴 High')).toBeInTheDocument());
  });

  // Test 18: Invalid/unknown enum values
  test('handles unknown regime/volatility/strategy values', async () => {
    const explanation: SignalExplanation = {
      regime: 'CUSTOM_REGIME',
      volatility_state: 'EXTREME',
      strategy_mode: 'UNKNOWN_MODE',
    };

    render(
      <SignalTooltip signalReason={JSON.stringify(explanation)}>
        <span>Entry</span>
      </SignalTooltip>
    );

    fireEvent.mouseEnter(screen.getByText('Entry').parentElement!);

    await waitFor(() => {
      expect(screen.getByText('CUSTOM_REGIME')).toBeInTheDocument();
      expect(screen.getByText('EXTREME')).toBeInTheDocument();
      expect(screen.getByText('UNKNOWN_MODE')).toBeInTheDocument();
    });
  });

  // Test 19: Null/undefined values in explanation
  test('handles null and undefined values gracefully', async () => {
    const explanation: SignalExplanation = {
      entry_trigger: 'Test entry',
      atr: undefined,
      rsi: null as any,
      risk_amount: undefined,
    };

    render(
      <SignalTooltip signalReason={JSON.stringify(explanation)}>
        <span>Entry</span>
      </SignalTooltip>
    );

    fireEvent.mouseEnter(screen.getByText('Entry').parentElement!);

    await waitFor(() => {
      expect(screen.getByText('Test entry')).toBeInTheDocument();
      // Sections with only N/A values should not be displayed
    });
  });

  // Test 20: Integration test - Trade history page usage
  test('integrates with trade history table row', async () => {
    const tradeSignalReason = JSON.stringify({
      entry_trigger: 'KAMA breakout above resistance',
      regime: 'STRONG_TREND',
      volatility_state: 'HIGH',
      strategy_mode: 'TREND_FOLLOWING',
      atr: 4.2,
      kama: 455.0,
      rsi: 60.0,
      adx: 32.0,
      risk_amount: 250.0,
      risk_percent: 0.01,
      stop_loss: 445.0,
      position_size: 125,
      noise_filtered: true,
      volume_valid: true,
      spread_valid: true,
    });

    render(
      <table>
        <tbody>
          <tr>
            <td>SPY</td>
            <td>LONG</td>
            <td>455.00</td>
            <td>
              <SignalTooltip signalReason={tradeSignalReason}>
                <span className="info-icon">ℹ️</span>
              </SignalTooltip>
            </td>
          </tr>
        </tbody>
      </table>
    );

    const icon = screen.getByText('ℹ️');
    fireEvent.mouseEnter(icon.parentElement!);

    await waitFor(() => {
      expect(screen.getByText('KAMA breakout above resistance')).toBeInTheDocument();
      expect(screen.getByText('📈 Strong Trend')).toBeInTheDocument();
      expect(screen.getByText('🔴 High')).toBeInTheDocument();
      expect(screen.getByText('Trend Following')).toBeInTheDocument();
      expect(screen.getByText('125 shares')).toBeInTheDocument();
    });
  });
});
