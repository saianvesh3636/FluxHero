'use client';

import React, { useState } from 'react';

/**
 * Signal Explanation Tooltip Component
 *
 * Displays comprehensive signal explanations on hover, including:
 * - Market regime and volatility state
 * - Indicator values (ATR, KAMA, RSI, ADX, R²)
 * - Risk parameters (risk amount, position size, stop loss)
 * - Entry trigger and signal reasoning
 * - Noise filter status
 *
 * Reference: FLUXHERO_REQUIREMENTS.md Feature 12.3 - Signal Explainer
 */

export interface SignalExplanation {
  // Signal identification
  symbol?: string;
  signal_type?: string;
  price?: number;
  timestamp?: number;

  // Market context
  strategy_mode?: string;
  regime?: string;
  volatility_state?: string;

  // Indicator values
  atr?: number;
  kama?: number;
  rsi?: number;
  adx?: number;
  r_squared?: number;

  // Risk parameters
  risk_amount?: number;
  risk_percent?: number;
  stop_loss?: number;
  position_size?: number;

  // Signal details
  entry_trigger?: string;
  noise_filtered?: boolean;
  volume_valid?: boolean;
  spread_valid?: boolean;
}

export interface SignalTooltipProps {
  signalReason?: string; // Raw signal_reason from backend (may be JSON string)
  children: React.ReactNode;
  placement?: 'top' | 'bottom' | 'left' | 'right';
}

/**
 * Parse signal reason string into structured data
 * Handles both JSON and plain text formats
 */
const parseSignalReason = (signalReason?: string): SignalExplanation | null => {
  if (!signalReason) return null;

  try {
    // Try to parse as JSON first
    const parsed = JSON.parse(signalReason);
    return parsed as SignalExplanation;
  } catch {
    // If not JSON, return a basic object with the raw string
    return {
      entry_trigger: signalReason,
    };
  }
};

/**
 * Format currency values
 */
const formatCurrency = (value?: number): string => {
  if (value === undefined || value === null) return 'N/A';
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
};

/**
 * Format percentage values
 */
const formatPercent = (value?: number): string => {
  if (value === undefined || value === null) return 'N/A';
  return `${(value * 100).toFixed(2)}%`;
};

/**
 * Format decimal values
 */
const formatDecimal = (value?: number, decimals: number = 2): string => {
  if (value === undefined || value === null) return 'N/A';
  return value.toFixed(decimals);
};

/**
 * Get display name for regime
 */
const getRegimeDisplay = (regime?: string): string => {
  if (!regime) return 'Unknown';
  const regimeMap: Record<string, string> = {
    STRONG_TREND: '📈 Strong Trend',
    MEAN_REVERSION: '↔️ Mean Reversion',
    NEUTRAL: '➖ Neutral',
    '0': '↔️ Mean Reversion',
    '1': '➖ Neutral',
    '2': '📈 Strong Trend',
  };
  return regimeMap[regime] || regime;
};

/**
 * Get display name for volatility state
 */
const getVolatilityDisplay = (state?: string): string => {
  if (!state) return 'Unknown';
  const stateMap: Record<string, string> = {
    LOW: '🟢 Low',
    NORMAL: '🟡 Normal',
    HIGH: '🔴 High',
    '0': '🟢 Low',
    '1': '🟡 Normal',
    '2': '🔴 High',
  };
  return stateMap[state] || state;
};

/**
 * Get display name for strategy mode
 */
const getStrategyDisplay = (mode?: string): string => {
  if (!mode) return 'Unknown';
  const modeMap: Record<string, string> = {
    TREND_FOLLOWING: 'Trend Following',
    MEAN_REVERSION: 'Mean Reversion',
    NEUTRAL: 'Neutral (Blended)',
    '1': 'Mean Reversion',
    '2': 'Trend Following',
    '3': 'Neutral (Blended)',
  };
  return modeMap[mode] || mode;
};

/**
 * SignalTooltip Component
 */
export const SignalTooltip: React.FC<SignalTooltipProps> = ({
  signalReason,
  children,
  placement = 'top',
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const explanation = parseSignalReason(signalReason);

  if (!explanation) {
    return <>{children}</>;
  }

  const tooltipContent = (
    <div className="signal-tooltip-content">
      {/* Header */}
      <div className="tooltip-header">
        <h4>Signal Explanation</h4>
      </div>

      {/* Entry Trigger */}
      {explanation.entry_trigger && (
        <div className="tooltip-section">
          <p className="section-title">Entry Trigger</p>
          <p className="section-value">{explanation.entry_trigger}</p>
        </div>
      )}

      {/* Market Context */}
      {(explanation.regime || explanation.volatility_state || explanation.strategy_mode) && (
        <div className="tooltip-section">
          <p className="section-title">Market Context</p>
          {explanation.regime && (
            <p className="detail-row">
              <span className="detail-label">Regime:</span>
              <span className="detail-value">{getRegimeDisplay(explanation.regime)}</span>
            </p>
          )}
          {explanation.volatility_state && (
            <p className="detail-row">
              <span className="detail-label">Volatility:</span>
              <span className="detail-value">
                {getVolatilityDisplay(explanation.volatility_state)}
              </span>
            </p>
          )}
          {explanation.strategy_mode && (
            <p className="detail-row">
              <span className="detail-label">Strategy:</span>
              <span className="detail-value">
                {getStrategyDisplay(explanation.strategy_mode)}
              </span>
            </p>
          )}
        </div>
      )}

      {/* Indicator Values */}
      {(explanation.atr ||
        explanation.kama ||
        explanation.rsi ||
        explanation.adx ||
        explanation.r_squared) && (
        <div className="tooltip-section">
          <p className="section-title">Indicator Values</p>
          {explanation.kama && (
            <p className="detail-row">
              <span className="detail-label">KAMA:</span>
              <span className="detail-value">{formatDecimal(explanation.kama)}</span>
            </p>
          )}
          {explanation.atr && (
            <p className="detail-row">
              <span className="detail-label">ATR:</span>
              <span className="detail-value">{formatDecimal(explanation.atr)}</span>
            </p>
          )}
          {explanation.rsi && (
            <p className="detail-row">
              <span className="detail-label">RSI:</span>
              <span className="detail-value">{formatDecimal(explanation.rsi, 1)}</span>
            </p>
          )}
          {explanation.adx && (
            <p className="detail-row">
              <span className="detail-label">ADX:</span>
              <span className="detail-value">{formatDecimal(explanation.adx, 1)}</span>
            </p>
          )}
          {explanation.r_squared && (
            <p className="detail-row">
              <span className="detail-label">R²:</span>
              <span className="detail-value">{formatDecimal(explanation.r_squared, 3)}</span>
            </p>
          )}
        </div>
      )}

      {/* Risk Parameters */}
      {(explanation.risk_amount ||
        explanation.risk_percent ||
        explanation.stop_loss ||
        explanation.position_size) && (
        <div className="tooltip-section">
          <p className="section-title">Risk Management</p>
          {explanation.position_size && (
            <p className="detail-row">
              <span className="detail-label">Position Size:</span>
              <span className="detail-value">{explanation.position_size} shares</span>
            </p>
          )}
          {explanation.risk_amount && (
            <p className="detail-row">
              <span className="detail-label">Risk Amount:</span>
              <span className="detail-value">{formatCurrency(explanation.risk_amount)}</span>
            </p>
          )}
          {explanation.risk_percent && (
            <p className="detail-row">
              <span className="detail-label">Risk %:</span>
              <span className="detail-value">{formatPercent(explanation.risk_percent)}</span>
            </p>
          )}
          {explanation.stop_loss && (
            <p className="detail-row">
              <span className="detail-label">Stop Loss:</span>
              <span className="detail-value">{formatCurrency(explanation.stop_loss)}</span>
            </p>
          )}
        </div>
      )}

      {/* Validation Status */}
      {(explanation.noise_filtered !== undefined ||
        explanation.volume_valid !== undefined ||
        explanation.spread_valid !== undefined) && (
        <div className="tooltip-section">
          <p className="section-title">Validation Checks</p>
          {explanation.noise_filtered !== undefined && (
            <p className="detail-row">
              <span className="detail-label">Noise Filter:</span>
              <span className="detail-value">
                {explanation.noise_filtered ? '✅ Passed' : '❌ Failed'}
              </span>
            </p>
          )}
          {explanation.volume_valid !== undefined && (
            <p className="detail-row">
              <span className="detail-label">Volume:</span>
              <span className="detail-value">
                {explanation.volume_valid ? '✅ Valid' : '❌ Invalid'}
              </span>
            </p>
          )}
          {explanation.spread_valid !== undefined && (
            <p className="detail-row">
              <span className="detail-label">Spread:</span>
              <span className="detail-value">
                {explanation.spread_valid ? '✅ Valid' : '❌ Invalid'}
              </span>
            </p>
          )}
        </div>
      )}
    </div>
  );

  return (
    <div
      className={`signal-tooltip-container placement-${placement}`}
      onMouseEnter={() => setIsVisible(true)}
      onMouseLeave={() => setIsVisible(false)}
    >
      {children}
      {isVisible && <div className={`signal-tooltip-popup placement-${placement}`}>{tooltipContent}</div>}

      <style jsx>{`
        .signal-tooltip-container {
          position: relative;
          display: inline-block;
        }

        .signal-tooltip-popup {
          position: absolute;
          z-index: 1000;
          min-width: 320px;
          max-width: 400px;
          background-color: #2d3748;
          color: #e2e8f0;
          border-radius: 8px;
          box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
          padding: 0;
          font-size: 0.9rem;
          line-height: 1.5;
        }

        /* Placement styles */
        .signal-tooltip-popup.placement-top {
          bottom: calc(100% + 12px);
          left: 50%;
          transform: translateX(-50%);
        }

        .signal-tooltip-popup.placement-bottom {
          top: calc(100% + 12px);
          left: 50%;
          transform: translateX(-50%);
        }

        .signal-tooltip-popup.placement-left {
          right: calc(100% + 12px);
          top: 50%;
          transform: translateY(-50%);
        }

        .signal-tooltip-popup.placement-right {
          left: calc(100% + 12px);
          top: 50%;
          transform: translateY(-50%);
        }

        /* Arrow styles */
        .signal-tooltip-popup.placement-top::after {
          content: '';
          position: absolute;
          top: 100%;
          left: 50%;
          transform: translateX(-50%);
          border: 8px solid transparent;
          border-top-color: #2d3748;
        }

        .signal-tooltip-popup.placement-bottom::after {
          content: '';
          position: absolute;
          bottom: 100%;
          left: 50%;
          transform: translateX(-50%);
          border: 8px solid transparent;
          border-bottom-color: #2d3748;
        }

        .signal-tooltip-popup.placement-left::after {
          content: '';
          position: absolute;
          left: 100%;
          top: 50%;
          transform: translateY(-50%);
          border: 8px solid transparent;
          border-left-color: #2d3748;
        }

        .signal-tooltip-popup.placement-right::after {
          content: '';
          position: absolute;
          right: 100%;
          top: 50%;
          transform: translateY(-50%);
          border: 8px solid transparent;
          border-right-color: #2d3748;
        }

        .signal-tooltip-content {
          padding: 1rem;
        }

        .tooltip-header {
          margin-bottom: 1rem;
          padding-bottom: 0.75rem;
          border-bottom: 1px solid #4a5568;
        }

        .tooltip-header h4 {
          margin: 0;
          font-size: 1.1rem;
          font-weight: 600;
          color: #f7fafc;
        }

        .tooltip-section {
          margin-bottom: 1rem;
          padding-bottom: 0.75rem;
          border-bottom: 1px solid #4a5568;
        }

        .tooltip-section:last-child {
          margin-bottom: 0;
          border-bottom: none;
          padding-bottom: 0;
        }

        .section-title {
          margin: 0 0 0.5rem 0;
          font-weight: 600;
          font-size: 0.85rem;
          text-transform: uppercase;
          letter-spacing: 0.05em;
          color: #a0aec0;
        }

        .section-value {
          margin: 0;
          color: #e2e8f0;
          line-height: 1.6;
        }

        .detail-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin: 0.4rem 0;
          gap: 1rem;
        }

        .detail-label {
          color: #cbd5e0;
          font-size: 0.9rem;
          flex-shrink: 0;
        }

        .detail-value {
          color: #f7fafc;
          font-weight: 500;
          text-align: right;
          font-family: 'Courier New', monospace;
        }

        @media (max-width: 768px) {
          .signal-tooltip-popup {
            min-width: 280px;
            max-width: 320px;
            font-size: 0.85rem;
          }

          .tooltip-header h4 {
            font-size: 1rem;
          }
        }
      `}</style>
    </div>
  );
};

export default SignalTooltip;
