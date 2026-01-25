/**
 * SignalExplanationPanel - Displays trade signal reasoning
 *
 * Shows:
 * - Entry trigger and reasoning
 * - Indicator values at entry
 * - Formatted explanation text
 *
 * Reusable in modals, pages, or embedded anywhere
 */

import React from 'react';
import { cn } from '../../lib/utils';
import { Badge } from '../ui';

export interface SignalExplanation {
  entry_trigger?: string;
  strategy?: string;
  regime?: string;
  formatted_reason?: string;
  raw_reason?: string;
  indicators?: {
    rsi?: number;
    adx?: number;
    kama_slope?: number;
    atr?: number;
    r_squared?: number;
    price?: number;
    kama?: number;
  };
  risk_params?: {
    stop_loss?: number;
    take_profit?: number;
    position_size?: number;
    risk_reward?: number;
  };
}

export interface SignalExplanationPanelProps {
  signalExplanation: SignalExplanation;
  className?: string;
  compact?: boolean;
}

function IndicatorValue({
  label,
  value,
  unit = '',
  color,
}: {
  label: string;
  value: number | undefined;
  unit?: string;
  color?: 'profit' | 'loss' | 'warning' | 'neutral';
}) {
  if (value === undefined) return null;

  const colorClass = {
    profit: 'text-profit-500',
    loss: 'text-loss-500',
    warning: 'text-warning-500',
    neutral: 'text-text-700',
  }[color || 'neutral'];

  return (
    <div className="flex flex-col">
      <span className="text-xs text-text-400">{label}</span>
      <span className={cn('font-mono tabular-nums font-medium', colorClass)}>
        {value.toFixed(2)}{unit}
      </span>
    </div>
  );
}

function getRsiColor(rsi: number): 'profit' | 'loss' | 'warning' | 'neutral' {
  if (rsi >= 70) return 'loss'; // Overbought
  if (rsi <= 30) return 'profit'; // Oversold
  return 'neutral';
}

function getAdxColor(adx: number): 'profit' | 'loss' | 'warning' | 'neutral' {
  if (adx >= 25) return 'profit'; // Strong trend
  return 'warning'; // Weak trend
}

export function SignalExplanationPanel({
  signalExplanation,
  className,
  compact = false,
}: SignalExplanationPanelProps) {
  const {
    entry_trigger,
    strategy,
    regime,
    formatted_reason,
    raw_reason,
    indicators,
    risk_params,
  } = signalExplanation;

  const displayReason = formatted_reason || raw_reason || 'No signal explanation available';

  if (compact) {
    return (
      <div className={cn('bg-panel-600 rounded-lg p-4', className)}>
        <div className="flex items-center gap-2 mb-2">
          {strategy && (
            <Badge variant={strategy === 'TREND' ? 'info' : strategy === 'MEAN_REVERSION' ? 'warning' : 'neutral'}>
              {strategy}
            </Badge>
          )}
          {regime && (
            <Badge variant={regime === 'TRENDING' ? 'success' : regime === 'MEAN_REVERTING' ? 'warning' : 'neutral'}>
              {regime}
            </Badge>
          )}
        </div>
        <p className="text-text-600 text-sm">{displayReason}</p>
      </div>
    );
  }

  return (
    <div className={cn('bg-panel-600 rounded-xl p-5', className)}>
      {/* Header with badges */}
      <div className="flex items-center gap-3 mb-4">
        {strategy && (
          <Badge
            variant={
              strategy === 'TREND' ? 'info' :
              strategy === 'MEAN_REVERSION' ? 'warning' :
              'neutral'
            }
          >
            {strategy}
          </Badge>
        )}
        {regime && (
          <Badge
            variant={
              regime === 'TRENDING' ? 'success' :
              regime === 'MEAN_REVERTING' ? 'warning' :
              'neutral'
            }
          >
            {regime}
          </Badge>
        )}
      </div>

      {/* Entry Trigger */}
      {entry_trigger && (
        <div className="mb-4">
          <h4 className="text-sm font-medium text-text-400 mb-1">Entry Trigger</h4>
          <p className="text-text-800 font-medium">{entry_trigger}</p>
        </div>
      )}

      {/* Signal Reason */}
      <div className="mb-5">
        <h4 className="text-sm font-medium text-text-400 mb-1">Signal Explanation</h4>
        <p className="text-text-700 text-sm leading-relaxed whitespace-pre-wrap">
          {displayReason}
        </p>
      </div>

      {/* Technical Indicators */}
      {indicators && Object.keys(indicators).length > 0 && (
        <div className="mb-5">
          <h4 className="text-sm font-medium text-text-400 mb-3">Indicators at Entry</h4>
          <div className="grid grid-cols-3 sm:grid-cols-5 gap-4">
            {indicators.rsi !== undefined && (
              <IndicatorValue
                label="RSI (14)"
                value={indicators.rsi}
                color={getRsiColor(indicators.rsi)}
              />
            )}
            {indicators.adx !== undefined && (
              <IndicatorValue
                label="ADX (14)"
                value={indicators.adx}
                color={getAdxColor(indicators.adx)}
              />
            )}
            {indicators.kama_slope !== undefined && (
              <IndicatorValue
                label="KAMA Slope"
                value={indicators.kama_slope}
                unit="%"
                color={indicators.kama_slope > 0 ? 'profit' : indicators.kama_slope < 0 ? 'loss' : 'neutral'}
              />
            )}
            {indicators.atr !== undefined && (
              <IndicatorValue
                label="ATR (14)"
                value={indicators.atr}
              />
            )}
            {indicators.r_squared !== undefined && (
              <IndicatorValue
                label="R-squared"
                value={indicators.r_squared}
                color={indicators.r_squared >= 0.8 ? 'profit' : indicators.r_squared >= 0.5 ? 'neutral' : 'loss'}
              />
            )}
            {indicators.price !== undefined && (
              <IndicatorValue
                label="Price"
                value={indicators.price}
              />
            )}
            {indicators.kama !== undefined && (
              <IndicatorValue
                label="KAMA"
                value={indicators.kama}
              />
            )}
          </div>
        </div>
      )}

      {/* Risk Parameters */}
      {risk_params && Object.keys(risk_params).length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-text-400 mb-3">Risk Parameters</h4>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            {risk_params.stop_loss !== undefined && (
              <div className="flex flex-col">
                <span className="text-xs text-text-400">Stop Loss</span>
                <span className="font-mono tabular-nums font-medium text-loss-500">
                  ${risk_params.stop_loss.toFixed(2)}
                </span>
              </div>
            )}
            {risk_params.take_profit !== undefined && (
              <div className="flex flex-col">
                <span className="text-xs text-text-400">Take Profit</span>
                <span className="font-mono tabular-nums font-medium text-profit-500">
                  ${risk_params.take_profit.toFixed(2)}
                </span>
              </div>
            )}
            {risk_params.position_size !== undefined && (
              <div className="flex flex-col">
                <span className="text-xs text-text-400">Position Size</span>
                <span className="font-mono tabular-nums font-medium text-text-700">
                  {risk_params.position_size} shares
                </span>
              </div>
            )}
            {risk_params.risk_reward !== undefined && (
              <div className="flex flex-col">
                <span className="text-xs text-text-400">Risk/Reward</span>
                <span className={cn(
                  'font-mono tabular-nums font-medium',
                  risk_params.risk_reward >= 2 ? 'text-profit-500' :
                  risk_params.risk_reward >= 1 ? 'text-warning-500' :
                  'text-loss-500'
                )}>
                  1:{risk_params.risk_reward.toFixed(1)}
                </span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default SignalExplanationPanel;
