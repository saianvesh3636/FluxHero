/**
 * TradeReasonCard - Signal information card with technical indicators
 *
 * Displays:
 * - Signal reason text
 * - Strategy and regime badges
 * - Technical indicators (RSI, ADX, KAMA slope, ATR, R-squared)
 * - Risk parameters (position size, stop/target, R:R ratio)
 * - Validation checks with checkmark/X icons
 */

import React from 'react';
import { cn, formatCurrency, formatPercent } from '../../lib/utils';
import { Badge } from '../ui';

export interface TechnicalIndicators {
  rsi?: number;
  adx?: number;
  kamaSlope?: number;
  atr?: number;
  rSquared?: number;
}

export interface RiskParameters {
  positionSize: number;
  positionPct?: number;
  stopLoss: number;
  takeProfit?: number;
  riskReward?: number;
}

export interface ValidationCheck {
  label: string;
  passed: boolean;
}

export interface TradeReasonCardProps {
  signalReason: string;
  strategy: string;
  regime: string;
  indicators?: TechnicalIndicators;
  riskParams?: RiskParameters;
  validationChecks?: ValidationCheck[];
  className?: string;
}

function IndicatorItem({
  label,
  value,
  unit = '',
  color,
}: {
  label: string;
  value: number | string | undefined;
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

  const displayValue = typeof value === 'number' ? value.toFixed(2) : value;

  return (
    <div className="flex flex-col">
      <span className="text-xs text-text-400">{label}</span>
      <span className={cn('font-mono tabular-nums font-medium', colorClass)}>
        {displayValue}{unit}
      </span>
    </div>
  );
}

function ValidationItem({ label, passed }: ValidationCheck) {
  return (
    <div className="flex items-center gap-2">
      <span
        className={cn(
          'w-5 h-5 rounded-full flex items-center justify-center text-xs',
          passed ? 'bg-profit-500/20 text-profit-500' : 'bg-loss-500/20 text-loss-500'
        )}
      >
        {passed ? '\u2713' : '\u2717'}
      </span>
      <span className={cn('text-sm', passed ? 'text-text-700' : 'text-text-400')}>
        {label}
      </span>
    </div>
  );
}

export function TradeReasonCard({
  signalReason,
  strategy,
  regime,
  indicators,
  riskParams,
  validationChecks,
  className,
}: TradeReasonCardProps) {
  // Determine indicator colors
  const getRsiColor = (rsi: number): 'profit' | 'loss' | 'warning' | 'neutral' => {
    if (rsi >= 70) return 'loss'; // Overbought
    if (rsi <= 30) return 'profit'; // Oversold
    return 'neutral';
  };

  const getAdxColor = (adx: number): 'profit' | 'loss' | 'warning' | 'neutral' => {
    if (adx >= 25) return 'profit'; // Strong trend
    return 'warning'; // Weak trend
  };

  return (
    <div className={cn('bg-panel-600 rounded-xl p-5', className)}>
      {/* Header with badges */}
      <div className="flex items-center gap-3 mb-4">
        <Badge
          variant={
            strategy === 'TREND' ? 'info' :
            strategy === 'MEAN_REVERSION' ? 'warning' :
            'neutral'
          }
        >
          {strategy}
        </Badge>
        <Badge
          variant={
            regime === 'TRENDING' ? 'success' :
            regime === 'MEAN_REVERTING' ? 'warning' :
            'neutral'
          }
        >
          {regime}
        </Badge>
      </div>

      {/* Signal Reason */}
      <div className="mb-5">
        <h4 className="text-sm font-medium text-text-400 mb-1">Signal Reason</h4>
        <p className="text-text-700 text-sm leading-relaxed">
          {signalReason || 'No signal reason provided'}
        </p>
      </div>

      {/* Technical Indicators */}
      {indicators && (
        <div className="mb-5">
          <h4 className="text-sm font-medium text-text-400 mb-3">Technical Indicators</h4>
          <div className="grid grid-cols-3 sm:grid-cols-5 gap-4">
            {indicators.rsi !== undefined && (
              <IndicatorItem
                label="RSI (14)"
                value={indicators.rsi}
                color={getRsiColor(indicators.rsi)}
              />
            )}
            {indicators.adx !== undefined && (
              <IndicatorItem
                label="ADX (14)"
                value={indicators.adx}
                color={getAdxColor(indicators.adx)}
              />
            )}
            {indicators.kamaSlope !== undefined && (
              <IndicatorItem
                label="KAMA Slope"
                value={indicators.kamaSlope}
                unit="%"
                color={indicators.kamaSlope > 0 ? 'profit' : indicators.kamaSlope < 0 ? 'loss' : 'neutral'}
              />
            )}
            {indicators.atr !== undefined && (
              <IndicatorItem
                label="ATR (14)"
                value={indicators.atr}
              />
            )}
            {indicators.rSquared !== undefined && (
              <IndicatorItem
                label="R-squared"
                value={indicators.rSquared}
                color={indicators.rSquared >= 0.8 ? 'profit' : indicators.rSquared >= 0.5 ? 'neutral' : 'loss'}
              />
            )}
          </div>
        </div>
      )}

      {/* Risk Parameters */}
      {riskParams && (
        <div className="mb-5">
          <h4 className="text-sm font-medium text-text-400 mb-3">Risk Parameters</h4>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="flex flex-col">
              <span className="text-xs text-text-400">Position Size</span>
              <span className="font-mono tabular-nums font-medium text-text-700">
                {riskParams.positionSize} shares
              </span>
              {riskParams.positionPct !== undefined && (
                <span className="text-xs text-text-400">
                  ({formatPercent(riskParams.positionPct)} of equity)
                </span>
              )}
            </div>
            <div className="flex flex-col">
              <span className="text-xs text-text-400">Stop Loss</span>
              <span className="font-mono tabular-nums font-medium text-loss-500">
                {formatCurrency(riskParams.stopLoss)}
              </span>
            </div>
            {riskParams.takeProfit !== undefined && (
              <div className="flex flex-col">
                <span className="text-xs text-text-400">Take Profit</span>
                <span className="font-mono tabular-nums font-medium text-profit-500">
                  {formatCurrency(riskParams.takeProfit)}
                </span>
              </div>
            )}
            {riskParams.riskReward !== undefined && (
              <div className="flex flex-col">
                <span className="text-xs text-text-400">Risk/Reward</span>
                <span className={cn(
                  'font-mono tabular-nums font-medium',
                  riskParams.riskReward >= 2 ? 'text-profit-500' :
                  riskParams.riskReward >= 1 ? 'text-warning-500' :
                  'text-loss-500'
                )}>
                  1:{riskParams.riskReward.toFixed(1)}
                </span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Validation Checks */}
      {validationChecks && validationChecks.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-text-400 mb-3">Validation Checks</h4>
          <div className="grid grid-cols-2 gap-2">
            {validationChecks.map((check, index) => (
              <ValidationItem key={index} {...check} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default TradeReasonCard;
