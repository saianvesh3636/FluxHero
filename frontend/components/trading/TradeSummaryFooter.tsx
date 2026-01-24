/**
 * TradeSummaryFooter - Fixed bottom bar with trading statistics
 *
 * Displays 8 key metrics in a compact horizontal layout:
 * - Closed/Open trade count
 * - Realized/Unrealized P&L
 * - Total P&L, Return %, vs BAH, Annualized
 */

import React from 'react';
import { cn, formatCurrency, formatPercent } from '../../lib/utils';

export interface TradeSummaryFooterProps {
  closedCount: number;
  openCount: number;
  realizedPnl: number;
  unrealizedPnl: number;
  totalPnl: number;
  totalReturnPct: number;
  vsBuyAndHold?: number; // vs BAH percentage
  annualizedReturn?: number;
  className?: string;
}

interface StatItemProps {
  label: string;
  value: string | number;
  color?: 'profit' | 'loss' | 'neutral';
  subValue?: string;
}

function StatItem({ label, value, color = 'neutral', subValue }: StatItemProps) {
  const colorClass = {
    profit: 'text-profit-500',
    loss: 'text-loss-500',
    neutral: 'text-text-900',
  }[color];

  return (
    <div className="flex flex-col items-center px-4 border-r border-panel-500 last:border-r-0">
      <span className="text-xs text-text-400 whitespace-nowrap">{label}</span>
      <span className={cn('text-sm font-semibold font-mono tabular-nums', colorClass)}>
        {value}
      </span>
      {subValue && (
        <span className="text-xs text-text-300">{subValue}</span>
      )}
    </div>
  );
}

export function TradeSummaryFooter({
  closedCount,
  openCount,
  realizedPnl,
  unrealizedPnl,
  totalPnl,
  totalReturnPct,
  vsBuyAndHold,
  annualizedReturn,
  className,
}: TradeSummaryFooterProps) {
  const getPnlColor = (value: number): 'profit' | 'loss' | 'neutral' => {
    if (value > 0) return 'profit';
    if (value < 0) return 'loss';
    return 'neutral';
  };

  return (
    <div
      className={cn(
        'fixed bottom-0 left-0 right-0 z-50',
        'bg-panel-700 border-t border-panel-500',
        'py-3 px-4',
        className
      )}
    >
      <div className="max-w-7xl mx-auto flex items-center justify-center overflow-x-auto">
        <StatItem
          label="Closed"
          value={closedCount}
        />

        <StatItem
          label="Open"
          value={openCount}
          color={openCount > 0 ? 'profit' : 'neutral'}
        />

        <StatItem
          label="Realized P&L"
          value={formatCurrency(realizedPnl, true)}
          color={getPnlColor(realizedPnl)}
        />

        <StatItem
          label="Unrealized P&L"
          value={formatCurrency(unrealizedPnl, true)}
          color={getPnlColor(unrealizedPnl)}
        />

        <StatItem
          label="Total P&L"
          value={formatCurrency(totalPnl, true)}
          color={getPnlColor(totalPnl)}
        />

        <StatItem
          label="Return"
          value={formatPercent(totalReturnPct, true)}
          color={getPnlColor(totalReturnPct)}
        />

        {vsBuyAndHold !== undefined && (
          <StatItem
            label="vs B&H"
            value={formatPercent(vsBuyAndHold, true)}
            color={getPnlColor(vsBuyAndHold)}
          />
        )}

        {annualizedReturn !== undefined && (
          <StatItem
            label="Annualized"
            value={formatPercent(annualizedReturn, true)}
            color={getPnlColor(annualizedReturn)}
          />
        )}
      </div>
    </div>
  );
}

export default TradeSummaryFooter;
