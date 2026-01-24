/**
 * TradeSummaryFooter - Compact fixed bottom bar with trading statistics
 *
 * Displays 10 key metrics in a dense horizontal layout:
 * - Closed/Open trade count
 * - Realized/Unrealized P&L
 * - Total P&L, Return %, VTBAM%, VTI, Annualized
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
  vsBuyAndHold?: number; // vs BAH percentage (VTBAM%)
  vtiValue?: number; // VTI benchmark value
  annualizedReturn?: number;
  className?: string;
}

export function TradeSummaryFooter({
  closedCount,
  openCount,
  realizedPnl,
  unrealizedPnl,
  totalPnl,
  totalReturnPct,
  vsBuyAndHold,
  vtiValue,
  annualizedReturn,
  className,
}: TradeSummaryFooterProps) {
  const getPnlColor = (value: number): string => {
    if (value > 0) return 'text-profit-500';
    if (value < 0) return 'text-loss-500';
    return 'text-text-600';
  };

  return (
    <div
      className={cn(
        'fixed bottom-0 left-0 right-0 z-50',
        'bg-panel-800 border-t border-panel-500',
        'py-1.5 px-2',
        className
      )}
    >
      <div className="max-w-7xl mx-auto flex items-center justify-center gap-1 overflow-x-auto text-xs">
        {/* Closed */}
        <span className="text-text-400">Closed:</span>
        <span className="text-text-800 font-medium">{closedCount}</span>
        <span className="text-text-500 mx-1">|</span>

        {/* Open */}
        <span className="text-text-400">Open:</span>
        <span className={cn('font-medium', openCount > 0 ? 'text-profit-500' : 'text-text-800')}>{openCount}</span>
        <span className="text-text-500 mx-1">|</span>

        {/* Realized */}
        <span className="text-text-400">Realized:</span>
        <span className={cn('font-mono tabular-nums font-medium', getPnlColor(realizedPnl))}>
          {formatCurrency(realizedPnl, true)}
        </span>
        <span className="text-text-500 mx-1">|</span>

        {/* Unrealized */}
        <span className="text-text-400">Unrealized:</span>
        <span className={cn('font-mono tabular-nums font-medium', getPnlColor(unrealizedPnl))}>
          {formatCurrency(unrealizedPnl, true)}
        </span>
        <span className="text-text-500 mx-1">|</span>

        {/* Total */}
        <span className="text-text-400">Total:</span>
        <span className={cn('font-mono tabular-nums font-semibold', getPnlColor(totalPnl))}>
          {formatCurrency(totalPnl, true)}
        </span>
        <span className="text-text-500 mx-1">|</span>

        {/* Return */}
        <span className="text-text-400">Return:</span>
        <span className={cn('font-mono tabular-nums font-medium', getPnlColor(totalReturnPct))}>
          {formatPercent(totalReturnPct, true)}
        </span>

        {/* VTBAM% */}
        {vsBuyAndHold !== undefined && (
          <>
            <span className="text-text-500 mx-1">|</span>
            <span className="text-text-400">VTBAM:</span>
            <span className={cn('font-mono tabular-nums font-medium', getPnlColor(vsBuyAndHold))}>
              {formatPercent(vsBuyAndHold, true)}
            </span>
          </>
        )}

        {/* VTI Value */}
        {vtiValue !== undefined && (
          <>
            <span className="text-text-500 mx-1">|</span>
            <span className="text-text-400">VTI:</span>
            <span className={cn('font-mono tabular-nums font-medium', getPnlColor(vtiValue))}>
              {formatCurrency(vtiValue, true)}
            </span>
          </>
        )}

        {/* Annualized */}
        {annualizedReturn !== undefined && (
          <>
            <span className="text-text-500 mx-1">|</span>
            <span className="text-text-400">Ann:</span>
            <span className={cn('font-mono tabular-nums font-medium', getPnlColor(annualizedReturn))}>
              {formatPercent(annualizedReturn, true)}
            </span>
          </>
        )}
      </div>
    </div>
  );
}

export default TradeSummaryFooter;
