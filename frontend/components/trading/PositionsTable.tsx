import React from 'react';
import { cn, formatCurrency, formatPrice, formatPercent } from '../../lib/utils';
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
  TableEmpty,
} from '../ui/Table';
import { Badge } from '../ui/Badge';
import { Button } from '../ui/Button';
import { PLDisplay, PLBadge } from './PLDisplay';

export interface PositionRow {
  symbol: string;
  side: 'long' | 'short' | 'buy' | 'sell';
  quantity: number;
  entryPrice: number;
  currentPrice: number;
  pnl: number;
  pnlPercent: number;
  marketValue?: number;
}

export interface PositionsTableProps {
  positions: PositionRow[];
  isLoading?: boolean;
  onRowClick?: (position: PositionRow) => void;
  onSellPosition?: (position: PositionRow) => void;
  showSellButton?: boolean;
  className?: string;
}

/**
 * PositionsTable - Compact table displaying positions
 * Follows design system: table styling, color-coded P&L
 * Optimized for information density
 */
export function PositionsTable({
  positions,
  isLoading = false,
  onRowClick,
  onSellPosition,
  showSellButton = false,
  className,
}: PositionsTableProps) {
  if (isLoading) {
    return (
      <div className={cn('bg-panel-600 overflow-hidden', className)}>
        <div className="p-2">
          <div className="space-y-1">
            {Array.from({ length: 3 }).map((_, i) => (
              <div key={i} className="flex gap-2">
                <div className="h-3 w-12 bg-panel-500 rounded" />
                <div className="h-3 w-8 bg-panel-500 rounded" />
                <div className="h-3 w-16 bg-panel-500 rounded" />
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  const normalizeSide = (side: string): 'long' | 'short' => {
    return side === 'buy' || side === 'long' ? 'long' : 'short';
  };

  return (
    <div className={cn('bg-panel-600 overflow-hidden', className)}>
      <table className="w-full text-xs">
        <thead>
          <tr className="bg-panel-700/50">
            <th className="px-2 py-1.5 text-left text-text-400 font-medium">Symbol</th>
            <th className="px-2 py-1.5 text-left text-text-400 font-medium">Side</th>
            <th className="px-2 py-1.5 text-right text-text-400 font-medium">Qty</th>
            <th className="px-2 py-1.5 text-right text-text-400 font-medium">Entry</th>
            <th className="px-2 py-1.5 text-right text-text-400 font-medium">Current</th>
            <th className="px-2 py-1.5 text-right text-text-400 font-medium">P&L</th>
            <th className="px-2 py-1.5 text-right text-text-400 font-medium">%</th>
            {showSellButton && <th className="px-2 py-1.5 text-center text-text-400 font-medium">Action</th>}
          </tr>
        </thead>
        <tbody>
          {positions.length === 0 ? (
            <tr>
              <td colSpan={showSellButton ? 8 : 7} className="px-2 py-3 text-center text-text-400 text-xs">
                No open positions
              </td>
            </tr>
          ) : (
            positions.map((position, index) => {
              const side = normalizeSide(position.side);
              return (
                <tr
                  key={`${position.symbol}-${index}`}
                  className={cn(
                    'border-t border-panel-500/30 hover:bg-panel-500/20',
                    onRowClick && 'cursor-pointer'
                  )}
                  onClick={() => onRowClick?.(position)}
                >
                  <td className="px-2 py-1.5 font-medium text-text-800">
                    {position.symbol}
                  </td>
                  <td className="px-2 py-1.5">
                    <span className={cn(
                      'text-xs font-medium',
                      side === 'long' ? 'text-profit-500' : 'text-loss-500'
                    )}>
                      {side.toUpperCase()}
                    </span>
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono tabular-nums text-text-600">
                    {Math.abs(position.quantity)}
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono tabular-nums text-text-600">
                    ${formatPrice(position.entryPrice)}
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono tabular-nums text-text-800">
                    ${formatPrice(position.currentPrice)}
                  </td>
                  <td className={cn(
                    'px-2 py-1.5 text-right font-mono tabular-nums font-medium',
                    position.pnl > 0 ? 'text-profit-500' : position.pnl < 0 ? 'text-loss-500' : 'text-text-400'
                  )}>
                    {position.pnl > 0 ? '+' : ''}{formatCurrency(position.pnl)}
                  </td>
                  <td className={cn(
                    'px-2 py-1.5 text-right font-mono tabular-nums',
                    position.pnlPercent > 0 ? 'text-profit-500' : position.pnlPercent < 0 ? 'text-loss-500' : 'text-text-400'
                  )}>
                    {position.pnlPercent > 0 && '+'}
                    {formatPercent(position.pnlPercent)}
                  </td>
                  {showSellButton && (
                    <td className="px-2 py-1.5 text-center">
                      <button
                        className="px-2 py-0.5 text-xs font-medium bg-loss-500 hover:bg-loss-600 text-white rounded transition-colors"
                        onClick={(e) => {
                          e.stopPropagation();
                          onSellPosition?.(position);
                        }}
                      >
                        Sell
                      </button>
                    </td>
                  )}
                </tr>
              );
            })
          )}
        </tbody>
      </table>
    </div>
  );
}

export default PositionsTable;
