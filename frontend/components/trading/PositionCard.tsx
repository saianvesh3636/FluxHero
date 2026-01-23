import React from 'react';
import { cn, formatCurrency, formatPercent, formatPrice } from '../../lib/utils';
import { Card } from '../ui/Card';
import { Badge } from '../ui/Badge';
import { PLDisplay } from './PLDisplay';

export interface Position {
  symbol: string;
  quantity: number;
  side: 'long' | 'short';
  entryPrice: number;
  currentPrice: number;
  pnl: number;
  pnlPercent: number;
  marketValue?: number;
}

export interface PositionCardProps {
  position: Position;
  onClick?: () => void;
  className?: string;
}

/**
 * PositionCard - displays a single position with key metrics
 * Follows design system: card styling, color-coded P&L
 */
export function PositionCard({
  position,
  onClick,
  className,
}: PositionCardProps) {
  const { symbol, quantity, side, entryPrice, currentPrice, pnl, pnlPercent, marketValue } = position;

  const isLong = side === 'long';

  return (
    <Card
      className={cn(
        onClick && 'cursor-pointer hover:bg-panel-500',
        className
      )}
      onClick={onClick}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-lg font-semibold text-text-900">{symbol}</span>
          <Badge variant={isLong ? 'success' : 'error'} size="sm">
            {isLong ? 'LONG' : 'SHORT'}
          </Badge>
        </div>
        <span className="text-sm text-text-400">
          {Math.abs(quantity)} shares
        </span>
      </div>

      {/* Prices */}
      <div className="grid grid-cols-2 gap-4 mb-3">
        <div>
          <span className="text-xs text-text-400 block mb-1">Entry</span>
          <span className="text-text-700 font-mono tabular-nums">
            ${formatPrice(entryPrice)}
          </span>
        </div>
        <div>
          <span className="text-xs text-text-400 block mb-1">Current</span>
          <span className="text-text-900 font-mono tabular-nums font-medium">
            ${formatPrice(currentPrice)}
          </span>
        </div>
      </div>

      {/* P&L */}
      <div className="pt-3 border-t border-panel-500">
        <div className="flex items-center justify-between">
          <span className="text-sm text-text-400">P&L</span>
          <PLDisplay value={pnl} percent={pnlPercent} size="md" />
        </div>
        {marketValue !== undefined && (
          <div className="flex items-center justify-between mt-1">
            <span className="text-xs text-text-300">Market Value</span>
            <span className="text-sm text-text-500 font-mono tabular-nums">
              {formatCurrency(marketValue)}
            </span>
          </div>
        )}
      </div>
    </Card>
  );
}

export default PositionCard;
