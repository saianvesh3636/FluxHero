import React from 'react';
import { cn, formatCurrency, formatPrice, formatDateTime, formatPercent } from '../../lib/utils';
import { Badge } from '../ui/Badge';
import { PLBadge } from './PLDisplay';

export interface TradeData {
  id: number | string;
  symbol: string;
  side: 'buy' | 'sell';
  entryTime?: string | number;
  exitTime?: string | number;
  entryPrice?: number;
  exitPrice?: number;
  quantity: number;
  pnl?: number;
  pnlPercent?: number;
  strategy?: string;
  regime?: string;
  signalReason?: string;
}

export interface TradeRowProps {
  trade: TradeData;
  onClick?: (trade: TradeData) => void;
  showDetails?: boolean;
  className?: string;
}

/**
 * TradeRow - displays a single trade with key metrics
 * Can be used in tables or as standalone cards
 */
export function TradeRow({
  trade,
  onClick,
  showDetails = false,
  className,
}: TradeRowProps) {
  const {
    symbol,
    side,
    entryTime,
    exitTime,
    entryPrice,
    exitPrice,
    quantity,
    pnl,
    pnlPercent,
    strategy,
    regime,
    signalReason,
  } = trade;

  const isBuy = side === 'buy';

  return (
    <div
      className={cn(
        'bg-panel-600 rounded-2xl p-4',
        onClick && 'cursor-pointer hover:bg-panel-500',
        className
      )}
      onClick={() => onClick?.(trade)}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-text-900">{symbol}</span>
          <Badge variant={isBuy ? 'success' : 'error'} size="sm">
            {side.toUpperCase()}
          </Badge>
          {strategy && (
            <Badge variant="info" size="sm">
              {strategy}
            </Badge>
          )}
          {regime && (
            <Badge variant="neutral" size="sm">
              {regime}
            </Badge>
          )}
        </div>
        {pnl !== undefined && <PLBadge value={pnl} />}
      </div>

      {/* Trade Details */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm">
        {entryTime && (
          <div>
            <span className="text-text-400 block text-xs mb-1">Entry Time</span>
            <span className="text-text-700">
              {typeof entryTime === 'number'
                ? formatDateTime(new Date(entryTime * 1000).toISOString())
                : formatDateTime(entryTime)}
            </span>
          </div>
        )}
        {exitTime && (
          <div>
            <span className="text-text-400 block text-xs mb-1">Exit Time</span>
            <span className="text-text-700">
              {typeof exitTime === 'number'
                ? formatDateTime(new Date(exitTime * 1000).toISOString())
                : formatDateTime(exitTime)}
            </span>
          </div>
        )}
        {entryPrice !== undefined && (
          <div>
            <span className="text-text-400 block text-xs mb-1">Entry Price</span>
            <span className="text-text-700 font-mono tabular-nums">
              ${formatPrice(entryPrice)}
            </span>
          </div>
        )}
        {exitPrice !== undefined && (
          <div>
            <span className="text-text-400 block text-xs mb-1">Exit Price</span>
            <span className="text-text-700 font-mono tabular-nums">
              ${formatPrice(exitPrice)}
            </span>
          </div>
        )}
        <div>
          <span className="text-text-400 block text-xs mb-1">Quantity</span>
          <span className="text-text-700 font-mono tabular-nums">{quantity}</span>
        </div>
        {pnlPercent !== undefined && (
          <div>
            <span className="text-text-400 block text-xs mb-1">Return</span>
            <span
              className={cn(
                'font-mono tabular-nums',
                pnlPercent > 0
                  ? 'text-profit-500'
                  : pnlPercent < 0
                  ? 'text-loss-500'
                  : 'text-text-500'
              )}
            >
              {pnlPercent > 0 && '+'}
              {formatPercent(pnlPercent)}
            </span>
          </div>
        )}
      </div>

      {/* Signal Reason (optional) */}
      {showDetails && signalReason && (
        <div className="mt-3 pt-3 border-t border-panel-500">
          <span className="text-text-400 text-xs block mb-1">Signal Reason</span>
          <p className="text-text-600 text-sm">{signalReason}</p>
        </div>
      )}
    </div>
  );
}

export default TradeRow;
