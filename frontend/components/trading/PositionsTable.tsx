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
 * PositionsTable - displays positions in a table format
 * Follows design system: table styling, color-coded P&L
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
      <div className={cn('bg-panel-600 rounded-2xl overflow-hidden', className)}>
        <div className="p-4">
          <div className="space-y-3">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="flex gap-4">
                <div className="h-4 w-16 bg-panel-500 rounded" />
                <div className="h-4 w-12 bg-panel-500 rounded" />
                <div className="h-4 w-20 bg-panel-500 rounded" />
                <div className="h-4 w-20 bg-panel-500 rounded" />
                <div className="h-4 w-24 bg-panel-500 rounded" />
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
    <div className={cn('bg-panel-600 rounded-2xl overflow-hidden', className)}>
      <Table>
        <TableHeader>
          <TableRow isHoverable={false}>
            <TableHead>Symbol</TableHead>
            <TableHead>Side</TableHead>
            <TableHead className="text-right">Qty</TableHead>
            <TableHead className="text-right">Entry</TableHead>
            <TableHead className="text-right">Current</TableHead>
            <TableHead className="text-right">P&L</TableHead>
            <TableHead className="text-right">P&L %</TableHead>
            {showSellButton && <TableHead className="text-center">Action</TableHead>}
          </TableRow>
        </TableHeader>
        <TableBody>
          {positions.length === 0 ? (
            <TableEmpty colSpan={showSellButton ? 8 : 7} message="No open positions" />
          ) : (
            positions.map((position, index) => {
              const side = normalizeSide(position.side);
              return (
                <TableRow
                  key={`${position.symbol}-${index}`}
                  className={onRowClick ? 'cursor-pointer' : ''}
                  onClick={() => onRowClick?.(position)}
                >
                  <TableCell>
                    <span className="font-medium text-text-900">
                      {position.symbol}
                    </span>
                  </TableCell>
                  <TableCell>
                    <Badge
                      variant={side === 'long' ? 'success' : 'error'}
                      size="sm"
                    >
                      {side.toUpperCase()}
                    </Badge>
                  </TableCell>
                  <TableCell numeric>
                    {Math.abs(position.quantity)}
                  </TableCell>
                  <TableCell numeric>
                    ${formatPrice(position.entryPrice)}
                  </TableCell>
                  <TableCell numeric className="text-text-900">
                    ${formatPrice(position.currentPrice)}
                  </TableCell>
                  <TableCell numeric>
                    <PLBadge value={position.pnl} />
                  </TableCell>
                  <TableCell
                    numeric
                    className={cn(
                      position.pnlPercent > 0
                        ? 'text-profit-500'
                        : position.pnlPercent < 0
                        ? 'text-loss-500'
                        : 'text-text-500'
                    )}
                  >
                    {position.pnlPercent > 0 && '+'}
                    {formatPercent(position.pnlPercent)}
                  </TableCell>
                  {showSellButton && (
                    <TableCell className="text-center">
                      <Button
                        variant="danger"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          onSellPosition?.(position);
                        }}
                      >
                        Sell
                      </Button>
                    </TableCell>
                  )}
                </TableRow>
              );
            })
          )}
        </TableBody>
      </Table>
    </div>
  );
}

export default PositionsTable;
