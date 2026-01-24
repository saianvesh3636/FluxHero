/**
 * DailyTradeGroup - Expandable accordion grouping trades by date
 *
 * Features:
 * - Collapsible accordion with date header
 * - P&L summary, trade count, and daily return %
 * - Green/red background tint based on daily P&L
 * - Expandable to show individual trades
 */

'use client';

import React, { useState } from 'react';
import { cn, formatCurrency, formatPercent } from '../../lib/utils';
import { Badge } from '../ui';
import { Trade } from '../../utils/api';

export interface DailyTradeGroupProps {
  date: string; // YYYY-MM-DD
  trades: Trade[];
  totalPnl: number;
  winCount: number;
  lossCount: number;
  dailyReturnPct: number;
  defaultExpanded?: boolean;
  onTradeClick?: (tradeId: number) => void;
  className?: string;
}

export function DailyTradeGroup({
  date,
  trades,
  totalPnl,
  winCount,
  lossCount,
  dailyReturnPct,
  defaultExpanded = false,
  onTradeClick,
  className,
}: DailyTradeGroupProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  const isProfit = totalPnl > 0;
  const isLoss = totalPnl < 0;

  // Format date for display
  const formattedDate = new Date(date).toLocaleDateString('en-US', {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });

  return (
    <div
      className={cn(
        'rounded-xl overflow-hidden mb-3',
        isProfit && 'bg-profit-500/5 border border-profit-500/20',
        isLoss && 'bg-loss-500/5 border border-loss-500/20',
        !isProfit && !isLoss && 'bg-panel-600 border border-panel-500',
        className
      )}
    >
      {/* Header - Clickable to expand/collapse */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className={cn(
          'w-full px-4 py-3 flex items-center justify-between',
          'hover:bg-panel-500/30 transition-colors',
          'text-left'
        )}
      >
        <div className="flex items-center gap-4">
          {/* Expand/Collapse indicator */}
          <span className={cn(
            'w-6 h-6 flex items-center justify-center rounded',
            'bg-panel-500 text-text-400',
            'transition-transform duration-200',
            isExpanded && 'rotate-90'
          )}>
            {'\u25B6'}
          </span>

          {/* Date */}
          <span className="font-medium text-text-900">{formattedDate}</span>

          {/* Trade count badge */}
          <Badge variant="neutral" size="sm">
            {trades.length} trade{trades.length !== 1 ? 's' : ''}
          </Badge>
        </div>

        <div className="flex items-center gap-6">
          {/* Win/Loss counts */}
          <div className="flex items-center gap-2 text-sm">
            <span className="text-profit-500">{winCount}W</span>
            <span className="text-text-400">/</span>
            <span className="text-loss-500">{lossCount}L</span>
          </div>

          {/* Daily return */}
          <span className={cn(
            'text-sm font-mono tabular-nums',
            isProfit && 'text-profit-500',
            isLoss && 'text-loss-500',
            !isProfit && !isLoss && 'text-text-400'
          )}>
            {formatPercent(dailyReturnPct, true)}
          </span>

          {/* Daily P&L */}
          <span className={cn(
            'font-semibold font-mono tabular-nums min-w-[100px] text-right',
            isProfit && 'text-profit-500',
            isLoss && 'text-loss-500',
            !isProfit && !isLoss && 'text-text-400'
          )}>
            {formatCurrency(totalPnl, true)}
          </span>
        </div>
      </button>

      {/* Expanded trade list */}
      {isExpanded && trades.length > 0 && (
        <div className="border-t border-panel-500">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-panel-600/50">
                <th className="px-4 py-2 text-left text-xs text-text-400 font-medium">Symbol</th>
                <th className="px-4 py-2 text-left text-xs text-text-400 font-medium">Side</th>
                <th className="px-4 py-2 text-right text-xs text-text-400 font-medium">Entry</th>
                <th className="px-4 py-2 text-right text-xs text-text-400 font-medium">Exit</th>
                <th className="px-4 py-2 text-right text-xs text-text-400 font-medium">Shares</th>
                <th className="px-4 py-2 text-right text-xs text-text-400 font-medium">P&L</th>
                <th className="px-4 py-2 text-center text-xs text-text-400 font-medium">Chart</th>
              </tr>
            </thead>
            <tbody>
              {trades.map((trade) => {
                const tradePnl = trade.realized_pnl || 0;
                const tradeProfit = tradePnl > 0;
                const tradeLoss = tradePnl < 0;

                return (
                  <tr
                    key={trade.id}
                    className="border-t border-panel-500/50 hover:bg-panel-500/30"
                  >
                    <td className="px-4 py-2 font-medium text-text-900">
                      {trade.symbol}
                    </td>
                    <td className="px-4 py-2">
                      <Badge
                        variant={trade.side === 'buy' ? 'success' : 'error'}
                        size="sm"
                      >
                        {trade.side.toUpperCase()}
                      </Badge>
                    </td>
                    <td className="px-4 py-2 text-right font-mono tabular-nums text-text-700">
                      {formatCurrency(trade.entry_price)}
                    </td>
                    <td className="px-4 py-2 text-right font-mono tabular-nums text-text-700">
                      {trade.exit_price ? formatCurrency(trade.exit_price) : '-'}
                    </td>
                    <td className="px-4 py-2 text-right font-mono tabular-nums">
                      {trade.shares}
                    </td>
                    <td className={cn(
                      'px-4 py-2 text-right font-mono tabular-nums font-semibold',
                      tradeProfit && 'text-profit-500',
                      tradeLoss && 'text-loss-500',
                      !tradeProfit && !tradeLoss && 'text-text-400'
                    )}>
                      {formatCurrency(tradePnl, true)}
                    </td>
                    <td className="px-4 py-2 text-center">
                      {trade.id && onTradeClick && (
                        <button
                          onClick={() => onTradeClick(trade.id!)}
                          className={cn(
                            'w-8 h-8 rounded-lg',
                            'bg-panel-500 hover:bg-accent-500',
                            'text-text-400 hover:text-white',
                            'flex items-center justify-center',
                            'transition-colors'
                          )}
                          title="View Chart"
                        >
                          {/* Chart icon */}
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            viewBox="0 0 20 20"
                            fill="currentColor"
                            className="w-4 h-4"
                          >
                            <path
                              fillRule="evenodd"
                              d="M12 2.25c-5.385 0-9.75 4.365-9.75 9.75s4.365 9.75 9.75 9.75 9.75-4.365 9.75-9.75S17.385 2.25 12 2.25zM12.75 9a.75.75 0 00-1.5 0v2.25H9a.75.75 0 000 1.5h2.25V15a.75.75 0 001.5 0v-2.25H15a.75.75 0 000-1.5h-2.25V9z"
                              clipRule="evenodd"
                            />
                          </svg>
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default DailyTradeGroup;
