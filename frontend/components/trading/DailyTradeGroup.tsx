/**
 * DailyTradeGroup - Compact expandable accordion grouping trades by date
 *
 * Features:
 * - Single-line header with all stats inline
 * - Collapsible accordion with date header
 * - P&L summary, trade count, and daily return %
 * - Green/red background tint based on daily P&L
 * - Compact rows for dense information display
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
  unrealizedPnl?: number;
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
  unrealizedPnl = 0,
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

  return (
    <div
      className={cn(
        'overflow-hidden',
        isProfit && 'bg-profit-500/5',
        isLoss && 'bg-loss-500/5',
        !isProfit && !isLoss && 'bg-panel-600/50',
        className
      )}
    >
      {/* Compact Single-Line Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className={cn(
          'w-full px-3 py-1.5 flex items-center gap-3',
          'hover:bg-panel-500/30 transition-colors',
          'text-left text-sm'
        )}
      >
        {/* Expand/Collapse indicator */}
        <span className={cn(
          'w-4 h-4 flex items-center justify-center text-xs',
          'text-text-400',
          'transition-transform duration-200',
          isExpanded && 'rotate-90'
        )}>
          {'\u25B6'}
        </span>

        {/* Date */}
        <span className="font-medium text-text-800 min-w-[90px]">{date}</span>

        {/* Realized/Unrealized inline */}
        <span className="text-xs text-text-400">[</span>
        <span className={cn(
          'text-xs font-mono tabular-nums',
          totalPnl > 0 ? 'text-profit-500' : totalPnl < 0 ? 'text-loss-500' : 'text-text-400'
        )}>
          R {formatCurrency(totalPnl, true)}
        </span>
        <span className="text-xs text-text-400">/</span>
        <span className={cn(
          'text-xs font-mono tabular-nums',
          unrealizedPnl > 0 ? 'text-profit-500' : unrealizedPnl < 0 ? 'text-loss-500' : 'text-text-400'
        )}>
          U {formatCurrency(unrealizedPnl, true)}
        </span>
        <span className="text-xs text-text-400">]</span>

        {/* Separator */}
        <span className="text-text-400">-</span>

        {/* Trade count */}
        <span className="text-xs text-text-400">
          Trades: <span className="text-text-700">{trades.length}</span>
        </span>

        {/* Win/Loss */}
        <span className="text-xs">
          <span className="text-profit-500">{winCount}W</span>
          <span className="text-text-400">/</span>
          <span className="text-loss-500">{lossCount}L</span>
        </span>

        {/* Spacer */}
        <span className="flex-1" />

        {/* Daily return - right aligned */}
        <span className="text-xs text-text-400">Day:</span>
        <span className={cn(
          'text-xs font-mono tabular-nums font-medium min-w-[50px] text-right',
          isProfit && 'text-profit-500',
          isLoss && 'text-loss-500',
          !isProfit && !isLoss && 'text-text-400'
        )}>
          {formatPercent(dailyReturnPct, true)}
        </span>
      </button>

      {/* Compact Expanded trade list */}
      {isExpanded && trades.length > 0 && (
        <div className="border-t border-panel-500/50">
          <table className="w-full text-xs">
            <thead>
              <tr className="bg-panel-700/50">
                <th className="px-2 py-1 text-left text-text-400 font-medium">Symbol</th>
                <th className="px-2 py-1 text-left text-text-400 font-medium">Side</th>
                <th className="px-2 py-1 text-right text-text-400 font-medium">Qty</th>
                <th className="px-2 py-1 text-right text-text-400 font-medium">Entry</th>
                <th className="px-2 py-1 text-right text-text-400 font-medium">Exit</th>
                <th className="px-2 py-1 text-right text-text-400 font-medium">P&L</th>
                <th className="px-2 py-1 text-right text-text-400 font-medium">%</th>
              </tr>
            </thead>
            <tbody>
              {trades.map((trade) => {
                const tradePnl = trade.realized_pnl || 0;
                const tradeProfit = tradePnl > 0;
                const tradeLoss = tradePnl < 0;
                const pnlPct = trade.entry_price && trade.shares
                  ? (tradePnl / (trade.entry_price * trade.shares)) * 100
                  : 0;

                return (
                  <tr
                    key={trade.id}
                    className={cn(
                      'border-t border-panel-500/30 hover:bg-panel-500/20',
                      onTradeClick && 'cursor-pointer'
                    )}
                    onClick={() => trade.id && onTradeClick?.(trade.id)}
                  >
                    <td className="px-2 py-1 font-medium text-text-800">
                      {trade.symbol}
                    </td>
                    <td className="px-2 py-1">
                      <span className={cn(
                        'text-xs font-medium',
                        trade.side === 'buy' ? 'text-profit-500' : 'text-loss-500'
                      )}>
                        {trade.side.toUpperCase()}
                      </span>
                    </td>
                    <td className="px-2 py-1 text-right font-mono tabular-nums text-text-600">
                      {trade.shares}
                    </td>
                    <td className="px-2 py-1 text-right font-mono tabular-nums text-text-600">
                      {formatCurrency(trade.entry_price)}
                    </td>
                    <td className="px-2 py-1 text-right font-mono tabular-nums text-text-600">
                      {trade.exit_price ? formatCurrency(trade.exit_price) : '-'}
                    </td>
                    <td className={cn(
                      'px-2 py-1 text-right font-mono tabular-nums font-medium',
                      tradeProfit && 'text-profit-500',
                      tradeLoss && 'text-loss-500',
                      !tradeProfit && !tradeLoss && 'text-text-400'
                    )}>
                      {formatCurrency(tradePnl, true)}
                    </td>
                    <td className={cn(
                      'px-2 py-1 text-right font-mono tabular-nums',
                      tradeProfit && 'text-profit-500',
                      tradeLoss && 'text-loss-500',
                      !tradeProfit && !tradeLoss && 'text-text-400'
                    )}>
                      {formatPercent(pnlPct, true)}
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
