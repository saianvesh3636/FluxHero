/**
 * TradeAnalysisView - Composite component for analyzing a single trade
 *
 * Combines:
 * - Chart with entry/exit markers
 * - Signal explanation panel
 * - Trade summary stats
 *
 * Used in: modal, dedicated page, or embedded
 */

'use client';

import React, { useEffect, useState } from 'react';
import { TradeDetailChart } from '../charts/composed/TradeDetailChart';
import { ChartContainer } from '../charts/core/ChartContainer';
import { SignalExplanationPanel, type SignalExplanation } from './SignalExplanationPanel';
import { apiClient, type TradeChartDataResponse, type TradeReviewResponse } from '../../utils/api';
import { cn, formatCurrency, formatPercent } from '../../lib/utils';
import { Badge } from '../ui';
import { CHART_COLORS } from '../charts/config/theme';

export interface TradeAnalysisViewProps {
  tradeId: number;
  mode?: 'live' | 'paper';
  showChart?: boolean;
  showExplanation?: boolean;
  showSummary?: boolean;
  chartHeight?: number;
  className?: string;
  onClose?: () => void;
}

interface TradeData {
  chartData: TradeChartDataResponse | null;
  reviewData: TradeReviewResponse | null;
}

export function TradeAnalysisView({
  tradeId,
  mode = 'paper',
  showChart = true,
  showExplanation = true,
  showSummary = true,
  chartHeight = 400,
  className,
  onClose,
}: TradeAnalysisViewProps) {
  const [data, setData] = useState<TradeData>({ chartData: null, reviewData: null });
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      setError(null);

      try {
        // Fetch chart data and review data in parallel
        const [chartData, reviewData] = await Promise.all([
          apiClient.getTradeChartData(tradeId),
          apiClient.reviewTrade(tradeId, mode),
        ]);

        setData({ chartData, reviewData });
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load trade analysis');
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, [tradeId, mode]);

  if (isLoading) {
    return (
      <div className={cn('space-y-4', className)}>
        {showChart && <ChartContainer height={chartHeight} isLoading={true}><div /></ChartContainer>}
        {showExplanation && (
          <div className="bg-panel-600 rounded-xl p-5 animate-pulse">
            <div className="h-4 bg-panel-500 rounded w-1/4 mb-4" />
            <div className="h-16 bg-panel-500 rounded" />
          </div>
        )}
      </div>
    );
  }

  if (error) {
    return (
      <div className={cn('bg-panel-600 rounded-xl p-6', className)}>
        <div className="text-center">
          <p className="text-loss-500 mb-2">{error}</p>
          <p className="text-text-400 text-sm">Trade ID: {tradeId}</p>
        </div>
      </div>
    );
  }

  const { chartData, reviewData } = data;
  if (!chartData || !reviewData) return null;

  const trade = chartData.trade;
  const isClosed = trade.exit_time !== null;
  const isProfitable = (trade.realized_pnl ?? 0) > 0;
  const side = trade.side === 1 ? 'LONG' : 'SHORT';

  // Build chart markers
  const markers = [];

  // Entry marker
  if (chartData.entry_index >= 0 && chartData.candles[chartData.entry_index]) {
    const entryCandle = chartData.candles[chartData.entry_index];
    markers.push({
      time: String(entryCandle.time),
      price: trade.entry_price,
      text: 'BUY',
      color: CHART_COLORS.profit,
      position: 'below' as const,
    });
  }

  // Exit marker
  if (isClosed && chartData.exit_index !== null && chartData.exit_index >= 0 && chartData.candles[chartData.exit_index]) {
    const exitCandle = chartData.candles[chartData.exit_index];
    markers.push({
      time: String(exitCandle.time),
      price: trade.exit_price!,
      text: 'SELL',
      color: CHART_COLORS.loss,
      position: 'above' as const,
    });
  }

  // Build price lines
  const priceLines: Array<{
    price: number;
    color: string;
    lineStyle: 'solid' | 'dashed' | 'dotted';
    label: string;
  }> = [
    {
      price: trade.entry_price,
      color: CHART_COLORS.accent,
      lineStyle: 'solid',
      label: 'Entry',
    },
    {
      price: trade.stop_loss,
      color: CHART_COLORS.loss,
      lineStyle: 'dashed',
      label: 'Stop',
    },
  ];

  if (trade.take_profit) {
    priceLines.push({
      price: trade.take_profit,
      color: CHART_COLORS.profit,
      lineStyle: 'dashed',
      label: 'Target',
    });
  }

  if (isClosed && trade.exit_price) {
    priceLines.push({
      price: trade.exit_price,
      color: isProfitable ? CHART_COLORS.profit : CHART_COLORS.loss,
      lineStyle: 'solid',
      label: 'Exit',
    });
  }

  // Build signal explanation from review data
  const signalExplanation: SignalExplanation = {
    entry_trigger: reviewData.signal_reason || undefined,
    strategy: reviewData.strategy,
    regime: reviewData.regime || undefined,
    formatted_reason: reviewData.signal_explanation || undefined,
    raw_reason: reviewData.signal_reason || undefined,
    risk_params: {
      stop_loss: reviewData.stop_loss,
      take_profit: reviewData.take_profit || undefined,
      position_size: reviewData.shares,
    },
  };

  return (
    <div className={cn('space-y-4', className)}>
      {/* Trade Summary Header */}
      {showSummary && (
        <div className="bg-panel-600 rounded-xl p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-3">
              <span className="text-xl font-bold text-text-900">{trade.symbol}</span>
              <Badge variant={side === 'LONG' ? 'success' : 'error'}>
                {side}
              </Badge>
              <Badge variant={isClosed ? 'neutral' : 'info'}>
                {isClosed ? 'CLOSED' : 'OPEN'}
              </Badge>
            </div>
            {onClose && (
              <button
                onClick={onClose}
                className="text-text-400 hover:text-text-700 transition-colors"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-4">
            <div>
              <span className="text-xs text-text-400">Entry</span>
              <p className="font-mono tabular-nums text-text-700">{formatCurrency(trade.entry_price)}</p>
            </div>
            <div>
              <span className="text-xs text-text-400">Shares</span>
              <p className="font-mono tabular-nums text-text-700">{trade.shares}</p>
            </div>
            {isClosed && (
              <>
                <div>
                  <span className="text-xs text-text-400">Exit</span>
                  <p className="font-mono tabular-nums text-text-700">{formatCurrency(trade.exit_price!)}</p>
                </div>
                <div>
                  <span className="text-xs text-text-400">P&L</span>
                  <p className={cn(
                    'font-mono tabular-nums font-medium',
                    isProfitable ? 'text-profit-500' : 'text-loss-500'
                  )}>
                    {formatCurrency(trade.realized_pnl!)}
                  </p>
                </div>
                <div>
                  <span className="text-xs text-text-400">Return</span>
                  <p className={cn(
                    'font-mono tabular-nums font-medium',
                    isProfitable ? 'text-profit-500' : 'text-loss-500'
                  )}>
                    {formatPercent(reviewData.return_pct ?? 0)}
                  </p>
                </div>
                {reviewData.holding_period_days !== null && (
                  <div>
                    <span className="text-xs text-text-400">Duration</span>
                    <p className="font-mono tabular-nums text-text-700">
                      {reviewData.holding_period_days} days
                    </p>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      )}

      {/* Chart */}
      {showChart && (
        <div className="bg-panel-600 rounded-xl overflow-hidden">
          <div className="px-4 py-3 border-b border-panel-500">
            <span className="text-text-700 font-medium">Trade Chart</span>
          </div>
          <TradeDetailChart
            candles={chartData.candles}
            indicators={chartData.indicators}
            markers={markers}
            priceLines={priceLines}
            height={chartHeight}
            showOHLCData={true}
          />
        </div>
      )}

      {/* Signal Explanation */}
      {showExplanation && (
        <SignalExplanationPanel signalExplanation={signalExplanation} />
      )}
    </div>
  );
}

export default TradeAnalysisView;
