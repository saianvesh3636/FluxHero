/**
 * BacktestAnalysisView - Composite component for analyzing backtest results
 *
 * Features:
 * - Tab toggle: Full Chart | Trade List
 * - Full Chart: MultiTradeChart with all markers
 * - Trade List: Table with drill-down to TradeAnalysisView
 * - Key metrics summary
 */

'use client';

import React, { useState, useEffect } from 'react';
import { cn, formatCurrency, formatPercent } from '../../lib/utils';
import { Badge } from '../ui';
import { TradeAnalysisView } from './TradeAnalysisView';
import { MultiTradeChart, type TradeMarker } from '../charts';
import { ChartContainer } from '../charts/core/ChartContainer';
import { apiClient, type BacktestResultDetail, type BacktestChartDataResponse } from '../../utils/api';

export interface BacktestTrade {
  id: number;
  symbol: string;
  side: 'buy' | 'sell';
  entry_price: number;
  exit_price: number | null;
  entry_time: string;
  exit_time: string | null;
  shares: number;
  realized_pnl: number | null;
  return_pct?: number;
}

export interface BacktestAnalysisViewProps {
  backtest: BacktestResultDetail;
  trades?: BacktestTrade[];
  className?: string;
}

type TabType = 'chart' | 'trades';

export function BacktestAnalysisView({
  backtest,
  trades = [],
  className,
}: BacktestAnalysisViewProps) {
  const [activeTab, setActiveTab] = useState<TabType>('trades');
  const [selectedTradeId, setSelectedTradeId] = useState<number | null>(null);
  const [chartData, setChartData] = useState<BacktestChartDataResponse | null>(null);
  const [chartLoading, setChartLoading] = useState(false);
  const [chartError, setChartError] = useState<string | null>(null);

  // Parse trades from JSON if not provided
  const parsedTrades = trades.length > 0 ? trades : (() => {
    if (!backtest.trades_json) return [];
    try {
      return JSON.parse(backtest.trades_json) as BacktestTrade[];
    } catch {
      return [];
    }
  })();

  // Fetch chart data when switching to chart tab
  useEffect(() => {
    if (activeTab === 'chart' && !chartData && !chartLoading) {
      const fetchChartData = async () => {
        setChartLoading(true);
        setChartError(null);
        try {
          const data = await apiClient.getBacktestChartData(backtest.run_id);
          setChartData(data);
        } catch (err) {
          setChartError(err instanceof Error ? err.message : 'Failed to load chart data');
        } finally {
          setChartLoading(false);
        }
      };
      fetchChartData();
    }
  }, [activeTab, backtest.run_id, chartData, chartLoading]);

  // Convert chart markers to MultiTradeChart format
  const chartMarkers: TradeMarker[] = chartData?.markers.map(m => ({
    tradeId: m.trade_id,
    time: m.time,
    price: m.price,
    type: m.type as 'entry' | 'exit',
    side: m.side as 'long' | 'short',
    pnl: m.pnl,
  })) || [];

  const handleMarkerClick = (tradeId: number) => {
    setSelectedTradeId(tradeId);
  };

  const isProfitable = (backtest.total_return_pct ?? 0) > 0;

  // Metrics summary
  const metrics = [
    { label: 'Total Return', value: formatPercent(backtest.total_return_pct ?? 0), isPnl: true },
    { label: 'Sharpe Ratio', value: (backtest.sharpe_ratio ?? 0).toFixed(2) },
    { label: 'Max Drawdown', value: formatPercent(backtest.max_drawdown_pct ?? 0), isNegative: true },
    { label: 'Win Rate', value: formatPercent(backtest.win_rate ?? 0) },
    { label: 'Trades', value: String(backtest.num_trades ?? 0) },
  ];

  // If a trade is selected, show the TradeAnalysisView
  if (selectedTradeId !== null) {
    return (
      <div className={className}>
        <button
          onClick={() => setSelectedTradeId(null)}
          className="flex items-center gap-2 text-accent-500 hover:text-accent-400 mb-4"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          Back to Backtest
        </button>
        <TradeAnalysisView
          tradeId={selectedTradeId}
          mode="paper"
          onClose={() => setSelectedTradeId(null)}
        />
      </div>
    );
  }

  return (
    <div className={cn('space-y-4', className)}>
      {/* Backtest Summary Header */}
      <div className="bg-panel-600 rounded-xl p-4">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <span className="text-xl font-bold text-text-900">{backtest.symbol}</span>
            <Badge variant="info">{backtest.strategy_mode}</Badge>
            <Badge variant={isProfitable ? 'success' : 'error'}>
              {isProfitable ? 'PROFITABLE' : 'LOSS'}
            </Badge>
          </div>
          <span className="text-text-400 text-sm">
            {backtest.start_date} — {backtest.end_date}
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-5 gap-4">
          {metrics.map((metric, i) => (
            <div key={i}>
              <span className="text-xs text-text-400">{metric.label}</span>
              <p className={cn(
                'font-mono tabular-nums font-medium',
                metric.isPnl ? (isProfitable ? 'text-profit-500' : 'text-loss-500') :
                metric.isNegative ? 'text-loss-500' :
                'text-text-700'
              )}>
                {metric.value}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Equity Curve (if available) */}
      {backtest.equity_curve_json && (
        <div className="bg-panel-600 rounded-xl p-4">
          <h3 className="text-text-700 font-medium mb-4">Equity Curve</h3>
          <div className="h-48 flex items-center justify-center text-text-400">
            {/* TODO: Add equity curve chart */}
            <span>Equity curve visualization coming soon</span>
          </div>
        </div>
      )}

      {/* Tab Navigation */}
      <div className="flex gap-2 border-b border-panel-500 pb-2">
        <button
          onClick={() => setActiveTab('trades')}
          className={cn(
            'px-4 py-2 text-sm font-medium rounded-t transition-colors',
            activeTab === 'trades'
              ? 'bg-panel-600 text-text-900 border-b-2 border-accent-500'
              : 'text-text-400 hover:text-text-700'
          )}
        >
          Trade List ({parsedTrades.length})
        </button>
        <button
          onClick={() => setActiveTab('chart')}
          className={cn(
            'px-4 py-2 text-sm font-medium rounded-t transition-colors',
            activeTab === 'chart'
              ? 'bg-panel-600 text-text-900 border-b-2 border-accent-500'
              : 'text-text-400 hover:text-text-700'
          )}
        >
          Full Chart
        </button>
      </div>

      {/* Tab Content */}
      {activeTab === 'trades' ? (
        <div className="bg-panel-600 rounded-xl overflow-hidden">
          {parsedTrades.length === 0 ? (
            <div className="p-8 text-center text-text-400">
              No trades available for this backtest
            </div>
          ) : (
            <table className="w-full">
              <thead>
                <tr className="border-b border-panel-500">
                  <th className="text-left px-4 py-3 text-xs font-medium text-text-400">Symbol</th>
                  <th className="text-left px-4 py-3 text-xs font-medium text-text-400">Side</th>
                  <th className="text-right px-4 py-3 text-xs font-medium text-text-400">Entry</th>
                  <th className="text-right px-4 py-3 text-xs font-medium text-text-400">Exit</th>
                  <th className="text-right px-4 py-3 text-xs font-medium text-text-400">Shares</th>
                  <th className="text-right px-4 py-3 text-xs font-medium text-text-400">P&L</th>
                  <th className="text-right px-4 py-3 text-xs font-medium text-text-400">Return</th>
                  <th className="text-center px-4 py-3 text-xs font-medium text-text-400">Action</th>
                </tr>
              </thead>
              <tbody>
                {parsedTrades.map((trade, index) => {
                  const tradePnl = trade.realized_pnl ?? 0;
                  const tradeIsProfitable = tradePnl > 0;
                  const returnPct = trade.return_pct ?? (
                    trade.entry_price && trade.shares
                      ? (tradePnl / (trade.entry_price * trade.shares)) * 100
                      : 0
                  );

                  return (
                    <tr
                      key={trade.id || index}
                      className="border-b border-panel-500 hover:bg-panel-500/50 transition-colors"
                    >
                      <td className="px-4 py-3 text-text-700 font-medium">{trade.symbol}</td>
                      <td className="px-4 py-3">
                        <Badge variant={trade.side === 'buy' ? 'success' : 'error'} size="sm">
                          {trade.side.toUpperCase()}
                        </Badge>
                      </td>
                      <td className="px-4 py-3 text-right font-mono tabular-nums text-text-600">
                        {formatCurrency(trade.entry_price)}
                      </td>
                      <td className="px-4 py-3 text-right font-mono tabular-nums text-text-600">
                        {trade.exit_price ? formatCurrency(trade.exit_price) : '—'}
                      </td>
                      <td className="px-4 py-3 text-right font-mono tabular-nums text-text-600">
                        {trade.shares}
                      </td>
                      <td className={cn(
                        'px-4 py-3 text-right font-mono tabular-nums font-medium',
                        tradeIsProfitable ? 'text-profit-500' : 'text-loss-500'
                      )}>
                        {formatCurrency(tradePnl)}
                      </td>
                      <td className={cn(
                        'px-4 py-3 text-right font-mono tabular-nums font-medium',
                        tradeIsProfitable ? 'text-profit-500' : 'text-loss-500'
                      )}>
                        {formatPercent(returnPct)}
                      </td>
                      <td className="px-4 py-3 text-center">
                        {trade.id && (
                          <button
                            onClick={() => setSelectedTradeId(trade.id)}
                            className="text-accent-500 hover:text-accent-400 text-sm font-medium"
                          >
                            Analyze
                          </button>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      ) : (
        <div className="bg-panel-600 rounded-xl overflow-hidden">
          <div className="px-4 py-3 border-b border-panel-500 flex items-center justify-between">
            <span className="text-text-700 font-medium">
              {backtest.symbol} - Full Backtest Chart
            </span>
            <span className="text-text-400 text-sm">
              {chartMarkers.length / 2} trades
            </span>
          </div>
          {chartLoading ? (
            <ChartContainer height={500} isLoading={true}>
              <div />
            </ChartContainer>
          ) : chartError ? (
            <div className="h-96 flex items-center justify-center text-text-400">
              <div className="text-center">
                <p className="text-loss-500 mb-2">{chartError}</p>
                <button
                  onClick={() => {
                    setChartData(null);
                    setChartError(null);
                  }}
                  className="text-accent-500 hover:text-accent-400 text-sm"
                >
                  Retry
                </button>
              </div>
            </div>
          ) : chartData && chartData.candles.length > 0 ? (
            <MultiTradeChart
              candles={chartData.candles}
              indicators={chartData.indicators}
              markers={chartMarkers}
              height={500}
              showOHLCData={true}
              onMarkerClick={handleMarkerClick}
            />
          ) : (
            <div className="h-96 flex items-center justify-center text-text-400">
              No chart data available
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default BacktestAnalysisView;
