/**
 * Live Analysis Page - Performance analytics dashboard
 *
 * Features:
 * - Cumulative P&L chart (algo vs benchmark) using Plotly
 * - Return % comparison chart using Plotly
 * - Summary stats panel with all risk ratios
 * - Daily breakdown table
 * - 5-second auto-refresh
 */

'use client';

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { PageContainer, PageHeader, StatsGrid } from '../../../components/layout';
import { Card, CardTitle, Button, Badge, Skeleton, Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from '../../../components/ui';
import { PLDisplay } from '../../../components/trading';
import { PnLComparisonChart, EquityCurveChart, CHART_COLORS } from '../../../components/charts';
import { formatCurrency, formatPercent } from '../../../lib/utils';

interface EquityCurvePoint {
  date: string;
  equity: number;
  benchmark_equity: number;
  daily_pnl: number;
  cumulative_pnl: number;
  cumulative_return_pct: number;
  benchmark_return_pct: number;
}

interface RiskMetrics {
  sharpe_ratio: number;
  sortino_ratio: number;
  calmar_ratio: number;
  max_drawdown: number;
  max_drawdown_pct: number;
  win_rate: number;
  profit_factor: number;
  avg_win: number;
  avg_loss: number;
}

interface DailyBreakdown {
  date: string;
  pnl: number;
  return_pct: number;
  trade_count: number;
  cumulative_pnl: number;
}

interface LiveAnalysisData {
  equity_curve: EquityCurvePoint[];
  risk_metrics: RiskMetrics;
  daily_breakdown: DailyBreakdown[];
  initial_capital: number;
  current_equity: number;
  benchmark_symbol: string;
  trading_days: number;
}

export default function LiveAnalysisPage() {
  const router = useRouter();

  const [analysisData, setAnalysisData] = useState<LiveAnalysisData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const isFetchingRef = useRef(false);

  // Fetch analysis data
  const fetchAnalysis = useCallback(async () => {
    if (isFetchingRef.current) return;
    isFetchingRef.current = true;

    try {
      setError(null);

      const response = await fetch('/api/live/analysis?benchmark=VTI', {
        headers: {
          Authorization: `Bearer ${process.env.NODE_ENV === 'development' ? 'fluxhero-dev-secret-change-in-production' : ''}`,
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch analysis: ${response.statusText}`);
      }

      const data: LiveAnalysisData = await response.json();
      setAnalysisData(data);
      setLastUpdate(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load analysis');
    } finally {
      setLoading(false);
      isFetchingRef.current = false;
    }
  }, []);

  // Initial fetch and polling
  useEffect(() => {
    fetchAnalysis();
    const interval = setInterval(fetchAnalysis, 5000);
    return () => clearInterval(interval);
  }, [fetchAnalysis]);

  if (loading) {
    return (
      <PageContainer>
        <PageHeader title="Live Analysis" subtitle="Loading..." />
        <StatsGrid columns={4} className="mb-6">
          {Array.from({ length: 4 }).map((_, i) => (
            <Card key={i}>
              <Skeleton variant="text" className="mb-2" />
              <Skeleton variant="title" />
            </Card>
          ))}
        </StatsGrid>
        <Card noPadding className="mb-6">
          <Skeleton height={300} className="rounded-2xl" />
        </Card>
      </PageContainer>
    );
  }

  if (error || !analysisData) {
    return (
      <PageContainer>
        <PageHeader title="Live Analysis" subtitle="Error" />
        <Card className="text-center py-12">
          <p className="text-loss-500 mb-4">{error || 'No analysis data available'}</p>
          <Button variant="secondary" onClick={fetchAnalysis}>
            Retry
          </Button>
        </Card>
      </PageContainer>
    );
  }

  const { risk_metrics, equity_curve, daily_breakdown } = analysisData;
  const totalReturn = equity_curve.length > 0
    ? equity_curve[equity_curve.length - 1].cumulative_return_pct
    : 0;
  const totalPnl = equity_curve.length > 0
    ? equity_curve[equity_curve.length - 1].cumulative_pnl
    : 0;
  const benchmarkReturn = equity_curve.length > 0
    ? equity_curve[equity_curve.length - 1].benchmark_return_pct
    : 0;
  const alpha = totalReturn - benchmarkReturn;

  // Prepare chart data
  const equityChartData = {
    times: equity_curve.map(p => p.date),
    equity: equity_curve.map(p => p.equity),
    benchmark: equity_curve.map(p => p.benchmark_equity),
  };

  const returnChartData = {
    times: equity_curve.map(p => p.date),
    series: [
      {
        name: 'Algo %',
        values: equity_curve.map(p => p.cumulative_return_pct),
        color: CHART_COLORS.profit,
      },
      {
        name: `${analysisData.benchmark_symbol} %`,
        values: equity_curve.map(p => p.benchmark_return_pct),
        color: '#6B7280',
      },
      {
        name: 'Diff %',
        values: equity_curve.map(p => p.cumulative_return_pct - p.benchmark_return_pct),
        color: CHART_COLORS.blue,
      },
    ],
  };

  const pnlChartData = {
    times: equity_curve.map(p => p.date),
    series: [
      {
        name: 'Algo P&L',
        values: equity_curve.map(p => p.cumulative_pnl),
        color: CHART_COLORS.profit,
      },
      {
        name: `${analysisData.benchmark_symbol} P&L`,
        values: equity_curve.map((p, i) => {
          // Calculate benchmark P&L
          const benchmarkReturn = p.benchmark_return_pct / 100;
          return analysisData.initial_capital * benchmarkReturn;
        }),
        color: CHART_COLORS.loss,
      },
    ],
  };

  return (
    <PageContainer>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <PageHeader
          title="P&L Analysis"
          subtitle={`Performance analytics vs ${analysisData.benchmark_symbol}`}
        />
        <div className="flex items-center gap-3">
          <Button variant="secondary" size="sm" onClick={() => {}}>
            Export HTML
          </Button>
          <Button variant="primary" size="sm" onClick={() => router.push('/trades')}>
            Live Trades
          </Button>
        </div>
      </div>

      {/* Main Layout: Charts on left, Stats on right */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 mb-4">
        {/* Charts Column - spans 3 columns */}
        <div className="lg:col-span-3 space-y-4">
          {/* Cumulative P&L Chart */}
          <Card noPadding className="overflow-hidden">
            <div className="px-3 py-2 border-b border-panel-500">
              <span className="text-sm font-medium text-text-800">Cumulative P&L vs. Date</span>
            </div>
            <div style={{ height: 280 }}>
              <PnLComparisonChart
                data={pnlChartData}
                height={280}
                formatAsCurrency={true}
              />
            </div>
          </Card>

          {/* Return % Comparison Chart */}
          <Card noPadding className="overflow-hidden">
            <div className="px-3 py-2 border-b border-panel-500">
              <span className="text-sm font-medium text-text-800">Return % vs. Date</span>
            </div>
            <div style={{ height: 250 }}>
              <PnLComparisonChart
                data={returnChartData}
                height={250}
                formatAsPercent={true}
              />
            </div>
          </Card>
        </div>

        {/* Summary Statistics Column */}
        <div className="lg:col-span-1">
          <Card className="h-full">
            <CardTitle className="text-sm mb-3">Summary Statistics</CardTitle>
            <div className="space-y-2 text-sm">
              <StatRow label="Initial Capital" value={formatCurrency(analysisData.initial_capital)} />
              <StatRow label="Trading Days" value={analysisData.trading_days.toString()} />
              <StatRow label="Risk-Free Rate" value="3.64%" />
              <StatRow
                label="Max Drawdown"
                value={`-${risk_metrics.max_drawdown_pct.toFixed(2)}%`}
                color="loss"
              />
              <StatRow
                label="Annualized Return"
                value={`${totalReturn >= 0 ? '+' : ''}${(totalReturn * (252 / Math.max(analysisData.trading_days, 1))).toFixed(2)}%`}
                color={totalReturn >= 0 ? 'profit' : 'loss'}
              />
              <StatRow
                label={`${analysisData.benchmark_symbol} Return`}
                value={`${benchmarkReturn >= 0 ? '+' : ''}${benchmarkReturn.toFixed(2)}%`}
                color={benchmarkReturn >= 0 ? 'profit' : 'loss'}
              />
              <StatRow
                label="Sharpe Ratio"
                value={risk_metrics.sharpe_ratio.toFixed(2)}
                color={risk_metrics.sharpe_ratio >= 1 ? 'profit' : risk_metrics.sharpe_ratio >= 0 ? 'neutral' : 'loss'}
              />
              <StatRow
                label="Sortino Ratio"
                value={risk_metrics.sortino_ratio.toFixed(2)}
                color={risk_metrics.sortino_ratio >= 1.5 ? 'profit' : risk_metrics.sortino_ratio >= 0 ? 'neutral' : 'loss'}
              />
              <StatRow
                label="Calmar Ratio"
                value={risk_metrics.calmar_ratio.toFixed(2)}
                color={risk_metrics.calmar_ratio >= 1 ? 'profit' : risk_metrics.calmar_ratio >= 0 ? 'neutral' : 'loss'}
              />
            </div>
          </Card>
        </div>
      </div>

      {/* Daily Breakdown Table */}
      <Card noPadding>
        <div className="overflow-x-auto max-h-64">
          <table className="w-full text-xs">
            <thead className="bg-panel-700 sticky top-0">
              <tr>
                <th className="px-2 py-1.5 text-left text-text-400 font-medium">DATE</th>
                <th className="px-2 py-1.5 text-right text-text-400 font-medium">VTI CLOSE</th>
                <th className="px-2 py-1.5 text-right text-text-400 font-medium">BAH VALUE</th>
                <th className="px-2 py-1.5 text-right text-text-400 font-medium">BAH CUM</th>
                <th className="px-2 py-1.5 text-right text-text-400 font-medium">BAH DAILY %</th>
                <th className="px-2 py-1.5 text-right text-text-400 font-medium">BAH DD</th>
                <th className="px-2 py-1.5 text-right text-text-400 font-medium">BAH %</th>
                <th className="px-2 py-1.5 text-right text-text-400 font-medium">DIFF %</th>
                <th className="px-2 py-1.5 text-right text-text-400 font-medium">ALGO %</th>
                <th className="px-2 py-1.5 text-right text-text-400 font-medium">ALGO VALUE</th>
                <th className="px-2 py-1.5 text-right text-text-400 font-medium">ALGO P&L</th>
                <th className="px-2 py-1.5 text-right text-text-400 font-medium">DAILY P&L</th>
                <th className="px-2 py-1.5 text-right text-text-400 font-medium">DAILY %</th>
                <th className="px-2 py-1.5 text-right text-text-400 font-medium">ALGO DD</th>
              </tr>
            </thead>
            <tbody>
              {equity_curve.length === 0 ? (
                <tr>
                  <td colSpan={14} className="px-2 py-4 text-center text-text-400">
                    No trading data available
                  </td>
                </tr>
              ) : (
                equity_curve.map((point, index) => {
                  const prevPoint = index > 0 ? equity_curve[index - 1] : null;
                  const dailyPnl = prevPoint ? point.cumulative_pnl - prevPoint.cumulative_pnl : point.cumulative_pnl;
                  const dailyPct = prevPoint ? point.cumulative_return_pct - prevPoint.cumulative_return_pct : point.cumulative_return_pct;
                  const bahDailyPct = prevPoint ? point.benchmark_return_pct - prevPoint.benchmark_return_pct : point.benchmark_return_pct;
                  const diff = point.cumulative_return_pct - point.benchmark_return_pct;
                  const bahValue = analysisData.initial_capital * (1 + point.benchmark_return_pct / 100);
                  const bahCum = bahValue - analysisData.initial_capital;

                  return (
                    <tr key={point.date} className="border-t border-panel-500/30 hover:bg-panel-500/20">
                      <td className="px-2 py-1 font-mono text-text-600">{point.date}</td>
                      <td className="px-2 py-1 text-right font-mono tabular-nums text-text-600">
                        ${point.benchmark_equity.toFixed(2)}
                      </td>
                      <td className="px-2 py-1 text-right font-mono tabular-nums text-text-600">
                        ${bahValue.toFixed(2)}
                      </td>
                      <td className={`px-2 py-1 text-right font-mono tabular-nums ${bahCum >= 0 ? 'text-profit-500' : 'text-loss-500'}`}>
                        ${bahCum.toFixed(2)}
                      </td>
                      <td className={`px-2 py-1 text-right font-mono tabular-nums ${bahDailyPct >= 0 ? 'text-profit-500' : 'text-loss-500'}`}>
                        {bahDailyPct >= 0 ? '+' : ''}{bahDailyPct.toFixed(2)}%
                      </td>
                      <td className="px-2 py-1 text-right font-mono tabular-nums text-text-400">0.00%</td>
                      <td className={`px-2 py-1 text-right font-mono tabular-nums ${point.benchmark_return_pct >= 0 ? 'text-profit-500' : 'text-loss-500'}`}>
                        {point.benchmark_return_pct >= 0 ? '+' : ''}{point.benchmark_return_pct.toFixed(2)}%
                      </td>
                      <td className={`px-2 py-1 text-right font-mono tabular-nums ${diff >= 0 ? 'text-profit-500' : 'text-loss-500'}`}>
                        {diff >= 0 ? '+' : ''}{diff.toFixed(2)}%
                      </td>
                      <td className={`px-2 py-1 text-right font-mono tabular-nums ${point.cumulative_return_pct >= 0 ? 'text-profit-500' : 'text-loss-500'}`}>
                        {point.cumulative_return_pct >= 0 ? '+' : ''}{point.cumulative_return_pct.toFixed(2)}%
                      </td>
                      <td className="px-2 py-1 text-right font-mono tabular-nums text-text-600">
                        ${point.equity.toFixed(2)}
                      </td>
                      <td className={`px-2 py-1 text-right font-mono tabular-nums ${point.cumulative_pnl >= 0 ? 'text-profit-500' : 'text-loss-500'}`}>
                        ${point.cumulative_pnl.toFixed(2)}
                      </td>
                      <td className={`px-2 py-1 text-right font-mono tabular-nums ${dailyPnl >= 0 ? 'text-profit-500' : 'text-loss-500'}`}>
                        ${dailyPnl.toFixed(2)}
                      </td>
                      <td className={`px-2 py-1 text-right font-mono tabular-nums ${dailyPct >= 0 ? 'text-profit-500' : 'text-loss-500'}`}>
                        {dailyPct >= 0 ? '+' : ''}{dailyPct.toFixed(2)}%
                      </td>
                      <td className="px-2 py-1 text-right font-mono tabular-nums text-text-400">0.00%</td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </PageContainer>
  );
}

// Helper component for stat rows
function StatRow({
  label,
  value,
  color = 'neutral'
}: {
  label: string;
  value: string;
  color?: 'profit' | 'loss' | 'neutral';
}) {
  const colorClass = {
    profit: 'text-profit-500',
    loss: 'text-loss-500',
    neutral: 'text-text-800',
  }[color];

  return (
    <div className="flex items-center justify-between">
      <span className="text-text-400">{label}</span>
      <span className={`font-mono tabular-nums font-medium ${colorClass}`}>{value}</span>
    </div>
  );
}
