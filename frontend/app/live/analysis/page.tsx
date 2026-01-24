/**
 * Live Analysis Page - Performance analytics dashboard
 *
 * Features:
 * - Cumulative P&L chart (algo vs benchmark)
 * - Return % comparison chart
 * - Summary stats panel with all risk ratios
 * - Daily breakdown table
 * - 5-second auto-refresh
 */

'use client';

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import {
  createChart,
  IChartApi,
  ISeriesApi,
  LineData,
  Time,
  LineSeries,
  AreaSeries,
} from 'lightweight-charts';
import { PageContainer, PageHeader, StatsGrid } from '../../../components/layout';
import { Card, CardTitle, Button, Badge, Skeleton, Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from '../../../components/ui';
import { PLDisplay } from '../../../components/trading';
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

  const equityChartRef = useRef<HTMLDivElement>(null);
  const returnChartRef = useRef<HTMLDivElement>(null);
  const equityChartApiRef = useRef<IChartApi | null>(null);
  const returnChartApiRef = useRef<IChartApi | null>(null);

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

  // Initialize equity chart
  useEffect(() => {
    if (!equityChartRef.current || !analysisData || analysisData.equity_curve.length === 0) return;

    // Clean up existing chart
    if (equityChartApiRef.current) {
      equityChartApiRef.current.remove();
    }

    const chart = createChart(equityChartRef.current, {
      width: equityChartRef.current.clientWidth,
      height: 300,
      layout: {
        background: { color: '#1C1C28' },
        textColor: '#CCCAD5',
      },
      grid: {
        vertLines: { color: '#21222F' },
        horzLines: { color: '#21222F' },
      },
      rightPriceScale: {
        borderColor: '#21222F',
      },
      timeScale: {
        borderColor: '#21222F',
        timeVisible: true,
      },
    });

    equityChartApiRef.current = chart;

    // Algo equity line
    const algoSeries = chart.addSeries(AreaSeries, {
      lineColor: '#A549FC',
      topColor: 'rgba(165, 73, 252, 0.3)',
      bottomColor: 'rgba(165, 73, 252, 0.05)',
      lineWidth: 2,
      title: 'Strategy',
    });

    // Benchmark equity line
    const benchmarkSeries = chart.addSeries(LineSeries, {
      color: '#6B7280',
      lineWidth: 1,
      lineStyle: 2,
      title: analysisData.benchmark_symbol,
    });

    // Set data
    const algoData: LineData<Time>[] = analysisData.equity_curve.map((p) => ({
      time: p.date as Time,
      value: p.equity,
    }));

    const benchmarkData: LineData<Time>[] = analysisData.equity_curve.map((p) => ({
      time: p.date as Time,
      value: p.benchmark_equity,
    }));

    algoSeries.setData(algoData);
    benchmarkSeries.setData(benchmarkData);

    chart.timeScale().fitContent();

    const handleResize = () => {
      if (equityChartRef.current && equityChartApiRef.current) {
        equityChartApiRef.current.applyOptions({
          width: equityChartRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [analysisData]);

  // Initialize return comparison chart
  useEffect(() => {
    if (!returnChartRef.current || !analysisData || analysisData.equity_curve.length === 0) return;

    // Clean up existing chart
    if (returnChartApiRef.current) {
      returnChartApiRef.current.remove();
    }

    const chart = createChart(returnChartRef.current, {
      width: returnChartRef.current.clientWidth,
      height: 250,
      layout: {
        background: { color: '#1C1C28' },
        textColor: '#CCCAD5',
      },
      grid: {
        vertLines: { color: '#21222F' },
        horzLines: { color: '#21222F' },
      },
      rightPriceScale: {
        borderColor: '#21222F',
      },
      timeScale: {
        borderColor: '#21222F',
        timeVisible: true,
      },
    });

    returnChartApiRef.current = chart;

    // Algo return line
    const algoReturnSeries = chart.addSeries(LineSeries, {
      color: '#22C55E',
      lineWidth: 2,
      title: 'Strategy %',
    });

    // Benchmark return line
    const benchmarkReturnSeries = chart.addSeries(LineSeries, {
      color: '#6B7280',
      lineWidth: 1,
      lineStyle: 2,
      title: `${analysisData.benchmark_symbol} %`,
    });

    // Set data
    const algoReturnData: LineData<Time>[] = analysisData.equity_curve.map((p) => ({
      time: p.date as Time,
      value: p.cumulative_return_pct,
    }));

    const benchmarkReturnData: LineData<Time>[] = analysisData.equity_curve.map((p) => ({
      time: p.date as Time,
      value: p.benchmark_return_pct,
    }));

    algoReturnSeries.setData(algoReturnData);
    benchmarkReturnSeries.setData(benchmarkReturnData);

    chart.timeScale().fitContent();

    const handleResize = () => {
      if (returnChartRef.current && returnChartApiRef.current) {
        returnChartApiRef.current.applyOptions({
          width: returnChartRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [analysisData]);

  // Clean up charts on unmount
  useEffect(() => {
    return () => {
      if (equityChartApiRef.current) {
        equityChartApiRef.current.remove();
      }
      if (returnChartApiRef.current) {
        returnChartApiRef.current.remove();
      }
    };
  }, []);

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

  return (
    <PageContainer>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <PageHeader
          title="Live Analysis"
          subtitle={`Performance analytics vs ${analysisData.benchmark_symbol}`}
        />
        <div className="flex items-center gap-4">
          <span className="text-xs text-text-400">
            Last update: {lastUpdate.toLocaleTimeString()}
          </span>
          <Button variant="secondary" onClick={() => router.push('/live')}>
            {'\u2190'} Back to Live
          </Button>
        </div>
      </div>

      {/* Summary Stats */}
      <StatsGrid columns={4} className="mb-6">
        <Card>
          <span className="text-sm text-text-400 block mb-1">Total Return</span>
          <PLDisplay value={totalPnl} percent={totalReturn} size="lg" />
        </Card>

        <Card>
          <span className="text-sm text-text-400 block mb-1">Current Equity</span>
          <span className="text-2xl font-bold text-text-900 font-mono tabular-nums">
            {formatCurrency(analysisData.current_equity)}
          </span>
          <span className="text-xs text-text-300 block mt-1">
            Initial: {formatCurrency(analysisData.initial_capital)}
          </span>
        </Card>

        <Card>
          <span className="text-sm text-text-400 block mb-1">Alpha vs {analysisData.benchmark_symbol}</span>
          <span className={`text-2xl font-bold font-mono tabular-nums ${
            alpha > 0 ? 'text-profit-500' : alpha < 0 ? 'text-loss-500' : 'text-text-700'
          }`}>
            {alpha >= 0 ? '+' : ''}{alpha.toFixed(2)}%
          </span>
          <span className="text-xs text-text-300 block mt-1">
            Benchmark: {formatPercent(benchmarkReturn, true)}
          </span>
        </Card>

        <Card>
          <span className="text-sm text-text-400 block mb-1">Trading Days</span>
          <span className="text-2xl font-bold text-text-900 font-mono tabular-nums">
            {analysisData.trading_days}
          </span>
        </Card>
      </StatsGrid>

      {/* Risk Metrics */}
      <h2 className="text-xl font-semibold text-text-900 mb-4">Risk Metrics</h2>
      <StatsGrid columns={5} className="mb-6">
        <Card>
          <span className="text-sm text-text-400 block mb-1">Sharpe Ratio</span>
          <span className={`text-xl font-bold font-mono tabular-nums ${
            risk_metrics.sharpe_ratio >= 1 ? 'text-profit-500' :
            risk_metrics.sharpe_ratio >= 0.5 ? 'text-warning-500' :
            'text-loss-500'
          }`}>
            {risk_metrics.sharpe_ratio.toFixed(2)}
          </span>
        </Card>

        <Card>
          <span className="text-sm text-text-400 block mb-1">Sortino Ratio</span>
          <span className={`text-xl font-bold font-mono tabular-nums ${
            risk_metrics.sortino_ratio >= 1.5 ? 'text-profit-500' :
            risk_metrics.sortino_ratio >= 1 ? 'text-warning-500' :
            'text-loss-500'
          }`}>
            {risk_metrics.sortino_ratio.toFixed(2)}
          </span>
        </Card>

        <Card>
          <span className="text-sm text-text-400 block mb-1">Calmar Ratio</span>
          <span className={`text-xl font-bold font-mono tabular-nums ${
            risk_metrics.calmar_ratio >= 1 ? 'text-profit-500' :
            risk_metrics.calmar_ratio >= 0.5 ? 'text-warning-500' :
            'text-loss-500'
          }`}>
            {risk_metrics.calmar_ratio.toFixed(2)}
          </span>
        </Card>

        <Card>
          <span className="text-sm text-text-400 block mb-1">Max Drawdown</span>
          <span className="text-xl font-bold text-loss-500 font-mono tabular-nums">
            -{risk_metrics.max_drawdown_pct.toFixed(1)}%
          </span>
          <span className="text-xs text-text-300 block mt-1">
            {formatCurrency(risk_metrics.max_drawdown)}
          </span>
        </Card>

        <Card>
          <span className="text-sm text-text-400 block mb-1">Win Rate</span>
          <span className={`text-xl font-bold font-mono tabular-nums ${
            risk_metrics.win_rate >= 55 ? 'text-profit-500' :
            risk_metrics.win_rate >= 45 ? 'text-warning-500' :
            'text-loss-500'
          }`}>
            {risk_metrics.win_rate.toFixed(1)}%
          </span>
        </Card>
      </StatsGrid>

      {/* Secondary metrics */}
      <StatsGrid columns={3} className="mb-8">
        <Card>
          <span className="text-sm text-text-400 block mb-1">Profit Factor</span>
          <span className={`text-xl font-bold font-mono tabular-nums ${
            risk_metrics.profit_factor >= 1.5 ? 'text-profit-500' :
            risk_metrics.profit_factor >= 1 ? 'text-warning-500' :
            'text-loss-500'
          }`}>
            {risk_metrics.profit_factor.toFixed(2)}
          </span>
        </Card>

        <Card>
          <span className="text-sm text-text-400 block mb-1">Avg Win</span>
          <span className="text-xl font-bold text-profit-500 font-mono tabular-nums">
            {formatCurrency(risk_metrics.avg_win)}
          </span>
        </Card>

        <Card>
          <span className="text-sm text-text-400 block mb-1">Avg Loss</span>
          <span className="text-xl font-bold text-loss-500 font-mono tabular-nums">
            {formatCurrency(risk_metrics.avg_loss)}
          </span>
        </Card>
      </StatsGrid>

      {/* Equity Curve Chart */}
      <Card noPadding className="mb-6 overflow-hidden">
        <div className="p-4 border-b border-panel-500 flex items-center justify-between">
          <CardTitle>Equity Curve</CardTitle>
          <div className="flex items-center gap-4 text-xs text-text-400">
            <div className="flex items-center gap-1">
              <div className="w-3 h-0.5 bg-accent-500" />
              <span>Strategy</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-0.5 bg-gray-500" style={{ borderTop: '1px dashed' }} />
              <span>{analysisData.benchmark_symbol}</span>
            </div>
          </div>
        </div>
        <div ref={equityChartRef} className="bg-panel-700" />
      </Card>

      {/* Return Comparison Chart */}
      <Card noPadding className="mb-6 overflow-hidden">
        <div className="p-4 border-b border-panel-500 flex items-center justify-between">
          <CardTitle>Cumulative Returns (%)</CardTitle>
          <div className="flex items-center gap-4 text-xs text-text-400">
            <div className="flex items-center gap-1">
              <div className="w-3 h-0.5 bg-profit-500" />
              <span>Strategy</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-0.5 bg-gray-500" style={{ borderTop: '1px dashed' }} />
              <span>{analysisData.benchmark_symbol}</span>
            </div>
          </div>
        </div>
        <div ref={returnChartRef} className="bg-panel-700" />
      </Card>

      {/* Daily Breakdown Table */}
      <Card noPadding>
        <div className="p-4 border-b border-panel-500">
          <CardTitle>Daily Breakdown</CardTitle>
        </div>
        <div className="overflow-x-auto max-h-96">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Date</TableHead>
                <TableHead align="right">Daily P&L</TableHead>
                <TableHead align="right">Return %</TableHead>
                <TableHead align="right">Trades</TableHead>
                <TableHead align="right">Cumulative P&L</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {daily_breakdown.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={5} className="text-center py-8 text-text-400">
                    No trading data available
                  </TableCell>
                </TableRow>
              ) : (
                daily_breakdown.slice().reverse().map((day) => (
                  <TableRow key={day.date}>
                    <TableCell className="font-medium text-text-700">
                      {new Date(day.date).toLocaleDateString('en-US', {
                        weekday: 'short',
                        month: 'short',
                        day: 'numeric',
                      })}
                    </TableCell>
                    <TableCell
                      align="right"
                      className={`font-mono tabular-nums ${
                        day.pnl > 0 ? 'text-profit-500' :
                        day.pnl < 0 ? 'text-loss-500' :
                        'text-text-400'
                      }`}
                    >
                      {formatCurrency(day.pnl, true)}
                    </TableCell>
                    <TableCell
                      align="right"
                      className={`font-mono tabular-nums ${
                        day.return_pct > 0 ? 'text-profit-500' :
                        day.return_pct < 0 ? 'text-loss-500' :
                        'text-text-400'
                      }`}
                    >
                      {formatPercent(day.return_pct, true)}
                    </TableCell>
                    <TableCell align="right" className="font-mono tabular-nums">
                      {day.trade_count}
                    </TableCell>
                    <TableCell
                      align="right"
                      className={`font-mono tabular-nums font-semibold ${
                        day.cumulative_pnl > 0 ? 'text-profit-500' :
                        day.cumulative_pnl < 0 ? 'text-loss-500' :
                        'text-text-400'
                      }`}
                    >
                      {formatCurrency(day.cumulative_pnl, true)}
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </div>
      </Card>
    </PageContainer>
  );
}
