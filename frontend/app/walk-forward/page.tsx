'use client';

import React, { useState, useRef, useEffect } from 'react';
import { createChart, IChartApi, LineSeries, Time } from 'lightweight-charts';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';
import { PageContainer, PageHeader } from '../../components/layout';
import {
  Card,
  CardTitle,
  Button,
  Badge,
  Select,
  Input,
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from '../../components/ui';
import { PLDisplay, SymbolSearch } from '../../components/trading';
import { formatCurrency, formatPercent } from '../../lib/utils';
import {
  apiClient,
  ApiError,
  WalkForwardRequest,
  WalkForwardResponse,
  WalkForwardWindowMetrics,
} from '../../utils/api';

interface WalkForwardFormConfig {
  symbol: string;
  startDate: string;
  endDate: string;
  initialCapital: number;
  commissionPerShare: number;
  slippagePct: number;
  trainBars: number;
  testBars: number;
  strategyMode: 'TREND' | 'MEAN_REVERSION' | 'DUAL';
  passThreshold: number;
}

const strategyModeOptions = [
  { value: 'DUAL', label: 'DUAL - Adaptive (Recommended)' },
  { value: 'TREND', label: 'TREND - Trend Following' },
  { value: 'MEAN_REVERSION', label: 'MEAN_REVERSION - Mean Reversion' },
];

export default function WalkForwardPage() {
  const [config, setConfig] = useState<WalkForwardFormConfig>({
    symbol: 'SPY',
    startDate: '2022-01-01',
    endDate: '2024-01-01',
    initialCapital: 100000,
    commissionPerShare: 0.005,
    slippagePct: 0.01,
    trainBars: 63,
    testBars: 21,
    strategyMode: 'DUAL',
    passThreshold: 0.6,
  });

  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [results, setResults] = useState<WalkForwardResponse | null>(null);
  const [showResults, setShowResults] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleConfigChange = (
    field: keyof WalkForwardFormConfig,
    value: string | number
  ) => {
    if (typeof value === 'number' && isNaN(value)) {
      return;
    }
    setConfig((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const runWalkForward = async () => {
    setIsRunning(true);
    setError(null);
    setResults(null);

    if (!config.symbol.trim()) {
      setError('Please enter a stock symbol');
      setIsRunning(false);
      return;
    }

    if (!config.startDate || !config.endDate) {
      setError('Please select both start and end dates');
      setIsRunning(false);
      return;
    }

    try {
      const request: WalkForwardRequest = {
        symbol: config.symbol.toUpperCase().trim(),
        start_date: config.startDate,
        end_date: config.endDate,
        initial_capital: config.initialCapital,
        commission_per_share: config.commissionPerShare,
        slippage_pct: config.slippagePct,
        train_bars: config.trainBars,
        test_bars: config.testBars,
        strategy_mode: config.strategyMode,
        pass_threshold: config.passThreshold,
      };

      const data = await apiClient.runWalkForwardBacktest(request);
      setResults(data);
      setShowResults(true);
    } catch (err) {
      if (err instanceof ApiError) {
        if (err.status === 404) {
          setError(
            `Symbol "${config.symbol.toUpperCase()}" not found. Please check the ticker is correct.`
          );
        } else if (err.status === 400) {
          setError(err.detail || 'Invalid request. Please check your inputs.');
        } else {
          setError(err.detail || `Server error (${err.status})`);
        }
      } else {
        setError(
          err instanceof Error ? err.message : 'An unknown error occurred'
        );
      }
      console.error('Walk-forward error:', err);
    } finally {
      setIsRunning(false);
    }
  };

  const closeResults = () => {
    setShowResults(false);
  };

  const exportToCSV = () => {
    if (!results || !results.window_results) return;

    const headers = [
      'Window ID',
      'Test Start',
      'Test End',
      'Initial Equity',
      'Final Equity',
      'Return %',
      'Sharpe',
      'Max DD %',
      'Win Rate',
      'Trades',
      'Profitable',
    ];
    const rows = results.window_results.map((w) => [
      w.window_id,
      w.test_start_date || '',
      w.test_end_date || '',
      w.initial_equity.toFixed(2),
      w.final_equity.toFixed(2),
      w.return_pct.toFixed(2),
      w.sharpe_ratio.toFixed(2),
      w.max_drawdown_pct.toFixed(2),
      (w.win_rate * 100).toFixed(1),
      w.num_trades,
      w.is_profitable ? 'Yes' : 'No',
    ]);

    const csvContent = [headers, ...rows].map((row) => row.join(',')).join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `walkforward_${config.symbol}_${config.startDate}_${config.endDate}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <PageContainer>
      <PageHeader
        title="Walk-Forward Testing"
        subtitle="Validate strategy robustness with out-of-sample testing"
      />

      {/* Configuration Form */}
      <Card className="mb-6">
        <CardTitle className="mb-6">Walk-Forward Configuration</CardTitle>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          <SymbolSearch
            value={config.symbol}
            onChange={(symbol) => handleConfigChange('symbol', symbol)}
            placeholder="Search stocks (e.g., Apple, AAPL)"
            disabled={isRunning}
          />
          <div>
            <label className="block text-sm font-medium text-text-600 mb-2">
              Start Date
            </label>
            <DatePicker
              selected={config.startDate ? new Date(config.startDate) : null}
              onChange={(date: Date | null) => {
                if (date) {
                  const formatted = date.toISOString().split('T')[0];
                  handleConfigChange('startDate', formatted);
                }
              }}
              dateFormat="yyyy-MM-dd"
              showMonthDropdown
              showYearDropdown
              dropdownMode="select"
              className="w-full bg-panel-300 text-text-800 rounded px-4 py-3 border-none focus:outline-none focus:ring-2 focus:ring-accent-500"
              calendarClassName="dark-datepicker"
              maxDate={new Date()}
              placeholderText="Select start date"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-text-600 mb-2">
              End Date
            </label>
            <DatePicker
              selected={config.endDate ? new Date(config.endDate) : null}
              onChange={(date: Date | null) => {
                if (date) {
                  const formatted = date.toISOString().split('T')[0];
                  handleConfigChange('endDate', formatted);
                }
              }}
              dateFormat="yyyy-MM-dd"
              showMonthDropdown
              showYearDropdown
              dropdownMode="select"
              className="w-full bg-panel-300 text-text-800 rounded px-4 py-3 border-none focus:outline-none focus:ring-2 focus:ring-accent-500"
              calendarClassName="dark-datepicker"
              maxDate={new Date()}
              minDate={config.startDate ? new Date(config.startDate) : undefined}
              placeholderText="Select end date"
            />
          </div>
          <Input
            label="Initial Capital ($)"
            type="number"
            value={config.initialCapital}
            onChange={(e) =>
              handleConfigChange('initialCapital', parseFloat(e.target.value))
            }
            min={1000}
          />
          <Input
            label="Commission ($/share)"
            type="number"
            value={config.commissionPerShare}
            onChange={(e) =>
              handleConfigChange('commissionPerShare', parseFloat(e.target.value))
            }
            min={0}
            step={0.001}
          />
          <Input
            label="Slippage (%)"
            type="number"
            value={config.slippagePct * 100}
            onChange={(e) =>
              handleConfigChange('slippagePct', parseFloat(e.target.value) / 100)
            }
            min={0}
            step={0.01}
          />
        </div>

        {/* Walk-Forward Parameters */}
        <h3 className="text-lg font-semibold text-text-900 mt-8 mb-4">
          Walk-Forward Parameters
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
          <Input
            label="Training Period (bars)"
            type="number"
            value={config.trainBars}
            onChange={(e) =>
              handleConfigChange('trainBars', parseInt(e.target.value))
            }
            min={21}
          />
          <Input
            label="Test Period (bars)"
            type="number"
            value={config.testBars}
            onChange={(e) =>
              handleConfigChange('testBars', parseInt(e.target.value))
            }
            min={5}
          />
          <Input
            label="Pass Threshold (%)"
            type="number"
            value={config.passThreshold * 100}
            onChange={(e) =>
              handleConfigChange('passThreshold', parseFloat(e.target.value) / 100)
            }
            min={0}
            max={100}
            step={5}
          />
          <Select
            label="Strategy Mode"
            options={strategyModeOptions}
            value={config.strategyMode}
            onChange={(e) =>
              handleConfigChange(
                'strategyMode',
                e.target.value as 'TREND' | 'MEAN_REVERSION' | 'DUAL'
              )
            }
          />
        </div>

        {/* Info Box */}
        <div className="mt-6 p-4 bg-panel-600 rounded-xl">
          <p className="text-sm text-text-400">
            <strong className="text-text-700">Walk-Forward Testing:</strong>{' '}
            Divides data into rolling train/test windows. Strategy is validated
            on each test window separately. Default: 63-bar (~3 month) training,
            21-bar (~1 month) testing. Strategy passes if &gt;{(config.passThreshold * 100).toFixed(0)}%
            of test windows are profitable.
          </p>
        </div>

        {/* Run Button */}
        <div className="mt-8 flex justify-center">
          <Button
            variant="primary"
            size="lg"
            onClick={runWalkForward}
            disabled={isRunning}
            className="min-w-48"
          >
            {isRunning ? 'Running Walk-Forward...' : 'Run Walk-Forward Test'}
          </Button>
        </div>

        {/* Error Display */}
        {error && (
          <Card variant="highlighted" className="mt-6 border-l-4 border-loss-500">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-loss-500 font-medium">Error</p>
                <p className="text-text-400 text-sm">{error}</p>
              </div>
              <Button variant="danger" onClick={runWalkForward} disabled={isRunning}>
                Retry
              </Button>
            </div>
          </Card>
        )}
      </Card>

      {/* Results Modal */}
      {showResults && results && (
        <div className="fixed inset-0 bg-panel-900/90 flex items-center justify-center z-50 p-5">
          <div className="bg-panel-700 rounded-2xl max-w-6xl w-full max-h-[90vh] overflow-hidden flex flex-col">
            {/* Modal Header */}
            <div className="px-6 py-5 flex justify-between items-center border-b border-panel-500">
              <h2 className="text-xl font-bold text-text-900">
                Walk-Forward Results
              </h2>
              <button
                onClick={closeResults}
                className="text-text-400 hover:text-text-900 text-2xl font-bold w-8 h-8 flex items-center justify-center"
              >
                ×
              </button>
            </div>

            {/* Modal Content */}
            <div className="p-6 overflow-y-auto flex-1">
              {/* Pass/Fail Status Banner */}
              <div
                className={`mb-6 p-4 rounded-xl flex items-center justify-between ${
                  results.passes_walk_forward_test
                    ? 'bg-profit-500/20 border border-profit-500'
                    : 'bg-loss-500/20 border border-loss-500'
                }`}
              >
                <div className="flex items-center gap-3">
                  <div
                    className={`w-12 h-12 rounded-full flex items-center justify-center ${
                      results.passes_walk_forward_test
                        ? 'bg-profit-500'
                        : 'bg-loss-500'
                    }`}
                  >
                    <span className="text-2xl">
                      {results.passes_walk_forward_test ? '✓' : '✗'}
                    </span>
                  </div>
                  <div>
                    <div
                      className={`text-xl font-bold ${
                        results.passes_walk_forward_test
                          ? 'text-profit-500'
                          : 'text-loss-500'
                      }`}
                    >
                      {results.passes_walk_forward_test
                        ? 'STRATEGY PASSED'
                        : 'STRATEGY FAILED'}
                    </div>
                    <div className="text-sm text-text-400">
                      Pass Rate: {(results.pass_rate * 100).toFixed(1)}% (Required:{' '}
                      {'>'}{(results.pass_threshold * 100).toFixed(0)}%)
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-sm text-text-400">Windows</div>
                  <div className="text-lg font-bold text-text-900">
                    {results.profitable_windows} / {results.total_windows}{' '}
                    profitable
                  </div>
                </div>
              </div>

              {/* Summary Metrics */}
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4 mb-6">
                <MetricCard
                  label="Total Return"
                  value={
                    <PLDisplay
                      value={results.final_capital - results.initial_capital}
                      percent={results.total_return_pct}
                      size="lg"
                    />
                  }
                />
                <MetricCard
                  label="Aggregate Sharpe"
                  value={results.aggregate_sharpe.toFixed(2)}
                  valueClass={
                    results.aggregate_sharpe > 0.8
                      ? 'text-profit-500'
                      : results.aggregate_sharpe > 0.5
                        ? 'text-warning-500'
                        : 'text-loss-500'
                  }
                  hint={
                    results.aggregate_sharpe > 0.8
                      ? 'Excellent'
                      : results.aggregate_sharpe > 0.5
                        ? 'Good'
                        : 'Poor'
                  }
                />
                <MetricCard
                  label="Max Drawdown"
                  value={`${results.aggregate_max_drawdown_pct.toFixed(2)}%`}
                  valueClass={
                    results.aggregate_max_drawdown_pct < 15
                      ? 'text-profit-500'
                      : results.aggregate_max_drawdown_pct < 25
                        ? 'text-warning-500'
                        : 'text-loss-500'
                  }
                  hint={
                    results.aggregate_max_drawdown_pct < 15
                      ? 'Low Risk'
                      : results.aggregate_max_drawdown_pct < 25
                        ? 'Moderate'
                        : 'High Risk'
                  }
                />
                <MetricCard
                  label="Aggregate Win Rate"
                  value={`${(results.aggregate_win_rate * 100).toFixed(1)}%`}
                  valueClass={
                    results.aggregate_win_rate >= 0.55
                      ? 'text-profit-500'
                      : results.aggregate_win_rate >= 0.45
                        ? 'text-warning-500'
                        : 'text-loss-500'
                  }
                  hint={`${results.total_trades} total trades`}
                />
                <MetricCard
                  label="Final Capital"
                  value={formatCurrency(results.final_capital)}
                  valueClass={
                    results.final_capital >= results.initial_capital
                      ? 'text-profit-500'
                      : 'text-loss-500'
                  }
                  hint={`From ${formatCurrency(results.initial_capital)}`}
                />
              </div>

              {/* Combined Equity Curve Chart */}
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-text-900 mb-4">
                  Combined Equity Curve
                </h3>
                <EquityCurveChart
                  equityCurve={results.combined_equity_curve}
                  timestamps={results.timestamps}
                  initialCapital={results.initial_capital}
                />
              </div>

              {/* Per-Window Results Table */}
              <div className="mb-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-text-900">
                    Per-Window Results
                  </h3>
                  <Button variant="secondary" onClick={exportToCSV}>
                    Export CSV
                  </Button>
                </div>
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Window</TableHead>
                        <TableHead>Test Period</TableHead>
                        <TableHead className="text-right">Return</TableHead>
                        <TableHead className="text-right">Sharpe</TableHead>
                        <TableHead className="text-right">Max DD</TableHead>
                        <TableHead className="text-right">Win Rate</TableHead>
                        <TableHead className="text-right">Trades</TableHead>
                        <TableHead className="text-center">Status</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {results.window_results.map((window) => (
                        <WindowResultRow key={window.window_id} window={window} />
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </div>

              {/* Configuration Summary */}
              <div>
                <h3 className="text-lg font-semibold text-text-900 mb-4">
                  Test Configuration
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-panel-600 rounded-xl p-4">
                    <div className="text-sm text-text-400 mb-1">Symbol</div>
                    <div className="text-xl font-bold text-text-900 font-mono">
                      {results.symbol}
                    </div>
                  </div>
                  <div className="bg-panel-600 rounded-xl p-4">
                    <div className="text-sm text-text-400 mb-1">Date Range</div>
                    <div className="text-sm font-bold text-text-900">
                      {results.start_date} to {results.end_date}
                    </div>
                  </div>
                  <div className="bg-panel-600 rounded-xl p-4">
                    <div className="text-sm text-text-400 mb-1">Train/Test</div>
                    <div className="text-xl font-bold text-text-900 font-mono">
                      {results.train_bars} / {results.test_bars} bars
                    </div>
                  </div>
                  <div className="bg-panel-600 rounded-xl p-4">
                    <div className="text-sm text-text-400 mb-1">Pass Threshold</div>
                    <div className="text-xl font-bold text-text-900 font-mono">
                      {'>'}{(results.pass_threshold * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Modal Footer */}
            <div className="px-6 py-4 border-t border-panel-500 flex justify-end">
              <Button variant="secondary" onClick={closeResults}>
                Close
              </Button>
            </div>
          </div>
        </div>
      )}
    </PageContainer>
  );
}

// Metric Card Component
interface MetricCardProps {
  label: string;
  value: React.ReactNode;
  valueClass?: string;
  hint?: string;
}

function MetricCard({ label, value, valueClass = '', hint }: MetricCardProps) {
  return (
    <div className="bg-panel-600 rounded-xl p-4">
      <div className="text-sm text-text-400 mb-1">{label}</div>
      {typeof value === 'string' || typeof value === 'number' ? (
        <div className={`text-2xl font-bold font-mono tabular-nums ${valueClass}`}>
          {value}
        </div>
      ) : (
        value
      )}
      {hint && <div className="text-xs text-text-300 mt-1">{hint}</div>}
    </div>
  );
}

// Window Result Row Component
interface WindowResultRowProps {
  window: WalkForwardWindowMetrics;
}

function WindowResultRow({ window }: WindowResultRowProps) {
  return (
    <TableRow>
      <TableCell className="font-mono">{window.window_id + 1}</TableCell>
      <TableCell className="text-sm">
        {window.test_start_date || 'N/A'} - {window.test_end_date || 'N/A'}
      </TableCell>
      <TableCell
        className={`text-right font-mono ${
          window.return_pct >= 0 ? 'text-profit-500' : 'text-loss-500'
        }`}
      >
        {window.return_pct >= 0 ? '+' : ''}
        {window.return_pct.toFixed(2)}%
      </TableCell>
      <TableCell
        className={`text-right font-mono ${
          window.sharpe_ratio > 0.5
            ? 'text-profit-500'
            : window.sharpe_ratio > 0
              ? 'text-warning-500'
              : 'text-loss-500'
        }`}
      >
        {window.sharpe_ratio.toFixed(2)}
      </TableCell>
      <TableCell className="text-right font-mono text-loss-500">
        {window.max_drawdown_pct.toFixed(2)}%
      </TableCell>
      <TableCell className="text-right font-mono">
        {(window.win_rate * 100).toFixed(1)}%
      </TableCell>
      <TableCell className="text-right font-mono">{window.num_trades}</TableCell>
      <TableCell className="text-center">
        <Badge variant={window.is_profitable ? 'success' : 'error'}>
          {window.is_profitable ? 'PASS' : 'FAIL'}
        </Badge>
      </TableCell>
    </TableRow>
  );
}

// Equity Curve Chart Component
interface EquityCurveChartProps {
  equityCurve: number[];
  timestamps: string[];
  initialCapital: number;
}

function EquityCurveChart({
  equityCurve,
  timestamps,
  initialCapital,
}: EquityCurveChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current || equityCurve.length === 0) return;

    if (chartRef.current) {
      chartRef.current.remove();
    }

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 350,
      layout: {
        background: { color: '#1C1C28' },
        textColor: '#CCCAD5',
      },
      grid: {
        vertLines: { color: '#21222F' },
        horzLines: { color: '#21222F' },
      },
      crosshair: {
        mode: 1,
      },
      rightPriceScale: {
        borderColor: '#21222F',
      },
      timeScale: {
        borderColor: '#21222F',
        timeVisible: true,
        secondsVisible: false,
      },
    });

    chartRef.current = chart;

    const equitySeries = chart.addSeries(LineSeries, {
      color: '#A549FC',
      lineWidth: 2,
      title: 'Equity',
      priceFormat: {
        type: 'custom',
        formatter: (price: number) =>
          `$${price.toLocaleString(undefined, {
            minimumFractionDigits: 0,
            maximumFractionDigits: 0,
          })}`,
      },
    });

    const baselineSeries = chart.addSeries(LineSeries, {
      color: '#6B6983',
      lineWidth: 1,
      lineStyle: 2,
      title: 'Initial Capital',
      crosshairMarkerVisible: false,
    });

    const chartData = equityCurve.map((value, i) => ({
      time: timestamps[i] as Time,
      value: value,
    }));

    const baselineData = equityCurve.map((_, i) => ({
      time: timestamps[i] as Time,
      value: initialCapital,
    }));

    equitySeries.setData(chartData);
    baselineSeries.setData(baselineData);

    chart.timeScale().fitContent();

    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
    };
  }, [equityCurve, timestamps, initialCapital]);

  return (
    <div className="bg-panel-600 rounded-xl p-4">
      <div ref={chartContainerRef} className="w-full" />
    </div>
  );
}
