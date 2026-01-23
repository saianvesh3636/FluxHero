'use client';

import React, { useState, useRef, useEffect } from 'react';
import { createChart, IChartApi, LineSeries, Time } from 'lightweight-charts';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';
import { PageContainer, PageHeader, StatsGrid } from '../../components/layout';
import { Card, CardTitle, Button, Badge, Select, Input, Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from '../../components/ui';
import { PLDisplay, SymbolSearch } from '../../components/trading';
import { formatCurrency, formatPercent } from '../../lib/utils';
import { apiClient, ApiError, BacktestRequest, BacktestResultResponse } from '../../utils/api';

// Form config (camelCase for UI)
interface BacktestFormConfig {
  symbol: string;
  startDate: string;
  endDate: string;
  initialCapital: number;
  commissionPerShare: number;
  slippagePct: number;
  strategyMode: 'TREND' | 'MEAN_REVERSION' | 'DUAL';
}

const strategyModeOptions = [
  { value: 'DUAL', label: 'DUAL - Adaptive (Recommended)' },
  { value: 'TREND', label: 'TREND - Trend Following' },
  { value: 'MEAN_REVERSION', label: 'MEAN_REVERSION - Mean Reversion' },
];

export default function BacktestPage() {
  const [config, setConfig] = useState<BacktestFormConfig>({
    symbol: 'SPY',
    startDate: '2023-01-01',
    endDate: '2024-01-01',
    initialCapital: 100000,
    commissionPerShare: 0.005,
    slippagePct: 0.01,
    strategyMode: 'DUAL',
  });

  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [results, setResults] = useState<BacktestResultResponse | null>(null);
  const [showResults, setShowResults] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleConfigChange = (field: keyof BacktestFormConfig, value: string | number) => {
    // Prevent NaN values from being stored
    if (typeof value === 'number' && isNaN(value)) {
      return;
    }
    setConfig((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const runBacktest = async () => {
    setIsRunning(true);
    setError(null);
    setResults(null);

    // Validate inputs
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
      // Transform form config to backend request format
      const request: BacktestRequest = {
        symbol: config.symbol.toUpperCase().trim(),
        start_date: config.startDate,
        end_date: config.endDate,
        initial_capital: config.initialCapital,
        commission_per_share: config.commissionPerShare,
        slippage_pct: config.slippagePct,
        strategy_mode: config.strategyMode,
      };

      const data = await apiClient.runBacktest(request);
      setResults(data);
      setShowResults(true);
    } catch (err) {
      // Handle specific error types
      if (err instanceof ApiError) {
        if (err.status === 404) {
          setError(`Symbol "${config.symbol.toUpperCase()}" not found. Please check the ticker is correct.`);
        } else if (err.status === 400) {
          setError(err.detail || 'Invalid request. Please check your inputs.');
        } else {
          setError(err.detail || `Server error (${err.status})`);
        }
      } else {
        setError(err instanceof Error ? err.message : 'An unknown error occurred');
      }
      console.error('Backtest error:', err);
    } finally {
      setIsRunning(false);
    }
  };

  const closeResults = () => {
    setShowResults(false);
  };

  const exportToCSV = () => {
    if (!results || !results.equity_curve) return;

    // Export equity curve data
    const headers = ['Date', 'Equity'];
    const rows = results.timestamps.map((timestamp, i) => [
      timestamp,
      results.equity_curve[i].toFixed(2),
    ]);

    const csvContent = [headers, ...rows].map((row) => row.join(',')).join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `backtest_${config.symbol}_${config.startDate}_${config.endDate}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <PageContainer>
      <PageHeader
        title="Backtesting Module"
        subtitle="Test your strategy against historical data"
      />

      {/* Configuration Form */}
      <Card className="mb-6">
        <CardTitle className="mb-6">Backtest Configuration</CardTitle>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          <SymbolSearch
            value={config.symbol}
            onChange={(symbol) => handleConfigChange('symbol', symbol)}
            placeholder="Search stocks (e.g., Apple, AAPL)"
            disabled={isRunning}
          />
          <div>
            <label className="block text-sm font-medium text-text-600 mb-2">Start Date</label>
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
            <label className="block text-sm font-medium text-text-600 mb-2">End Date</label>
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
            onChange={(e) => handleConfigChange('initialCapital', parseFloat(e.target.value))}
            min={1000}
          />
          <Input
            label="Commission ($/share)"
            type="number"
            value={config.commissionPerShare}
            onChange={(e) => handleConfigChange('commissionPerShare', parseFloat(e.target.value))}
            min={0}
            step={0.001}
          />
          <Input
            label="Slippage (%)"
            type="number"
            value={config.slippagePct * 100}
            onChange={(e) => handleConfigChange('slippagePct', parseFloat(e.target.value) / 100)}
            min={0}
            step={0.01}
          />
        </div>

        {/* Strategy Parameters */}
        <h3 className="text-lg font-semibold text-text-900 mt-8 mb-4">Strategy Parameters</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          <Select
            label="Strategy Mode"
            options={strategyModeOptions}
            value={config.strategyMode}
            onChange={(e) => handleConfigChange('strategyMode', e.target.value as 'TREND' | 'MEAN_REVERSION' | 'DUAL')}
          />
          <div className="flex items-end">
            <p className="text-sm text-text-400 pb-2">
              <strong>DUAL</strong>: Adapts between trend-following and mean-reversion based on market regime.<br />
              <strong>TREND</strong>: Follows momentum signals only.<br />
              <strong>MEAN_REVERSION</strong>: Trades reversals only.
            </p>
          </div>
        </div>

        {/* Run Button */}
        <div className="mt-8 flex justify-center">
          <Button
            variant="primary"
            size="lg"
            onClick={runBacktest}
            disabled={isRunning}
            className="min-w-48"
          >
            {isRunning ? 'Running Backtest...' : 'Run Backtest'}
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
              <Button variant="danger" onClick={runBacktest} disabled={isRunning}>
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
              <h2 className="text-xl font-bold text-text-900">Backtest Results</h2>
              <button
                onClick={closeResults}
                className="text-text-400 hover:text-text-900 text-2xl font-bold w-8 h-8 flex items-center justify-center"
              >
                ×
              </button>
            </div>

            {/* Modal Content */}
            <div className="p-6 overflow-y-auto flex-1">
              {/* Summary Metrics */}
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4 mb-6">
                <MetricCard
                  label="Total Return"
                  value={<PLDisplay value={results.total_return} percent={results.total_return_pct} size="lg" />}
                />
                <MetricCard
                  label="Sharpe Ratio"
                  value={results.sharpe_ratio.toFixed(2)}
                  valueClass={
                    results.sharpe_ratio > 0.8 ? 'text-profit-500' :
                    results.sharpe_ratio > 0.5 ? 'text-warning-500' :
                    'text-loss-500'
                  }
                  hint={
                    results.sharpe_ratio > 0.8 ? 'Excellent' :
                    results.sharpe_ratio > 0.5 ? 'Good' : 'Poor'
                  }
                />
                <MetricCard
                  label="Max Drawdown"
                  value={`${results.max_drawdown_pct.toFixed(2)}%`}
                  valueClass={
                    results.max_drawdown_pct < 15 ? 'text-profit-500' :
                    results.max_drawdown_pct < 25 ? 'text-warning-500' :
                    'text-loss-500'
                  }
                  hint={
                    results.max_drawdown_pct < 15 ? 'Low Risk' :
                    results.max_drawdown_pct < 25 ? 'Moderate' : 'High Risk'
                  }
                />
                <MetricCard
                  label="Win Rate"
                  value={`${(results.win_rate * 100).toFixed(1)}%`}
                  valueClass={
                    results.win_rate >= 0.55 ? 'text-profit-500' :
                    results.win_rate >= 0.45 ? 'text-warning-500' :
                    'text-loss-500'
                  }
                  hint={`${results.num_trades} trades`}
                />
                <MetricCard
                  label="Win/Loss Ratio"
                  value={results.avg_win_loss_ratio.toFixed(2)}
                  valueClass={
                    results.avg_win_loss_ratio >= 2 ? 'text-profit-500' :
                    results.avg_win_loss_ratio >= 1.5 ? 'text-warning-500' :
                    'text-loss-500'
                  }
                  hint={`Final: ${formatCurrency(results.final_equity)}`}
                />
              </div>

              {/* Success Criteria Check */}
              <Card className="mb-6">
                <h3 className="text-lg font-semibold text-text-900 mb-4">Success Criteria</h3>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <CriteriaCheck
                    label="All Criteria Met"
                    passed={results.success_criteria_met}
                  />
                  <CriteriaCheck
                    label="Sharpe > 0.8"
                    passed={results.sharpe_ratio > 0.8}
                  />
                  <CriteriaCheck
                    label="Max DD < 25%"
                    passed={results.max_drawdown_pct < 25}
                  />
                  <CriteriaCheck
                    label="Win Rate > 45%"
                    passed={results.win_rate > 0.45}
                  />
                </div>
              </Card>

              {/* Equity Curve Chart */}
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-text-900 mb-4">Equity Curve</h3>
                <EquityCurveChart
                  equityCurve={results.equity_curve}
                  timestamps={results.timestamps}
                  initialCapital={results.initial_capital}
                />
              </div>

              {/* Export Button */}
              <div className="mb-6">
                <Button variant="primary" onClick={exportToCSV}>
                  Export Equity Curve (CSV)
                </Button>
              </div>

              {/* Equity Curve Summary */}
              <div>
                <h3 className="text-lg font-semibold text-text-900 mb-4">Equity Curve Statistics</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-panel-600 rounded-xl p-4">
                    <div className="text-sm text-text-400 mb-1">Initial Capital</div>
                    <div className="text-xl font-bold text-text-900 font-mono tabular-nums">
                      {formatCurrency(results.initial_capital)}
                    </div>
                  </div>
                  <div className="bg-panel-600 rounded-xl p-4">
                    <div className="text-sm text-text-400 mb-1">Final Equity</div>
                    <div className={`text-xl font-bold font-mono tabular-nums ${
                      results.final_equity >= results.initial_capital ? 'text-profit-500' : 'text-loss-500'
                    }`}>
                      {formatCurrency(results.final_equity)}
                    </div>
                  </div>
                  <div className="bg-panel-600 rounded-xl p-4">
                    <div className="text-sm text-text-400 mb-1">Max Drawdown</div>
                    <div className="text-xl font-bold text-loss-500 font-mono tabular-nums">
                      {formatCurrency(results.max_drawdown)}
                    </div>
                  </div>
                  <div className="bg-panel-600 rounded-xl p-4">
                    <div className="text-sm text-text-400 mb-1">Data Points</div>
                    <div className="text-xl font-bold text-text-900 font-mono tabular-nums">
                      {results.equity_curve.length}
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

// Slider Input Component
interface SliderInputProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step?: number;
  suffix?: string;
  decimals?: number;
}

function SliderInput({ label, value, onChange, min, max, step = 1, suffix = '', decimals = 0 }: SliderInputProps) {
  return (
    <div>
      <label className="block text-sm font-medium text-text-400 mb-2">{label}</label>
      <input
        type="range"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        min={min}
        max={max}
        step={step}
        className="w-full h-2 bg-panel-500 rounded-lg appearance-none cursor-pointer accent-accent-500"
      />
      <div className="text-center text-sm text-text-300 mt-2 font-mono tabular-nums">
        {decimals > 0 ? value.toFixed(decimals) : value}{suffix}
      </div>
    </div>
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
      {hint && (
        <div className="text-xs text-text-300 mt-1">{hint}</div>
      )}
    </div>
  );
}

// Criteria Check Component
interface CriteriaCheckProps {
  label: string;
  passed: boolean;
}

function CriteriaCheck({ label, passed }: CriteriaCheckProps) {
  return (
    <div className="flex items-center gap-3">
      <div className={`w-4 h-4 rounded-full ${passed ? 'bg-profit-500' : 'bg-loss-500'}`} />
      <span className="text-text-700">
        {label}: <span className={passed ? 'text-profit-500' : 'text-loss-500'}>{passed ? 'PASS' : 'FAIL'}</span>
      </span>
    </div>
  );
}

// Equity Curve Chart Component
interface EquityCurveChartProps {
  equityCurve: number[];
  timestamps: string[];
  initialCapital: number;
}

function EquityCurveChart({ equityCurve, timestamps, initialCapital }: EquityCurveChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current || equityCurve.length === 0) return;

    // Clean up previous chart
    if (chartRef.current) {
      chartRef.current.remove();
    }

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 350,
      layout: {
        background: { color: '#1C1C28' }, // panel-700
        textColor: '#CCCAD5', // text-400
      },
      grid: {
        vertLines: { color: '#21222F' }, // panel-500
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

    // Add equity line series
    const equitySeries = chart.addSeries(LineSeries, {
      color: '#A549FC', // accent-500 (purple)
      lineWidth: 2,
      title: 'Equity',
      priceFormat: {
        type: 'custom',
        formatter: (price: number) => `$${price.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`,
      },
    });

    // Add initial capital reference line
    const baselineSeries = chart.addSeries(LineSeries, {
      color: '#6B6983', // text-300 (gray)
      lineWidth: 1,
      lineStyle: 2, // dashed
      title: 'Initial Capital',
      crosshairMarkerVisible: false,
    });

    // Convert data to chart format
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

    // Fit content
    chart.timeScale().fitContent();

    // Handle resize
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
