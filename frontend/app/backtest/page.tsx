'use client';

import React, { useState } from 'react';
import { PageContainer, PageHeader, StatsGrid } from '../../components/layout';
import { Card, CardTitle, Button, Badge, Select, Input, Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from '../../components/ui';
import { PLDisplay } from '../../components/trading';
import { formatCurrency, formatPercent } from '../../lib/utils';

interface BacktestConfig {
  symbol: string;
  startDate: string;
  endDate: string;
  initialCapital: number;
  commission: number;
  slippage: number;
  emaPeriod: number;
  rsiPeriod: number;
  atrPeriod: number;
  kamaPeriod: number;
  maxPositionSize: number;
  stopLossPct: number;
}

interface BacktestResult {
  totalReturn: number;
  totalReturnPct: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  totalTrades: number;
  avgWin: number;
  avgLoss: number;
  profitFactor: number;
  equity_curve: number[];
  trade_log: Array<{
    entry_date: string;
    exit_date: string;
    symbol: string;
    side: string;
    entry_price: number;
    exit_price: number;
    shares: number;
    pnl: number;
  }>;
}

const symbolOptions = [
  { value: 'SPY', label: 'SPY - S&P 500 ETF' },
  { value: 'QQQ', label: 'QQQ - Nasdaq 100 ETF' },
  { value: 'AAPL', label: 'AAPL - Apple Inc.' },
  { value: 'TSLA', label: 'TSLA - Tesla Inc.' },
  { value: 'MSFT', label: 'MSFT - Microsoft Corp.' },
  { value: 'NVDA', label: 'NVDA - NVIDIA Corp.' },
  { value: 'AMZN', label: 'AMZN - Amazon.com Inc.' },
];

export default function BacktestPage() {
  const [config, setConfig] = useState<BacktestConfig>({
    symbol: 'SPY',
    startDate: '2023-01-01',
    endDate: '2024-01-01',
    initialCapital: 100000,
    commission: 0.005,
    slippage: 0.01,
    emaPeriod: 20,
    rsiPeriod: 14,
    atrPeriod: 14,
    kamaPeriod: 10,
    maxPositionSize: 20,
    stopLossPct: 3.0,
  });

  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [results, setResults] = useState<BacktestResult | null>(null);
  const [showResults, setShowResults] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleConfigChange = (field: keyof BacktestConfig, value: string | number) => {
    setConfig((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const runBacktest = async () => {
    setIsRunning(true);
    setError(null);
    setResults(null);

    try {
      const response = await fetch('/api/backtest', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
      });

      if (!response.ok) {
        throw new Error(`Backtest failed: ${response.statusText}`);
      }

      const data: BacktestResult = await response.json();
      setResults(data);
      setShowResults(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
      console.error('Backtest error:', err);
    } finally {
      setIsRunning(false);
    }
  };

  const closeResults = () => {
    setShowResults(false);
  };

  const exportToCSV = () => {
    if (!results || !results.trade_log) return;

    const headers = ['Entry Date', 'Exit Date', 'Symbol', 'Side', 'Entry Price', 'Exit Price', 'Shares', 'P&L'];
    const rows = results.trade_log.map((trade) => [
      trade.entry_date,
      trade.exit_date,
      trade.symbol,
      trade.side,
      trade.entry_price.toFixed(2),
      trade.exit_price.toFixed(2),
      trade.shares.toString(),
      trade.pnl.toFixed(2),
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
          <Select
            label="Symbol"
            options={symbolOptions}
            value={config.symbol}
            onChange={(e) => handleConfigChange('symbol', e.target.value)}
          />
          <Input
            label="Start Date"
            type="date"
            value={config.startDate}
            onChange={(e) => handleConfigChange('startDate', e.target.value)}
          />
          <Input
            label="End Date"
            type="date"
            value={config.endDate}
            onChange={(e) => handleConfigChange('endDate', e.target.value)}
          />
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
            value={config.commission}
            onChange={(e) => handleConfigChange('commission', parseFloat(e.target.value))}
            min={0}
            step={0.001}
          />
          <Input
            label="Slippage (%)"
            type="number"
            value={config.slippage}
            onChange={(e) => handleConfigChange('slippage', parseFloat(e.target.value))}
            min={0}
            step={0.01}
          />
        </div>

        {/* Strategy Parameters */}
        <h3 className="text-lg font-semibold text-text-900 mt-8 mb-4">Strategy Parameters</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
          <SliderInput
            label="EMA Period"
            value={config.emaPeriod}
            onChange={(val) => handleConfigChange('emaPeriod', val)}
            min={5}
            max={50}
          />
          <SliderInput
            label="RSI Period"
            value={config.rsiPeriod}
            onChange={(val) => handleConfigChange('rsiPeriod', val)}
            min={5}
            max={30}
          />
          <SliderInput
            label="ATR Period"
            value={config.atrPeriod}
            onChange={(val) => handleConfigChange('atrPeriod', val)}
            min={5}
            max={30}
          />
          <SliderInput
            label="KAMA Period"
            value={config.kamaPeriod}
            onChange={(val) => handleConfigChange('kamaPeriod', val)}
            min={5}
            max={20}
          />
        </div>

        {/* Risk Parameters */}
        <h3 className="text-lg font-semibold text-text-900 mt-8 mb-4">Risk Parameters</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          <SliderInput
            label="Max Position Size"
            value={config.maxPositionSize}
            onChange={(val) => handleConfigChange('maxPositionSize', val)}
            min={5}
            max={50}
            suffix="%"
          />
          <SliderInput
            label="Stop Loss"
            value={config.stopLossPct}
            onChange={(val) => handleConfigChange('stopLossPct', val)}
            min={1}
            max={10}
            step={0.5}
            suffix="%"
            decimals={1}
          />
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
                  value={<PLDisplay value={results.totalReturn} percent={results.totalReturnPct} size="lg" />}
                />
                <MetricCard
                  label="Sharpe Ratio"
                  value={results.sharpeRatio.toFixed(2)}
                  valueClass={
                    results.sharpeRatio > 0.8 ? 'text-profit-500' :
                    results.sharpeRatio > 0.5 ? 'text-warning-500' :
                    'text-loss-500'
                  }
                  hint={
                    results.sharpeRatio > 0.8 ? 'Excellent' :
                    results.sharpeRatio > 0.5 ? 'Good' : 'Poor'
                  }
                />
                <MetricCard
                  label="Max Drawdown"
                  value={`${results.maxDrawdown.toFixed(2)}%`}
                  valueClass={
                    results.maxDrawdown < 15 ? 'text-profit-500' :
                    results.maxDrawdown < 25 ? 'text-warning-500' :
                    'text-loss-500'
                  }
                  hint={
                    results.maxDrawdown < 15 ? 'Low Risk' :
                    results.maxDrawdown < 25 ? 'Moderate' : 'High Risk'
                  }
                />
                <MetricCard
                  label="Win Rate"
                  value={`${results.winRate.toFixed(1)}%`}
                  valueClass={
                    results.winRate >= 55 ? 'text-profit-500' :
                    results.winRate >= 45 ? 'text-warning-500' :
                    'text-loss-500'
                  }
                  hint={`${results.totalTrades} trades`}
                />
                <MetricCard
                  label="Profit Factor"
                  value={results.profitFactor.toFixed(2)}
                  valueClass={
                    results.profitFactor >= 2 ? 'text-profit-500' :
                    results.profitFactor >= 1.5 ? 'text-warning-500' :
                    'text-loss-500'
                  }
                  hint={`Avg Win: ${formatCurrency(results.avgWin)}`}
                />
              </div>

              {/* Success Criteria Check */}
              <Card className="mb-6">
                <h3 className="text-lg font-semibold text-text-900 mb-4">Success Criteria</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <CriteriaCheck
                    label="Sharpe > 0.8"
                    passed={results.sharpeRatio > 0.8}
                  />
                  <CriteriaCheck
                    label="Max DD < 25%"
                    passed={results.maxDrawdown < 25}
                  />
                  <CriteriaCheck
                    label="Win Rate > 45%"
                    passed={results.winRate > 45}
                  />
                </div>
              </Card>

              {/* Export Button */}
              <div className="mb-6">
                <Button variant="primary" onClick={exportToCSV}>
                  Export Trade Log (CSV)
                </Button>
              </div>

              {/* Trade Log Table */}
              <div>
                <h3 className="text-lg font-semibold text-text-900 mb-4">Trade Log (Last 20 trades)</h3>
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Entry Date</TableHead>
                        <TableHead>Exit Date</TableHead>
                        <TableHead>Side</TableHead>
                        <TableHead align="right">Entry</TableHead>
                        <TableHead align="right">Exit</TableHead>
                        <TableHead align="right">Shares</TableHead>
                        <TableHead align="right">P&L</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {results.trade_log.slice(-20).reverse().map((trade, idx) => (
                        <TableRow key={idx}>
                          <TableCell>{trade.entry_date}</TableCell>
                          <TableCell>{trade.exit_date}</TableCell>
                          <TableCell>
                            <Badge
                              variant={trade.side === 'LONG' ? 'success' : 'error'}
                              size="sm"
                            >
                              {trade.side}
                            </Badge>
                          </TableCell>
                          <TableCell align="right" className="font-mono tabular-nums">
                            {formatCurrency(trade.entry_price)}
                          </TableCell>
                          <TableCell align="right" className="font-mono tabular-nums">
                            {formatCurrency(trade.exit_price)}
                          </TableCell>
                          <TableCell align="right" className="font-mono tabular-nums">
                            {trade.shares}
                          </TableCell>
                          <TableCell
                            align="right"
                            className={`font-mono tabular-nums font-semibold ${
                              trade.pnl >= 0 ? 'text-profit-500' : 'text-loss-500'
                            }`}
                          >
                            {formatCurrency(trade.pnl)}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
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
