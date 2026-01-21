'use client';

import React, { useState } from 'react';

// Type definitions
interface BacktestConfig {
  symbol: string;
  startDate: string;
  endDate: string;
  initialCapital: number;
  commission: number;
  slippage: number;
  // Strategy parameters
  emaPeriod: number;
  rsiPeriod: number;
  atrPeriod: number;
  kamaPeriod: number;
  // Risk parameters
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
      // Call backend API
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
    <div className="min-h-screen bg-gray-900 text-white p-6 page-container">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6 page-header">
          <h1 className="text-3xl font-bold mb-2">Backtesting Module</h1>
          <p className="text-gray-400">Test your strategy against historical data</p>
        </div>

        {/* Configuration Form */}
        <div className="bg-gray-800 rounded-lg p-6 mb-6 card card-padding">
          <h2 className="text-xl font-bold mb-4">Backtest Configuration</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 single-col-tablet">
            {/* Symbol Selector */}
            <div>
              <label className="block text-sm font-medium mb-2">Symbol</label>
              <select
                value={config.symbol}
                onChange={(e) => handleConfigChange('symbol', e.target.value)}
                className="w-full bg-gray-700 border border-gray-600 rounded px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="SPY">SPY - S&P 500 ETF</option>
                <option value="QQQ">QQQ - Nasdaq 100 ETF</option>
                <option value="AAPL">AAPL - Apple Inc.</option>
                <option value="TSLA">TSLA - Tesla Inc.</option>
                <option value="MSFT">MSFT - Microsoft Corp.</option>
                <option value="NVDA">NVDA - NVIDIA Corp.</option>
                <option value="AMZN">AMZN - Amazon.com Inc.</option>
              </select>
            </div>

            {/* Start Date */}
            <div>
              <label className="block text-sm font-medium mb-2">Start Date</label>
              <input
                type="date"
                value={config.startDate}
                onChange={(e) => handleConfigChange('startDate', e.target.value)}
                className="w-full bg-gray-700 border border-gray-600 rounded px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            {/* End Date */}
            <div>
              <label className="block text-sm font-medium mb-2">End Date</label>
              <input
                type="date"
                value={config.endDate}
                onChange={(e) => handleConfigChange('endDate', e.target.value)}
                className="w-full bg-gray-700 border border-gray-600 rounded px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            {/* Initial Capital */}
            <div>
              <label className="block text-sm font-medium mb-2">Initial Capital ($)</label>
              <input
                type="number"
                value={config.initialCapital}
                onChange={(e) => handleConfigChange('initialCapital', parseFloat(e.target.value))}
                min="1000"
                step="1000"
                className="w-full bg-gray-700 border border-gray-600 rounded px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            {/* Commission */}
            <div>
              <label className="block text-sm font-medium mb-2">Commission ($/share)</label>
              <input
                type="number"
                value={config.commission}
                onChange={(e) => handleConfigChange('commission', parseFloat(e.target.value))}
                min="0"
                step="0.001"
                className="w-full bg-gray-700 border border-gray-600 rounded px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            {/* Slippage */}
            <div>
              <label className="block text-sm font-medium mb-2">Slippage (%)</label>
              <input
                type="number"
                value={config.slippage}
                onChange={(e) => handleConfigChange('slippage', parseFloat(e.target.value))}
                min="0"
                step="0.01"
                className="w-full bg-gray-700 border border-gray-600 rounded px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>

          {/* Strategy Parameters */}
          <h3 className="text-lg font-semibold mt-6 mb-4">Strategy Parameters</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {/* EMA Period */}
            <div>
              <label className="block text-sm font-medium mb-2">EMA Period</label>
              <input
                type="range"
                value={config.emaPeriod}
                onChange={(e) => handleConfigChange('emaPeriod', parseInt(e.target.value))}
                min="5"
                max="50"
                className="w-full"
              />
              <div className="text-center text-sm text-gray-400 mt-1">{config.emaPeriod}</div>
            </div>

            {/* RSI Period */}
            <div>
              <label className="block text-sm font-medium mb-2">RSI Period</label>
              <input
                type="range"
                value={config.rsiPeriod}
                onChange={(e) => handleConfigChange('rsiPeriod', parseInt(e.target.value))}
                min="5"
                max="30"
                className="w-full"
              />
              <div className="text-center text-sm text-gray-400 mt-1">{config.rsiPeriod}</div>
            </div>

            {/* ATR Period */}
            <div>
              <label className="block text-sm font-medium mb-2">ATR Period</label>
              <input
                type="range"
                value={config.atrPeriod}
                onChange={(e) => handleConfigChange('atrPeriod', parseInt(e.target.value))}
                min="5"
                max="30"
                className="w-full"
              />
              <div className="text-center text-sm text-gray-400 mt-1">{config.atrPeriod}</div>
            </div>

            {/* KAMA Period */}
            <div>
              <label className="block text-sm font-medium mb-2">KAMA Period</label>
              <input
                type="range"
                value={config.kamaPeriod}
                onChange={(e) => handleConfigChange('kamaPeriod', parseInt(e.target.value))}
                min="5"
                max="20"
                className="w-full"
              />
              <div className="text-center text-sm text-gray-400 mt-1">{config.kamaPeriod}</div>
            </div>
          </div>

          {/* Risk Parameters */}
          <h3 className="text-lg font-semibold mt-6 mb-4">Risk Parameters</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Max Position Size */}
            <div>
              <label className="block text-sm font-medium mb-2">Max Position Size (%)</label>
              <input
                type="range"
                value={config.maxPositionSize}
                onChange={(e) => handleConfigChange('maxPositionSize', parseInt(e.target.value))}
                min="5"
                max="50"
                className="w-full"
              />
              <div className="text-center text-sm text-gray-400 mt-1">{config.maxPositionSize}%</div>
            </div>

            {/* Stop Loss */}
            <div>
              <label className="block text-sm font-medium mb-2">Stop Loss (%)</label>
              <input
                type="range"
                value={config.stopLossPct}
                onChange={(e) => handleConfigChange('stopLossPct', parseFloat(e.target.value))}
                min="1"
                max="10"
                step="0.5"
                className="w-full"
              />
              <div className="text-center text-sm text-gray-400 mt-1">{config.stopLossPct.toFixed(1)}%</div>
            </div>
          </div>

          {/* Run Button */}
          <div className="mt-6 flex justify-center">
            <button
              onClick={runBacktest}
              disabled={isRunning}
              className={`px-8 py-3 rounded-lg font-semibold text-lg ${
                isRunning
                  ? 'bg-gray-600 cursor-not-allowed'
                  : 'bg-blue-600 hover:bg-blue-700 active:bg-blue-800'
              } transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500`}
            >
              {isRunning ? (
                <div className="flex items-center gap-3">
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                  <span>Running Backtest...</span>
                </div>
              ) : (
                'Run Backtest'
              )}
            </button>
          </div>

          {/* Error Display */}
          {error && (
            <div className="mt-4 bg-red-900 border border-red-700 rounded-lg p-4 text-red-200">
              <div className="font-semibold mb-1">Error</div>
              <div>{error}</div>
            </div>
          )}
        </div>

        {/* Results Modal */}
        {showResults && results && (
          <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-6">
            <div className="bg-gray-800 rounded-lg max-w-6xl w-full max-h-[90vh] overflow-y-auto">
              {/* Modal Header */}
              <div className="sticky top-0 bg-gray-800 border-b border-gray-700 p-6 flex justify-between items-center">
                <h2 className="text-2xl font-bold">Backtest Results</h2>
                <button
                  onClick={closeResults}
                  className="text-gray-400 hover:text-white text-2xl font-bold"
                >
                  ×
                </button>
              </div>

              {/* Modal Content */}
              <div className="p-6">
                {/* Summary Metrics */}
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4 mb-6">
                  <div className="bg-gray-700 rounded-lg p-4">
                    <div className="text-sm text-gray-400 mb-1">Total Return</div>
                    <div className={`text-2xl font-bold ${
                      results.totalReturn >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      ${results.totalReturn.toLocaleString('en-US', { minimumFractionDigits: 2 })}
                    </div>
                    <div className={`text-sm ${
                      results.totalReturnPct >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {results.totalReturnPct >= 0 ? '+' : ''}{results.totalReturnPct.toFixed(2)}%
                    </div>
                  </div>

                  <div className="bg-gray-700 rounded-lg p-4">
                    <div className="text-sm text-gray-400 mb-1">Sharpe Ratio</div>
                    <div className={`text-2xl font-bold ${
                      results.sharpeRatio > 0.8 ? 'text-green-400' :
                      results.sharpeRatio > 0.5 ? 'text-yellow-400' :
                      'text-red-400'
                    }`}>
                      {results.sharpeRatio.toFixed(2)}
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      {results.sharpeRatio > 0.8 ? 'Excellent' :
                       results.sharpeRatio > 0.5 ? 'Good' :
                       'Poor'}
                    </div>
                  </div>

                  <div className="bg-gray-700 rounded-lg p-4">
                    <div className="text-sm text-gray-400 mb-1">Max Drawdown</div>
                    <div className={`text-2xl font-bold ${
                      results.maxDrawdown < 15 ? 'text-green-400' :
                      results.maxDrawdown < 25 ? 'text-yellow-400' :
                      'text-red-400'
                    }`}>
                      {results.maxDrawdown.toFixed(2)}%
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      {results.maxDrawdown < 15 ? 'Low Risk' :
                       results.maxDrawdown < 25 ? 'Moderate' :
                       'High Risk'}
                    </div>
                  </div>

                  <div className="bg-gray-700 rounded-lg p-4">
                    <div className="text-sm text-gray-400 mb-1">Win Rate</div>
                    <div className={`text-2xl font-bold ${
                      results.winRate >= 55 ? 'text-green-400' :
                      results.winRate >= 45 ? 'text-yellow-400' :
                      'text-red-400'
                    }`}>
                      {results.winRate.toFixed(1)}%
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      {results.totalTrades} trades
                    </div>
                  </div>

                  <div className="bg-gray-700 rounded-lg p-4">
                    <div className="text-sm text-gray-400 mb-1">Profit Factor</div>
                    <div className={`text-2xl font-bold ${
                      results.profitFactor >= 2 ? 'text-green-400' :
                      results.profitFactor >= 1.5 ? 'text-yellow-400' :
                      'text-red-400'
                    }`}>
                      {results.profitFactor.toFixed(2)}
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      Avg Win: ${results.avgWin.toFixed(2)}
                    </div>
                  </div>
                </div>

                {/* Success Criteria Check */}
                <div className="bg-gray-700 rounded-lg p-4 mb-6">
                  <h3 className="text-lg font-semibold mb-3">Success Criteria</h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="flex items-center gap-2">
                      <div className={`w-4 h-4 rounded-full ${
                        results.sharpeRatio > 0.8 ? 'bg-green-400' : 'bg-red-400'
                      }`}></div>
                      <span>Sharpe &gt; 0.8: {results.sharpeRatio > 0.8 ? 'PASS' : 'FAIL'}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className={`w-4 h-4 rounded-full ${
                        results.maxDrawdown < 25 ? 'bg-green-400' : 'bg-red-400'
                      }`}></div>
                      <span>Max DD &lt; 25%: {results.maxDrawdown < 25 ? 'PASS' : 'FAIL'}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className={`w-4 h-4 rounded-full ${
                        results.winRate > 45 ? 'bg-green-400' : 'bg-red-400'
                      }`}></div>
                      <span>Win Rate &gt; 45%: {results.winRate > 45 ? 'PASS' : 'FAIL'}</span>
                    </div>
                  </div>
                </div>

                {/* Export Button */}
                <div className="mb-6">
                  <button
                    onClick={exportToCSV}
                    className="px-6 py-2 bg-green-600 hover:bg-green-700 rounded-lg font-semibold transition-colors"
                  >
                    Export Trade Log (CSV)
                  </button>
                </div>

                {/* Trade Log Table */}
                <div>
                  <h3 className="text-lg font-semibold mb-3">Trade Log (Last 20 trades)</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead className="bg-gray-700">
                        <tr>
                          <th className="px-4 py-2 text-left">Entry Date</th>
                          <th className="px-4 py-2 text-left">Exit Date</th>
                          <th className="px-4 py-2 text-left">Side</th>
                          <th className="px-4 py-2 text-right">Entry</th>
                          <th className="px-4 py-2 text-right">Exit</th>
                          <th className="px-4 py-2 text-right">Shares</th>
                          <th className="px-4 py-2 text-right">P&L</th>
                        </tr>
                      </thead>
                      <tbody>
                        {results.trade_log.slice(-20).reverse().map((trade, idx) => (
                          <tr key={idx} className="border-t border-gray-700 hover:bg-gray-700">
                            <td className="px-4 py-2">{trade.entry_date}</td>
                            <td className="px-4 py-2">{trade.exit_date}</td>
                            <td className="px-4 py-2">
                              <span className={`px-2 py-1 rounded text-xs ${
                                trade.side === 'LONG' ? 'bg-green-900 text-green-200' : 'bg-red-900 text-red-200'
                              }`}>
                                {trade.side}
                              </span>
                            </td>
                            <td className="px-4 py-2 text-right">${trade.entry_price.toFixed(2)}</td>
                            <td className="px-4 py-2 text-right">${trade.exit_price.toFixed(2)}</td>
                            <td className="px-4 py-2 text-right">{trade.shares}</td>
                            <td className={`px-4 py-2 text-right font-semibold ${
                              trade.pnl >= 0 ? 'text-green-400' : 'text-red-400'
                            }`}>
                              ${trade.pnl.toFixed(2)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>

              {/* Modal Footer */}
              <div className="sticky bottom-0 bg-gray-800 border-t border-gray-700 p-6 flex justify-end">
                <button
                  onClick={closeResults}
                  className="px-6 py-2 bg-gray-600 hover:bg-gray-700 rounded-lg font-semibold transition-colors"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
