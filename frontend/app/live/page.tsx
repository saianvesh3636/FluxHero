'use client';

import { useEffect, useState } from 'react';
import { apiClient, Position, AccountInfo, SystemStatus } from '../../utils/api';

export default function LiveTradingPage() {
  const [positions, setPositions] = useState<Position[]>([]);
  const [accountInfo, setAccountInfo] = useState<AccountInfo | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  // Fetch data function
  const fetchData = async () => {
    try {
      setError(null);
      const [positionsData, accountData, statusData] = await Promise.all([
        apiClient.getPositions(),
        apiClient.getAccountInfo(),
        apiClient.getSystemStatus(),
      ]);

      setPositions(positionsData);
      setAccountInfo(accountData);
      setSystemStatus(statusData);
      setLastUpdate(new Date());
      setLoading(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch data');
      setLoading(false);
    }
  };

  // Initial fetch and 5-second auto-refresh
  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5000); // 5-second refresh
    return () => clearInterval(interval);
  }, []);

  // Format currency
  const formatCurrency = (value: number): string => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value);
  };

  // Format percentage
  const formatPercent = (value: number): string => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  // Get color for P&L
  const getPnlColor = (value: number): string => {
    if (value > 0) return 'text-green-600';
    if (value < 0) return 'text-red-600';
    return 'text-gray-600';
  };

  // Get system status indicator
  const getStatusIndicator = (status: string | undefined): { color: string; emoji: string } => {
    switch (status) {
      case 'active':
        return { color: 'bg-green-500', emoji: '🟢' };
      case 'delayed':
        return { color: 'bg-yellow-500', emoji: '🟡' };
      case 'offline':
        return { color: 'bg-red-500', emoji: '🔴' };
      default:
        return { color: 'bg-gray-500', emoji: '⚪' };
    }
  };

  // Calculate total exposure
  const totalExposure = positions.reduce(
    (sum, pos) => sum + pos.current_price * pos.quantity,
    0
  );

  // Calculate current drawdown (placeholder, would need historical data)
  const currentDrawdown = accountInfo
    ? ((accountInfo.equity - accountInfo.equity) / accountInfo.equity) * 100
    : 0;

  const statusIndicator = getStatusIndicator(systemStatus?.status);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading live data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6 page-container">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6 page-header">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Live Trading</h1>
          <p className="text-gray-600">
            Last updated: {lastUpdate.toLocaleTimeString()}
          </p>
        </div>

        {/* Error Banner */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
            <p className="text-red-800">⚠️ {error}</p>
          </div>
        )}

        {/* System Heartbeat & Quick Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6 stats-grid">
          {/* System Status */}
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600 mb-1">System Status</p>
                <p className="text-lg font-semibold text-gray-900 capitalize">
                  {systemStatus?.status || 'Unknown'}
                </p>
              </div>
              <div className="text-3xl">{statusIndicator.emoji}</div>
            </div>
          </div>

          {/* Daily P&L */}
          <div className="bg-white rounded-lg shadow p-6">
            <p className="text-sm text-gray-600 mb-1">Daily P&L</p>
            <p className={`text-2xl font-bold ${getPnlColor(accountInfo?.daily_pnl || 0)}`}>
              {formatCurrency(accountInfo?.daily_pnl || 0)}
            </p>
          </div>

          {/* Current Drawdown */}
          <div className="bg-white rounded-lg shadow p-6">
            <p className="text-sm text-gray-600 mb-1">Current Drawdown</p>
            <p className={`text-2xl font-bold ${getPnlColor(-Math.abs(currentDrawdown))}`}>
              {formatPercent(currentDrawdown)}
            </p>
          </div>

          {/* Total Exposure */}
          <div className="bg-white rounded-lg shadow p-6">
            <p className="text-sm text-gray-600 mb-1">Total Exposure</p>
            <p className="text-2xl font-bold text-gray-900">
              {formatCurrency(totalExposure)}
            </p>
            <p className="text-xs text-gray-500 mt-1">
              {accountInfo?.equity
                ? `${((totalExposure / accountInfo.equity) * 100).toFixed(1)}% of equity`
                : ''}
            </p>
          </div>
        </div>

        {/* Open Positions Table */}
        <div className="bg-white rounded-lg shadow overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-200">
            <h2 className="text-xl font-semibold text-gray-900">
              Open Positions ({positions.length})
            </h2>
          </div>

          {positions.length === 0 ? (
            <div className="p-12 text-center">
              <p className="text-gray-500 text-lg">No open positions</p>
              <p className="text-gray-400 text-sm mt-2">
                Positions will appear here when trades are executed
              </p>
            </div>
          ) : (
            <div className="overflow-x-auto table-container">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Symbol
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Quantity
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Entry Price
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Current Price
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      P&L
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      P&L %
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Market Value
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {positions.map((position, index) => (
                    <tr key={`${position.symbol}-${index}`} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className="text-sm font-medium text-gray-900">
                          {position.symbol}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {position.quantity}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {formatCurrency(position.entry_price)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {formatCurrency(position.current_price)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`text-sm font-semibold ${getPnlColor(position.pnl)}`}>
                          {formatCurrency(position.pnl)}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`text-sm font-semibold ${getPnlColor(position.pnl_percent)}`}>
                          {formatPercent(position.pnl_percent)}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {formatCurrency(position.current_price * position.quantity)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Account Info Summary */}
        {accountInfo && (
          <div className="mt-6 bg-white rounded-lg shadow p-6 card card-padding">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Account Summary</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 stats-grid">
              <div>
                <p className="text-sm text-gray-600">Equity</p>
                <p className="text-lg font-semibold text-gray-900">
                  {formatCurrency(accountInfo.equity)}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Cash</p>
                <p className="text-lg font-semibold text-gray-900">
                  {formatCurrency(accountInfo.cash)}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Buying Power</p>
                <p className="text-lg font-semibold text-gray-900">
                  {formatCurrency(accountInfo.buying_power)}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Total P&L</p>
                <p className={`text-lg font-semibold ${getPnlColor(accountInfo.total_pnl)}`}>
                  {formatCurrency(accountInfo.total_pnl)}
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
