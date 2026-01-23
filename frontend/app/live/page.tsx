'use client';

import { useEffect, useState, useRef, useCallback } from 'react';
import { apiClient, Position, AccountInfo as ApiAccountInfo, SystemStatus } from '../../utils/api';
import { PageContainer, PageHeader, StatsGrid } from '../../components/layout';
import { Card, CardTitle, Button, Badge, StatusDot, Skeleton } from '../../components/ui';
import { AccountSummary, PositionsTable, PLDisplay } from '../../components/trading';
import type { AccountInfo } from '../../components/trading';
import type { PositionRow } from '../../components/trading';
import { formatCurrency, formatPercent } from '../../lib/utils';

export default function LiveTradingPage() {
  const [positions, setPositions] = useState<Position[]>([]);
  const [accountInfo, setAccountInfo] = useState<ApiAccountInfo | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isBackendOffline, setIsBackendOffline] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  // Track if a fetch is already in progress to prevent duplicate calls
  const isFetchingRef = useRef(false);

  const fetchData = useCallback(async () => {
    // Prevent duplicate concurrent fetches
    if (isFetchingRef.current) return;
    isFetchingRef.current = true;

    try {
      setError(null);
      setIsBackendOffline(false);
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
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch data';
      setError(errorMessage);
      setIsBackendOffline(true);
      setLoading(false);
    } finally {
      isFetchingRef.current = false;
    }
  }, []);

  const handleRetry = () => {
    setLoading(true);
    setError(null);
    fetchData();
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, [fetchData]);

  // Calculate metrics
  const totalExposure = positions.reduce(
    (sum, pos) => sum + pos.current_price * pos.quantity,
    0
  );

  // Convert positions to table format
  const positionRows: PositionRow[] = positions.map((pos) => ({
    symbol: pos.symbol,
    side: pos.quantity > 0 ? 'long' : 'short',
    quantity: Math.abs(pos.quantity),
    entryPrice: pos.entry_price,
    currentPrice: pos.current_price,
    pnl: pos.pnl,
    pnlPercent: pos.pnl_percent,
    marketValue: pos.current_price * pos.quantity,
  }));

  // Convert account info to component format
  const accountData: AccountInfo | null = accountInfo
    ? {
        equity: accountInfo.equity,
        cash: accountInfo.cash,
        buyingPower: accountInfo.buying_power,
        dailyPnl: accountInfo.daily_pnl,
        totalPnl: accountInfo.total_pnl,
        totalExposure: accountInfo.equity > 0 ? (totalExposure / accountInfo.equity) * 100 : 0,
      }
    : null;

  if (loading) {
    return (
      <PageContainer>
        <PageHeader title="Live Trading" subtitle="Loading..." />
        <StatsGrid columns={4} className="mb-8">
          {Array.from({ length: 4 }).map((_, i) => (
            <Card key={i}>
              <Skeleton variant="text" width="60%" className="mb-2" />
              <Skeleton variant="title" width="80%" />
            </Card>
          ))}
        </StatsGrid>
        <Card>
          <Skeleton variant="title" className="mb-4" />
          <div className="space-y-3">
            {Array.from({ length: 5 }).map((_, i) => (
              <Skeleton key={i} variant="text" />
            ))}
          </div>
        </Card>
      </PageContainer>
    );
  }

  return (
    <PageContainer>
      <PageHeader
        title="Live Trading"
        subtitle={`Last updated: ${lastUpdate.toLocaleTimeString()}`}
        actions={
          <div className="flex items-center gap-3">
            <StatusDot
              status={systemStatus?.status === 'active' ? 'connected' : 'disconnected'}
              showLabel
              label={systemStatus?.status || 'Unknown'}
            />
          </div>
        }
      />

      {/* Error Banner */}
      {isBackendOffline && (
        <Card variant="highlighted" className="mb-6 border-l-4 border-loss-500">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <StatusDot status="error" size="lg" />
              <div>
                <p className="text-loss-500 font-medium">Backend Offline</p>
                <p className="text-text-400 text-sm">{error}</p>
              </div>
            </div>
            <Button variant="danger" onClick={handleRetry}>
              Retry Connection
            </Button>
          </div>
        </Card>
      )}

      {/* Quick Stats */}
      <StatsGrid columns={4} className="mb-8">
        <Card>
          <span className="text-sm text-text-400 block mb-1">System Status</span>
          <div className="flex items-center gap-2">
            <Badge
              variant={
                systemStatus?.status === 'active'
                  ? 'success'
                  : systemStatus?.status === 'delayed'
                  ? 'warning'
                  : 'error'
              }
              size="md"
            >
              {(systemStatus?.status || 'Unknown').toUpperCase()}
            </Badge>
          </div>
        </Card>

        <Card>
          <span className="text-sm text-text-400 block mb-1">Daily P&L</span>
          <PLDisplay
            value={accountInfo?.daily_pnl || 0}
            size="lg"
            showPercent={false}
          />
        </Card>

        <Card>
          <span className="text-sm text-text-400 block mb-1">Total Exposure</span>
          <span className="text-xl font-semibold text-text-900 font-mono tabular-nums">
            {formatCurrency(totalExposure)}
          </span>
          {accountInfo?.equity && (
            <span className="text-xs text-text-400 block mt-1">
              {formatPercent((totalExposure / accountInfo.equity) * 100)} of equity
            </span>
          )}
        </Card>

        <Card>
          <span className="text-sm text-text-400 block mb-1">Total P&L</span>
          <PLDisplay
            value={accountInfo?.total_pnl || 0}
            size="lg"
            showPercent={false}
          />
        </Card>
      </StatsGrid>

      {/* Positions Table */}
      <Card noPadding className="mb-8">
        <div className="px-5 py-4 border-b border-panel-500">
          <div className="flex items-center justify-between">
            <CardTitle>Open Positions ({positions.length})</CardTitle>
          </div>
        </div>
        <PositionsTable positions={positionRows} />
      </Card>

      {/* Account Summary */}
      {accountData && (
        <div>
          <h2 className="text-xl font-semibold text-text-900 mb-4">Account Summary</h2>
          <AccountSummary account={accountData} />
        </div>
      )}
    </PageContainer>
  );
}
