'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { apiClient, SystemStatus } from '../utils/api';
import { PageContainer, PageHeader, CardGrid } from '../components/layout';
import { Card, CardTitle, CardDescription, Button, Badge, StatusDot } from '../components/ui';

interface FeatureCardProps {
  title: string;
  description: string;
  href: string;
  icon: React.ReactNode;
}

function FeatureCard({ title, description, href, icon }: FeatureCardProps) {
  return (
    <Link href={href}>
      <Card className="h-full hover:bg-panel-500 cursor-pointer">
        <div className="flex items-start gap-4">
          <div className="icon-container-accent shrink-0">
            {icon}
          </div>
          <div>
            <CardTitle className="text-lg mb-1">{title}</CardTitle>
            <CardDescription>{description}</CardDescription>
          </div>
        </div>
      </Card>
    </Link>
  );
}

export default function Home() {
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [isBackendOffline, setIsBackendOffline] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const checkBackend = async () => {
      try {
        const status = await apiClient.getSystemStatus();
        setSystemStatus(status);
        setIsBackendOffline(false);
      } catch {
        setIsBackendOffline(true);
      } finally {
        setLoading(false);
      }
    };

    checkBackend();
  }, []);

  const handleRetryConnection = async () => {
    setLoading(true);
    try {
      const status = await apiClient.getSystemStatus();
      setSystemStatus(status);
      setIsBackendOffline(false);
    } catch {
      setIsBackendOffline(true);
    } finally {
      setLoading(false);
    }
  };

  return (
    <PageContainer>
      <PageHeader
        title="FluxHero Trading System"
        subtitle="Adaptive retail quantitative trading platform"
      />

      {/* Feature Cards */}
      <CardGrid columns={2} className="mb-8">
        <FeatureCard
          title="Live Trading"
          description="Monitor open positions and real-time P&L"
          href="/live"
          icon={<ChartIcon />}
        />
        <FeatureCard
          title="Analytics"
          description="Charts, indicators, and performance metrics"
          href="/analytics"
          icon={<AnalyticsIcon />}
        />
        <FeatureCard
          title="Backtesting"
          description="Test strategies on historical data"
          href="/backtest"
          icon={<BacktestIcon />}
        />
        <FeatureCard
          title="Trade History"
          description="View past trades and export data"
          href="/history"
          icon={<HistoryIcon />}
        />
      </CardGrid>

      {/* System Status */}
      <Card>
        <div className="flex items-center justify-between mb-4">
          <CardTitle>System Status</CardTitle>
          {!loading && !isBackendOffline && systemStatus && (
            <Badge
              variant={
                systemStatus.status === 'active'
                  ? 'success'
                  : systemStatus.status === 'delayed'
                  ? 'warning'
                  : 'error'
              }
            >
              {systemStatus.status.toUpperCase()}
            </Badge>
          )}
        </div>

        {loading ? (
          <div className="flex items-center gap-3">
            <StatusDot status="connecting" />
            <span className="text-text-400">Checking backend status...</span>
          </div>
        ) : isBackendOffline ? (
          <div>
            <div className="flex items-center gap-3 mb-4">
              <StatusDot status="disconnected" size="lg" />
              <span className="text-loss-500 font-medium">Backend Offline</span>
            </div>
            <p className="text-text-400 mb-4">
              Unable to connect to the backend server. Please ensure the backend is running on port 8000.
            </p>
            <Button variant="danger" onClick={handleRetryConnection}>
              Retry Connection
            </Button>
          </div>
        ) : (
          <div>
            <div className="flex items-center gap-3 mb-2">
              <StatusDot status="connected" size="lg" />
              <span className="text-profit-500 font-medium">Connected</span>
            </div>
            <p className="text-text-400">
              Backend API is connected and ready.
            </p>
            {systemStatus && (
              <div className="mt-4 pt-4 border-t border-panel-500">
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-text-400 block mb-1">Last Update</span>
                    <span className="text-text-700">
                      {new Date(systemStatus.last_update).toLocaleString()}
                    </span>
                  </div>
                  <div>
                    <span className="text-text-400 block mb-1">Uptime</span>
                    <span className="text-text-700">
                      {formatUptime(systemStatus.uptime_seconds)}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </Card>
    </PageContainer>
  );
}

function formatUptime(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  return `${minutes}m`;
}

// Icons as inline SVGs
function ChartIcon() {
  return (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4v16" />
    </svg>
  );
}

function AnalyticsIcon() {
  return (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
    </svg>
  );
}

function BacktestIcon() {
  return (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  );
}

function HistoryIcon() {
  return (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
    </svg>
  );
}
