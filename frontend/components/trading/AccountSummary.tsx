import React from 'react';
import { cn, formatCurrency, formatPercent } from '../../lib/utils';
import { Card, CardTitle } from '../ui/Card';
import { StatsGrid } from '../layout/Grid';
import { PLDisplay } from './PLDisplay';

export interface AccountInfo {
  equity: number;
  cash: number;
  buyingPower: number;
  dailyPnl: number;
  dailyPnlPercent?: number;
  totalPnl: number;
  totalPnlPercent?: number;
  currentDrawdown?: number;
  totalExposure?: number;
}

export interface AccountSummaryProps {
  account: AccountInfo;
  isLoading?: boolean;
  className?: string;
}

/**
 * AccountSummary - displays account metrics in a grid
 * Follows design system: stats-grid layout, card styling
 */
export function AccountSummary({
  account,
  isLoading = false,
  className,
}: AccountSummaryProps) {
  const {
    equity,
    cash,
    buyingPower,
    dailyPnl,
    dailyPnlPercent,
    totalPnl,
    totalPnlPercent,
    currentDrawdown,
    totalExposure,
  } = account;

  if (isLoading) {
    return (
      <div className={cn('grid gap-5 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4', className)}>
        {Array.from({ length: 4 }).map((_, i) => (
          <Card key={i}>
            <div className="h-4 w-20 bg-panel-500 rounded mb-2" />
            <div className="h-8 w-32 bg-panel-500 rounded" />
          </Card>
        ))}
      </div>
    );
  }

  return (
    <StatsGrid columns={4} className={className}>
      {/* Equity */}
      <StatCard
        label="Equity"
        value={formatCurrency(equity)}
        valueClassName="text-text-900"
      />

      {/* Cash */}
      <StatCard
        label="Cash"
        value={formatCurrency(cash)}
        valueClassName="text-text-700"
      />

      {/* Buying Power */}
      <StatCard
        label="Buying Power"
        value={formatCurrency(buyingPower)}
        valueClassName="text-text-700"
      />

      {/* Daily P&L */}
      <Card>
        <span className="text-sm text-text-400 block mb-1">Daily P&L</span>
        <PLDisplay
          value={dailyPnl}
          percent={dailyPnlPercent}
          size="lg"
        />
      </Card>

      {/* Total P&L */}
      <Card>
        <span className="text-sm text-text-400 block mb-1">Total P&L</span>
        <PLDisplay
          value={totalPnl}
          percent={totalPnlPercent}
          size="lg"
        />
      </Card>

      {/* Current Drawdown (optional) */}
      {currentDrawdown !== undefined && (
        <StatCard
          label="Current Drawdown"
          value={formatPercent(currentDrawdown)}
          valueClassName={currentDrawdown > 10 ? 'text-loss-500' : 'text-text-700'}
        />
      )}

      {/* Total Exposure (optional) */}
      {totalExposure !== undefined && (
        <StatCard
          label="Total Exposure"
          value={formatPercent(totalExposure)}
          valueClassName="text-text-700"
        />
      )}
    </StatsGrid>
  );
}

interface StatCardProps {
  label: string;
  value: string;
  subtitle?: string;
  valueClassName?: string;
}

function StatCard({ label, value, subtitle, valueClassName }: StatCardProps) {
  return (
    <Card>
      <span className="text-sm text-text-400 block mb-1">{label}</span>
      <span className={cn('text-xl font-semibold font-mono tabular-nums', valueClassName)}>
        {value}
      </span>
      {subtitle && (
        <span className="text-xs text-text-300 block mt-1">{subtitle}</span>
      )}
    </Card>
  );
}

export default AccountSummary;
