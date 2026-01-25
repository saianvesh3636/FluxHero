/**
 * Backtest Analysis Page - /analysis/backtest
 *
 * List of backtest runs with drill-down capability
 */

'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { PageContainer, PageHeader } from '../../../components/layout';
import { Badge } from '../../../components/ui';
import { apiClient, type BacktestResultSummary } from '../../../utils/api';
import { cn, formatPercent } from '../../../lib/utils';

export default function BacktestAnalysisPage() {
  const [results, setResults] = useState<BacktestResultSummary[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchResults = async () => {
      setIsLoading(true);
      setError(null);

      try {
        const data = await apiClient.getBacktestResults(50);
        setResults(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load backtest results');
      } finally {
        setIsLoading(false);
      }
    };

    fetchResults();
  }, []);

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  };

  return (
    <PageContainer>
      <PageHeader
        title="Backtest Analysis"
        subtitle="Review and analyze past backtest results"
      />

      {/* Results List */}
      <div className="bg-panel-600 rounded-xl overflow-hidden">
        {isLoading ? (
          <div className="p-8 text-center">
            <div className="animate-pulse space-y-4">
              {[1, 2, 3, 4, 5].map((i) => (
                <div key={i} className="h-16 bg-panel-500 rounded" />
              ))}
            </div>
          </div>
        ) : error ? (
          <div className="p-8 text-center text-loss-500">{error}</div>
        ) : results.length === 0 ? (
          <div className="p-8 text-center">
            <p className="text-text-400 mb-4">No backtest results found.</p>
            <Link
              href="/backtest"
              className="text-accent-500 hover:text-accent-400"
            >
              Run your first backtest
            </Link>
          </div>
        ) : (
          <table className="w-full">
            <thead>
              <tr className="border-b border-panel-500">
                <th className="text-left px-4 py-3 text-xs font-medium text-text-400">Symbol</th>
                <th className="text-left px-4 py-3 text-xs font-medium text-text-400">Strategy</th>
                <th className="text-left px-4 py-3 text-xs font-medium text-text-400">Period</th>
                <th className="text-right px-4 py-3 text-xs font-medium text-text-400">Return</th>
                <th className="text-right px-4 py-3 text-xs font-medium text-text-400">Sharpe</th>
                <th className="text-right px-4 py-3 text-xs font-medium text-text-400">Max DD</th>
                <th className="text-right px-4 py-3 text-xs font-medium text-text-400">Win Rate</th>
                <th className="text-right px-4 py-3 text-xs font-medium text-text-400">Trades</th>
                <th className="text-center px-4 py-3 text-xs font-medium text-text-400">Action</th>
              </tr>
            </thead>
            <tbody>
              {results.map((result) => {
                const isProfitable = (result.total_return_pct ?? 0) > 0;

                return (
                  <tr
                    key={result.run_id}
                    className="border-b border-panel-500 hover:bg-panel-500/50 transition-colors"
                  >
                    <td className="px-4 py-3 text-text-700 font-medium">
                      {result.symbol}
                    </td>
                    <td className="px-4 py-3">
                      <Badge
                        variant={
                          result.strategy_mode === 'TREND' ? 'info' :
                          result.strategy_mode === 'MEAN_REVERSION' ? 'warning' :
                          'neutral'
                        }
                        size="sm"
                      >
                        {result.strategy_mode}
                      </Badge>
                    </td>
                    <td className="px-4 py-3 text-text-400 text-sm">
                      {formatDate(result.start_date)} - {formatDate(result.end_date)}
                    </td>
                    <td className={cn(
                      'px-4 py-3 text-right font-mono tabular-nums font-medium',
                      isProfitable ? 'text-profit-500' : 'text-loss-500'
                    )}>
                      {formatPercent(result.total_return_pct ?? 0)}
                    </td>
                    <td className="px-4 py-3 text-right font-mono tabular-nums text-text-600">
                      {(result.sharpe_ratio ?? 0).toFixed(2)}
                    </td>
                    <td className="px-4 py-3 text-right font-mono tabular-nums text-loss-500">
                      {formatPercent(result.max_drawdown_pct ?? 0)}
                    </td>
                    <td className="px-4 py-3 text-right font-mono tabular-nums text-text-600">
                      {formatPercent(result.win_rate ?? 0)}
                    </td>
                    <td className="px-4 py-3 text-right font-mono tabular-nums text-text-600">
                      {result.num_trades ?? 0}
                    </td>
                    <td className="px-4 py-3 text-center">
                      <Link
                        href={`/analysis/backtest/${result.run_id}`}
                        className="text-accent-500 hover:text-accent-400 text-sm font-medium"
                      >
                        Analyze
                      </Link>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </PageContainer>
  );
}
