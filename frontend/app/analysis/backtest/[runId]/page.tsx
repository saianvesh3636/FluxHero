/**
 * Backtest Detail Page - /analysis/backtest/[runId]
 *
 * Detailed analysis of a specific backtest run
 */

'use client';

import React, { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { PageContainer } from '../../../../components/layout';
import { BacktestAnalysisView } from '../../../../components/analysis';
import { apiClient, type BacktestResultDetail } from '../../../../utils/api';

export default function BacktestDetailPage() {
  const params = useParams();
  const runId = params.runId as string;

  const [backtest, setBacktest] = useState<BacktestResultDetail | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchBacktest = async () => {
      if (!runId) return;

      setIsLoading(true);
      setError(null);

      try {
        const data = await apiClient.getBacktestResultDetail(runId);
        setBacktest(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load backtest');
      } finally {
        setIsLoading(false);
      }
    };

    fetchBacktest();
  }, [runId]);

  if (isLoading) {
    return (
      <PageContainer>
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-panel-600 rounded w-48" />
          <div className="h-64 bg-panel-600 rounded" />
          <div className="h-96 bg-panel-600 rounded" />
        </div>
      </PageContainer>
    );
  }

  if (error || !backtest) {
    return (
      <PageContainer>
        <div className="bg-panel-600 rounded-xl p-8 text-center">
          <p className="text-loss-500 mb-4">{error || 'Backtest not found'}</p>
          <Link
            href="/analysis/backtest"
            className="text-accent-500 hover:text-accent-400"
          >
            Back to Backtest List
          </Link>
        </div>
      </PageContainer>
    );
  }

  return (
    <PageContainer>
      <Link
        href="/analysis/backtest"
        className="flex items-center gap-2 text-accent-500 hover:text-accent-400 mb-4"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
        </svg>
        Back to Backtest List
      </Link>

      <BacktestAnalysisView backtest={backtest} />
    </PageContainer>
  );
}
