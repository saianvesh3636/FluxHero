/**
 * Standalone Chart Page - /chart/[symbol]
 *
 * Quick access to any stock chart with timeframe selection
 */

'use client';

import React from 'react';
import { useParams, useSearchParams } from 'next/navigation';
import { PageContainer, PageHeader } from '../../../components/layout';
import { ChartViewer } from '../../../components/analysis';

export default function ChartPage() {
  const params = useParams();
  const searchParams = useSearchParams();

  const symbol = (params.symbol as string)?.toUpperCase() || 'SPY';
  const timeframe = searchParams.get('timeframe') || '1h';

  return (
    <PageContainer>
      <PageHeader
        title={`${symbol} Chart`}
        subtitle="Real-time price chart with technical indicators"
      />

      <ChartViewer
        symbol={symbol}
        timeframe={timeframe}
        height={600}
        showControls={true}
      />
    </PageContainer>
  );
}
