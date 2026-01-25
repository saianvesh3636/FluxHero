/**
 * Chart Index Page - /chart
 *
 * Symbol search and quick access to charts
 */

'use client';

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import { PageContainer, PageHeader } from '../../components/layout';
import { Input, Button } from '../../components/ui';

const POPULAR_SYMBOLS = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'];

export default function ChartIndexPage() {
  const router = useRouter();
  const [symbol, setSymbol] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (symbol.trim()) {
      router.push(`/chart/${symbol.trim().toUpperCase()}`);
    }
  };

  const handleQuickAccess = (sym: string) => {
    router.push(`/chart/${sym}`);
  };

  return (
    <PageContainer>
      <PageHeader
        title="Charts"
        subtitle="View real-time price charts for any symbol"
      />

      {/* Symbol Search */}
      <div className="bg-panel-600 rounded-xl p-6 mb-6">
        <form onSubmit={handleSubmit} className="flex gap-4">
          <Input
            type="text"
            placeholder="Enter symbol (e.g., SPY, AAPL)"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            className="flex-1"
          />
          <Button type="submit" disabled={!symbol.trim()}>
            View Chart
          </Button>
        </form>
      </div>

      {/* Popular Symbols */}
      <div className="bg-panel-600 rounded-xl p-6">
        <h3 className="text-text-700 font-medium mb-4">Popular Symbols</h3>
        <div className="flex flex-wrap gap-2">
          {POPULAR_SYMBOLS.map((sym) => (
            <button
              key={sym}
              onClick={() => handleQuickAccess(sym)}
              className="px-4 py-2 bg-panel-500 hover:bg-panel-400 text-text-700 rounded-lg transition-colors font-mono"
            >
              {sym}
            </button>
          ))}
        </div>
      </div>
    </PageContainer>
  );
}
