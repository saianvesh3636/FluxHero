/**
 * ChartViewer - Standalone chart component for viewing any symbol
 *
 * Features:
 * - Wraps TradeDetailChart for standalone use
 * - Fetches chart data for any symbol/timeframe
 * - Optional markers and indicators
 * - Can be embedded anywhere or used in dedicated route
 */

'use client';

import React, { useEffect, useState, useCallback } from 'react';
import { TradeDetailChart } from '../charts/composed/TradeDetailChart';
import { ChartContainer } from '../charts/core/ChartContainer';
import { apiClient, type ChartCandleData } from '../../utils/api';
import { cn } from '../../lib/utils';

export interface ChartMarker {
  time: string;
  price: number;
  text: string;
  color: string;
  position: 'above' | 'below';
}

export interface ChartPriceLine {
  price: number;
  color: string;
  lineStyle: 'solid' | 'dashed' | 'dotted';
  label: string;
  labelVisible?: boolean;
}

export interface ChartViewerProps {
  symbol: string;
  timeframe?: string;
  markers?: ChartMarker[];
  priceLines?: ChartPriceLine[];
  height?: number;
  showControls?: boolean;
  className?: string;
  onError?: (error: Error) => void;
}

const TIMEFRAME_OPTIONS = [
  { value: '5m', label: '5m' },
  { value: '15m', label: '15m' },
  { value: '1h', label: '1H' },
  { value: '4h', label: '4H' },
  { value: '1d', label: '1D' },
];

export function ChartViewer({
  symbol,
  timeframe: initialTimeframe = '1h',
  markers = [],
  priceLines = [],
  height = 500,
  showControls = true,
  className,
  onError,
}: ChartViewerProps) {
  const [timeframe, setTimeframe] = useState(initialTimeframe);
  const [candles, setCandles] = useState<ChartCandleData[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchChartData = useCallback(async () => {
    if (!symbol) return;

    setIsLoading(true);
    setError(null);

    try {
      const data = await apiClient.getChartData(symbol, timeframe);
      setCandles(data);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to load chart data';
      setError(errorMessage);
      onError?.(err instanceof Error ? err : new Error(errorMessage));
    } finally {
      setIsLoading(false);
    }
  }, [symbol, timeframe, onError]);

  useEffect(() => {
    fetchChartData();
  }, [fetchChartData]);

  // Transform candles to the format TradeDetailChart expects
  const chartCandles = candles.map(c => ({
    time: c.time,
    open: c.open,
    high: c.high,
    low: c.low,
    close: c.close,
    volume: c.volume,
  }));

  // Empty indicators for basic chart view
  const indicators = candles.map(c => ({
    time: c.time,
    kama: null,
    atr_upper: null,
    atr_lower: null,
  }));

  if (error) {
    return (
      <div className={cn('bg-panel-600 rounded-xl p-6', className)}>
        <div className="flex flex-col items-center justify-center h-64 text-center">
          <p className="text-loss-500 mb-4">{error}</p>
          <button
            onClick={fetchChartData}
            className="px-4 py-2 bg-accent-500 text-white rounded-lg hover:bg-accent-600 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={cn('bg-panel-600 rounded-xl overflow-hidden', className)}>
      {/* Controls Header */}
      {showControls && (
        <div className="flex items-center justify-between px-4 py-3 border-b border-panel-500">
          <div className="flex items-center gap-2">
            <span className="text-text-900 font-semibold">{symbol}</span>
            <span className="text-text-400 text-sm">{timeframe}</span>
          </div>
          <div className="flex gap-1">
            {TIMEFRAME_OPTIONS.map(opt => (
              <button
                key={opt.value}
                onClick={() => setTimeframe(opt.value)}
                className={cn(
                  'px-3 py-1.5 text-sm rounded transition-colors',
                  timeframe === opt.value
                    ? 'bg-accent-500 text-white'
                    : 'bg-panel-500 text-text-400 hover:bg-panel-400 hover:text-text-700'
                )}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Chart */}
      {isLoading ? (
        <ChartContainer height={height} isLoading={true}>
          <div />
        </ChartContainer>
      ) : chartCandles.length > 0 ? (
        <TradeDetailChart
          candles={chartCandles}
          indicators={indicators}
          markers={markers}
          priceLines={priceLines}
          height={height}
          showOHLCData={true}
        />
      ) : (
        <div className="flex items-center justify-center" style={{ height }}>
          <p className="text-text-400">No data available for {symbol}</p>
        </div>
      )}
    </div>
  );
}

export default ChartViewer;
