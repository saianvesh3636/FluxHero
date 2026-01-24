/**
 * useCrosshairOHLC - Hook for tracking OHLC data on crosshair move
 *
 * Subscribes to crosshair move events and extracts OHLC data from
 * a candlestick series. Returns the current hovered data point.
 */

'use client';

import { useEffect, useState, useCallback, useRef } from 'react';
import type { IChartApi, ISeriesApi, MouseEventParams, CandlestickData } from 'lightweight-charts';
import { formatChartTime } from '../utils/formatters';
import type { OHLCDisplayData } from '../core/OHLCDataBox';

export interface UseCrosshairOHLCOptions {
  /** The chart instance */
  chart: IChartApi | null;
  /** The candlestick series to track */
  series: ISeriesApi<'Candlestick'> | null;
  /** Volume data indexed by time for lookup */
  volumeData?: Map<string | number, number>;
  /** Whether to include time in the formatted date */
  includeTime?: boolean;
}

export interface UseCrosshairOHLCReturn {
  /** Current OHLC data (null when not hovering) */
  ohlcData: OHLCDisplayData | null;
  /** Whether currently hovering over data */
  isHovering: boolean;
}

/**
 * Hook for tracking OHLC data when crosshair moves over candlestick series
 */
export function useCrosshairOHLC({
  chart,
  series,
  volumeData,
  includeTime = false,
}: UseCrosshairOHLCOptions): UseCrosshairOHLCReturn {
  const [ohlcData, setOhlcData] = useState<OHLCDisplayData | null>(null);
  const [isHovering, setIsHovering] = useState(false);
  const handlerRef = useRef<((param: MouseEventParams) => void) | null>(null);

  const handleCrosshairMove = useCallback(
    (param: MouseEventParams) => {
      if (!series || !param.time || !param.seriesData) {
        setOhlcData(null);
        setIsHovering(false);
        return;
      }

      const data = param.seriesData.get(series) as CandlestickData | undefined;
      if (!data) {
        setOhlcData(null);
        setIsHovering(false);
        return;
      }

      // Get volume if available
      let volume: number | undefined;
      if (volumeData && param.time) {
        volume = volumeData.get(param.time as string | number);
      }

      // Format time for display
      const formattedTime = formatChartTime(param.time as number | string, includeTime);

      setOhlcData({
        time: formattedTime,
        open: data.open,
        high: data.high,
        low: data.low,
        close: data.close,
        volume,
      });
      setIsHovering(true);
    },
    [series, volumeData, includeTime]
  );

  useEffect(() => {
    if (!chart || !series) return;

    // Store handler reference for cleanup
    handlerRef.current = handleCrosshairMove;
    chart.subscribeCrosshairMove(handlerRef.current);

    return () => {
      if (handlerRef.current) {
        chart.unsubscribeCrosshairMove(handlerRef.current);
        handlerRef.current = null;
      }
    };
  }, [chart, series, handleCrosshairMove]);

  return { ohlcData, isHovering };
}

export default useCrosshairOHLC;
