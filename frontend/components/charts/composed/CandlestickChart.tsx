/**
 * CandlestickChart - OHLC candlestick chart with OHLC data display
 *
 * Features:
 * - Candlestick OHLC display with profit/loss colors
 * - OHLC data box showing Open/High/Low/Close/Volume on hover
 * - Optional volume histogram (separate price scale)
 * - Support for indicator overlays (line series)
 * - Magnet crosshair mode for precise data selection
 * - Theme-matched colors
 */

'use client';

import React, { useEffect, useRef, useState } from 'react';
import type { IChartApi, ISeriesApi, MouseEventParams, CandlestickData } from 'lightweight-charts';
import { ChartContainer } from '../core/ChartContainer';
import { OHLCDataBox, type OHLCDisplayData } from '../core/OHLCDataBox';
import { CHART_COLORS, createChartOptions, CROSSHAIR_MODE } from '../config/theme';
import {
  LINE_WIDTH,
  LINE_STYLE,
  CHART_HEIGHT,
  VOLUME_OPACITY,
} from '../config/constants';
import {
  toCandlestickData,
  toVolumeData,
  toIndicatorData,
} from '../utils/dataTransformers';
import { withOpacity } from '../utils/colorUtils';
import { formatPrice, formatVolume, formatChartTime } from '../utils/formatters';

export interface CandlestickChartProps {
  data: {
    times: (string | number)[];
    open: number[];
    high: number[];
    low: number[];
    close: number[];
    volume?: number[];
  };
  indicators?: {
    name: string;
    values: (number | null)[];
    color: string;
    dash?: 'solid' | 'dash' | 'dot';
  }[];
  height?: number;
  showVolume?: boolean;
  showOHLCData?: boolean;
  useMagnetCrosshair?: boolean;
  title?: string;
  className?: string;
}

export function CandlestickChart({
  data,
  indicators = [],
  height = CHART_HEIGHT.lg,
  showVolume = false,
  showOHLCData = true,
  useMagnetCrosshair = true,
  title,
  className,
}: CandlestickChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const indicatorSeriesRef = useRef<ISeriesApi<'Line'>[]>([]);
  const crosshairHandlerRef = useRef<((param: MouseEventParams) => void) | null>(null);
  const lastTimeRef = useRef<string | number | null>(null);
  const volumeMapRef = useRef<Map<string | number, number>>(new Map());
  const isFirstDataLoadRef = useRef(true);
  const [isLoading, setIsLoading] = useState(true);
  const [ohlcData, setOhlcData] = useState<OHLCDisplayData | null>(null);

  // Effect 1: Create chart (only on mount or when structural options change)
  useEffect(() => {
    if (!containerRef.current) return;

    let isMounted = true;

    const initChart = async () => {
      const {
        createChart,
        CandlestickSeries,
        LineSeries,
        HistogramSeries,
      } = await import('lightweight-charts');

      if (!isMounted || !containerRef.current) return;

      const container = containerRef.current;
      const { width } = container.getBoundingClientRect();

      // Create chart
      const chart = createChart(container, createChartOptions(width, height, {
        rightPriceScale: {
          scaleMargins: showVolume
            ? { top: 0.1, bottom: 0.3 }
            : { top: 0.1, bottom: 0.1 },
        },
        timeScale: {
          timeVisible: true,
          secondsVisible: false,
        },
        crosshair: {
          mode: useMagnetCrosshair ? CROSSHAIR_MODE.magnet : CROSSHAIR_MODE.normal,
        },
      }));

      chartRef.current = chart;

      // Add candlestick series
      const candlestickSeries = chart.addSeries(CandlestickSeries, {
        upColor: CHART_COLORS.profit,
        downColor: CHART_COLORS.loss,
        borderUpColor: CHART_COLORS.profit,
        borderDownColor: CHART_COLORS.loss,
        wickUpColor: CHART_COLORS.profit,
        wickDownColor: CHART_COLORS.loss,
        priceFormat: {
          type: 'custom',
          formatter: (price: number) => formatPrice(price, 2),
        },
      });
      candlestickSeriesRef.current = candlestickSeries;

      // Add volume series if needed
      if (showVolume) {
        const volumeSeries = chart.addSeries(HistogramSeries, {
          priceFormat: {
            type: 'custom',
            formatter: (price: number) => formatVolume(price),
          },
          priceScaleId: 'volume',
        });
        chart.priceScale('volume').applyOptions({
          scaleMargins: { top: 0.85, bottom: 0 },
          borderColor: CHART_COLORS.grid,
        });
        volumeSeriesRef.current = volumeSeries;
      }

      // Subscribe to crosshair move for OHLC data display
      if (showOHLCData) {
        crosshairHandlerRef.current = (param: MouseEventParams) => {
          if (!param.time || !param.seriesData) {
            if (lastTimeRef.current !== null) {
              lastTimeRef.current = null;
              setOhlcData(null);
            }
            return;
          }

          if (lastTimeRef.current === param.time) {
            return;
          }
          lastTimeRef.current = param.time as string | number;

          const candleData = param.seriesData.get(candlestickSeries) as CandlestickData | undefined;
          if (!candleData) {
            setOhlcData(null);
            return;
          }

          const volume = volumeMapRef.current.get(param.time as string | number);
          const isIntraday = typeof param.time === 'number';
          const formattedTime = formatChartTime(param.time as number | string, isIntraday);

          setOhlcData({
            time: formattedTime,
            open: candleData.open,
            high: candleData.high,
            low: candleData.low,
            close: candleData.close,
            volume,
          });
        };
        chart.subscribeCrosshairMove(crosshairHandlerRef.current);
      }

      setIsLoading(false);

      // Handle resize
      const resizeObserver = new ResizeObserver((entries) => {
        if (entries[0] && chartRef.current) {
          const { width: newWidth } = entries[0].contentRect;
          chartRef.current.applyOptions({ width: newWidth });
        }
      });
      resizeObserver.observe(container);

      return () => {
        resizeObserver.disconnect();
      };
    };

    initChart();

    return () => {
      isMounted = false;
      if (chartRef.current) {
        if (crosshairHandlerRef.current) {
          chartRef.current.unsubscribeCrosshairMove(crosshairHandlerRef.current);
          crosshairHandlerRef.current = null;
        }
        chartRef.current.remove();
        chartRef.current = null;
      }
      candlestickSeriesRef.current = null;
      volumeSeriesRef.current = null;
      indicatorSeriesRef.current = [];
      isFirstDataLoadRef.current = true; // Reset for next chart creation
    };
  }, [height, showVolume, showOHLCData, useMagnetCrosshair]);

  // Effect 2: Update data (without recreating chart)
  useEffect(() => {
    if (!candlestickSeriesRef.current || data.times.length === 0) return;

    // Update candlestick data
    candlestickSeriesRef.current.setData(
      toCandlestickData(data.times, data.open, data.high, data.low, data.close)
    );

    // Update volume map for OHLC display
    volumeMapRef.current.clear();
    if (data.volume) {
      data.times.forEach((time, i) => {
        if (data.volume && data.volume[i] !== undefined) {
          volumeMapRef.current.set(time, data.volume[i]);
        }
      });
    }

    // Update volume series if exists
    if (volumeSeriesRef.current && data.volume && data.volume.length > 0) {
      volumeSeriesRef.current.setData(
        toVolumeData(
          data.times,
          data.volume,
          data.close,
          withOpacity(CHART_COLORS.profit, VOLUME_OPACITY),
          withOpacity(CHART_COLORS.loss, VOLUME_OPACITY),
          withOpacity(CHART_COLORS.text, VOLUME_OPACITY)
        )
      );
    }

    // Fit content only on first data load (not on subsequent updates)
    if (chartRef.current && isFirstDataLoadRef.current) {
      chartRef.current.timeScale().fitContent();
      isFirstDataLoadRef.current = false;
    }
  }, [data.times, data.open, data.high, data.low, data.close, data.volume]);

  // Effect 3: Update indicators (separate from main data)
  useEffect(() => {
    if (!chartRef.current || data.times.length === 0) return;

    const updateIndicators = async () => {
      const { LineSeries } = await import('lightweight-charts');

      // Remove old indicator series
      indicatorSeriesRef.current.forEach(series => {
        try {
          chartRef.current?.removeSeries(series);
        } catch (e) {
          // Series might already be removed
        }
      });
      indicatorSeriesRef.current = [];

      // Add new indicator series
      indicators.forEach((indicator) => {
        if (!chartRef.current) return;

        const lineStyle =
          indicator.dash === 'dash'
            ? LINE_STYLE.dashed
            : indicator.dash === 'dot'
            ? LINE_STYLE.dotted
            : LINE_STYLE.solid;

        const indicatorSeries = chartRef.current.addSeries(LineSeries, {
          color: indicator.color,
          lineWidth: LINE_WIDTH.thin,
          lineStyle,
          priceFormat: {
            type: 'custom',
            formatter: (price: number) => formatPrice(price, 2),
          },
        });

        indicatorSeries.setData(toIndicatorData(data.times, indicator.values));
        indicatorSeriesRef.current.push(indicatorSeries);
      });
    };

    updateIndicators();
  }, [indicators, data.times]);

  return (
    <div className="relative">
      <ChartContainer
        height={height}
        isLoading={isLoading}
        className={className}
        title={title}
      >
        <div ref={containerRef} className="w-full h-full" />
      </ChartContainer>

      {showOHLCData && !isLoading && (
        <OHLCDataBox
          data={ohlcData}
          showVolume={showVolume && !!data.volume}
        />
      )}
    </div>
  );
}

export default CandlestickChart;
