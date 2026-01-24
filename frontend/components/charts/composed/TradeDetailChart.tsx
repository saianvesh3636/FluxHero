/**
 * TradeDetailChart - Candlestick chart with trade visualization
 *
 * Features:
 * - Candlestick OHLC with indicator overlays
 * - OHLC data box showing Open/High/Low/Close/Volume on hover
 * - Horizontal price lines (entry, exit, stop loss, take profit)
 * - BUY/SELL markers at trade entry/exit points
 * - Magnet crosshair mode for precise data selection
 * - Theme-matched colors
 */

'use client';

import React, { useEffect, useRef, useState } from 'react';
import type { IChartApi, ISeriesApi, CreatePriceLineOptions, MouseEventParams, CandlestickData } from 'lightweight-charts';
import { ChartContainer } from '../core/ChartContainer';
import { OHLCDataBox, type OHLCDisplayData } from '../core/OHLCDataBox';
import { CHART_COLORS, createChartOptions, CROSSHAIR_MODE } from '../config/theme';
import { LINE_WIDTH, LINE_STYLE } from '../config/constants';
import { toCandlestickData, toIndicatorData } from '../utils/dataTransformers';
import { formatPrice, formatChartTime } from '../utils/formatters';

interface Candle {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface Indicator {
  time: number;
  kama: number | null;
  atr_upper: number | null;
  atr_lower: number | null;
}

interface PriceLine {
  price: number;
  color: string;
  lineStyle: 'solid' | 'dashed' | 'dotted';
  label: string;
  labelVisible?: boolean;
}

interface Marker {
  time: string;
  price: number;
  text: string;
  color: string;
  position: 'above' | 'below';
}

export interface TradeDetailChartProps {
  candles: Candle[];
  indicators: Indicator[];
  priceLines?: PriceLine[];
  markers?: Marker[];
  height?: number;
  showOHLCData?: boolean;
  className?: string;
}

export function TradeDetailChart({
  candles,
  indicators,
  priceLines = [],
  markers = [],
  height = 500,
  showOHLCData = true,
  className,
}: TradeDetailChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const crosshairHandlerRef = useRef<((param: MouseEventParams) => void) | null>(null);
  const lastTimeRef = useRef<number | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [ohlcData, setOhlcData] = useState<OHLCDisplayData | null>(null);

  useEffect(() => {
    if (!containerRef.current || candles.length === 0) return;

    let isMounted = true;

    const initChart = async () => {
      const { createChart, CandlestickSeries, LineSeries } = await import('lightweight-charts');

      if (!isMounted || !containerRef.current) return;

      const container = containerRef.current;
      const { width } = container.getBoundingClientRect();

      // Build volume lookup map for OHLC display
      const volumeMap = new Map<number, number>();
      candles.forEach((c) => {
        volumeMap.set(c.time, c.volume);
      });

      // Create chart with magnet crosshair for precise selection
      const chart = createChart(container, createChartOptions(width, height, {
        timeScale: {
          timeVisible: true,
          secondsVisible: false,
        },
        crosshair: {
          mode: CROSSHAIR_MODE.magnet,
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

      // Pass raw timestamps - toCandlestickData handles the conversion
      candlestickSeries.setData(
        toCandlestickData(
          candles.map((c) => c.time),
          candles.map((c) => c.open),
          candles.map((c) => c.high),
          candles.map((c) => c.low),
          candles.map((c) => c.close)
        )
      );
      candlestickSeriesRef.current = candlestickSeries;

      // Add KAMA indicator line
      const kamaData = indicators.filter((i) => i.kama !== null);
      if (kamaData.length > 0) {
        const kamaSeries = chart.addSeries(LineSeries, {
          color: CHART_COLORS.accent,
          lineWidth: LINE_WIDTH.normal,
          priceFormat: {
            type: 'custom',
            formatter: (price: number) => formatPrice(price, 2),
          },
        });
        kamaSeries.setData(
          toIndicatorData(
            kamaData.map((i) => i.time),
            kamaData.map((i) => i.kama)
          )
        );
      }

      // Add ATR Upper band
      const atrUpperData = indicators.filter((i) => i.atr_upper !== null);
      if (atrUpperData.length > 0) {
        const atrUpperSeries = chart.addSeries(LineSeries, {
          color: CHART_COLORS.blue,
          lineWidth: LINE_WIDTH.thin,
          lineStyle: LINE_STYLE.dashed,
          priceFormat: {
            type: 'custom',
            formatter: (price: number) => formatPrice(price, 2),
          },
        });
        atrUpperSeries.setData(
          toIndicatorData(
            atrUpperData.map((i) => i.time),
            atrUpperData.map((i) => i.atr_upper)
          )
        );
      }

      // Add ATR Lower band
      const atrLowerData = indicators.filter((i) => i.atr_lower !== null);
      if (atrLowerData.length > 0) {
        const atrLowerSeries = chart.addSeries(LineSeries, {
          color: CHART_COLORS.blue,
          lineWidth: LINE_WIDTH.thin,
          lineStyle: LINE_STYLE.dashed,
          priceFormat: {
            type: 'custom',
            formatter: (price: number) => formatPrice(price, 2),
          },
        });
        atrLowerSeries.setData(
          toIndicatorData(
            atrLowerData.map((i) => i.time),
            atrLowerData.map((i) => i.atr_lower)
          )
        );
      }

      // Add horizontal price lines
      priceLines.forEach((priceLine) => {
        const lineStyle =
          priceLine.lineStyle === 'dashed'
            ? LINE_STYLE.dashed
            : priceLine.lineStyle === 'dotted'
            ? LINE_STYLE.dotted
            : LINE_STYLE.solid;

        const options: CreatePriceLineOptions = {
          price: priceLine.price,
          color: priceLine.color,
          lineWidth: LINE_WIDTH.thin,
          lineStyle,
          axisLabelVisible: priceLine.labelVisible ?? true,
          title: priceLine.label,
        };

        candlestickSeries.createPriceLine(options);
      });

      // Add markers using the plugin API
      if (markers.length > 0) {
        const { createSeriesMarkers } = await import('lightweight-charts');
        const chartMarkers = markers.map((m) => ({
          time: m.time,
          position: m.position === 'above' ? ('aboveBar' as const) : ('belowBar' as const),
          color: m.color,
          shape: m.position === 'above' ? ('arrowDown' as const) : ('arrowUp' as const),
          text: m.text,
        }));
        createSeriesMarkers(candlestickSeries, chartMarkers);
      }

      // Subscribe to crosshair move for OHLC data display (inline handler)
      if (showOHLCData) {
        crosshairHandlerRef.current = (param: MouseEventParams) => {
          if (!param.time || !param.seriesData) {
            if (lastTimeRef.current !== null) {
              lastTimeRef.current = null;
              setOhlcData(null);
            }
            return;
          }

          // Skip update if hovering over same candle (prevents flicker)
          if (lastTimeRef.current === param.time) {
            return;
          }
          lastTimeRef.current = param.time as number;

          const candleData = param.seriesData.get(candlestickSeries) as CandlestickData | undefined;
          if (!candleData) {
            setOhlcData(null);
            return;
          }

          // Get volume from our map
          const volume = volumeMap.get(param.time as number);

          // Format time - this is intraday data so include time
          const formattedTime = formatChartTime(param.time as number, true);

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

      // Fit content
      chart.timeScale().fitContent();
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
    };
  }, [candles, indicators, priceLines, markers, height, showOHLCData]);

  return (
    <div className="relative">
      <ChartContainer height={height} isLoading={isLoading} className={className}>
        <div ref={containerRef} className="w-full h-full" />
      </ChartContainer>

      {/* OHLC Data Display - positioned outside ChartContainer to avoid overflow:hidden */}
      {showOHLCData && !isLoading && (
        <OHLCDataBox
          data={ohlcData}
          showVolume={true}
        />
      )}
    </div>
  );
}

export default TradeDetailChart;
