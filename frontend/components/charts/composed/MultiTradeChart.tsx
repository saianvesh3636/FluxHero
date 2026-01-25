/**
 * MultiTradeChart - Chart with multiple trade markers for backtest visualization
 *
 * Features:
 * - Display multiple BUY/SELL markers from backtest trades
 * - Clickable markers for drill-down to individual trades
 * - Indicator overlays (KAMA, ATR bands)
 * - Hover effects on markers
 */

'use client';

import React, { useEffect, useRef, useState, useCallback } from 'react';
import type { IChartApi, ISeriesApi, MouseEventParams, CandlestickData, Time } from 'lightweight-charts';
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

export interface TradeMarker {
  tradeId: number;
  time: number;  // Unix timestamp
  price: number;
  type: 'entry' | 'exit';
  side: 'long' | 'short';
  pnl?: number;
}

export interface MultiTradeChartProps {
  candles: Candle[];
  indicators?: Indicator[];
  markers: TradeMarker[];
  height?: number;
  showOHLCData?: boolean;
  className?: string;
  onMarkerClick?: (tradeId: number, marker: TradeMarker) => void;
}

export function MultiTradeChart({
  candles,
  indicators = [],
  markers,
  height = 500,
  showOHLCData = true,
  className,
  onMarkerClick,
}: MultiTradeChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const crosshairHandlerRef = useRef<((param: MouseEventParams) => void) | null>(null);
  const lastTimeRef = useRef<number | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [ohlcData, setOhlcData] = useState<OHLCDisplayData | null>(null);
  const [hoveredMarker, setHoveredMarker] = useState<TradeMarker | null>(null);

  // Find nearest marker to a given time
  const findNearestMarker = useCallback((time: Time): TradeMarker | null => {
    if (!time || markers.length === 0) return null;

    const timeNum = typeof time === 'number' ? time :
      typeof time === 'string' ? new Date(time).getTime() / 1000 :
      0;

    // Find marker within 2 candles of the hovered time
    const tolerance = 2 * 3600; // 2 hours tolerance

    return markers.find(m => Math.abs(m.time - timeNum) < tolerance) || null;
  }, [markers]);

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

      // Create chart with magnet crosshair
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

      // Add indicator lines if provided
      if (indicators.length > 0) {
        // KAMA line
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

        // ATR Upper band
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

        // ATR Lower band
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
      }

      // Add markers using the plugin API
      if (markers.length > 0) {
        const { createSeriesMarkers } = await import('lightweight-charts');

        const chartMarkers = markers.map((m) => {
          const isEntry = m.type === 'entry';
          const isLong = m.side === 'long';

          // Entry markers: green arrow up for long, red arrow down for short
          // Exit markers: opposite
          const isPositiveAction = isEntry ? isLong : !isLong;

          return {
            time: String(m.time),
            position: isPositiveAction ? ('belowBar' as const) : ('aboveBar' as const),
            color: isPositiveAction ? CHART_COLORS.profit : CHART_COLORS.loss,
            shape: isPositiveAction ? ('arrowUp' as const) : ('arrowDown' as const),
            text: isEntry ? 'BUY' : 'SELL',
          };
        });

        createSeriesMarkers(candlestickSeries, chartMarkers);
      }

      // Subscribe to crosshair move
      if (showOHLCData || onMarkerClick) {
        crosshairHandlerRef.current = (param: MouseEventParams) => {
          if (!param.time || !param.seriesData) {
            if (lastTimeRef.current !== null) {
              lastTimeRef.current = null;
              setOhlcData(null);
              setHoveredMarker(null);
            }
            return;
          }

          // Skip if same time
          if (lastTimeRef.current === param.time) {
            return;
          }
          lastTimeRef.current = param.time as number;

          // Check for hovered marker
          const nearestMarker = findNearestMarker(param.time);
          setHoveredMarker(nearestMarker);

          // Get candle data for OHLC display
          if (showOHLCData) {
            const candleData = param.seriesData.get(candlestickSeries) as CandlestickData | undefined;
            if (!candleData) {
              setOhlcData(null);
              return;
            }

            const volume = volumeMap.get(param.time as number);
            const formattedTime = formatChartTime(param.time as number, true);

            setOhlcData({
              time: formattedTime,
              open: candleData.open,
              high: candleData.high,
              low: candleData.low,
              close: candleData.close,
              volume,
            });
          }
        };
        chart.subscribeCrosshairMove(crosshairHandlerRef.current);
      }

      // Handle click for marker selection
      if (onMarkerClick) {
        chart.subscribeClick((param: MouseEventParams) => {
          if (param.time) {
            const clickedMarker = findNearestMarker(param.time);
            if (clickedMarker) {
              onMarkerClick(clickedMarker.tradeId, clickedMarker);
            }
          }
        });
      }

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
  }, [candles, indicators, markers, height, showOHLCData, onMarkerClick, findNearestMarker]);

  return (
    <div className="relative">
      <ChartContainer height={height} isLoading={isLoading} className={className}>
        <div ref={containerRef} className="w-full h-full" />
      </ChartContainer>

      {/* OHLC Data Display */}
      {showOHLCData && !isLoading && (
        <OHLCDataBox data={ohlcData} showVolume={true} />
      )}

      {/* Hovered marker tooltip */}
      {hoveredMarker && onMarkerClick && (
        <div className="absolute top-4 right-4 bg-panel-700 rounded-lg p-3 shadow-lg border border-panel-500">
          <p className="text-xs text-text-400 mb-1">
            {hoveredMarker.type === 'entry' ? 'Entry' : 'Exit'} • Trade #{hoveredMarker.tradeId}
          </p>
          <p className="font-mono tabular-nums text-text-700">
            ${hoveredMarker.price.toFixed(2)}
          </p>
          {hoveredMarker.pnl !== undefined && (
            <p className={`font-mono tabular-nums text-sm ${hoveredMarker.pnl >= 0 ? 'text-profit-500' : 'text-loss-500'}`}>
              P&L: ${hoveredMarker.pnl.toFixed(2)}
            </p>
          )}
          <p className="text-xs text-accent-500 mt-2">Click to analyze</p>
        </div>
      )}
    </div>
  );
}

export default MultiTradeChart;
