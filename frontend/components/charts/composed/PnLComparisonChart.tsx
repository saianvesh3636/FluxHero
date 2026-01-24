/**
 * PnLComparisonChart - Multi-line chart for P&L analysis
 *
 * Features:
 * - Multiple line series with different colors
 * - Interactive legend with hover highlighting
 * - Crosshair value display
 * - Currency/percent/raw formatting
 */

'use client';

import React, { useEffect, useRef, useState, useCallback } from 'react';
import type { IChartApi, ISeriesApi, MouseEventParams } from 'lightweight-charts';
import { ChartContainer } from '../core/ChartContainer';
import { CHART_COLORS, createChartOptions } from '../config/theme';
import { LINE_WIDTH, CHART_HEIGHT } from '../config/constants';
import { toLineData } from '../utils/dataTransformers';
import { withOpacity } from '../utils/colorUtils';
import { formatCurrency, formatPercent, formatPrice } from '../utils/formatters';

export interface PnLComparisonChartProps {
  data: {
    times: (string | number)[];  // Accept both date strings and Unix timestamps
    series: {
      name: string;
      values: number[];
      color: string;
    }[];
  };
  height?: number;
  title?: string;
  formatAsPercent?: boolean;
  formatAsCurrency?: boolean;
  className?: string;
}

interface LegendValue {
  name: string;
  color: string;
  value: number | null;
}

export function PnLComparisonChart({
  data,
  height = CHART_HEIGHT.md,
  title,
  formatAsPercent = false,
  formatAsCurrency = false,
  className,
}: PnLComparisonChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesMapRef = useRef<Map<string, ISeriesApi<'Line'>>>(new Map());
  const crosshairHandlerRef = useRef<((param: MouseEventParams) => void) | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [legendValues, setLegendValues] = useState<LegendValue[]>([]);

  // Format value based on props
  const formatValue = useCallback(
    (value: number): string => {
      if (formatAsPercent) return formatPercent(value, 2);
      if (formatAsCurrency) return formatCurrency(value, 2);
      return formatPrice(value, 2);
    },
    [formatAsPercent, formatAsCurrency]
  );

  useEffect(() => {
    if (!containerRef.current || data.times.length === 0) return;

    let isMounted = true;

    const initChart = async () => {
      const { createChart, LineSeries } = await import('lightweight-charts');

      if (!isMounted || !containerRef.current) return;

      const container = containerRef.current;
      const { width } = container.getBoundingClientRect();

      // Create chart
      const chart = createChart(container, createChartOptions(width, height, {
        rightPriceScale: {
          scaleMargins: { top: 0.15, bottom: 0.1 },
        },
        timeScale: {
          timeVisible: false,
        },
        crosshair: {
          mode: 1, // Normal mode
        },
      }));

      chartRef.current = chart;
      seriesMapRef.current.clear();

      // Add each series
      data.series.forEach((s) => {
        const lineSeries = chart.addSeries(LineSeries, {
          color: s.color,
          lineWidth: LINE_WIDTH.normal,
          priceFormat: {
            type: 'custom',
            formatter: (price: number) => formatValue(price),
          },
        });

        lineSeries.setData(toLineData(data.times, s.values));
        seriesMapRef.current.set(s.name, lineSeries);
      });

      // Initialize legend with final values
      setLegendValues(
        data.series.map((s) => ({
          name: s.name,
          color: s.color,
          value: s.values[s.values.length - 1],
        }))
      );

      // Subscribe to crosshair move for legend updates
      crosshairHandlerRef.current = (param: MouseEventParams) => {
        if (!param.time || !param.seriesData) {
          // Reset to final values when not hovering
          setLegendValues(
            data.series.map((s) => ({
              name: s.name,
              color: s.color,
              value: s.values[s.values.length - 1],
            }))
          );
          return;
        }

        // Update legend with hovered values
        const newValues: LegendValue[] = [];
        data.series.forEach((s) => {
          const series = seriesMapRef.current.get(s.name);
          if (series) {
            const seriesData = param.seriesData.get(series);
            if (seriesData && 'value' in seriesData) {
              newValues.push({
                name: s.name,
                color: s.color,
                value: seriesData.value as number,
              });
            }
          }
        });

        if (newValues.length > 0) {
          setLegendValues(newValues);
        }
      };
      chart.subscribeCrosshairMove(crosshairHandlerRef.current);

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
  }, [data, height, formatValue]);

  return (
    <ChartContainer
      height={height}
      isLoading={isLoading}
      className={className}
      title={title}
    >
      <div ref={containerRef} className="w-full h-full" />

      {/* Interactive legend */}
      {!isLoading && legendValues.length > 0 && (
        <div
          className="absolute top-2 left-2 flex flex-wrap gap-3 text-xs"
          style={{ backgroundColor: withOpacity(CHART_COLORS.background, 0.9) }}
        >
          {legendValues.map((item) => (
            <div key={item.name} className="flex items-center gap-1.5">
              <div
                className="w-3 h-0.5"
                style={{ backgroundColor: item.color }}
              />
              <span style={{ color: item.color }}>
                {item.name}: {item.value !== null ? formatValue(item.value) : '—'}
              </span>
            </div>
          ))}
        </div>
      )}
    </ChartContainer>
  );
}

export default PnLComparisonChart;
