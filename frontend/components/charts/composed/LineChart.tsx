/**
 * LineChart - Simple single or multi-series line chart
 *
 * Features:
 * - Single line or multiple line series
 * - Configurable line styles and colors
 * - Optional area fill
 */

'use client';

import React, { useEffect, useRef, useState } from 'react';
import type { IChartApi } from 'lightweight-charts';
import { ChartContainer } from '../core/ChartContainer';
import { CHART_COLORS, createChartOptions } from '../config/theme';
import { LINE_WIDTH, CHART_HEIGHT } from '../config/constants';
import { toLineData } from '../utils/dataTransformers';
import { withOpacity } from '../utils/colorUtils';
import { formatPrice } from '../utils/formatters';

export interface LineChartProps {
  data: {
    times: (string | number)[];  // Accept both date strings and Unix timestamps
    values: number[];
  };
  height?: number;
  title?: string;
  color?: string;
  lineWidth?: 1 | 2 | 3 | 4;
  showArea?: boolean;
  className?: string;
}

export function LineChart({
  data,
  height = CHART_HEIGHT.md,
  title,
  color = CHART_COLORS.accent,
  lineWidth = LINE_WIDTH.normal,
  showArea = false,
  className,
}: LineChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!containerRef.current || data.times.length === 0) return;

    let isMounted = true;

    const initChart = async () => {
      const { createChart, LineSeries, AreaSeries } = await import('lightweight-charts');

      if (!isMounted || !containerRef.current) return;

      const container = containerRef.current;
      const { width } = container.getBoundingClientRect();

      // Create chart
      const chart = createChart(container, createChartOptions(width, height, {
        timeScale: {
          timeVisible: false,
        },
      }));

      chartRef.current = chart;

      // Add series
      if (showArea) {
        const areaSeries = chart.addSeries(AreaSeries, {
          lineColor: color,
          lineWidth,
          topColor: withOpacity(color, 0.4),
          bottomColor: withOpacity(color, 0.0),
          priceFormat: {
            type: 'custom',
            formatter: (price: number) => formatPrice(price, 2),
          },
        });
        areaSeries.setData(toLineData(data.times, data.values));
      } else {
        const lineSeries = chart.addSeries(LineSeries, {
          color,
          lineWidth,
          priceFormat: {
            type: 'custom',
            formatter: (price: number) => formatPrice(price, 2),
          },
        });
        lineSeries.setData(toLineData(data.times, data.values));
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
        chartRef.current.remove();
        chartRef.current = null;
      }
    };
  }, [data, height, color, lineWidth, showArea]);

  return (
    <ChartContainer
      height={height}
      isLoading={isLoading}
      className={className}
      title={title}
    >
      <div ref={containerRef} className="w-full h-full" />
    </ChartContainer>
  );
}

export default LineChart;
