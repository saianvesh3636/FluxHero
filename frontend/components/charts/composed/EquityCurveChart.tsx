/**
 * EquityCurveChart - Equity curve for backtest and walk-forward pages
 *
 * Features:
 * - Area chart showing equity over time with gradient fill
 * - Optional dashed benchmark line
 * - Optional dotted initial capital reference line
 * - Return percentage annotation
 */

'use client';

import React, { useEffect, useRef, useState } from 'react';
import type { IChartApi, ISeriesApi } from 'lightweight-charts';
import { ChartContainer } from '../core/ChartContainer';
import { CHART_COLORS, createChartOptions } from '../config/theme';
import { LINE_WIDTH, LINE_STYLE, CHART_HEIGHT } from '../config/constants';
import { toLineData, toHorizontalLine } from '../utils/dataTransformers';
import { withOpacity } from '../utils/colorUtils';
import { formatCurrency, formatPercent } from '../utils/formatters';

export interface EquityCurveChartProps {
  data: {
    times: (string | number)[];  // Accept both date strings and Unix timestamps
    equity: number[];
    benchmark?: number[];
    totalReturnPct?: number;
  };
  initialCapital?: number;
  height?: number;
  title?: string;
  showBenchmark?: boolean;
  benchmarkLabel?: string;
  hideReturnAnnotation?: boolean;
  className?: string;
}

export function EquityCurveChart({
  data,
  initialCapital,
  height = CHART_HEIGHT.lg,
  title,
  showBenchmark = false,
  benchmarkLabel = 'Benchmark',
  hideReturnAnnotation = false,
  className,
}: EquityCurveChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<'Area'> | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Calculate return percentage
  const endValue = data.equity[data.equity.length - 1];
  const returnPct =
    data.totalReturnPct !== undefined
      ? data.totalReturnPct
      : (() => {
          const startValue = data.equity[0];
          return startValue > 0
            ? ((endValue - startValue) / startValue) * 100
            : 0;
        })();
  const isPositiveReturn = returnPct >= 0;

  useEffect(() => {
    if (!containerRef.current || data.times.length === 0) return;

    let isMounted = true;

    const initChart = async () => {
      const { createChart, AreaSeries, LineSeries } = await import('lightweight-charts');

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

      // Add main equity area series
      const equitySeries = chart.addSeries(AreaSeries, {
        lineColor: CHART_COLORS.accent,
        lineWidth: LINE_WIDTH.normal,
        topColor: withOpacity(CHART_COLORS.accent, 0.4),
        bottomColor: withOpacity(CHART_COLORS.accent, 0.0),
        priceFormat: {
          type: 'custom',
          formatter: (price: number) => formatCurrency(price, 0),
        },
      });

      equitySeries.setData(toLineData(data.times, data.equity));
      seriesRef.current = equitySeries;

      // Add benchmark line if provided
      if (showBenchmark && data.benchmark && data.benchmark.length > 0) {
        const benchmarkSeries = chart.addSeries(LineSeries, {
          color: CHART_COLORS.text,
          lineWidth: LINE_WIDTH.thin,
          lineStyle: LINE_STYLE.dashed,
          priceFormat: {
            type: 'custom',
            formatter: (price: number) => formatCurrency(price, 0),
          },
        });
        benchmarkSeries.setData(toLineData(data.times, data.benchmark));
      }

      // Add initial capital reference line if provided
      if (initialCapital !== undefined && data.times.length >= 2) {
        const capitalSeries = chart.addSeries(LineSeries, {
          color: CHART_COLORS.warning,
          lineWidth: LINE_WIDTH.thin,
          lineStyle: LINE_STYLE.dotted,
          priceFormat: {
            type: 'custom',
            formatter: (price: number) => formatCurrency(price, 0),
          },
        });
        capitalSeries.setData(
          toHorizontalLine(data.times[0], data.times[data.times.length - 1], initialCapital)
        );
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
  }, [data, initialCapital, height, showBenchmark, benchmarkLabel]);

  return (
    <ChartContainer
      height={height}
      isLoading={isLoading}
      className={className}
      title={title}
    >
      <div ref={containerRef} className="w-full h-full" />

      {/* Legend */}
      {!isLoading && (showBenchmark || !hideReturnAnnotation) && (
        <div
          className="absolute top-2 left-2 flex flex-wrap gap-3 text-xs px-2 py-1 rounded"
          style={{ backgroundColor: withOpacity(CHART_COLORS.background, 0.9) }}
        >
          {/* Equity line label */}
          <div className="flex items-center gap-1.5">
            <div
              className="w-3 h-0.5"
              style={{ backgroundColor: CHART_COLORS.accent }}
            />
            <span style={{ color: CHART_COLORS.text }}>Equity</span>
          </div>

          {/* Benchmark line label */}
          {showBenchmark && data.benchmark && data.benchmark.length > 0 && (
            <div className="flex items-center gap-1.5">
              <div
                className="w-3 h-0.5 border-dashed"
                style={{
                  backgroundColor: CHART_COLORS.text,
                  borderStyle: 'dashed',
                }}
              />
              <span style={{ color: CHART_COLORS.text }}>{benchmarkLabel}</span>
            </div>
          )}

          {/* Initial capital line label */}
          {initialCapital !== undefined && (
            <div className="flex items-center gap-1.5">
              <div
                className="w-3 h-0.5"
                style={{
                  backgroundColor: CHART_COLORS.warning,
                  borderStyle: 'dotted',
                }}
              />
              <span style={{ color: CHART_COLORS.text }}>Initial Capital</span>
            </div>
          )}
        </div>
      )}

      {/* Return annotation */}
      {!hideReturnAnnotation && !isLoading && (
        <div
          className="absolute top-2 right-2 px-2 py-1 rounded text-xs font-semibold"
          style={{
            backgroundColor: withOpacity(CHART_COLORS.background, 0.9),
            color: isPositiveReturn ? CHART_COLORS.profit : CHART_COLORS.loss,
          }}
        >
          {formatPercent(returnPct, 2, true)}
        </div>
      )}
    </ChartContainer>
  );
}

export default EquityCurveChart;
