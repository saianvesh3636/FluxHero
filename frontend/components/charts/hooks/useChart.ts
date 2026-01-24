/**
 * useChart - Core hook for chart lifecycle management
 *
 * Handles:
 * - SSR-safe dynamic import of lightweight-charts
 * - Chart creation and cleanup
 * - Responsive resize handling via ResizeObserver
 */

'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import type { IChartApi, ChartOptions, DeepPartial } from 'lightweight-charts';
import { DEFAULT_CHART_OPTIONS } from '../config/theme';
import { RESIZE_DEBOUNCE_MS, MIN_CHART_WIDTH, MIN_CHART_HEIGHT } from '../config/constants';

export interface UseChartOptions {
  /** Custom chart options to merge with defaults */
  options?: DeepPartial<ChartOptions>;
  /** Whether to auto-fit content after initial render */
  autoFit?: boolean;
}

export interface UseChartReturn {
  /** Ref to attach to the container div */
  chartRef: React.RefObject<HTMLDivElement | null>;
  /** The chart API instance (null until loaded) */
  chart: IChartApi | null;
  /** Whether the chart is still loading */
  isLoading: boolean;
}

/**
 * Hook for creating and managing a lightweight-charts instance
 */
export function useChart(options: UseChartOptions = {}): UseChartReturn {
  const { options: chartOptions, autoFit = true } = options;

  const chartRef = useRef<HTMLDivElement | null>(null);
  const chartInstance = useRef<IChartApi | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Debounced resize handler
  const resizeTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const handleResize = useCallback(() => {
    if (resizeTimeoutRef.current) {
      clearTimeout(resizeTimeoutRef.current);
    }

    resizeTimeoutRef.current = setTimeout(() => {
      if (chartInstance.current && chartRef.current) {
        const { width, height } = chartRef.current.getBoundingClientRect();
        if (width >= MIN_CHART_WIDTH && height >= MIN_CHART_HEIGHT) {
          chartInstance.current.applyOptions({ width, height });
        }
      }
    }, RESIZE_DEBOUNCE_MS);
  }, []);

  useEffect(() => {
    let isMounted = true;
    let resizeObserver: ResizeObserver | null = null;

    const initChart = async () => {
      // Dynamic import for SSR safety
      const { createChart } = await import('lightweight-charts');

      if (!isMounted || !chartRef.current) return;

      const container = chartRef.current;
      const { width, height } = container.getBoundingClientRect();

      // Create chart with merged options
      const chart = createChart(container, {
        ...DEFAULT_CHART_OPTIONS,
        ...chartOptions,
        width: Math.max(width, MIN_CHART_WIDTH),
        height: Math.max(height, MIN_CHART_HEIGHT),
      } as DeepPartial<ChartOptions>);

      chartInstance.current = chart;
      setIsLoading(false);

      // Auto-fit content after a brief delay to allow series to be added
      if (autoFit) {
        requestAnimationFrame(() => {
          if (chart && isMounted) {
            chart.timeScale().fitContent();
          }
        });
      }

      // Setup ResizeObserver
      resizeObserver = new ResizeObserver(handleResize);
      resizeObserver.observe(container);
    };

    initChart();

    // Cleanup
    return () => {
      isMounted = false;

      if (resizeTimeoutRef.current) {
        clearTimeout(resizeTimeoutRef.current);
      }

      if (resizeObserver) {
        resizeObserver.disconnect();
      }

      if (chartInstance.current) {
        chartInstance.current.remove();
        chartInstance.current = null;
      }
    };
  }, []); // Only run on mount

  // Update chart options when they change
  useEffect(() => {
    if (chartInstance.current && chartOptions) {
      chartInstance.current.applyOptions(chartOptions as DeepPartial<ChartOptions>);
    }
  }, [chartOptions]);

  return {
    chartRef,
    chart: chartInstance.current,
    isLoading,
  };
}

export default useChart;
