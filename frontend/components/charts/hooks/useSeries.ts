/**
 * useSeries - Hook for managing chart series
 *
 * Provides utilities for creating and managing series in lightweight-charts v5
 * Note: The v5 API uses chart.addSeries(SeriesDefinition, options)
 */

'use client';

import { useCallback, useRef } from 'react';
import type {
  IChartApi,
  ISeriesApi,
  SeriesType,
  SeriesDataItemTypeMap,
} from 'lightweight-charts';

type AnySeriesApi = ISeriesApi<SeriesType>;

export interface UseSeriesReturn {
  /** Set data on a series */
  setData: <T extends SeriesType>(
    series: ISeriesApi<T>,
    data: SeriesDataItemTypeMap[T][]
  ) => void;

  /** Update series options */
  updateOptions: (
    series: AnySeriesApi,
    options: Record<string, unknown>
  ) => void;

  /** Remove a series from the chart */
  removeSeries: (chart: IChartApi, series: AnySeriesApi) => void;

  /** Track a series for cleanup */
  trackSeries: (series: AnySeriesApi) => void;

  /** Remove all tracked series from the chart */
  removeAllSeries: (chart: IChartApi) => void;
}

/**
 * Hook for managing chart series lifecycle
 *
 * Note: In lightweight-charts v5, series are created using:
 *   const series = chart.addSeries(LineSeries, options);
 *
 * The series definitions (LineSeries, AreaSeries, etc.) are imported from 'lightweight-charts'
 */
export function useSeries(): UseSeriesReturn {
  // Track all created series for cleanup
  const seriesRefs = useRef<Set<AnySeriesApi>>(new Set());

  const setData = useCallback(
    <T extends SeriesType>(
      series: ISeriesApi<T>,
      data: SeriesDataItemTypeMap[T][]
    ) => {
      series.setData(data);
    },
    []
  );

  const updateOptions = useCallback(
    (series: AnySeriesApi, options: Record<string, unknown>) => {
      series.applyOptions(options);
    },
    []
  );

  const trackSeries = useCallback(
    (series: AnySeriesApi) => {
      seriesRefs.current.add(series);
    },
    []
  );

  const removeSeries = useCallback(
    (chart: IChartApi, series: AnySeriesApi) => {
      chart.removeSeries(series);
      seriesRefs.current.delete(series);
    },
    []
  );

  const removeAllSeries = useCallback(
    (chart: IChartApi) => {
      seriesRefs.current.forEach((series) => {
        try {
          chart.removeSeries(series);
        } catch {
          // Series may already be removed
        }
      });
      seriesRefs.current.clear();
    },
    []
  );

  return {
    setData,
    updateOptions,
    trackSeries,
    removeSeries,
    removeAllSeries,
  };
}

export default useSeries;
