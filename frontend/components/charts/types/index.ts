/**
 * TypeScript types for chart components
 */

import type {
  IChartApi,
  ISeriesApi,
  SeriesType,
  Time,
  LineData,
  CandlestickData,
  HistogramData,
  AreaData,
} from 'lightweight-charts';

// Re-export commonly used types from lightweight-charts
export type {
  IChartApi,
  ISeriesApi,
  SeriesType,
  Time,
  LineData,
  CandlestickData,
  HistogramData,
  AreaData,
};

/**
 * Base props shared by all chart components
 */
export interface BaseChartProps {
  height?: number;
  className?: string;
  title?: string;
}

/**
 * Data point with time for line/area series
 */
export interface TimeValueData {
  time: Time;
  value: number;
}

/**
 * OHLC data point for candlestick series
 */
export interface OHLCData {
  time: Time;
  open: number;
  high: number;
  low: number;
  close: number;
}

/**
 * Volume data point
 */
export interface VolumeData {
  time: Time;
  value: number;
  color?: string;
}

/**
 * Props for EquityCurveChart component
 */
export interface EquityCurveChartProps extends BaseChartProps {
  data: {
    times: (string | number)[];  // Accept both date strings and Unix timestamps
    equity: number[];
    benchmark?: number[];
    totalReturnPct?: number;
  };
  initialCapital?: number;
  showBenchmark?: boolean;
  benchmarkLabel?: string;
  hideReturnAnnotation?: boolean;
}

/**
 * Indicator overlay configuration
 */
export interface IndicatorConfig {
  name: string;
  values: (number | null)[];
  color: string;
  dash?: 'solid' | 'dash' | 'dot';
}

/**
 * Props for CandlestickChart component
 */
export interface CandlestickChartProps extends BaseChartProps {
  data: {
    times: (string | number)[];  // Accept both date strings and Unix timestamps
    open: number[];
    high: number[];
    low: number[];
    close: number[];
    volume?: number[];
  };
  indicators?: IndicatorConfig[];
  showVolume?: boolean;
}

/**
 * Series configuration for multi-line charts
 */
export interface SeriesConfig {
  name: string;
  values: number[];
  color: string;
}

/**
 * Props for PnLComparisonChart component
 */
export interface PnLComparisonChartProps extends BaseChartProps {
  data: {
    times: (string | number)[];  // Accept both date strings and Unix timestamps
    series: SeriesConfig[];
  };
  formatAsPercent?: boolean;
  formatAsCurrency?: boolean;
}

/**
 * Props for simple LineChart component
 */
export interface LineChartProps extends BaseChartProps {
  data: {
    times: (string | number)[];  // Accept both date strings and Unix timestamps
    values: number[];
  };
  color?: string;
  lineWidth?: 1 | 2 | 3 | 4;
  showArea?: boolean;
}

/**
 * Chart ref type for imperative handle
 */
export interface ChartRef {
  chart: IChartApi | null;
  container: HTMLDivElement | null;
}

/**
 * Return value from useChart hook
 */
export interface UseChartReturn {
  chartRef: React.RefObject<HTMLDivElement | null>;
  chart: IChartApi | null;
  isLoading: boolean;
}

/**
 * Options for creating a series
 */
export interface CreateSeriesOptions {
  type: 'line' | 'area' | 'candlestick' | 'histogram';
  priceScaleId?: string;
  color?: string;
  lineWidth?: number;
  lineStyle?: number;
  priceFormat?: {
    type: 'price' | 'volume' | 'percent' | 'custom';
    precision?: number;
    minMove?: number;
    formatter?: (price: number) => string;
  };
}

/**
 * Legend item for charts with multiple series
 */
export interface LegendItem {
  name: string;
  color: string;
  value?: number | string;
  visible: boolean;
}
