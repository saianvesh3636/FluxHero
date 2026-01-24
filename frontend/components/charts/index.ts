// Chart Components - Public API
// Using TradingView lightweight-charts library

// Configuration
export { CHART_COLORS, DEFAULT_CHART_OPTIONS, CHART_FONT_FAMILY, createChartOptions, CROSSHAIR_MODE } from './config/theme';
export type { CrosshairMode } from './config/theme';
export { CHART_HEIGHT, LINE_WIDTH, LINE_STYLE } from './config/constants';

// Utilities
export { withOpacity, lighten, darken, blendColors } from './utils/colorUtils';
export {
  formatPrice,
  formatCurrency,
  formatPercent,
  formatVolume,
  formatCompact,
  formatDate,
  formatChartTime,
  createPriceFormatter,
  createCurrencyFormatter,
  createPercentFormatter,
  createVolumeFormatter,
} from './utils/formatters';
export {
  toUTCTimestamp,
  toLineData,
  toCandlestickData,
  toVolumeData,
  toIndicatorData,
  toHorizontalLine,
  timestampToDateString,
} from './utils/dataTransformers';

// Hooks
export { useChart } from './hooks/useChart';
export { useSeries } from './hooks/useSeries';

// Core Components
export { ChartContainer } from './core/ChartContainer';
export { OHLCDataBox } from './core/OHLCDataBox';
export type { OHLCDisplayData, OHLCDataBoxProps } from './core/OHLCDataBox';

// Composed Chart Components
export { EquityCurveChart } from './composed/EquityCurveChart';
export type { EquityCurveChartProps } from './composed/EquityCurveChart';

export { CandlestickChart } from './composed/CandlestickChart';
export type { CandlestickChartProps } from './composed/CandlestickChart';

export { LineChart } from './composed/LineChart';
export type { LineChartProps } from './composed/LineChart';

export { PnLComparisonChart } from './composed/PnLComparisonChart';
export type { PnLComparisonChartProps } from './composed/PnLComparisonChart';

export { TradeDetailChart } from './composed/TradeDetailChart';
export type { TradeDetailChartProps } from './composed/TradeDetailChart';

// Types
export type {
  BaseChartProps,
  TimeValueData,
  OHLCData,
  VolumeData,
  IndicatorConfig,
  SeriesConfig,
  ChartRef,
  UseChartReturn,
  CreateSeriesOptions,
  LegendItem,
} from './types';
