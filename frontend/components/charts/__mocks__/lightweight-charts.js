/**
 * Jest mock for lightweight-charts library v5
 * Provides mock implementations of chart APIs for testing
 */

const createMockSeries = () => ({
  setData: jest.fn(),
  update: jest.fn(),
  applyOptions: jest.fn(),
  createPriceLine: jest.fn().mockReturnValue({
    applyOptions: jest.fn(),
    options: jest.fn().mockReturnValue({}),
  }),
  priceScale: jest.fn().mockReturnValue({
    applyOptions: jest.fn(),
  }),
});

const mockTimeScale = {
  fitContent: jest.fn(),
  scrollToPosition: jest.fn(),
  scrollToRealTime: jest.fn(),
  getVisibleRange: jest.fn().mockReturnValue({ from: 0, to: 100 }),
  setVisibleRange: jest.fn(),
  applyOptions: jest.fn(),
  subscribeVisibleTimeRangeChange: jest.fn(),
  unsubscribeVisibleTimeRangeChange: jest.fn(),
};

const mockPriceScale = {
  applyOptions: jest.fn(),
  options: jest.fn().mockReturnValue({}),
  width: jest.fn().mockReturnValue(50),
};

const mockChart = {
  // v5 API: addSeries(SeriesDefinition, options)
  addSeries: jest.fn().mockImplementation(() => createMockSeries()),
  removeSeries: jest.fn(),
  timeScale: jest.fn().mockReturnValue(mockTimeScale),
  priceScale: jest.fn().mockReturnValue(mockPriceScale),
  applyOptions: jest.fn(),
  options: jest.fn().mockReturnValue({}),
  resize: jest.fn(),
  remove: jest.fn(),
  subscribeCrosshairMove: jest.fn(),
  unsubscribeCrosshairMove: jest.fn(),
  subscribeClick: jest.fn(),
  unsubscribeClick: jest.fn(),
};

export const createChart = jest.fn().mockReturnValue(mockChart);

// v5 Series Definitions (used with addSeries)
export const LineSeries = { type: 'Line' };
export const AreaSeries = { type: 'Area' };
export const CandlestickSeries = { type: 'Candlestick' };
export const HistogramSeries = { type: 'Histogram' };
export const BarSeries = { type: 'Bar' };
export const BaselineSeries = { type: 'Baseline' };

// v5 Plugin API for markers
export const createSeriesMarkers = jest.fn();

// Export enums
export const LineStyle = {
  Solid: 0,
  Dotted: 1,
  Dashed: 2,
  LargeDashed: 3,
  SparseDotted: 4,
};

export const LineType = {
  Simple: 0,
  WithSteps: 1,
};

export const CrosshairMode = {
  Normal: 0,
  Magnet: 1,
};

export const PriceScaleMode = {
  Normal: 0,
  Logarithmic: 1,
  Percentage: 2,
  IndexedTo100: 3,
};

export const ColorType = {
  Solid: 'solid',
  VerticalGradient: 'gradient',
};

// Reset function for tests
export const resetMocks = () => {
  createChart.mockClear();
  createSeriesMarkers.mockClear();
  Object.values(mockChart).forEach((fn) => {
    if (typeof fn === 'function' && fn.mockClear) {
      fn.mockClear();
    }
  });
  Object.values(mockTimeScale).forEach((fn) => {
    if (typeof fn === 'function' && fn.mockClear) {
      fn.mockClear();
    }
  });
};

export default {
  createChart,
  LineSeries,
  AreaSeries,
  CandlestickSeries,
  HistogramSeries,
  BarSeries,
  BaselineSeries,
  createSeriesMarkers,
  LineStyle,
  LineType,
  CrosshairMode,
  PriceScaleMode,
  ColorType,
  resetMocks,
};
