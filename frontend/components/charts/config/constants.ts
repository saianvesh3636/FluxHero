/**
 * Chart dimension and style constants
 * All magic numbers should be defined here with semantic names
 */

// Chart heights in pixels
export const CHART_HEIGHT = {
  sm: 200,
  md: 300,
  lg: 400,
  xl: 500,
} as const;

// Line widths in pixels
export const LINE_WIDTH = {
  thin: 1,
  normal: 2,
  thick: 3,
} as const;

// Line styles (maps to lightweight-charts LineStyle enum)
export const LINE_STYLE = {
  solid: 0,       // LineStyle.Solid
  dotted: 1,      // LineStyle.Dotted
  dashed: 2,      // LineStyle.Dashed
  largeDashed: 3, // LineStyle.LargeDashed
  sparseDotted: 4, // LineStyle.SparseDotted
} as const;

// Volume histogram opacity
export const VOLUME_OPACITY = 0.5;

// Area fill opacity for equity curves
export const AREA_FILL_OPACITY = {
  top: 0.4,
  bottom: 0.0,
} as const;

// Price scale margins
export const PRICE_SCALE_MARGINS = {
  default: { top: 0.1, bottom: 0.1 },
  withVolume: { top: 0.1, bottom: 0.3 },
  volumePane: { top: 0.7, bottom: 0 },
} as const;

// Animation durations in milliseconds
export const ANIMATION_DURATION = {
  fast: 150,
  normal: 300,
  slow: 500,
} as const;

// Minimum chart dimensions
export const MIN_CHART_WIDTH = 200;
export const MIN_CHART_HEIGHT = 100;

// Debounce delay for resize events
export const RESIZE_DEBOUNCE_MS = 100;
