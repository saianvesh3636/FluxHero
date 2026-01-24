/**
 * Chart theme configuration - Single source of truth for all chart colors
 * Maps to the app's Tailwind design system tokens
 */

export const CHART_COLORS = {
  // Backgrounds
  background: '#1C1C28',    // panel-700
  surface: '#21222F',       // panel-500
  tooltip: '#292D3C',       // panel-400

  // Text
  text: '#CCCAD5',          // text-400
  textMuted: '#716f7a',     // text-100

  // Grid
  grid: '#21222F',          // panel-500

  // Semantic colors
  profit: '#22C55E',        // profit-500
  loss: '#EF4444',          // loss-500
  accent: '#A549FC',        // accent-500
  blue: '#3E7AEE',          // blue-500
  warning: '#F59E0B',       // warning-500
  cyan: '#06B6D4',          // cyan-500
  emerald: '#10B981',       // emerald-500
} as const;

export type ChartColorKey = keyof typeof CHART_COLORS;

/**
 * Crosshair mode constants
 * Normal: Crosshair moves freely with cursor
 * Magnet: Crosshair snaps to nearest data point (better for candlestick charts)
 */
export const CROSSHAIR_MODE = {
  normal: 1,
  magnet: 0,
} as const;

export type CrosshairMode = keyof typeof CROSSHAIR_MODE;

/**
 * Font family used across all charts
 */
export const CHART_FONT_FAMILY = 'Inter, system-ui, sans-serif';

/**
 * Default chart options for lightweight-charts
 */
export const DEFAULT_CHART_OPTIONS = {
  layout: {
    background: { color: CHART_COLORS.background },
    textColor: CHART_COLORS.text,
    fontFamily: CHART_FONT_FAMILY,
    fontSize: 11,
  },
  grid: {
    vertLines: { color: CHART_COLORS.grid },
    horzLines: { color: CHART_COLORS.grid },
  },
  crosshair: {
    mode: 1, // CrosshairMode.Normal
    vertLine: {
      color: CHART_COLORS.textMuted,
      width: 1,
      style: 2, // LineStyle.Dashed
      labelBackgroundColor: CHART_COLORS.surface,
    },
    horzLine: {
      color: CHART_COLORS.textMuted,
      width: 1,
      style: 2, // LineStyle.Dashed
      labelBackgroundColor: CHART_COLORS.surface,
    },
  },
  rightPriceScale: {
    borderColor: CHART_COLORS.grid,
    scaleMargins: {
      top: 0.1,
      bottom: 0.1,
    },
  },
  timeScale: {
    borderColor: CHART_COLORS.grid,
    timeVisible: true,
    secondsVisible: false,
  },
  handleScroll: {
    mouseWheel: true,
    pressedMouseMove: true,
    horzTouchDrag: true,
    vertTouchDrag: false,
  },
  handleScale: {
    axisPressedMouseMove: true,
    mouseWheel: true,
    pinch: true,
  },
} as const;

/** Deep partial type helper */
type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

/** Line width type matching lightweight-charts */
type LineWidth = 1 | 2 | 3 | 4;

/** Chart options override type - allows any valid chart configuration */
interface ChartOptionsOverride {
  layout?: {
    background?: { color?: string };
    textColor?: string;
    fontFamily?: string;
    fontSize?: number;
  };
  grid?: {
    vertLines?: { color?: string };
    horzLines?: { color?: string };
  };
  crosshair?: {
    mode?: number;
    vertLine?: {
      color?: string;
      width?: LineWidth;
      style?: number;
      labelBackgroundColor?: string;
    };
    horzLine?: {
      color?: string;
      width?: LineWidth;
      style?: number;
      labelBackgroundColor?: string;
    };
  };
  rightPriceScale?: {
    borderColor?: string;
    scaleMargins?: {
      top?: number;
      bottom?: number;
    };
  };
  timeScale?: {
    borderColor?: string;
    timeVisible?: boolean;
    secondsVisible?: boolean;
  };
  handleScroll?: {
    mouseWheel?: boolean;
    pressedMouseMove?: boolean;
    horzTouchDrag?: boolean;
    vertTouchDrag?: boolean;
  };
  handleScale?: {
    axisPressedMouseMove?: boolean;
    mouseWheel?: boolean;
    pinch?: boolean;
  };
}

/**
 * Create chart options with custom overrides
 */
export function createChartOptions(
  width: number,
  height: number,
  overrides?: ChartOptionsOverride
) {
  return {
    width,
    height,
    layout: {
      background: { color: CHART_COLORS.background },
      textColor: CHART_COLORS.text,
      fontFamily: CHART_FONT_FAMILY,
      fontSize: 11,
      ...overrides?.layout,
    },
    grid: {
      vertLines: { color: CHART_COLORS.grid },
      horzLines: { color: CHART_COLORS.grid },
      ...overrides?.grid,
    },
    crosshair: {
      mode: 1,
      vertLine: {
        color: CHART_COLORS.textMuted,
        width: 1 as LineWidth,
        style: 2,
        labelBackgroundColor: CHART_COLORS.surface,
      },
      horzLine: {
        color: CHART_COLORS.textMuted,
        width: 1 as LineWidth,
        style: 2,
        labelBackgroundColor: CHART_COLORS.surface,
      },
      ...overrides?.crosshair,
    },
    rightPriceScale: {
      borderColor: CHART_COLORS.grid,
      scaleMargins: {
        top: 0.1,
        bottom: 0.1,
      },
      ...overrides?.rightPriceScale,
    },
    timeScale: {
      borderColor: CHART_COLORS.grid,
      timeVisible: true,
      secondsVisible: false,
      ...overrides?.timeScale,
    },
    handleScroll: {
      mouseWheel: true,
      pressedMouseMove: true,
      horzTouchDrag: true,
      vertTouchDrag: false,
      ...overrides?.handleScroll,
    },
    handleScale: {
      axisPressedMouseMove: true,
      mouseWheel: true,
      pinch: true,
      ...overrides?.handleScale,
    },
  };
}
