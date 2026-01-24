/**
 * Data transformation utilities for converting app data formats
 * to lightweight-charts compatible formats
 */

import type { Time, LineData, CandlestickData, HistogramData } from 'lightweight-charts';

/**
 * Convert a date string or timestamp to lightweight-charts Time format
 * Handles: ISO date strings, Unix timestamps (seconds or milliseconds)
 *
 * For timestamps: Returns Unix timestamp in SECONDS (lightweight-charts expects seconds)
 * For date strings: Returns as-is if YYYY-MM-DD format, otherwise extracts date part
 */
export function toUTCTimestamp(date: string | number): Time {
  if (typeof date === 'number') {
    // Handle Unix timestamp - lightweight-charts expects seconds
    // Timestamps in seconds are ~10 digits, milliseconds are ~13 digits
    const seconds = date > 10_000_000_000 ? Math.floor(date / 1000) : date;
    return seconds as Time;
  }

  // Handle date string
  if (date.includes('T')) {
    // ISO datetime string - extract date part for daily data
    return date.split('T')[0] as Time;
  }

  // Already YYYY-MM-DD format
  return date as Time;
}

/**
 * Convert arrays of times and values to LineData format
 * Accepts either date strings or Unix timestamps
 */
export function toLineData(times: (string | number)[], values: number[]): LineData[] {
  if (times.length !== values.length) {
    throw new Error('Times and values arrays must have the same length');
  }

  return times.map((time, i) => ({
    time: toUTCTimestamp(time),
    value: values[i],
  }));
}

/**
 * Convert OHLC arrays to CandlestickData format
 * Accepts either date strings or Unix timestamps
 */
export function toCandlestickData(
  times: (string | number)[],
  open: number[],
  high: number[],
  low: number[],
  close: number[]
): CandlestickData[] {
  const length = times.length;
  if (
    open.length !== length ||
    high.length !== length ||
    low.length !== length ||
    close.length !== length
  ) {
    throw new Error('All OHLC arrays must have the same length as times');
  }

  return times.map((time, i) => ({
    time: toUTCTimestamp(time),
    open: open[i],
    high: high[i],
    low: low[i],
    close: close[i],
  }));
}

/**
 * Convert volume arrays to HistogramData format with colors based on price direction
 * Accepts either date strings or Unix timestamps
 */
export function toVolumeData(
  times: (string | number)[],
  volume: number[],
  close: number[],
  profitColor: string,
  lossColor: string,
  neutralColor: string
): HistogramData[] {
  if (times.length !== volume.length || times.length !== close.length) {
    throw new Error('All arrays must have the same length');
  }

  return times.map((time, i) => {
    let color = neutralColor;
    if (i > 0) {
      color = close[i] >= close[i - 1] ? profitColor : lossColor;
    }

    return {
      time: toUTCTimestamp(time),
      value: volume[i],
      color,
    };
  });
}

/**
 * Convert indicator values (with possible nulls) to LineData format
 * Filters out null values to create disconnected segments
 * Accepts either date strings or Unix timestamps
 */
export function toIndicatorData(
  times: (string | number)[],
  values: (number | null)[]
): LineData[] {
  if (times.length !== values.length) {
    throw new Error('Times and values arrays must have the same length');
  }

  return times
    .map((time, i) => {
      const value = values[i];
      if (value === null || value === undefined || isNaN(value)) {
        return null;
      }
      return {
        time: toUTCTimestamp(time),
        value,
      };
    })
    .filter((item): item is LineData => item !== null);
}

/**
 * Create a horizontal line dataset (e.g., for initial capital reference)
 * Accepts either date strings or Unix timestamps
 */
export function toHorizontalLine(
  startTime: string | number,
  endTime: string | number,
  value: number
): LineData[] {
  return [
    { time: toUTCTimestamp(startTime), value },
    { time: toUTCTimestamp(endTime), value },
  ];
}

/**
 * Convert a Unix timestamp to ISO date string.
 * Handles both seconds and milliseconds formats by detecting magnitude.
 * @param timestamp - Unix timestamp (seconds or milliseconds)
 * @deprecated Use toUTCTimestamp for new code
 */
export function timestampToDateString(timestamp: number): string {
  // Timestamps in seconds are ~10 digits (1.7B for 2024)
  // Timestamps in milliseconds are ~13 digits (1.7T for 2024)
  // If timestamp is less than 10 billion, assume seconds
  const ms = timestamp < 10_000_000_000 ? timestamp * 1000 : timestamp;
  return new Date(ms).toISOString().split('T')[0];
}
