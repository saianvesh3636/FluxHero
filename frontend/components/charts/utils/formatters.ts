/**
 * Number and value formatting utilities for chart labels and tooltips
 */

/**
 * Format a number as price with appropriate decimal places
 * Uses locale formatting with commas
 */
export function formatPrice(value: number, decimals: number = 2): string {
  return value.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

/**
 * Format a number as currency (USD)
 */
export function formatCurrency(value: number, decimals: number = 2): string {
  return '$' + formatPrice(value, decimals);
}

/**
 * Format a number as percentage
 */
export function formatPercent(value: number, decimals: number = 2, showSign: boolean = false): string {
  const formatted = value.toFixed(decimals);
  const prefix = showSign && value > 0 ? '+' : '';
  return `${prefix}${formatted}%`;
}

/**
 * Format large numbers with abbreviations (K, M, B)
 */
export function formatCompact(value: number): string {
  const absValue = Math.abs(value);
  const sign = value < 0 ? '-' : '';

  if (absValue >= 1_000_000_000) {
    return `${sign}${(absValue / 1_000_000_000).toFixed(2)}B`;
  }
  if (absValue >= 1_000_000) {
    return `${sign}${(absValue / 1_000_000).toFixed(2)}M`;
  }
  if (absValue >= 1_000) {
    return `${sign}${(absValue / 1_000).toFixed(2)}K`;
  }
  return `${sign}${absValue.toFixed(2)}`;
}

/**
 * Format volume with abbreviated suffixes
 */
export function formatVolume(value: number): string {
  return formatCompact(value);
}

/**
 * Format a date for display
 * @param date - ISO date string or Date object
 * @param format - Format type: 'short' (Jan 15), 'medium' (Jan 15, 2024), 'long' (January 15, 2024)
 */
export function formatDate(
  date: string | Date,
  format: 'short' | 'medium' | 'long' = 'short'
): string {
  const d = typeof date === 'string' ? new Date(date) : date;

  switch (format) {
    case 'short':
      return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    case 'medium':
      return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
    case 'long':
      return d.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' });
    default:
      return d.toLocaleDateString('en-US');
  }
}

/**
 * Create a price formatter function for lightweight-charts
 */
export function createPriceFormatter(decimals: number = 2): (price: number) => string {
  return (price: number) => formatPrice(price, decimals);
}

/**
 * Create a currency formatter function for lightweight-charts
 */
export function createCurrencyFormatter(decimals: number = 2): (price: number) => string {
  return (price: number) => formatCurrency(price, decimals);
}

/**
 * Create a percent formatter function for lightweight-charts
 */
export function createPercentFormatter(decimals: number = 2): (price: number) => string {
  return (price: number) => formatPercent(price, decimals);
}

/**
 * Create a volume formatter function for lightweight-charts
 */
export function createVolumeFormatter(): (price: number) => string {
  return (price: number) => formatVolume(price);
}

/**
 * Format a Unix timestamp or date string for chart display
 * @param time - Unix timestamp (seconds) or date string
 * @param includeTime - Whether to include time portion
 */
export function formatChartTime(time: number | string, includeTime: boolean = false): string {
  let date: Date;

  if (typeof time === 'number') {
    // Unix timestamp in seconds
    date = new Date(time * 1000);
  } else if (typeof time === 'string') {
    // Date string (YYYY-MM-DD or ISO)
    date = new Date(time);
  } else {
    return '';
  }

  if (includeTime) {
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  }

  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });
}
