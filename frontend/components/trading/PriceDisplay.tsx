import React from 'react';
import { cn, formatPrice, formatPercent } from '../../lib/utils';

export interface PriceDisplayProps {
  price: number;
  previousPrice?: number;
  change?: number;
  changePercent?: number;
  decimals?: number;
  showCurrency?: boolean;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  className?: string;
}

/**
 * PriceDisplay - displays a price with optional change indicator
 * Follows design system: tabular-nums, color-coded changes
 */
export function PriceDisplay({
  price,
  previousPrice,
  change,
  changePercent,
  decimals = 2,
  showCurrency = true,
  size = 'md',
  className,
}: PriceDisplayProps) {
  // Calculate change if previous price provided
  const actualChange = change ?? (previousPrice ? price - previousPrice : 0);
  const actualChangePercent = changePercent ?? (previousPrice ? ((price - previousPrice) / previousPrice) * 100 : 0);

  const isPositive = actualChange > 0;
  const isNegative = actualChange < 0;

  const sizeClasses = {
    sm: 'text-sm',
    md: 'text-base',
    lg: 'text-xl',
    xl: 'text-2xl',
  };

  const changeColor = isPositive
    ? 'text-profit-500'
    : isNegative
    ? 'text-loss-500'
    : 'text-text-500';

  return (
    <div className={cn('font-mono tabular-nums', className)}>
      <span className={cn('text-text-900', sizeClasses[size])}>
        {showCurrency && '$'}
        {formatPrice(price, decimals)}
      </span>
      {(previousPrice || change !== undefined) && (
        <span className={cn('ml-2 text-sm', changeColor)}>
          {isPositive && '+'}
          {formatPrice(actualChange, decimals)}
          {actualChangePercent !== 0 && (
            <span className="ml-1">
              ({isPositive && '+'}
              {formatPercent(actualChangePercent)})
            </span>
          )}
        </span>
      )}
    </div>
  );
}

export default PriceDisplay;
