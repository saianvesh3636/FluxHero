import React from 'react';
import { cn, formatCurrency, formatPercent, getPLColorClass } from '../../lib/utils';

export interface PLDisplayProps {
  value: number;
  percent?: number;
  showSign?: boolean;
  showPercent?: boolean;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  className?: string;
}

/**
 * PLDisplay - displays P&L value with color coding
 * Follows design system: profit-500 for positive, loss-500 for negative
 */
export function PLDisplay({
  value,
  percent,
  showSign = true,
  showPercent = true,
  size = 'md',
  className,
}: PLDisplayProps) {
  const colorClass = getPLColorClass(value);

  const sizeClasses = {
    sm: 'text-sm',
    md: 'text-base',
    lg: 'text-xl',
    xl: 'text-2xl',
  };

  return (
    <div className={cn('font-mono tabular-nums', colorClass, sizeClasses[size], className)}>
      <span>
        {formatCurrency(value, showSign)}
      </span>
      {showPercent && percent !== undefined && (
        <span className="ml-1 text-sm">
          ({formatPercent(percent, showSign)})
        </span>
      )}
    </div>
  );
}

/**
 * PLBadge - compact P&L display for inline use
 */
export function PLBadge({
  value,
  className,
}: {
  value: number;
  className?: string;
}) {
  const isPositive = value > 0;
  const isNegative = value < 0;

  return (
    <span
      className={cn(
        'inline-flex items-center px-2 py-0.5 rounded text-xs font-medium font-mono tabular-nums',
        isPositive && 'bg-profit-500/20 text-profit-500',
        isNegative && 'bg-loss-500/20 text-loss-500',
        !isPositive && !isNegative && 'bg-panel-400 text-text-400',
        className
      )}
    >
      {formatCurrency(value, true)}
    </span>
  );
}

export default PLDisplay;
