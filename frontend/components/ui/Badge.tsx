import React from 'react';
import { cn } from '../../lib/utils';

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: 'success' | 'error' | 'warning' | 'info' | 'neutral';
  size?: 'sm' | 'md';
}

/**
 * Badge component for status indicators and labels
 * Follows design system: 20% opacity background with solid text color
 */
export function Badge({
  className,
  variant = 'neutral',
  size = 'sm',
  children,
  ...props
}: BadgeProps) {
  const variantClasses = {
    success: 'bg-profit-500/20 text-profit-500',
    error: 'bg-loss-500/20 text-loss-500',
    warning: 'bg-warning-500/20 text-warning-500',
    info: 'bg-blue-500/20 text-blue-500',
    neutral: 'bg-panel-400 text-text-400',
  };

  const sizeClasses = {
    sm: 'text-xs px-2 py-0.5',
    md: 'text-sm px-2.5 py-1',
  };

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded font-medium',
        variantClasses[variant],
        sizeClasses[size],
        className
      )}
      {...props}
    >
      {children}
    </span>
  );
}

export default Badge;
