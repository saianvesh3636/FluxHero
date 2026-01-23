import React from 'react';
import { cn } from '../../lib/utils';

export interface StatusDotProps extends React.HTMLAttributes<HTMLSpanElement> {
  status: 'connected' | 'disconnected' | 'connecting' | 'error' | 'idle';
  size?: 'sm' | 'md' | 'lg';
  label?: string;
  showLabel?: boolean;
}

/**
 * StatusDot component for connection/status indicators
 * Follows design system: solid colors, no animations
 */
export function StatusDot({
  className,
  status,
  size = 'md',
  label,
  showLabel = false,
  ...props
}: StatusDotProps) {
  const statusClasses = {
    connected: 'bg-profit-500',
    disconnected: 'bg-loss-500',
    connecting: 'bg-warning-500',
    error: 'bg-loss-500',
    idle: 'bg-text-300',
  };

  const sizeClasses = {
    sm: 'w-1.5 h-1.5',
    md: 'w-2 h-2',
    lg: 'w-3 h-3',
  };

  const statusLabels = {
    connected: 'Connected',
    disconnected: 'Disconnected',
    connecting: 'Connecting',
    error: 'Error',
    idle: 'Idle',
  };

  const displayLabel = label || statusLabels[status];

  return (
    <span
      className={cn('inline-flex items-center gap-2', className)}
      {...props}
    >
      <span
        className={cn(
          'rounded-full',
          statusClasses[status],
          sizeClasses[size]
        )}
      />
      {showLabel && (
        <span className="text-sm text-text-400">{displayLabel}</span>
      )}
    </span>
  );
}

export default StatusDot;
