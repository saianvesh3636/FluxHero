import React from 'react';
import { cn } from '../../lib/utils';

export interface SkeletonProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'text' | 'title' | 'card' | 'circle' | 'custom';
  width?: string | number;
  height?: string | number;
}

/**
 * Skeleton component for loading states
 * Follows design system: static (no shimmer animation), panel-500 background
 */
export function Skeleton({
  className,
  variant = 'custom',
  width,
  height,
  style,
  ...props
}: SkeletonProps) {
  const variantClasses = {
    text: 'h-4 w-full rounded',
    title: 'h-6 w-3/4 rounded',
    card: 'h-32 w-full rounded-2xl',
    circle: 'rounded-full',
    custom: 'rounded',
  };

  return (
    <div
      className={cn(
        'bg-panel-500',
        variantClasses[variant],
        className
      )}
      style={{
        width: width,
        height: height,
        ...style,
      }}
      {...props}
    />
  );
}

export interface SkeletonTextProps extends React.HTMLAttributes<HTMLDivElement> {
  lines?: number;
  lastLineWidth?: string;
}

/**
 * SkeletonText for multiple lines of text loading
 */
export function SkeletonText({
  className,
  lines = 3,
  lastLineWidth = '60%',
  ...props
}: SkeletonTextProps) {
  return (
    <div className={cn('space-y-2', className)} {...props}>
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton
          key={i}
          variant="text"
          style={{
            width: i === lines - 1 ? lastLineWidth : '100%',
          }}
        />
      ))}
    </div>
  );
}

export interface SkeletonCardProps extends React.HTMLAttributes<HTMLDivElement> {
  hasHeader?: boolean;
  hasContent?: boolean;
}

/**
 * SkeletonCard for card loading state
 */
export function SkeletonCard({
  className,
  hasHeader = true,
  hasContent = true,
  ...props
}: SkeletonCardProps) {
  return (
    <div
      className={cn('bg-panel-600 rounded-2xl p-5', className)}
      {...props}
    >
      {hasHeader && (
        <div className="mb-4">
          <Skeleton variant="title" className="mb-2" />
          <Skeleton variant="text" width="40%" />
        </div>
      )}
      {hasContent && (
        <SkeletonText lines={3} />
      )}
    </div>
  );
}

export interface SkeletonTableProps extends React.HTMLAttributes<HTMLDivElement> {
  rows?: number;
  columns?: number;
}

/**
 * SkeletonTable for table loading state
 */
export function SkeletonTable({
  className,
  rows = 5,
  columns = 4,
  ...props
}: SkeletonTableProps) {
  return (
    <div className={cn('overflow-x-auto', className)} {...props}>
      <div className="min-w-full">
        {/* Header */}
        <div className="flex gap-4 bg-panel-700 p-3 rounded-t">
          {Array.from({ length: columns }).map((_, i) => (
            <Skeleton key={i} variant="text" width={i === 0 ? '20%' : '15%'} />
          ))}
        </div>
        {/* Rows */}
        {Array.from({ length: rows }).map((_, rowIndex) => (
          <div
            key={rowIndex}
            className="flex gap-4 p-3 border-b border-panel-600"
          >
            {Array.from({ length: columns }).map((_, colIndex) => (
              <Skeleton
                key={colIndex}
                variant="text"
                width={colIndex === 0 ? '20%' : '15%'}
              />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

export default Skeleton;
