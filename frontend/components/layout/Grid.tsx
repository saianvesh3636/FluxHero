import React from 'react';
import { cn } from '../../lib/utils';

export interface GridProps extends React.HTMLAttributes<HTMLDivElement> {
  columns?: 1 | 2 | 3 | 4 | 'auto';
  gap?: 'sm' | 'md' | 'lg';
}

/**
 * Grid component - responsive grid layout
 * Follows design system: 20px gaps, responsive columns
 */
export function Grid({
  className,
  columns = 'auto',
  gap = 'md',
  children,
  ...props
}: GridProps) {
  const columnClasses = {
    1: 'grid-cols-1',
    2: 'grid-cols-1 sm:grid-cols-2',
    3: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3',
    4: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-4',
    auto: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4',
  };

  const gapClasses = {
    sm: 'gap-3',
    md: 'gap-5',
    lg: 'gap-8',
  };

  return (
    <div
      className={cn(
        'grid',
        columnClasses[columns],
        gapClasses[gap],
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}

export interface StatsGridProps extends React.HTMLAttributes<HTMLDivElement> {
  columns?: 2 | 3 | 4 | 5;
}

/**
 * StatsGrid - specialized grid for stat cards
 * Responsive: 1 col mobile, 2 col tablet, specified cols desktop
 */
export function StatsGrid({
  className,
  columns = 4,
  children,
  ...props
}: StatsGridProps) {
  const columnClasses = {
    2: 'grid-cols-1 sm:grid-cols-2',
    3: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3',
    4: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-4',
    5: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5',
  };

  return (
    <div
      className={cn(
        'grid gap-5',
        columnClasses[columns],
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}

export interface CardGridProps extends React.HTMLAttributes<HTMLDivElement> {
  columns?: 2 | 3 | 4;
}

/**
 * CardGrid - specialized grid for card layouts
 */
export function CardGrid({
  className,
  columns = 3,
  children,
  ...props
}: CardGridProps) {
  const columnClasses = {
    2: 'grid-cols-1 md:grid-cols-2',
    3: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3',
    4: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4',
  };

  return (
    <div
      className={cn(
        'grid gap-5',
        columnClasses[columns],
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}

export default Grid;
