/**
 * ChartContainer - Base wrapper component for all charts
 *
 * Provides:
 * - Loading state overlay
 * - Consistent sizing and styling
 * - Ref forwarding for composed charts
 */

'use client';

import React, { forwardRef } from 'react';
import { CHART_HEIGHT } from '../config/constants';

export interface ChartContainerProps {
  /** Chart height - can be a named size or pixel value */
  height?: keyof typeof CHART_HEIGHT | number;
  /** Whether the chart is still loading */
  isLoading?: boolean;
  /** Additional CSS classes */
  className?: string;
  /** Optional title displayed above the chart */
  title?: string;
  /** Children (usually the chart canvas container) */
  children: React.ReactNode;
}

/**
 * Base container component for charts
 */
export const ChartContainer = forwardRef<HTMLDivElement, ChartContainerProps>(
  function ChartContainer(
    { height = 'md', isLoading = false, className = '', title, children },
    ref
  ) {
    // Resolve height value
    const resolvedHeight =
      typeof height === 'number' ? height : CHART_HEIGHT[height];

    return (
      <div className={`relative w-full ${className}`}>
        {title && (
          <h3 className="text-sm font-medium text-text-400 mb-2">{title}</h3>
        )}

        <div
          ref={ref}
          className="relative w-full bg-panel-700 rounded overflow-hidden"
          style={{ height: resolvedHeight }}
        >
          {children}

          {/* Loading overlay */}
          {isLoading && (
            <div className="absolute inset-0 bg-panel-700 flex items-center justify-center z-10">
              <div className="flex items-center gap-2 text-text-400">
                <div className="w-4 h-4 border-2 border-text-400 border-t-transparent rounded-full animate-spin" />
                <span className="text-sm">Loading chart...</span>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }
);

export default ChartContainer;
