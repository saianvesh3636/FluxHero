import React from 'react';
import { cn } from '../../lib/utils';

export interface PageHeaderProps extends React.HTMLAttributes<HTMLDivElement> {
  title: string;
  subtitle?: string;
  actions?: React.ReactNode;
}

/**
 * PageHeader - consistent page header with title, subtitle, and optional actions
 * Follows design system: text-3xl title, text-lg subtitle
 */
export function PageHeader({
  className,
  title,
  subtitle,
  actions,
  ...props
}: PageHeaderProps) {
  return (
    <div
      className={cn('mb-8', className)}
      {...props}
    >
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-semibold text-text-900">
            {title}
          </h1>
          {subtitle && (
            <p className="text-lg text-text-400 mt-1">
              {subtitle}
            </p>
          )}
        </div>
        {actions && (
          <div className="flex items-center gap-2">
            {actions}
          </div>
        )}
      </div>
    </div>
  );
}

export default PageHeader;
