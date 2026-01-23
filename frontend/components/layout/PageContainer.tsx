import React from 'react';
import { cn } from '../../lib/utils';

export interface PageContainerProps extends React.HTMLAttributes<HTMLDivElement> {
  maxWidth?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
  noPadding?: boolean;
}

/**
 * PageContainer - wraps page content with consistent max-width and padding
 * Follows design system: max-width 1280px, 24px padding
 */
export function PageContainer({
  className,
  maxWidth = 'xl',
  noPadding = false,
  children,
  ...props
}: PageContainerProps) {
  const maxWidthClasses = {
    sm: 'max-w-2xl',
    md: 'max-w-4xl',
    lg: 'max-w-6xl',
    xl: 'max-w-7xl',
    full: 'max-w-full',
  };

  return (
    <div
      className={cn(
        'mx-auto w-full',
        maxWidthClasses[maxWidth],
        !noPadding && 'px-4 py-6 sm:px-6',
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}

export default PageContainer;
