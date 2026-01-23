'use client';

import React, { ReactNode } from 'react';
import ErrorBoundary from './ErrorBoundary';
import ErrorFallback from './ErrorFallback';

interface PageErrorBoundaryProps {
  children: ReactNode;
  pageName?: string;
}

/**
 * Page-level error boundary wrapper
 *
 * Wraps individual pages with error handling and custom fallback UI.
 * Logs errors specific to the page for easier debugging.
 *
 * Usage:
 * ```tsx
 * export default function MyPage() {
 *   return (
 *     <PageErrorBoundary pageName="Analytics">
 *       <YourPageContent />
 *     </PageErrorBoundary>
 *   );
 * }
 * ```
 */
export default function PageErrorBoundary({
  children,
  pageName = 'Page',
}: PageErrorBoundaryProps): React.ReactElement {
  const handleError = (error: Error): void => {
    // Log error with page context
    console.error(`[${pageName}] Error occurred:`, error);

    // In production, you could send error to logging service
    // Example: logErrorToService({ page: pageName, error });
  };

  return (
    <ErrorBoundary
      fallback={
        <ErrorFallback
          title={`${pageName} Error`}
          message={`An error occurred while loading the ${pageName} page. Please try refreshing.`}
        />
      }
      onError={handleError}
    >
      {children}
    </ErrorBoundary>
  );
}
