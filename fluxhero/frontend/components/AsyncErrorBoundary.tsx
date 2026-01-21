'use client';

import React, { ReactNode, useState, useEffect } from 'react';
import ErrorBoundary from './ErrorBoundary';
import { APIErrorFallback, DataLoadErrorFallback } from './ErrorFallback';

interface AsyncErrorBoundaryProps {
  children: ReactNode;
  isLoading?: boolean;
  error?: Error | null;
  onRetry?: () => void;
  fallbackType?: 'api' | 'data' | 'default';
}

/**
 * Error boundary specialized for async operations
 *
 * Handles errors from async data fetching and provides appropriate fallback UI.
 * Can be used with loading states and retry logic.
 *
 * Features:
 * - Different fallback UIs for API/data errors
 * - Retry functionality
 * - Loading state integration
 * - Automatic error reset on retry
 *
 * Usage:
 * ```tsx
 * const { data, error, isLoading, refetch } = useQuery();
 *
 * <AsyncErrorBoundary
 *   isLoading={isLoading}
 *   error={error}
 *   onRetry={refetch}
 *   fallbackType="api"
 * >
 *   <YourComponent data={data} />
 * </AsyncErrorBoundary>
 * ```
 */
export default function AsyncErrorBoundary({
  children,
  isLoading = false,
  error = null,
  onRetry,
  fallbackType = 'default',
}: AsyncErrorBoundaryProps): JSX.Element {
  const [resetKey, setResetKey] = useState(0);

  // Reset error boundary when retry is triggered
  const handleRetry = (): void => {
    setResetKey((prev) => prev + 1);
    if (onRetry) {
      onRetry();
    }
  };

  // Show error fallback if async error occurred
  if (error && !isLoading) {
    const FallbackComponent =
      fallbackType === 'api'
        ? APIErrorFallback
        : fallbackType === 'data'
        ? DataLoadErrorFallback
        : null;

    if (FallbackComponent) {
      return <FallbackComponent resetErrorBoundary={handleRetry} />;
    }
  }

  // Wrap children in error boundary for runtime errors
  return (
    <ErrorBoundary key={resetKey} resetKeys={[resetKey]}>
      {children}
    </ErrorBoundary>
  );
}

/**
 * Hook for managing async error boundaries
 *
 * Example:
 * ```tsx
 * const { error, setError, resetError } = useAsyncError();
 * ```
 */
export function useAsyncError() {
  const [error, setError] = useState<Error | null>(null);

  const resetError = (): void => {
    setError(null);
  };

  const throwError = (err: Error): void => {
    setError(err);
  };

  return { error, setError: throwError, resetError };
}
