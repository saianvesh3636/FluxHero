'use client';

import React, { Component, ErrorInfo, ReactNode } from 'react';

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  resetKeys?: Array<string | number>;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

/**
 * ErrorBoundary component for catching React errors and displaying fallback UI
 *
 * Features:
 * - Catches JavaScript errors anywhere in child component tree
 * - Logs error information for debugging
 * - Displays fallback UI when errors occur
 * - Supports custom error handlers
 * - Can reset error state when props change (resetKeys)
 *
 * Usage:
 * ```tsx
 * <ErrorBoundary fallback={<ErrorFallback />}>
 *   <YourComponent />
 * </ErrorBoundary>
 * ```
 */
class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    // Update state so the next render will show the fallback UI
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    // Log error to console for development
    console.error('ErrorBoundary caught an error:', error, errorInfo);

    // Update state with error details
    this.setState({
      error,
      errorInfo,
    });

    // Call custom error handler if provided
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }
  }

  componentDidUpdate(prevProps: ErrorBoundaryProps): void {
    // Reset error state when resetKeys change
    if (this.state.hasError && this.props.resetKeys) {
      const prevResetKeys = prevProps.resetKeys || [];
      const currentResetKeys = this.props.resetKeys;

      const keysChanged = currentResetKeys.some(
        (key, index) => key !== prevResetKeys[index]
      );

      if (keysChanged) {
        this.resetErrorBoundary();
      }
    }
  }

  resetErrorBoundary = (): void => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
  };

  render(): ReactNode {
    if (this.state.hasError) {
      // Render custom fallback if provided
      if (this.props.fallback) {
        return this.props.fallback;
      }

      // Render default fallback UI
      return (
        <DefaultErrorFallback
          error={this.state.error}
          errorInfo={this.state.errorInfo}
          resetErrorBoundary={this.resetErrorBoundary}
        />
      );
    }

    return this.props.children;
  }
}

interface DefaultErrorFallbackProps {
  error: Error | null;
  errorInfo: ErrorInfo | null;
  resetErrorBoundary: () => void;
}

/**
 * Default error fallback UI component
 */
function DefaultErrorFallback({
  error,
  errorInfo,
  resetErrorBoundary,
}: DefaultErrorFallbackProps): JSX.Element {
  return (
    <div
      style={{
        padding: '2rem',
        margin: '2rem',
        border: '2px solid #ef4444',
        borderRadius: '8px',
        backgroundColor: '#fef2f2',
        color: '#991b1b',
      }}
    >
      <h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', marginBottom: '1rem' }}>
        Something went wrong
      </h2>
      <p style={{ marginBottom: '1rem' }}>
        An unexpected error occurred. The error has been logged for investigation.
      </p>

      {error && (
        <details style={{ marginBottom: '1rem' }}>
          <summary style={{ cursor: 'pointer', fontWeight: 'bold' }}>
            Error Details
          </summary>
          <pre
            style={{
              marginTop: '0.5rem',
              padding: '1rem',
              backgroundColor: '#fff',
              border: '1px solid #fca5a5',
              borderRadius: '4px',
              overflow: 'auto',
              fontSize: '0.875rem',
            }}
          >
            {error.toString()}
            {errorInfo && '\n\n'}
            {errorInfo?.componentStack}
          </pre>
        </details>
      )}

      <button
        onClick={resetErrorBoundary}
        style={{
          padding: '0.5rem 1rem',
          backgroundColor: '#dc2626',
          color: '#fff',
          border: 'none',
          borderRadius: '4px',
          cursor: 'pointer',
          fontSize: '1rem',
          fontWeight: '500',
        }}
      >
        Try Again
      </button>
    </div>
  );
}

export default ErrorBoundary;
