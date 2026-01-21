'use client';

import React from 'react';

interface ErrorFallbackProps {
  error?: Error | null;
  resetErrorBoundary?: () => void;
  title?: string;
  message?: string;
}

/**
 * Customizable error fallback component for use with ErrorBoundary
 *
 * Features:
 * - Clean, user-friendly error display
 * - Customizable title and message
 * - Optional error details in collapsible section
 * - Reset button to retry
 * - Responsive design
 * - Theme-aware styling
 */
export default function ErrorFallback({
  error,
  resetErrorBoundary,
  title = 'Something went wrong',
  message = 'An unexpected error occurred. Please try again.',
}: ErrorFallbackProps): JSX.Element {
  return (
    <div className="error-fallback-container">
      <div className="error-fallback-card">
        <div className="error-icon">⚠️</div>
        <h2 className="error-title">{title}</h2>
        <p className="error-message">{message}</p>

        {error && (
          <details className="error-details">
            <summary className="error-details-summary">Technical Details</summary>
            <pre className="error-details-content">
              {error.name}: {error.message}
              {error.stack && `\n\n${error.stack}`}
            </pre>
          </details>
        )}

        {resetErrorBoundary && (
          <button onClick={resetErrorBoundary} className="error-reset-button">
            Try Again
          </button>
        )}
      </div>

      <style jsx>{`
        .error-fallback-container {
          display: flex;
          align-items: center;
          justify-content: center;
          min-height: 400px;
          padding: 2rem;
        }

        .error-fallback-card {
          max-width: 600px;
          padding: 2rem;
          border: 2px solid var(--error-border, #ef4444);
          border-radius: 12px;
          background-color: var(--error-bg, #fef2f2);
          color: var(--error-text, #991b1b);
          text-align: center;
        }

        .error-icon {
          font-size: 3rem;
          margin-bottom: 1rem;
        }

        .error-title {
          font-size: 1.5rem;
          font-weight: bold;
          margin-bottom: 1rem;
        }

        .error-message {
          font-size: 1rem;
          margin-bottom: 1.5rem;
          color: var(--error-text-secondary, #7f1d1d);
        }

        .error-details {
          margin-bottom: 1.5rem;
          text-align: left;
        }

        .error-details-summary {
          cursor: pointer;
          font-weight: 600;
          color: var(--error-text, #991b1b);
          margin-bottom: 0.5rem;
        }

        .error-details-summary:hover {
          text-decoration: underline;
        }

        .error-details-content {
          margin-top: 0.5rem;
          padding: 1rem;
          background-color: var(--error-details-bg, #fff);
          border: 1px solid var(--error-border-light, #fca5a5);
          border-radius: 6px;
          overflow: auto;
          font-size: 0.875rem;
          font-family: 'Courier New', monospace;
          max-height: 300px;
        }

        .error-reset-button {
          padding: 0.75rem 1.5rem;
          background-color: var(--error-button-bg, #dc2626);
          color: white;
          border: none;
          border-radius: 6px;
          cursor: pointer;
          font-size: 1rem;
          font-weight: 600;
          transition: background-color 0.2s;
        }

        .error-reset-button:hover {
          background-color: var(--error-button-hover, #b91c1c);
        }

        .error-reset-button:active {
          transform: scale(0.98);
        }

        @media (max-width: 640px) {
          .error-fallback-container {
            padding: 1rem;
          }

          .error-fallback-card {
            padding: 1.5rem;
          }

          .error-title {
            font-size: 1.25rem;
          }

          .error-message {
            font-size: 0.875rem;
          }
        }
      `}</style>
    </div>
  );
}

/**
 * Specific error fallback for API/network errors
 */
export function APIErrorFallback({
  resetErrorBoundary,
}: {
  resetErrorBoundary?: () => void;
}): JSX.Element {
  return (
    <ErrorFallback
      title="Connection Error"
      message="Unable to connect to the server. Please check your internet connection and try again."
      resetErrorBoundary={resetErrorBoundary}
    />
  );
}

/**
 * Specific error fallback for data loading errors
 */
export function DataLoadErrorFallback({
  resetErrorBoundary,
}: {
  resetErrorBoundary?: () => void;
}): JSX.Element {
  return (
    <ErrorFallback
      title="Data Load Error"
      message="Failed to load data. This might be a temporary issue. Please try again."
      resetErrorBoundary={resetErrorBoundary}
    />
  );
}

/**
 * Specific error fallback for component errors
 */
export function ComponentErrorFallback({
  error,
  resetErrorBoundary,
}: ErrorFallbackProps): JSX.Element {
  return (
    <ErrorFallback
      error={error}
      title="Component Error"
      message="This component encountered an error. Our team has been notified."
      resetErrorBoundary={resetErrorBoundary}
    />
  );
}
