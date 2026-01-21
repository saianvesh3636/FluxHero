/**
 * Loading Spinner Component
 *
 * Reusable loading spinner with various sizes and styles
 * Used across the application for async operation feedback
 */

import React from 'react';

export type LoadingSize = 'sm' | 'md' | 'lg' | 'xl';
export type LoadingVariant = 'spinner' | 'dots' | 'pulse' | 'skeleton';

interface LoadingSpinnerProps {
  size?: LoadingSize;
  variant?: LoadingVariant;
  message?: string;
  fullScreen?: boolean;
  className?: string;
}

const sizeClasses = {
  sm: 'h-4 w-4',
  md: 'h-8 w-8',
  lg: 'h-12 w-12',
  xl: 'h-16 w-16',
};

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 'md',
  variant = 'spinner',
  message,
  fullScreen = false,
  className = '',
}) => {
  const containerClass = fullScreen
    ? 'fixed inset-0 bg-gray-900 bg-opacity-50 flex items-center justify-center z-50'
    : 'flex items-center justify-center';

  const renderSpinner = () => {
    switch (variant) {
      case 'spinner':
        return (
          <div
            className={`animate-spin rounded-full border-b-2 border-blue-500 ${sizeClasses[size]} ${className}`}
            role="status"
            aria-label="Loading"
          />
        );

      case 'dots':
        return (
          <div className="flex space-x-2" role="status" aria-label="Loading">
            <div className="w-3 h-3 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
            <div className="w-3 h-3 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
            <div className="w-3 h-3 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
          </div>
        );

      case 'pulse':
        return (
          <div
            className={`bg-blue-500 rounded-full animate-pulse ${sizeClasses[size]} ${className}`}
            role="status"
            aria-label="Loading"
          />
        );

      case 'skeleton':
        return (
          <div className="space-y-3 w-full max-w-md" role="status" aria-label="Loading">
            <div className="h-4 bg-gray-700 rounded animate-pulse"></div>
            <div className="h-4 bg-gray-700 rounded animate-pulse w-5/6"></div>
            <div className="h-4 bg-gray-700 rounded animate-pulse w-4/6"></div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className={containerClass}>
      <div className="flex flex-col items-center space-y-4">
        {renderSpinner()}
        {message && (
          <p className="text-gray-400 text-sm md:text-base animate-pulse">
            {message}
          </p>
        )}
      </div>
    </div>
  );
};

export default LoadingSpinner;

/**
 * Loading Overlay Component
 *
 * Overlays a loading spinner on top of content
 */
interface LoadingOverlayProps {
  isLoading: boolean;
  message?: string;
  children: React.ReactNode;
}

export const LoadingOverlay: React.FC<LoadingOverlayProps> = ({
  isLoading,
  message = 'Loading...',
  children,
}) => {
  return (
    <div className="relative">
      {children}
      {isLoading && (
        <div className="absolute inset-0 bg-gray-900 bg-opacity-75 flex items-center justify-center z-10 rounded-lg">
          <LoadingSpinner size="lg" message={message} />
        </div>
      )}
    </div>
  );
};

/**
 * Skeleton Loader for Cards
 */
export const SkeletonCard: React.FC = () => {
  return (
    <div className="bg-gray-800 rounded-lg p-6 animate-pulse">
      <div className="h-4 bg-gray-700 rounded w-1/3 mb-4"></div>
      <div className="h-8 bg-gray-700 rounded w-1/2 mb-2"></div>
      <div className="h-3 bg-gray-700 rounded w-2/3"></div>
    </div>
  );
};

/**
 * Skeleton Loader for Table
 */
export const SkeletonTable: React.FC<{ rows?: number; cols?: number }> = ({
  rows = 5,
  cols = 6,
}) => {
  return (
    <div className="bg-white rounded-lg shadow overflow-hidden animate-pulse">
      <div className="px-6 py-4 border-b border-gray-200">
        <div className="h-6 bg-gray-200 rounded w-1/4"></div>
      </div>
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              {Array.from({ length: cols }).map((_, idx) => (
                <th key={idx} className="px-6 py-3">
                  <div className="h-4 bg-gray-200 rounded"></div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {Array.from({ length: rows }).map((_, rowIdx) => (
              <tr key={rowIdx}>
                {Array.from({ length: cols }).map((_, colIdx) => (
                  <td key={colIdx} className="px-6 py-4">
                    <div className="h-4 bg-gray-100 rounded"></div>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

/**
 * Button Loading State
 */
interface LoadingButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  isLoading: boolean;
  loadingText?: string;
  children: React.ReactNode;
}

export const LoadingButton: React.FC<LoadingButtonProps> = ({
  isLoading,
  loadingText,
  children,
  disabled,
  className = '',
  ...props
}) => {
  return (
    <button
      {...props}
      disabled={disabled || isLoading}
      className={`relative ${className}`}
    >
      {isLoading && (
        <span className="absolute inset-0 flex items-center justify-center">
          <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
          {loadingText && <span>{loadingText}</span>}
        </span>
      )}
      <span className={isLoading ? 'invisible' : ''}>{children}</span>
    </button>
  );
};
