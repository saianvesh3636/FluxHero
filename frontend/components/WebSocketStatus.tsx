/**
 * WebSocket Status Indicator Component
 * Displays current WebSocket connection status with visual feedback
 */

'use client';

import React from 'react';
import { useWebSocketContext } from '../contexts/WebSocketContext';
import { WebSocketState } from '../hooks/useWebSocket';

interface WebSocketStatusProps {
  /** Show detailed status text (default: false) */
  showText?: boolean;
  /** Custom className for styling */
  className?: string;
}

/**
 * WebSocket Status Indicator
 * Shows connection state with color-coded status indicators
 *
 * @example
 * ```tsx
 * // Simple indicator (icon only)
 * <WebSocketStatus />
 *
 * // With status text
 * <WebSocketStatus showText />
 *
 * // Custom styling
 * <WebSocketStatus className="my-custom-class" showText />
 * ```
 */
export function WebSocketStatus({ showText = false, className = '' }: WebSocketStatusProps) {
  const { connectionState, error, reconnectAttempts, reconnect } = useWebSocketContext();

  const getStatusConfig = () => {
    switch (connectionState) {
      case WebSocketState.CONNECTED:
        return {
          color: 'bg-green-500',
          text: 'Connected',
          emoji: '🟢',
        };
      case WebSocketState.CONNECTING:
        return {
          color: 'bg-yellow-500',
          text: 'Connecting...',
          emoji: '🟡',
        };
      case WebSocketState.RECONNECTING:
        return {
          color: 'bg-orange-500',
          text: `Reconnecting (${reconnectAttempts})...`,
          emoji: '🟠',
        };
      case WebSocketState.DISCONNECTED:
        return {
          color: 'bg-gray-500',
          text: 'Disconnected',
          emoji: '⚪',
        };
      case WebSocketState.FAILED:
        return {
          color: 'bg-red-500',
          text: 'Connection Failed',
          emoji: '🔴',
        };
      default:
        return {
          color: 'bg-gray-500',
          text: 'Unknown',
          emoji: '⚪',
        };
    }
  };

  const statusConfig = getStatusConfig();
  const showRetry = connectionState === WebSocketState.FAILED;

  return (
    <div className={`flex items-center gap-2 ${className}`}>
      {/* Status indicator */}
      <div className="flex items-center gap-2">
        <span className="text-lg" role="img" aria-label={statusConfig.text}>
          {statusConfig.emoji}
        </span>
        {showText && (
          <span className="text-sm font-medium">
            {statusConfig.text}
          </span>
        )}
      </div>

      {/* Error message */}
      {error && showText && (
        <span className="text-xs text-red-600 dark:text-red-400">
          {error}
        </span>
      )}

      {/* Retry button */}
      {showRetry && (
        <button
          onClick={reconnect}
          className="px-2 py-1 text-xs bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
          aria-label="Retry connection"
        >
          Retry
        </button>
      )}
    </div>
  );
}

/**
 * WebSocket Status Badge (Compact Version)
 * Minimal badge for header/navbar usage
 */
export function WebSocketStatusBadge({ className = '' }: { className?: string }) {
  const { connectionState } = useWebSocketContext();

  const getStatusConfig = () => {
    switch (connectionState) {
      case WebSocketState.CONNECTED:
        return { color: 'bg-green-500', pulse: false };
      case WebSocketState.CONNECTING:
      case WebSocketState.RECONNECTING:
        return { color: 'bg-yellow-500', pulse: true };
      case WebSocketState.DISCONNECTED:
        return { color: 'bg-gray-500', pulse: false };
      case WebSocketState.FAILED:
        return { color: 'bg-red-500', pulse: false };
      default:
        return { color: 'bg-gray-500', pulse: false };
    }
  };

  const statusConfig = getStatusConfig();

  return (
    <div
      className={`w-3 h-3 rounded-full ${statusConfig.color} ${
        statusConfig.pulse ? 'animate-pulse' : ''
      } ${className}`}
      title={connectionState}
      role="status"
      aria-label={`WebSocket status: ${connectionState}`}
    />
  );
}
