/**
 * Custom React hook for WebSocket connection management
 * Provides automatic reconnection, connection state tracking, and lifecycle management
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import { getAuthToken } from '../utils/api';

export interface PriceUpdate {
  symbol: string;
  price: number;
  timestamp: string;
  volume?: number;
  bid?: number;
  ask?: number;
  // Additional fields from CSV replay mode
  open?: number;
  high?: number;
  low?: number;
  close?: number;
  type?: string;
  replay_index?: number;
  total_rows?: number;
}

export interface WebSocketOptions {
  /** Auto-reconnect on connection loss (default: true) */
  autoReconnect?: boolean;
  /** Maximum number of reconnection attempts (default: 5) */
  maxReconnectAttempts?: number;
  /** Initial reconnection delay in ms (default: 1000) */
  reconnectDelay?: number;
  /** Maximum reconnection delay in ms (default: 30000) */
  maxReconnectDelay?: number;
  /** Authentication token (if not set, uses getAuthToken from api.ts) */
  authToken?: string | null;
  /** Callback for connection open event */
  onOpen?: () => void;
  /** Callback for connection close event */
  onClose?: () => void;
  /** Callback for connection error event */
  onError?: (error: Event) => void;
}

export enum WebSocketState {
  CONNECTING = 'CONNECTING',
  CONNECTED = 'CONNECTED',
  DISCONNECTED = 'DISCONNECTED',
  RECONNECTING = 'RECONNECTING',
  FAILED = 'FAILED',
}

export interface UseWebSocketReturn {
  /** Current connection state */
  state: WebSocketState;
  /** Latest price update received */
  data: PriceUpdate | null;
  /** Error message if connection failed */
  error: string | null;
  /** Number of reconnection attempts made */
  reconnectAttempts: number;
  /** Manually reconnect to WebSocket */
  reconnect: () => void;
  /** Close WebSocket connection */
  disconnect: () => void;
  /** Send message to WebSocket (e.g., subscribe to symbols) */
  send: (message: string | object) => void;
}

/**
 * React hook for managing WebSocket connections with automatic reconnection
 *
 * @param url WebSocket URL (e.g., '/ws/prices')
 * @param options Configuration options
 * @returns WebSocket state and control methods
 *
 * @example
 * ```tsx
 * const { state, data, error, reconnect } = useWebSocket('/ws/prices', {
 *   autoReconnect: true,
 *   maxReconnectAttempts: 5,
 *   onOpen: () => console.log('Connected'),
 * });
 *
 * // Use data in your component
 * {data && <div>Price: ${data.price}</div>}
 *
 * // Handle reconnection
 * {state === WebSocketState.FAILED && (
 *   <button onClick={reconnect}>Retry Connection</button>
 * )}
 * ```
 */
export function useWebSocket(
  url: string,
  options: WebSocketOptions = {}
): UseWebSocketReturn {
  const {
    autoReconnect = true,
    maxReconnectAttempts = 5,
    reconnectDelay = 1000,
    maxReconnectDelay = 30000,
    authToken,
    onOpen,
    onClose,
    onError,
  } = options;

  // Get auth token from options or from api client
  const token = authToken ?? getAuthToken();

  const [state, setState] = useState<WebSocketState>(WebSocketState.DISCONNECTED);
  const [data, setData] = useState<PriceUpdate | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const shouldReconnectRef = useRef(true);

  /**
   * Calculate exponential backoff delay for reconnection
   */
  const getReconnectDelay = useCallback(
    (attempt: number): number => {
      const delay = reconnectDelay * Math.pow(2, attempt);
      return Math.min(delay, maxReconnectDelay);
    },
    [reconnectDelay, maxReconnectDelay]
  );

  // Track consecutive failures without successful connection (likely auth issues)
  const consecutiveFailuresRef = useRef(0);

  /**
   * Connect to WebSocket
   */
  const connect = useCallback(() => {
    // Don't connect if URL is empty or invalid
    if (!url || url.trim() === '') {
      setState(WebSocketState.DISCONNECTED);
      return;
    }

    // Don't connect if we're already connected or connecting
    if (
      wsRef.current &&
      (wsRef.current.readyState === WebSocket.CONNECTING ||
        wsRef.current.readyState === WebSocket.OPEN)
    ) {
      return;
    }

    setState(WebSocketState.CONNECTING);
    setError(null);

    try {
      // Build WebSocket URL (handle both relative and absolute URLs)
      let wsUrl = url;
      if (url.startsWith('/')) {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        wsUrl = `${protocol}//${window.location.host}${url}`;
      }

      // Add auth token as query parameter if available
      // Browser WebSocket API doesn't support custom headers, so token must be in URL
      if (token) {
        const separator = wsUrl.includes('?') ? '&' : '?';
        wsUrl = `${wsUrl}${separator}token=${encodeURIComponent(token)}`;
      }

      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;
      let connectionOpened = false;

      ws.onopen = () => {
        connectionOpened = true;
        consecutiveFailuresRef.current = 0; // Reset on successful connection
        setState(WebSocketState.CONNECTED);
        setReconnectAttempts(0);
        setError(null);
        onOpen?.();
      };

      ws.onmessage = (event) => {
        try {
          const parsedData = JSON.parse(event.data);
          setData(parsedData);
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      };

      ws.onerror = (event) => {
        const errorMessage = 'WebSocket connection error';
        setError(errorMessage);
        onError?.(event);
      };

      ws.onclose = (event) => {
        setState(WebSocketState.DISCONNECTED);
        onClose?.();

        // Track consecutive failures (connection closed without ever opening)
        // This typically indicates auth issues (403) or server rejection
        if (!connectionOpened) {
          consecutiveFailuresRef.current += 1;
        }

        // Stop reconnecting if we've had 3+ consecutive failures without connection
        // This likely means auth is misconfigured and retrying won't help
        const likelyAuthFailure = consecutiveFailuresRef.current >= 3;
        if (likelyAuthFailure) {
          setState(WebSocketState.FAILED);
          setError('WebSocket connection rejected - check authentication');
          return;
        }

        // Auto-reconnect if enabled
        if (
          autoReconnect &&
          shouldReconnectRef.current &&
          reconnectAttempts < maxReconnectAttempts
        ) {
          const delay = getReconnectDelay(reconnectAttempts);
          setState(WebSocketState.RECONNECTING);
          setReconnectAttempts((prev) => prev + 1);

          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, delay);
        } else if (reconnectAttempts >= maxReconnectAttempts) {
          setState(WebSocketState.FAILED);
          setError(`Failed to connect after ${maxReconnectAttempts} attempts`);
        }
      };
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(errorMessage);
      setState(WebSocketState.FAILED);
    }
  }, [
    url,
    autoReconnect,
    maxReconnectAttempts,
    reconnectAttempts,
    getReconnectDelay,
    onOpen,
    onClose,
    onError,
  ]);

  /**
   * Disconnect from WebSocket
   */
  const disconnect = useCallback(() => {
    shouldReconnectRef.current = false;

    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setState(WebSocketState.DISCONNECTED);
    setReconnectAttempts(0);
  }, []);

  /**
   * Manually reconnect to WebSocket
   */
  const reconnect = useCallback(() => {
    shouldReconnectRef.current = true;
    consecutiveFailuresRef.current = 0; // Reset failure counter on manual reconnect
    setReconnectAttempts(0);
    disconnect();
    setTimeout(() => connect(), 100);
  }, [connect, disconnect]);

  /**
   * Send message to WebSocket
   */
  const send = useCallback((message: string | object) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      const payload = typeof message === 'string' ? message : JSON.stringify(message);
      wsRef.current.send(payload);
    } else {
      console.warn('WebSocket is not connected. Cannot send message.');
    }
  }, []);

  // Connect on mount, disconnect on unmount
  useEffect(() => {
    connect();

    return () => {
      shouldReconnectRef.current = false;
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url]); // Only reconnect when URL changes

  return {
    state,
    data,
    error,
    reconnectAttempts,
    reconnect,
    disconnect,
    send,
  };
}
