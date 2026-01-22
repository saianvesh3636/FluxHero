/**
 * WebSocket Context Provider for managing live price updates across the application
 */

'use client';

import React, { createContext, useContext, useCallback, useState } from 'react';
import { useWebSocket, WebSocketState, PriceUpdate } from '../hooks/useWebSocket';

interface PriceMap {
  [symbol: string]: PriceUpdate;
}

interface WebSocketContextValue {
  /** Current connection state */
  connectionState: WebSocketState;
  /** Map of symbol to latest price update */
  prices: PriceMap;
  /** Get price for specific symbol */
  getPrice: (symbol: string) => PriceUpdate | undefined;
  /** Subscribe to symbol updates */
  subscribe: (symbols: string[]) => void;
  /** Unsubscribe from symbol updates */
  unsubscribe: (symbols: string[]) => void;
  /** Manually reconnect to WebSocket */
  reconnect: () => void;
  /** Error message if connection failed */
  error: string | null;
  /** Number of reconnection attempts made */
  reconnectAttempts: number;
}

const WebSocketContext = createContext<WebSocketContextValue | undefined>(undefined);

interface WebSocketProviderProps {
  children: React.ReactNode;
  /** WebSocket URL (default: '/ws/prices') */
  url?: string;
  /** Enable WebSocket connection (default: true) */
  enabled?: boolean;
}

/**
 * WebSocket Provider Component
 * Manages WebSocket connection and distributes price updates to child components
 *
 * @example
 * ```tsx
 * // In app layout
 * <WebSocketProvider>
 *   <YourApp />
 * </WebSocketProvider>
 *
 * // In any child component
 * const { prices, connectionState, subscribe } = useWebSocketContext();
 *
 * useEffect(() => {
 *   subscribe(['SPY', 'AAPL']);
 * }, []);
 *
 * const spyPrice = prices['SPY'];
 * ```
 */
export function WebSocketProvider({
  children,
  url = '/ws/prices',
  enabled = true,
}: WebSocketProviderProps) {
  const [prices, setPrices] = useState<PriceMap>({});
  const [subscribedSymbols, setSubscribedSymbols] = useState<Set<string>>(new Set());

  const handleMessage = useCallback((data: PriceUpdate) => {
    if (data && data.symbol) {
      setPrices((prev) => ({
        ...prev,
        [data.symbol]: data,
      }));
    }
  }, []);

  const {
    state: connectionState,
    data,
    error,
    reconnectAttempts,
    reconnect,
    send,
  } = useWebSocket(enabled ? url : '', {
    autoReconnect: true,
    maxReconnectAttempts: 5,
    reconnectDelay: 1000,
    maxReconnectDelay: 30000,
    onOpen: () => {
      // Re-subscribe to symbols on reconnect
      if (subscribedSymbols.size > 0) {
        send({
          action: 'subscribe',
          symbols: Array.from(subscribedSymbols),
        });
      }
    },
  });

  // Update data when WebSocket receives messages
  React.useEffect(() => {
    if (data) {
      handleMessage(data);
    }
  }, [data, handleMessage]);

  const getPrice = useCallback(
    (symbol: string): PriceUpdate | undefined => {
      return prices[symbol];
    },
    [prices]
  );

  const subscribe = useCallback(
    (symbols: string[]) => {
      const newSymbols = symbols.filter((s) => !subscribedSymbols.has(s));

      if (newSymbols.length > 0) {
        setSubscribedSymbols((prev) => {
          const updated = new Set(prev);
          newSymbols.forEach((s) => updated.add(s));
          return updated;
        });

        // Send subscribe message if connected
        if (connectionState === WebSocketState.CONNECTED) {
          send({
            action: 'subscribe',
            symbols: newSymbols,
          });
        }
      }
    },
    [subscribedSymbols, connectionState, send]
  );

  const unsubscribe = useCallback(
    (symbols: string[]) => {
      setSubscribedSymbols((prev) => {
        const updated = new Set(prev);
        symbols.forEach((s) => updated.delete(s));
        return updated;
      });

      // Send unsubscribe message if connected
      if (connectionState === WebSocketState.CONNECTED) {
        send({
          action: 'unsubscribe',
          symbols,
        });
      }
    },
    [connectionState, send]
  );

  const value: WebSocketContextValue = {
    connectionState,
    prices,
    getPrice,
    subscribe,
    unsubscribe,
    reconnect,
    error,
    reconnectAttempts,
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
}

/**
 * Hook to access WebSocket context
 * Must be used within a WebSocketProvider
 *
 * @example
 * ```tsx
 * const { prices, connectionState, subscribe } = useWebSocketContext();
 *
 * useEffect(() => {
 *   subscribe(['SPY', 'AAPL']);
 * }, [subscribe]);
 *
 * const spyPrice = prices['SPY']?.price;
 * ```
 */
export function useWebSocketContext(): WebSocketContextValue {
  const context = useContext(WebSocketContext);

  if (!context) {
    throw new Error('useWebSocketContext must be used within a WebSocketProvider');
  }

  return context;
}
