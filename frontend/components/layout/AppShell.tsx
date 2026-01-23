'use client';

import React from 'react';
import { Navigation } from './Navigation';
import { useWebSocketContext } from '../../contexts/WebSocketContext';

interface AppShellProps {
  children: React.ReactNode;
}

/**
 * AppShell - wraps the app with navigation and connects to WebSocket status
 */
export function AppShell({ children }: AppShellProps) {
  const { connectionState } = useWebSocketContext();

  const statusMap: Record<string, 'connected' | 'disconnected' | 'connecting'> = {
    connected: 'connected',
    connecting: 'connecting',
    disconnected: 'disconnected',
    error: 'disconnected',
  };

  return (
    <div className="min-h-screen bg-panel-900">
      <Navigation connectionStatus={statusMap[connectionState] || 'disconnected'} />
      <main>{children}</main>
    </div>
  );
}

export default AppShell;
