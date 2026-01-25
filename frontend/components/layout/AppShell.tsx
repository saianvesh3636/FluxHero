'use client';

import React from 'react';
import { Navigation } from './Navigation';
import { Sidebar, SidebarProvider, useSidebar } from './Sidebar';
import { useWebSocketContext } from '../../contexts/WebSocketContext';

interface AppShellProps {
  children: React.ReactNode;
}

/**
 * MainContent - wrapper that adjusts for sidebar width
 */
function MainContent({ children }: { children: React.ReactNode }) {
  const { isCollapsed } = useSidebar();

  return (
    <main
      className="transition-all duration-300 ease-in-out lg:ml-16"
      style={{
        marginLeft: isCollapsed ? undefined : undefined,
      }}
    >
      <div
        className="transition-all duration-300 ease-in-out"
        style={{
          marginLeft: isCollapsed ? '0' : 'calc(14rem - 4rem)', // 224px - 64px = 160px additional margin when expanded
        }}
      >
        {children}
      </div>
    </main>
  );
}

/**
 * AppShell - wraps the app with navigation, sidebar, and connects to WebSocket status
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
    <SidebarProvider>
      <div className="min-h-screen bg-panel-900">
        <Navigation connectionStatus={statusMap[connectionState] || 'disconnected'} />
        <Sidebar />
        <MainContent>{children}</MainContent>
      </div>
    </SidebarProvider>
  );
}

export default AppShell;
