'use client';

import React from 'react';
import Link from 'next/link';
import { StatusDot } from '../ui/StatusDot';
import { TradingModeToggle } from '../TradingModeToggle';
import { useTradingMode } from '../../contexts/TradingModeContext';
import { MobileMenuButton } from './Sidebar';

export interface NavItem {
  label: string;
  href: string;
  icon?: React.ReactNode;
}

export interface NavigationProps {
  connectionStatus?: 'connected' | 'disconnected' | 'connecting';
}

/**
 * Navigation component - horizontal nav bar (top bar only)
 * Follows design system: panel-800 background, 56px height
 * Nav items moved to Sidebar - this now contains logo, mode toggle, and status
 */
export function Navigation({
  connectionStatus = 'disconnected',
}: NavigationProps) {
  const { mode, modeState, switchMode, isLoading: modeLoading } = useTradingMode();

  const handleModeChange = async (newMode: 'live' | 'paper', confirmLive = false) => {
    await switchMode(newMode, confirmLive);
  };

  return (
    <nav className="bg-panel-800 h-14 sticky top-0 z-50">
      <div className="px-4 h-full flex items-center justify-between">
        {/* Left: Mobile menu + Logo */}
        <div className="flex items-center gap-2">
          <MobileMenuButton />
          <Link href="/" className="flex items-center gap-2">
            <div className="w-8 h-8 bg-gradient-to-r from-accent-500 to-accent-900 rounded-lg flex items-center justify-center">
              <span className="text-text-900 font-bold text-sm">FH</span>
            </div>
            <span className="text-text-900 font-semibold text-lg hidden sm:block">
              FluxHero
            </span>
          </Link>
        </div>

        {/* Right: Trading Mode Toggle & Status Indicator */}
        <div className="flex items-center gap-4">
          <TradingModeToggle
            mode={mode}
            isLiveBrokerConfigured={modeState?.is_live_broker_configured || false}
            isLoading={modeLoading}
            onModeChange={handleModeChange}
          />
          <div className="flex items-center gap-2">
            <StatusDot
              status={connectionStatus}
              showLabel
              className="hidden sm:flex"
            />
            <StatusDot
              status={connectionStatus}
              className="sm:hidden"
            />
          </div>
        </div>
      </div>
    </nav>
  );
}

export default Navigation;
