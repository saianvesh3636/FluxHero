'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '../../lib/utils';
import { StatusDot } from '../ui/StatusDot';

export interface NavItem {
  label: string;
  href: string;
  icon?: React.ReactNode;
}

export interface NavigationProps {
  items?: NavItem[];
  connectionStatus?: 'connected' | 'disconnected' | 'connecting';
}

const defaultNavItems: NavItem[] = [
  { label: 'Home', href: '/' },
  { label: 'Live', href: '/live' },
  { label: 'Analytics', href: '/analytics' },
  { label: 'Backtest', href: '/backtest' },
  { label: 'History', href: '/history' },
  { label: 'Signals', href: '/signals' },
];

/**
 * Navigation component - horizontal nav bar
 * Follows design system: panel-800 background, 56px height
 */
export function Navigation({
  items = defaultNavItems,
  connectionStatus = 'disconnected',
}: NavigationProps) {
  const pathname = usePathname();

  return (
    <nav className="bg-panel-800 h-14 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 h-full flex items-center justify-between">
        {/* Logo/Brand */}
        <Link href="/" className="flex items-center gap-2">
          <div className="w-8 h-8 bg-gradient-to-r from-accent-500 to-accent-900 rounded-lg flex items-center justify-center">
            <span className="text-text-900 font-bold text-sm">FH</span>
          </div>
          <span className="text-text-900 font-semibold text-lg hidden sm:block">
            FluxHero
          </span>
        </Link>

        {/* Nav Items */}
        <div className="flex items-center gap-1 overflow-x-auto scrollbar-hide">
          {items.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  'px-3 py-2 text-sm font-medium rounded whitespace-nowrap',
                  isActive
                    ? 'bg-panel-600 text-text-900 border-l-2 border-accent-500'
                    : 'text-text-400 hover:text-text-700 hover:bg-panel-700'
                )}
              >
                {item.icon && <span className="mr-2">{item.icon}</span>}
                {item.label}
              </Link>
            );
          })}
        </div>

        {/* Status Indicator */}
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
    </nav>
  );
}

export default Navigation;
