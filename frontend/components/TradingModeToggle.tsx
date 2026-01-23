'use client';

import { useState, useEffect, useCallback } from 'react';
import { cn } from '../lib/utils';
import { Button } from './ui';

export type TradingMode = 'paper' | 'live';

const STORAGE_KEY = 'fluxhero_trading_mode';

export interface TradingModeToggleProps {
  className?: string;
  onModeChange?: (mode: TradingMode) => void;
}

/**
 * Toggle component for switching between paper and live trading modes.
 * Persists selection in localStorage and shows confirmation for live mode.
 */
export function TradingModeToggle({ className, onModeChange }: TradingModeToggleProps) {
  const [mode, setMode] = useState<TradingMode>('paper');
  const [showConfirmation, setShowConfirmation] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);

  // Load saved mode from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === 'paper' || saved === 'live') {
      setMode(saved);
    }
    setIsLoaded(true);
  }, []);

  // Save mode to localStorage and notify parent
  const updateMode = useCallback((newMode: TradingMode) => {
    setMode(newMode);
    localStorage.setItem(STORAGE_KEY, newMode);
    onModeChange?.(newMode);
  }, [onModeChange]);

  const handleModeClick = (newMode: TradingMode) => {
    if (newMode === mode) return;

    if (newMode === 'live') {
      // Show confirmation dialog when switching to live
      setShowConfirmation(true);
    } else {
      // Switch to paper directly
      updateMode(newMode);
    }
  };

  const handleConfirmLive = () => {
    updateMode('live');
    setShowConfirmation(false);
  };

  const handleCancelLive = () => {
    setShowConfirmation(false);
  };

  // Don't render until we've loaded the saved mode
  if (!isLoaded) {
    return (
      <div className={cn('inline-flex items-center gap-2', className)}>
        <div className="w-32 h-8 bg-panel-500 rounded animate-pulse" />
      </div>
    );
  }

  return (
    <div className={cn('relative inline-flex items-center gap-2', className)}>
      {/* Mode indicator */}
      <div className="flex items-center gap-1.5 mr-2">
        <span
          className={cn(
            'w-2.5 h-2.5 rounded-full animate-pulse',
            mode === 'paper' ? 'bg-profit-500' : 'bg-loss-500'
          )}
        />
        <span
          className={cn(
            'text-xs font-semibold uppercase tracking-wider',
            mode === 'paper' ? 'text-profit-500' : 'text-loss-500'
          )}
        >
          {mode === 'paper' ? 'Paper' : 'Live'}
        </span>
      </div>

      {/* Toggle buttons */}
      <div className="inline-flex rounded bg-panel-500 p-0.5">
        <button
          type="button"
          onClick={() => handleModeClick('paper')}
          className={cn(
            'px-3 py-1.5 text-sm font-medium rounded transition-colors',
            mode === 'paper'
              ? 'bg-profit-500 text-text-900'
              : 'text-text-400 hover:text-text-700'
          )}
        >
          Paper
        </button>
        <button
          type="button"
          onClick={() => handleModeClick('live')}
          className={cn(
            'px-3 py-1.5 text-sm font-medium rounded transition-colors',
            mode === 'live'
              ? 'bg-loss-500 text-text-900'
              : 'text-text-400 hover:text-text-700'
          )}
        >
          Live
        </button>
      </div>

      {/* Confirmation dialog overlay */}
      {showConfirmation && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-panel-900/80"
            onClick={handleCancelLive}
          />

          {/* Dialog */}
          <div className="relative z-10 w-full max-w-md bg-panel-700 rounded-lg p-6 shadow-xl border border-panel-500">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-full bg-loss-500/20 flex items-center justify-center">
                <svg
                  className="w-5 h-5 text-loss-500"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                  />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-text-900">
                Switch to Live Trading?
              </h3>
            </div>

            <p className="text-text-400 mb-6">
              You are about to switch to <span className="text-loss-500 font-semibold">live trading mode</span>.
              This will use real money and execute actual trades. Make sure you have configured
              your broker credentials correctly.
            </p>

            <div className="flex gap-3 justify-end">
              <Button variant="secondary" onClick={handleCancelLive}>
                Cancel
              </Button>
              <Button variant="danger" onClick={handleConfirmLive}>
                Switch to Live
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Hook to get and set the current trading mode
 */
export function useTradingMode() {
  const [mode, setMode] = useState<TradingMode>('paper');
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === 'paper' || saved === 'live') {
      setMode(saved);
    }
    setIsLoaded(true);

    // Listen for changes from other tabs/components
    const handleStorage = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY && (e.newValue === 'paper' || e.newValue === 'live')) {
        setMode(e.newValue);
      }
    };

    window.addEventListener('storage', handleStorage);
    return () => window.removeEventListener('storage', handleStorage);
  }, []);

  const updateMode = useCallback((newMode: TradingMode) => {
    setMode(newMode);
    localStorage.setItem(STORAGE_KEY, newMode);
    // Dispatch storage event for other components
    window.dispatchEvent(new StorageEvent('storage', {
      key: STORAGE_KEY,
      newValue: newMode,
    }));
  }, []);

  return { mode, setMode: updateMode, isLoaded };
}

export default TradingModeToggle;
