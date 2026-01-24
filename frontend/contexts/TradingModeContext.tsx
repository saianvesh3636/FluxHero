'use client';

import { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import { useSearchParams, useRouter, usePathname } from 'next/navigation';
import { apiClient } from '../utils/api';

export type TradingMode = 'live' | 'paper';

export interface ModeState {
  active_mode: TradingMode;
  last_mode_change: string | null;
  paper_balance: number;
  paper_realized_pnl: number;
  is_live_broker_configured: boolean;
}

interface TradingModeContextValue {
  mode: TradingMode;
  modeState: ModeState | null;
  isLive: boolean;
  isPaper: boolean;
  isLoading: boolean;
  error: string | null;
  switchMode: (newMode: TradingMode, confirmLive?: boolean) => Promise<void>;
  refreshModeState: () => Promise<void>;
}

const TradingModeContext = createContext<TradingModeContextValue | null>(null);

const STORAGE_KEY = 'fluxhero_trading_mode';

export function TradingModeProvider({ children }: { children: ReactNode }) {
  const [modeState, setModeState] = useState<ModeState | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const mode = modeState?.active_mode || 'paper';

  const refreshModeState = useCallback(async () => {
    try {
      setError(null);
      const state = await apiClient.getModeState();
      setModeState(state);
      // Sync with localStorage for cross-tab consistency
      localStorage.setItem(STORAGE_KEY, state.active_mode);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch mode state';
      setError(message);
      console.error('Failed to fetch mode state:', err);
      // Fall back to localStorage
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved === 'paper' || saved === 'live') {
        setModeState({
          active_mode: saved,
          last_mode_change: null,
          paper_balance: 100000,
          paper_realized_pnl: 0,
          is_live_broker_configured: false,
        });
      }
    } finally {
      setIsLoading(false);
    }
  }, []);

  const switchMode = useCallback(async (newMode: TradingMode, confirmLive = false) => {
    // Clear any previous error
    setError(null);

    try {
      const result = await apiClient.switchMode(newMode, confirmLive);

      // Only update state after successful API call
      setModeState(result);
      localStorage.setItem(STORAGE_KEY, result.active_mode);

      // Update URL with the actual mode returned by backend (source of truth)
      const params = new URLSearchParams(searchParams.toString());
      params.set('mode', result.active_mode);
      router.push(`${pathname}?${params.toString()}`);

      // Dispatch storage event for other tabs
      window.dispatchEvent(new StorageEvent('storage', {
        key: STORAGE_KEY,
        newValue: result.active_mode,
      }));
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to switch mode';
      setError(message);

      // Refresh mode state from backend to ensure we're in sync
      // This handles cases where the switch partially succeeded or failed
      try {
        const currentState = await apiClient.getModeState();
        setModeState(currentState);
        localStorage.setItem(STORAGE_KEY, currentState.active_mode);
      } catch {
        // Ignore refresh error - we already have the switch error
      }

      throw err;
    }
  }, [pathname, router, searchParams]);

  // Initial load
  useEffect(() => {
    refreshModeState();
  }, [refreshModeState]);

  // Sync URL mode param with backend on navigation
  useEffect(() => {
    const urlMode = searchParams.get('mode') as TradingMode | null;
    if (urlMode && modeState && urlMode !== modeState.active_mode) {
      // URL and backend out of sync - backend is source of truth
      const params = new URLSearchParams(searchParams.toString());
      params.set('mode', modeState.active_mode);
      router.replace(`${pathname}?${params.toString()}`);
    }
  }, [modeState, searchParams, pathname, router]);

  // Listen for cross-tab changes
  useEffect(() => {
    const handleStorage = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY && (e.newValue === 'paper' || e.newValue === 'live')) {
        // Another tab changed the mode, refresh our state
        refreshModeState();
      }
    };

    window.addEventListener('storage', handleStorage);
    return () => window.removeEventListener('storage', handleStorage);
  }, [refreshModeState]);

  return (
    <TradingModeContext.Provider value={{
      mode,
      modeState,
      isLive: mode === 'live',
      isPaper: mode === 'paper',
      isLoading,
      error,
      switchMode,
      refreshModeState,
    }}>
      {children}
    </TradingModeContext.Provider>
  );
}

export function useTradingMode() {
  const context = useContext(TradingModeContext);
  if (!context) {
    throw new Error('useTradingMode must be used within TradingModeProvider');
  }
  return context;
}

export default TradingModeContext;
