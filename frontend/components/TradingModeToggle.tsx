'use client';

import { useState, useCallback } from 'react';
import { cn } from '../lib/utils';
import { Button } from './ui';

export type TradingMode = 'paper' | 'live';

export interface TradingModeToggleProps {
  className?: string;
  mode: TradingMode;
  isLiveBrokerConfigured?: boolean;
  isLoading?: boolean;
  onModeChange?: (mode: TradingMode, confirmLive?: boolean) => Promise<void>;
}

/**
 * Toggle component for switching between paper and live trading modes.
 * Connects to backend API and shows confirmation dialog with acknowledgment for live mode.
 */
export function TradingModeToggle({
  className,
  mode,
  isLiveBrokerConfigured = false,
  isLoading = false,
  onModeChange,
}: TradingModeToggleProps) {
  const [showConfirmation, setShowConfirmation] = useState(false);
  const [acknowledged, setAcknowledged] = useState(false);
  const [switchError, setSwitchError] = useState<string | null>(null);
  const [isSwitching, setIsSwitching] = useState(false);

  const handleModeClick = (newMode: TradingMode) => {
    if (newMode === mode || isSwitching) return;

    if (newMode === 'live') {
      // Show confirmation dialog when switching to live
      setShowConfirmation(true);
      setAcknowledged(false);
      setSwitchError(null);
    } else {
      // Switch to paper directly
      handleSwitch(newMode);
    }
  };

  const handleSwitch = async (newMode: TradingMode, confirmLive = false) => {
    if (!onModeChange) return;

    setIsSwitching(true);
    setSwitchError(null);

    try {
      await onModeChange(newMode, confirmLive);
      setShowConfirmation(false);
      setAcknowledged(false);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to switch mode';
      setSwitchError(message);
    } finally {
      setIsSwitching(false);
    }
  };

  const handleConfirmLive = () => {
    if (!acknowledged) return;
    handleSwitch('live', true);
  };

  const handleCancelLive = () => {
    setShowConfirmation(false);
    setAcknowledged(false);
    setSwitchError(null);
  };

  if (isLoading) {
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
            'w-2.5 h-2.5 rounded-full',
            mode === 'paper' ? 'bg-profit-500 animate-pulse' : 'bg-loss-500 animate-pulse'
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
          disabled={isSwitching}
          className={cn(
            'px-3 py-1.5 text-sm font-medium rounded transition-colors',
            mode === 'paper'
              ? 'bg-profit-500 text-text-900'
              : 'text-text-400 hover:text-text-700',
            isSwitching && 'opacity-50 cursor-not-allowed'
          )}
        >
          Paper
        </button>
        <div className="relative group">
          <button
            type="button"
            onClick={() => handleModeClick('live')}
            disabled={isSwitching}
            className={cn(
              'px-3 py-1.5 text-sm font-medium rounded transition-colors',
              mode === 'live'
                ? 'bg-loss-500 text-text-900'
                : 'text-text-400 hover:text-text-700',
              isSwitching && 'opacity-50 cursor-not-allowed'
            )}
          >
            Live
          </button>
          {/* Tooltip when broker not configured */}
          {!isLiveBrokerConfigured && mode !== 'live' && (
            <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-2 bg-panel-800 text-text-400 text-xs rounded shadow-lg whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50 border border-panel-500">
              <div className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-loss-500" />
                <span>Broker not configured</span>
              </div>
              <div className="absolute top-full left-1/2 transform -translate-x-1/2 border-4 border-transparent border-t-panel-800" />
            </div>
          )}
        </div>
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

            <p className="text-text-400 mb-4">
              You are about to switch to <span className="text-loss-500 font-semibold">live trading mode</span>.
              This will use real money and execute actual trades.
            </p>

            {/* Broker status */}
            <div className={cn(
              'p-3 rounded-lg mb-4',
              isLiveBrokerConfigured ? 'bg-profit-500/10' : 'bg-loss-500/10'
            )}>
              <div className="flex items-center gap-2">
                <span className={cn(
                  'w-2 h-2 rounded-full',
                  isLiveBrokerConfigured ? 'bg-profit-500' : 'bg-loss-500'
                )} />
                <span className={cn(
                  'text-sm font-medium',
                  isLiveBrokerConfigured ? 'text-profit-500' : 'text-loss-500'
                )}>
                  {isLiveBrokerConfigured
                    ? 'Broker configured and ready'
                    : 'No broker configured - add broker credentials first'}
                </span>
              </div>
            </div>

            {/* Acknowledgment checkbox */}
            <label className="flex items-start gap-3 mb-6 cursor-pointer">
              <input
                type="checkbox"
                checked={acknowledged}
                onChange={(e) => setAcknowledged(e.target.checked)}
                className="mt-1 w-4 h-4 rounded border-panel-400 bg-panel-600 text-loss-500 focus:ring-loss-500"
              />
              <span className="text-sm text-text-400">
                I understand that live trading involves real money and I accept responsibility
                for any trades executed in this mode.
              </span>
            </label>

            {/* Error message */}
            {switchError && (
              <div className="mb-4 p-3 bg-loss-500/10 rounded-lg">
                <p className="text-sm text-loss-500">{switchError}</p>
              </div>
            )}

            <div className="flex gap-3 justify-end">
              <Button variant="secondary" onClick={handleCancelLive} disabled={isSwitching}>
                Cancel
              </Button>
              <Button
                variant="danger"
                onClick={handleConfirmLive}
                disabled={!acknowledged || !isLiveBrokerConfigured || isSwitching}
              >
                {isSwitching ? 'Switching...' : 'Switch to Live'}
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default TradingModeToggle;
