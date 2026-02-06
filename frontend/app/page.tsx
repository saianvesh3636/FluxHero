/**
 * Home Page - FluxHero Trading System Dashboard
 * 
 * Features:
 * - Backend status checking
 * - Quick navigation to main features
 * - System health indicators
 */

'use client';

import React, { useEffect, useState } from 'react';
import { apiClient } from '../utils/api';

interface SystemStatus {
  status: 'active' | 'delayed' | 'offline';
  last_update: string;
  uptime_seconds: number;
}

export default function Home() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const checkStatus = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await apiClient.getSystemStatus();
      setStatus(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to connect');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    checkStatus();
  }, []);

  const getStatusBadgeClass = (statusValue: string) => {
    switch (statusValue) {
      case 'active':
        return 'bg-profit-500 text-text-900';
      case 'delayed':
        return 'bg-warning-500 text-text-900';
      case 'offline':
        return 'bg-loss-500 text-text-900';
      default:
        return 'bg-panel-400 text-text-400';
    }
  };

  return (
    <div className="mx-auto w-full max-w-7xl px-4 py-6 sm:px-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-semibold text-text-900">
          FluxHero Trading System
        </h1>
        <p className="text-lg text-text-400 mt-1">
          Adaptive retail quantitative trading platform
        </p>
      </div>

      {/* Backend Status Card */}
      <div className="rounded-2xl bg-panel-600 p-5 mb-6">
        <h3 className="text-xl font-semibold text-text-900 mb-4">
          System Status
        </h3>

        {isLoading ? (
          <div className="text-text-400">
            Checking backend status...
          </div>
        ) : error ? (
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-loss-500" />
              <span className="text-loss-500 font-medium">Backend Offline</span>
            </div>
            <p className="text-text-400 text-sm">
              Unable to connect to the backend server. Please ensure the API is running.
            </p>
            <button
              onClick={checkStatus}
              className="inline-flex items-center justify-center gap-2 font-medium rounded px-4 py-2 bg-accent-500 text-text-900 hover:bg-accent-600"
            >
              Retry Connection
            </button>
          </div>
        ) : status ? (
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <div className={`w-3 h-3 rounded-full ${status.status === 'active' ? 'bg-profit-500' : status.status === 'delayed' ? 'bg-warning-500' : 'bg-loss-500'}`} />
              <span className="text-text-900 font-medium">Connected</span>
              <span className={`px-2 py-0.5 rounded text-xs font-medium ${getStatusBadgeClass(status.status)}`}>
                {status.status.toUpperCase()}
              </span>
            </div>
            <p className="text-text-400 text-sm">
              Backend API is connected and ready.
            </p>
          </div>
        ) : null}
      </div>

      {/* Quick Navigation */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <a href="/backtest" className="rounded-2xl bg-panel-600 p-5 hover:bg-panel-500 transition-colors">
          <h3 className="text-lg font-semibold text-text-900 mb-2">Backtesting</h3>
          <p className="text-text-400 text-sm">Test strategies against historical data</p>
        </a>
        <a href="/live" className="rounded-2xl bg-panel-600 p-5 hover:bg-panel-500 transition-colors">
          <h3 className="text-lg font-semibold text-text-900 mb-2">Live Trading</h3>
          <p className="text-text-400 text-sm">Monitor and execute live trades</p>
        </a>
        <a href="/walk-forward" className="rounded-2xl bg-panel-600 p-5 hover:bg-panel-500 transition-colors">
          <h3 className="text-lg font-semibold text-text-900 mb-2">Walk-Forward</h3>
          <p className="text-text-400 text-sm">Validate strategy robustness</p>
        </a>
        <a href="/signals" className="rounded-2xl bg-panel-600 p-5 hover:bg-panel-500 transition-colors">
          <h3 className="text-lg font-semibold text-text-900 mb-2">Signals</h3>
          <p className="text-text-400 text-sm">View trading signals and alerts</p>
        </a>
        <a href="/chart" className="rounded-2xl bg-panel-600 p-5 hover:bg-panel-500 transition-colors">
          <h3 className="text-lg font-semibold text-text-900 mb-2">Charts</h3>
          <p className="text-text-400 text-sm">Technical analysis and charting</p>
        </a>
        <a href="/settings" className="rounded-2xl bg-panel-600 p-5 hover:bg-panel-500 transition-colors">
          <h3 className="text-lg font-semibold text-text-900 mb-2">Settings</h3>
          <p className="text-text-400 text-sm">Configure system preferences</p>
        </a>
      </div>
    </div>
  );
}
