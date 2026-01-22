'use client';

import { useEffect, useState } from 'react';
import { apiClient, SystemStatus } from '../utils/api';

export default function Home() {
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [isBackendOffline, setIsBackendOffline] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const checkBackend = async () => {
      try {
        const status = await apiClient.getSystemStatus();
        setSystemStatus(status);
        setIsBackendOffline(false);
      } catch (err) {
        setIsBackendOffline(true);
      } finally {
        setLoading(false);
      }
    };

    checkBackend();
  }, []);

  const handleRetryConnection = async () => {
    setLoading(true);
    try {
      const status = await apiClient.getSystemStatus();
      setSystemStatus(status);
      setIsBackendOffline(false);
    } catch (err) {
      setIsBackendOffline(true);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main style={{ padding: '2rem', maxWidth: '1200px', margin: '0 auto' }}>
      <h1 style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>
        FluxHero Trading System
      </h1>
      <p style={{ fontSize: '1.2rem', marginBottom: '2rem', color: '#666' }}>
        Adaptive retail quantitative trading platform
      </p>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
          gap: '1.5rem',
          marginTop: '2rem',
        }}
      >
        <div
          style={{
            padding: '1.5rem',
            background: 'white',
            borderRadius: '8px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
          }}
        >
          <h2 style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>
            Live Trading
          </h2>
          <p style={{ color: '#666' }}>
            Monitor open positions and real-time P&L
          </p>
        </div>

        <div
          style={{
            padding: '1.5rem',
            background: 'white',
            borderRadius: '8px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
          }}
        >
          <h2 style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>
            Analytics
          </h2>
          <p style={{ color: '#666' }}>
            Charts, indicators, and performance metrics
          </p>
        </div>

        <div
          style={{
            padding: '1.5rem',
            background: 'white',
            borderRadius: '8px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
          }}
        >
          <h2 style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>
            Trade History
          </h2>
          <p style={{ color: '#666' }}>View past trades and export data</p>
        </div>

        <div
          style={{
            padding: '1.5rem',
            background: 'white',
            borderRadius: '8px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
          }}
        >
          <h2 style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>
            Backtesting
          </h2>
          <p style={{ color: '#666' }}>Test strategies on historical data</p>
        </div>
      </div>

      <div
        style={{
          marginTop: '3rem',
          padding: '1.5rem',
          background: 'white',
          borderRadius: '8px',
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
        }}
      >
        <h3 style={{ fontSize: '1.3rem', marginBottom: '1rem' }}>
          System Status
        </h3>
        {loading ? (
          <p style={{ color: '#666' }}>Checking backend status...</p>
        ) : isBackendOffline ? (
          <div>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                marginBottom: '1rem',
              }}
            >
              <span style={{ fontSize: '1.5rem' }}>🔴</span>
              <span style={{ color: '#dc2626', fontWeight: '600' }}>
                Backend Offline
              </span>
            </div>
            <p style={{ color: '#666', marginBottom: '1rem' }}>
              Unable to connect to the backend server. Please ensure the backend is running on port 8000.
            </p>
            <button
              onClick={handleRetryConnection}
              style={{
                padding: '0.5rem 1rem',
                background: '#dc2626',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                cursor: 'pointer',
                fontWeight: '500',
              }}
            >
              Retry Connection
            </button>
          </div>
        ) : (
          <div>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                marginBottom: '0.5rem',
              }}
            >
              <span style={{ fontSize: '1.5rem' }}>
                {systemStatus?.status === 'active' ? '🟢' :
                 systemStatus?.status === 'delayed' ? '🟡' : '🔴'}
              </span>
              <span style={{ color: '#059669', fontWeight: '600', textTransform: 'capitalize' }}>
                {systemStatus?.status || 'Unknown'}
              </span>
            </div>
            <p style={{ color: '#666' }}>
              Backend API is connected and ready.
            </p>
          </div>
        )}
      </div>
    </main>
  );
}
