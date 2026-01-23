'use client';

import { useEffect, useState, useRef, useCallback } from 'react';
import {
  apiClient,
  ApiError,
  BrokerConfigResponse,
  BrokerConfigRequest,
  BrokerHealthResponse,
} from '../../utils/api';
import { PageContainer, PageHeader } from '../../components/layout';
import {
  Card,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
  Button,
  Input,
  Select,
  Badge,
  StatusDot,
  Skeleton,
} from '../../components/ui';
import { cn, formatDateTime } from '../../lib/utils';

// Available broker types
const BROKER_TYPES = [
  { value: 'alpaca', label: 'Alpaca' },
];

// Default Alpaca URLs
const ALPACA_URLS = {
  paper: 'https://paper-api.alpaca.markets',
  live: 'https://api.alpaca.markets',
};

interface BrokerFormData {
  broker_type: string;
  name: string;
  api_key: string;
  api_secret: string;
  base_url: string;
}

const initialFormData: BrokerFormData = {
  broker_type: 'alpaca',
  name: '',
  api_key: '',
  api_secret: '',
  base_url: ALPACA_URLS.paper,
};

export default function SettingsPage() {
  // Broker list state
  const [brokers, setBrokers] = useState<BrokerConfigResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Form state
  const [showForm, setShowForm] = useState(false);
  const [formData, setFormData] = useState<BrokerFormData>(initialFormData);
  const [formError, setFormError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Health check state
  const [healthStatus, setHealthStatus] = useState<Record<string, BrokerHealthResponse>>({});
  const [checkingHealth, setCheckingHealth] = useState<Record<string, boolean>>({});

  // Delete confirmation state
  const [deletingId, setDeletingId] = useState<string | null>(null);

  // Track if a fetch is already in progress
  const isFetchingRef = useRef(false);

  const fetchBrokers = useCallback(async () => {
    if (isFetchingRef.current) return;
    isFetchingRef.current = true;

    try {
      setError(null);
      const response = await apiClient.getBrokers();
      setBrokers(response.brokers);
      setLoading(false);
    } catch (err) {
      const message = err instanceof ApiError ? err.detail : 'Failed to fetch brokers';
      setError(message);
      setLoading(false);
    } finally {
      isFetchingRef.current = false;
    }
  }, []);

  useEffect(() => {
    fetchBrokers();
  }, [fetchBrokers]);

  const handleFormChange = (field: keyof BrokerFormData, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
    setFormError(null);
  };

  const handleEnvironmentChange = (env: 'paper' | 'live') => {
    setFormData((prev) => ({
      ...prev,
      base_url: ALPACA_URLS[env],
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setFormError(null);

    // Validation
    if (!formData.name.trim()) {
      setFormError('Name is required');
      return;
    }
    if (!formData.api_key.trim()) {
      setFormError('API Key is required');
      return;
    }
    if (!formData.api_secret.trim()) {
      setFormError('API Secret is required');
      return;
    }

    setIsSubmitting(true);

    try {
      const request: BrokerConfigRequest = {
        broker_type: formData.broker_type,
        name: formData.name.trim(),
        api_key: formData.api_key.trim(),
        api_secret: formData.api_secret.trim(),
        base_url: formData.base_url || undefined,
      };

      await apiClient.addBroker(request);

      // Reset form and refresh list
      setFormData(initialFormData);
      setShowForm(false);
      await fetchBrokers();
    } catch (err) {
      const message = err instanceof ApiError ? err.detail : 'Failed to add broker';
      setFormError(message);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCancel = () => {
    setFormData(initialFormData);
    setFormError(null);
    setShowForm(false);
  };

  const handleTestConnection = async (brokerId: string) => {
    setCheckingHealth((prev) => ({ ...prev, [brokerId]: true }));

    try {
      const health = await apiClient.getBrokerHealth(brokerId);
      setHealthStatus((prev) => ({ ...prev, [brokerId]: health }));
    } catch (err) {
      const message = err instanceof ApiError ? err.detail : 'Connection test failed';
      setHealthStatus((prev) => ({
        ...prev,
        [brokerId]: {
          id: brokerId,
          name: '',
          broker_type: '',
          is_connected: false,
          is_authenticated: false,
          latency_ms: null,
          last_heartbeat: null,
          error_message: message,
        },
      }));
    } finally {
      setCheckingHealth((prev) => ({ ...prev, [brokerId]: false }));
    }
  };

  const handleDelete = async (brokerId: string) => {
    try {
      await apiClient.deleteBroker(brokerId);
      setDeletingId(null);
      // Remove from health status
      setHealthStatus((prev) => {
        const next = { ...prev };
        delete next[brokerId];
        return next;
      });
      await fetchBrokers();
    } catch (err) {
      const message = err instanceof ApiError ? err.detail : 'Failed to delete broker';
      setError(message);
      setDeletingId(null);
    }
  };

  const getConnectionStatus = (
    brokerId: string
  ): 'connected' | 'disconnected' | 'connecting' => {
    if (checkingHealth[brokerId]) return 'connecting';
    const health = healthStatus[brokerId];
    if (!health) return 'disconnected';
    return health.is_connected && health.is_authenticated ? 'connected' : 'disconnected';
  };

  if (loading) {
    return (
      <PageContainer>
        <PageHeader title="Settings" subtitle="Configure broker connections" />
        <div className="space-y-4">
          <Skeleton className="h-48 w-full rounded-2xl" />
          <Skeleton className="h-48 w-full rounded-2xl" />
        </div>
      </PageContainer>
    );
  }

  return (
    <PageContainer>
      <PageHeader
        title="Settings"
        subtitle="Configure broker connections"
        actions={
          !showForm && (
            <Button onClick={() => setShowForm(true)}>Add Broker</Button>
          )
        }
      />

      {error && (
        <Card variant="elevated" className="mb-6 border-l-4 border-loss-500">
          <p className="text-loss-500">{error}</p>
          <Button variant="ghost" size="sm" onClick={() => setError(null)} className="mt-2">
            Dismiss
          </Button>
        </Card>
      )}

      {/* Add Broker Form */}
      {showForm && (
        <Card className="mb-6">
          <CardTitle>Add Broker</CardTitle>
          <CardDescription>Configure a new broker connection</CardDescription>

          <form onSubmit={handleSubmit} className="mt-6 space-y-5">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              <Select
                label="Broker Type"
                options={BROKER_TYPES}
                value={formData.broker_type}
                onChange={(e) => handleFormChange('broker_type', e.target.value)}
              />

              <Input
                label="Display Name"
                type="text"
                placeholder="My Alpaca Account"
                value={formData.name}
                onChange={(e) => handleFormChange('name', e.target.value)}
                maxLength={100}
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              <Input
                label="API Key"
                type="password"
                placeholder="Enter your API key"
                value={formData.api_key}
                onChange={(e) => handleFormChange('api_key', e.target.value)}
                autoComplete="off"
              />

              <Input
                label="API Secret"
                type="password"
                placeholder="Enter your API secret"
                value={formData.api_secret}
                onChange={(e) => handleFormChange('api_secret', e.target.value)}
                autoComplete="off"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-text-600 mb-2">
                Environment
              </label>
              <div className="flex gap-3">
                <button
                  type="button"
                  onClick={() => handleEnvironmentChange('paper')}
                  className={cn(
                    'px-4 py-2 rounded text-sm font-medium',
                    formData.base_url === ALPACA_URLS.paper
                      ? 'bg-profit-500 text-text-900'
                      : 'bg-panel-400 text-text-600 hover:bg-panel-300'
                  )}
                >
                  Paper Trading
                </button>
                <button
                  type="button"
                  onClick={() => handleEnvironmentChange('live')}
                  className={cn(
                    'px-4 py-2 rounded text-sm font-medium',
                    formData.base_url === ALPACA_URLS.live
                      ? 'bg-loss-500 text-text-900'
                      : 'bg-panel-400 text-text-600 hover:bg-panel-300'
                  )}
                >
                  Live Trading
                </button>
              </div>
              <p className="mt-2 text-sm text-text-300">
                {formData.base_url === ALPACA_URLS.live
                  ? 'Warning: Live trading uses real money'
                  : 'Paper trading uses simulated funds'}
              </p>
            </div>

            {formError && (
              <p className="text-loss-500 text-sm">{formError}</p>
            )}

            <div className="flex gap-3">
              <Button type="submit" isLoading={isSubmitting}>
                Add Broker
              </Button>
              <Button type="button" variant="secondary" onClick={handleCancel}>
                Cancel
              </Button>
            </div>
          </form>
        </Card>
      )}

      {/* Broker List */}
      <div className="space-y-4">
        {brokers.length === 0 ? (
          <Card className="text-center py-12">
            <p className="text-text-400 mb-4">No brokers configured</p>
            {!showForm && (
              <Button onClick={() => setShowForm(true)}>Add Your First Broker</Button>
            )}
          </Card>
        ) : (
          brokers.map((broker) => {
            const health = healthStatus[broker.id];
            const isChecking = checkingHealth[broker.id];
            const status = getConnectionStatus(broker.id);

            return (
              <Card key={broker.id}>
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <CardTitle>{broker.name}</CardTitle>
                      <Badge variant={broker.is_connected ? 'success' : 'neutral'}>
                        {broker.broker_type}
                      </Badge>
                      <StatusDot status={status} showLabel />
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 text-sm">
                      <div>
                        <span className="text-text-400">API Key:</span>{' '}
                        <span className="text-text-700 font-mono">
                          {broker.api_key_masked}
                        </span>
                      </div>
                      <div>
                        <span className="text-text-400">URL:</span>{' '}
                        <span className="text-text-700">
                          {broker.base_url.includes('paper') ? 'Paper' : 'Live'}
                        </span>
                      </div>
                      <div>
                        <span className="text-text-400">Created:</span>{' '}
                        <span className="text-text-700">
                          {formatDateTime(new Date(broker.created_at))}
                        </span>
                      </div>
                    </div>

                    {/* Health Status Details */}
                    {health && (
                      <div className="mt-4 p-3 bg-panel-500 rounded">
                        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-sm">
                          <div>
                            <span className="text-text-400">Connected:</span>{' '}
                            <span
                              className={
                                health.is_connected ? 'text-profit-500' : 'text-loss-500'
                              }
                            >
                              {health.is_connected ? 'Yes' : 'No'}
                            </span>
                          </div>
                          <div>
                            <span className="text-text-400">Authenticated:</span>{' '}
                            <span
                              className={
                                health.is_authenticated ? 'text-profit-500' : 'text-loss-500'
                              }
                            >
                              {health.is_authenticated ? 'Yes' : 'No'}
                            </span>
                          </div>
                          {health.latency_ms !== null && (
                            <div>
                              <span className="text-text-400">Latency:</span>{' '}
                              <span className="text-text-700">
                                {health.latency_ms.toFixed(0)}ms
                              </span>
                            </div>
                          )}
                          {health.error_message && (
                            <div className="col-span-full">
                              <span className="text-text-400">Error:</span>{' '}
                              <span className="text-loss-500">{health.error_message}</span>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                <CardFooter className="flex gap-3">
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => handleTestConnection(broker.id)}
                    isLoading={isChecking}
                  >
                    Test Connection
                  </Button>

                  {deletingId === broker.id ? (
                    <>
                      <span className="text-sm text-text-400 self-center">
                        Delete this broker?
                      </span>
                      <Button
                        variant="danger"
                        size="sm"
                        onClick={() => handleDelete(broker.id)}
                      >
                        Confirm
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setDeletingId(null)}
                      >
                        Cancel
                      </Button>
                    </>
                  ) : (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setDeletingId(broker.id)}
                    >
                      Delete
                    </Button>
                  )}
                </CardFooter>
              </Card>
            );
          })
        )}
      </div>
    </PageContainer>
  );
}
