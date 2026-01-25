/**
 * Trade Review Modal - Detailed view of a completed trade
 *
 * Shows:
 * - Entry/exit details
 * - P&L and return
 * - Strategy and signal information
 * - Risk management levels
 */

'use client';

import React, { useState, useEffect } from 'react';
import { Button, Badge, Skeleton } from '../ui';
import { PLDisplay } from './PLDisplay';
import { apiClient, ApiError, TradeReviewResponse } from '../../utils/api';
import { formatCurrency } from '../../lib/utils';

export interface TradeReviewModalProps {
  isOpen: boolean;
  onClose: () => void;
  tradeId: number | null;
  mode: 'live' | 'paper';
}

export function TradeReviewModal({
  isOpen,
  onClose,
  tradeId,
  mode,
}: TradeReviewModalProps) {
  const [trade, setTrade] = useState<TradeReviewResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen && tradeId !== null) {
      fetchTradeDetails();
    }
  }, [isOpen, tradeId, mode]);

  const fetchTradeDetails = async () => {
    if (tradeId === null) return;

    setLoading(true);
    setError(null);

    try {
      const data = await apiClient.reviewTrade(tradeId, mode);
      setTrade(data);
    } catch (err) {
      const message = err instanceof ApiError ? err.detail : 'Failed to load trade details';
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-panel-900/90 flex items-center justify-center z-50 p-5">
      <div className="bg-panel-700 rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Modal Header */}
        <div className="px-6 py-5 flex justify-between items-center border-b border-panel-500">
          <h2 className="text-xl font-bold text-text-900">Trade Review</h2>
          <button
            onClick={onClose}
            className="text-text-400 hover:text-text-900 text-2xl font-bold w-8 h-8 flex items-center justify-center"
          >
            &times;
          </button>
        </div>

        {/* Modal Content */}
        <div className="p-6 overflow-y-auto flex-1">
          {loading && (
            <div className="space-y-4">
              <Skeleton variant="title" />
              <Skeleton variant="text" />
              <Skeleton variant="text" />
              <Skeleton variant="text" />
            </div>
          )}

          {error && (
            <div className="p-4 bg-loss-500/20 border border-loss-500 rounded-xl">
              <p className="text-loss-500">{error}</p>
              <Button variant="secondary" onClick={fetchTradeDetails} className="mt-2">
                Retry
              </Button>
            </div>
          )}

          {trade && !loading && (
            <div className="space-y-6">
              {/* Trade Header */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <span className="text-2xl font-bold text-text-900 font-mono">
                    {trade.symbol}
                  </span>
                  <Badge variant={trade.side === 'LONG' ? 'success' : 'error'}>
                    {trade.side}
                  </Badge>
                  <Badge
                    variant={
                      trade.status === 'CLOSED'
                        ? 'neutral'
                        : trade.status === 'OPEN'
                          ? 'success'
                          : 'warning'
                    }
                  >
                    {trade.status}
                  </Badge>
                </div>
                {trade.realized_pnl !== null && (
                  <PLDisplay value={trade.realized_pnl} percent={trade.return_pct ?? undefined} size="lg" />
                )}
              </div>

              {/* Entry Details */}
              <div className="bg-panel-600 rounded-xl p-4">
                <h3 className="text-sm font-semibold text-text-400 uppercase tracking-wide mb-3">
                  Entry Details
                </h3>
                <div className="grid grid-cols-2 gap-4">
                  <DetailRow label="Entry Price" value={formatCurrency(trade.entry_price)} />
                  <DetailRow label="Shares" value={trade.shares.toString()} />
                  <DetailRow
                    label="Entry Time"
                    value={new Date(trade.entry_time).toLocaleString()}
                  />
                  <DetailRow
                    label="Position Value"
                    value={formatCurrency(trade.entry_price * trade.shares)}
                  />
                </div>
              </div>

              {/* Exit Details (if closed) */}
              {trade.status === 'CLOSED' && trade.exit_price && (
                <div className="bg-panel-600 rounded-xl p-4">
                  <h3 className="text-sm font-semibold text-text-400 uppercase tracking-wide mb-3">
                    Exit Details
                  </h3>
                  <div className="grid grid-cols-2 gap-4">
                    <DetailRow label="Exit Price" value={formatCurrency(trade.exit_price)} />
                    <DetailRow
                      label="Exit Time"
                      value={
                        trade.exit_time
                          ? new Date(trade.exit_time).toLocaleString()
                          : '-'
                      }
                    />
                    <DetailRow
                      label="Holding Period"
                      value={
                        trade.holding_period_days !== null
                          ? `${trade.holding_period_days.toFixed(1)} days`
                          : '-'
                      }
                    />
                    <DetailRow
                      label="Return"
                      value={
                        trade.return_pct !== null
                          ? `${trade.return_pct >= 0 ? '+' : ''}${trade.return_pct.toFixed(2)}%`
                          : '-'
                      }
                      valueClass={
                        trade.return_pct !== null
                          ? trade.return_pct >= 0
                            ? 'text-profit-500'
                            : 'text-loss-500'
                          : ''
                      }
                    />
                  </div>
                </div>
              )}

              {/* Risk Management */}
              <div className="bg-panel-600 rounded-xl p-4">
                <h3 className="text-sm font-semibold text-text-400 uppercase tracking-wide mb-3">
                  Risk Management
                </h3>
                <div className="grid grid-cols-2 gap-4">
                  <DetailRow
                    label="Stop Loss"
                    value={formatCurrency(trade.stop_loss)}
                    valueClass="text-loss-500"
                  />
                  <DetailRow
                    label="Take Profit"
                    value={trade.take_profit ? formatCurrency(trade.take_profit) : 'Not Set'}
                    valueClass={trade.take_profit ? 'text-profit-500' : 'text-text-400'}
                  />
                  <DetailRow
                    label="Risk Amount"
                    value={formatCurrency(
                      Math.abs(trade.entry_price - trade.stop_loss) * trade.shares
                    )}
                  />
                  {trade.take_profit && (
                    <DetailRow
                      label="Risk/Reward"
                      value={`1:${(
                        Math.abs(trade.take_profit - trade.entry_price) /
                        Math.abs(trade.entry_price - trade.stop_loss)
                      ).toFixed(2)}`}
                    />
                  )}
                </div>
              </div>

              {/* Strategy Info */}
              <div className="bg-panel-600 rounded-xl p-4">
                <h3 className="text-sm font-semibold text-text-400 uppercase tracking-wide mb-3">
                  Strategy Information
                </h3>
                <div className="space-y-3">
                  <DetailRow label="Strategy" value={trade.strategy} />
                  {trade.regime && <DetailRow label="Market Regime" value={trade.regime} />}
                  {trade.signal_reason && (
                    <DetailRow label="Signal Reason" value={trade.signal_reason} />
                  )}
                  {trade.signal_explanation && (
                    <div>
                      <span className="text-sm text-text-400">Signal Explanation</span>
                      <p className="text-text-900 mt-1 text-sm leading-relaxed">
                        {trade.signal_explanation}
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-4 border-t border-panel-500 flex justify-end">
          <Button variant="secondary" onClick={onClose}>
            Close
          </Button>
        </div>
      </div>
    </div>
  );
}

// Helper component for detail rows
function DetailRow({
  label,
  value,
  valueClass = 'text-text-900',
}: {
  label: string;
  value: string;
  valueClass?: string;
}) {
  return (
    <div>
      <span className="text-sm text-text-400">{label}</span>
      <div className={`font-semibold font-mono ${valueClass}`}>{value}</div>
    </div>
  );
}

export default TradeReviewModal;
