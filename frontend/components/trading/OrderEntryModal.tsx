/**
 * OrderEntryModal - Simple modal for placing buy/sell orders
 *
 * Features:
 * - Mode toggle (Paper/Live)
 * - Symbol search using existing SymbolSearch component
 * - Side toggle (Buy/Sell)
 * - Quantity input
 * - Order type (Market/Limit)
 * - Limit price input (conditional)
 * - Confirmation for live orders
 */

'use client';

import React, { useState, useEffect } from 'react';
import { cn } from '../../lib/utils';
import { apiClient } from '../../utils/api';
import { SymbolSearch } from './SymbolSearch';

export interface OrderEntryModalProps {
  isOpen: boolean;
  onClose: () => void;
  defaultMode?: 'paper' | 'live';
  onOrderPlaced?: () => void;
}

export function OrderEntryModal({
  isOpen,
  onClose,
  defaultMode = 'paper',
  onOrderPlaced,
}: OrderEntryModalProps) {
  const [mode, setMode] = useState<'paper' | 'live'>(defaultMode);
  const [symbol, setSymbol] = useState('');
  const [symbolName, setSymbolName] = useState('');
  const [side, setSide] = useState<'buy' | 'sell'>('buy');
  const [quantity, setQuantity] = useState<string>('');
  const [orderType, setOrderType] = useState<'market' | 'limit'>('market');
  const [limitPrice, setLimitPrice] = useState<string>('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showConfirmation, setShowConfirmation] = useState(false);

  // Reset form when modal opens
  useEffect(() => {
    if (isOpen) {
      setMode(defaultMode);
      setSymbol('');
      setSymbolName('');
      setSide('buy');
      setQuantity('');
      setOrderType('market');
      setLimitPrice('');
      setError(null);
      setShowConfirmation(false);
    }
  }, [isOpen, defaultMode]);

  // Close on ESC key
  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleEsc);
    return () => window.removeEventListener('keydown', handleEsc);
  }, [isOpen, onClose]);

  const handleSubmit = async () => {
    // Validate inputs
    if (!symbol.trim()) {
      setError('Please enter a symbol');
      return;
    }
    const qty = parseInt(quantity, 10);
    if (isNaN(qty) || qty <= 0) {
      setError('Please enter a valid quantity');
      return;
    }
    if (orderType === 'limit') {
      const price = parseFloat(limitPrice);
      if (isNaN(price) || price <= 0) {
        setError('Please enter a valid limit price');
        return;
      }
    }

    // For live mode, require confirmation
    if (mode === 'live' && !showConfirmation) {
      setShowConfirmation(true);
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      const price = orderType === 'limit' ? parseFloat(limitPrice) : undefined;

      const orderRequest = {
        symbol,
        qty,
        side,
        order_type: orderType,
        limit_price: price || null,
      };

      if (mode === 'paper') {
        await apiClient.placePaperOrder(orderRequest);
      } else {
        await apiClient.placeLiveOrder(orderRequest, true);
      }

      // Success - close modal and notify parent
      onClose();
      onOrderPlaced?.();
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to place order';
      setError(message);
      setShowConfirmation(false);
    } finally {
      setIsSubmitting(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-panel-900/80"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative bg-panel-700 rounded-xl shadow-xl w-full max-w-md mx-4 overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-panel-500">
          <h2 className="text-lg font-semibold text-text-900">Place Order</h2>
          <button
            onClick={onClose}
            className="w-8 h-8 flex items-center justify-center rounded-lg hover:bg-panel-500 text-text-400 hover:text-text-800 transition-colors"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Body */}
        <div className="p-4 space-y-4">
          {/* Mode Toggle */}
          <div>
            <label className="block text-xs text-text-400 mb-1">Mode</label>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => setMode('paper')}
                className={cn(
                  'flex-1 py-2 px-3 rounded-lg text-sm font-medium transition-colors',
                  mode === 'paper'
                    ? 'bg-profit-500 text-white'
                    : 'bg-panel-500 text-text-400 hover:bg-panel-400'
                )}
              >
                Paper
              </button>
              <button
                type="button"
                onClick={() => setMode('live')}
                className={cn(
                  'flex-1 py-2 px-3 rounded-lg text-sm font-medium transition-colors',
                  mode === 'live'
                    ? 'bg-loss-500 text-white'
                    : 'bg-panel-500 text-text-400 hover:bg-panel-400'
                )}
              >
                Live
              </button>
            </div>
          </div>

          {/* Symbol Search */}
          <div>
            <SymbolSearch
              value={symbol}
              onChange={(sym, name) => {
                setSymbol(sym);
                if (name) setSymbolName(name);
              }}
              label="Symbol"
              placeholder="Search symbol..."
            />
            {symbolName && (
              <p className="text-xs text-text-400 mt-1">{symbolName}</p>
            )}
          </div>

          {/* Side Toggle */}
          <div>
            <label className="block text-xs text-text-400 mb-1">Side</label>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => setSide('buy')}
                className={cn(
                  'flex-1 py-2 px-3 rounded-lg text-sm font-medium transition-colors',
                  side === 'buy'
                    ? 'bg-profit-500 text-white'
                    : 'bg-panel-500 text-text-400 hover:bg-panel-400'
                )}
              >
                BUY
              </button>
              <button
                type="button"
                onClick={() => setSide('sell')}
                className={cn(
                  'flex-1 py-2 px-3 rounded-lg text-sm font-medium transition-colors',
                  side === 'sell'
                    ? 'bg-loss-500 text-white'
                    : 'bg-panel-500 text-text-400 hover:bg-panel-400'
                )}
              >
                SELL
              </button>
            </div>
          </div>

          {/* Quantity */}
          <div>
            <label className="block text-xs text-text-400 mb-1">Quantity (shares)</label>
            <input
              type="number"
              value={quantity}
              onChange={(e) => setQuantity(e.target.value)}
              placeholder="0"
              min="1"
              className="w-full bg-panel-500 text-text-800 rounded-lg px-3 py-2 text-sm border-none focus:outline-none focus:ring-2 focus:ring-accent-500"
            />
          </div>

          {/* Order Type */}
          <div>
            <label className="block text-xs text-text-400 mb-1">Order Type</label>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => setOrderType('market')}
                className={cn(
                  'flex-1 py-2 px-3 rounded-lg text-sm font-medium transition-colors',
                  orderType === 'market'
                    ? 'bg-accent-500 text-white'
                    : 'bg-panel-500 text-text-400 hover:bg-panel-400'
                )}
              >
                Market
              </button>
              <button
                type="button"
                onClick={() => setOrderType('limit')}
                className={cn(
                  'flex-1 py-2 px-3 rounded-lg text-sm font-medium transition-colors',
                  orderType === 'limit'
                    ? 'bg-accent-500 text-white'
                    : 'bg-panel-500 text-text-400 hover:bg-panel-400'
                )}
              >
                Limit
              </button>
            </div>
          </div>

          {/* Limit Price (conditional) */}
          {orderType === 'limit' && (
            <div>
              <label className="block text-xs text-text-400 mb-1">Limit Price ($)</label>
              <input
                type="number"
                value={limitPrice}
                onChange={(e) => setLimitPrice(e.target.value)}
                placeholder="0.00"
                step="0.01"
                min="0.01"
                className="w-full bg-panel-500 text-text-800 rounded-lg px-3 py-2 text-sm border-none focus:outline-none focus:ring-2 focus:ring-accent-500"
              />
            </div>
          )}

          {/* Error Display */}
          {error && (
            <div className="p-2 bg-loss-500/20 border border-loss-500/50 rounded-lg">
              <p className="text-loss-500 text-sm">{error}</p>
            </div>
          )}

          {/* Live Mode Confirmation */}
          {showConfirmation && mode === 'live' && (
            <div className="p-3 bg-loss-500/20 border border-loss-500 rounded-lg">
              <p className="text-loss-500 font-medium text-sm mb-1">Confirm Live Order</p>
              <p className="text-text-400 text-xs">
                This will place a real order using real money. Are you sure?
              </p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex gap-2 px-4 py-3 border-t border-panel-500 bg-panel-600">
          <button
            type="button"
            onClick={onClose}
            disabled={isSubmitting}
            className="flex-1 py-2 px-4 rounded-lg text-sm font-medium bg-panel-500 text-text-400 hover:bg-panel-400 transition-colors disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleSubmit}
            disabled={isSubmitting}
            className={cn(
              'flex-1 py-2 px-4 rounded-lg text-sm font-medium text-white transition-colors disabled:opacity-50',
              showConfirmation && mode === 'live'
                ? 'bg-loss-500 hover:bg-loss-600'
                : side === 'buy'
                ? 'bg-profit-500 hover:bg-profit-600'
                : 'bg-loss-500 hover:bg-loss-600'
            )}
          >
            {isSubmitting ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Placing...
              </span>
            ) : showConfirmation && mode === 'live' ? (
              'Confirm Order'
            ) : (
              'Place Order'
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

export default OrderEntryModal;
