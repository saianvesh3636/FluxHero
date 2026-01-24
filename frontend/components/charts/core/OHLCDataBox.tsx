/**
 * OHLCDataBox - Displays OHLC data for candlestick charts
 *
 * Shows Open, High, Low, Close, Volume, and Change % when hovering
 * on candlestick data points. Updates in real-time with crosshair movement.
 */

'use client';

import React from 'react';
import { CHART_COLORS } from '../config/theme';
import { withOpacity } from '../utils/colorUtils';
import { formatPrice, formatVolume, formatPercent } from '../utils/formatters';

export interface OHLCDisplayData {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

export interface OHLCDataBoxProps {
  data: OHLCDisplayData | null;
  className?: string;
  showVolume?: boolean;
  precision?: number;
}

export function OHLCDataBox({
  data,
  className = '',
  showVolume = true,
  precision = 2,
}: OHLCDataBoxProps) {
  if (!data) return null;

  const change = data.close - data.open;
  const changePercent = data.open !== 0 ? (change / data.open) * 100 : 0;
  const isPositive = change >= 0;

  return (
    <div
      className={`absolute top-2 left-2 flex items-center gap-4 text-xs px-2 py-1.5 rounded z-20 ${className}`}
      style={{ backgroundColor: withOpacity(CHART_COLORS.background, 0.95) }}
    >
      {/* Date/Time */}
      <span style={{ color: CHART_COLORS.textMuted }}>{data.time}</span>

      {/* OHLC Values */}
      <div className="flex items-center gap-3">
        <span style={{ color: CHART_COLORS.text }}>
          <span style={{ color: CHART_COLORS.textMuted }}>O</span>{' '}
          {formatPrice(data.open, precision)}
        </span>
        <span style={{ color: CHART_COLORS.text }}>
          <span style={{ color: CHART_COLORS.textMuted }}>H</span>{' '}
          {formatPrice(data.high, precision)}
        </span>
        <span style={{ color: CHART_COLORS.text }}>
          <span style={{ color: CHART_COLORS.textMuted }}>L</span>{' '}
          {formatPrice(data.low, precision)}
        </span>
        <span style={{ color: CHART_COLORS.text }}>
          <span style={{ color: CHART_COLORS.textMuted }}>C</span>{' '}
          {formatPrice(data.close, precision)}
        </span>
      </div>

      {/* Volume */}
      {showVolume && data.volume !== undefined && (
        <span style={{ color: CHART_COLORS.text }}>
          <span style={{ color: CHART_COLORS.textMuted }}>Vol</span>{' '}
          {formatVolume(data.volume)}
        </span>
      )}

      {/* Change */}
      <span style={{ color: isPositive ? CHART_COLORS.profit : CHART_COLORS.loss }}>
        {isPositive ? '+' : ''}
        {formatPrice(change, precision)} ({formatPercent(changePercent, 2, true)})
      </span>
    </div>
  );
}

export default OHLCDataBox;
