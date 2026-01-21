'use client';

import React, { useState, useEffect } from 'react';
import { api, Trade } from '../utils/api';

/**
 * Trade History Page Component
 *
 * Features:
 * - Displays trade log table with key trade details
 * - Pagination (20 trades per page)
 * - CSV export functionality
 * - Trade detail tooltips with signal explanations
 * - Color-coded P&L (green profit, red loss)
 * - Responsive design
 */

// Helper function to format currency
const formatCurrency = (value: number): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
};

// Helper function to format date
const formatDate = (timestamp: number): string => {
  return new Date(timestamp * 1000).toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

// Helper function to format percentage
const formatPercent = (value: number): string => {
  return `${(value * 100).toFixed(2)}%`;
};

// Helper function to convert trades to CSV
const convertToCSV = (trades: Trade[]): string => {
  const headers = [
    'Symbol',
    'Side',
    'Entry Time',
    'Exit Time',
    'Entry Price',
    'Exit Price',
    'Shares',
    'Realized P&L',
    'Return %',
    'Strategy',
    'Regime',
    'Stop Loss',
    'Take Profit',
    'Signal Reason',
  ];

  const rows = trades.map((trade) => [
    trade.symbol,
    trade.side,
    trade.entry_time ? formatDate(trade.entry_time) : '',
    trade.exit_time ? formatDate(trade.exit_time) : '',
    trade.entry_price?.toFixed(2) || '',
    trade.exit_price?.toFixed(2) || '',
    trade.shares?.toString() || '',
    trade.realized_pnl?.toFixed(2) || '',
    trade.realized_pnl && trade.entry_price && trade.shares
      ? formatPercent(trade.realized_pnl / (trade.entry_price * trade.shares))
      : '',
    trade.strategy || '',
    trade.regime || '',
    trade.stop_loss?.toFixed(2) || '',
    trade.take_profit?.toFixed(2) || '',
    trade.signal_reason || '',
  ]);

  const csvContent = [
    headers.join(','),
    ...rows.map((row) =>
      row.map((cell) => (cell.includes(',') ? `"${cell}"` : cell)).join(',')
    ),
  ].join('\n');

  return csvContent;
};

// Helper function to download CSV
const downloadCSV = (trades: Trade[], filename: string = 'trades.csv'): void => {
  const csv = convertToCSV(trades);
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  const url = URL.createObjectURL(blob);

  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  link.style.visibility = 'hidden';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

// Tooltip component for trade details
interface TooltipProps {
  trade: Trade;
  children: React.ReactNode;
}

const TradeTooltip: React.FC<TooltipProps> = ({ trade, children }) => {
  const [isVisible, setIsVisible] = useState(false);

  const tooltipContent = (
    <div className="tooltip-content">
      <h4>Trade Details</h4>
      <div className="tooltip-section">
        <p>
          <strong>Signal Reason:</strong>
        </p>
        <p>{trade.signal_reason || 'N/A'}</p>
      </div>
      <div className="tooltip-section">
        <p>
          <strong>Strategy:</strong> {trade.strategy || 'N/A'}
        </p>
        <p>
          <strong>Market Regime:</strong> {trade.regime || 'N/A'}
        </p>
      </div>
      <div className="tooltip-section">
        <p>
          <strong>Entry:</strong> {formatCurrency(trade.entry_price || 0)} @{' '}
          {trade.entry_time ? formatDate(trade.entry_time) : 'N/A'}
        </p>
        <p>
          <strong>Exit:</strong> {formatCurrency(trade.exit_price || 0)} @{' '}
          {trade.exit_time ? formatDate(trade.exit_time) : 'N/A'}
        </p>
      </div>
      <div className="tooltip-section">
        <p>
          <strong>Stop Loss:</strong> {formatCurrency(trade.stop_loss || 0)}
        </p>
        <p>
          <strong>Take Profit:</strong> {formatCurrency(trade.take_profit || 0)}
        </p>
      </div>
    </div>
  );

  return (
    <div
      className="tooltip-container"
      onMouseEnter={() => setIsVisible(true)}
      onMouseLeave={() => setIsVisible(false)}
    >
      {children}
      {isVisible && <div className="tooltip-popup">{tooltipContent}</div>}
    </div>
  );
};

// Main Trade History Page Component
const TradeHistoryPage: React.FC = () => {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState<number>(1);
  const tradesPerPage = 20;

  // Fetch trades from API
  useEffect(() => {
    const fetchTrades = async () => {
      try {
        setLoading(true);
        setError(null);
        const data = await api.getTrades(currentPage, tradesPerPage);
        setTrades(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch trades');
      } finally {
        setLoading(false);
      }
    };

    fetchTrades();
  }, [currentPage]);

  // Calculate pagination info
  const totalPages = Math.max(1, Math.ceil(trades.length / tradesPerPage));
  const startIndex = 0; // API returns paginated data
  const endIndex = trades.length;
  const currentTrades = trades.slice(startIndex, endIndex);

  // Handle page change
  const handlePageChange = (newPage: number) => {
    if (newPage >= 1 && newPage <= totalPages) {
      setCurrentPage(newPage);
    }
  };

  // Handle CSV export
  const handleExport = () => {
    if (trades.length === 0) {
      alert('No trades to export');
      return;
    }
    const timestamp = new Date().toISOString().split('T')[0];
    downloadCSV(trades, `trades_${timestamp}.csv`);
  };

  return (
    <div className="trade-history-page">
      <div className="page-header">
        <h1>Trade History</h1>
        <button onClick={handleExport} className="export-button" disabled={loading}>
          📥 Export CSV
        </button>
      </div>

      {error && (
        <div className="error-message">
          <p>Error: {error}</p>
        </div>
      )}

      {loading ? (
        <div className="loading-spinner">
          <p>Loading trades...</p>
        </div>
      ) : (
        <>
          <div className="table-container">
            <table className="trades-table">
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Side</th>
                  <th>Entry</th>
                  <th>Exit</th>
                  <th>Shares</th>
                  <th>P&L</th>
                  <th>Return %</th>
                  <th>Strategy</th>
                  <th>Details</th>
                </tr>
              </thead>
              <tbody>
                {currentTrades.length === 0 ? (
                  <tr>
                    <td colSpan={9} className="no-trades">
                      No trades found
                    </td>
                  </tr>
                ) : (
                  currentTrades.map((trade) => {
                    const returnPct =
                      trade.realized_pnl && trade.entry_price && trade.shares
                        ? (trade.realized_pnl / (trade.entry_price * trade.shares)) * 100
                        : 0;
                    const pnlClass =
                      (trade.realized_pnl || 0) > 0
                        ? 'pnl-positive'
                        : (trade.realized_pnl || 0) < 0
                        ? 'pnl-negative'
                        : 'pnl-neutral';

                    return (
                      <tr key={trade.id}>
                        <td className="symbol">{trade.symbol}</td>
                        <td className={`side ${trade.side?.toLowerCase()}`}>
                          {trade.side}
                        </td>
                        <td className="price">
                          {formatCurrency(trade.entry_price || 0)}
                          <br />
                          <span className="timestamp">
                            {trade.entry_time ? formatDate(trade.entry_time) : 'N/A'}
                          </span>
                        </td>
                        <td className="price">
                          {formatCurrency(trade.exit_price || 0)}
                          <br />
                          <span className="timestamp">
                            {trade.exit_time ? formatDate(trade.exit_time) : 'N/A'}
                          </span>
                        </td>
                        <td className="shares">{trade.shares || 0}</td>
                        <td className={`pnl ${pnlClass}`}>
                          {formatCurrency(trade.realized_pnl || 0)}
                        </td>
                        <td className={`return ${pnlClass}`}>
                          {returnPct.toFixed(2)}%
                        </td>
                        <td className="strategy">{trade.strategy || 'N/A'}</td>
                        <td className="details">
                          <TradeTooltip trade={trade}>
                            <button className="info-button">ℹ️</button>
                          </TradeTooltip>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>

          {/* Pagination Controls */}
          <div className="pagination">
            <button
              onClick={() => handlePageChange(currentPage - 1)}
              disabled={currentPage === 1}
              className="pagination-button"
            >
              ← Previous
            </button>
            <span className="pagination-info">
              Page {currentPage} of {totalPages}
            </span>
            <button
              onClick={() => handlePageChange(currentPage + 1)}
              disabled={currentPage >= totalPages}
              className="pagination-button"
            >
              Next →
            </button>
          </div>
        </>
      )}

      <style jsx>{`
        .trade-history-page {
          padding: 2rem;
          max-width: 1400px;
          margin: 0 auto;
        }

        .page-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 2rem;
        }

        .page-header h1 {
          font-size: 2rem;
          font-weight: 600;
          color: #1a1a1a;
        }

        .export-button {
          padding: 0.75rem 1.5rem;
          background-color: #0070f3;
          color: white;
          border: none;
          border-radius: 6px;
          font-size: 1rem;
          cursor: pointer;
          transition: background-color 0.2s;
        }

        .export-button:hover:not(:disabled) {
          background-color: #0051cc;
        }

        .export-button:disabled {
          background-color: #ccc;
          cursor: not-allowed;
        }

        .error-message {
          padding: 1rem;
          background-color: #fee;
          border: 1px solid #fcc;
          border-radius: 6px;
          color: #c00;
          margin-bottom: 1rem;
        }

        .loading-spinner {
          text-align: center;
          padding: 3rem;
          font-size: 1.2rem;
          color: #666;
        }

        .table-container {
          overflow-x: auto;
          border-radius: 8px;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
          margin-bottom: 1.5rem;
        }

        .trades-table {
          width: 100%;
          border-collapse: collapse;
          background-color: white;
        }

        .trades-table th {
          background-color: #f5f5f5;
          padding: 1rem;
          text-align: left;
          font-weight: 600;
          color: #333;
          border-bottom: 2px solid #ddd;
        }

        .trades-table td {
          padding: 1rem;
          border-bottom: 1px solid #eee;
          color: #333;
        }

        .trades-table tbody tr:hover {
          background-color: #f9f9f9;
        }

        .no-trades {
          text-align: center;
          padding: 3rem !important;
          color: #999;
          font-style: italic;
        }

        .symbol {
          font-weight: 600;
          font-size: 1.1rem;
        }

        .side {
          font-weight: 600;
          text-transform: uppercase;
        }

        .side.long {
          color: #10b981;
        }

        .side.short {
          color: #ef4444;
        }

        .price {
          font-family: 'Courier New', monospace;
        }

        .timestamp {
          font-size: 0.85rem;
          color: #666;
        }

        .shares {
          text-align: right;
        }

        .pnl,
        .return {
          font-weight: 600;
          font-family: 'Courier New', monospace;
        }

        .pnl-positive {
          color: #10b981;
        }

        .pnl-negative {
          color: #ef4444;
        }

        .pnl-neutral {
          color: #666;
        }

        .strategy {
          font-size: 0.9rem;
          color: #666;
        }

        .details {
          text-align: center;
        }

        .info-button {
          background: none;
          border: none;
          font-size: 1.2rem;
          cursor: pointer;
          padding: 0.25rem 0.5rem;
          border-radius: 4px;
          transition: background-color 0.2s;
        }

        .info-button:hover {
          background-color: #f0f0f0;
        }

        .tooltip-container {
          position: relative;
          display: inline-block;
        }

        .tooltip-popup {
          position: absolute;
          bottom: 100%;
          left: 50%;
          transform: translateX(-50%);
          background-color: #333;
          color: white;
          padding: 1rem;
          border-radius: 6px;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
          z-index: 1000;
          min-width: 300px;
          margin-bottom: 0.5rem;
        }

        .tooltip-popup::after {
          content: '';
          position: absolute;
          top: 100%;
          left: 50%;
          transform: translateX(-50%);
          border: 8px solid transparent;
          border-top-color: #333;
        }

        .tooltip-content h4 {
          margin: 0 0 0.75rem 0;
          font-size: 1rem;
          border-bottom: 1px solid #555;
          padding-bottom: 0.5rem;
        }

        .tooltip-section {
          margin-bottom: 0.75rem;
        }

        .tooltip-section:last-child {
          margin-bottom: 0;
        }

        .tooltip-section p {
          margin: 0.25rem 0;
          font-size: 0.9rem;
          line-height: 1.4;
        }

        .pagination {
          display: flex;
          justify-content: center;
          align-items: center;
          gap: 1rem;
          margin-top: 2rem;
        }

        .pagination-button {
          padding: 0.5rem 1rem;
          background-color: #f5f5f5;
          border: 1px solid #ddd;
          border-radius: 6px;
          cursor: pointer;
          transition: background-color 0.2s;
        }

        .pagination-button:hover:not(:disabled) {
          background-color: #e0e0e0;
        }

        .pagination-button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .pagination-info {
          font-weight: 500;
          color: #666;
        }

        @media (max-width: 768px) {
          .trade-history-page {
            padding: 1rem;
          }

          .page-header {
            flex-direction: column;
            gap: 1rem;
            align-items: flex-start;
          }

          .trades-table th,
          .trades-table td {
            padding: 0.5rem;
            font-size: 0.9rem;
          }

          .tooltip-popup {
            min-width: 250px;
          }
        }
      `}</style>
    </div>
  );
};

export default TradeHistoryPage;
