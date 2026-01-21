'use client';

import React, { useState, useEffect } from 'react';
import { api, Trade } from '../utils/api';
import LoadingSpinner, { SkeletonTable } from '../components/LoadingSpinner';

/**
 * Signal Explainer Archive Viewer Component
 *
 * Features (Phase 15 - Task 5):
 * - View all historical trade signals with complete explanations
 * - Filter by symbol, strategy, regime, date range
 * - Search signal explanations
 * - Sort by various criteria (date, P&L, strategy)
 * - Detailed signal analysis view
 * - Export filtered results to CSV
 * - Color-coded by signal outcome (profitable/loss)
 */

// Signal explanation interface (parsed from Trade.signal_explanation JSON)
interface SignalExplanation {
  symbol: string;
  signal_type: number;
  price: number;
  timestamp: number;
  strategy_mode: number;
  regime: number;
  volatility_state: number;
  atr: number;
  kama: number;
  rsi?: number;
  adx?: number;
  r_squared?: number;
  risk_amount: number;
  risk_percent: number;
  stop_loss: number;
  position_size: number;
  entry_trigger: string;
  noise_filtered: boolean;
  volume_validated: boolean;
  formatted_reason?: string;
  compact_reason?: string;
}

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

// Helper function to format date for input
const formatDateForInput = (timestamp: number): string => {
  return new Date(timestamp * 1000).toISOString().split('T')[0];
};

// Helper function to get signal type name
const getSignalTypeName = (signalType: number): string => {
  const types: Record<number, string> = {
    0: 'NONE',
    1: 'LONG',
    '-1': 'SHORT',
    2: 'EXIT_LONG',
    '-2': 'EXIT_SHORT',
  };
  return types[signalType] || 'UNKNOWN';
};

// Helper function to get regime name
const getRegimeName = (regime: number | string): string => {
  if (typeof regime === 'string') return regime;
  const regimes: Record<number, string> = {
    0: 'MEAN_REVERSION',
    1: 'NEUTRAL',
    2: 'STRONG_TREND',
  };
  return regimes[regime] || 'UNKNOWN';
};

// Helper function to get volatility state name
const getVolatilityStateName = (state: number): string => {
  const states: Record<number, string> = {
    0: 'LOW',
    1: 'NORMAL',
    2: 'HIGH',
  };
  return states[state] || 'UNKNOWN';
};

// Parse signal explanation from trade
const parseSignalExplanation = (trade: Trade): SignalExplanation | null => {
  if (!trade.signal_explanation) return null;

  try {
    // If it's already an object, return it
    if (typeof trade.signal_explanation === 'object') {
      return trade.signal_explanation as SignalExplanation;
    }
    // Otherwise parse as JSON
    return JSON.parse(trade.signal_explanation) as SignalExplanation;
  } catch (e) {
    console.error('Failed to parse signal explanation:', e);
    return null;
  }
};

// Detail modal component
interface DetailModalProps {
  trade: Trade;
  explanation: SignalExplanation | null;
  onClose: () => void;
}

const DetailModal: React.FC<DetailModalProps> = ({ trade, explanation, onClose }) => {
  if (!explanation) return null;

  const pnl = trade.realized_pnl || 0;
  const pnlClass = pnl > 0 ? 'positive' : pnl < 0 ? 'negative' : 'neutral';
  const outcome = pnl > 0 ? '✅ PROFITABLE' : pnl < 0 ? '❌ LOSS' : '⚪ BREAKEVEN';

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Signal Analysis: {trade.symbol}</h2>
          <button onClick={onClose} className="close-button">✕</button>
        </div>

        <div className="modal-body">
          {/* Outcome Banner */}
          <div className={`outcome-banner ${pnlClass}`}>
            <h3>{outcome}</h3>
            <div className="outcome-stats">
              <span>P&L: {formatCurrency(pnl)}</span>
              <span>Return: {((pnl / ((trade.entry_price || 0) * (trade.shares || 1))) * 100).toFixed(2)}%</span>
            </div>
          </div>

          {/* Signal Details */}
          <div className="detail-section">
            <h4>Signal Details</h4>
            <div className="detail-grid">
              <div className="detail-item">
                <span className="label">Signal Type:</span>
                <span className="value">{getSignalTypeName(explanation.signal_type)}</span>
              </div>
              <div className="detail-item">
                <span className="label">Entry Price:</span>
                <span className="value">{formatCurrency(explanation.price)}</span>
              </div>
              <div className="detail-item">
                <span className="label">Timestamp:</span>
                <span className="value">{formatDate(explanation.timestamp)}</span>
              </div>
              <div className="detail-item">
                <span className="label">Entry Trigger:</span>
                <span className="value">{explanation.entry_trigger}</span>
              </div>
            </div>
          </div>

          {/* Market Context */}
          <div className="detail-section">
            <h4>Market Context</h4>
            <div className="detail-grid">
              <div className="detail-item">
                <span className="label">Regime:</span>
                <span className="value">{getRegimeName(explanation.regime)}</span>
              </div>
              <div className="detail-item">
                <span className="label">Volatility:</span>
                <span className="value">{getVolatilityStateName(explanation.volatility_state)}</span>
              </div>
              <div className="detail-item">
                <span className="label">ATR:</span>
                <span className="value">{explanation.atr.toFixed(2)}</span>
              </div>
              <div className="detail-item">
                <span className="label">KAMA:</span>
                <span className="value">{formatCurrency(explanation.kama)}</span>
              </div>
            </div>
          </div>

          {/* Technical Indicators */}
          <div className="detail-section">
            <h4>Technical Indicators</h4>
            <div className="detail-grid">
              {explanation.rsi !== undefined && explanation.rsi !== null && (
                <div className="detail-item">
                  <span className="label">RSI:</span>
                  <span className="value">{explanation.rsi.toFixed(1)}</span>
                </div>
              )}
              {explanation.adx !== undefined && explanation.adx !== null && (
                <div className="detail-item">
                  <span className="label">ADX:</span>
                  <span className="value">{explanation.adx.toFixed(1)}</span>
                </div>
              )}
              {explanation.r_squared !== undefined && explanation.r_squared !== null && (
                <div className="detail-item">
                  <span className="label">R²:</span>
                  <span className="value">{explanation.r_squared.toFixed(3)}</span>
                </div>
              )}
            </div>
          </div>

          {/* Risk Management */}
          <div className="detail-section">
            <h4>Risk Management</h4>
            <div className="detail-grid">
              <div className="detail-item">
                <span className="label">Risk Amount:</span>
                <span className="value">{formatCurrency(explanation.risk_amount)}</span>
              </div>
              <div className="detail-item">
                <span className="label">Risk %:</span>
                <span className="value">{(explanation.risk_percent * 100).toFixed(2)}%</span>
              </div>
              <div className="detail-item">
                <span className="label">Position Size:</span>
                <span className="value">{explanation.position_size} shares</span>
              </div>
              <div className="detail-item">
                <span className="label">Stop Loss:</span>
                <span className="value">{formatCurrency(explanation.stop_loss)}</span>
              </div>
            </div>
          </div>

          {/* Validation Checks */}
          <div className="detail-section">
            <h4>Validation Checks</h4>
            <div className="detail-grid">
              <div className="detail-item">
                <span className="label">Noise Filter:</span>
                <span className="value">{explanation.noise_filtered ? '✅ Passed' : '❌ Failed'}</span>
              </div>
              <div className="detail-item">
                <span className="label">Volume Validation:</span>
                <span className="value">{explanation.volume_validated ? '✅ Passed' : '❌ Failed'}</span>
              </div>
            </div>
          </div>

          {/* Formatted Explanation */}
          {explanation.formatted_reason && (
            <div className="detail-section">
              <h4>Signal Explanation</h4>
              <pre className="formatted-reason">{explanation.formatted_reason}</pre>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// Main Signal Archive Viewer Component
const SignalArchiveViewer: React.FC = () => {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [filteredTrades, setFilteredTrades] = useState<Trade[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedTrade, setSelectedTrade] = useState<Trade | null>(null);

  // Filters
  const [symbolFilter, setSymbolFilter] = useState<string>('');
  const [strategyFilter, setStrategyFilter] = useState<string>('');
  const [regimeFilter, setRegimeFilter] = useState<string>('');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [startDate, setStartDate] = useState<string>('');
  const [endDate, setEndDate] = useState<string>('');
  const [sortBy, setSortBy] = useState<string>('date-desc');

  // Pagination
  const [currentPage, setCurrentPage] = useState<number>(1);
  const itemsPerPage = 20;

  // Fetch all trades
  useEffect(() => {
    const fetchAllTrades = async () => {
      try {
        setLoading(true);
        setError(null);
        // Fetch multiple pages to get all trades
        const allTrades: Trade[] = [];
        let page = 1;
        let hasMore = true;

        while (hasMore && page <= 10) { // Limit to 10 pages (200 trades)
          const data = await api.getTrades(page, 20);
          if (data.length === 0) {
            hasMore = false;
          } else {
            allTrades.push(...data);
            page++;
          }
        }

        setTrades(allTrades);
        setFilteredTrades(allTrades);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch trades');
      } finally {
        setLoading(false);
      }
    };

    fetchAllTrades();
  }, []);

  // Apply filters
  useEffect(() => {
    let filtered = [...trades];

    // Symbol filter
    if (symbolFilter) {
      filtered = filtered.filter(t =>
        t.symbol?.toLowerCase().includes(symbolFilter.toLowerCase())
      );
    }

    // Strategy filter
    if (strategyFilter) {
      filtered = filtered.filter(t => t.strategy === strategyFilter);
    }

    // Regime filter
    if (regimeFilter) {
      filtered = filtered.filter(t => t.regime === regimeFilter);
    }

    // Search query (searches in signal_reason and explanation)
    if (searchQuery) {
      filtered = filtered.filter(t => {
        const reason = t.signal_reason?.toLowerCase() || '';
        const explanation = parseSignalExplanation(t);
        const trigger = explanation?.entry_trigger?.toLowerCase() || '';
        const query = searchQuery.toLowerCase();
        return reason.includes(query) || trigger.includes(query);
      });
    }

    // Date range filter
    if (startDate) {
      const startTimestamp = new Date(startDate).getTime() / 1000;
      filtered = filtered.filter(t => (t.entry_time || 0) >= startTimestamp);
    }
    if (endDate) {
      const endTimestamp = new Date(endDate).getTime() / 1000;
      filtered = filtered.filter(t => (t.entry_time || 0) <= endTimestamp);
    }

    // Sort
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'date-desc':
          return (b.entry_time || 0) - (a.entry_time || 0);
        case 'date-asc':
          return (a.entry_time || 0) - (b.entry_time || 0);
        case 'pnl-desc':
          return (b.realized_pnl || 0) - (a.realized_pnl || 0);
        case 'pnl-asc':
          return (a.realized_pnl || 0) - (b.realized_pnl || 0);
        case 'symbol':
          return (a.symbol || '').localeCompare(b.symbol || '');
        default:
          return 0;
      }
    });

    setFilteredTrades(filtered);
    setCurrentPage(1); // Reset to first page when filters change
  }, [trades, symbolFilter, strategyFilter, regimeFilter, searchQuery, startDate, endDate, sortBy]);

  // Pagination
  const totalPages = Math.max(1, Math.ceil(filteredTrades.length / itemsPerPage));
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const currentTrades = filteredTrades.slice(startIndex, endIndex);

  // Get unique values for filters
  const uniqueSymbols = [...new Set(trades.map(t => t.symbol).filter(Boolean))];
  const uniqueStrategies = [...new Set(trades.map(t => t.strategy).filter(Boolean))];
  const uniqueRegimes = [...new Set(trades.map(t => t.regime).filter(Boolean))];

  // Reset filters
  const resetFilters = () => {
    setSymbolFilter('');
    setStrategyFilter('');
    setRegimeFilter('');
    setSearchQuery('');
    setStartDate('');
    setEndDate('');
    setSortBy('date-desc');
  };

  // Export to CSV
  const exportToCSV = () => {
    const headers = [
      'Symbol', 'Entry Time', 'Signal Type', 'Entry Price', 'Exit Price',
      'P&L', 'Strategy', 'Regime', 'Volatility', 'ATR', 'KAMA',
      'RSI', 'ADX', 'R²', 'Entry Trigger', 'Risk Amount', 'Stop Loss',
    ];

    const rows = filteredTrades.map(trade => {
      const explanation = parseSignalExplanation(trade);
      return [
        trade.symbol,
        trade.entry_time ? formatDate(trade.entry_time) : '',
        explanation ? getSignalTypeName(explanation.signal_type) : '',
        trade.entry_price?.toFixed(2) || '',
        trade.exit_price?.toFixed(2) || '',
        trade.realized_pnl?.toFixed(2) || '',
        trade.strategy || '',
        trade.regime || '',
        explanation ? getVolatilityStateName(explanation.volatility_state) : '',
        explanation?.atr.toFixed(2) || '',
        explanation?.kama.toFixed(2) || '',
        explanation?.rsi?.toFixed(1) || '',
        explanation?.adx?.toFixed(1) || '',
        explanation?.r_squared?.toFixed(3) || '',
        explanation?.entry_trigger || '',
        explanation?.risk_amount.toFixed(2) || '',
        explanation?.stop_loss.toFixed(2) || '',
      ];
    });

    const csvContent = [
      headers.join(','),
      ...rows.map(row =>
        row.map(cell => (cell.toString().includes(',') ? `"${cell}"` : cell)).join(',')
      ),
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `signal_archive_${new Date().toISOString().split('T')[0]}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  if (loading) {
    return (
      <div className="signal-archive-page">
        <h1>Signal Explainer Archive</h1>
        <SkeletonTable rows={10} cols={7} />
      </div>
    );
  }

  return (
    <div className="signal-archive-page">
      <div className="page-header">
        <h1>Signal Explainer Archive</h1>
        <p className="subtitle">
          Analyze historical trading signals with complete context and explanations
        </p>
      </div>

      {error && (
        <div className="error-message">
          <p>Error: {error}</p>
        </div>
      )}

      {/* Filters Section */}
      <div className="filters-section">
        <div className="filters-row">
          <div className="filter-group">
            <label>Symbol</label>
            <select value={symbolFilter} onChange={(e) => setSymbolFilter(e.target.value)}>
              <option value="">All Symbols</option>
              {uniqueSymbols.map(symbol => (
                <option key={symbol} value={symbol}>{symbol}</option>
              ))}
            </select>
          </div>

          <div className="filter-group">
            <label>Strategy</label>
            <select value={strategyFilter} onChange={(e) => setStrategyFilter(e.target.value)}>
              <option value="">All Strategies</option>
              {uniqueStrategies.map(strategy => (
                <option key={strategy} value={strategy}>{strategy}</option>
              ))}
            </select>
          </div>

          <div className="filter-group">
            <label>Regime</label>
            <select value={regimeFilter} onChange={(e) => setRegimeFilter(e.target.value)}>
              <option value="">All Regimes</option>
              {uniqueRegimes.map(regime => (
                <option key={regime} value={regime}>{regime}</option>
              ))}
            </select>
          </div>

          <div className="filter-group">
            <label>Sort By</label>
            <select value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
              <option value="date-desc">Date (Newest)</option>
              <option value="date-asc">Date (Oldest)</option>
              <option value="pnl-desc">P&L (Highest)</option>
              <option value="pnl-asc">P&L (Lowest)</option>
              <option value="symbol">Symbol (A-Z)</option>
            </select>
          </div>
        </div>

        <div className="filters-row">
          <div className="filter-group search-group">
            <label>Search</label>
            <input
              type="text"
              placeholder="Search signals..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>

          <div className="filter-group">
            <label>Start Date</label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
            />
          </div>

          <div className="filter-group">
            <label>End Date</label>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
            />
          </div>

          <div className="filter-group">
            <button onClick={resetFilters} className="reset-button">
              Reset Filters
            </button>
          </div>

          <div className="filter-group">
            <button onClick={exportToCSV} className="export-button">
              📥 Export CSV
            </button>
          </div>
        </div>

        <div className="results-info">
          Showing {currentTrades.length} of {filteredTrades.length} signals
        </div>
      </div>

      {/* Signals Table */}
      <div className="table-container">
        <table className="signals-table">
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Date</th>
              <th>Signal</th>
              <th>Entry Trigger</th>
              <th>Strategy / Regime</th>
              <th>P&L</th>
              <th>Details</th>
            </tr>
          </thead>
          <tbody>
            {currentTrades.length === 0 ? (
              <tr>
                <td colSpan={7} className="no-data">
                  No signals found matching your filters
                </td>
              </tr>
            ) : (
              currentTrades.map((trade) => {
                const explanation = parseSignalExplanation(trade);
                const pnl = trade.realized_pnl || 0;
                const pnlClass = pnl > 0 ? 'positive' : pnl < 0 ? 'negative' : 'neutral';

                return (
                  <tr key={trade.id} className={`signal-row ${pnlClass}`}>
                    <td className="symbol">{trade.symbol}</td>
                    <td className="date">
                      {trade.entry_time ? formatDate(trade.entry_time) : 'N/A'}
                    </td>
                    <td className="signal-type">
                      {explanation ? getSignalTypeName(explanation.signal_type) : 'N/A'}
                    </td>
                    <td className="trigger">
                      {explanation?.entry_trigger || 'N/A'}
                    </td>
                    <td className="strategy">
                      <div>{trade.strategy || 'N/A'}</div>
                      <div className="regime-tag">{trade.regime || 'N/A'}</div>
                    </td>
                    <td className={`pnl ${pnlClass}`}>
                      {formatCurrency(pnl)}
                    </td>
                    <td className="actions">
                      <button
                        onClick={() => setSelectedTrade(trade)}
                        className="view-button"
                      >
                        View Details
                      </button>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      <div className="pagination">
        <button
          onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
          disabled={currentPage === 1}
          className="pagination-button"
        >
          ← Previous
        </button>
        <span className="pagination-info">
          Page {currentPage} of {totalPages}
        </span>
        <button
          onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
          disabled={currentPage >= totalPages}
          className="pagination-button"
        >
          Next →
        </button>
      </div>

      {/* Detail Modal */}
      {selectedTrade && (
        <DetailModal
          trade={selectedTrade}
          explanation={parseSignalExplanation(selectedTrade)}
          onClose={() => setSelectedTrade(null)}
        />
      )}

      <style jsx>{`
        .signal-archive-page {
          padding: 2rem;
          max-width: 1600px;
          margin: 0 auto;
        }

        .page-header {
          margin-bottom: 2rem;
        }

        .page-header h1 {
          font-size: 2rem;
          font-weight: 600;
          color: #1a1a1a;
          margin-bottom: 0.5rem;
        }

        .subtitle {
          color: #666;
          font-size: 1rem;
        }

        .error-message {
          padding: 1rem;
          background-color: #fee;
          border: 1px solid #fcc;
          border-radius: 6px;
          color: #c00;
          margin-bottom: 1rem;
        }

        .filters-section {
          background-color: white;
          padding: 1.5rem;
          border-radius: 8px;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
          margin-bottom: 1.5rem;
        }

        .filters-row {
          display: flex;
          gap: 1rem;
          margin-bottom: 1rem;
          flex-wrap: wrap;
        }

        .filters-row:last-child {
          margin-bottom: 0;
        }

        .filter-group {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
          flex: 1;
          min-width: 150px;
        }

        .search-group {
          flex: 2;
        }

        .filter-group label {
          font-size: 0.875rem;
          font-weight: 500;
          color: #333;
        }

        .filter-group select,
        .filter-group input {
          padding: 0.5rem;
          border: 1px solid #ddd;
          border-radius: 6px;
          font-size: 0.875rem;
        }

        .reset-button,
        .export-button {
          padding: 0.5rem 1rem;
          border: none;
          border-radius: 6px;
          font-size: 0.875rem;
          cursor: pointer;
          transition: background-color 0.2s;
          margin-top: 1.5rem;
        }

        .reset-button {
          background-color: #f5f5f5;
          color: #333;
        }

        .reset-button:hover {
          background-color: #e0e0e0;
        }

        .export-button {
          background-color: #0070f3;
          color: white;
        }

        .export-button:hover {
          background-color: #0051cc;
        }

        .results-info {
          color: #666;
          font-size: 0.875rem;
          margin-top: 0.5rem;
        }

        .table-container {
          overflow-x: auto;
          border-radius: 8px;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
          background-color: white;
          margin-bottom: 1.5rem;
        }

        .signals-table {
          width: 100%;
          border-collapse: collapse;
        }

        .signals-table th {
          background-color: #f5f5f5;
          padding: 1rem;
          text-align: left;
          font-weight: 600;
          color: #333;
          border-bottom: 2px solid #ddd;
        }

        .signals-table td {
          padding: 1rem;
          border-bottom: 1px solid #eee;
          color: #333;
        }

        .signals-table tbody tr:hover {
          background-color: #f9f9f9;
        }

        .no-data {
          text-align: center;
          padding: 3rem !important;
          color: #999;
          font-style: italic;
        }

        .signal-row.positive {
          border-left: 3px solid #10b981;
        }

        .signal-row.negative {
          border-left: 3px solid #ef4444;
        }

        .symbol {
          font-weight: 600;
          font-size: 1.1rem;
        }

        .date {
          font-size: 0.875rem;
          color: #666;
        }

        .signal-type {
          font-weight: 600;
        }

        .trigger {
          font-size: 0.875rem;
          max-width: 250px;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .strategy {
          font-size: 0.875rem;
        }

        .regime-tag {
          display: inline-block;
          padding: 0.25rem 0.5rem;
          background-color: #f0f0f0;
          border-radius: 4px;
          font-size: 0.75rem;
          margin-top: 0.25rem;
          color: #666;
        }

        .pnl {
          font-weight: 600;
          font-family: 'Courier New', monospace;
        }

        .pnl.positive {
          color: #10b981;
        }

        .pnl.negative {
          color: #ef4444;
        }

        .pnl.neutral {
          color: #666;
        }

        .view-button {
          padding: 0.5rem 1rem;
          background-color: #0070f3;
          color: white;
          border: none;
          border-radius: 6px;
          font-size: 0.875rem;
          cursor: pointer;
          transition: background-color 0.2s;
        }

        .view-button:hover {
          background-color: #0051cc;
        }

        .pagination {
          display: flex;
          justify-content: center;
          align-items: center;
          gap: 1rem;
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

        /* Modal Styles */
        .modal-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background-color: rgba(0, 0, 0, 0.5);
          display: flex;
          justify-content: center;
          align-items: center;
          z-index: 1000;
          padding: 1rem;
        }

        .modal-content {
          background-color: white;
          border-radius: 8px;
          max-width: 800px;
          width: 100%;
          max-height: 90vh;
          overflow-y: auto;
          box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }

        .modal-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 1.5rem;
          border-bottom: 1px solid #eee;
        }

        .modal-header h2 {
          margin: 0;
          font-size: 1.5rem;
          font-weight: 600;
        }

        .close-button {
          background: none;
          border: none;
          font-size: 1.5rem;
          cursor: pointer;
          color: #666;
          padding: 0.5rem;
          line-height: 1;
        }

        .close-button:hover {
          color: #333;
        }

        .modal-body {
          padding: 1.5rem;
        }

        .outcome-banner {
          padding: 1.5rem;
          border-radius: 8px;
          margin-bottom: 1.5rem;
          text-align: center;
        }

        .outcome-banner.positive {
          background-color: #d1fae5;
          border: 2px solid #10b981;
        }

        .outcome-banner.negative {
          background-color: #fee2e2;
          border: 2px solid #ef4444;
        }

        .outcome-banner.neutral {
          background-color: #f3f4f6;
          border: 2px solid #9ca3af;
        }

        .outcome-banner h3 {
          margin: 0 0 0.5rem 0;
          font-size: 1.25rem;
        }

        .outcome-stats {
          display: flex;
          justify-content: center;
          gap: 2rem;
          font-weight: 600;
        }

        .detail-section {
          margin-bottom: 1.5rem;
        }

        .detail-section h4 {
          margin: 0 0 1rem 0;
          font-size: 1.125rem;
          font-weight: 600;
          color: #333;
          border-bottom: 1px solid #eee;
          padding-bottom: 0.5rem;
        }

        .detail-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 1rem;
        }

        .detail-item {
          display: flex;
          justify-content: space-between;
        }

        .detail-item .label {
          font-weight: 500;
          color: #666;
        }

        .detail-item .value {
          font-weight: 600;
          color: #333;
        }

        .formatted-reason {
          background-color: #f5f5f5;
          padding: 1rem;
          border-radius: 6px;
          white-space: pre-wrap;
          font-family: 'Courier New', monospace;
          font-size: 0.875rem;
          line-height: 1.6;
          color: #333;
        }

        @media (max-width: 768px) {
          .signal-archive-page {
            padding: 1rem;
          }

          .filters-row {
            flex-direction: column;
          }

          .filter-group {
            min-width: 100%;
          }

          .detail-grid {
            grid-template-columns: 1fr;
          }

          .signals-table th,
          .signals-table td {
            padding: 0.5rem;
            font-size: 0.875rem;
          }
        }
      `}</style>
    </div>
  );
};

export default SignalArchiveViewer;
