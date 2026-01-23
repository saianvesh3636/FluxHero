'use client';

import React, { useState, useEffect } from 'react';
import { apiClient, Trade } from '../../utils/api';
import { PageContainer, PageHeader, StatsGrid } from '../../components/layout';
import { Card, CardTitle, Button, Badge, Select, Input, Skeleton, Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from '../../components/ui';
import { formatCurrency } from '../../lib/utils';

// Signal explanation interface
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

const ITEMS_PER_PAGE = 20;

// Helper functions
const formatDate = (timestamp: number): string => {
  return new Date(timestamp * 1000).toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

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

const getRegimeName = (regime: number | string): string => {
  if (typeof regime === 'string') return regime;
  const regimes: Record<number, string> = {
    0: 'MEAN_REVERSION',
    1: 'NEUTRAL',
    2: 'STRONG_TREND',
  };
  return regimes[regime] || 'UNKNOWN';
};

const getVolatilityStateName = (state: number): string => {
  const states: Record<number, string> = {
    0: 'LOW',
    1: 'NORMAL',
    2: 'HIGH',
  };
  return states[state] || 'UNKNOWN';
};

const parseSignalExplanation = (trade: Trade): SignalExplanation | null => {
  if (!trade.signal_explanation) return null;
  try {
    if (typeof trade.signal_explanation === 'object') {
      return trade.signal_explanation as SignalExplanation;
    }
    return JSON.parse(trade.signal_explanation) as SignalExplanation;
  } catch {
    return null;
  }
};

const sortOptions = [
  { value: 'date-desc', label: 'Date (Newest)' },
  { value: 'date-asc', label: 'Date (Oldest)' },
  { value: 'pnl-desc', label: 'P&L (Highest)' },
  { value: 'pnl-asc', label: 'P&L (Lowest)' },
  { value: 'symbol', label: 'Symbol (A-Z)' },
];

export default function SignalsPage() {
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

  // Fetch all trades
  useEffect(() => {
    const fetchAllTrades = async () => {
      try {
        setLoading(true);
        setError(null);
        const allTrades: Trade[] = [];
        let page = 1;
        let hasMore = true;

        while (hasMore && page <= 10) {
          const data = await apiClient.getTrades(page, 20);
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

    if (symbolFilter) {
      filtered = filtered.filter(t =>
        t.symbol?.toLowerCase().includes(symbolFilter.toLowerCase())
      );
    }

    if (strategyFilter) {
      filtered = filtered.filter(t => t.strategy === strategyFilter);
    }

    if (regimeFilter) {
      filtered = filtered.filter(t => t.regime === regimeFilter);
    }

    if (searchQuery) {
      filtered = filtered.filter(t => {
        const reason = t.signal_reason?.toLowerCase() || '';
        const explanation = parseSignalExplanation(t);
        const trigger = explanation?.entry_trigger?.toLowerCase() || '';
        const query = searchQuery.toLowerCase();
        return reason.includes(query) || trigger.includes(query);
      });
    }

    if (startDate) {
      const startTimestamp = new Date(startDate).getTime() / 1000;
      filtered = filtered.filter(t => (t.entry_time || 0) >= startTimestamp);
    }
    if (endDate) {
      const endTimestamp = new Date(endDate).getTime() / 1000;
      filtered = filtered.filter(t => (t.entry_time || 0) <= endTimestamp);
    }

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
    setCurrentPage(1);
  }, [trades, symbolFilter, strategyFilter, regimeFilter, searchQuery, startDate, endDate, sortBy]);

  // Pagination
  const totalPages = Math.max(1, Math.ceil(filteredTrades.length / ITEMS_PER_PAGE));
  const startIndex = (currentPage - 1) * ITEMS_PER_PAGE;
  const currentTrades = filteredTrades.slice(startIndex, startIndex + ITEMS_PER_PAGE);

  // Get unique values for filters
  const uniqueSymbols = [...new Set(trades.map(t => t.symbol).filter(Boolean))];
  const uniqueStrategies = [...new Set(trades.map(t => t.strategy).filter(Boolean))];
  const uniqueRegimes = [...new Set(trades.map(t => t.regime).filter(Boolean))];

  const symbolOptions = [{ value: '', label: 'All Symbols' }, ...uniqueSymbols.map(s => ({ value: s!, label: s! }))];
  const strategyOptions = [{ value: '', label: 'All Strategies' }, ...uniqueStrategies.map(s => ({ value: s!, label: s! }))];
  const regimeOptions = [{ value: '', label: 'All Regimes' }, ...uniqueRegimes.map(s => ({ value: s!, label: s! }))];

  const resetFilters = () => {
    setSymbolFilter('');
    setStrategyFilter('');
    setRegimeFilter('');
    setSearchQuery('');
    setStartDate('');
    setEndDate('');
    setSortBy('date-desc');
  };

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
      <PageContainer>
        <PageHeader title="Signal Explainer Archive" subtitle="Loading..." />
        <Card noPadding>
          <div className="p-5">
            {Array.from({ length: 10 }).map((_, i) => (
              <Skeleton key={i} variant="text" className="mb-3" />
            ))}
          </div>
        </Card>
      </PageContainer>
    );
  }

  return (
    <PageContainer>
      <PageHeader
        title="Signal Explainer Archive"
        subtitle="Analyze historical trading signals with complete context"
        actions={
          <Button variant="primary" onClick={exportToCSV} disabled={filteredTrades.length === 0}>
            Export CSV
          </Button>
        }
      />

      {/* Error Display */}
      {error && (
        <Card variant="highlighted" className="mb-6 border-l-4 border-loss-500">
          <p className="text-loss-500 font-medium">Error: {error}</p>
        </Card>
      )}

      {/* Filters */}
      <Card className="mb-6">
        <CardTitle className="mb-4">Filters</CardTitle>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
          <Select
            label="Symbol"
            options={symbolOptions}
            value={symbolFilter}
            onChange={(e) => setSymbolFilter(e.target.value)}
          />
          <Select
            label="Strategy"
            options={strategyOptions}
            value={strategyFilter}
            onChange={(e) => setStrategyFilter(e.target.value)}
          />
          <Select
            label="Regime"
            options={regimeOptions}
            value={regimeFilter}
            onChange={(e) => setRegimeFilter(e.target.value)}
          />
          <Select
            label="Sort By"
            options={sortOptions}
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
          />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
          <Input
            label="Search"
            type="text"
            placeholder="Search signals..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          <Input
            label="Start Date"
            type="date"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
          />
          <Input
            label="End Date"
            type="date"
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
          />
          <div className="flex items-end">
            <Button variant="secondary" onClick={resetFilters} className="w-full">
              Reset Filters
            </Button>
          </div>
        </div>
        <p className="text-sm text-text-400">
          Showing {currentTrades.length} of {filteredTrades.length} signals
        </p>
      </Card>

      {/* Signals Table */}
      <Card noPadding className="mb-6">
        <div className="px-5 py-4 border-b border-panel-500">
          <CardTitle>Signal Log</CardTitle>
        </div>
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Symbol</TableHead>
                <TableHead>Date</TableHead>
                <TableHead>Signal</TableHead>
                <TableHead>Entry Trigger</TableHead>
                <TableHead>Strategy / Regime</TableHead>
                <TableHead align="right">P&L</TableHead>
                <TableHead align="center">Details</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {currentTrades.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={7} className="text-center py-8 text-text-400">
                    No signals found matching your filters
                  </TableCell>
                </TableRow>
              ) : (
                currentTrades.map((trade) => {
                  const explanation = parseSignalExplanation(trade);
                  const pnl = trade.realized_pnl || 0;

                  return (
                    <TableRow
                      key={trade.id}
                      className={`border-l-4 ${
                        pnl > 0 ? 'border-l-profit-500' :
                        pnl < 0 ? 'border-l-loss-500' :
                        'border-l-panel-500'
                      }`}
                    >
                      <TableCell className="font-semibold text-text-900">
                        {trade.symbol}
                      </TableCell>
                      <TableCell className="text-text-400 text-sm">
                        {trade.entry_time ? formatDate(trade.entry_time) : 'N/A'}
                      </TableCell>
                      <TableCell>
                        <Badge
                          variant={explanation?.signal_type === 1 ? 'success' : explanation?.signal_type === -1 ? 'error' : 'neutral'}
                          size="sm"
                        >
                          {explanation ? getSignalTypeName(explanation.signal_type) : 'N/A'}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-sm text-text-400 max-w-48 truncate">
                        {explanation?.entry_trigger || 'N/A'}
                      </TableCell>
                      <TableCell>
                        <div className="text-sm text-text-700">{trade.strategy || 'N/A'}</div>
                        <Badge variant="neutral" size="sm" className="mt-1">
                          {trade.regime || 'N/A'}
                        </Badge>
                      </TableCell>
                      <TableCell
                        align="right"
                        className={`font-mono tabular-nums font-semibold ${
                          pnl > 0 ? 'text-profit-500' :
                          pnl < 0 ? 'text-loss-500' :
                          'text-text-400'
                        }`}
                      >
                        {formatCurrency(pnl)}
                      </TableCell>
                      <TableCell align="center">
                        <Button
                          variant="secondary"
                          size="sm"
                          onClick={() => setSelectedTrade(trade)}
                        >
                          View
                        </Button>
                      </TableCell>
                    </TableRow>
                  );
                })
              )}
            </TableBody>
          </Table>
        </div>
      </Card>

      {/* Pagination */}
      <div className="flex items-center justify-center gap-4">
        <Button
          variant="secondary"
          onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
          disabled={currentPage === 1}
        >
          Previous
        </Button>
        <span className="text-text-400 font-medium">
          Page {currentPage} of {totalPages}
        </span>
        <Button
          variant="secondary"
          onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
          disabled={currentPage >= totalPages}
        >
          Next
        </Button>
      </div>

      {/* Detail Modal */}
      {selectedTrade && (
        <DetailModal
          trade={selectedTrade}
          explanation={parseSignalExplanation(selectedTrade)}
          onClose={() => setSelectedTrade(null)}
        />
      )}
    </PageContainer>
  );
}

// Detail Modal Component
interface DetailModalProps {
  trade: Trade;
  explanation: SignalExplanation | null;
  onClose: () => void;
}

function DetailModal({ trade, explanation, onClose }: DetailModalProps) {
  if (!explanation) return null;

  const pnl = trade.realized_pnl || 0;
  const returnPct = trade.entry_price && trade.shares
    ? (pnl / (trade.entry_price * trade.shares)) * 100
    : 0;

  return (
    <div
      className="fixed inset-0 bg-panel-900/90 flex items-center justify-center z-50 p-5"
      onClick={onClose}
    >
      <div
        className="bg-panel-700 rounded-2xl max-w-3xl w-full max-h-[90vh] overflow-hidden flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="px-6 py-5 flex justify-between items-center border-b border-panel-500">
          <h2 className="text-xl font-bold text-text-900">Signal Analysis: {trade.symbol}</h2>
          <button
            onClick={onClose}
            className="text-text-400 hover:text-text-900 text-2xl font-bold w-8 h-8 flex items-center justify-center"
          >
            ×
          </button>
        </div>

        {/* Modal Content */}
        <div className="p-6 overflow-y-auto flex-1">
          {/* Outcome Banner */}
          <div className={`p-5 rounded-xl mb-6 ${
            pnl > 0 ? 'bg-profit-500/20' :
            pnl < 0 ? 'bg-loss-500/20' :
            'bg-panel-600'
          }`}>
            <h3 className={`text-xl font-bold text-center mb-2 ${
              pnl > 0 ? 'text-profit-500' :
              pnl < 0 ? 'text-loss-500' :
              'text-text-400'
            }`}>
              {pnl > 0 ? 'PROFITABLE' : pnl < 0 ? 'LOSS' : 'BREAKEVEN'}
            </h3>
            <div className="flex justify-center gap-8 text-sm">
              <span className="text-text-400">P&L: <span className="font-semibold text-text-700">{formatCurrency(pnl)}</span></span>
              <span className="text-text-400">Return: <span className="font-semibold text-text-700">{returnPct.toFixed(2)}%</span></span>
            </div>
          </div>

          {/* Signal Details */}
          <DetailSection title="Signal Details">
            <DetailGrid>
              <DetailItem label="Signal Type" value={getSignalTypeName(explanation.signal_type)} />
              <DetailItem label="Entry Price" value={formatCurrency(explanation.price)} />
              <DetailItem label="Timestamp" value={formatDate(explanation.timestamp)} />
              <DetailItem label="Entry Trigger" value={explanation.entry_trigger} />
            </DetailGrid>
          </DetailSection>

          {/* Market Context */}
          <DetailSection title="Market Context">
            <DetailGrid>
              <DetailItem label="Regime" value={getRegimeName(explanation.regime)} />
              <DetailItem label="Volatility" value={getVolatilityStateName(explanation.volatility_state)} />
              <DetailItem label="ATR" value={explanation.atr.toFixed(2)} />
              <DetailItem label="KAMA" value={formatCurrency(explanation.kama)} />
            </DetailGrid>
          </DetailSection>

          {/* Technical Indicators */}
          <DetailSection title="Technical Indicators">
            <DetailGrid>
              {explanation.rsi !== undefined && explanation.rsi !== null && (
                <DetailItem label="RSI" value={explanation.rsi.toFixed(1)} />
              )}
              {explanation.adx !== undefined && explanation.adx !== null && (
                <DetailItem label="ADX" value={explanation.adx.toFixed(1)} />
              )}
              {explanation.r_squared !== undefined && explanation.r_squared !== null && (
                <DetailItem label="R²" value={explanation.r_squared.toFixed(3)} />
              )}
            </DetailGrid>
          </DetailSection>

          {/* Risk Management */}
          <DetailSection title="Risk Management">
            <DetailGrid>
              <DetailItem label="Risk Amount" value={formatCurrency(explanation.risk_amount)} />
              <DetailItem label="Risk %" value={`${(explanation.risk_percent * 100).toFixed(2)}%`} />
              <DetailItem label="Position Size" value={`${explanation.position_size} shares`} />
              <DetailItem label="Stop Loss" value={formatCurrency(explanation.stop_loss)} />
            </DetailGrid>
          </DetailSection>

          {/* Validation Checks */}
          <DetailSection title="Validation Checks">
            <div className="flex gap-6">
              <div className="flex items-center gap-2">
                <div className={`w-4 h-4 rounded-full ${explanation.noise_filtered ? 'bg-profit-500' : 'bg-loss-500'}`} />
                <span className="text-text-700">Noise Filter: {explanation.noise_filtered ? 'Passed' : 'Failed'}</span>
              </div>
              <div className="flex items-center gap-2">
                <div className={`w-4 h-4 rounded-full ${explanation.volume_validated ? 'bg-profit-500' : 'bg-loss-500'}`} />
                <span className="text-text-700">Volume Validation: {explanation.volume_validated ? 'Passed' : 'Failed'}</span>
              </div>
            </div>
          </DetailSection>

          {/* Formatted Explanation */}
          {explanation.formatted_reason && (
            <DetailSection title="Signal Explanation">
              <pre className="bg-panel-600 p-4 rounded-xl text-sm text-text-400 whitespace-pre-wrap font-mono">
                {explanation.formatted_reason}
              </pre>
            </DetailSection>
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

// Helper Components
function DetailSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mb-6">
      <h4 className="text-lg font-semibold text-text-900 mb-3 pb-2 border-b border-panel-500">{title}</h4>
      {children}
    </div>
  );
}

function DetailGrid({ children }: { children: React.ReactNode }) {
  return <div className="grid grid-cols-2 gap-4">{children}</div>;
}

function DetailItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-text-400">{label}:</span>
      <span className="text-text-700 font-medium">{value}</span>
    </div>
  );
}
