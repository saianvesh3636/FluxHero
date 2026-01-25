'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import Link from 'next/link';
import { apiClient, Trade, ApiError, ReportResponse } from '../../utils/api';
import { PageContainer, PageHeader } from '../../components/layout';
import { Card, CardTitle, Button, Badge, Skeleton, Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from '../../components/ui';
import { formatCurrency, formatPercent } from '../../lib/utils';

const TRADES_PER_PAGE = 20;

// Helper function to format date (handles both ISO strings and Unix timestamps)
const formatDate = (timestamp: string | number | null | undefined): string => {
  if (!timestamp) return 'N/A';

  let date: Date;
  if (typeof timestamp === 'string') {
    // ISO string format
    date = new Date(timestamp);
  } else {
    // Unix timestamp (seconds)
    date = new Date(timestamp * 1000);
  }

  if (isNaN(date.getTime())) return 'N/A';

  return date.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
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
      ? ((trade.realized_pnl / (trade.entry_price * trade.shares)) * 100).toFixed(2) + '%'
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

export default function HistoryPage() {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [initialLoad, setInitialLoad] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [totalPages, setTotalPages] = useState<number>(1);
  const [totalCount, setTotalCount] = useState<number>(0);
  const [expandedTradeId, setExpandedTradeId] = useState<number | null>(null);
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const [reportSuccess, setReportSuccess] = useState<ReportResponse | null>(null);
  const [reportError, setReportError] = useState<string | null>(null);

  // Track if a fetch is already in progress to prevent duplicate calls
  const isFetchingRef = useRef(false);
  const lastFetchedPageRef = useRef<number | null>(null);

  // Generate report handler
  const handleGenerateReport = async () => {
    setIsGeneratingReport(true);
    setReportError(null);
    setReportSuccess(null);

    try {
      const response = await apiClient.generateReport({
        mode: 'paper',
        benchmark: 'SPY',
        title: 'Trade History Report',
      });
      setReportSuccess(response);
    } catch (err) {
      const message = err instanceof ApiError ? err.detail : 'Failed to generate report';
      setReportError(message);
    } finally {
      setIsGeneratingReport(false);
    }
  };

  const handleDownloadReport = async () => {
    if (reportSuccess) {
      try {
        await apiClient.downloadReport(reportSuccess.report_id);
      } catch (err) {
        const message = err instanceof ApiError ? err.detail : 'Failed to download';
        setReportError(message);
      }
    }
  };

  // Fetch trades from API
  const fetchTrades = useCallback(async (page: number) => {
    // Prevent duplicate concurrent fetches for the same page
    if (isFetchingRef.current && lastFetchedPageRef.current === page) return;
    isFetchingRef.current = true;
    lastFetchedPageRef.current = page;

    try {
      setLoading(true);
      setError(null);
      const response = await apiClient.getTrades(page, TRADES_PER_PAGE);
      setTrades(response.trades);
      setTotalPages(response.totalPages);
      setTotalCount(response.totalCount);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch trades');
    } finally {
      setLoading(false);
      setInitialLoad(false);
      isFetchingRef.current = false;
    }
  }, []);

  useEffect(() => {
    fetchTrades(currentPage);
  }, [currentPage, fetchTrades]);

  // Handle CSV export
  const handleExport = () => {
    if (trades.length === 0) {
      alert('No trades to export');
      return;
    }
    const timestamp = new Date().toISOString().split('T')[0];
    downloadCSV(trades, `trades_${timestamp}.csv`);
  };

  // Toggle trade details
  const toggleTradeDetails = (tradeId: number) => {
    setExpandedTradeId(expandedTradeId === tradeId ? null : tradeId);
  };

  // Show initial loading state
  if (initialLoad) {
    return (
      <PageContainer>
        <PageHeader title="Trade History" subtitle="Loading..." />
        <Card noPadding>
          <div className="p-5 border-b border-panel-500">
            <Skeleton variant="title" width="40%" />
          </div>
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
        title="Trade History"
        subtitle={`${totalCount} total trades - Page ${currentPage} of ${totalPages}`}
        actions={
          <div className="flex items-center gap-2">
            <Button
              variant="secondary"
              onClick={handleGenerateReport}
              disabled={isGeneratingReport || trades.length === 0}
            >
              {isGeneratingReport ? 'Generating...' : 'Generate Report'}
            </Button>
            <Button variant="primary" onClick={handleExport} disabled={loading || trades.length === 0}>
              Export CSV
            </Button>
          </div>
        }
      />

      {/* Error Display */}
      {error && (
        <Card variant="highlighted" className="mb-6 border-l-4 border-loss-500">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-loss-500 font-medium">Error</p>
              <p className="text-text-400 text-sm">{error}</p>
            </div>
            <Button variant="danger" onClick={() => setCurrentPage(currentPage)}>
              Retry
            </Button>
          </div>
        </Card>
      )}

      {/* Report Success Banner */}
      {reportSuccess && (
        <div className="mb-6 px-4 py-3 rounded-lg border-l-4 border-profit-500 bg-profit-500/10 flex items-center justify-between">
          <div>
            <span className="text-profit-500 font-medium">{reportSuccess.title}</span>
            <span className="text-text-400 text-sm ml-2">Generated successfully</span>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="primary" size="sm" onClick={handleDownloadReport}>
              Download HTML
            </Button>
            <Button variant="secondary" size="sm" onClick={() => setReportSuccess(null)}>
              Dismiss
            </Button>
          </div>
        </div>
      )}

      {/* Report Error Banner */}
      {reportError && (
        <div className="mb-6 px-4 py-3 rounded-lg border-l-4 border-loss-500 bg-loss-500/10 flex items-center justify-between">
          <span className="text-loss-500">{reportError}</span>
          <Button variant="secondary" size="sm" onClick={() => setReportError(null)}>
            Dismiss
          </Button>
        </div>
      )}

      {/* Trades Table */}
      <Card noPadding className="mb-6">
        <div className="px-5 py-4 border-b border-panel-500 flex items-center justify-between">
          <CardTitle>Trade Log</CardTitle>
          {loading && <Badge variant="warning">Loading...</Badge>}
        </div>

        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Symbol</TableHead>
                <TableHead>Side</TableHead>
                <TableHead align="right">Entry</TableHead>
                <TableHead align="right">Exit</TableHead>
                <TableHead align="right">Shares</TableHead>
                <TableHead align="right">P&L</TableHead>
                <TableHead align="right">Return</TableHead>
                <TableHead>Strategy</TableHead>
                <TableHead align="center">Chart</TableHead>
                <TableHead align="center">Details</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {trades.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={10} className="text-center py-8 text-text-400">
                    No trades found
                  </TableCell>
                </TableRow>
              ) : (
                trades.map((trade) => {
                  const returnPct =
                    trade.realized_pnl && trade.entry_price && trade.shares
                      ? (trade.realized_pnl / (trade.entry_price * trade.shares)) * 100
                      : 0;
                  const isExpanded = expandedTradeId === trade.id;

                  return (
                    <React.Fragment key={trade.id}>
                      <TableRow className={isExpanded ? 'bg-panel-600' : ''}>
                        <TableCell className="font-semibold text-text-900">
                          {trade.symbol}
                        </TableCell>
                        <TableCell>
                          <Badge
                            variant={trade.side === 'buy' ? 'success' : 'error'}
                            size="sm"
                          >
                            {trade.side.toUpperCase()}
                          </Badge>
                        </TableCell>
                        <TableCell align="right">
                          <div className="font-mono tabular-nums text-text-700">
                            {formatCurrency(trade.entry_price || 0)}
                          </div>
                          <div className="text-xs text-text-300">
                            {trade.entry_time ? formatDate(trade.entry_time) : 'N/A'}
                          </div>
                        </TableCell>
                        <TableCell align="right">
                          <div className="font-mono tabular-nums text-text-700">
                            {formatCurrency(trade.exit_price || 0)}
                          </div>
                          <div className="text-xs text-text-300">
                            {trade.exit_time ? formatDate(trade.exit_time) : 'N/A'}
                          </div>
                        </TableCell>
                        <TableCell align="right" className="font-mono tabular-nums">
                          {trade.shares || 0}
                        </TableCell>
                        <TableCell
                          align="right"
                          className={`font-mono tabular-nums font-semibold ${
                            (trade.realized_pnl || 0) > 0
                              ? 'text-profit-500'
                              : (trade.realized_pnl || 0) < 0
                              ? 'text-loss-500'
                              : 'text-text-400'
                          }`}
                        >
                          {formatCurrency(trade.realized_pnl || 0)}
                        </TableCell>
                        <TableCell
                          align="right"
                          className={`font-mono tabular-nums ${
                            returnPct > 0
                              ? 'text-profit-500'
                              : returnPct < 0
                              ? 'text-loss-500'
                              : 'text-text-400'
                          }`}
                        >
                          {returnPct >= 0 ? '+' : ''}{returnPct.toFixed(2)}%
                        </TableCell>
                        <TableCell className="text-text-400">
                          {trade.strategy || 'N/A'}
                        </TableCell>
                        <TableCell align="center">
                          {trade.id !== null && (
                            <Link
                              href={`/trades/${trade.id}`}
                              className="w-8 h-8 flex items-center justify-center rounded-lg bg-panel-500 hover:bg-accent-500 text-text-400 hover:text-white transition-colors"
                              title="View Chart"
                            >
                              <svg
                                xmlns="http://www.w3.org/2000/svg"
                                viewBox="0 0 20 20"
                                fill="currentColor"
                                className="w-4 h-4"
                              >
                                <path d="M12 9a1 1 0 01-1-1V3c0-.553.45-1.008.997-.93a7.004 7.004 0 015.933 5.933c.078.547-.378.997-.93.997h-5z" />
                                <path d="M8.003 4.07C8.55 3.992 9 4.447 9 5v5a1 1 0 001 1h5c.552 0 1.008.45.93.997A7.001 7.001 0 012 11a7.002 7.002 0 016.003-6.93z" />
                              </svg>
                            </Link>
                          )}
                        </TableCell>
                        <TableCell align="center">
                          <button
                            onClick={() => trade.id !== null && toggleTradeDetails(trade.id)}
                            className="w-8 h-8 flex items-center justify-center rounded-lg bg-panel-500 hover:bg-panel-400 text-text-400 hover:text-text-900"
                          >
                            {isExpanded ? '−' : '+'}
                          </button>
                        </TableCell>
                      </TableRow>

                      {/* Expanded Trade Details */}
                      {isExpanded && (
                        <TableRow className="bg-panel-600">
                          <TableCell colSpan={10}>
                            <div className="p-4 grid grid-cols-2 md:grid-cols-4 gap-4">
                              <div>
                                <span className="text-xs text-text-400 block mb-1">Signal Reason</span>
                                <span className="text-sm text-text-700">
                                  {trade.signal_reason || 'N/A'}
                                </span>
                              </div>
                              <div>
                                <span className="text-xs text-text-400 block mb-1">Market Regime</span>
                                <Badge variant="neutral" size="sm">
                                  {trade.regime || 'N/A'}
                                </Badge>
                              </div>
                              <div>
                                <span className="text-xs text-text-400 block mb-1">Stop Loss</span>
                                <span className="text-sm text-loss-500 font-mono tabular-nums">
                                  {trade.stop_loss ? formatCurrency(trade.stop_loss) : 'N/A'}
                                </span>
                              </div>
                              <div>
                                <span className="text-xs text-text-400 block mb-1">Take Profit</span>
                                <span className="text-sm text-profit-500 font-mono tabular-nums">
                                  {trade.take_profit ? formatCurrency(trade.take_profit) : 'N/A'}
                                </span>
                              </div>
                              {trade.id !== null && (
                                <div className="md:col-span-4 pt-3 border-t border-panel-500 mt-2">
                                  <Link href={`/trades/${trade.id}`}>
                                    <Button variant="primary" className="w-full sm:w-auto">
                                      View Chart
                                    </Button>
                                  </Link>
                                </div>
                              )}
                            </div>
                          </TableCell>
                        </TableRow>
                      )}
                    </React.Fragment>
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
          disabled={currentPage === 1 || loading}
        >
          Previous
        </Button>
        <span className="text-text-400 font-medium">
          Page {currentPage} of {totalPages}
        </span>
        <Button
          variant="secondary"
          onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
          disabled={currentPage >= totalPages || loading}
        >
          Next
        </Button>
      </div>
    </PageContainer>
  );
}
