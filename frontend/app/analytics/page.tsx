'use client';

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { createChart, IChartApi, ISeriesApi, CandlestickData, Time, CandlestickSeries } from 'lightweight-charts';
import { useWebSocketContext } from '../../contexts/WebSocketContext';
import { PageContainer, PageHeader, StatsGrid } from '../../components/layout';
import { Card, CardTitle, Badge } from '../../components/ui';
import { PLDisplay, SymbolSearch } from '../../components/trading';
import { apiClient, ChartCandleData, IntervalInfo } from '../../utils/api';

interface PerformanceMetrics {
  totalReturn: number;
  totalReturnPct: number;
  sharpeRatio: number;
  winRate: number;
  maxDrawdown: number;
}

export default function AnalyticsPage() {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);

  const [symbol, setSymbol] = useState<string>('SPY');
  const [symbolName, setSymbolName] = useState<string>('SPDR S&P 500 ETF Trust');
  const [selectedSymbol, setSelectedSymbol] = useState<string>('SPY'); // Only updates on selection
  const [interval, setInterval] = useState<string>('1d'); // Current interval (1m, 5m, 1h, 1d, etc.)
  const [availableIntervals, setAvailableIntervals] = useState<IntervalInfo[]>([]);
  const [candles, setCandles] = useState<ChartCandleData[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [chartReady, setChartReady] = useState(false);

  const [indicators, setIndicators] = useState({
    atr: 0,
    rsi: 50,
    adx: 25,
    regime: 'NEUTRAL'
  });

  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    totalReturn: 0,
    totalReturnPct: 0,
    sharpeRatio: 0,
    winRate: 0,
    maxDrawdown: 0
  });

  const { prices, getPrice, subscribe } = useWebSocketContext();

  // Fetch chart data from API
  const fetchChartData = useCallback(async () => {
    // Don't fetch if no symbol selected
    if (!selectedSymbol || selectedSymbol.trim() === '') {
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Get max_days from provider's interval info
      const intervalInfo = availableIntervals.find(i => i.name === interval);
      const maxDays = intervalInfo?.max_days;

      // Fetch chart data with provider's limit
      const data = await apiClient.getChartData(selectedSymbol, interval, 300, true, maxDays);
      setCandles(data);

      // Calculate simple indicators from the data
      if (data.length > 14) {
        const closes = data.map(c => c.close);
        const atr = calculateATR(data);
        const rsi = calculateRSI(closes);
        const adx = calculateADX(data);

        setIndicators({
          atr: atr,
          rsi: rsi,
          adx: adx,
          regime: adx > 25 ? 'TRENDING' : 'MEAN_REVERTING',
        });

        // Calculate performance metrics
        const firstPrice = data[0]?.close || 0;
        const lastPrice = data[data.length - 1]?.close || 0;
        const returnPct = firstPrice > 0 ? ((lastPrice - firstPrice) / firstPrice) * 100 : 0;

        setMetrics({
          totalReturn: lastPrice - firstPrice,
          totalReturnPct: returnPct,
          sharpeRatio: calculateSharpe(closes),
          winRate: calculateWinRate(data),
          maxDrawdown: calculateMaxDrawdown(closes),
        });
      }
    } catch (err) {
      let message = 'Failed to fetch chart data';
      if (err instanceof Error) {
        // Handle case where message might be an object
        message = typeof err.message === 'string' ? err.message : JSON.stringify(err.message);
      } else if (typeof err === 'object' && err !== null) {
        message = JSON.stringify(err);
      }
      setError(message);
      console.error('Chart data error:', err);
    } finally {
      setLoading(false);
    }
  }, [selectedSymbol, interval, availableIntervals]);

  // Initialize chart after component mounts
  useEffect(() => {
    // Small delay to ensure DOM is ready
    const timer = setTimeout(() => {
      const container = chartContainerRef.current;
      if (!container || chartRef.current) return;

      try {
        const chart = createChart(container, {
          width: container.clientWidth,
          height: container.clientHeight,
          layout: {
            background: { color: '#1C1C28' },
            textColor: '#CCCAD5',
          },
          grid: {
            vertLines: { color: '#21222F' },
            horzLines: { color: '#21222F' },
          },
          crosshair: {
            mode: 1,
          },
          rightPriceScale: {
            borderColor: '#21222F',
            autoScale: true,
            scaleMargins: {
              top: 0.1,
              bottom: 0.1,
            },
          },
          timeScale: {
            borderColor: '#21222F',
            timeVisible: true,
            secondsVisible: false,
          },
        });

        chartRef.current = chart;

        const candlestickSeries = chart.addSeries(CandlestickSeries, {
          upColor: '#22C55E',
          downColor: '#EF4444',
          borderVisible: false,
          wickUpColor: '#22C55E',
          wickDownColor: '#EF4444',
        });
        candlestickSeriesRef.current = candlestickSeries;

        setChartReady(true);
        console.log('Chart initialized successfully');
      } catch (err) {
        console.error('Error creating chart:', err);
      }
    }, 100);

    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      clearTimeout(timer);
      window.removeEventListener('resize', handleResize);
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
        candlestickSeriesRef.current = null;
        setChartReady(false);
      }
    };
  }, []);


  // Fetch available intervals on mount
  useEffect(() => {
    const fetchIntervals = async () => {
      try {
        const intervals = await apiClient.getChartIntervals();
        setAvailableIntervals(intervals);
      } catch (err) {
        console.error('Failed to fetch intervals:', err);
        // Fallback to common intervals if API fails
        setAvailableIntervals([
          { name: '5m', label: '5 Min', seconds: 300, max_days: 60, native: true },
          { name: '1h', label: '1 Hour', seconds: 3600, max_days: 730, native: true },
          { name: '1d', label: '1 Day', seconds: 86400, max_days: 1825, native: true },
        ]);
      }
    };
    fetchIntervals();
  }, []);

  // Fetch data when symbol/timeframe changes
  useEffect(() => {
    fetchChartData();
  }, [fetchChartData]);

  // Update chart when data changes
  useEffect(() => {
    if (!chartReady || !candlestickSeriesRef.current || candles.length === 0) return;

    const chartData: CandlestickData[] = candles.map(c => ({
      time: c.time as Time,
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
    }));

    candlestickSeriesRef.current.setData(chartData);
    chartRef.current?.timeScale().fitContent();
    console.log(`Chart updated with ${chartData.length} candles`);
  }, [candles, chartReady]);

  // Subscribe to WebSocket for real-time updates
  useEffect(() => {
    if (selectedSymbol) {
      subscribe([selectedSymbol]);
    }
  }, [selectedSymbol, subscribe]);

  // Handle real-time price updates
  useEffect(() => {
    const priceData = getPrice(selectedSymbol);
    if (priceData && candlestickSeriesRef.current && candles.length > 0) {
      // Update the last candle with real-time data
      const lastCandle = candles[candles.length - 1];
      if (lastCandle) {
        candlestickSeriesRef.current.update({
          time: lastCandle.time as Time,
          open: lastCandle.open,
          high: Math.max(lastCandle.high, priceData.price),
          low: Math.min(lastCandle.low, priceData.price),
          close: priceData.price,
        });
      }
    }
  }, [prices, selectedSymbol, getPrice, candles]);

  return (
    <PageContainer>
      {/* Header with search inline on left */}
      <div className="flex items-center gap-6 mb-6">
        <div className="w-64">
          <SymbolSearch
            value={symbol}
            onChange={(sym, name) => {
              setSymbol(sym);
              // Only fetch data when a symbol is SELECTED from dropdown (name is provided)
              if (name) {
                setSymbolName(name);
                setSelectedSymbol(sym); // This triggers data fetch
              }
            }}
            placeholder="Search stocks..."
          />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-text-900">Analytics Dashboard</h1>
          <p className="text-text-400 text-sm">Real-time market analysis and performance metrics</p>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <Card variant="highlighted" className="mb-6 border-l-4 border-loss-500">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-loss-500 font-medium">Error loading chart data</p>
              <p className="text-text-400 text-sm">{error}</p>
            </div>
            <button
              onClick={fetchChartData}
              className="px-4 py-2 bg-panel-400 hover:bg-panel-300 rounded text-text-700"
            >
              Retry
            </button>
          </div>
        </Card>
      )}

      {/* Chart */}
      <Card noPadding className="mb-6">
        <div className="p-4 border-b border-panel-500 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <CardTitle>
              {selectedSymbol}{symbolName && ` - ${symbolName}`}
            </CardTitle>
            {candles.length > 0 && (
              <span className="text-sm text-text-300">
                ({candles.length} bars)
              </span>
            )}
          </div>

          {/* Interval selector pills */}
          <div className="flex bg-panel-400 rounded-lg p-1 gap-0.5">
            {availableIntervals.map((option) => (
              <button
                key={option.name}
                onClick={() => setInterval(option.name)}
                className={`px-3 py-1.5 text-sm font-medium rounded-md transition-all ${
                  interval === option.name
                    ? 'bg-panel-700 text-text-900 shadow-sm'
                    : 'text-text-400 hover:text-text-700 hover:bg-panel-500'
                }`}
                title={`${option.label} - up to ${option.max_days} days of history`}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>
        <div className="relative">
          <div
            ref={chartContainerRef}
            className="bg-panel-700 w-full h-[50vh] min-h-[350px]"
          />
          {/* Loading overlay - top right corner */}
          {loading && (
            <div className="absolute top-3 right-3 z-10">
              <Badge variant="warning">Loading...</Badge>
            </div>
          )}
        </div>
      </Card>

      {/* Indicators */}
      <h2 className="text-xl font-semibold text-text-900 mb-4">Technical Indicators</h2>
      <StatsGrid columns={4} className="mb-8">
        <Card>
          <span className="text-sm text-text-400 block mb-1">ATR (14)</span>
          <span className="text-2xl font-bold text-text-900 font-mono tabular-nums">
            {indicators.atr.toFixed(2)}
          </span>
          <span className="text-xs text-text-300 block mt-1">Average True Range</span>
        </Card>

        <Card>
          <span className="text-sm text-text-400 block mb-1">RSI (14)</span>
          <span className={`text-2xl font-bold font-mono tabular-nums ${
            indicators.rsi > 70 ? 'text-loss-500' :
            indicators.rsi < 30 ? 'text-profit-500' :
            'text-text-900'
          }`}>
            {indicators.rsi.toFixed(1)}
          </span>
          <span className="text-xs text-text-300 block mt-1">
            {indicators.rsi > 70 ? 'Overbought' :
             indicators.rsi < 30 ? 'Oversold' :
             'Neutral'}
          </span>
        </Card>

        <Card>
          <span className="text-sm text-text-400 block mb-1">ADX (14)</span>
          <span className={`text-2xl font-bold font-mono tabular-nums ${
            indicators.adx > 25 ? 'text-profit-500' : 'text-warning-500'
          }`}>
            {indicators.adx.toFixed(1)}
          </span>
          <span className="text-xs text-text-300 block mt-1">
            {indicators.adx > 25 ? 'Strong Trend' : 'Weak Trend'}
          </span>
        </Card>

        <Card>
          <span className="text-sm text-text-400 block mb-1">Market Regime</span>
          <Badge
            variant={indicators.regime === 'TRENDING' ? 'info' : 'neutral'}
            size="md"
            className="mt-1"
          >
            {indicators.regime}
          </Badge>
          <span className="text-xs text-text-300 block mt-2">Detected Mode</span>
        </Card>
      </StatsGrid>

      {/* Performance Metrics */}
      <h2 className="text-xl font-semibold text-text-900 mb-4">Performance Metrics</h2>
      <Card>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-6">
          <div>
            <span className="text-sm text-text-400 block mb-1">Price Change</span>
            <PLDisplay value={metrics.totalReturn} percent={metrics.totalReturnPct} size="lg" />
          </div>
          <div>
            <span className="text-sm text-text-400 block mb-1">Sharpe Ratio</span>
            <span className={`text-2xl font-bold font-mono tabular-nums ${
              metrics.sharpeRatio > 1.0 ? 'text-profit-500' :
              metrics.sharpeRatio > 0.5 ? 'text-warning-500' :
              'text-loss-500'
            }`}>
              {metrics.sharpeRatio.toFixed(2)}
            </span>
            <span className="text-xs text-text-300 block mt-1">
              {metrics.sharpeRatio > 1.0 ? 'Excellent' :
               metrics.sharpeRatio > 0.5 ? 'Good' : 'Poor'}
            </span>
          </div>
          <div>
            <span className="text-sm text-text-400 block mb-1">Win Rate</span>
            <span className={`text-2xl font-bold font-mono tabular-nums ${
              metrics.winRate >= 55 ? 'text-profit-500' :
              metrics.winRate >= 45 ? 'text-warning-500' :
              'text-loss-500'
            }`}>
              {metrics.winRate.toFixed(1)}%
            </span>
          </div>
          <div>
            <span className="text-sm text-text-400 block mb-1">Max Drawdown</span>
            <span className={`text-2xl font-bold font-mono tabular-nums ${
              metrics.maxDrawdown < 15 ? 'text-profit-500' :
              metrics.maxDrawdown < 25 ? 'text-warning-500' :
              'text-loss-500'
            }`}>
              {metrics.maxDrawdown.toFixed(1)}%
            </span>
          </div>
          <div>
            <span className="text-sm text-text-400 block mb-1">Status</span>
            <Badge variant={error ? 'error' : 'success'} size="md">
              {error ? 'ERROR' : 'ACTIVE'}
            </Badge>
          </div>
        </div>
      </Card>

      {/* Legend */}
      <div className="mt-6 flex flex-wrap gap-6 text-sm text-text-400">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-profit-500 rounded-sm" />
          <span>Bullish Candle</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-loss-500 rounded-sm" />
          <span>Bearish Candle</span>
        </div>
      </div>
    </PageContainer>
  );
}

// Simple indicator calculations
function calculateATR(candles: ChartCandleData[], period: number = 14): number {
  if (candles.length < period + 1) return 0;

  let atrSum = 0;
  for (let i = candles.length - period; i < candles.length; i++) {
    const high = candles[i].high;
    const low = candles[i].low;
    const prevClose = candles[i - 1]?.close || candles[i].open;
    const tr = Math.max(high - low, Math.abs(high - prevClose), Math.abs(low - prevClose));
    atrSum += tr;
  }
  return atrSum / period;
}

function calculateRSI(closes: number[], period: number = 14): number {
  if (closes.length < period + 1) return 50;

  let gains = 0;
  let losses = 0;

  for (let i = closes.length - period; i < closes.length; i++) {
    const change = closes[i] - closes[i - 1];
    if (change > 0) gains += change;
    else losses -= change;
  }

  if (losses === 0) return 100;
  const rs = gains / losses;
  return 100 - (100 / (1 + rs));
}

function calculateADX(candles: ChartCandleData[], period: number = 14): number {
  if (candles.length < period * 2) return 25;

  // Simplified ADX calculation
  let sumDM = 0;
  for (let i = candles.length - period; i < candles.length; i++) {
    const high = candles[i].high;
    const low = candles[i].low;
    const prevHigh = candles[i - 1]?.high || high;
    const prevLow = candles[i - 1]?.low || low;

    const plusDM = high - prevHigh > prevLow - low ? Math.max(high - prevHigh, 0) : 0;
    const minusDM = prevLow - low > high - prevHigh ? Math.max(prevLow - low, 0) : 0;
    sumDM += Math.abs(plusDM - minusDM);
  }

  return (sumDM / period) * 10; // Scaled approximation
}

function calculateSharpe(closes: number[]): number {
  if (closes.length < 2) return 0;

  const returns = [];
  for (let i = 1; i < closes.length; i++) {
    returns.push((closes[i] - closes[i - 1]) / closes[i - 1]);
  }

  const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance = returns.reduce((a, b) => a + Math.pow(b - avgReturn, 2), 0) / returns.length;
  const stdDev = Math.sqrt(variance);

  if (stdDev === 0) return 0;
  return (avgReturn / stdDev) * Math.sqrt(252); // Annualized
}

function calculateWinRate(candles: ChartCandleData[]): number {
  if (candles.length < 2) return 50;

  let wins = 0;
  for (let i = 1; i < candles.length; i++) {
    if (candles[i].close > candles[i - 1].close) wins++;
  }
  return (wins / (candles.length - 1)) * 100;
}

function calculateMaxDrawdown(closes: number[]): number {
  if (closes.length < 2) return 0;

  let maxDrawdown = 0;
  let peak = closes[0];

  for (const close of closes) {
    if (close > peak) peak = close;
    const drawdown = ((peak - close) / peak) * 100;
    if (drawdown > maxDrawdown) maxDrawdown = drawdown;
  }

  return maxDrawdown;
}
