'use client';

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { createChart, IChartApi, ISeriesApi, CandlestickData, LineData, Time, CandlestickSeries, LineSeries, createSeriesMarkers, ISeriesMarkersPluginApi, SeriesMarker } from 'lightweight-charts';
import { useWebSocketContext } from '../../contexts/WebSocketContext';
import { PageContainer, PageHeader, StatsGrid } from '../../components/layout';
import { Card, CardTitle, Select, Badge, Skeleton } from '../../components/ui';
import { PLDisplay } from '../../components/trading';
import { formatCurrency, formatPercent } from '../../lib/utils';

interface SignalMarker {
  time: Time;
  position: 'aboveBar' | 'belowBar';
  color: string;
  shape: 'arrowUp' | 'arrowDown';
  text: string;
}

interface PerformanceMetrics {
  totalReturn: number;
  totalReturnPct: number;
  sharpeRatio: number;
  winRate: number;
  maxDrawdown: number;
}

const symbolOptions = [
  { value: 'SPY', label: 'SPY' },
  { value: 'QQQ', label: 'QQQ' },
  { value: 'AAPL', label: 'AAPL' },
  { value: 'TSLA', label: 'TSLA' },
  { value: 'MSFT', label: 'MSFT' },
];

const timeframeOptions = [
  { value: '1m', label: '1 Minute' },
  { value: '5m', label: '5 Minutes' },
  { value: '15m', label: '15 Minutes' },
  { value: '1h', label: '1 Hour' },
  { value: '4h', label: '4 Hours' },
  { value: '1d', label: '1 Day' },
];

export default function AnalyticsPage() {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const kamaSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const atrUpperBandRef = useRef<ISeriesApi<'Line'> | null>(null);
  const atrLowerBandRef = useRef<ISeriesApi<'Line'> | null>(null);
  const markersRef = useRef<ISeriesMarkersPluginApi<Time> | null>(null);

  const [symbol, setSymbol] = useState<string>('SPY');
  const [timeframe, setTimeframe] = useState<string>('1h');
  const [indicators, setIndicators] = useState({
    atr: 2.5,
    rsi: 45.2,
    adx: 28.5,
    regime: 'TRENDING'
  });
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    totalReturn: 0,
    totalReturnPct: 0,
    sharpeRatio: 0,
    winRate: 0,
    maxDrawdown: 0
  });
  const [loading, setLoading] = useState<boolean>(true);
  const [initialLoad, setInitialLoad] = useState<boolean>(true);

  // Track if a fetch is already in progress to prevent duplicate calls
  const isFetchingRef = useRef(false);

  const { prices, getPrice, subscribe } = useWebSocketContext();

  // Initialize chart with design system colors
  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 500,
      layout: {
        background: { color: '#1C1C28' }, // panel-700
        textColor: '#CCCAD5', // text-400
      },
      grid: {
        vertLines: { color: '#21222F' }, // panel-500
        horzLines: { color: '#21222F' },
      },
      crosshair: {
        mode: 1,
      },
      rightPriceScale: {
        borderColor: '#21222F',
      },
      timeScale: {
        borderColor: '#21222F',
        timeVisible: true,
        secondsVisible: false,
      },
    });

    chartRef.current = chart;

    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#22C55E', // profit-500
      downColor: '#EF4444', // loss-500
      borderVisible: false,
      wickUpColor: '#22C55E',
      wickDownColor: '#EF4444',
    });
    candlestickSeriesRef.current = candlestickSeries;

    const markers = createSeriesMarkers(candlestickSeries, []);
    markersRef.current = markers;

    const kamaSeries = chart.addSeries(LineSeries, {
      color: '#A549FC', // accent-500
      lineWidth: 2,
      title: 'KAMA',
    });
    kamaSeriesRef.current = kamaSeries;

    const atrUpperBand = chart.addSeries(LineSeries, {
      color: '#3E7AEE', // blue-500
      lineWidth: 1,
      lineStyle: 2,
      title: 'ATR Upper',
    });
    atrUpperBandRef.current = atrUpperBand;

    const atrLowerBand = chart.addSeries(LineSeries, {
      color: '#3E7AEE',
      lineWidth: 1,
      lineStyle: 2,
      title: 'ATR Lower',
    });
    atrLowerBandRef.current = atrLowerBand;

    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, []);

  // Fetch chart data
  const fetchChartData = useCallback(async () => {
    // Prevent duplicate concurrent fetches
    if (isFetchingRef.current) return;
    isFetchingRef.current = true;

    setLoading(true);
    try {
      const mockData = generateMockData(100);

      if (candlestickSeriesRef.current) {
        candlestickSeriesRef.current.setData(mockData.candles);
      }
      if (kamaSeriesRef.current) {
        kamaSeriesRef.current.setData(mockData.kama);
      }
      if (atrUpperBandRef.current) {
        atrUpperBandRef.current.setData(mockData.atrUpper);
      }
      if (atrLowerBandRef.current) {
        atrLowerBandRef.current.setData(mockData.atrLower);
      }
      if (markersRef.current) {
        markersRef.current.setMarkers(mockData.signals as SeriesMarker<Time>[]);
      }

      setIndicators({
        atr: mockData.latestIndicators.atr,
        rsi: mockData.latestIndicators.rsi,
        adx: mockData.latestIndicators.adx,
        regime: mockData.latestIndicators.regime,
      });

      setMetrics({
        totalReturn: 15420.50,
        totalReturnPct: 15.42,
        sharpeRatio: 1.85,
        winRate: 58.3,
        maxDrawdown: 12.5,
      });
    } catch (error) {
      console.error('Error fetching chart data:', error);
    } finally {
      setLoading(false);
      setInitialLoad(false);
      isFetchingRef.current = false;
    }
  }, []);

  useEffect(() => {
    // Reset fetch flag when symbol or timeframe changes
    isFetchingRef.current = false;
    fetchChartData();
    const interval = setInterval(fetchChartData, 5000);
    return () => clearInterval(interval);
  }, [symbol, timeframe, fetchChartData]);

  useEffect(() => {
    subscribe([symbol]);
  }, [symbol, subscribe]);

  useEffect(() => {
    const priceData = getPrice(symbol);
    if (priceData && candlestickSeriesRef.current) {
      console.log('Real-time price update:', priceData);
    }
  }, [prices, symbol, getPrice]);

  if (initialLoad) {
    return (
      <PageContainer>
        <PageHeader title="Analytics Dashboard" subtitle="Loading..." />
        <Card className="mb-6">
          <Skeleton height={500} className="rounded-lg" />
        </Card>
        <StatsGrid columns={4}>
          {Array.from({ length: 4 }).map((_, i) => (
            <Card key={i}>
              <Skeleton variant="text" className="mb-2" />
              <Skeleton variant="title" />
            </Card>
          ))}
        </StatsGrid>
      </PageContainer>
    );
  }

  return (
    <PageContainer>
      <PageHeader
        title="Analytics Dashboard"
        subtitle="Real-time market analysis and performance metrics"
      />

      {/* Controls */}
      <div className="flex flex-wrap gap-4 mb-6">
        <div className="w-40">
          <Select
            label="Symbol"
            options={symbolOptions}
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
          />
        </div>
        <div className="w-40">
          <Select
            label="Timeframe"
            options={timeframeOptions}
            value={timeframe}
            onChange={(e) => setTimeframe(e.target.value)}
          />
        </div>
      </div>

      {/* Chart */}
      <Card noPadding className="mb-6 overflow-hidden">
        <div className="p-4 border-b border-panel-500 flex items-center justify-between">
          <CardTitle>{symbol} - {timeframeOptions.find(t => t.value === timeframe)?.label}</CardTitle>
          {loading && <Badge variant="warning">Updating...</Badge>}
        </div>
        <div ref={chartContainerRef} className="bg-panel-700" />
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
            <span className="text-sm text-text-400 block mb-1">Total Return</span>
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
            <Badge variant="success" size="md">ACTIVE</Badge>
          </div>
        </div>
      </Card>

      {/* Legend */}
      <div className="mt-6 flex flex-wrap gap-6 text-sm text-text-400">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-accent-500 rounded-full" />
          <span>KAMA Line</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-blue-500 rounded-full" />
          <span>ATR Bands</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-0 h-0 border-l-4 border-l-transparent border-r-4 border-r-transparent border-b-8 border-b-profit-500" />
          <span>Buy Signal</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-0 h-0 border-l-4 border-l-transparent border-r-4 border-r-transparent border-t-8 border-t-loss-500" />
          <span>Sell Signal</span>
        </div>
      </div>
    </PageContainer>
  );
}

function generateMockData(numCandles: number) {
  const candles: CandlestickData[] = [];
  const kama: LineData[] = [];
  const atrUpper: LineData[] = [];
  const atrLower: LineData[] = [];
  const signals: SignalMarker[] = [];

  let basePrice = 450;
  let basetime = Math.floor(Date.now() / 1000) - (numCandles * 3600);

  for (let i = 0; i < numCandles; i++) {
    const time = (basetime + i * 3600) as Time;
    const volatility = 2 + Math.random() * 3;

    basePrice += (Math.random() - 0.5) * 5;

    const open = basePrice;
    const close = basePrice + (Math.random() - 0.5) * 4;
    const high = Math.max(open, close) + Math.random() * 2;
    const low = Math.min(open, close) - Math.random() * 2;

    candles.push({ time, open, high, low, close });

    const kamaValue = basePrice + Math.sin(i / 10) * 3;
    kama.push({ time, value: kamaValue });
    atrUpper.push({ time, value: kamaValue + volatility * 0.5 });
    atrLower.push({ time, value: kamaValue - volatility * 0.5 });

    if (i % 15 === 0 && i > 0) {
      signals.push({
        time,
        position: 'belowBar',
        color: '#22C55E',
        shape: 'arrowUp',
        text: 'B',
      });
    } else if (i % 20 === 0 && i > 0) {
      signals.push({
        time,
        position: 'aboveBar',
        color: '#EF4444',
        shape: 'arrowDown',
        text: 'S',
      });
    }
  }

  return {
    candles,
    kama,
    atrUpper,
    atrLower,
    signals,
    latestIndicators: {
      atr: 2.5,
      rsi: 45.2,
      adx: 28.5,
      regime: 'TRENDING',
    },
  };
}
