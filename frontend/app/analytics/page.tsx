'use client';

import React, { useEffect, useRef, useState } from 'react';
import { createChart, IChartApi, ISeriesApi, CandlestickData, LineData, Time } from 'lightweight-charts';
import LoadingSpinner, { LoadingOverlay, SkeletonCard } from '../../components/LoadingSpinner';
import { WebSocketStatus } from '../../components/WebSocketStatus';
import { useWebSocketContext } from '../../contexts/WebSocketContext';

// Type definitions for chart data
interface CandleWithIndicators extends CandlestickData {
  kama?: number;
  atr?: number;
  rsi?: number;
  adx?: number;
  regime?: number;
}

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

export default function AnalyticsPage() {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const kamaSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const atrUpperBandRef = useRef<ISeriesApi<'Line'> | null>(null);
  const atrLowerBandRef = useRef<ISeriesApi<'Line'> | null>(null);

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

  // WebSocket context for real-time price updates
  const { connectionState, prices, getPrice, subscribe } = useWebSocketContext();

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 600,
      layout: {
        background: { color: '#1a1a1a' },
        textColor: '#d1d4dc',
      },
      grid: {
        vertLines: { color: '#2b2b2b' },
        horzLines: { color: '#2b2b2b' },
      },
      crosshair: {
        mode: 1,
      },
      rightPriceScale: {
        borderColor: '#2b2b2b',
      },
      timeScale: {
        borderColor: '#2b2b2b',
        timeVisible: true,
        secondsVisible: false,
      },
    });

    chartRef.current = chart;

    // Create candlestick series
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderVisible: false,
      wickUpColor: '#26a69a',
      wickDownColor: '#ef5350',
    });
    candlestickSeriesRef.current = candlestickSeries;

    // Create KAMA line series
    const kamaSeries = chart.addLineSeries({
      color: '#2962ff',
      lineWidth: 2,
      title: 'KAMA',
    });
    kamaSeriesRef.current = kamaSeries;

    // Create ATR upper band series
    const atrUpperBand = chart.addLineSeries({
      color: '#ff6b6b',
      lineWidth: 1,
      lineStyle: 2, // dashed
      title: 'ATR Upper',
    });
    atrUpperBandRef.current = atrUpperBand;

    // Create ATR lower band series
    const atrLowerBand = chart.addLineSeries({
      color: '#ff6b6b',
      lineWidth: 1,
      lineStyle: 2, // dashed
      title: 'ATR Lower',
    });
    atrLowerBandRef.current = atrLowerBand;

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, []);

  // Fetch chart data
  useEffect(() => {
    const fetchChartData = async () => {
      setLoading(true);
      try {
        // TODO: Replace with actual API call
        // const response = await fetch(`/api/chart-data?symbol=${symbol}&timeframe=${timeframe}`);
        // const data = await response.json();

        // Mock data for demonstration
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

        // Add signal markers
        if (candlestickSeriesRef.current) {
          candlestickSeriesRef.current.setMarkers(mockData.signals);
        }

        // Update indicators
        setIndicators({
          atr: mockData.latestIndicators.atr,
          rsi: mockData.latestIndicators.rsi,
          adx: mockData.latestIndicators.adx,
          regime: mockData.latestIndicators.regime,
        });

        // Fetch and update performance metrics
        // TODO: Replace with actual API call
        // const metricsResponse = await fetch('/api/performance-metrics');
        // const metricsData = await metricsResponse.json();
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
      }
    };

    fetchChartData();

    // Set up auto-refresh every 5 seconds
    const interval = setInterval(fetchChartData, 5000);
    return () => clearInterval(interval);
  }, [symbol, timeframe]);

  // Subscribe to WebSocket price updates for current symbol
  useEffect(() => {
    subscribe([symbol]);
  }, [symbol, subscribe]);

  // Update chart with real-time price data from WebSocket
  useEffect(() => {
    const priceData = getPrice(symbol);
    if (priceData && candlestickSeriesRef.current) {
      // Update the latest candle with real-time price
      // This is a simple implementation that updates the last candle
      // In production, you'd want more sophisticated logic to update or add candles
      const timestamp = Math.floor(new Date(priceData.timestamp).getTime() / 1000) as Time;

      // For demo purposes, we're just logging the price update
      // In a real implementation, you would update the chart data
      console.log('Real-time price update:', {
        symbol: priceData.symbol,
        price: priceData.price,
        timestamp: priceData.timestamp,
      });
    }
  }, [prices, symbol, getPrice]);

  // Show full-screen loading on initial load
  if (initialLoad) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <LoadingSpinner size="xl" message="Loading analytics dashboard..." />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6 page-container">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6 page-header">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold mb-2">Analytics Dashboard</h1>
              <p className="text-gray-400">Real-time market analysis and performance metrics</p>
            </div>
            <WebSocketStatus showText className="ml-4" />
          </div>
        </div>

        {/* Controls */}
        <div className="mb-6 flex gap-4 flex-col-tablet">
          <div>
            <label className="block text-sm font-medium mb-2">Symbol</label>
            <select
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              className="bg-gray-800 border border-gray-700 rounded px-4 py-2 text-white"
            >
              <option value="SPY">SPY</option>
              <option value="QQQ">QQQ</option>
              <option value="AAPL">AAPL</option>
              <option value="TSLA">TSLA</option>
              <option value="MSFT">MSFT</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">Timeframe</label>
            <select
              value={timeframe}
              onChange={(e) => setTimeframe(e.target.value)}
              className="bg-gray-800 border border-gray-700 rounded px-4 py-2 text-white"
            >
              <option value="1m">1 Minute</option>
              <option value="5m">5 Minutes</option>
              <option value="15m">15 Minutes</option>
              <option value="1h">1 Hour</option>
              <option value="4h">4 Hours</option>
              <option value="1d">1 Day</option>
            </select>
          </div>
        </div>

        {/* Chart Container */}
        <div className="bg-gray-800 rounded-lg p-4 mb-6 chart-container card relative">
          <LoadingOverlay isLoading={loading} message="Updating chart data...">
            <div ref={chartContainerRef} />
          </LoadingOverlay>
        </div>

        {/* Indicators Panel */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6 stats-grid">
          <div className="bg-gray-800 rounded-lg p-4">
            <div className="text-sm text-gray-400 mb-1">ATR (14)</div>
            <div className="text-2xl font-bold">{indicators.atr.toFixed(2)}</div>
            <div className="text-xs text-gray-500 mt-1">Average True Range</div>
          </div>
          <div className="bg-gray-800 rounded-lg p-4">
            <div className="text-sm text-gray-400 mb-1">RSI (14)</div>
            <div className={`text-2xl font-bold ${
              indicators.rsi > 70 ? 'text-red-400' :
              indicators.rsi < 30 ? 'text-green-400' :
              'text-white'
            }`}>
              {indicators.rsi.toFixed(1)}
            </div>
            <div className="text-xs text-gray-500 mt-1">
              {indicators.rsi > 70 ? 'Overbought' :
               indicators.rsi < 30 ? 'Oversold' :
               'Neutral'}
            </div>
          </div>
          <div className="bg-gray-800 rounded-lg p-4">
            <div className="text-sm text-gray-400 mb-1">ADX (14)</div>
            <div className={`text-2xl font-bold ${
              indicators.adx > 25 ? 'text-green-400' : 'text-yellow-400'
            }`}>
              {indicators.adx.toFixed(1)}
            </div>
            <div className="text-xs text-gray-500 mt-1">
              {indicators.adx > 25 ? 'Strong Trend' : 'Weak Trend'}
            </div>
          </div>
          <div className="bg-gray-800 rounded-lg p-4">
            <div className="text-sm text-gray-400 mb-1">Market Regime</div>
            <div className={`text-2xl font-bold ${
              indicators.regime === 'TRENDING' ? 'text-blue-400' :
              indicators.regime === 'MEAN_REVERSION' ? 'text-purple-400' :
              'text-gray-400'
            }`}>
              {indicators.regime}
            </div>
            <div className="text-xs text-gray-500 mt-1">Detected Mode</div>
          </div>
        </div>

        {/* Performance Metrics */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-bold mb-4">Performance Metrics</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
            <div>
              <div className="text-sm text-gray-400 mb-1">Total Return</div>
              <div className={`text-2xl font-bold ${
                metrics.totalReturn >= 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                ${metrics.totalReturn.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </div>
              <div className={`text-sm ${
                metrics.totalReturnPct >= 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                {metrics.totalReturnPct >= 0 ? '+' : ''}{metrics.totalReturnPct.toFixed(2)}%
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-400 mb-1">Sharpe Ratio</div>
              <div className={`text-2xl font-bold ${
                metrics.sharpeRatio > 1.0 ? 'text-green-400' :
                metrics.sharpeRatio > 0.5 ? 'text-yellow-400' :
                'text-red-400'
              }`}>
                {metrics.sharpeRatio.toFixed(2)}
              </div>
              <div className="text-xs text-gray-500 mt-1">
                {metrics.sharpeRatio > 1.0 ? 'Excellent' :
                 metrics.sharpeRatio > 0.5 ? 'Good' :
                 'Poor'}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-400 mb-1">Win Rate</div>
              <div className={`text-2xl font-bold ${
                metrics.winRate >= 55 ? 'text-green-400' :
                metrics.winRate >= 45 ? 'text-yellow-400' :
                'text-red-400'
              }`}>
                {metrics.winRate.toFixed(1)}%
              </div>
              <div className="text-xs text-gray-500 mt-1">
                {metrics.winRate >= 55 ? 'Strong' :
                 metrics.winRate >= 45 ? 'Average' :
                 'Weak'}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-400 mb-1">Max Drawdown</div>
              <div className={`text-2xl font-bold ${
                metrics.maxDrawdown < 15 ? 'text-green-400' :
                metrics.maxDrawdown < 25 ? 'text-yellow-400' :
                'text-red-400'
              }`}>
                {metrics.maxDrawdown.toFixed(1)}%
              </div>
              <div className="text-xs text-gray-500 mt-1">
                {metrics.maxDrawdown < 15 ? 'Low Risk' :
                 metrics.maxDrawdown < 25 ? 'Moderate Risk' :
                 'High Risk'}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-400 mb-1">Status</div>
              <div className="text-2xl font-bold text-green-400">Active</div>
              <div className="text-xs text-gray-500 mt-1">System Running</div>
            </div>
          </div>
        </div>

        {/* Legend */}
        <div className="mt-4 text-sm text-gray-400">
          <div className="flex gap-6">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
              <span>KAMA Line</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-red-400 rounded-full"></div>
              <span>ATR Bands</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-0 h-0 border-l-4 border-l-transparent border-r-4 border-r-transparent border-b-8 border-b-green-400"></div>
              <span>Buy Signal</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-0 h-0 border-l-4 border-l-transparent border-r-4 border-r-transparent border-t-8 border-t-red-400"></div>
              <span>Sell Signal</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Helper function to generate mock data
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

    // Random price movement
    basePrice += (Math.random() - 0.5) * 5;

    const open = basePrice;
    const close = basePrice + (Math.random() - 0.5) * 4;
    const high = Math.max(open, close) + Math.random() * 2;
    const low = Math.min(open, close) - Math.random() * 2;

    candles.push({
      time,
      open,
      high,
      low,
      close,
    });

    // KAMA line (smoother than price)
    const kamaValue = basePrice + Math.sin(i / 10) * 3;
    kama.push({ time, value: kamaValue });

    // ATR bands
    atrUpper.push({ time, value: kamaValue + volatility * 0.5 });
    atrLower.push({ time, value: kamaValue - volatility * 0.5 });

    // Add some buy/sell signals
    if (i % 15 === 0 && i > 0) {
      signals.push({
        time,
        position: 'belowBar',
        color: '#26a69a',
        shape: 'arrowUp',
        text: 'B',
      });
    } else if (i % 20 === 0 && i > 0) {
      signals.push({
        time,
        position: 'aboveBar',
        color: '#ef5350',
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
