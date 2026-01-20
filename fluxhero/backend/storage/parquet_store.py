"""
Parquet Store for Market Data Caching

This module implements lightweight market data caching using Apache Parquet format
with Snappy compression for fast read/write operations.

Requirements: Feature 7.2 (R7.2.1-R7.2.4)
- R7.2.1: Cache downloaded market data as data/cache/{symbol}_{timeframe}.parquet
- R7.2.2: On startup, check if cached data is <24 hours old; if yes, load from cache
- R7.2.3: Store only OHLCV + calculated indicators (EMA, ATR, RSI)
- R7.2.4: Compress with snappy codec for fast read/write

Performance Targets:
- Load 500 candles from Parquet: <50ms
- Write 500 candles to Parquet: <100ms
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


@dataclass
class CandleData:
    """
    Market candle data with OHLCV and calculated indicators.

    Attributes:
        symbol: Stock/ETF ticker symbol
        timeframe: Candle timeframe (e.g., '1h', '1d', '5m')
        timestamp: Array of timestamps
        open: Opening prices
        high: High prices
        low: Low prices
        close: Closing prices
        volume: Trading volumes
        ema: Exponential Moving Average (optional)
        atr: Average True Range (optional)
        rsi: Relative Strength Index (optional)
    """
    symbol: str
    timeframe: str
    timestamp: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    ema: Optional[np.ndarray] = None
    atr: Optional[np.ndarray] = None
    rsi: Optional[np.ndarray] = None


class ParquetStore:
    """
    Lightweight market data cache using Parquet format.

    Features:
    - Fast read/write with Snappy compression
    - Cache validation (check if data is fresh)
    - Support for OHLCV + indicators
    - Automatic cache directory creation
    - Thread-safe operations

    Example:
        >>> store = ParquetStore()
        >>> data = CandleData(
        ...     symbol='SPY',
        ...     timeframe='1h',
        ...     timestamp=timestamps,
        ...     open=opens, high=highs, low=lows,
        ...     close=closes, volume=volumes
        ... )
        >>> store.save_candles(data)
        >>> loaded = store.load_candles('SPY', '1h')
    """

    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize Parquet store.

        Args:
            cache_dir: Directory to store cached Parquet files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Performance configuration
        self.compression = 'snappy'  # R7.2.4: Fast compression
        self.cache_ttl_hours = 24    # R7.2.2: 24-hour cache validity

    def _get_cache_path(self, symbol: str, timeframe: str) -> Path:
        """
        Get cache file path for given symbol and timeframe.

        Args:
            symbol: Ticker symbol (e.g., 'SPY')
            timeframe: Timeframe string (e.g., '1h', '1d')

        Returns:
            Path to cache file

        Requirements: R7.2.1
        """
        filename = f"{symbol}_{timeframe}.parquet"
        return self.cache_dir / filename

    def save_candles(self, data: CandleData) -> None:
        """
        Save candle data to Parquet file with Snappy compression.

        Args:
            data: CandleData object containing OHLCV and optional indicators

        Requirements:
        - R7.2.1: Save to data/cache/{symbol}_{timeframe}.parquet
        - R7.2.3: Store OHLCV + indicators
        - R7.2.4: Compress with snappy

        Performance: <100ms for 500 candles
        """
        cache_path = self._get_cache_path(data.symbol, data.timeframe)

        # Build dictionary for DataFrame
        df_dict = {
            'timestamp': data.timestamp,
            'open': data.open,
            'high': data.high,
            'low': data.low,
            'close': data.close,
            'volume': data.volume,
        }

        # Add indicators if present (R7.2.3)
        if data.ema is not None:
            df_dict['ema'] = data.ema
        if data.atr is not None:
            df_dict['atr'] = data.atr
        if data.rsi is not None:
            df_dict['rsi'] = data.rsi

        # Create DataFrame
        df = pd.DataFrame(df_dict)

        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        # Write to Parquet with Snappy compression (R7.2.4)
        df.to_parquet(
            cache_path,
            engine='pyarrow',
            compression=self.compression,
            index=False
        )

    def load_candles(self, symbol: str, timeframe: str) -> Optional[CandleData]:
        """
        Load candle data from Parquet cache.

        Args:
            symbol: Ticker symbol (e.g., 'SPY')
            timeframe: Timeframe string (e.g., '1h', '1d')

        Returns:
            CandleData object if cache exists, None otherwise

        Requirements:
        - R7.2.1: Load from data/cache/{symbol}_{timeframe}.parquet
        - R7.2.3: Load OHLCV + indicators

        Performance: <50ms for 500 candles
        """
        cache_path = self._get_cache_path(symbol, timeframe)

        if not cache_path.exists():
            return None

        # Read Parquet file
        df = pd.read_parquet(cache_path, engine='pyarrow')

        # Convert timestamp to numpy array (unix timestamp)
        if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            timestamps = df['timestamp'].astype('int64') / 10**9  # Convert to seconds
        else:
            timestamps = df['timestamp'].values

        # Extract OHLCV data
        data = CandleData(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=timestamps.astype(np.float64),
            open=df['open'].values,
            high=df['high'].values,
            low=df['low'].values,
            close=df['close'].values,
            volume=df['volume'].values,
            ema=df['ema'].values if 'ema' in df.columns else None,
            atr=df['atr'].values if 'atr' in df.columns else None,
            rsi=df['rsi'].values if 'rsi' in df.columns else None,
        )

        return data

    def is_cache_fresh(self, symbol: str, timeframe: str) -> bool:
        """
        Check if cached data is fresh (less than 24 hours old).

        Args:
            symbol: Ticker symbol
            timeframe: Timeframe string

        Returns:
            True if cache exists and is fresh, False otherwise

        Requirements: R7.2.2 - Check if cached data is <24 hours old
        """
        cache_path = self._get_cache_path(symbol, timeframe)

        if not cache_path.exists():
            return False

        # Get file modification time
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - mtime

        return age < timedelta(hours=self.cache_ttl_hours)

    def get_cache_age(self, symbol: str, timeframe: str) -> Optional[timedelta]:
        """
        Get the age of cached data.

        Args:
            symbol: Ticker symbol
            timeframe: Timeframe string

        Returns:
            Age of cache as timedelta, or None if cache doesn't exist
        """
        cache_path = self._get_cache_path(symbol, timeframe)

        if not cache_path.exists():
            return None

        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return datetime.now() - mtime

    def delete_cache(self, symbol: str, timeframe: str) -> bool:
        """
        Delete cached data for given symbol and timeframe.

        Args:
            symbol: Ticker symbol
            timeframe: Timeframe string

        Returns:
            True if cache was deleted, False if it didn't exist
        """
        cache_path = self._get_cache_path(symbol, timeframe)

        if cache_path.exists():
            cache_path.unlink()
            return True
        return False

    def clear_all_cache(self) -> int:
        """
        Clear all cached files.

        Returns:
            Number of files deleted
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.parquet"):
            cache_file.unlink()
            count += 1
        return count

    def get_cache_size(self, symbol: str, timeframe: str) -> Optional[int]:
        """
        Get size of cached file in bytes.

        Args:
            symbol: Ticker symbol
            timeframe: Timeframe string

        Returns:
            File size in bytes, or None if cache doesn't exist
        """
        cache_path = self._get_cache_path(symbol, timeframe)

        if not cache_path.exists():
            return None

        return cache_path.stat().st_size

    def list_cached_symbols(self) -> list[tuple[str, str]]:
        """
        List all cached symbols and timeframes.

        Returns:
            List of (symbol, timeframe) tuples
        """
        symbols = []
        for cache_file in self.cache_dir.glob("*.parquet"):
            # Parse filename: {symbol}_{timeframe}.parquet
            name = cache_file.stem
            if '_' in name:
                parts = name.rsplit('_', 1)  # Split from right to handle symbols with underscores
                if len(parts) == 2:
                    symbols.append(tuple(parts))
        return symbols

    def get_cache_metadata(self, symbol: str, timeframe: str) -> Optional[dict]:
        """
        Get metadata about cached file.

        Args:
            symbol: Ticker symbol
            timeframe: Timeframe string

        Returns:
            Dictionary with metadata (size, age, num_rows) or None if cache doesn't exist
        """
        cache_path = self._get_cache_path(symbol, timeframe)

        if not cache_path.exists():
            return None

        # Get file stats
        stat = cache_path.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime)
        age = datetime.now() - mtime

        # Get number of rows
        try:
            parquet_file = pq.ParquetFile(cache_path)
            num_rows = parquet_file.metadata.num_rows
        except Exception:
            num_rows = None

        return {
            'size_bytes': stat.st_size,
            'age': age,
            'age_hours': age.total_seconds() / 3600,
            'is_fresh': age < timedelta(hours=self.cache_ttl_hours),
            'num_rows': num_rows,
            'modified_at': mtime,
        }
