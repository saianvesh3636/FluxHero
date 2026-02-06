"""
Local CSV Data Provider for FluxHero.

Reads historical data from local CSV files in the test_data directory.
Useful for development, testing, and when Yahoo Finance is unavailable.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from backend.data.provider import (
    DataProvider,
    HistoricalData,
    IntervalInfo,
    ProviderType,
    SymbolInfo,
    SymbolNotFoundError,
    DateRangeError,
    InsufficientDataError,
    register_provider,
)

logger = logging.getLogger(__name__)


@register_provider(ProviderType.YAHOO_FINANCE)  # Register as default provider
class LocalCSVProvider(DataProvider):
    """Data provider that reads from local CSV files."""

    PROVIDER_NAME = "local_csv"

    @property
    def name(self) -> str:
        """Provider name for logging."""
        return self.PROVIDER_NAME

    @property
    def supported_intervals(self) -> dict:
        """Return supported intervals (only daily for CSV files)."""
        return {
            "1d": IntervalInfo(
                name="1d",
                seconds=86400,
                native=True,
                aggregation_base=None,
            ),
        }

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize local CSV provider.

        Args:
            data_dir: Directory containing CSV files. Defaults to backend/test_data
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "test_data"
        self.data_dir = data_dir
        self._symbol_cache: dict[str, pd.DataFrame] = {}
        logger.info(f"LocalCSVProvider initialized with data_dir: {data_dir}")

    def _get_csv_path(self, symbol: str) -> Path:
        """Get path to CSV file for symbol."""
        return self.data_dir / f"{symbol.lower()}_daily.csv"

    def _load_symbol_data(self, symbol: str) -> pd.DataFrame:
        """Load and cache symbol data from CSV."""
        symbol = symbol.upper()
        if symbol in self._symbol_cache:
            return self._symbol_cache[symbol]

        csv_path = self._get_csv_path(symbol)
        if not csv_path.exists():
            raise SymbolNotFoundError(
                symbol=symbol,
                provider=self.PROVIDER_NAME,
                message=f"No local data file found for '{symbol}'. "
                f"Expected: {csv_path}"
            )

        try:
            df = pd.read_csv(csv_path)
            # Parse the Price column as datetime, converting to UTC first then removing tz
            df['date'] = pd.to_datetime(df['Price'], utc=True).dt.tz_localize(None)
            df = df.sort_values('date').reset_index(drop=True)
            self._symbol_cache[symbol] = df
            logger.info(f"Loaded {len(df)} rows for {symbol} from {csv_path}")
            return df
        except Exception as e:
            raise SymbolNotFoundError(
                symbol=symbol,
                provider=self.PROVIDER_NAME,
                message=f"Failed to load data for '{symbol}': {e}"
            )

    async def validate_symbol(self, symbol: str) -> SymbolInfo:
        """Check if symbol has local data."""
        symbol = symbol.upper()
        csv_path = self._get_csv_path(symbol)
        is_valid = csv_path.exists()

        return SymbolInfo(
            symbol=symbol,
            name=symbol,
            exchange="LOCAL",
            currency="USD",
            type="stock",
            is_valid=is_valid,
        )

    async def fetch_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> HistoricalData:
        """Fetch historical data from local CSV file."""
        symbol = symbol.upper()

        # Parse dates
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError as e:
            raise DateRangeError(f"Invalid date format: {e}")

        # Load data
        df = self._load_symbol_data(symbol)

        # Filter by date range
        mask = (df['date'] >= start_dt) & (df['date'] <= end_dt)
        filtered = df[mask].copy()

        if len(filtered) == 0:
            available_start = df['date'].min().strftime("%Y-%m-%d")
            available_end = df['date'].max().strftime("%Y-%m-%d")
            raise DateRangeError(
                f"No data for '{symbol}' between {start_date} and {end_date}. "
                f"Available range: {available_start} to {available_end}"
            )

        if len(filtered) < 20:
            raise InsufficientDataError(
                symbol=symbol,
                bars_found=len(filtered),
                bars_required=20,
            )

        # Build numpy arrays in OHLCV format
        bars = np.column_stack([
            filtered['open'].values,
            filtered['high'].values,
            filtered['low'].values,
            filtered['close'].values,
            filtered['volume'].values,
        ]).astype(np.float64)

        timestamps = (filtered['date'].astype(np.int64) // 10**9).values.astype(np.float64)
        dates = filtered['date'].dt.strftime("%Y-%m-%d").tolist()

        logger.info(f"Returning {len(bars)} bars for {symbol} ({dates[0]} to {dates[-1]})")

        return HistoricalData(
            symbol=symbol,
            bars=bars,
            timestamps=timestamps,
            dates=dates,
            provider=self.PROVIDER_NAME,
        )


    async def search_symbols(self, query: str, limit: int = 10) -> list[SymbolInfo]:
        """Search for symbols in local data directory."""
        query = query.upper()
        results = []
        
        # List all CSV files and match against query
        for csv_file in self.data_dir.glob("*_daily.csv"):
            symbol = csv_file.stem.replace("_daily", "").upper()
            if query in symbol:
                results.append(SymbolInfo(
                    symbol=symbol,
                    name=symbol,
                    exchange="LOCAL",
                    currency="USD",
                    type="stock",
                    is_valid=True,
                ))
                if len(results) >= limit:
                    break
        
        return results

    async def fetch_max_available(
        self,
        symbol: str,
        interval: str = "1d",
    ) -> HistoricalData:
        """Fetch all available data for a symbol from local CSV."""
        symbol = symbol.upper()
        df = self._load_symbol_data(symbol)
        
        # Build numpy arrays in OHLCV format
        bars = np.column_stack([
            df['open'].values,
            df['high'].values,
            df['low'].values,
            df['close'].values,
            df['volume'].values,
        ]).astype(np.float64)

        timestamps = (df['date'].astype(np.int64) // 10**9).values.astype(np.float64)
        dates = df['date'].dt.strftime("%Y-%m-%d").tolist()

        logger.info(f"Returning max {len(bars)} bars for {symbol} ({dates[0]} to {dates[-1]})")

        return HistoricalData(
            symbol=symbol,
            bars=bars,
            timestamps=timestamps,
            dates=dates,
            provider=self.PROVIDER_NAME,
        )


# Provider is registered via decorator above
