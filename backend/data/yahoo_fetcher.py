"""
Yahoo Finance Data Fetcher for FluxHero Backtesting.

Provides historical OHLCV data from Yahoo Finance for backtest execution.
Converts data to the format expected by BacktestEngine.
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class YahooFinanceError(Exception):
    """Custom exception for Yahoo Finance fetch errors."""

    pass


class YahooFinanceFetcher:
    """
    Fetches historical market data from Yahoo Finance.

    Features:
    - Symbol validation
    - Date range validation
    - Conversion to numpy arrays for BacktestEngine
    - Error handling with meaningful messages
    """

    def __init__(self, retry_count: int = 3):
        """
        Initialize the fetcher.

        Args:
            retry_count: Number of retries on failure (not used currently,
                        yfinance handles retries internally)
        """
        self.retry_count = retry_count

    def validate_symbol(self, symbol: str) -> bool:
        """
        Check if symbol is valid and has data.

        Args:
            symbol: Ticker symbol (e.g., "SPY", "AAPL")

        Returns:
            True if symbol exists and has data
        """
        try:
            ticker = yf.Ticker(symbol)
            # Try to get info - this will fail for invalid symbols
            info = ticker.info
            return info is not None and "symbol" in info
        except Exception:
            return False

    def fetch_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> dict:
        """
        Fetch OHLCV data for a symbol.

        Args:
            symbol: Ticker symbol (e.g., "SPY", "AAPL")
            start_date: Start date "YYYY-MM-DD"
            end_date: End date "YYYY-MM-DD"
            interval: Data interval ("1d", "1h", "1wk", etc.)

        Returns:
            dict with:
            - 'bars': numpy array (N, 5) [open, high, low, close, volume]
            - 'timestamps': numpy array of Unix timestamps (float64)
            - 'dates': list of date strings for display

        Raises:
            YahooFinanceError: If fetch fails or data is invalid
        """
        logger.info(f"Fetching {symbol} data from {start_date} to {end_date}")

        # Validate date format
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError as e:
            raise YahooFinanceError(f"Invalid date format: {e}")

        if end_dt <= start_dt:
            raise YahooFinanceError("End date must be after start date")

        # Fetch data from Yahoo Finance
        try:
            df = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
                auto_adjust=True,  # Adjust for splits/dividends
            )
        except Exception as e:
            raise YahooFinanceError(f"Failed to fetch data for {symbol}: {e}")

        # Check if we got data
        if df.empty:
            raise YahooFinanceError(
                f"No data returned for {symbol} between {start_date} and {end_date}. "
                "Check if the symbol is valid and the date range has trading days."
            )

        # Handle multi-level columns (yfinance sometimes returns these)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Validate required columns exist
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise YahooFinanceError(f"Missing columns in data: {missing_cols}")

        # Drop any rows with NaN values
        df = df.dropna()

        if len(df) < 1:
            raise YahooFinanceError(
                f"No valid data after removing NaN values for {symbol}"
            )

        logger.info(f"Fetched {len(df)} bars for {symbol}")

        # Convert to numpy arrays in BacktestEngine format
        return self._convert_to_backtest_format(df, symbol)

    def _convert_to_backtest_format(self, df: pd.DataFrame, symbol: str) -> dict:
        """
        Convert pandas DataFrame to numpy arrays for BacktestEngine.

        The BacktestEngine expects:
        - bars: numpy array (N, 5) with columns [open, high, low, close, volume]
        - timestamps: numpy array of Unix timestamps (float64)

        Args:
            df: DataFrame with OHLCV data (index is DatetimeIndex)
            symbol: Symbol name for logging

        Returns:
            dict with 'bars', 'timestamps', 'dates'
        """
        # Extract OHLCV columns in the exact order BacktestEngine expects
        opens = df["Open"].values.astype(np.float64)
        highs = df["High"].values.astype(np.float64)
        lows = df["Low"].values.astype(np.float64)
        closes = df["Close"].values.astype(np.float64)
        volumes = df["Volume"].values.astype(np.float64)

        # Stack into (N, 5) array: [open, high, low, close, volume]
        bars = np.column_stack([opens, highs, lows, closes, volumes])

        # Convert index to Unix timestamps (float64)
        timestamps = np.array(
            [ts.timestamp() for ts in df.index.to_pydatetime()], dtype=np.float64
        )

        # Create date strings for display
        dates = [ts.strftime("%Y-%m-%d") for ts in df.index.to_pydatetime()]

        logger.debug(f"Converted {symbol} data: {bars.shape[0]} bars")

        return {
            "bars": bars,
            "timestamps": timestamps,
            "dates": dates,
            "symbol": symbol,
        }

    def fetch_multiple_symbols(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> dict[str, dict]:
        """
        Fetch data for multiple symbols.

        Args:
            symbols: List of ticker symbols
            start_date: Start date "YYYY-MM-DD"
            end_date: End date "YYYY-MM-DD"
            interval: Data interval

        Returns:
            dict mapping symbol -> data dict

        Raises:
            YahooFinanceError: If any symbol fails
        """
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.fetch_historical_data(
                    symbol, start_date, end_date, interval
                )
            except YahooFinanceError as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                raise
        return results
