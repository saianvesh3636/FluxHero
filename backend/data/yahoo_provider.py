"""
Yahoo Finance Data Provider for FluxHero.

Implements the DataProvider interface using yfinance library.
"""

import asyncio
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from backend.data.provider import (
    DataProvider,
    DataProviderError,
    DataValidationError,
    DateRangeError,
    HistoricalData,
    InsufficientDataError,
    ProviderType,
    SymbolInfo,
    SymbolNotFoundError,
    register_provider,
)

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not installed. Yahoo Finance provider unavailable.")


def validate_ohlcv_data(
    df: pd.DataFrame,
    symbol: str,
    max_gap_days: int = 5,
) -> list[str]:
    """
    Validate OHLCV data for common issues.

    Checks for:
    - NaN values in OHLCV columns
    - Negative prices (Open, High, Low, Close)
    - Zero volume
    - Invalid OHLC relationships (High < Low)
    - Large gaps in data (> max_gap_days trading days missing)

    Args:
        df: DataFrame with OHLCV columns and DatetimeIndex
        symbol: Symbol for error messages
        max_gap_days: Maximum allowed gap between trading days

    Returns:
        List of validation issues found (empty if data is valid)
    """
    issues: list[str] = []

    # Check for NaN values
    nan_cols = []
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns and df[col].isna().any():
            nan_count = df[col].isna().sum()
            nan_cols.append(f"{col}({nan_count})")
    if nan_cols:
        issues.append(f"NaN values in: {', '.join(nan_cols)}")

    # Check for negative prices
    price_cols = ["Open", "High", "Low", "Close"]
    neg_cols = []
    for col in price_cols:
        if col in df.columns and (df[col] < 0).any():
            neg_count = (df[col] < 0).sum()
            neg_cols.append(f"{col}({neg_count})")
    if neg_cols:
        issues.append(f"Negative prices in: {', '.join(neg_cols)}")

    # Check for zero volume
    if "Volume" in df.columns:
        zero_vol_count = (df["Volume"] == 0).sum()
        if zero_vol_count > 0:
            zero_vol_pct = (zero_vol_count / len(df)) * 100
            issues.append(
                f"Zero volume on {zero_vol_count} bars ({zero_vol_pct:.1f}%)"
            )

    # Check for High < Low (invalid OHLC relationship)
    if "High" in df.columns and "Low" in df.columns:
        invalid_hl = (df["High"] < df["Low"]).sum()
        if invalid_hl > 0:
            issues.append(f"High < Low on {invalid_hl} bars")

    # Check for large gaps in data
    if len(df) > 1 and hasattr(df.index, 'to_pydatetime'):
        dates = df.index.to_pydatetime()
        for i in range(1, len(dates)):
            gap_days = (dates[i] - dates[i - 1]).days
            # Account for weekends (2 days gap is normal for Mon-Fri)
            # Gap > max_gap_days means more than max_gap_days calendar days
            if gap_days > max_gap_days:
                issues.append(
                    f"Data gap of {gap_days} days between "
                    f"{dates[i-1].strftime('%Y-%m-%d')} and "
                    f"{dates[i].strftime('%Y-%m-%d')}"
                )

    if issues:
        logger.warning(
            f"Data validation issues for {symbol}: {'; '.join(issues)}"
        )

    return issues


@register_provider(ProviderType.YAHOO_FINANCE)
class YahooFinanceProvider(DataProvider):
    """
    Yahoo Finance data provider using yfinance library.

    Features:
    - Symbol validation with metadata
    - Historical OHLCV data fetching
    - Symbol search (basic)
    - Async-compatible (uses run_in_executor for blocking calls)
    """

    def __init__(self):
        if not YFINANCE_AVAILABLE:
            raise DataProviderError(
                "yfinance library not installed. Run: pip install yfinance"
            )

    @property
    def name(self) -> str:
        return "Yahoo Finance"

    @property
    def supported_intervals(self) -> dict[str, "IntervalInfo"]:
        """
        Yahoo Finance supported intervals.

        Native: 1m, 5m, 15m, 30m, 1h, 1d, 1wk
        Aggregated: 4h (from 1h)

        Note: Intraday data has limited history:
        - 1m: last 7 days
        - 5m, 15m, 30m: last 60 days
        - 1h: last 730 days
        """
        from backend.data.provider import IntervalInfo

        return {
            "1m": IntervalInfo("1m", 60, native=True),
            "5m": IntervalInfo("5m", 300, native=True),
            "15m": IntervalInfo("15m", 900, native=True),
            "30m": IntervalInfo("30m", 1800, native=True),
            "1h": IntervalInfo("1h", 3600, native=True),
            "4h": IntervalInfo("4h", 14400, native=False, aggregate_from="1h"),
            "1d": IntervalInfo("1d", 86400, native=True),
            "1wk": IntervalInfo("1wk", 604800, native=True),
        }

    async def validate_symbol(self, symbol: str) -> SymbolInfo:
        """
        Validate a symbol exists on Yahoo Finance.

        Returns SymbolInfo with company name, exchange, etc.
        """
        symbol = symbol.upper().strip()

        if not symbol:
            raise SymbolNotFoundError(symbol, self.name, "Empty symbol provided")

        # Run blocking yfinance call in executor
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._validate_symbol_sync, symbol)
        return result

    def _validate_symbol_sync(self, symbol: str) -> SymbolInfo:
        """Synchronous symbol validation."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Check if we got valid data
            if not info:
                raise SymbolNotFoundError(symbol, self.name)

            # yfinance returns empty dict or dict with just 'trailingPegRatio' for invalid symbols
            # Check for actual symbol data
            if "symbol" not in info and "shortName" not in info:
                # Try to fetch some price data as final check
                hist = ticker.history(period="5d")
                if hist.empty:
                    raise SymbolNotFoundError(
                        symbol, self.name,
                        f"Symbol '{symbol}' not found. Check if the ticker is correct."
                    )

            return SymbolInfo(
                symbol=symbol,
                name=info.get("shortName") or info.get("longName") or symbol,
                exchange=info.get("exchange"),
                currency=info.get("currency"),
                type=info.get("quoteType", "").lower() or None,
                is_valid=True,
            )

        except SymbolNotFoundError:
            raise
        except Exception as e:
            error_msg = str(e).lower()
            if "404" in error_msg or "not found" in error_msg or "delisted" in error_msg:
                raise SymbolNotFoundError(
                    symbol, self.name,
                    f"Symbol '{symbol}' not found or may be delisted."
                )
            raise DataProviderError(f"Error validating symbol {symbol}: {e}")

    async def fetch_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> HistoricalData:
        """
        Fetch historical OHLCV data from Yahoo Finance.

        Args:
            symbol: Ticker symbol (e.g., "SPY", "AAPL")
            start_date: Start date "YYYY-MM-DD"
            end_date: End date "YYYY-MM-DD"
            interval: Data interval ("1d", "1h", "4h", etc.)

        Returns:
            HistoricalData with bars, timestamps, dates

        Raises:
            SymbolNotFoundError: If symbol doesn't exist
            DateRangeError: If date range is invalid
            UnsupportedIntervalError: If interval not supported
            DataProviderError: On other errors
        """
        symbol = symbol.upper().strip()

        # Check if interval is supported and get info
        interval_info = self.get_interval_info(interval)

        # Validate date format and range
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError as e:
            raise DateRangeError(f"Invalid date format: {e}. Use YYYY-MM-DD format.")

        if end_dt <= start_dt:
            raise DateRangeError("End date must be after start date")

        if end_dt > datetime.now():
            raise DateRangeError("End date cannot be in the future")

        # Determine which interval to fetch
        fetch_interval = interval_info.aggregate_from if interval_info.aggregate_from else interval

        # Run blocking yfinance call in executor
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._fetch_data_sync,
            symbol,
            start_date,
            end_date,
            fetch_interval,
        )

        # Aggregate if needed (e.g., 1h -> 4h)
        if interval_info.aggregate_from:
            result = self._aggregate_bars(result, interval_info)

        return result

    def _aggregate_bars(
        self, data: HistoricalData, target_interval: "IntervalInfo"
    ) -> HistoricalData:
        """
        Aggregate bars from a smaller interval to a larger one.

        Args:
            data: HistoricalData with source bars
            target_interval: Target interval info (e.g., 4h)

        Returns:
            HistoricalData with aggregated bars
        """
        if data.bars.shape[0] == 0:
            return data

        source_info = self.supported_intervals.get(target_interval.aggregate_from)
        if not source_info:
            return data

        # Calculate how many source bars per target bar
        ratio = target_interval.seconds // source_info.seconds
        if ratio <= 1:
            return data

        # Aggregate bars
        aggregated_bars = []
        aggregated_timestamps = []
        aggregated_dates = []

        for i in range(0, len(data.bars), ratio):
            chunk = data.bars[i:i + ratio]
            if len(chunk) == 0:
                continue

            agg_bar = np.array([
                chunk[0, 0],  # open from first
                chunk[:, 1].max(),  # high is max
                chunk[:, 2].min(),  # low is min
                chunk[-1, 3],  # close from last
                chunk[:, 4].sum() if chunk.shape[1] > 4 else 0,  # volume sum
            ])
            aggregated_bars.append(agg_bar)
            aggregated_timestamps.append(data.timestamps[i])
            aggregated_dates.append(data.dates[i])

        return HistoricalData(
            symbol=data.symbol,
            bars=np.array(aggregated_bars),
            timestamps=np.array(aggregated_timestamps),
            dates=aggregated_dates,
            provider=data.provider,
        )

    def _fetch_data_sync(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str,
    ) -> HistoricalData:
        """Synchronous data fetching."""
        logger.info(f"Fetching {symbol} from Yahoo Finance: {start_date} to {end_date}")

        try:
            df = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
                auto_adjust=True,
            )
        except Exception as e:
            error_msg = str(e).lower()
            if "404" in error_msg or "not found" in error_msg:
                raise SymbolNotFoundError(symbol, self.name)
            raise DataProviderError(f"Failed to fetch data for {symbol}: {e}")

        # Check for empty data
        if df.empty:
            raise SymbolNotFoundError(
                symbol, self.name,
                f"No data returned for '{symbol}' between {start_date} and {end_date}. "
                "The symbol may be invalid, delisted, or have no trading data for this period."
            )

        # Handle multi-level columns (yfinance sometimes returns these)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Validate required columns
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise DataProviderError(f"Missing columns in data: {missing_cols}")

        # Validate data quality before processing
        validation_issues = validate_ohlcv_data(df, symbol, max_gap_days=5)

        # Separate critical issues (that should raise) from warnings
        critical_issues = [
            issue for issue in validation_issues
            if any(kw in issue.lower() for kw in ["nan values", "negative prices", "high < low"])
        ]

        if critical_issues:
            raise DataValidationError(symbol, critical_issues)

        # Drop rows with NaN values
        df = df.dropna()

        if len(df) < 1:
            raise InsufficientDataError(symbol, 0, 1)

        logger.info(f"Fetched {len(df)} bars for {symbol}")

        # Convert to numpy arrays in standardized format
        bars = np.column_stack([
            df["Open"].values.astype(np.float64),
            df["High"].values.astype(np.float64),
            df["Low"].values.astype(np.float64),
            df["Close"].values.astype(np.float64),
            df["Volume"].values.astype(np.float64),
        ])

        timestamps = np.array(
            [ts.timestamp() for ts in df.index.to_pydatetime()],
            dtype=np.float64,
        )

        dates = [ts.strftime("%Y-%m-%d") for ts in df.index.to_pydatetime()]

        return HistoricalData(
            symbol=symbol,
            bars=bars,
            timestamps=timestamps,
            dates=dates,
            provider=self.name,
        )

    async def search_symbols(self, query: str, limit: int = 10) -> list[SymbolInfo]:
        """
        Search for symbols matching a query (symbol or company name).

        Uses Yahoo Finance's search API for fuzzy matching.
        """
        if not query or len(query) < 1:
            return []

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, self._search_sync, query, limit)
        return results

    def _search_sync(self, query: str, limit: int) -> list[SymbolInfo]:
        """Synchronous symbol search using Yahoo Finance search API."""
        import json
        import urllib.parse
        import urllib.request

        results = []
        query = query.strip()

        try:
            # Use Yahoo Finance's search API
            encoded_query = urllib.parse.quote(query)
            url = f"https://query1.finance.yahoo.com/v1/finance/search?q={encoded_query}&quotesCount={limit}&newsCount=0&enableFuzzyQuery=true&quotesQueryId=tss_match_phrase_query"

            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
            )

            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())

            quotes = data.get("quotes", [])

            for quote in quotes[:limit]:
                # Filter to stocks and ETFs (skip crypto, futures, etc.)
                quote_type = quote.get("quoteType", "").upper()
                if quote_type not in ("EQUITY", "ETF", "MUTUALFUND"):
                    continue

                symbol = quote.get("symbol", "")
                if not symbol:
                    continue

                results.append(SymbolInfo(
                    symbol=symbol,
                    name=quote.get("shortname") or quote.get("longname") or symbol,
                    exchange=quote.get("exchange"),
                    currency=None,  # Not in search results
                    type=quote_type.lower(),
                    is_valid=True,
                ))

        except Exception as e:
            logger.warning(f"Yahoo search failed for '{query}': {e}")
            # Fallback: try exact symbol match
            try:
                ticker = yf.Ticker(query.upper())
                info = ticker.info
                if info and ("symbol" in info or "shortName" in info):
                    results.append(SymbolInfo(
                        symbol=query.upper(),
                        name=info.get("shortName") or info.get("longName") or query.upper(),
                        exchange=info.get("exchange"),
                        currency=info.get("currency"),
                        type=info.get("quoteType", "").lower() or None,
                        is_valid=True,
                    ))
            except Exception:
                pass

        return results[:limit]
