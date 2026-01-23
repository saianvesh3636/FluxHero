"""
Data Provider Abstraction Layer for FluxHero.

Provides a common interface for different data sources (Yahoo Finance, Alpaca, etc.)
Allows switching providers with minimal code changes.

Usage:
    from backend.data.provider import get_provider, DataProviderError

    provider = get_provider()  # Returns configured provider (Yahoo Finance by default)

    # Validate symbol before backtest
    is_valid = await provider.validate_symbol("AAPL")

    # Fetch historical data
    data = await provider.fetch_historical_data("AAPL", "2024-01-01", "2024-12-31")
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Supported data providers."""
    YAHOO_FINANCE = "yahoo"
    ALPACA = "alpaca"
    # Future: POLYGON = "polygon", IEX = "iex", etc.


class DataProviderError(Exception):
    """Base exception for data provider errors."""
    pass


class SymbolNotFoundError(DataProviderError):
    """Raised when symbol is not found or invalid."""
    def __init__(self, symbol: str, provider: str, message: str = ""):
        self.symbol = symbol
        self.provider = provider
        super().__init__(message or f"Symbol '{symbol}' not found on {provider}")


class InsufficientDataError(DataProviderError):
    """Raised when not enough data is available for the requested period."""
    def __init__(self, symbol: str, bars_found: int, bars_required: int):
        self.symbol = symbol
        self.bars_found = bars_found
        self.bars_required = bars_required
        super().__init__(
            f"Insufficient data for {symbol}: got {bars_found} bars, need {bars_required}"
        )


class DateRangeError(DataProviderError):
    """Raised when date range is invalid."""
    pass


class DataValidationError(DataProviderError):
    """Raised when data fails validation checks."""
    def __init__(self, symbol: str, issues: list[str]):
        self.symbol = symbol
        self.issues = issues
        issues_str = "; ".join(issues)
        super().__init__(f"Data validation failed for {symbol}: {issues_str}")


@dataclass
class SymbolInfo:
    """Symbol metadata returned by validation."""
    symbol: str
    name: str
    exchange: Optional[str] = None
    currency: Optional[str] = None
    type: Optional[str] = None  # stock, etf, etc.
    is_valid: bool = True


@dataclass
class HistoricalData:
    """Standardized historical data format for all providers."""
    symbol: str
    bars: np.ndarray  # Shape (N, 5): [open, high, low, close, volume]
    timestamps: np.ndarray  # Unix timestamps (float64)
    dates: list[str]  # Date strings for display
    provider: str


class DataProvider(ABC):
    """
    Abstract base class for data providers.

    All data providers must implement this interface.
    This allows switching between Yahoo Finance, Alpaca, Polygon, etc.
    with minimal code changes.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging and error messages."""
        pass

    @abstractmethod
    async def validate_symbol(self, symbol: str) -> SymbolInfo:
        """
        Validate if a symbol exists and get its metadata.

        Args:
            symbol: Ticker symbol (e.g., "AAPL", "SPY")

        Returns:
            SymbolInfo with symbol metadata

        Raises:
            SymbolNotFoundError: If symbol doesn't exist
            DataProviderError: On provider errors
        """
        pass

    @abstractmethod
    async def fetch_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> HistoricalData:
        """
        Fetch historical OHLCV data.

        Args:
            symbol: Ticker symbol
            start_date: Start date "YYYY-MM-DD"
            end_date: End date "YYYY-MM-DD"
            interval: Data interval ("1d", "1h", etc.)

        Returns:
            HistoricalData with bars, timestamps, dates

        Raises:
            SymbolNotFoundError: If symbol doesn't exist
            DateRangeError: If date range is invalid
            InsufficientDataError: If not enough data
            DataProviderError: On provider errors
        """
        pass

    @abstractmethod
    async def search_symbols(self, query: str, limit: int = 10) -> list[SymbolInfo]:
        """
        Search for symbols matching a query.

        Args:
            query: Search query (symbol or company name)
            limit: Maximum results to return

        Returns:
            List of matching SymbolInfo objects
        """
        pass


# Provider registry
_providers: dict[ProviderType, type[DataProvider]] = {}
_default_provider: Optional[ProviderType] = None


def register_provider(provider_type: ProviderType):
    """Decorator to register a data provider class."""
    def decorator(cls: type[DataProvider]):
        _providers[provider_type] = cls
        return cls
    return decorator


def set_default_provider(provider_type: ProviderType) -> None:
    """Set the default data provider."""
    global _default_provider
    if provider_type not in _providers:
        raise ValueError(f"Provider {provider_type} not registered")
    _default_provider = provider_type


def get_provider(provider_type: Optional[ProviderType] = None) -> DataProvider:
    """
    Get a data provider instance.

    Args:
        provider_type: Specific provider to use, or None for default

    Returns:
        DataProvider instance

    Raises:
        ValueError: If provider not available
    """
    if provider_type is None:
        if _default_provider is None:
            # Auto-select first available provider
            if ProviderType.YAHOO_FINANCE in _providers:
                provider_type = ProviderType.YAHOO_FINANCE
            elif _providers:
                provider_type = next(iter(_providers.keys()))
            else:
                raise ValueError("No data providers registered")
        else:
            provider_type = _default_provider

    if provider_type not in _providers:
        raise ValueError(f"Provider {provider_type} not available")

    return _providers[provider_type]()


# Import providers to trigger registration
# This is done at module level to ensure providers are registered
def _register_builtin_providers():
    """Register built-in providers."""
    try:
        from backend.data.yahoo_provider import YahooFinanceProvider  # noqa: F401
        logger.debug("Yahoo Finance provider registered")
    except ImportError as e:
        logger.warning(f"Yahoo Finance provider not available: {e}")


_register_builtin_providers()
