"""
Cache Service - Reusable time-series caching with incremental updates.

This module provides a generic caching pattern for time-series data:
- Check if cache exists and is fresh
- Fetch only missing data (incremental updates)
- No magic numbers - provider determines data availability

Usage:
    cache_service = CacheService(sqlite_store)

    # Get cached data with automatic incremental update
    data, needs_fetch, fetch_from = await cache_service.get_with_freshness_check(
        cache_key={"symbol": "SPY", "interval": "1d"},
        get_cached=lambda: store.get_cached_candles(...),
        get_date_from_item=lambda c: c.date,
    )

    if needs_fetch:
        # Fetch only missing data from fetch_from date
        new_data = await provider.fetch(start_date=fetch_from, ...)
        await cache_service.save_incremental(new_data)
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, TypeVar, Generic, Protocol

logger = logging.getLogger(__name__)

T = TypeVar('T')


class DateExtractor(Protocol):
    """Protocol for extracting date from cached items."""
    def __call__(self, item: Any) -> str: ...


@dataclass
class CacheFreshnessResult(Generic[T]):
    """Result of cache freshness check."""

    # All cached items
    cached_items: list[T]

    # Whether cache is current (has data up to reference date)
    is_current: bool

    # Whether any cache exists
    has_cache: bool

    # Date range of cached data
    min_date: str | None
    max_date: str | None

    # If not current, the date to fetch from (day after max_date)
    fetch_from_date: str | None

    # Reference date used for freshness check (usually yesterday)
    reference_date: str


class CacheService:
    """
    Generic time-series cache service with incremental update support.

    Implements the pattern:
    1. Check cache freshness
    2. If fresh: return cached data
    3. If stale: return cached data + date to fetch from
    4. If no cache: signal full fetch needed
    """

    def __init__(self, reference_date: str | None = None):
        """
        Initialize cache service.

        Args:
            reference_date: Date to check freshness against (default: yesterday)
        """
        if reference_date is None:
            reference_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        self.reference_date = reference_date

    def check_freshness(
        self,
        cached_items: list[T],
        get_date: Callable[[T], str],
        today: str | None = None,
    ) -> CacheFreshnessResult[T]:
        """
        Check if cached items are fresh (have data up to reference date).

        Args:
            cached_items: List of cached items
            get_date: Function to extract date string from each item
            today: Today's date (items with this date are excluded from cache check)

        Returns:
            CacheFreshnessResult with freshness info and fetch_from_date if stale
        """
        if today is None:
            today = datetime.now().strftime("%Y-%m-%d")

        if not cached_items:
            return CacheFreshnessResult(
                cached_items=[],
                is_current=False,
                has_cache=False,
                min_date=None,
                max_date=None,
                fetch_from_date=None,  # Full fetch needed
                reference_date=self.reference_date,
            )

        # Get date range (excluding today's data)
        dates = [get_date(item) for item in cached_items if get_date(item) != today]

        if not dates:
            return CacheFreshnessResult(
                cached_items=cached_items,
                is_current=False,
                has_cache=True,
                min_date=None,
                max_date=None,
                fetch_from_date=None,  # Full fetch needed
                reference_date=self.reference_date,
            )

        min_date = min(dates)
        max_date = max(dates)

        # Cache is current if it has data up to reference date (usually yesterday)
        is_current = max_date >= self.reference_date

        # If stale, calculate the date to fetch from (day after max cached date)
        fetch_from_date = None
        if not is_current:
            max_dt = datetime.strptime(max_date, "%Y-%m-%d")
            fetch_from_date = (max_dt + timedelta(days=1)).strftime("%Y-%m-%d")

        return CacheFreshnessResult(
            cached_items=cached_items,
            is_current=is_current,
            has_cache=True,
            min_date=min_date,
            max_date=max_date,
            fetch_from_date=fetch_from_date,
            reference_date=self.reference_date,
        )

    @staticmethod
    def filter_for_caching(
        items: list[T],
        get_date: Callable[[T], str],
        exclude_date: str,
    ) -> list[T]:
        """
        Filter items for caching (exclude today's data which may still update).

        Args:
            items: Items to filter
            get_date: Function to extract date from item
            exclude_date: Date to exclude (usually today)

        Returns:
            Items that should be cached
        """
        return [item for item in items if get_date(item) != exclude_date]

    @staticmethod
    def get_today() -> str:
        """Get today's date as YYYY-MM-DD string."""
        return datetime.now().strftime("%Y-%m-%d")

    @staticmethod
    def get_yesterday() -> str:
        """Get yesterday's date as YYYY-MM-DD string."""
        return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")


def create_cache_service() -> CacheService:
    """Factory function to create a cache service with current date reference."""
    return CacheService(reference_date=CacheService.get_yesterday())
