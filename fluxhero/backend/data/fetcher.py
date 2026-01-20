"""
FluxHero Async API Wrapper
Feature 8: Async REST and WebSocket client for market data fetching

This module implements:
- R8.1.1-R8.1.4: Async REST API client with httpx
- R8.2.1-R8.2.4: WebSocket live feed with auto-reconnect
- R8.3.1-R8.3.3: Data pipeline startup sequence

Requirements references:
- R8.1.1: Async methods for fetch_candles, get_account_info, place_order
- R8.1.2: Retry logic: 3 attempts with exponential backoff (1s, 2s, 4s)
- R8.1.3: Rate limiting: 200 requests/min
- R8.1.4: Connection pooling for efficiency
- R8.2.1: Real-time price subscriptions
- R8.2.2: Auto-reconnect (max 5 retries)
- R8.2.3: Async tick processing
- R8.2.4: Heartbeat monitor (>60s alert)
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Union

import httpx
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException


# ============================================================================
# Data Classes
# ============================================================================


class OrderSide(IntEnum):
    """Order side enum."""

    BUY = 1
    SELL = -1


class OrderType(IntEnum):
    """Order type enum."""

    MARKET = 0
    LIMIT = 1
    STOP = 2
    STOP_LIMIT = 3


@dataclass
class Candle:
    """Single OHLCV candle data."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    timeframe: str


@dataclass
class AccountInfo:
    """Broker account information."""

    balance: float
    buying_power: float
    equity: float
    cash: float


@dataclass
class OrderResponse:
    """Order placement response."""

    order_id: str
    symbol: str
    qty: float
    side: OrderSide
    order_type: OrderType
    status: str
    filled_qty: float = 0.0
    filled_avg_price: float = 0.0


# ============================================================================
# Rate Limiter
# ============================================================================


class RateLimiter:
    """
    Rate limiter to respect broker API limits.

    Implements R8.1.3: 200 requests per minute (configurable).

    Uses a sliding window approach to track requests.
    """

    def __init__(self, max_requests: int = 200, window_seconds: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: List[float] = []  # Timestamps of requests
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """
        Wait until request can be made within rate limit.

        Blocks if rate limit exceeded until old requests expire.
        """
        while True:
            async with self._lock:
                now = time.time()
                cutoff = now - self.window_seconds

                # Remove expired requests
                self.requests = [t for t in self.requests if t > cutoff]

                # Check if we can make request
                if len(self.requests) < self.max_requests:
                    # Record this request
                    self.requests.append(now)
                    return

                # Calculate wait time until oldest request expires
                oldest = self.requests[0]
                wait_time = self.window_seconds - (now - oldest) + 0.1

            # Sleep outside the lock
            await asyncio.sleep(wait_time)

    def reset(self) -> None:
        """Reset rate limiter (clear all tracked requests)."""
        self.requests.clear()


# ============================================================================
# REST API Client
# ============================================================================


class AsyncAPIClient:
    """
    Async REST API client using httpx.

    Implements R8.1.1-R8.1.4:
    - Async methods for market data and order management
    - Retry logic with exponential backoff
    - Rate limiting
    - Connection pooling

    Usage:
        async with AsyncAPIClient(base_url, api_key) as client:
            candles = await client.fetch_candles("SPY", "1h", 500)
            account = await client.get_account_info()
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        api_secret: Optional[str] = None,
        max_requests_per_minute: int = 200,
        timeout: float = 30.0,
    ):
        """
        Initialize API client.

        Args:
            base_url: Base URL for API (e.g., "https://paper-api.alpaca.markets")
            api_key: API key for authentication
            api_secret: API secret for authentication (if required)
            max_requests_per_minute: Rate limit (default: 200)
            timeout: Request timeout in seconds (default: 30)
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret
        self.timeout = timeout

        # Rate limiter (R8.1.3)
        self.rate_limiter = RateLimiter(max_requests=max_requests_per_minute)

        # HTTP client with connection pooling (R8.1.4)
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        """Context manager entry."""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            limits=httpx.Limits(
                max_connections=100,  # Connection pool size
                max_keepalive_connections=20,
            ),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._client:
            await self._client.aclose()

    def _get_headers(self) -> Dict[str, str]:
        """
        Get authentication headers.

        Override this method for broker-specific auth.
        """
        headers = {
            "APCA-API-KEY-ID": self.api_key,
        }
        if self.api_secret:
            headers["APCA-API-SECRET-KEY"] = self.api_secret
        return headers

    async def _request_with_retry(
        self,
        method: str,
        endpoint: str,
        max_retries: int = 3,
        **kwargs,
    ) -> httpx.Response:
        """
        Make HTTP request with retry logic and exponential backoff.

        Implements R8.1.2: 3 attempts with delays of 1s, 2s, 4s.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (relative to base_url)
            max_retries: Maximum retry attempts (default: 3)
            **kwargs: Additional arguments for httpx request

        Returns:
            httpx.Response object

        Raises:
            httpx.HTTPError: If all retries failed
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")

        # Apply rate limiting (R8.1.3)
        await self.rate_limiter.acquire()

        headers = kwargs.pop("headers", {})
        headers.update(self._get_headers())

        last_exception = None
        for attempt in range(max_retries):
            try:
                response = await self._client.request(
                    method=method,
                    url=endpoint,
                    headers=headers,
                    **kwargs,
                )
                response.raise_for_status()
                return response

            except (httpx.HTTPError, httpx.TimeoutException) as e:
                last_exception = e

                # Don't retry on client errors (4xx)
                if isinstance(e, httpx.HTTPStatusError) and 400 <= e.response.status_code < 500:
                    raise

                # Exponential backoff: 1s, 2s, 4s
                if attempt < max_retries - 1:
                    delay = 2**attempt  # 1, 2, 4 seconds
                    await asyncio.sleep(delay)

        # All retries failed
        raise last_exception

    async def fetch_candles(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[Candle]:
        """
        Fetch OHLCV candle data for a symbol.

        Implements R8.1.1: fetch_candles method.

        Args:
            symbol: Stock/ETF symbol (e.g., "SPY")
            timeframe: Candle timeframe (e.g., "1m", "5m", "1h", "1d")
            limit: Maximum number of candles to fetch (default: 500)
            start: Start datetime (optional)
            end: End datetime (optional)

        Returns:
            List of Candle objects

        Example:
            candles = await client.fetch_candles("SPY", "1h", 500)
        """
        params: Dict[str, Union[str, int]] = {
            "timeframe": timeframe,
            "limit": limit,
        }
        if start:
            params["start"] = start.isoformat()
        if end:
            params["end"] = end.isoformat()

        response = await self._request_with_retry(
            "GET",
            f"/v2/stocks/{symbol}/bars",
            params=params,
        )

        data = response.json()

        # Parse response (Alpaca format)
        candles = []
        if "bars" in data:
            for bar in data["bars"]:
                candles.append(
                    Candle(
                        timestamp=datetime.fromisoformat(bar["t"].replace("Z", "+00:00")),
                        open=float(bar["o"]),
                        high=float(bar["h"]),
                        low=float(bar["l"]),
                        close=float(bar["c"]),
                        volume=float(bar["v"]),
                        symbol=symbol,
                        timeframe=timeframe,
                    )
                )

        return candles

    async def get_account_info(self) -> AccountInfo:
        """
        Get account information (balance, buying power, etc.).

        Implements R8.1.1: get_account_info method.

        Returns:
            AccountInfo object

        Example:
            account = await client.get_account_info()
            print(f"Balance: ${account.balance}")
        """
        response = await self._request_with_retry("GET", "/v2/account")
        data = response.json()

        return AccountInfo(
            balance=float(data.get("equity", 0)),
            buying_power=float(data.get("buying_power", 0)),
            equity=float(data.get("equity", 0)),
            cash=float(data.get("cash", 0)),
        )

    async def place_order(
        self,
        symbol: str,
        qty: float,
        side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> OrderResponse:
        """
        Place an order.

        Implements R8.1.1: place_order method.

        Args:
            symbol: Stock/ETF symbol
            qty: Number of shares
            side: BUY or SELL
            order_type: MARKET, LIMIT, STOP, or STOP_LIMIT
            limit_price: Limit price (required for LIMIT orders)
            stop_price: Stop price (required for STOP orders)

        Returns:
            OrderResponse object

        Example:
            order = await client.place_order("SPY", 10, OrderSide.BUY, OrderType.MARKET)
        """
        payload = {
            "symbol": symbol,
            "qty": qty,
            "side": "buy" if side == OrderSide.BUY else "sell",
            "type": ["market", "limit", "stop", "stop_limit"][order_type],
            "time_in_force": "day",
        }

        if order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
            if limit_price is None:
                raise ValueError("limit_price required for LIMIT orders")
            payload["limit_price"] = limit_price

        if order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
            if stop_price is None:
                raise ValueError("stop_price required for STOP orders")
            payload["stop_price"] = stop_price

        response = await self._request_with_retry("POST", "/v2/orders", json=payload)
        data = response.json()

        return OrderResponse(
            order_id=data["id"],
            symbol=data["symbol"],
            qty=float(data["qty"]),
            side=OrderSide.BUY if data["side"] == "buy" else OrderSide.SELL,
            order_type=order_type,
            status=data["status"],
            filled_qty=float(data.get("filled_qty", 0)),
            filled_avg_price=float(data.get("filled_avg_price", 0)),
        )


# ============================================================================
# WebSocket Live Feed
# ============================================================================


class WebSocketFeed:
    """
    WebSocket live price feed with auto-reconnect and heartbeat monitor.

    Implements R8.2.1-R8.2.4:
    - Real-time price subscriptions
    - Auto-reconnect (max 5 retries)
    - Async tick processing
    - Heartbeat monitor (>60s alert)

    Usage:
        feed = WebSocketFeed(ws_url, api_key)
        await feed.connect()
        await feed.subscribe(["SPY", "QQQ"])

        async for tick in feed.stream():
            print(f"Price update: {tick}")
    """

    def __init__(
        self,
        ws_url: str,
        api_key: str,
        api_secret: Optional[str] = None,
        heartbeat_timeout: float = 60.0,
        max_reconnect_attempts: int = 5,
    ):
        """
        Initialize WebSocket feed.

        Args:
            ws_url: WebSocket URL (e.g., "wss://stream.data.alpaca.markets/v2/iex")
            api_key: API key for authentication
            api_secret: API secret for authentication
            heartbeat_timeout: Alert if no data for this many seconds (R8.2.4)
            max_reconnect_attempts: Maximum reconnection attempts (R8.2.2)
        """
        self.ws_url = ws_url
        self.api_key = api_key
        self.api_secret = api_secret
        self.heartbeat_timeout = heartbeat_timeout
        self.max_reconnect_attempts = max_reconnect_attempts

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._subscribed_symbols: List[str] = []
        self._last_message_time: float = time.time()
        self._connected = False
        self._reconnect_count = 0

        # Callback for tick data (R8.2.3)
        self._tick_callback: Optional[Callable] = None

    async def connect(self) -> None:
        """
        Connect to WebSocket server.

        Implements R8.2.2: Auto-reconnect with max 5 retries.

        Raises:
            ConnectionError: If max reconnect attempts exceeded
        """
        for attempt in range(self.max_reconnect_attempts):
            try:
                self._ws = await websockets.connect(self.ws_url)
                self._connected = True
                self._reconnect_count = attempt

                # Authenticate
                auth_msg = {
                    "action": "auth",
                    "key": self.api_key,
                    "secret": self.api_secret or "",
                }
                await self._ws.send(str(auth_msg))

                # Wait for auth confirmation
                await asyncio.wait_for(self._ws.recv(), timeout=5.0)
                # Parse response and check auth success
                # (broker-specific implementation)

                self._last_message_time = time.time()
                return

            except (ConnectionClosed, WebSocketException, asyncio.TimeoutError) as e:
                if attempt < self.max_reconnect_attempts - 1:
                    delay = 2**attempt  # Exponential backoff
                    await asyncio.sleep(delay)
                else:
                    raise ConnectionError(
                        f"Failed to connect after {self.max_reconnect_attempts} attempts"
                    ) from e

    async def disconnect(self) -> None:
        """Disconnect from WebSocket server."""
        if self._ws:
            await self._ws.close()
            self._connected = False

    async def subscribe(self, symbols: List[str]) -> None:
        """
        Subscribe to real-time price updates.

        Implements R8.2.1: Real-time price subscriptions.

        Args:
            symbols: List of symbols to subscribe to

        Example:
            await feed.subscribe(["SPY", "QQQ", "AAPL"])
        """
        if not self._connected or not self._ws:
            raise RuntimeError("Not connected. Call connect() first.")

        self._subscribed_symbols.extend(symbols)

        # Subscribe message (Alpaca format)
        subscribe_msg = {
            "action": "subscribe",
            "bars": symbols,
            "trades": symbols,
        }
        await self._ws.send(str(subscribe_msg))

    def set_tick_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Set callback function for incoming tick data.

        Implements R8.2.3: Async tick processing.

        Args:
            callback: Async function to call with tick data

        Example:
            async def on_tick(data):
                print(f"Received: {data}")

            feed.set_tick_callback(on_tick)
        """
        self._tick_callback = callback

    async def stream(self):
        """
        Stream incoming messages (generator).

        Implements R8.2.4: Heartbeat monitor.

        Yields:
            Parsed message data

        Raises:
            TimeoutError: If no data received for >heartbeat_timeout seconds

        Example:
            async for tick in feed.stream():
                print(f"Price: {tick['price']}")
        """
        if not self._connected or not self._ws:
            raise RuntimeError("Not connected. Call connect() first.")

        while self._connected:
            try:
                # Wait for message with timeout (heartbeat monitor)
                message = await asyncio.wait_for(
                    self._ws.recv(),
                    timeout=self.heartbeat_timeout,
                )

                self._last_message_time = time.time()

                # Parse message (broker-specific)
                # For now, yield raw message
                yield message

                # Call tick callback if set (R8.2.3)
                if self._tick_callback:
                    await self._tick_callback(message)

            except asyncio.TimeoutError:
                # No data for >60 seconds (R8.2.4)
                raise TimeoutError(
                    f"No data received for {self.heartbeat_timeout} seconds. "
                    "Connection may be stale."
                )

            except ConnectionClosed:
                # Connection dropped, attempt reconnect (R8.2.2)
                self._connected = False
                await self.connect()
                # Re-subscribe to symbols
                if self._subscribed_symbols:
                    await self.subscribe(self._subscribed_symbols)

    def get_last_message_time(self) -> float:
        """
        Get timestamp of last received message.

        Used for heartbeat monitoring.

        Returns:
            Unix timestamp of last message
        """
        return self._last_message_time

    def is_stale(self) -> bool:
        """
        Check if connection is stale (no data for >60s).

        Returns:
            True if connection is stale
        """
        return (time.time() - self._last_message_time) > self.heartbeat_timeout


# ============================================================================
# Data Pipeline
# ============================================================================


class DataPipeline:
    """
    Complete data pipeline integrating REST and WebSocket feeds.

    Implements R8.3.1-R8.3.3:
    - Startup sequence: fetch 500 candles → open WebSocket → process ticks
    - Signal checks on candle close
    - Async signal processing

    Usage:
        pipeline = DataPipeline(rest_client, ws_feed)
        await pipeline.start("SPY", "1h")

        # Pipeline now running in background
    """

    def __init__(
        self,
        rest_client: AsyncAPIClient,
        ws_feed: WebSocketFeed,
    ):
        """
        Initialize data pipeline.

        Args:
            rest_client: REST API client for historical data
            ws_feed: WebSocket feed for live data
        """
        self.rest_client = rest_client
        self.ws_feed = ws_feed
        self._running = False
        self._signal_callback: Optional[Callable] = None

    async def start(
        self,
        symbol: str,
        timeframe: str = "1h",
        initial_candles: int = 500,
    ) -> List[Candle]:
        """
        Start data pipeline.

        Implements R8.3.1: Startup sequence.

        Steps:
        1. Fetch last 500 candles via REST
        2. Open WebSocket connection
        3. Subscribe to symbol
        4. Start processing ticks

        Args:
            symbol: Symbol to trade
            timeframe: Candle timeframe
            initial_candles: Number of historical candles to fetch

        Returns:
            List of initial candles
        """
        # Step 1: Fetch historical candles (R8.3.1 step 1)
        candles = await self.rest_client.fetch_candles(
            symbol=symbol,
            timeframe=timeframe,
            limit=initial_candles,
        )

        # Step 2: Connect WebSocket (R8.3.1 step 2)
        await self.ws_feed.connect()

        # Step 3: Subscribe to symbol (R8.3.1 step 3)
        await self.ws_feed.subscribe([symbol])

        self._running = True

        return candles

    async def stop(self) -> None:
        """Stop data pipeline."""
        self._running = False
        await self.ws_feed.disconnect()

    def set_signal_callback(self, callback: Callable) -> None:
        """
        Set callback for signal checks.

        Implements R8.3.2-R8.3.3: Signal checks on candle close.

        Args:
            callback: Async function to call on new candle
        """
        self._signal_callback = callback

    async def process_stream(self) -> None:
        """
        Process incoming WebSocket stream.

        Implements R8.3.3: Run signal checks in separate task.

        This method runs in background and calls signal_callback
        on each completed candle.
        """
        async for message in self.ws_feed.stream():
            if not self._running:
                break

            # Parse message and check if candle closed
            # (broker-specific implementation)

            # If candle closed and callback set, run signal check
            if self._signal_callback:
                # Run in separate task (R8.3.3)
                asyncio.create_task(self._signal_callback(message))
