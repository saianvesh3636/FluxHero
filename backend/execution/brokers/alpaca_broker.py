"""
Alpaca Broker Adapter

This module implements the BrokerInterface for Alpaca trading API.
Handles Alpaca-specific authentication, order mapping, and connection management.

Feature: Multi-Broker Architecture (Phase A)
"""

import asyncio
import time
from typing import Any

import httpx
from loguru import logger

from backend.core.config import get_settings
from backend.execution.broker_base import (
    Account,
    BrokerHealth,
    BrokerInterface,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)


class AlpacaBroker(BrokerInterface):
    """
    Alpaca trading broker adapter implementing BrokerInterface.

    Provides connection pooling, retry logic, and unified response mapping
    for the Alpaca trading API.

    Attributes:
        api_key: Alpaca API key
        api_secret: Alpaca API secret
        base_url: Alpaca API base URL (paper or live)
    """

    # Alpaca API endpoints
    ACCOUNT_ENDPOINT = "/v2/account"
    POSITIONS_ENDPOINT = "/v2/positions"
    ORDERS_ENDPOINT = "/v2/orders"

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
    ):
        """
        Initialize Alpaca broker adapter.

        Args:
            api_key: Alpaca API key (defaults to config)
            api_secret: Alpaca API secret (defaults to config)
            base_url: Alpaca API base URL (defaults to config)
            timeout: Request timeout in seconds (default: 30.0)
        """
        settings = get_settings()

        self.api_key = api_key or settings.alpaca_api_key
        self.api_secret = api_secret or settings.alpaca_api_secret
        self.base_url = base_url or settings.alpaca_api_url
        self.timeout = timeout

        self._client: httpx.AsyncClient | None = None
        self._connected = False
        self._last_heartbeat: float | None = None

    def _get_headers(self) -> dict[str, str]:
        """Get Alpaca authentication headers."""
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json",
        }

    async def _request_with_retry(
        self,
        method: str,
        endpoint: str,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> httpx.Response:
        """
        Make HTTP request with retry logic and exponential backoff.

        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            endpoint: API endpoint (relative to base_url)
            max_retries: Maximum retry attempts (default: 3)
            **kwargs: Additional arguments for httpx request

        Returns:
            httpx.Response object

        Raises:
            httpx.HTTPError: If all retries failed
            RuntimeError: If client not connected
        """
        if not self._client:
            raise RuntimeError("Broker not connected. Call connect() first.")

        headers = kwargs.pop("headers", {})
        headers.update(self._get_headers())

        last_exception: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = await self._client.request(
                    method=method,
                    url=endpoint,
                    headers=headers,
                    **kwargs,
                )
                response.raise_for_status()
                self._last_heartbeat = time.time()
                return response

            except (httpx.HTTPError, httpx.TimeoutException) as e:
                last_exception = e
                logger.warning(f"Alpaca request failed (attempt {attempt + 1}/{max_retries}): {e}")

                # Don't retry on client errors (4xx)
                if isinstance(e, httpx.HTTPStatusError) and 400 <= e.response.status_code < 500:
                    raise

                # Exponential backoff: 1s, 2s, 4s
                if attempt < max_retries - 1:
                    delay = 2**attempt
                    await asyncio.sleep(delay)

        # All retries failed
        if last_exception:
            raise last_exception
        raise RuntimeError("Request failed with no exception")

    # -------------------------------------------------------------------------
    # Connection Lifecycle Methods
    # -------------------------------------------------------------------------

    async def connect(self) -> bool:
        """
        Establish connection to Alpaca API.

        Validates credentials by making a test request to the account endpoint.

        Returns:
            True if connection successful, False otherwise
        """
        if not self.api_key or not self.api_secret:
            logger.error("Alpaca API credentials not configured")
            return False

        try:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=20,
                ),
            )

            # Test connection by fetching account
            response = await self._request_with_retry("GET", self.ACCOUNT_ENDPOINT)
            response.raise_for_status()

            self._connected = True
            self._last_heartbeat = time.time()
            logger.info(f"Connected to Alpaca API at {self.base_url}")
            return True

        except httpx.HTTPStatusError as e:
            logger.error(f"Alpaca authentication failed: {e.response.status_code}")
            await self.disconnect()
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            await self.disconnect()
            return False

    async def disconnect(self) -> None:
        """Close connection to Alpaca API."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._connected = False
        self._last_heartbeat = None
        logger.info("Disconnected from Alpaca API")

    async def health_check(self) -> BrokerHealth:
        """
        Check the health of the Alpaca connection.

        Returns:
            BrokerHealth object with connection status
        """
        if not self._connected or not self._client:
            return BrokerHealth(
                is_connected=False,
                is_authenticated=False,
                latency_ms=None,
                last_heartbeat=self._last_heartbeat,
                error_message="Not connected",
            )

        try:
            start_time = time.time()
            response = await self._request_with_retry("GET", self.ACCOUNT_ENDPOINT)
            latency_ms = (time.time() - start_time) * 1000

            account_data = response.json()
            is_active = account_data.get("status") == "ACTIVE"

            return BrokerHealth(
                is_connected=True,
                is_authenticated=is_active,
                latency_ms=latency_ms,
                last_heartbeat=self._last_heartbeat,
                error_message=None if is_active else "Account not active",
            )

        except Exception as e:
            logger.warning(f"Alpaca health check failed: {e}")
            return BrokerHealth(
                is_connected=False,
                is_authenticated=False,
                latency_ms=None,
                last_heartbeat=self._last_heartbeat,
                error_message=str(e),
            )

    # -------------------------------------------------------------------------
    # Account and Position Methods
    # -------------------------------------------------------------------------

    async def get_account(self) -> Account:
        """
        Get Alpaca account information.

        Returns:
            Account object with balance, buying power, equity, etc.
        """
        response = await self._request_with_retry("GET", self.ACCOUNT_ENDPOINT)
        data = response.json()

        return Account(
            account_id=data.get("account_number", data.get("id", "")),
            balance=float(data.get("cash", 0)),
            buying_power=float(data.get("buying_power", 0)),
            equity=float(data.get("equity", 0)),
            cash=float(data.get("cash", 0)),
            positions_value=float(data.get("long_market_value", 0))
            + float(data.get("short_market_value", 0)),
        )

    async def get_positions(self) -> list[Position]:
        """
        Get all open positions from Alpaca.

        Returns:
            List of Position objects
        """
        response = await self._request_with_retry("GET", self.POSITIONS_ENDPOINT)
        positions_data = response.json()

        positions = []
        for data in positions_data:
            qty = int(data.get("qty", 0))
            side = data.get("side", "long")
            if side == "short":
                qty = -qty

            position = Position(
                symbol=data.get("symbol", ""),
                qty=qty,
                entry_price=float(data.get("avg_entry_price", 0)),
                current_price=float(data.get("current_price", 0)),
                unrealized_pnl=float(data.get("unrealized_pl", 0)),
                market_value=float(data.get("market_value", 0)),
            )
            positions.append(position)

        return positions

    # -------------------------------------------------------------------------
    # Order Methods
    # -------------------------------------------------------------------------

    def _map_order_side(self, side: OrderSide) -> str:
        """Map internal OrderSide to Alpaca side string."""
        return "buy" if side == OrderSide.BUY else "sell"

    def _map_order_type(self, order_type: OrderType) -> str:
        """Map internal OrderType to Alpaca type string."""
        mapping = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP: "stop",
            OrderType.STOP_LIMIT: "stop_limit",
        }
        return mapping.get(order_type, "market")

    def _map_alpaca_status(self, status: str) -> OrderStatus:
        """Map Alpaca status string to internal OrderStatus."""
        status_mapping = {
            "new": OrderStatus.PENDING,
            "accepted": OrderStatus.PENDING,
            "pending_new": OrderStatus.PENDING,
            "accepted_for_bidding": OrderStatus.PENDING,
            "filled": OrderStatus.FILLED,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "canceled": OrderStatus.CANCELLED,
            "cancelled": OrderStatus.CANCELLED,
            "expired": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
            "pending_cancel": OrderStatus.PENDING,
            "pending_replace": OrderStatus.PENDING,
            "replaced": OrderStatus.PENDING,
        }
        return status_mapping.get(status.lower(), OrderStatus.PENDING)

    def _parse_alpaca_order(self, data: dict[str, Any]) -> Order:
        """Parse Alpaca order response into Order object."""
        filled_qty = int(data.get("filled_qty") or 0)
        filled_price = data.get("filled_avg_price")

        return Order(
            order_id=data.get("id", ""),
            symbol=data.get("symbol", ""),
            qty=int(data.get("qty", 0)),
            side=OrderSide.BUY if data.get("side") == "buy" else OrderSide.SELL,
            order_type=self._parse_order_type(data.get("type", "market")),
            limit_price=float(data["limit_price"]) if data.get("limit_price") else None,
            stop_price=float(data["stop_price"]) if data.get("stop_price") else None,
            status=self._map_alpaca_status(data.get("status", "new")),
            filled_qty=filled_qty,
            filled_price=float(filled_price) if filled_price else None,
            created_at=time.time(),
            updated_at=time.time(),
        )

    def _parse_order_type(self, type_str: str) -> OrderType:
        """Parse Alpaca order type string to OrderType."""
        mapping = {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT,
            "stop": OrderType.STOP,
            "stop_limit": OrderType.STOP_LIMIT,
        }
        return mapping.get(type_str.lower(), OrderType.MARKET)

    async def place_order(
        self,
        symbol: str,
        qty: int,
        side: OrderSide,
        order_type: OrderType,
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> Order:
        """
        Place an order with Alpaca.

        Args:
            symbol: Trading symbol
            qty: Number of shares
            side: BUY or SELL
            order_type: MARKET, LIMIT, STOP, or STOP_LIMIT
            limit_price: Limit price (required for LIMIT and STOP_LIMIT)
            stop_price: Stop price (required for STOP and STOP_LIMIT)

        Returns:
            Order object with order details
        """
        # Validate required prices
        if order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and limit_price is None:
            raise ValueError(f"limit_price required for {order_type.name} orders")

        if order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and stop_price is None:
            raise ValueError(f"stop_price required for {order_type.name} orders")

        # Build order payload
        payload: dict[str, Any] = {
            "symbol": symbol,
            "qty": str(qty),
            "side": self._map_order_side(side),
            "type": self._map_order_type(order_type),
            "time_in_force": "day",
        }

        if limit_price is not None:
            payload["limit_price"] = str(limit_price)

        if stop_price is not None:
            payload["stop_price"] = str(stop_price)

        logger.info(f"Placing Alpaca order: {payload}")

        response = await self._request_with_retry(
            "POST",
            self.ORDERS_ENDPOINT,
            json=payload,
        )

        order_data = response.json()
        order = self._parse_alpaca_order(order_data)

        logger.info(f"Order placed: {order.order_id} - {symbol} {side.name} {qty}")
        return order

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order on Alpaca.

        Args:
            order_id: Alpaca order ID

        Returns:
            True if cancellation successful, False otherwise
        """
        try:
            await self._request_with_retry(
                "DELETE",
                f"{self.ORDERS_ENDPOINT}/{order_id}",
            )
            logger.info(f"Order cancelled: {order_id}")
            return True

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"Order not found: {order_id}")
                return False
            if e.response.status_code == 422:
                # Order cannot be cancelled (already filled/cancelled)
                logger.warning(f"Order cannot be cancelled: {order_id}")
                return False
            raise

    async def get_order_status(self, order_id: str) -> Order | None:
        """
        Get the current status of an order from Alpaca.

        Args:
            order_id: Alpaca order ID

        Returns:
            Order object with current status, or None if not found
        """
        try:
            response = await self._request_with_retry(
                "GET",
                f"{self.ORDERS_ENDPOINT}/{order_id}",
            )
            order_data = response.json()
            return self._parse_alpaca_order(order_data)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
