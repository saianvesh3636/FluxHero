"""
Rate Limiting Middleware for FluxHero API

This module implements rate limiting middleware to protect API endpoints
from excessive requests. Adapted from the RateLimiter pattern in
backend/data/fetcher.py.

Requirements:
- Phase 5, Task 2: Add rate limiting middleware to API endpoints
- Prevent API abuse and ensure fair resource allocation
- Use sliding window approach for accurate rate limiting
"""

import asyncio
import time
from typing import List, Dict, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import HTTP_429_TOO_MANY_REQUESTS
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Rate Limiter Core
# ============================================================================


class RateLimiter:
    """
    Rate limiter using sliding window approach.

    Tracks requests per client IP and enforces configurable limits.
    Adapted from backend/data/fetcher.py RateLimiter.
    """

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed in window per client
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds

        # Track requests per client IP: {ip: [timestamp1, timestamp2, ...]}
        self.requests: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()

    async def is_allowed(self, client_id: str) -> bool:
        """
        Check if request from client is allowed within rate limit.

        Args:
            client_id: Unique identifier for client (usually IP address)

        Returns:
            True if request is allowed, False if rate limit exceeded
        """
        async with self._lock:
            now = time.time()
            cutoff = now - self.window_seconds

            # Initialize client if not seen before
            if client_id not in self.requests:
                self.requests[client_id] = []

            # Remove expired requests
            self.requests[client_id] = [
                t for t in self.requests[client_id] if t > cutoff
            ]

            # Check if we can make request
            if len(self.requests[client_id]) < self.max_requests:
                # Record this request
                self.requests[client_id].append(now)
                return True

            # Rate limit exceeded
            return False

    async def get_retry_after(self, client_id: str) -> int:
        """
        Get number of seconds until client can make another request.

        Args:
            client_id: Unique identifier for client

        Returns:
            Seconds to wait before retry (0 if allowed now)
        """
        async with self._lock:
            now = time.time()
            cutoff = now - self.window_seconds

            if client_id not in self.requests:
                return 0

            # Remove expired requests
            self.requests[client_id] = [
                t for t in self.requests[client_id] if t > cutoff
            ]

            # If under limit, no wait needed
            if len(self.requests[client_id]) < self.max_requests:
                return 0

            # Calculate wait time until oldest request expires
            oldest = self.requests[client_id][0]
            wait_time = int(self.window_seconds - (now - oldest) + 1)
            return max(0, wait_time)

    def reset(self, client_id: Optional[str] = None) -> None:
        """
        Reset rate limiter for specific client or all clients.

        Args:
            client_id: Client to reset, or None to reset all clients
        """
        if client_id:
            self.requests.pop(client_id, None)
        else:
            self.requests.clear()


# ============================================================================
# FastAPI Middleware
# ============================================================================


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting HTTP requests.

    Enforces per-IP rate limits and returns 429 status when exceeded.

    Usage:
        from backend.api.rate_limit import RateLimitMiddleware

        app.add_middleware(
            RateLimitMiddleware,
            max_requests=100,
            window_seconds=60,
            exclude_paths=["/health", "/"]
        )
    """

    def __init__(
        self,
        app,
        max_requests: int = 100,
        window_seconds: int = 60,
        exclude_paths: Optional[List[str]] = None,
    ):
        """
        Initialize rate limiting middleware.

        Args:
            app: FastAPI application instance
            max_requests: Maximum requests per window (default: 100)
            window_seconds: Time window in seconds (default: 60)
            exclude_paths: List of paths to exclude from rate limiting
        """
        super().__init__(app)
        self.rate_limiter = RateLimiter(
            max_requests=max_requests,
            window_seconds=window_seconds
        )
        self.exclude_paths = exclude_paths or []

        logger.info(
            "Rate limiting middleware initialized",
            extra={
                "max_requests": max_requests,
                "window_seconds": window_seconds,
                "exclude_paths": self.exclude_paths,
            }
        )

    async def dispatch(self, request: Request, call_next):
        """
        Process request and enforce rate limiting.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain

        Returns:
            HTTP response or 429 error if rate limited
        """
        # Skip rate limiting for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Get client identifier (IP address)
        client_ip = request.client.host if request.client else "unknown"

        # Check rate limit
        allowed = await self.rate_limiter.is_allowed(client_ip)

        if not allowed:
            # Rate limit exceeded
            retry_after = await self.rate_limiter.get_retry_after(client_ip)

            logger.warning(
                "Rate limit exceeded",
                extra={
                    "client_ip": client_ip,
                    "path": request.url.path,
                    "retry_after": retry_after,
                }
            )

            # Return 429 Too Many Requests
            return Response(
                content=f'{{"detail":"Rate limit exceeded. Try again in {retry_after} seconds."}}',
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.rate_limiter.max_requests),
                    "X-RateLimit-Window": str(self.rate_limiter.window_seconds),
                },
                media_type="application/json",
            )

        # Request allowed - proceed
        response = await call_next(request)

        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(self.rate_limiter.max_requests)
        response.headers["X-RateLimit-Window"] = str(self.rate_limiter.window_seconds)

        return response


# ============================================================================
# Dependency Injection (Alternative to Middleware)
# ============================================================================


async def rate_limit_dependency(
    request: Request,
    max_requests: int = 100,
    window_seconds: int = 60,
) -> None:
    """
    FastAPI dependency for per-endpoint rate limiting.

    Can be used as an alternative to global middleware for selective
    rate limiting on specific endpoints.

    Usage:
        @app.get("/api/expensive-operation", dependencies=[Depends(rate_limit_dependency)])
        async def expensive_operation():
            return {"status": "ok"}

    Args:
        request: FastAPI request object
        max_requests: Maximum requests per window
        window_seconds: Time window in seconds

    Raises:
        HTTPException: 429 if rate limit exceeded
    """
    # This is a simplified version - in production, use a shared RateLimiter instance
    # stored in app.state to maintain state across requests

    # For now, this serves as a template for dependency-based rate limiting
    # Note: This dependency needs a shared rate limiter instance in app.state
    # Example: rate_limiter = request.app.state.rate_limiter
    # client_ip = request.client.host if request.client else "unknown"

    raise NotImplementedError(
        "Dependency-based rate limiting requires shared RateLimiter in app.state. "
        "Use RateLimitMiddleware instead for global rate limiting."
    )
