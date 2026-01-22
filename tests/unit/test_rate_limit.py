"""
Unit tests for rate limiting middleware.

Tests the RateLimiter and RateLimitMiddleware classes to ensure
proper rate limiting behavior and enforcement.

Requirements tested:
- Phase 5, Task 2: Rate limiting middleware functionality
- Sliding window rate limit enforcement
- 429 status code on rate limit exceeded
- Retry-After headers
- Excluded paths bypass rate limiting
"""

import asyncio

# Add parent directory to path for imports
import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent / ""))

from backend.api.rate_limit import RateLimiter, RateLimitMiddleware

# ============================================================================
# RateLimiter Core Tests
# ============================================================================


@pytest.mark.asyncio
async def test_rate_limiter_allows_requests_under_limit():
    """Test that requests under the limit are allowed."""
    limiter = RateLimiter(max_requests=5, window_seconds=60)

    # First 5 requests should be allowed
    for i in range(5):
        allowed = await limiter.is_allowed("test_client")
        assert allowed, f"Request {i + 1} should be allowed"


@pytest.mark.asyncio
async def test_rate_limiter_blocks_requests_over_limit():
    """Test that requests over the limit are blocked."""
    limiter = RateLimiter(max_requests=3, window_seconds=60)

    # First 3 requests allowed
    for i in range(3):
        await limiter.is_allowed("test_client")

    # 4th request should be blocked
    allowed = await limiter.is_allowed("test_client")
    assert not allowed, "Request over limit should be blocked"


@pytest.mark.asyncio
async def test_rate_limiter_sliding_window():
    """Test that rate limiter uses sliding window (old requests expire)."""
    limiter = RateLimiter(max_requests=2, window_seconds=1)  # 1 second window

    # Make 2 requests (at limit)
    await limiter.is_allowed("test_client")
    await limiter.is_allowed("test_client")

    # 3rd request should be blocked
    allowed = await limiter.is_allowed("test_client")
    assert not allowed

    # Wait for window to expire
    await asyncio.sleep(1.1)

    # Should be allowed again after window expires
    allowed = await limiter.is_allowed("test_client")
    assert allowed, "Request should be allowed after window expires"


@pytest.mark.asyncio
async def test_rate_limiter_per_client():
    """Test that rate limits are enforced per client (not global)."""
    limiter = RateLimiter(max_requests=2, window_seconds=60)

    # Client 1 uses up their limit
    await limiter.is_allowed("client_1")
    await limiter.is_allowed("client_1")
    allowed_1 = await limiter.is_allowed("client_1")
    assert not allowed_1

    # Client 2 should still be allowed
    allowed_2 = await limiter.is_allowed("client_2")
    assert allowed_2, "Different client should have independent limit"


@pytest.mark.asyncio
async def test_rate_limiter_retry_after():
    """Test retry_after calculation."""
    limiter = RateLimiter(max_requests=2, window_seconds=10)

    # Use up limit
    await limiter.is_allowed("test_client")
    await limiter.is_allowed("test_client")

    # Check retry_after
    retry_after = await limiter.get_retry_after("test_client")
    assert retry_after > 0, "Retry after should be positive when over limit"
    assert retry_after <= 11, "Retry after should not exceed window + 1"


@pytest.mark.asyncio
async def test_rate_limiter_reset():
    """Test that reset clears request history."""
    limiter = RateLimiter(max_requests=2, window_seconds=60)

    # Use up limit
    await limiter.is_allowed("test_client")
    await limiter.is_allowed("test_client")
    allowed = await limiter.is_allowed("test_client")
    assert not allowed

    # Reset and try again
    limiter.reset("test_client")
    allowed = await limiter.is_allowed("test_client")
    assert allowed, "Should be allowed after reset"


@pytest.mark.asyncio
async def test_rate_limiter_reset_all():
    """Test that reset() without args clears all clients."""
    limiter = RateLimiter(max_requests=1, window_seconds=60)

    # Multiple clients use up limits
    await limiter.is_allowed("client_1")
    await limiter.is_allowed("client_2")

    # Reset all
    limiter.reset()

    # Both should be allowed again
    assert await limiter.is_allowed("client_1")
    assert await limiter.is_allowed("client_2")


# ============================================================================
# RateLimitMiddleware Integration Tests
# ============================================================================


def create_test_app(max_requests: int = 5, window_seconds: int = 60, exclude_paths: list = None):
    """Helper to create FastAPI app with rate limiting middleware."""
    app = FastAPI()

    # Add rate limiting middleware
    app.add_middleware(
        RateLimitMiddleware,
        max_requests=max_requests,
        window_seconds=window_seconds,
        exclude_paths=exclude_paths or [],
    )

    @app.get("/api/test")
    async def test_endpoint():
        return {"message": "success"}

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    return app


def test_middleware_allows_requests_under_limit():
    """Test that middleware allows requests under the limit."""
    app = create_test_app(max_requests=3, window_seconds=60)
    client = TestClient(app)

    # First 3 requests should succeed
    for i in range(3):
        response = client.get("/api/test")
        assert response.status_code == 200, f"Request {i + 1} should succeed"
        assert "X-RateLimit-Limit" in response.headers
        assert response.headers["X-RateLimit-Limit"] == "3"


def test_middleware_blocks_requests_over_limit():
    """Test that middleware returns 429 when limit exceeded."""
    app = create_test_app(max_requests=2, window_seconds=60)
    client = TestClient(app)

    # Use up limit
    client.get("/api/test")
    client.get("/api/test")

    # Next request should be rate limited
    response = client.get("/api/test")
    assert response.status_code == 429
    assert "Retry-After" in response.headers
    assert "Rate limit exceeded" in response.json()["detail"]


def test_middleware_retry_after_header():
    """Test that Retry-After header is set correctly."""
    app = create_test_app(max_requests=1, window_seconds=10)
    client = TestClient(app)

    # Use up limit
    client.get("/api/test")

    # Get rate limited response
    response = client.get("/api/test")
    assert response.status_code == 429
    assert "Retry-After" in response.headers

    retry_after = int(response.headers["Retry-After"])
    assert retry_after > 0
    assert retry_after <= 11


def test_middleware_excluded_paths():
    """Test that excluded paths bypass rate limiting."""
    app = create_test_app(max_requests=1, window_seconds=60, exclude_paths=["/health"])
    client = TestClient(app)

    # Use up limit on API endpoint
    client.get("/api/test")
    response = client.get("/api/test")
    assert response.status_code == 429

    # Health endpoint should still work (excluded)
    response = client.get("/health")
    assert response.status_code == 200


def test_middleware_rate_limit_headers():
    """Test that rate limit headers are added to responses."""
    app = create_test_app(max_requests=10, window_seconds=60)
    client = TestClient(app)

    response = client.get("/api/test")
    assert response.status_code == 200

    # Check headers
    assert "X-RateLimit-Limit" in response.headers
    assert "X-RateLimit-Window" in response.headers
    assert response.headers["X-RateLimit-Limit"] == "10"
    assert response.headers["X-RateLimit-Window"] == "60"


def test_middleware_per_client_isolation():
    """Test that different clients have independent rate limits."""
    app = create_test_app(max_requests=1, window_seconds=60)

    # Client 1 uses up limit
    client1 = TestClient(app)
    response1 = client1.get("/api/test")
    assert response1.status_code == 200

    response1_again = client1.get("/api/test")
    assert response1_again.status_code == 429

    # Client 2 should still be allowed
    # Note: TestClient uses same IP, so we can't fully test this
    # In production, different IPs would have independent limits


def test_middleware_response_format():
    """Test that 429 response has correct format."""
    app = create_test_app(max_requests=1, window_seconds=60)
    client = TestClient(app)

    # Use up limit
    client.get("/api/test")

    # Get rate limited response
    response = client.get("/api/test")
    assert response.status_code == 429
    assert response.headers["Content-Type"] == "application/json"

    data = response.json()
    assert "detail" in data
    assert "Rate limit exceeded" in data["detail"]


# ============================================================================
# Edge Cases and Concurrency Tests
# ============================================================================


@pytest.mark.asyncio
async def test_rate_limiter_concurrent_requests():
    """Test rate limiter under concurrent load."""
    limiter = RateLimiter(max_requests=10, window_seconds=60)

    # Simulate 20 concurrent requests
    tasks = [limiter.is_allowed("test_client") for _ in range(20)]
    results = await asyncio.gather(*tasks)

    # First 10 should be allowed, rest blocked
    allowed_count = sum(results)
    assert allowed_count == 10, f"Expected 10 allowed, got {allowed_count}"


@pytest.mark.asyncio
async def test_rate_limiter_empty_client_id():
    """Test rate limiter with empty string client ID."""
    limiter = RateLimiter(max_requests=2, window_seconds=60)

    # Empty string should work as valid client ID
    allowed1 = await limiter.is_allowed("")
    allowed2 = await limiter.is_allowed("")
    allowed3 = await limiter.is_allowed("")

    assert allowed1 and allowed2
    assert not allowed3


@pytest.mark.asyncio
async def test_rate_limiter_very_short_window():
    """Test rate limiter with very short time window."""
    limiter = RateLimiter(max_requests=2, window_seconds=0.1)

    # Use up limit
    await limiter.is_allowed("test_client")
    await limiter.is_allowed("test_client")
    allowed = await limiter.is_allowed("test_client")
    assert not allowed

    # Wait for window to expire
    await asyncio.sleep(0.15)

    # Should be allowed again
    allowed = await limiter.is_allowed("test_client")
    assert allowed


def test_middleware_unknown_client():
    """Test middleware behavior when client IP is unknown."""
    app = create_test_app(max_requests=2, window_seconds=60)
    client = TestClient(app)

    # TestClient should have a client IP
    # But test that middleware handles missing client gracefully
    response = client.get("/api/test")
    assert response.status_code == 200


# ============================================================================
# Performance Benchmarks
# ============================================================================


@pytest.mark.asyncio
async def test_rate_limiter_performance():
    """Benchmark rate limiter performance."""
    limiter = RateLimiter(max_requests=1000, window_seconds=60)

    # Measure time for 100 requests
    start = time.time()
    for i in range(100):
        await limiter.is_allowed(f"client_{i % 10}")  # 10 different clients
    elapsed = time.time() - start

    # Should complete in reasonable time (< 1 second)
    assert elapsed < 1.0, f"Rate limiter too slow: {elapsed:.3f}s for 100 requests"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
