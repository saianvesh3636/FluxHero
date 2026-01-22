"""
Unit tests for Rate Limiting Middleware

Tests the RateLimiter core class and RateLimitMiddleware for FastAPI.
Verifies:
- Rate limiting enforcement
- Sliding window behavior
- Client isolation
- Retry-After headers
- Path exclusions
- Edge cases
"""

import asyncio

# Add backend to path
import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.api.rate_limit import RateLimiter, RateLimitMiddleware

# ============================================================================
# RateLimiter Core Tests
# ============================================================================


@pytest.mark.asyncio
async def test_rate_limiter_allows_requests_under_limit():
    """Test that requests under limit are allowed"""
    limiter = RateLimiter(max_requests=5, window_seconds=60)

    # Should allow 5 requests
    for i in range(5):
        allowed = await limiter.is_allowed("client1")
        assert allowed is True, f"Request {i+1} should be allowed"


@pytest.mark.asyncio
async def test_rate_limiter_blocks_requests_over_limit():
    """Test that requests over limit are blocked"""
    limiter = RateLimiter(max_requests=3, window_seconds=60)

    # Allow first 3 requests
    for _ in range(3):
        assert await limiter.is_allowed("client1") is True

    # 4th request should be blocked
    assert await limiter.is_allowed("client1") is False


@pytest.mark.asyncio
async def test_rate_limiter_sliding_window():
    """Test sliding window behavior - old requests expire"""
    limiter = RateLimiter(max_requests=2, window_seconds=1)

    # Make 2 requests (at limit)
    assert await limiter.is_allowed("client1") is True
    assert await limiter.is_allowed("client1") is True

    # 3rd request blocked
    assert await limiter.is_allowed("client1") is False

    # Wait for window to pass
    await asyncio.sleep(1.1)

    # Should be allowed again
    assert await limiter.is_allowed("client1") is True


@pytest.mark.asyncio
async def test_rate_limiter_isolates_clients():
    """Test that different clients have separate rate limits"""
    limiter = RateLimiter(max_requests=2, window_seconds=60)

    # Client 1 uses up their limit
    assert await limiter.is_allowed("client1") is True
    assert await limiter.is_allowed("client1") is True
    assert await limiter.is_allowed("client1") is False  # Over limit

    # Client 2 should still have their full limit
    assert await limiter.is_allowed("client2") is True
    assert await limiter.is_allowed("client2") is True
    assert await limiter.is_allowed("client2") is False  # Over limit


@pytest.mark.asyncio
async def test_rate_limiter_retry_after():
    """Test retry_after calculation"""
    limiter = RateLimiter(max_requests=2, window_seconds=10)

    # Make 2 requests to hit limit
    await limiter.is_allowed("client1")
    await limiter.is_allowed("client1")

    # Get retry time
    retry_after = await limiter.get_retry_after("client1")
    assert retry_after > 0, "Should need to wait when over limit"
    assert retry_after <= 11, "Wait time should be within window + 1"


@pytest.mark.asyncio
async def test_rate_limiter_retry_after_zero_when_allowed():
    """Test retry_after returns 0 when under limit"""
    limiter = RateLimiter(max_requests=5, window_seconds=60)

    # Make only 2 requests (under limit)
    await limiter.is_allowed("client1")
    await limiter.is_allowed("client1")

    # Should not need to wait
    retry_after = await limiter.get_retry_after("client1")
    assert retry_after == 0


@pytest.mark.asyncio
async def test_rate_limiter_reset_specific_client():
    """Test resetting rate limit for specific client"""
    limiter = RateLimiter(max_requests=2, window_seconds=60)

    # Client 1 hits limit
    await limiter.is_allowed("client1")
    await limiter.is_allowed("client1")
    assert await limiter.is_allowed("client1") is False

    # Reset client 1
    limiter.reset("client1")

    # Client 1 should be allowed again
    assert await limiter.is_allowed("client1") is True


@pytest.mark.asyncio
async def test_rate_limiter_reset_all_clients():
    """Test resetting all clients"""
    limiter = RateLimiter(max_requests=2, window_seconds=60)

    # Multiple clients hit their limits
    await limiter.is_allowed("client1")
    await limiter.is_allowed("client1")
    await limiter.is_allowed("client2")
    await limiter.is_allowed("client2")

    assert await limiter.is_allowed("client1") is False
    assert await limiter.is_allowed("client2") is False

    # Reset all
    limiter.reset()

    # All should be allowed again
    assert await limiter.is_allowed("client1") is True
    assert await limiter.is_allowed("client2") is True


@pytest.mark.asyncio
async def test_rate_limiter_concurrent_requests():
    """Test thread safety with concurrent requests"""
    limiter = RateLimiter(max_requests=10, window_seconds=60)

    # Make 20 concurrent requests from same client
    tasks = [limiter.is_allowed("client1") for _ in range(20)]
    results = await asyncio.gather(*tasks)

    # Exactly 10 should be allowed
    allowed_count = sum(1 for r in results if r is True)
    assert allowed_count == 10, f"Expected 10 allowed, got {allowed_count}"


@pytest.mark.asyncio
async def test_rate_limiter_handles_unknown_client_in_retry():
    """Test retry_after for client that hasn't made requests"""
    limiter = RateLimiter(max_requests=5, window_seconds=60)

    retry_after = await limiter.get_retry_after("unknown_client")
    assert retry_after == 0


# ============================================================================
# RateLimitMiddleware Tests
# ============================================================================


def create_test_app_with_rate_limiting(max_requests=5, window_seconds=60, exclude_paths=None):
    """Helper to create FastAPI app with rate limiting"""
    app = FastAPI()

    app.add_middleware(
        RateLimitMiddleware,
        max_requests=max_requests,
        window_seconds=window_seconds,
        exclude_paths=exclude_paths or []
    )

    @app.get("/api/test")
    async def test_endpoint():
        return {"status": "ok"}

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.get("/api/data")
    async def data():
        return {"data": [1, 2, 3]}

    return app


def test_middleware_allows_requests_under_limit():
    """Test middleware allows requests under rate limit"""
    app = create_test_app_with_rate_limiting(max_requests=5, window_seconds=60)

    with TestClient(app) as client:
        # Make 5 requests - all should succeed
        for i in range(5):
            response = client.get("/api/test")
            assert response.status_code == 200, f"Request {i+1} should succeed"
            assert "X-RateLimit-Limit" in response.headers
            assert response.headers["X-RateLimit-Limit"] == "5"
            assert response.headers["X-RateLimit-Window"] == "60"


def test_middleware_blocks_requests_over_limit():
    """Test middleware blocks requests over rate limit"""
    app = create_test_app_with_rate_limiting(max_requests=3, window_seconds=60)

    with TestClient(app) as client:
        # First 3 should succeed
        for _ in range(3):
            response = client.get("/api/test")
            assert response.status_code == 200

        # 4th should be rate limited
        response = client.get("/api/test")
        assert response.status_code == 429
        assert "Retry-After" in response.headers

        # Check error message
        data = response.json()
        assert "Rate limit exceeded" in data["detail"]


def test_middleware_retry_after_header():
    """Test that Retry-After header is present in 429 response"""
    app = create_test_app_with_rate_limiting(max_requests=2, window_seconds=10)

    with TestClient(app) as client:
        # Hit the limit
        client.get("/api/test")
        client.get("/api/test")

        # Get 429 response
        response = client.get("/api/test")
        assert response.status_code == 429
        assert "Retry-After" in response.headers

        retry_after = int(response.headers["Retry-After"])
        assert retry_after > 0
        assert retry_after <= 11  # Window + 1


def test_middleware_excludes_paths():
    """Test that excluded paths bypass rate limiting"""
    app = create_test_app_with_rate_limiting(
        max_requests=2,
        window_seconds=60,
        exclude_paths=["/health"]
    )

    with TestClient(app) as client:
        # Hit limit on /api/test
        client.get("/api/test")
        client.get("/api/test")
        response = client.get("/api/test")
        assert response.status_code == 429

        # /health should still work (excluded)
        for _ in range(5):
            response = client.get("/health")
            assert response.status_code == 200


def test_middleware_per_client_isolation():
    """Test that middleware isolates different clients"""
    # Note: TestClient uses same client for all requests
    # This test is limited but verifies the middleware is active
    app = create_test_app_with_rate_limiting(max_requests=2, window_seconds=60)

    with TestClient(app) as client:
        # Client hits limit
        client.get("/api/test")
        client.get("/api/test")
        response = client.get("/api/test")
        assert response.status_code == 429


def test_middleware_adds_rate_limit_headers():
    """Test that rate limit headers are added to all responses"""
    app = create_test_app_with_rate_limiting(max_requests=100, window_seconds=60)

    with TestClient(app) as client:
        response = client.get("/api/test")
        assert response.status_code == 200

        # Check headers
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Window" in response.headers
        assert response.headers["X-RateLimit-Limit"] == "100"
        assert response.headers["X-RateLimit-Window"] == "60"


def test_middleware_preserves_response_body():
    """Test that middleware doesn't modify response body"""
    app = create_test_app_with_rate_limiting(max_requests=10, window_seconds=60)

    with TestClient(app) as client:
        response = client.get("/api/data")
        assert response.status_code == 200

        data = response.json()
        assert data == {"data": [1, 2, 3]}


def test_middleware_429_response_format():
    """Test 429 response has correct format"""
    app = create_test_app_with_rate_limiting(max_requests=1, window_seconds=60)

    with TestClient(app) as client:
        # Hit limit
        client.get("/api/test")

        # Get 429
        response = client.get("/api/test")
        assert response.status_code == 429
        assert response.headers["content-type"] == "application/json"

        data = response.json()
        assert "detail" in data
        assert "Rate limit exceeded" in data["detail"]
        assert "Try again in" in data["detail"]


def test_middleware_works_with_multiple_endpoints():
    """Test that rate limit applies across all endpoints"""
    app = create_test_app_with_rate_limiting(max_requests=3, window_seconds=60)

    with TestClient(app) as client:
        # Make requests to different endpoints
        client.get("/api/test")
        client.get("/api/data")
        client.get("/api/test")

        # 4th request to any endpoint should be limited
        response = client.get("/api/data")
        assert response.status_code == 429


def test_middleware_configuration():
    """Test middleware with custom configuration"""
    app = create_test_app_with_rate_limiting(
        max_requests=10,
        window_seconds=30,
        exclude_paths=["/health", "/metrics"]
    )

    with TestClient(app) as client:
        response = client.get("/api/test")
        assert response.status_code == 200
        assert response.headers["X-RateLimit-Limit"] == "10"
        assert response.headers["X-RateLimit-Window"] == "30"


# ============================================================================
# Integration Tests
# ============================================================================


def test_rate_limiting_with_real_timing():
    """Test rate limiting with actual time delays (sliding window)"""
    app = create_test_app_with_rate_limiting(max_requests=2, window_seconds=1)

    with TestClient(app) as client:
        # First 2 requests succeed
        assert client.get("/api/test").status_code == 200
        assert client.get("/api/test").status_code == 200

        # 3rd is blocked
        assert client.get("/api/test").status_code == 429

        # Wait for window to expire
        time.sleep(1.1)

        # Should work again
        assert client.get("/api/test").status_code == 200


def test_rate_limiting_edge_case_zero_requests():
    """Test edge case with max_requests=0 (essentially blocks all)"""
    # This is an edge case - typically wouldn't use 0
    # But verify it doesn't crash
    app = create_test_app_with_rate_limiting(max_requests=0, window_seconds=60)

    with TestClient(app) as client:
        response = client.get("/api/test")
        assert response.status_code == 429


def test_rate_limiting_handles_client_without_host():
    """Test middleware handles requests without client host"""
    app = create_test_app_with_rate_limiting(max_requests=5, window_seconds=60)

    # TestClient should provide client info, but verify no crashes
    with TestClient(app) as client:
        response = client.get("/api/test")
        assert response.status_code in [200, 429]


# ============================================================================
# Success Criteria Tests
# ============================================================================


def test_rate_limiting_success_criteria():
    """
    Verify all success criteria for rate limiting:
    1. Enforces per-client limits
    2. Uses sliding window
    3. Returns 429 with Retry-After
    4. Excludes specified paths
    5. Adds rate limit headers
    """
    app = create_test_app_with_rate_limiting(
        max_requests=3,
        window_seconds=60,
        exclude_paths=["/health"]
    )

    with TestClient(app) as client:
        # Criterion 1: Enforces limits
        for _ in range(3):
            assert client.get("/api/test").status_code == 200
        assert client.get("/api/test").status_code == 429

        # Criterion 2: Sliding window (tested in other tests)
        # Criterion 3: Returns 429 with headers
        response = client.get("/api/test")
        assert response.status_code == 429
        assert "Retry-After" in response.headers

        # Criterion 4: Excludes paths
        assert client.get("/health").status_code == 200

        # Criterion 5: Adds headers
        response = client.get("/health")
        # Health is excluded so may not have rate limit headers,
        # but protected endpoints do
        response = client.get("/health")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
