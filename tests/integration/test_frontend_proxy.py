"""
Integration tests for frontend-backend proxy connectivity.

Tests verify that:
1. Backend API endpoints are accessible
2. Frontend proxy correctly forwards requests
3. CORS is not an issue (handled by Next.js proxy)
"""

import httpx
import pytest


class TestBackendDirectAccess:
    """Test direct access to backend endpoints."""

    BASE_URL = "http://localhost:8000"

    @pytest.mark.asyncio
    async def test_backend_status_endpoint(self):
        """Backend status endpoint should return valid data."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.BASE_URL}/api/status")
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "uptime_seconds" in data
            assert data["status"] in ["ACTIVE", "OFFLINE", "delayed"]

    @pytest.mark.asyncio
    async def test_backend_positions_endpoint(self):
        """Backend positions endpoint should return array."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.BASE_URL}/api/positions")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_backend_account_endpoint(self):
        """Backend account endpoint should return account data."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.BASE_URL}/api/account")
            assert response.status_code == 200
            data = response.json()
            assert "equity" in data
            assert "cash" in data
            assert "buying_power" in data
            assert isinstance(data["equity"], (int, float))


class TestFrontendProxyAccess:
    """Test access to backend through frontend proxy."""

    FRONTEND_URL = "http://localhost:3000"

    @pytest.mark.asyncio
    async def test_proxy_status_endpoint(self):
        """Frontend proxy should forward status requests correctly."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.FRONTEND_URL}/api/status")
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "uptime_seconds" in data

    @pytest.mark.asyncio
    async def test_proxy_positions_endpoint(self):
        """Frontend proxy should forward positions requests correctly."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.FRONTEND_URL}/api/positions")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_proxy_account_endpoint(self):
        """Frontend proxy should forward account requests correctly."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.FRONTEND_URL}/api/account")
            assert response.status_code == 200
            data = response.json()
            assert "equity" in data
            assert "cash" in data

    @pytest.mark.asyncio
    async def test_proxy_no_cors_errors(self):
        """
        Verify that CORS is not an issue when accessing through proxy.

        Since the proxy makes requests server-side, there should be no CORS
        preflight requests. The frontend sees all requests as same-origin.
        """
        async with httpx.AsyncClient() as client:
            # A successful request without CORS headers indicates the proxy is working
            response = await client.get(f"{self.FRONTEND_URL}/api/status")
            assert response.status_code == 200

            # Frontend proxy should not need CORS headers since it's same-origin
            # from the browser's perspective
            assert "access-control-allow-origin" not in response.headers


class TestDataConsistency:
    """Test that data is consistent between direct and proxied access."""

    BACKEND_URL = "http://localhost:8000"
    FRONTEND_URL = "http://localhost:3000"

    @pytest.mark.asyncio
    async def test_status_data_consistency(self):
        """Status data should be the same via direct and proxied access."""
        async with httpx.AsyncClient() as client:
            # Get data directly from backend
            backend_response = await client.get(f"{self.BACKEND_URL}/api/status")
            backend_data = backend_response.json()

            # Get data through frontend proxy
            frontend_response = await client.get(f"{self.FRONTEND_URL}/api/status")
            frontend_data = frontend_response.json()

            # Both should have same structure
            assert "status" in backend_data
            assert "status" in frontend_data
            assert "uptime_seconds" in backend_data
            assert "uptime_seconds" in frontend_data

    @pytest.mark.asyncio
    async def test_account_data_consistency(self):
        """Account data should be the same via direct and proxied access."""
        async with httpx.AsyncClient() as client:
            # Get data directly from backend
            backend_response = await client.get(f"{self.BACKEND_URL}/api/account")
            backend_data = backend_response.json()

            # Get data through frontend proxy
            frontend_response = await client.get(f"{self.FRONTEND_URL}/api/account")
            frontend_data = frontend_response.json()

            # Both should return identical data
            assert backend_data == frontend_data
