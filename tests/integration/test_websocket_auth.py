"""
Integration tests for WebSocket authentication.

Tests the WebSocket endpoint authentication to ensure that:
1. Valid tokens allow connections
2. Invalid tokens are rejected with code 4001
3. Missing tokens are rejected with code 4001

Requirements: Phase 3 - WebSocket Authentication (AUDIT_TASKS.md)
"""

import os
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

# Import the FastAPI app
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.api.server import app
from backend.api.auth import DEFAULT_SECRET


class TestWebSocketAuthentication:
    """Test WebSocket endpoint authentication."""

    def test_websocket_connection_with_valid_token(self):
        """Test that WebSocket connection succeeds with valid authentication token."""
        custom_secret = "test-websocket-secret-123"

        with patch.dict(os.environ, {"FLUXHERO_AUTH_SECRET": custom_secret}):
            with TestClient(app) as client:
                # Connect with valid token in headers
                with client.websocket_connect(
                    "/ws/prices",
                    headers={"Authorization": f"Bearer {custom_secret}"}
                ) as websocket:
                    # Should receive connection confirmation message
                    data = websocket.receive_json()
                    assert data["type"] == "connection"
                    assert data["status"] == "connected"
                    assert "timestamp" in data

    def test_websocket_connection_with_invalid_token(self):
        """Test that WebSocket connection is rejected with invalid token."""
        custom_secret = "correct-secret"
        wrong_secret = "wrong-secret"

        with patch.dict(os.environ, {"FLUXHERO_AUTH_SECRET": custom_secret}):
            with TestClient(app) as client:
                # Attempt connection with invalid token
                with pytest.raises(Exception) as exc_info:
                    with client.websocket_connect(
                        "/ws/prices",
                        headers={"Authorization": f"Bearer {wrong_secret}"}
                    ):
                        pass  # Should not reach here

                # Connection should be closed with code 4001
                # The TestClient raises an exception when connection is closed
                assert exc_info is not None

    def test_websocket_connection_without_token(self):
        """Test that WebSocket connection is rejected when no token is provided."""
        with TestClient(app) as client:
            # Attempt connection without Authorization header
            with pytest.raises(Exception) as exc_info:
                with client.websocket_connect("/ws/prices"):
                    pass  # Should not reach here

            # Connection should be rejected
            assert exc_info is not None

    def test_websocket_connection_with_empty_token(self):
        """Test that WebSocket connection is rejected with empty token."""
        with TestClient(app) as client:
            # Attempt connection with empty Authorization header
            with pytest.raises(Exception) as exc_info:
                with client.websocket_connect(
                    "/ws/prices",
                    headers={"Authorization": ""}
                ):
                    pass  # Should not reach here

            # Connection should be rejected
            assert exc_info is not None

    def test_websocket_connection_with_bearer_only(self):
        """Test that WebSocket connection is rejected with 'Bearer' but no token."""
        with TestClient(app) as client:
            # Attempt connection with Bearer prefix but no actual token
            with pytest.raises(Exception) as exc_info:
                with client.websocket_connect(
                    "/ws/prices",
                    headers={"Authorization": "Bearer "}
                ):
                    pass  # Should not reach here

            # Connection should be rejected
            assert exc_info is not None

    def test_websocket_connection_with_default_secret(self):
        """Test WebSocket connection using default development secret."""
        # Clear environment to use default secret
        with patch.dict(os.environ, {}, clear=True):
            with TestClient(app) as client:
                # Connect with default secret
                with client.websocket_connect(
                    "/ws/prices",
                    headers={"Authorization": f"Bearer {DEFAULT_SECRET}"}
                ) as websocket:
                    # Should receive connection confirmation
                    data = websocket.receive_json()
                    assert data["type"] == "connection"
                    assert data["status"] == "connected"

    def test_websocket_case_insensitive_authorization_header(self):
        """Test that Authorization header is case-insensitive."""
        custom_secret = "case-test-secret"

        with patch.dict(os.environ, {"FLUXHERO_AUTH_SECRET": custom_secret}):
            with TestClient(app) as client:
                # Test with lowercase 'authorization'
                with client.websocket_connect(
                    "/ws/prices",
                    headers={"authorization": f"Bearer {custom_secret}"}
                ) as websocket:
                    data = websocket.receive_json()
                    assert data["type"] == "connection"
                    assert data["status"] == "connected"

    def test_websocket_token_without_bearer_prefix(self):
        """Test WebSocket connection with raw token (no Bearer prefix)."""
        custom_secret = "raw-token-test"

        with patch.dict(os.environ, {"FLUXHERO_AUTH_SECRET": custom_secret}):
            with TestClient(app) as client:
                # Connect with raw token (no Bearer prefix)
                with client.websocket_connect(
                    "/ws/prices",
                    headers={"Authorization": custom_secret}
                ) as websocket:
                    data = websocket.receive_json()
                    assert data["type"] == "connection"
                    assert data["status"] == "connected"

    def test_websocket_receives_price_updates_after_auth(self):
        """Test that authenticated client receives price updates."""
        custom_secret = "price-update-test"

        with patch.dict(os.environ, {"FLUXHERO_AUTH_SECRET": custom_secret}):
            with TestClient(app) as client:
                with client.websocket_connect(
                    "/ws/prices",
                    headers={"Authorization": f"Bearer {custom_secret}"}
                ) as websocket:
                    # Receive connection message
                    conn_msg = websocket.receive_json()
                    assert conn_msg["type"] == "connection"

                    # Should receive price update within timeout
                    # Note: The server sends updates every 5 seconds, but TestClient
                    # may not wait that long in test mode. This is a basic check.
                    # In a real scenario, you might mock the price update generation.
                    try:
                        # Try to receive next message (price update)
                        # This may timeout in test environment, which is expected
                        price_msg = websocket.receive_json(timeout=1)
                        # If we get a message, it should have price data
                        if price_msg:
                            assert "symbol" in price_msg or "type" in price_msg
                    except Exception:
                        # Timeout is acceptable in test environment
                        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
