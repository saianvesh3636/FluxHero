"""
Unit tests for Broker Management API Endpoints

Tests the broker configuration CRUD operations and health check:
- GET /api/brokers - List configured brokers
- POST /api/brokers - Add broker configuration
- DELETE /api/brokers/{id} - Delete broker configuration
- GET /api/brokers/{id}/health - Check broker connection health

Feature: Multi-Broker Architecture (Phase A)
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.api.server import app, app_state
from backend.storage.sqlite_store import SQLiteStore


@pytest.fixture
def test_db_path(tmp_path):
    """Create a temporary test database path"""
    # Just return the path - the store will be created in the lifespan
    db_path = tmp_path / "test.db"
    return str(db_path)


@pytest.fixture
def client(test_db_path):
    """Create a test client with properly initialized database"""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def test_lifespan(app):
        # Create and initialize store within the lifespan's event loop
        # This ensures the async write worker runs in the correct loop
        store = SQLiteStore(db_path=test_db_path)
        await store.initialize()
        app_state.sqlite_store = store
        app_state.data_feed_active = False
        app_state.start_time = datetime.now()
        app_state.last_update = datetime.now()
        yield
        app_state.websocket_clients.clear()
        await store.close()

    app.router.lifespan_context = test_lifespan

    with TestClient(app) as test_client:
        yield test_client


# ============================================================================
# List Brokers Tests (GET /api/brokers)
# ============================================================================


def test_list_brokers_empty(client):
    """Test listing brokers when none exist"""
    response = client.get("/api/brokers")
    assert response.status_code == 200
    data = response.json()
    assert data["brokers"] == []
    assert data["total"] == 0


def test_list_brokers_after_add(client):
    """Test listing brokers after adding one"""
    # Add a broker
    add_response = client.post(
        "/api/brokers",
        json={
            "broker_type": "alpaca",
            "name": "Test Broker",
            "api_key": "PKTEST12345678901234",
            "api_secret": "abcdefghijklmnopqrstuvwxyz123456789012345678",
        },
    )
    assert add_response.status_code == 201

    # List brokers
    response = client.get("/api/brokers")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert len(data["brokers"]) == 1

    broker = data["brokers"][0]
    assert broker["broker_type"] == "alpaca"
    assert broker["name"] == "Test Broker"
    # Credentials should be masked
    assert "****" in broker["api_key_masked"]
    assert "PKTEST12345678901234" not in broker["api_key_masked"]


def test_list_brokers_multiple(client):
    """Test listing multiple brokers"""
    # Add two brokers
    for i in range(2):
        client.post(
            "/api/brokers",
            json={
                "broker_type": "alpaca",
                "name": f"Broker {i}",
                "api_key": f"PKTEST{i}2345678901234",
                "api_secret": f"abcdefghijklmnopqrstuvwxyz{i}23456789012345678",
            },
        )

    response = client.get("/api/brokers")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["brokers"]) == 2


# ============================================================================
# Add Broker Tests (POST /api/brokers)
# ============================================================================


def test_add_broker_success(client):
    """Test adding a broker configuration"""
    response = client.post(
        "/api/brokers",
        json={
            "broker_type": "alpaca",
            "name": "My Alpaca Account",
            "api_key": "PKTEST12345678901234",
            "api_secret": "abcdefghijklmnopqrstuvwxyz123456789012345678",
        },
    )

    assert response.status_code == 201
    data = response.json()
    assert data["broker_type"] == "alpaca"
    assert data["name"] == "My Alpaca Account"
    assert "id" in data
    assert len(data["id"]) == 8  # UUID prefix
    assert "****" in data["api_key_masked"]
    assert "created_at" in data
    assert "updated_at" in data


def test_add_broker_with_custom_base_url(client):
    """Test adding broker with custom base URL"""
    response = client.post(
        "/api/brokers",
        json={
            "broker_type": "alpaca",
            "name": "Live Alpaca",
            "api_key": "PKTEST12345678901234",
            "api_secret": "abcdefghijklmnopqrstuvwxyz123456789012345678",
            "base_url": "https://api.alpaca.markets",
        },
    )

    assert response.status_code == 201
    data = response.json()
    assert data["base_url"] == "https://api.alpaca.markets"


def test_add_broker_invalid_type(client):
    """Test adding broker with invalid type"""
    response = client.post(
        "/api/brokers",
        json={
            "broker_type": "invalid_broker",
            "name": "Invalid Broker",
            "api_key": "PKTEST12345678901234",
            "api_secret": "abcdefghijklmnopqrstuvwxyz123456789012345678",
        },
    )

    assert response.status_code == 400
    assert "Unknown broker type" in response.json()["detail"]


def test_add_broker_missing_api_key(client):
    """Test adding broker without API key"""
    response = client.post(
        "/api/brokers",
        json={
            "broker_type": "alpaca",
            "name": "Missing Key",
            "api_secret": "abcdefghijklmnopqrstuvwxyz123456789012345678",
        },
    )

    assert response.status_code == 422  # Pydantic validation error


def test_add_broker_missing_api_secret(client):
    """Test adding broker without API secret"""
    response = client.post(
        "/api/brokers",
        json={
            "broker_type": "alpaca",
            "name": "Missing Secret",
            "api_key": "PKTEST12345678901234",
        },
    )

    assert response.status_code == 422  # Pydantic validation error


def test_add_broker_empty_name(client):
    """Test adding broker with empty name"""
    response = client.post(
        "/api/brokers",
        json={
            "broker_type": "alpaca",
            "name": "",
            "api_key": "PKTEST12345678901234",
            "api_secret": "abcdefghijklmnopqrstuvwxyz123456789012345678",
        },
    )

    assert response.status_code == 422  # Pydantic validation error


def test_add_broker_empty_api_key(client):
    """Test adding broker with empty API key"""
    response = client.post(
        "/api/brokers",
        json={
            "broker_type": "alpaca",
            "name": "Test",
            "api_key": "",
            "api_secret": "abcdefghijklmnopqrstuvwxyz123456789012345678",
        },
    )

    assert response.status_code == 422  # Pydantic validation error


# ============================================================================
# Delete Broker Tests (DELETE /api/brokers/{id})
# ============================================================================


def test_delete_broker_success(client):
    """Test deleting a broker configuration"""
    # Add a broker first
    add_response = client.post(
        "/api/brokers",
        json={
            "broker_type": "alpaca",
            "name": "To Delete",
            "api_key": "PKTEST12345678901234",
            "api_secret": "abcdefghijklmnopqrstuvwxyz123456789012345678",
        },
    )
    broker_id = add_response.json()["id"]

    # Delete it
    delete_response = client.delete(f"/api/brokers/{broker_id}")
    assert delete_response.status_code == 204

    # Verify it's gone
    list_response = client.get("/api/brokers")
    assert list_response.json()["total"] == 0


def test_delete_broker_not_found(client):
    """Test deleting non-existent broker"""
    response = client.delete("/api/brokers/nonexistent")
    assert response.status_code == 404
    assert "Broker not found" in response.json()["detail"]


def test_delete_broker_idempotent(client):
    """Test deleting same broker twice returns 404 on second attempt"""
    # Add a broker
    add_response = client.post(
        "/api/brokers",
        json={
            "broker_type": "alpaca",
            "name": "Delete Twice",
            "api_key": "PKTEST12345678901234",
            "api_secret": "abcdefghijklmnopqrstuvwxyz123456789012345678",
        },
    )
    broker_id = add_response.json()["id"]

    # Delete it twice
    first_delete = client.delete(f"/api/brokers/{broker_id}")
    assert first_delete.status_code == 204

    second_delete = client.delete(f"/api/brokers/{broker_id}")
    assert second_delete.status_code == 404


# ============================================================================
# Health Check Tests (GET /api/brokers/{id}/health)
# ============================================================================


def test_broker_health_not_found(client):
    """Test health check for non-existent broker"""
    response = client.get("/api/brokers/nonexistent/health")
    assert response.status_code == 404
    assert "Broker not found" in response.json()["detail"]


@patch("backend.execution.broker_factory.BrokerFactory")
def test_broker_health_success(mock_factory_class, client):
    """Test successful health check"""
    # Add a broker first
    add_response = client.post(
        "/api/brokers",
        json={
            "broker_type": "alpaca",
            "name": "Health Check Test",
            "api_key": "PKTEST12345678901234",
            "api_secret": "abcdefghijklmnopqrstuvwxyz123456789012345678",
        },
    )
    broker_id = add_response.json()["id"]

    # Mock the broker and health check
    mock_broker = MagicMock()
    mock_health = MagicMock()
    mock_health.is_connected = True
    mock_health.is_authenticated = True
    mock_health.latency_ms = 50.0
    mock_health.last_heartbeat = datetime.now().timestamp()
    mock_health.error_message = None

    mock_broker.health_check = AsyncMock(return_value=mock_health)
    mock_broker.disconnect = AsyncMock()

    mock_factory = MagicMock()
    mock_factory.create_broker.return_value = mock_broker
    mock_factory_class.return_value = mock_factory

    # Check health
    response = client.get(f"/api/brokers/{broker_id}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == broker_id
    assert data["is_connected"] is True
    assert data["is_authenticated"] is True
    assert data["latency_ms"] == 50.0
    assert data["error_message"] is None


@patch("backend.execution.broker_factory.BrokerFactory")
def test_broker_health_connection_failed(mock_factory_class, client):
    """Test health check when connection fails"""
    # Add a broker
    add_response = client.post(
        "/api/brokers",
        json={
            "broker_type": "alpaca",
            "name": "Failed Connection",
            "api_key": "PKTEST12345678901234",
            "api_secret": "abcdefghijklmnopqrstuvwxyz123456789012345678",
        },
    )
    broker_id = add_response.json()["id"]

    # Mock connection failure
    mock_broker = MagicMock()
    mock_health = MagicMock()
    mock_health.is_connected = False
    mock_health.is_authenticated = False
    mock_health.latency_ms = None
    mock_health.last_heartbeat = None
    mock_health.error_message = "Connection refused"

    mock_broker.health_check = AsyncMock(return_value=mock_health)
    mock_broker.disconnect = AsyncMock()

    mock_factory = MagicMock()
    mock_factory.create_broker.return_value = mock_broker
    mock_factory_class.return_value = mock_factory

    response = client.get(f"/api/brokers/{broker_id}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["is_connected"] is False
    assert data["error_message"] == "Connection refused"


@patch("backend.execution.broker_factory.BrokerFactory")
def test_broker_health_exception_handling(mock_factory_class, client):
    """Test health check handles exceptions gracefully"""
    # Add a broker
    add_response = client.post(
        "/api/brokers",
        json={
            "broker_type": "alpaca",
            "name": "Exception Test",
            "api_key": "PKTEST12345678901234",
            "api_secret": "abcdefghijklmnopqrstuvwxyz123456789012345678",
        },
    )
    broker_id = add_response.json()["id"]

    # Mock exception
    mock_broker = MagicMock()
    mock_broker.health_check = AsyncMock(side_effect=Exception("Network error"))
    mock_broker.disconnect = AsyncMock()

    mock_factory = MagicMock()
    mock_factory.create_broker.return_value = mock_broker
    mock_factory_class.return_value = mock_factory

    response = client.get(f"/api/brokers/{broker_id}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["is_connected"] is False
    assert "Health check failed" in data["error_message"]
    assert "Network error" in data["error_message"]


# ============================================================================
# Credential Encryption Tests
# ============================================================================


def test_credentials_are_encrypted_in_storage(client):
    """Test that credentials are stored encrypted, not in plaintext"""
    # Add a broker
    api_key = "PKTEST12345678901234"
    api_secret = "abcdefghijklmnopqrstuvwxyz123456789012345678"

    client.post(
        "/api/brokers",
        json={
            "broker_type": "alpaca",
            "name": "Encryption Test",
            "api_key": api_key,
            "api_secret": api_secret,
        },
    )

    # Check storage directly via app_state
    # Note: get_all_settings is a sync read, so we can use asyncio.run
    all_settings = asyncio.run(app_state.sqlite_store.get_all_settings())

    # Find the broker config
    broker_key = None
    for key in all_settings:
        if key.startswith("broker_config:"):
            broker_key = key
            break

    assert broker_key is not None
    config_json = all_settings[broker_key]

    # Plaintext credentials should NOT appear in storage
    assert api_key not in config_json
    assert api_secret not in config_json

    # Encrypted fields should exist
    assert "api_key_encrypted" in config_json
    assert "api_secret_encrypted" in config_json


def test_credentials_masked_in_response(client):
    """Test that credentials are masked in API responses"""
    api_key = "PKTEST12345678901234"

    response = client.post(
        "/api/brokers",
        json={
            "broker_type": "alpaca",
            "name": "Masking Test",
            "api_key": api_key,
            "api_secret": "abcdefghijklmnopqrstuvwxyz123456789012345678",
        },
    )

    data = response.json()

    # Full API key should not appear
    assert api_key not in str(data)
    # Masked version should appear
    assert "****" in data["api_key_masked"]


# ============================================================================
# Database Connection Tests
# ============================================================================


def test_list_brokers_no_db(client):
    """Test list brokers returns 503 when database unavailable"""
    original = app_state.sqlite_store
    app_state.sqlite_store = None

    response = client.get("/api/brokers")
    assert response.status_code == 503
    assert "Database not initialized" in response.json()["detail"]

    app_state.sqlite_store = original


def test_add_broker_no_db(client):
    """Test add broker returns 503 when database unavailable"""
    original = app_state.sqlite_store
    app_state.sqlite_store = None

    response = client.post(
        "/api/brokers",
        json={
            "broker_type": "alpaca",
            "name": "Test",
            "api_key": "PKTEST12345678901234",
            "api_secret": "abcdefghijklmnopqrstuvwxyz123456789012345678",
        },
    )
    assert response.status_code == 503

    app_state.sqlite_store = original


def test_delete_broker_no_db(client):
    """Test delete broker returns 503 when database unavailable"""
    original = app_state.sqlite_store
    app_state.sqlite_store = None

    response = client.delete("/api/brokers/test-id")
    assert response.status_code == 503

    app_state.sqlite_store = original


def test_health_check_no_db(client):
    """Test health check returns 503 when database unavailable"""
    original = app_state.sqlite_store
    app_state.sqlite_store = None

    response = client.get("/api/brokers/test-id/health")
    assert response.status_code == 503

    app_state.sqlite_store = original


# ============================================================================
# Response Model Tests
# ============================================================================


def test_broker_config_response_model():
    """Test BrokerConfigResponse model structure"""
    from backend.api.server import BrokerConfigResponse

    response = BrokerConfigResponse(
        id="abc12345",
        broker_type="alpaca",
        name="Test Broker",
        api_key_masked="PKTE****",
        base_url="https://paper-api.alpaca.markets",
        is_connected=False,
        created_at="2024-01-01T00:00:00",
        updated_at="2024-01-01T00:00:00",
    )

    assert response.id == "abc12345"
    assert response.broker_type == "alpaca"
    assert response.api_key_masked == "PKTE****"


def test_broker_list_response_model():
    """Test BrokerListResponse model structure"""
    from backend.api.server import BrokerConfigResponse, BrokerListResponse

    broker = BrokerConfigResponse(
        id="abc12345",
        broker_type="alpaca",
        name="Test Broker",
        api_key_masked="PKTE****",
        base_url="https://paper-api.alpaca.markets",
        is_connected=False,
        created_at="2024-01-01T00:00:00",
        updated_at="2024-01-01T00:00:00",
    )

    response = BrokerListResponse(brokers=[broker], total=1)
    assert response.total == 1
    assert len(response.brokers) == 1


def test_broker_health_response_model():
    """Test BrokerHealthResponse model structure"""
    from backend.api.server import BrokerHealthResponse

    response = BrokerHealthResponse(
        id="abc12345",
        name="Test Broker",
        broker_type="alpaca",
        is_connected=True,
        is_authenticated=True,
        latency_ms=50.0,
        last_heartbeat="2024-01-01T00:00:00",
        error_message=None,
    )

    assert response.is_connected is True
    assert response.latency_ms == 50.0


def test_broker_config_request_model():
    """Test BrokerConfigRequest model validation"""
    from backend.api.server import BrokerConfigRequest

    # Valid request
    request = BrokerConfigRequest(
        broker_type="alpaca",
        name="Test",
        api_key="PKTEST123",
        api_secret="secret123",
    )
    assert request.broker_type == "alpaca"
    assert request.base_url is None  # Optional field


# ============================================================================
# Edge Cases
# ============================================================================


def test_add_broker_long_name(client):
    """Test adding broker with maximum length name"""
    response = client.post(
        "/api/brokers",
        json={
            "broker_type": "alpaca",
            "name": "A" * 100,  # Max length
            "api_key": "PKTEST12345678901234",
            "api_secret": "abcdefghijklmnopqrstuvwxyz123456789012345678",
        },
    )
    assert response.status_code == 201


def test_add_broker_name_too_long(client):
    """Test adding broker with name exceeding max length"""
    response = client.post(
        "/api/brokers",
        json={
            "broker_type": "alpaca",
            "name": "A" * 101,  # Over max length
            "api_key": "PKTEST12345678901234",
            "api_secret": "abcdefghijklmnopqrstuvwxyz123456789012345678",
        },
    )
    assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
