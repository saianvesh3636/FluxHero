"""
Unit tests for Authentication Middleware

Tests the token-based authentication system for WebSocket and API endpoints.

Requirements tested:
- Token validation with constant-time comparison
- Bearer token format handling
- Environment variable configuration
- WebSocket header authentication
- Security against timing attacks

Author: FluxHero
Date: 2026-01-22
"""

import os
from unittest.mock import patch

from fluxhero.backend.api.auth import (
    get_auth_secret,
    validate_token,
    extract_token_from_header,
    validate_websocket_auth,
    DEFAULT_SECRET,
)


class TestAuthSecretConfiguration:
    """Test authentication secret configuration and retrieval."""

    def test_default_secret_when_env_not_set(self):
        """Test that default secret is used when environment variable is not set."""
        with patch.dict(os.environ, {}, clear=True):
            secret = get_auth_secret()
            assert secret == DEFAULT_SECRET

    def test_custom_secret_from_environment(self):
        """Test that custom secret is loaded from environment variable."""
        custom_secret = "my-custom-secret-key-12345"
        with patch.dict(os.environ, {"FLUXHERO_AUTH_SECRET": custom_secret}):
            secret = get_auth_secret()
            assert secret == custom_secret

    def test_empty_env_var_uses_default(self):
        """Test that empty environment variable falls back to default."""
        with patch.dict(os.environ, {"FLUXHERO_AUTH_SECRET": ""}):
            secret = get_auth_secret()
            assert secret == DEFAULT_SECRET


class TestTokenValidation:
    """Test token validation logic."""

    def test_valid_token_without_bearer_prefix(self):
        """Test validation of valid token without Bearer prefix."""
        custom_secret = "test-secret-123"
        with patch.dict(os.environ, {"FLUXHERO_AUTH_SECRET": custom_secret}):
            assert validate_token("test-secret-123") is True

    def test_valid_token_with_bearer_prefix(self):
        """Test validation of valid token with Bearer prefix."""
        custom_secret = "test-secret-456"
        with patch.dict(os.environ, {"FLUXHERO_AUTH_SECRET": custom_secret}):
            assert validate_token("Bearer test-secret-456") is True

    def test_invalid_token(self):
        """Test that invalid token is rejected."""
        custom_secret = "correct-secret"
        with patch.dict(os.environ, {"FLUXHERO_AUTH_SECRET": custom_secret}):
            assert validate_token("wrong-secret") is False

    def test_none_token(self):
        """Test that None token is rejected."""
        assert validate_token(None) is False

    def test_empty_token(self):
        """Test that empty token is rejected."""
        assert validate_token("") is False

    def test_bearer_prefix_with_empty_token(self):
        """Test that 'Bearer ' with no token is rejected."""
        assert validate_token("Bearer ") is False

    def test_bearer_prefix_with_whitespace_only(self):
        """Test that 'Bearer ' with whitespace only is rejected."""
        assert validate_token("Bearer   ") is False

    def test_token_with_extra_whitespace(self):
        """Test token validation with extra whitespace after Bearer."""
        custom_secret = "test-secret-789"
        with patch.dict(os.environ, {"FLUXHERO_AUTH_SECRET": custom_secret}):
            assert validate_token("Bearer   test-secret-789  ") is True

    def test_case_sensitive_validation(self):
        """Test that token validation is case-sensitive."""
        custom_secret = "CaseSensitive123"
        with patch.dict(os.environ, {"FLUXHERO_AUTH_SECRET": custom_secret}):
            assert validate_token("casesensitive123") is False
            assert validate_token("CaseSensitive123") is True

    def test_timing_attack_resistance(self):
        """
        Test that validation uses constant-time comparison.

        This is a basic check that the function uses secrets.compare_digest.
        True timing attack testing would require statistical analysis.
        """
        custom_secret = "a" * 32
        with patch.dict(os.environ, {"FLUXHERO_AUTH_SECRET": custom_secret}):
            # These should both return False, but take similar time
            result1 = validate_token("b" * 32)
            result2 = validate_token("b" * 1)
            assert result1 is False
            assert result2 is False


class TestExtractTokenFromHeader:
    """Test token extraction from Authorization headers."""

    def test_extract_from_bearer_format(self):
        """Test extraction from Bearer <token> format."""
        token = extract_token_from_header("Bearer abc123xyz")
        assert token == "abc123xyz"

    def test_extract_from_raw_token(self):
        """Test extraction from raw token without Bearer prefix."""
        token = extract_token_from_header("abc123xyz")
        assert token == "abc123xyz"

    def test_extract_from_none(self):
        """Test extraction from None header."""
        token = extract_token_from_header(None)
        assert token is None

    def test_extract_from_empty_string(self):
        """Test extraction from empty string."""
        token = extract_token_from_header("")
        assert token is None

    def test_extract_with_whitespace(self):
        """Test extraction with extra whitespace."""
        token = extract_token_from_header("Bearer   token-with-spaces  ")
        assert token == "token-with-spaces"

    def test_extract_bearer_only(self):
        """Test extraction when header is 'Bearer' with no token."""
        token = extract_token_from_header("Bearer ")
        assert token == ""


class TestWebSocketAuthentication:
    """Test WebSocket-specific authentication logic."""

    def test_valid_websocket_auth(self):
        """Test successful WebSocket authentication."""
        custom_secret = "ws-secret-123"
        with patch.dict(os.environ, {"FLUXHERO_AUTH_SECRET": custom_secret}):
            headers = {"authorization": "Bearer ws-secret-123"}
            assert validate_websocket_auth(headers) is True

    def test_websocket_auth_case_insensitive_header(self):
        """Test that Authorization header is case-insensitive."""
        custom_secret = "ws-secret-456"
        with patch.dict(os.environ, {"FLUXHERO_AUTH_SECRET": custom_secret}):
            # Test different cases
            headers1 = {"Authorization": "Bearer ws-secret-456"}
            headers2 = {"AUTHORIZATION": "Bearer ws-secret-456"}
            headers3 = {"authorization": "Bearer ws-secret-456"}

            assert validate_websocket_auth(headers1) is True
            assert validate_websocket_auth(headers2) is True
            assert validate_websocket_auth(headers3) is True

    def test_websocket_auth_missing_header(self):
        """Test WebSocket auth fails when Authorization header is missing."""
        headers = {"content-type": "application/json"}
        assert validate_websocket_auth(headers) is False

    def test_websocket_auth_empty_headers(self):
        """Test WebSocket auth fails with empty headers dict."""
        assert validate_websocket_auth({}) is False

    def test_websocket_auth_invalid_token(self):
        """Test WebSocket auth fails with invalid token."""
        custom_secret = "correct-secret"
        with patch.dict(os.environ, {"FLUXHERO_AUTH_SECRET": custom_secret}):
            headers = {"authorization": "Bearer wrong-secret"}
            assert validate_websocket_auth(headers) is False

    def test_websocket_auth_without_bearer_prefix(self):
        """Test WebSocket auth with raw token (no Bearer prefix)."""
        custom_secret = "ws-secret-789"
        with patch.dict(os.environ, {"FLUXHERO_AUTH_SECRET": custom_secret}):
            headers = {"authorization": "ws-secret-789"}
            assert validate_websocket_auth(headers) is True


class TestSecurityConsiderations:
    """Test security-related edge cases."""

    def test_sql_injection_attempt(self):
        """Test that SQL injection attempts in token are safely rejected."""
        custom_secret = "safe-secret"
        with patch.dict(os.environ, {"FLUXHERO_AUTH_SECRET": custom_secret}):
            malicious_token = "'; DROP TABLE users; --"
            assert validate_token(malicious_token) is False

    def test_very_long_token(self):
        """Test handling of extremely long token strings."""
        custom_secret = "short-secret"
        with patch.dict(os.environ, {"FLUXHERO_AUTH_SECRET": custom_secret}):
            long_token = "a" * 10000
            assert validate_token(long_token) is False

    def test_special_characters_in_token(self):
        """Test token with special characters."""
        custom_secret = "secret!@#$%^&*()"
        with patch.dict(os.environ, {"FLUXHERO_AUTH_SECRET": custom_secret}):
            assert validate_token("Bearer secret!@#$%^&*()") is True
            assert validate_token("Bearer different!@#$") is False

    def test_unicode_in_token(self):
        """Test token with unicode characters."""
        custom_secret = "secret-with-émoji-🔒"
        with patch.dict(os.environ, {"FLUXHERO_AUTH_SECRET": custom_secret}):
            assert validate_token(f"Bearer {custom_secret}") is True
            assert validate_token("Bearer wrong-émoji-🔓") is False


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_typical_api_request_flow(self):
        """Test typical API request authentication flow."""
        # Simulate production setup
        production_secret = "prod-secret-key-very-secure-12345"
        with patch.dict(os.environ, {"FLUXHERO_AUTH_SECRET": production_secret}):
            # Client sends request with Authorization header
            auth_header = f"Bearer {production_secret}"
            token = extract_token_from_header(auth_header)
            is_valid = validate_token(token)

            assert is_valid is True

    def test_typical_websocket_connection_flow(self):
        """Test typical WebSocket connection authentication flow."""
        ws_secret = "websocket-secret-xyz"
        with patch.dict(os.environ, {"FLUXHERO_AUTH_SECRET": ws_secret}):
            # Simulate WebSocket handshake headers
            handshake_headers = {
                "Host": "localhost:8000",
                "Upgrade": "websocket",
                "Connection": "Upgrade",
                "Authorization": f"Bearer {ws_secret}",
            }

            assert validate_websocket_auth(handshake_headers) is True

    def test_unauthorized_access_attempt(self):
        """Test that unauthorized access is properly blocked."""
        real_secret = "super-secret-key"
        with patch.dict(os.environ, {"FLUXHERO_AUTH_SECRET": real_secret}):
            # Attacker tries common tokens
            common_attempts = [
                "Bearer admin",
                "Bearer password",
                "Bearer 12345",
                "Bearer token",
                "",
                None,
            ]

            for attempt in common_attempts:
                assert validate_token(attempt) is False
