"""
Authentication Middleware for FluxHero API

This module provides token-based authentication for WebSocket and API endpoints.
It validates bearer tokens against a configurable secret key.

Usage:
    from backend.api.auth import validate_token, get_auth_secret

    # Validate a token from request headers
    is_valid = validate_token(token)

    # In WebSocket endpoints:
    token = websocket.headers.get("Authorization")
    if not validate_token(token):
        await websocket.close(code=4001)
        return

Security:
    - Tokens are validated using constant-time comparison to prevent timing attacks
    - Secret key should be set via FLUXHERO_AUTH_SECRET environment variable
    - Default secret is insecure and should only be used for development

Author: FluxHero
Date: 2026-01-22
"""

import os
import secrets

from backend.core.logging_config import get_logger

logger = get_logger(__name__)

# Default insecure secret for development only
DEFAULT_SECRET = "fluxhero-dev-secret-change-in-production"


def get_auth_secret() -> str:
    """
    Get the authentication secret key from environment or use default.

    The secret is loaded from FLUXHERO_AUTH_SECRET environment variable.
    If not set or empty, a default development secret is used (insecure for production).

    Returns:
        Authentication secret key string

    Warnings:
        Logs a warning if using the default development secret
    """
    secret = os.environ.get("FLUXHERO_AUTH_SECRET", DEFAULT_SECRET)

    # Use default if environment variable is empty
    if not secret:
        secret = DEFAULT_SECRET

    if secret == DEFAULT_SECRET:
        logger.warning(
            "Using default authentication secret - not secure for production",
            extra={"env_var": "FLUXHERO_AUTH_SECRET"}
        )

    return secret


def validate_token(token: str | None) -> bool:
    """
    Validate an authentication token.

    This function performs token validation using constant-time comparison
    to prevent timing attacks. The token should be provided as a bearer token.

    Args:
        token: The token to validate. Can be:
               - "Bearer <token>" format (extracts token part)
               - Raw token string
               - None (returns False)

    Returns:
        True if token is valid, False otherwise

    Examples:
        >>> validate_token("Bearer my-secret-token")
        True  # If my-secret-token matches FLUXHERO_AUTH_SECRET

        >>> validate_token("invalid-token")
        False

        >>> validate_token(None)
        False

    Security:
        Uses secrets.compare_digest() for constant-time comparison
        to prevent timing-based attacks
    """
    if not token:
        logger.debug("Token validation failed: no token provided")
        return False

    # Extract token from "Bearer <token>" format if present
    if token.startswith("Bearer "):
        token = token[7:].strip()

    if not token:
        logger.debug("Token validation failed: empty token after Bearer prefix")
        return False

    # Get expected secret
    expected_secret = get_auth_secret()

    # Use constant-time comparison to prevent timing attacks
    # Note: secrets.compare_digest requires ASCII-only strings or bytes
    # For Unicode support, convert to bytes using UTF-8 encoding
    try:
        token_bytes = token.encode('utf-8')
        secret_bytes = expected_secret.encode('utf-8')
        is_valid = secrets.compare_digest(token_bytes, secret_bytes)
    except (UnicodeEncodeError, AttributeError) as e:
        logger.warning(
            "Token validation failed: encoding error",
            extra={"error": str(e)}
        )
        return False

    if not is_valid:
        logger.warning(
            "Token validation failed: invalid token",
            extra={"token_length": len(token)}
        )
    else:
        logger.debug("Token validation successful")

    return is_valid


def extract_token_from_header(auth_header: str | None) -> str | None:
    """
    Extract token from Authorization header.

    Handles both "Bearer <token>" format and raw token strings.

    Args:
        auth_header: The Authorization header value

    Returns:
        Extracted token string or None if header is invalid

    Examples:
        >>> extract_token_from_header("Bearer abc123")
        "abc123"

        >>> extract_token_from_header("abc123")
        "abc123"

        >>> extract_token_from_header(None)
        None
    """
    if not auth_header:
        return None

    # Remove "Bearer " prefix if present
    if auth_header.startswith("Bearer "):
        return auth_header[7:].strip()

    return auth_header.strip()


def validate_websocket_auth(headers: dict) -> bool:
    """
    Validate authentication for WebSocket connections.

    Checks the Authorization header from WebSocket handshake headers.

    Args:
        headers: Dictionary of WebSocket headers (case-insensitive)

    Returns:
        True if authentication is valid, False otherwise

    Examples:
        >>> headers = {"authorization": "Bearer my-token"}
        >>> validate_websocket_auth(headers)
        True  # If my-token is valid
    """
    # WebSocket headers are case-insensitive
    auth_header = None
    for key, value in headers.items():
        if key.lower() == "authorization":
            auth_header = value
            break

    if not auth_header:
        logger.debug("WebSocket auth failed: no Authorization header")
        return False

    return validate_token(auth_header)
