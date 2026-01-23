"""
Broker Credential Encryption - Secure Storage for API Keys and Secrets

This module provides AES-256-GCM encryption for sensitive broker credentials.
It uses the cryptography library to implement authenticated encryption,
ensuring both confidentiality and integrity of stored credentials.

Feature: Multi-Broker Architecture (Phase A)
"""

import base64
import os
import secrets
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class CredentialEncryptionError(Exception):
    """Raised when encryption or decryption fails."""

    pass


def _get_encryption_key() -> bytes:
    """
    Get the encryption key from environment configuration.

    The key is loaded from FLUXHERO_ENCRYPTION_KEY environment variable.
    If not set, raises an error in production or uses a dev key for development.

    Returns:
        32-byte encryption key for AES-256

    Raises:
        CredentialEncryptionError: If key is missing in production or invalid
    """
    from backend.core.config import get_settings

    settings = get_settings()

    # Check for encryption key in settings
    encryption_key = getattr(settings, "encryption_key", None)

    if encryption_key is None or encryption_key == "":
        # Check environment directly as fallback
        encryption_key = os.environ.get("FLUXHERO_ENCRYPTION_KEY", "")

    if not encryption_key:
        # Check if we're in production
        env = os.environ.get("FLUXHERO_ENV", "development")
        if env == "production":
            raise CredentialEncryptionError(
                "FLUXHERO_ENCRYPTION_KEY must be set in production environment. "
                "Generate a key with: python -c 'import secrets; print(secrets.token_hex(32))'"
            )
        # Use a development-only key (32 bytes = 64 hex chars)
        logger.warning(
            "Using development encryption key. Set FLUXHERO_ENCRYPTION_KEY for production."
        )
        encryption_key = "fluxhero_dev_encryption_key_32b!"  # Exactly 32 bytes

    # Convert to bytes and ensure correct length
    if isinstance(encryption_key, str):
        key_bytes = encryption_key.encode("utf-8")
    else:
        key_bytes = encryption_key

    # Pad or truncate to exactly 32 bytes for AES-256
    if len(key_bytes) < 32:
        key_bytes = key_bytes.ljust(32, b"\0")
    elif len(key_bytes) > 32:
        key_bytes = key_bytes[:32]

    return key_bytes


def _get_aesgcm() -> "AESGCM":
    """
    Get an AESGCM cipher instance with the configured encryption key.

    Returns:
        AESGCM instance for encryption/decryption

    Raises:
        CredentialEncryptionError: If cryptography library not installed
    """
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as e:
        raise CredentialEncryptionError(
            "cryptography library not installed. Install with: uv add cryptography"
        ) from e

    key = _get_encryption_key()
    return AESGCM(key)


def encrypt_credential(plaintext: str) -> str:
    """
    Encrypt a credential using AES-256-GCM.

    The encrypted output format is: base64(nonce + ciphertext + tag)
    - nonce: 12 bytes (96 bits) random
    - ciphertext: variable length
    - tag: 16 bytes (128 bits) authentication tag

    Args:
        plaintext: The credential to encrypt (e.g., API key or secret)

    Returns:
        Base64-encoded encrypted credential string

    Raises:
        CredentialEncryptionError: If encryption fails
    """
    if not plaintext:
        raise CredentialEncryptionError("Cannot encrypt empty credential")

    try:
        aesgcm = _get_aesgcm()

        # Generate a random 12-byte nonce (recommended for GCM)
        nonce = secrets.token_bytes(12)

        # Encrypt the plaintext
        plaintext_bytes = plaintext.encode("utf-8")
        ciphertext = aesgcm.encrypt(nonce, plaintext_bytes, associated_data=None)

        # Combine nonce + ciphertext and encode as base64
        encrypted_data = nonce + ciphertext
        encrypted_b64 = base64.b64encode(encrypted_data).decode("utf-8")

        logger.debug("Credential encrypted successfully (length: {} chars)", len(plaintext))
        return encrypted_b64

    except CredentialEncryptionError:
        raise
    except Exception as e:
        # Never log the plaintext value
        raise CredentialEncryptionError(f"Encryption failed: {type(e).__name__}") from e


def decrypt_credential(encrypted: str) -> str:
    """
    Decrypt a credential that was encrypted with encrypt_credential().

    Args:
        encrypted: Base64-encoded encrypted credential string

    Returns:
        Decrypted plaintext credential

    Raises:
        CredentialEncryptionError: If decryption fails (invalid data, wrong key, or tampered)
    """
    if not encrypted:
        raise CredentialEncryptionError("Cannot decrypt empty credential")

    try:
        aesgcm = _get_aesgcm()

        # Decode from base64
        encrypted_data = base64.b64decode(encrypted)

        # Extract nonce (first 12 bytes) and ciphertext
        if len(encrypted_data) < 12 + 16:  # nonce + minimum tag
            raise CredentialEncryptionError("Invalid encrypted data: too short")

        nonce = encrypted_data[:12]
        ciphertext = encrypted_data[12:]

        # Decrypt and authenticate
        plaintext_bytes = aesgcm.decrypt(nonce, ciphertext, associated_data=None)
        plaintext = plaintext_bytes.decode("utf-8")

        # Never log the decrypted value
        logger.debug("Credential decrypted successfully")
        return plaintext

    except CredentialEncryptionError:
        raise
    except Exception as e:
        # Could be InvalidTag (tampered/wrong key) or other errors
        # Never include the encrypted data in error messages
        raise CredentialEncryptionError(
            f"Decryption failed: {type(e).__name__}. "
            "Check that FLUXHERO_ENCRYPTION_KEY matches the key used for encryption."
        ) from e


def generate_encryption_key() -> str:
    """
    Generate a secure random encryption key.

    This utility function generates a 32-byte (256-bit) random key
    encoded as a hexadecimal string suitable for FLUXHERO_ENCRYPTION_KEY.

    Returns:
        64-character hexadecimal string (32 bytes)
    """
    return secrets.token_hex(32)


def is_encrypted(value: str) -> bool:
    """
    Check if a value appears to be an encrypted credential.

    This is a heuristic check based on the expected format.
    The encrypted format is base64-encoded and should be at least
    28 bytes (12 nonce + 16 tag) when decoded.

    Args:
        value: String to check

    Returns:
        True if value appears to be encrypted, False otherwise
    """
    if not value:
        return False

    try:
        decoded = base64.b64decode(value)
        # Minimum length: 12 (nonce) + 16 (tag) = 28 bytes
        return len(decoded) >= 28
    except Exception:
        return False


def mask_credential(credential: str, visible_chars: int = 4) -> str:
    """
    Mask a credential for safe logging/display.

    Shows only the first few characters, masking the rest with asterisks.
    Useful for displaying API keys in logs or UI without exposing full values.

    Args:
        credential: The credential to mask
        visible_chars: Number of characters to show at start (default: 4)

    Returns:
        Masked credential string (e.g., "PKAJ****" for "PKAJ1234567890")
    """
    if not credential:
        return "****"

    if len(credential) <= visible_chars:
        return "*" * len(credential)

    return credential[:visible_chars] + "*" * (len(credential) - visible_chars)
