"""
Unit tests for broker credential encryption.

Tests cover:
- Encryption and decryption round-trip
- Error handling for invalid inputs
- Encryption key management
- Utility functions (is_encrypted, mask_credential, generate_encryption_key)
"""

import base64
import os

import pytest

from backend.execution.broker_credentials import (
    CredentialEncryptionError,
    decrypt_credential,
    encrypt_credential,
    generate_encryption_key,
    is_encrypted,
    mask_credential,
)


class TestEncryptDecryptRoundTrip:
    """Tests for encrypt/decrypt round-trip functionality."""

    def test_encrypt_decrypt_api_key(self):
        """Test encrypting and decrypting an API key."""
        original = "PKAJ1234567890ABCDEF"

        encrypted = encrypt_credential(original)
        decrypted = decrypt_credential(encrypted)

        assert decrypted == original

    def test_encrypt_decrypt_api_secret(self):
        """Test encrypting and decrypting an API secret."""
        original = "abcdefghijklmnopqrstuvwxyz0123456789ABCD"

        encrypted = encrypt_credential(original)
        decrypted = decrypt_credential(encrypted)

        assert decrypted == original

    def test_encrypt_decrypt_special_characters(self):
        """Test encrypting credentials with special characters."""
        original = "key!@#$%^&*()_+-=[]{}|;':\",./<>?"

        encrypted = encrypt_credential(original)
        decrypted = decrypt_credential(encrypted)

        assert decrypted == original

    def test_encrypt_decrypt_unicode(self):
        """Test encrypting credentials with unicode characters."""
        original = "key_\u00e9\u00e8\u00ea_\u4e2d\u6587"

        encrypted = encrypt_credential(original)
        decrypted = decrypt_credential(encrypted)

        assert decrypted == original

    def test_encrypt_decrypt_long_credential(self):
        """Test encrypting a long credential."""
        original = "x" * 1000

        encrypted = encrypt_credential(original)
        decrypted = decrypt_credential(encrypted)

        assert decrypted == original

    def test_encrypt_decrypt_short_credential(self):
        """Test encrypting a short credential."""
        original = "a"

        encrypted = encrypt_credential(original)
        decrypted = decrypt_credential(encrypted)

        assert decrypted == original

    def test_encrypted_is_different_from_original(self):
        """Test that encrypted value is different from original."""
        original = "test_api_key"

        encrypted = encrypt_credential(original)

        assert encrypted != original
        assert original not in encrypted

    def test_encrypt_produces_different_ciphertext_each_time(self):
        """Test that encrypting the same value produces different ciphertexts (random nonce)."""
        original = "test_api_key"

        encrypted1 = encrypt_credential(original)
        encrypted2 = encrypt_credential(original)

        # Should be different due to random nonce
        assert encrypted1 != encrypted2

        # But both should decrypt to the same value
        assert decrypt_credential(encrypted1) == original
        assert decrypt_credential(encrypted2) == original


class TestEncryptionErrors:
    """Tests for encryption error handling."""

    def test_encrypt_empty_string_raises(self):
        """Test encrypting empty string raises error."""
        with pytest.raises(CredentialEncryptionError, match="Cannot encrypt empty"):
            encrypt_credential("")

    def test_decrypt_empty_string_raises(self):
        """Test decrypting empty string raises error."""
        with pytest.raises(CredentialEncryptionError, match="Cannot decrypt empty"):
            decrypt_credential("")

    def test_decrypt_invalid_base64_raises(self):
        """Test decrypting invalid base64 raises error."""
        with pytest.raises(CredentialEncryptionError):
            decrypt_credential("not-valid-base64!!!")

    def test_decrypt_too_short_raises(self):
        """Test decrypting data that's too short raises error."""
        # Less than 28 bytes (12 nonce + 16 tag)
        short_data = base64.b64encode(b"short").decode()

        with pytest.raises(CredentialEncryptionError, match="too short"):
            decrypt_credential(short_data)

    def test_decrypt_tampered_data_raises(self):
        """Test decrypting tampered data raises error."""
        original = "test_api_key"
        encrypted = encrypt_credential(original)

        # Decode, tamper with the ciphertext, re-encode
        encrypted_bytes = base64.b64decode(encrypted)
        tampered_bytes = encrypted_bytes[:-1] + bytes([encrypted_bytes[-1] ^ 0xFF])
        tampered = base64.b64encode(tampered_bytes).decode()

        with pytest.raises(CredentialEncryptionError, match="Decryption failed"):
            decrypt_credential(tampered)


class TestEncryptionKey:
    """Tests for encryption key management."""

    def test_development_mode_uses_dev_key(self, monkeypatch):
        """Test development mode uses a development key."""
        # Clear any existing encryption key
        monkeypatch.delenv("FLUXHERO_ENCRYPTION_KEY", raising=False)
        monkeypatch.setenv("FLUXHERO_ENV", "development")

        # Clear lru_cache
        from backend.core.config import get_settings

        get_settings.cache_clear()

        # Should work with dev key
        encrypted = encrypt_credential("test")
        decrypted = decrypt_credential(encrypted)

        assert decrypted == "test"

    def test_generate_encryption_key_format(self):
        """Test generate_encryption_key produces valid format."""
        key = generate_encryption_key()

        # Should be 64 hex characters (32 bytes)
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_generate_encryption_key_unique(self):
        """Test generate_encryption_key produces unique keys."""
        key1 = generate_encryption_key()
        key2 = generate_encryption_key()

        assert key1 != key2

    def test_custom_encryption_key_works(self, monkeypatch):
        """Test custom encryption key from environment works."""
        custom_key = generate_encryption_key()
        monkeypatch.setenv("FLUXHERO_ENCRYPTION_KEY", custom_key)

        # Clear lru_cache
        from backend.core.config import get_settings

        get_settings.cache_clear()

        original = "test_api_key"
        encrypted = encrypt_credential(original)
        decrypted = decrypt_credential(encrypted)

        assert decrypted == original

    def test_wrong_key_fails_decryption(self, monkeypatch):
        """Test decryption fails with wrong key."""
        # Encrypt with key1
        key1 = generate_encryption_key()
        monkeypatch.setenv("FLUXHERO_ENCRYPTION_KEY", key1)

        from backend.core.config import get_settings

        get_settings.cache_clear()

        encrypted = encrypt_credential("test_api_key")

        # Try to decrypt with key2
        key2 = generate_encryption_key()
        monkeypatch.setenv("FLUXHERO_ENCRYPTION_KEY", key2)
        get_settings.cache_clear()

        with pytest.raises(CredentialEncryptionError, match="Decryption failed"):
            decrypt_credential(encrypted)


class TestIsEncrypted:
    """Tests for is_encrypted utility function."""

    def test_is_encrypted_true_for_encrypted_data(self):
        """Test is_encrypted returns True for encrypted data."""
        encrypted = encrypt_credential("test_api_key")

        assert is_encrypted(encrypted) is True

    def test_is_encrypted_false_for_plaintext(self):
        """Test is_encrypted returns False for plaintext."""
        assert is_encrypted("PKAJ1234567890") is False

    def test_is_encrypted_false_for_empty_string(self):
        """Test is_encrypted returns False for empty string."""
        assert is_encrypted("") is False

    def test_is_encrypted_false_for_short_base64(self):
        """Test is_encrypted returns False for short base64."""
        short = base64.b64encode(b"short").decode()

        assert is_encrypted(short) is False

    def test_is_encrypted_false_for_invalid_base64(self):
        """Test is_encrypted returns False for invalid base64."""
        assert is_encrypted("not-base64!!!") is False

    def test_is_encrypted_true_for_valid_length_base64(self):
        """Test is_encrypted returns True for base64 >= 28 bytes."""
        # 28+ bytes of random data encoded as base64
        data = os.urandom(30)
        encoded = base64.b64encode(data).decode()

        assert is_encrypted(encoded) is True


class TestMaskCredential:
    """Tests for mask_credential utility function."""

    def test_mask_api_key(self):
        """Test masking an API key."""
        result = mask_credential("PKAJ1234567890ABCDEF")

        assert result == "PKAJ****************"

    def test_mask_with_custom_visible_chars(self):
        """Test masking with custom visible characters."""
        result = mask_credential("PKAJ1234567890", visible_chars=6)

        assert result == "PKAJ12********"

    def test_mask_empty_string(self):
        """Test masking empty string."""
        result = mask_credential("")

        assert result == "****"

    def test_mask_short_string(self):
        """Test masking string shorter than visible_chars."""
        result = mask_credential("ABC", visible_chars=4)

        assert result == "***"

    def test_mask_exact_length(self):
        """Test masking string with exact visible_chars length."""
        result = mask_credential("ABCD", visible_chars=4)

        assert result == "****"

    def test_mask_with_zero_visible(self):
        """Test masking with zero visible characters."""
        result = mask_credential("SECRET", visible_chars=0)

        assert result == "******"


class TestCredentialEncryptionError:
    """Tests for CredentialEncryptionError exception."""

    def test_exception_message(self):
        """Test exception stores message correctly."""
        error = CredentialEncryptionError("Test error")

        assert str(error) == "Test error"

    def test_exception_inheritance(self):
        """Test CredentialEncryptionError is an Exception."""
        error = CredentialEncryptionError("Test")

        assert isinstance(error, Exception)


class TestModuleImports:
    """Tests for module imports."""

    def test_import_encrypt_credential(self):
        """Test encrypt_credential can be imported."""
        from backend.execution.broker_credentials import encrypt_credential

        assert callable(encrypt_credential)

    def test_import_decrypt_credential(self):
        """Test decrypt_credential can be imported."""
        from backend.execution.broker_credentials import decrypt_credential

        assert callable(decrypt_credential)

    def test_import_error_class(self):
        """Test CredentialEncryptionError can be imported."""
        from backend.execution.broker_credentials import CredentialEncryptionError

        assert issubclass(CredentialEncryptionError, Exception)

    def test_import_utilities(self):
        """Test utility functions can be imported."""
        from backend.execution.broker_credentials import (
            generate_encryption_key,
            is_encrypted,
            mask_credential,
        )

        assert callable(generate_encryption_key)
        assert callable(is_encrypted)
        assert callable(mask_credential)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
