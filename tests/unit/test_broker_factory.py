"""
Unit tests for broker factory.

Tests cover:
- Factory singleton pattern
- Config validation using Pydantic
- Broker creation for each supported type
- Cache behavior (singleton reuse)
- Error handling for invalid types and configs
"""

import pytest

from backend.execution.broker_base import BrokerInterface
from backend.execution.broker_factory import (
    BROKER_CONFIG_MODELS,
    AlpacaBrokerConfig,
    BrokerFactory,
    BrokerFactoryError,
    create_broker,
)
from backend.execution.brokers.alpaca_broker import AlpacaBroker


class TestAlpacaBrokerConfig:
    """Tests for AlpacaBrokerConfig Pydantic model."""

    def test_valid_config(self):
        """Test valid config parses correctly."""
        config = AlpacaBrokerConfig(
            api_key="test-key",
            api_secret="test-secret",
        )

        assert config.api_key == "test-key"
        assert config.api_secret == "test-secret"
        assert config.base_url == "https://paper-api.alpaca.markets"
        assert config.timeout == 30.0

    def test_valid_config_with_custom_values(self):
        """Test valid config with custom base_url and timeout."""
        config = AlpacaBrokerConfig(
            api_key="key",
            api_secret="secret",
            base_url="https://api.alpaca.markets",
            timeout=60.0,
        )

        assert config.base_url == "https://api.alpaca.markets"
        assert config.timeout == 60.0

    def test_missing_api_key_raises(self):
        """Test missing api_key raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="api_key"):
            AlpacaBrokerConfig(api_secret="secret")

    def test_missing_api_secret_raises(self):
        """Test missing api_secret raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="api_secret"):
            AlpacaBrokerConfig(api_key="key")

    def test_empty_api_key_raises(self):
        """Test empty api_key raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="api_key"):
            AlpacaBrokerConfig(api_key="", api_secret="secret")

    def test_invalid_timeout_raises(self):
        """Test non-positive timeout raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="timeout"):
            AlpacaBrokerConfig(api_key="key", api_secret="secret", timeout=0)

        with pytest.raises(ValidationError, match="timeout"):
            AlpacaBrokerConfig(api_key="key", api_secret="secret", timeout=-1)


class TestBrokerConfigModels:
    """Tests for BROKER_CONFIG_MODELS registry."""

    def test_alpaca_in_registry(self):
        """Test alpaca is registered."""
        assert "alpaca" in BROKER_CONFIG_MODELS
        assert BROKER_CONFIG_MODELS["alpaca"] is AlpacaBrokerConfig

    def test_supported_broker_types(self):
        """Test only expected broker types are registered."""
        expected_types = {"alpaca"}
        assert set(BROKER_CONFIG_MODELS.keys()) == expected_types


class TestBrokerFactorySingleton:
    """Tests for BrokerFactory singleton pattern."""

    def test_singleton_instance(self):
        """Test BrokerFactory returns same instance."""
        factory1 = BrokerFactory()
        factory2 = BrokerFactory()

        assert factory1 is factory2

    def test_singleton_preserves_cache(self):
        """Test singleton preserves broker cache across instances."""
        factory1 = BrokerFactory()
        factory1.clear_cache()

        config = {"api_key": "key", "api_secret": "secret"}
        broker1 = factory1.create_broker("alpaca", config)

        factory2 = BrokerFactory()
        broker2 = factory2.get_cached_broker("alpaca", config)

        assert broker2 is broker1


class TestBrokerFactoryValidateConfig:
    """Tests for BrokerFactory.validate_config()."""

    def setup_method(self):
        """Reset singleton state for each test."""
        self.factory = BrokerFactory()
        self.factory.clear_cache()

    def test_validate_alpaca_config_success(self):
        """Test validating valid Alpaca config."""
        config = {"api_key": "test-key", "api_secret": "test-secret"}

        validated = self.factory.validate_config("alpaca", config)

        assert isinstance(validated, AlpacaBrokerConfig)
        assert validated.api_key == "test-key"
        assert validated.api_secret == "test-secret"

    def test_validate_unknown_broker_type(self):
        """Test validating unknown broker type raises error."""
        config = {"api_key": "key", "api_secret": "secret"}

        with pytest.raises(BrokerFactoryError, match="Unknown broker type"):
            self.factory.validate_config("unknown_broker", config)

    def test_validate_invalid_config(self):
        """Test validating invalid config raises error."""
        config = {"api_key": "key"}  # Missing api_secret

        with pytest.raises(BrokerFactoryError, match="Invalid alpaca config"):
            self.factory.validate_config("alpaca", config)


class TestBrokerFactoryCreateBroker:
    """Tests for BrokerFactory.create_broker()."""

    def setup_method(self):
        """Reset singleton state for each test."""
        self.factory = BrokerFactory()
        self.factory.clear_cache()

    def test_create_alpaca_broker(self):
        """Test creating Alpaca broker."""
        config = {
            "api_key": "test-key",
            "api_secret": "test-secret",
        }

        broker = self.factory.create_broker("alpaca", config)

        assert isinstance(broker, AlpacaBroker)
        assert isinstance(broker, BrokerInterface)
        assert broker.api_key == "test-key"
        assert broker.api_secret == "test-secret"

    def test_create_alpaca_broker_with_custom_url(self):
        """Test creating Alpaca broker with custom base_url."""
        config = {
            "api_key": "key",
            "api_secret": "secret",
            "base_url": "https://api.alpaca.markets",
            "timeout": 60.0,
        }

        broker = self.factory.create_broker("alpaca", config)

        assert broker.base_url == "https://api.alpaca.markets"
        assert broker.timeout == 60.0

    def test_create_unknown_broker_raises(self):
        """Test creating unknown broker type raises error."""
        config = {"api_key": "key", "api_secret": "secret"}

        with pytest.raises(BrokerFactoryError, match="Unknown broker type"):
            self.factory.create_broker("unknown_broker", config)

    def test_create_broker_invalid_config_raises(self):
        """Test creating broker with invalid config raises error."""
        config = {}  # Missing required fields

        with pytest.raises(BrokerFactoryError, match="Invalid alpaca config"):
            self.factory.create_broker("alpaca", config)


class TestBrokerFactoryCache:
    """Tests for BrokerFactory caching behavior."""

    def setup_method(self):
        """Reset singleton state for each test."""
        self.factory = BrokerFactory()
        self.factory.clear_cache()

    def test_cache_returns_same_instance(self):
        """Test cache returns same broker instance."""
        config = {"api_key": "key", "api_secret": "secret"}

        broker1 = self.factory.create_broker("alpaca", config)
        broker2 = self.factory.create_broker("alpaca", config)

        assert broker1 is broker2

    def test_different_configs_create_different_brokers(self):
        """Test different configs create different broker instances."""
        config1 = {"api_key": "key1", "api_secret": "secret1"}
        config2 = {"api_key": "key2", "api_secret": "secret2"}

        broker1 = self.factory.create_broker("alpaca", config1)
        broker2 = self.factory.create_broker("alpaca", config2)

        assert broker1 is not broker2

    def test_use_cache_false_creates_new_instance(self):
        """Test use_cache=False creates new instance."""
        config = {"api_key": "key", "api_secret": "secret"}

        broker1 = self.factory.create_broker("alpaca", config)
        broker2 = self.factory.create_broker("alpaca", config, use_cache=False)

        assert broker1 is not broker2

    def test_get_cached_broker_returns_instance(self):
        """Test get_cached_broker returns cached instance."""
        config = {"api_key": "key", "api_secret": "secret"}

        self.factory.create_broker("alpaca", config)
        cached = self.factory.get_cached_broker("alpaca", config)

        assert cached is not None
        assert isinstance(cached, AlpacaBroker)

    def test_get_cached_broker_returns_none_when_not_cached(self):
        """Test get_cached_broker returns None when not in cache."""
        config = {"api_key": "key", "api_secret": "secret"}

        cached = self.factory.get_cached_broker("alpaca", config)

        assert cached is None

    def test_clear_cache(self):
        """Test clear_cache removes all cached brokers."""
        config = {"api_key": "key", "api_secret": "secret"}

        self.factory.create_broker("alpaca", config)
        assert self.factory.cache_size == 1

        self.factory.clear_cache()

        assert self.factory.cache_size == 0
        assert self.factory.get_cached_broker("alpaca", config) is None

    def test_remove_from_cache(self):
        """Test remove_from_cache removes specific broker."""
        config1 = {"api_key": "key1", "api_secret": "secret1"}
        config2 = {"api_key": "key2", "api_secret": "secret2"}

        self.factory.create_broker("alpaca", config1)
        self.factory.create_broker("alpaca", config2)
        assert self.factory.cache_size == 2

        result = self.factory.remove_from_cache("alpaca", config1)

        assert result is True
        assert self.factory.cache_size == 1
        assert self.factory.get_cached_broker("alpaca", config1) is None
        assert self.factory.get_cached_broker("alpaca", config2) is not None

    def test_remove_from_cache_returns_false_when_not_found(self):
        """Test remove_from_cache returns False when not in cache."""
        config = {"api_key": "key", "api_secret": "secret"}

        result = self.factory.remove_from_cache("alpaca", config)

        assert result is False


class TestBrokerFactoryProperties:
    """Tests for BrokerFactory properties."""

    def setup_method(self):
        """Reset singleton state for each test."""
        self.factory = BrokerFactory()
        self.factory.clear_cache()

    def test_supported_broker_types(self):
        """Test supported_broker_types returns correct list."""
        types = self.factory.supported_broker_types

        assert "alpaca" in types
        assert len(types) == 1

    def test_cache_size(self):
        """Test cache_size property."""
        assert self.factory.cache_size == 0

        self.factory.create_broker("alpaca", {"api_key": "k1", "api_secret": "s1"})
        assert self.factory.cache_size == 1

        self.factory.create_broker("alpaca", {"api_key": "k2", "api_secret": "s2"})
        assert self.factory.cache_size == 2


class TestCreateBrokerFunction:
    """Tests for module-level create_broker() convenience function."""

    def setup_method(self):
        """Reset singleton state for each test."""
        factory = BrokerFactory()
        factory.clear_cache()

    def test_create_broker_convenience_function(self):
        """Test create_broker convenience function works."""
        config = {"api_key": "test-key", "api_secret": "test-secret"}

        broker = create_broker("alpaca", config)

        assert isinstance(broker, AlpacaBroker)
        assert broker.api_key == "test-key"

    def test_create_broker_uses_cache(self):
        """Test create_broker convenience function uses cache."""
        config = {"api_key": "key", "api_secret": "secret"}

        broker1 = create_broker("alpaca", config)
        broker2 = create_broker("alpaca", config)

        assert broker1 is broker2

    def test_create_broker_without_cache(self):
        """Test create_broker convenience function can bypass cache."""
        config = {"api_key": "key", "api_secret": "secret"}

        broker1 = create_broker("alpaca", config, use_cache=False)
        broker2 = create_broker("alpaca", config, use_cache=False)

        assert broker1 is not broker2


class TestBrokerFactoryError:
    """Tests for BrokerFactoryError exception."""

    def test_exception_message(self):
        """Test exception stores message correctly."""
        error = BrokerFactoryError("Test error message")

        assert str(error) == "Test error message"

    def test_exception_inheritance(self):
        """Test BrokerFactoryError is an Exception."""
        error = BrokerFactoryError("Test")

        assert isinstance(error, Exception)


class TestBrokerFactoryImports:
    """Tests for module imports."""

    def test_import_factory(self):
        """Test BrokerFactory can be imported."""
        from backend.execution.broker_factory import BrokerFactory

        assert BrokerFactory is not None

    def test_import_create_broker(self):
        """Test create_broker can be imported."""
        from backend.execution.broker_factory import create_broker

        assert callable(create_broker)

    def test_import_error(self):
        """Test BrokerFactoryError can be imported."""
        from backend.execution.broker_factory import BrokerFactoryError

        assert issubclass(BrokerFactoryError, Exception)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
