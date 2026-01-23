"""
Broker Factory - Factory Pattern for Creating Broker Instances

This module implements a factory pattern for creating broker instances based on
configuration. It handles broker type validation, config validation using Pydantic,
and maintains a singleton cache for connection reuse.

Feature: Multi-Broker Architecture (Phase A)
"""

from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from backend.execution.broker_base import BrokerInterface
from backend.execution.brokers.alpaca_broker import AlpacaBroker

# -----------------------------------------------------------------------------
# Config Models (Pydantic validation per broker type)
# -----------------------------------------------------------------------------


class AlpacaBrokerConfig(BaseModel):
    """Configuration for Alpaca broker."""

    api_key: str = Field(..., min_length=1, description="Alpaca API key")
    api_secret: str = Field(..., min_length=1, description="Alpaca API secret")
    base_url: str = Field(
        default="https://paper-api.alpaca.markets",
        description="Alpaca API base URL (paper or live)",
    )
    timeout: float = Field(default=30.0, gt=0, description="Request timeout in seconds")


# Type alias for supported broker types
BrokerType = Literal["alpaca"]

# Registry mapping broker types to their config models
BROKER_CONFIG_MODELS: dict[str, type[BaseModel]] = {
    "alpaca": AlpacaBrokerConfig,
}


# -----------------------------------------------------------------------------
# Broker Factory
# -----------------------------------------------------------------------------


class BrokerFactoryError(Exception):
    """Raised when broker creation fails."""

    pass


class BrokerFactory:
    """
    Factory for creating broker instances.

    Implements factory pattern with:
    - Type-based broker creation
    - Pydantic config validation per broker type
    - Singleton caching for connection reuse

    Usage:
        factory = BrokerFactory()

        # Create a new broker
        broker = factory.create_broker("alpaca", {
            "api_key": "your-key",
            "api_secret": "your-secret",
        })

        # Get cached broker (same config returns same instance)
        broker2 = factory.create_broker("alpaca", same_config)
        assert broker is broker2  # Same instance

        # Clear cache if needed
        factory.clear_cache()
    """

    # Singleton instance
    _instance: "BrokerFactory | None" = None

    def __new__(cls) -> "BrokerFactory":
        """Ensure only one factory instance exists (singleton)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._broker_cache = {}
        return cls._instance

    def __init__(self) -> None:
        """Initialize broker factory."""
        # Ensure cache exists (for repeated __init__ calls)
        if not hasattr(self, "_broker_cache"):
            self._broker_cache: dict[str, BrokerInterface] = {}

    def _get_cache_key(self, broker_type: str, config: dict) -> str:
        """
        Generate a cache key from broker type and config.

        Args:
            broker_type: Type of broker
            config: Broker configuration dict

        Returns:
            Cache key string
        """
        # Use sorted items for consistent key generation
        config_str = str(sorted(config.items()))
        return f"{broker_type}:{config_str}"

    def validate_config(self, broker_type: str, config: dict) -> BaseModel:
        """
        Validate broker configuration using Pydantic models.

        Args:
            broker_type: Type of broker ("alpaca", etc.)
            config: Configuration dictionary

        Returns:
            Validated Pydantic config model

        Raises:
            BrokerFactoryError: If broker type is unknown or config is invalid
        """
        if broker_type not in BROKER_CONFIG_MODELS:
            supported = list(BROKER_CONFIG_MODELS.keys())
            raise BrokerFactoryError(
                f"Unknown broker type: '{broker_type}'. Supported types: {supported}"
            )

        config_model = BROKER_CONFIG_MODELS[broker_type]

        try:
            return config_model(**config)
        except ValidationError as e:
            raise BrokerFactoryError(f"Invalid {broker_type} config: {e}") from e

    def create_broker(
        self,
        broker_type: str,
        config: dict,
        use_cache: bool = True,
    ) -> BrokerInterface:
        """
        Create a broker instance.

        Args:
            broker_type: Type of broker ("alpaca")
            config: Broker configuration dictionary
            use_cache: Whether to use cached instance if available (default: True)

        Returns:
            BrokerInterface implementation

        Raises:
            BrokerFactoryError: If broker type unknown or config invalid
        """
        # Validate config first
        validated_config = self.validate_config(broker_type, config)

        # Check cache for existing instance
        cache_key = self._get_cache_key(broker_type, config)
        if use_cache and cache_key in self._broker_cache:
            logger.debug(f"Returning cached broker instance for {broker_type}")
            return self._broker_cache[cache_key]

        # Create new broker instance based on type
        broker: BrokerInterface

        if broker_type == "alpaca":
            alpaca_config = validated_config
            assert isinstance(alpaca_config, AlpacaBrokerConfig)
            broker = AlpacaBroker(
                api_key=alpaca_config.api_key,
                api_secret=alpaca_config.api_secret,
                base_url=alpaca_config.base_url,
                timeout=alpaca_config.timeout,
            )
        else:
            # This shouldn't happen due to validation, but handle gracefully
            raise BrokerFactoryError(f"Broker type not implemented: {broker_type}")

        # Cache the instance
        if use_cache:
            self._broker_cache[cache_key] = broker
            logger.info(f"Created and cached new {broker_type} broker instance")
        else:
            logger.info(f"Created new {broker_type} broker instance (not cached)")

        return broker

    def get_cached_broker(self, broker_type: str, config: dict) -> BrokerInterface | None:
        """
        Get a cached broker instance if it exists.

        Args:
            broker_type: Type of broker
            config: Broker configuration

        Returns:
            Cached broker instance or None if not found
        """
        cache_key = self._get_cache_key(broker_type, config)
        return self._broker_cache.get(cache_key)

    def clear_cache(self) -> None:
        """Clear the broker cache."""
        self._broker_cache.clear()
        logger.info("Broker factory cache cleared")

    def remove_from_cache(self, broker_type: str, config: dict) -> bool:
        """
        Remove a specific broker from cache.

        Args:
            broker_type: Type of broker
            config: Broker configuration

        Returns:
            True if broker was removed, False if not found
        """
        cache_key = self._get_cache_key(broker_type, config)
        if cache_key in self._broker_cache:
            del self._broker_cache[cache_key]
            logger.debug(f"Removed {broker_type} broker from cache")
            return True
        return False

    @property
    def supported_broker_types(self) -> list[str]:
        """Get list of supported broker types."""
        return list(BROKER_CONFIG_MODELS.keys())

    @property
    def cache_size(self) -> int:
        """Get number of cached broker instances."""
        return len(self._broker_cache)


# -----------------------------------------------------------------------------
# Module-level convenience function
# -----------------------------------------------------------------------------


def create_broker(
    broker_type: str,
    config: dict,
    use_cache: bool = True,
) -> BrokerInterface:
    """
    Convenience function to create a broker instance.

    Uses the singleton BrokerFactory internally.

    Args:
        broker_type: Type of broker ("alpaca")
        config: Broker configuration dictionary
        use_cache: Whether to use cached instance if available (default: True)

    Returns:
        BrokerInterface implementation

    Raises:
        BrokerFactoryError: If broker type unknown or config invalid
    """
    factory = BrokerFactory()
    return factory.create_broker(broker_type, config, use_cache)
