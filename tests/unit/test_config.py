"""
Unit tests for centralized configuration module.

Tests cover:
- Default configuration values
- Environment variable loading
- Settings validation
- Settings caching behavior
- Field constraints

Author: FluxHero
Date: 2026-01-22
"""

import os
from unittest.mock import patch

import pytest

from backend.core.config import Settings, get_settings


class TestSettingsDefaults:
    """Test default configuration values."""

    def test_default_auth_secret(self):
        """Test default authentication secret."""
        settings = Settings()
        assert settings.auth_secret == "fluxhero-dev-secret-change-in-production"

    def test_default_api_settings(self):
        """Test default API configuration."""
        settings = Settings()
        assert settings.api_title == "FluxHero API"
        assert settings.api_description == "REST API for FluxHero adaptive quant trading system"
        assert settings.api_version == "1.0.0"

    def test_default_cors_origins(self):
        """Test default CORS origins."""
        settings = Settings()
        assert "http://localhost:3000" in settings.cors_origins
        assert "http://localhost:3001" in settings.cors_origins
        assert "http://127.0.0.1:3000" in settings.cors_origins
        assert settings.cors_allow_credentials is True
        assert settings.cors_allow_methods == ["*"]
        assert settings.cors_allow_headers == ["*"]

    def test_default_alpaca_settings(self):
        """Test default Alpaca API configuration."""
        settings = Settings()
        assert settings.alpaca_api_url == "https://paper-api.alpaca.markets"
        assert settings.alpaca_ws_url == "wss://stream.data.alpaca.markets"
        assert settings.alpaca_api_key == ""
        assert settings.alpaca_api_secret == ""

    def test_default_risk_position_limits(self):
        """Test default position-level risk limits."""
        settings = Settings()
        assert settings.max_risk_pct_trend == 0.01
        assert settings.max_risk_pct_mean_rev == 0.0075
        assert settings.max_position_size_pct == 0.20

    def test_default_risk_portfolio_limits(self):
        """Test default portfolio-level risk limits."""
        settings = Settings()
        assert settings.max_total_exposure_pct == 0.50
        assert settings.max_open_positions == 5
        assert settings.correlation_threshold == 0.7
        assert settings.correlation_size_reduction == 0.50

    def test_default_risk_stop_settings(self):
        """Test default stop loss configuration."""
        settings = Settings()
        assert settings.trend_stop_atr_multiplier == 2.5
        assert settings.mean_rev_stop_pct == 0.03

    def test_default_storage_settings(self):
        """Test default storage configuration."""
        settings = Settings()
        assert settings.cache_dir == "data/cache"
        assert settings.log_file == "logs/daily_reboot.log"

    def test_default_market_data_settings(self):
        """Test default market data configuration."""
        settings = Settings()
        assert settings.default_timeframe == "1h"
        assert settings.initial_candles == 500


class TestEnvironmentVariables:
    """Test environment variable loading."""

    def test_env_var_with_prefix(self):
        """Test loading environment variable with FLUXHERO_ prefix."""
        with patch.dict(os.environ, {"FLUXHERO_AUTH_SECRET": "test-secret-123"}):
            settings = Settings()
            assert settings.auth_secret == "test-secret-123"

    def test_env_var_cors_origins(self):
        """Test loading CORS origins from environment."""
        # pydantic-settings handles list parsing from JSON string
        with patch.dict(os.environ, {
            "FLUXHERO_CORS_ORIGINS": '["http://example.com", "http://test.com"]'
        }):
            settings = Settings()
            assert "http://example.com" in settings.cors_origins
            assert "http://test.com" in settings.cors_origins

    def test_env_var_alpaca_settings(self):
        """Test loading Alpaca settings from environment."""
        with patch.dict(os.environ, {
            "FLUXHERO_ALPACA_API_KEY": "test-key",
            "FLUXHERO_ALPACA_API_SECRET": "test-secret",
            "FLUXHERO_ALPACA_API_URL": "https://api.alpaca.markets",
        }):
            settings = Settings()
            assert settings.alpaca_api_key == "test-key"
            assert settings.alpaca_api_secret == "test-secret"
            assert settings.alpaca_api_url == "https://api.alpaca.markets"

    def test_env_var_risk_settings(self):
        """Test loading risk settings from environment."""
        with patch.dict(os.environ, {
            "FLUXHERO_MAX_RISK_PCT_TREND": "0.02",
            "FLUXHERO_MAX_OPEN_POSITIONS": "10",
            "FLUXHERO_CORRELATION_THRESHOLD": "0.8",
        }):
            settings = Settings()
            assert settings.max_risk_pct_trend == 0.02
            assert settings.max_open_positions == 10
            assert settings.correlation_threshold == 0.8

    def test_env_var_case_insensitive(self):
        """Test that environment variables are case-insensitive."""
        with patch.dict(os.environ, {"fluxhero_auth_secret": "lowercase-secret"}):
            settings = Settings()
            assert settings.auth_secret == "lowercase-secret"


class TestSettingsValidation:
    """Test field validation constraints."""

    def test_max_risk_pct_trend_validation(self):
        """Test max_risk_pct_trend must be between 0 and 1."""
        with pytest.raises(Exception):  # ValidationError from pydantic
            Settings(max_risk_pct_trend=-0.1)

        with pytest.raises(Exception):
            Settings(max_risk_pct_trend=1.5)

        # Valid values should work
        settings = Settings(max_risk_pct_trend=0.5)
        assert settings.max_risk_pct_trend == 0.5

    def test_max_risk_pct_mean_rev_validation(self):
        """Test max_risk_pct_mean_rev must be between 0 and 1."""
        with pytest.raises(Exception):
            Settings(max_risk_pct_mean_rev=-0.1)

        with pytest.raises(Exception):
            Settings(max_risk_pct_mean_rev=1.5)

        settings = Settings(max_risk_pct_mean_rev=0.01)
        assert settings.max_risk_pct_mean_rev == 0.01

    def test_max_position_size_pct_validation(self):
        """Test max_position_size_pct must be between 0 and 1."""
        with pytest.raises(Exception):
            Settings(max_position_size_pct=-0.1)

        with pytest.raises(Exception):
            Settings(max_position_size_pct=1.5)

        settings = Settings(max_position_size_pct=0.3)
        assert settings.max_position_size_pct == 0.3

    def test_max_total_exposure_pct_validation(self):
        """Test max_total_exposure_pct must be between 0 and 1."""
        with pytest.raises(Exception):
            Settings(max_total_exposure_pct=-0.1)

        with pytest.raises(Exception):
            Settings(max_total_exposure_pct=1.5)

        settings = Settings(max_total_exposure_pct=0.8)
        assert settings.max_total_exposure_pct == 0.8

    def test_max_open_positions_validation(self):
        """Test max_open_positions must be at least 1."""
        with pytest.raises(Exception):
            Settings(max_open_positions=0)

        with pytest.raises(Exception):
            Settings(max_open_positions=-1)

        settings = Settings(max_open_positions=10)
        assert settings.max_open_positions == 10

    def test_correlation_threshold_validation(self):
        """Test correlation_threshold must be between 0 and 1."""
        with pytest.raises(Exception):
            Settings(correlation_threshold=-0.1)

        with pytest.raises(Exception):
            Settings(correlation_threshold=1.5)

        settings = Settings(correlation_threshold=0.9)
        assert settings.correlation_threshold == 0.9

    def test_correlation_size_reduction_validation(self):
        """Test correlation_size_reduction must be between 0 and 1."""
        with pytest.raises(Exception):
            Settings(correlation_size_reduction=-0.1)

        with pytest.raises(Exception):
            Settings(correlation_size_reduction=1.5)

        settings = Settings(correlation_size_reduction=0.25)
        assert settings.correlation_size_reduction == 0.25

    def test_trend_stop_atr_multiplier_validation(self):
        """Test trend_stop_atr_multiplier must be non-negative."""
        with pytest.raises(Exception):
            Settings(trend_stop_atr_multiplier=-1.0)

        settings = Settings(trend_stop_atr_multiplier=3.0)
        assert settings.trend_stop_atr_multiplier == 3.0

    def test_mean_rev_stop_pct_validation(self):
        """Test mean_rev_stop_pct must be between 0 and 1."""
        with pytest.raises(Exception):
            Settings(mean_rev_stop_pct=-0.1)

        with pytest.raises(Exception):
            Settings(mean_rev_stop_pct=1.5)

        settings = Settings(mean_rev_stop_pct=0.05)
        assert settings.mean_rev_stop_pct == 0.05

    def test_initial_candles_validation(self):
        """Test initial_candles must be at least 1."""
        with pytest.raises(Exception):
            Settings(initial_candles=0)

        with pytest.raises(Exception):
            Settings(initial_candles=-1)

        settings = Settings(initial_candles=1000)
        assert settings.initial_candles == 1000


class TestGetSettings:
    """Test the get_settings() caching function."""

    def test_get_settings_returns_settings_instance(self):
        """Test get_settings returns a Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_get_settings_caches_instance(self):
        """Test get_settings returns the same instance on multiple calls."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2  # Same object in memory

    def test_get_settings_with_env_vars(self):
        """Test get_settings respects environment variables."""
        # Clear the cache first
        get_settings.cache_clear()

        with patch.dict(os.environ, {"FLUXHERO_AUTH_SECRET": "cached-secret"}):
            settings = get_settings()
            assert settings.auth_secret == "cached-secret"

        # Clear cache for other tests
        get_settings.cache_clear()


class TestConfigIntegration:
    """Integration tests for configuration usage."""

    def test_all_risk_parameters_accessible(self):
        """Test all risk parameters can be accessed together."""
        settings = Settings()

        # Position-level
        assert isinstance(settings.max_risk_pct_trend, float)
        assert isinstance(settings.max_risk_pct_mean_rev, float)
        assert isinstance(settings.max_position_size_pct, float)

        # Portfolio-level
        assert isinstance(settings.max_total_exposure_pct, float)
        assert isinstance(settings.max_open_positions, int)
        assert isinstance(settings.correlation_threshold, float)
        assert isinstance(settings.correlation_size_reduction, float)

        # Stop losses
        assert isinstance(settings.trend_stop_atr_multiplier, float)
        assert isinstance(settings.mean_rev_stop_pct, float)

    def test_all_api_parameters_accessible(self):
        """Test all API parameters can be accessed together."""
        settings = Settings()

        assert isinstance(settings.api_title, str)
        assert isinstance(settings.api_description, str)
        assert isinstance(settings.api_version, str)
        assert isinstance(settings.cors_origins, list)
        assert isinstance(settings.cors_allow_credentials, bool)

    def test_all_alpaca_parameters_accessible(self):
        """Test all Alpaca parameters can be accessed together."""
        settings = Settings()

        assert isinstance(settings.alpaca_api_url, str)
        assert isinstance(settings.alpaca_ws_url, str)
        assert isinstance(settings.alpaca_api_key, str)
        assert isinstance(settings.alpaca_api_secret, str)

    def test_settings_can_be_modified(self):
        """Test settings values can be overridden at instantiation."""
        settings = Settings(
            auth_secret="custom-secret",
            max_open_positions=20,
            cors_origins=["http://custom.com"]
        )

        assert settings.auth_secret == "custom-secret"
        assert settings.max_open_positions == 20
        assert settings.cors_origins == ["http://custom.com"]

    def test_extra_fields_ignored(self):
        """Test that extra/unknown fields are ignored."""
        # Should not raise an error due to extra="ignore"
        settings = Settings(unknown_field="value")
        assert not hasattr(settings, "unknown_field")


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_string_env_var(self):
        """Test handling of empty string environment variables."""
        with patch.dict(os.environ, {"FLUXHERO_AUTH_SECRET": ""}):
            settings = Settings()
            # Empty string should override default
            assert settings.auth_secret == ""

    def test_boundary_values_for_percentages(self):
        """Test boundary values (0 and 1) for percentage fields."""
        settings = Settings(
            max_risk_pct_trend=0.0,
            max_position_size_pct=1.0,
            correlation_threshold=0.0,
            mean_rev_stop_pct=1.0,
        )

        assert settings.max_risk_pct_trend == 0.0
        assert settings.max_position_size_pct == 1.0
        assert settings.correlation_threshold == 0.0
        assert settings.mean_rev_stop_pct == 1.0

    def test_minimum_valid_positions(self):
        """Test minimum valid value for max_open_positions."""
        settings = Settings(max_open_positions=1)
        assert settings.max_open_positions == 1

    def test_minimum_valid_candles(self):
        """Test minimum valid value for initial_candles."""
        settings = Settings(initial_candles=1)
        assert settings.initial_candles == 1

    def test_zero_atr_multiplier(self):
        """Test zero is valid for ATR multiplier."""
        settings = Settings(trend_stop_atr_multiplier=0.0)
        assert settings.trend_stop_atr_multiplier == 0.0
