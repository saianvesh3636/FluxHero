"""
Centralized Configuration for FluxHero

This module provides centralized configuration management using pydantic-settings.
All configuration values should be defined here and loaded from environment variables
or .env files.

Usage:
    from backend.core.config import get_settings

    settings = get_settings()
    print(settings.auth_secret)

Environment Variables:
    Configuration can be set via environment variables or .env file.
    See .env.example for available options.

Author: FluxHero
Date: 2026-01-22
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    FluxHero application settings.

    All settings are loaded from environment variables with FLUXHERO_ prefix
    or from a .env file in the backend directory.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="FLUXHERO_",
        case_sensitive=False,
        extra="ignore",
    )

    # ========================================================================
    # Authentication Settings
    # ========================================================================

    auth_secret: str = Field(
        default="fluxhero-dev-secret-change-in-production",
        description="Secret key for API and WebSocket authentication. "
        "MUST be changed in production.",
    )

    # ========================================================================
    # API Settings
    # ========================================================================

    api_title: str = Field(
        default="FluxHero API",
        description="API title displayed in documentation",
    )

    api_description: str = Field(
        default="REST API for FluxHero adaptive quant trading system",
        description="API description displayed in documentation",
    )

    api_version: str = Field(
        default="1.0.0",
        description="API version",
    )

    # ========================================================================
    # CORS Settings
    # ========================================================================

    cors_origins: list[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3000",
        ],
        description="Allowed CORS origins for API requests",
    )

    cors_allow_credentials: bool = Field(
        default=True,
        description="Allow credentials in CORS requests",
    )

    cors_allow_methods: list[str] = Field(
        default=["*"],
        description="Allowed HTTP methods for CORS",
    )

    cors_allow_headers: list[str] = Field(
        default=["*"],
        description="Allowed headers for CORS",
    )

    # ========================================================================
    # Security Settings
    # ========================================================================

    encryption_key: str = Field(
        default="",
        description="AES-256 encryption key for broker credentials. "
        "Generate with: python -c 'import secrets; print(secrets.token_hex(32))'. "
        "MUST be set in production.",
    )

    # ========================================================================
    # Alpaca API Settings
    # ========================================================================

    alpaca_api_url: str = Field(
        default="https://paper-api.alpaca.markets",
        description="Alpaca API base URL (paper or live trading)",
    )

    alpaca_ws_url: str = Field(
        default="wss://stream.data.alpaca.markets",
        description="Alpaca WebSocket URL for market data",
    )

    alpaca_api_key: str = Field(
        default="",
        description="Alpaca API key",
    )

    alpaca_api_secret: str = Field(
        default="",
        description="Alpaca API secret",
    )

    # ========================================================================
    # Risk Management Settings
    # ========================================================================

    # Position-level limits (R11.1)
    max_risk_pct_trend: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Maximum risk per position for trend-following strategies (1%)",
    )

    max_risk_pct_mean_rev: float = Field(
        default=0.0075,
        ge=0.0,
        le=1.0,
        description="Maximum risk per position for mean reversion strategies (0.75%)",
    )

    max_position_size_pct: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Maximum position size as percentage of account (20%)",
    )

    # Portfolio-level limits (R11.2)
    max_total_exposure_pct: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Maximum total portfolio exposure (50%)",
    )

    max_open_positions: int = Field(
        default=5,
        ge=1,
        description="Maximum number of open positions",
    )

    correlation_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Correlation threshold for position size reduction",
    )

    correlation_size_reduction: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Position size reduction when correlation exceeds threshold (50%)",
    )

    # ATR stop multipliers (R11.1.4)
    trend_stop_atr_multiplier: float = Field(
        default=2.5,
        ge=0.0,
        description="ATR multiplier for trend-following stop losses",
    )

    mean_rev_stop_pct: float = Field(
        default=0.03,
        ge=0.0,
        le=1.0,
        description="Fixed stop loss percentage for mean reversion (3%)",
    )

    # ========================================================================
    # Data Storage Settings
    # ========================================================================

    cache_dir: str = Field(
        default="data/cache",
        description="Directory for data caching",
    )

    log_file: str = Field(
        default="logs/daily_reboot.log",
        description="Log file path for daily reboot operations",
    )

    # ========================================================================
    # Market Data Settings
    # ========================================================================

    default_timeframe: str = Field(
        default="1h",
        description="Default timeframe for market data",
    )

    initial_candles: int = Field(
        default=500,
        ge=1,
        description="Number of initial candles to fetch on startup",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    This function uses lru_cache to ensure only one Settings instance
    is created and reused throughout the application lifecycle.

    Returns:
        Settings instance with configuration loaded from environment
    """
    return Settings()
