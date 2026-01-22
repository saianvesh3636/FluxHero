"""
Unit tests for Daily Reboot Script.

Tests cover:
- Configuration loading (file, args, defaults)
- Reboot orchestrator initialization
- Historical data fetching with cache hit/miss
- WebSocket connection initialization
- System readiness verification
- Error handling and cleanup
- End-to-end reboot sequence
"""

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / ""))

from backend.data.fetcher import Candle
from backend.maintenance.daily_reboot import (
    DailyRebootOrchestrator,
    RebootConfig,
)
from backend.storage.parquet_store import CandleData

# ============================================================================
# Configuration Tests
# ============================================================================


def test_reboot_config_defaults():
    """Test RebootConfig with default values."""
    config = RebootConfig(symbols=["SPY"])

    assert config.symbols == ["SPY"]
    assert config.timeframe == "1h"
    assert config.initial_candles == 500
    assert config.api_url == "https://paper-api.alpaca.markets"
    assert config.ws_url == "wss://stream.data.alpaca.markets"
    assert config.cache_dir == "data/cache"
    assert config.log_file == "logs/daily_reboot.log"


def test_reboot_config_custom_values():
    """Test RebootConfig with custom values."""
    config = RebootConfig(
        symbols=["SPY", "QQQ"],
        timeframe="1d",
        initial_candles=1000,
        api_url="https://custom-api.example.com",
        ws_url="wss://custom-ws.example.com",
        api_key="test_key",
        api_secret="test_secret",
        cache_dir="custom/cache",
        log_file="custom/log.log",
    )

    assert config.symbols == ["SPY", "QQQ"]
    assert config.timeframe == "1d"
    assert config.initial_candles == 1000
    assert config.api_url == "https://custom-api.example.com"
    assert config.ws_url == "wss://custom-ws.example.com"
    assert config.api_key == "test_key"
    assert config.api_secret == "test_secret"
    assert config.cache_dir == "custom/cache"
    assert config.log_file == "custom/log.log"


def test_reboot_config_from_file(tmp_path):
    """Test loading RebootConfig from JSON file."""
    config_data = {
        "symbols": ["SPY", "QQQ", "IWM"],
        "timeframe": "1h",
        "initial_candles": 500,
        "api_url": "https://paper-api.alpaca.markets",
        "ws_url": "wss://stream.data.alpaca.markets",
        "api_key": "file_key",
        "api_secret": "file_secret",
        "cache_dir": "data/cache",
        "log_file": "logs/daily_reboot.log",
    }

    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data))

    config = RebootConfig.from_file(str(config_file))

    assert config.symbols == ["SPY", "QQQ", "IWM"]
    assert config.timeframe == "1h"
    assert config.initial_candles == 500
    assert config.api_key == "file_key"
    assert config.api_secret == "file_secret"


def test_reboot_config_from_args():
    """Test creating RebootConfig from command-line arguments."""
    mock_args = MagicMock()
    mock_args.symbols = "SPY,QQQ"
    mock_args.timeframe = "1d"
    mock_args.initial_candles = 1000
    mock_args.api_url = "https://custom-api.example.com"
    mock_args.ws_url = "wss://custom-ws.example.com"
    mock_args.api_key = "args_key"
    mock_args.api_secret = "args_secret"
    mock_args.cache_dir = "data/cache"
    mock_args.log_file = "logs/daily_reboot.log"

    config = RebootConfig.from_args(mock_args)

    assert config.symbols == ["SPY", "QQQ"]
    assert config.timeframe == "1d"
    assert config.initial_candles == 1000
    assert config.api_key == "args_key"


def test_reboot_config_uses_centralized_config():
    """Test RebootConfig loads defaults from centralized config when not specified."""
    # Clear environment variables to ensure clean test
    env_vars_to_clear = [
        "FLUXHERO_ALPACA_API_URL",
        "FLUXHERO_ALPACA_WS_URL",
        "FLUXHERO_DEFAULT_TIMEFRAME",
        "FLUXHERO_INITIAL_CANDLES",
    ]
    original_env = {}
    for var in env_vars_to_clear:
        original_env[var] = os.environ.pop(var, None)

    try:
        # Create config with only required parameters
        config = RebootConfig(symbols=["SPY"])

        # Verify defaults come from centralized config
        assert config.symbols == ["SPY"]
        assert config.api_url == "https://paper-api.alpaca.markets"  # from centralized config
        assert config.ws_url == "wss://stream.data.alpaca.markets"  # from centralized config
        assert config.timeframe == "1h"  # from centralized config
        assert config.initial_candles == 500  # from centralized config
        assert config.cache_dir == "data/cache"  # from centralized config
        assert config.log_file == "logs/daily_reboot.log"  # from centralized config
    finally:
        # Restore environment
        for var, value in original_env.items():
            if value is not None:
                os.environ[var] = value


def test_reboot_config_overrides_centralized_config():
    """Test RebootConfig allows overriding centralized config values."""
    config = RebootConfig(
        symbols=["QQQ"],
        api_url="https://custom-api.example.com",
        ws_url="wss://custom-ws.example.com",
        timeframe="5m",
        initial_candles=1000,
    )

    # Verify overrides work
    assert config.symbols == ["QQQ"]
    assert config.api_url == "https://custom-api.example.com"
    assert config.ws_url == "wss://custom-ws.example.com"
    assert config.timeframe == "5m"
    assert config.initial_candles == 1000


# ============================================================================
# Orchestrator Initialization Tests
# ============================================================================


def test_orchestrator_initialization():
    """Test DailyRebootOrchestrator initialization."""
    config = RebootConfig(symbols=["SPY"])
    orchestrator = DailyRebootOrchestrator(config)

    assert orchestrator.config == config
    assert orchestrator.logger is not None
    assert orchestrator.parquet_store is not None
    assert orchestrator.candle_buffers == {}
    assert orchestrator.rest_client is None
    assert orchestrator.ws_feeds == {}
    assert orchestrator.pipelines == {}


def test_orchestrator_logging_setup(tmp_path):
    """Test logging setup creates log directory."""
    log_file = tmp_path / "logs" / "test_reboot.log"
    config = RebootConfig(symbols=["SPY"], log_file=str(log_file))
    orchestrator = DailyRebootOrchestrator(config)

    assert log_file.parent.exists()
    assert orchestrator.logger.name == "DailyReboot"


# ============================================================================
# Candle Fetching Tests
# ============================================================================


@pytest.mark.asyncio
async def test_fetch_candles_cache_hit(tmp_path):
    """Test fetching candles with cache hit (fresh cache)."""
    config = RebootConfig(symbols=["SPY"], initial_candles=500)
    config.cache_dir = str(tmp_path)
    orchestrator = DailyRebootOrchestrator(config)

    # Mock cache hit (fresh)
    mock_candle_data = CandleData(
        symbol="SPY",
        timeframe="1h",
        timestamp=np.array(
            [datetime.now(UTC) for _ in range(500)], dtype="datetime64[ns]"
        ),
        open=np.random.uniform(400, 410, 500),
        high=np.random.uniform(410, 420, 500),
        low=np.random.uniform(390, 400, 500),
        close=np.random.uniform(400, 410, 500),
        volume=np.random.uniform(1e6, 2e6, 500),
    )

    orchestrator.parquet_store.is_cache_fresh = MagicMock(return_value=True)
    orchestrator.parquet_store.load_candles = MagicMock(return_value=mock_candle_data)

    # Initialize mock REST client
    orchestrator.rest_client = AsyncMock()

    await orchestrator._fetch_and_cache_candles("SPY")

    # Verify cache was used (no API call)
    orchestrator.rest_client.fetch_candles.assert_not_called()

    # Verify buffer populated
    assert "SPY" in orchestrator.candle_buffers
    assert orchestrator.candle_buffers["SPY"].size() == 500


@pytest.mark.asyncio
async def test_fetch_candles_cache_miss(tmp_path):
    """Test fetching candles with cache miss (stale/missing)."""
    config = RebootConfig(symbols=["SPY"], initial_candles=500)
    config.cache_dir = str(tmp_path)
    orchestrator = DailyRebootOrchestrator(config)

    # Mock cache miss
    orchestrator.parquet_store.is_cache_fresh = MagicMock(return_value=False)

    # Mock API response
    mock_candles = [
        Candle(
            symbol="SPY",
            timeframe="1h",
            timestamp=datetime.now(UTC),
            open=400.0 + i,
            high=410.0 + i,
            low=390.0 + i,
            close=405.0 + i,
            volume=1e6,
        )
        for i in range(500)
    ]

    mock_rest_client = AsyncMock()
    mock_rest_client.fetch_candles = AsyncMock(return_value=mock_candles)
    orchestrator.rest_client = mock_rest_client

    await orchestrator._fetch_and_cache_candles("SPY")

    # Verify API was called
    mock_rest_client.fetch_candles.assert_called_once_with(
        symbol="SPY", timeframe="1h", limit=500
    )

    # Verify buffer populated
    assert "SPY" in orchestrator.candle_buffers
    assert orchestrator.candle_buffers["SPY"].size() == 500


@pytest.mark.asyncio
async def test_fetch_candles_no_data(tmp_path):
    """Test fetching candles with empty API response."""
    config = RebootConfig(symbols=["SPY"], initial_candles=500)
    config.cache_dir = str(tmp_path)
    orchestrator = DailyRebootOrchestrator(config)

    orchestrator.parquet_store.is_cache_fresh = MagicMock(return_value=False)

    mock_rest_client = AsyncMock()
    mock_rest_client.fetch_candles = AsyncMock(return_value=[])
    orchestrator.rest_client = mock_rest_client

    with pytest.raises(ValueError, match="No candles returned"):
        await orchestrator._fetch_and_cache_candles("SPY")


# ============================================================================
# WebSocket Initialization Tests
# ============================================================================


@pytest.mark.asyncio
async def test_initialize_websocket_success(tmp_path):
    """Test WebSocket initialization success."""
    config = RebootConfig(symbols=["SPY"])
    config.cache_dir = str(tmp_path)
    orchestrator = DailyRebootOrchestrator(config)

    # Mock WebSocketFeed
    mock_ws = AsyncMock()
    mock_ws.connect = AsyncMock()
    mock_ws.subscribe = AsyncMock()

    mock_rest_client = AsyncMock()
    orchestrator.rest_client = mock_rest_client

    with patch(
        "backend.maintenance.daily_reboot.WebSocketFeed", return_value=mock_ws
    ):
        await orchestrator._initialize_websocket("SPY")

    # Verify WebSocket connected and subscribed
    mock_ws.connect.assert_called_once()
    mock_ws.subscribe.assert_called_once_with(["SPY"])

    # Verify stored
    assert "SPY" in orchestrator.ws_feeds
    assert "SPY" in orchestrator.pipelines


# ============================================================================
# System Readiness Tests
# ============================================================================


@pytest.mark.asyncio
async def test_verify_system_readiness_success(tmp_path):
    """Test system readiness verification with all checks passing."""
    config = RebootConfig(symbols=["SPY", "QQQ"], initial_candles=500)
    config.cache_dir = str(tmp_path)
    orchestrator = DailyRebootOrchestrator(config)

    # Mock buffers
    mock_buffer_spy = MagicMock()
    mock_buffer_spy.size = MagicMock(return_value=500)
    orchestrator.candle_buffers["SPY"] = mock_buffer_spy

    mock_buffer_qqq = MagicMock()
    mock_buffer_qqq.size = MagicMock(return_value=500)
    orchestrator.candle_buffers["QQQ"] = mock_buffer_qqq

    # Mock WebSocket feeds
    mock_ws_spy = MagicMock()
    mock_ws_spy.is_stale = MagicMock(return_value=False)
    orchestrator.ws_feeds["SPY"] = mock_ws_spy

    mock_ws_qqq = MagicMock()
    mock_ws_qqq.is_stale = MagicMock(return_value=False)
    orchestrator.ws_feeds["QQQ"] = mock_ws_qqq

    # Should not raise
    await orchestrator._verify_system_readiness()


@pytest.mark.asyncio
async def test_verify_system_readiness_missing_buffer(tmp_path):
    """Test system readiness fails with missing buffer."""
    config = RebootConfig(symbols=["SPY", "QQQ"], initial_candles=500)
    config.cache_dir = str(tmp_path)
    orchestrator = DailyRebootOrchestrator(config)

    # Only SPY buffer, missing QQQ
    mock_buffer = MagicMock()
    mock_buffer.size = MagicMock(return_value=500)
    orchestrator.candle_buffers["SPY"] = mock_buffer

    with pytest.raises(RuntimeError, match="Missing candle buffer for QQQ"):
        await orchestrator._verify_system_readiness()


@pytest.mark.asyncio
async def test_verify_system_readiness_insufficient_candles(tmp_path):
    """Test system readiness fails with insufficient candles."""
    config = RebootConfig(symbols=["SPY"], initial_candles=500)
    config.cache_dir = str(tmp_path)
    orchestrator = DailyRebootOrchestrator(config)

    # Buffer with only 100 candles
    mock_buffer = MagicMock()
    mock_buffer.size = MagicMock(return_value=100)
    orchestrator.candle_buffers["SPY"] = mock_buffer

    with pytest.raises(RuntimeError, match="Insufficient candles for SPY"):
        await orchestrator._verify_system_readiness()


@pytest.mark.asyncio
async def test_verify_system_readiness_stale_websocket(tmp_path):
    """Test system readiness fails with stale WebSocket."""
    config = RebootConfig(symbols=["SPY"], initial_candles=500)
    config.cache_dir = str(tmp_path)
    orchestrator = DailyRebootOrchestrator(config)

    # Valid buffer
    mock_buffer = MagicMock()
    mock_buffer.size = MagicMock(return_value=500)
    orchestrator.candle_buffers["SPY"] = mock_buffer

    # Stale WebSocket
    mock_ws = MagicMock()
    mock_ws.is_stale = MagicMock(return_value=True)
    orchestrator.ws_feeds["SPY"] = mock_ws

    with pytest.raises(RuntimeError, match="WebSocket connection stale for SPY"):
        await orchestrator._verify_system_readiness()


# ============================================================================
# Cleanup Tests
# ============================================================================


@pytest.mark.asyncio
async def test_cleanup_all_resources(tmp_path):
    """Test cleanup closes all resources."""
    config = RebootConfig(symbols=["SPY"])
    config.cache_dir = str(tmp_path)
    orchestrator = DailyRebootOrchestrator(config)

    # Mock WebSocket
    mock_ws = AsyncMock()
    mock_ws.disconnect = AsyncMock()
    orchestrator.ws_feeds["SPY"] = mock_ws

    # Mock REST client
    mock_rest = AsyncMock()
    mock_rest.close = AsyncMock()
    orchestrator.rest_client = mock_rest

    await orchestrator.cleanup()

    # Verify cleanup
    mock_ws.disconnect.assert_called_once()
    mock_rest.close.assert_called_once()


@pytest.mark.asyncio
async def test_cleanup_handles_errors(tmp_path):
    """Test cleanup handles errors gracefully."""
    config = RebootConfig(symbols=["SPY"])
    config.cache_dir = str(tmp_path)
    orchestrator = DailyRebootOrchestrator(config)

    # Mock WebSocket that raises error
    mock_ws = AsyncMock()
    mock_ws.disconnect = AsyncMock(side_effect=Exception("Disconnect error"))
    orchestrator.ws_feeds["SPY"] = mock_ws

    # Should not raise
    await orchestrator.cleanup()


# ============================================================================
# End-to-End Tests
# ============================================================================


@pytest.mark.asyncio
async def test_run_full_reboot_success(tmp_path):
    """Test full reboot sequence succeeds."""
    config = RebootConfig(symbols=["SPY"], initial_candles=100)
    config.cache_dir = str(tmp_path)
    config.log_file = str(tmp_path / "reboot.log")
    orchestrator = DailyRebootOrchestrator(config)

    # Mock all dependencies
    mock_candles = [
        Candle(
            symbol="SPY",
            timeframe="1h",
            timestamp=datetime.now(UTC),
            open=400.0,
            high=410.0,
            low=390.0,
            close=405.0,
            volume=1e6,
        )
        for _ in range(100)
    ]

    mock_rest_client = AsyncMock()
    mock_rest_client.fetch_candles = AsyncMock(return_value=mock_candles)
    mock_rest_client.close = AsyncMock()

    mock_ws = AsyncMock()
    mock_ws.connect = AsyncMock()
    mock_ws.subscribe = AsyncMock()
    mock_ws.is_stale = MagicMock(return_value=False)
    mock_ws.disconnect = AsyncMock()

    with patch(
        "backend.maintenance.daily_reboot.AsyncAPIClient",
        return_value=mock_rest_client,
    ):
        with patch(
            "backend.maintenance.daily_reboot.WebSocketFeed",
            return_value=mock_ws,
        ):
            results = await orchestrator.run()

    # Verify success
    assert results["success"] is True
    assert results["symbols"] == ["SPY"]
    assert "total_duration_seconds" in results["metrics"]
    assert results["metrics"]["total_candles_loaded"] == 100
    assert len(results["errors"]) == 0


@pytest.mark.asyncio
async def test_run_reboot_failure(tmp_path):
    """Test reboot sequence fails and reports error."""
    config = RebootConfig(symbols=["SPY"], initial_candles=100)
    config.cache_dir = str(tmp_path)
    config.log_file = str(tmp_path / "reboot.log")
    orchestrator = DailyRebootOrchestrator(config)

    # Mock REST client that raises error
    mock_rest_client = AsyncMock()
    mock_rest_client.fetch_candles = AsyncMock(
        side_effect=Exception("API connection failed")
    )
    mock_rest_client.close = AsyncMock()

    with patch(
        "backend.maintenance.daily_reboot.AsyncAPIClient",
        return_value=mock_rest_client,
    ):
        with pytest.raises(Exception, match="API connection failed"):
            await orchestrator.run()


# ============================================================================
# Performance Tests
# ============================================================================


@pytest.mark.asyncio
async def test_reboot_performance_single_symbol(tmp_path):
    """Test reboot completes in reasonable time for single symbol."""
    config = RebootConfig(symbols=["SPY"], initial_candles=500)
    config.cache_dir = str(tmp_path)
    config.log_file = str(tmp_path / "reboot.log")
    orchestrator = DailyRebootOrchestrator(config)

    # Mock dependencies
    mock_candles = [
        Candle(
            symbol="SPY",
            timeframe="1h",
            timestamp=datetime.now(UTC),
            open=400.0,
            high=410.0,
            low=390.0,
            close=405.0,
            volume=1e6,
        )
        for _ in range(500)
    ]

    mock_rest_client = AsyncMock()
    mock_rest_client.fetch_candles = AsyncMock(return_value=mock_candles)
    mock_rest_client.close = AsyncMock()

    mock_ws = AsyncMock()
    mock_ws.connect = AsyncMock()
    mock_ws.subscribe = AsyncMock()
    mock_ws.is_stale = MagicMock(return_value=False)
    mock_ws.disconnect = AsyncMock()

    with patch(
        "backend.maintenance.daily_reboot.AsyncAPIClient",
        return_value=mock_rest_client,
    ):
        with patch(
            "backend.maintenance.daily_reboot.WebSocketFeed",
            return_value=mock_ws,
        ):
            start = datetime.now()
            results = await orchestrator.run()
            duration = (datetime.now() - start).total_seconds()

    # Verify reasonable performance (<10 seconds for mocked version)
    assert duration < 10.0
    assert results["success"] is True


# ============================================================================
# Summary
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
