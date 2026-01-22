"""
Daily Reboot Script for FluxHero Trading System.

This script performs a daily system reboot at 9:00 AM EST:
1. Fetch last 500 candles for configured symbols
2. Reconnect WebSocket feeds
3. Initialize trading system components
4. Resume trading operations

Designed to be run via cron job or system scheduler.

Usage:
    python -m fluxhero.backend.maintenance.daily_reboot --config config.json
    python -m fluxhero.backend.maintenance.daily_reboot --symbols SPY,QQQ --timeframe 1h
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.core.config import get_settings
from backend.data.fetcher import AsyncAPIClient, DataPipeline, WebSocketFeed
from backend.storage.candle_buffer import CandleBuffer
from backend.storage.parquet_store import ParquetStore

# ============================================================================
# Configuration
# ============================================================================


class RebootConfig:
    """Configuration for daily reboot script."""

    def __init__(
        self,
        symbols: list[str],
        timeframe: str | None = None,
        initial_candles: int | None = None,
        api_url: str | None = None,
        ws_url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        cache_dir: str | None = None,
        log_file: str | None = None,
    ):
        """
        Initialize reboot configuration.

        Args:
            symbols: List of symbols to fetch (e.g., ["SPY", "QQQ"])
            timeframe: Candle timeframe (e.g., "1h", "1d"), defaults to centralized config
            initial_candles: Number of candles to fetch on startup, defaults to centralized config
            api_url: REST API base URL, defaults to centralized config
            ws_url: WebSocket feed URL, defaults to centralized config
            api_key: API key for authentication, defaults to centralized config
            api_secret: API secret for authentication, defaults to centralized config
            cache_dir: Directory for Parquet cache files, defaults to centralized config
            log_file: Path to log file, defaults to centralized config
        """
        settings = get_settings()

        self.symbols = symbols
        self.timeframe = timeframe if timeframe is not None else settings.default_timeframe
        self.initial_candles = (
            initial_candles if initial_candles is not None else settings.initial_candles
        )
        self.api_url = api_url if api_url is not None else settings.alpaca_api_url
        self.ws_url = ws_url if ws_url is not None else settings.alpaca_ws_url
        self.api_key = api_key if api_key is not None else settings.alpaca_api_key
        self.api_secret = api_secret if api_secret is not None else settings.alpaca_api_secret
        self.cache_dir = cache_dir if cache_dir is not None else settings.cache_dir
        self.log_file = log_file if log_file is not None else settings.log_file

    @classmethod
    def from_file(cls, config_path: str) -> "RebootConfig":
        """
        Load configuration from JSON file.

        Args:
            config_path: Path to JSON config file

        Returns:
            RebootConfig instance
        """
        with open(config_path) as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "RebootConfig":
        """
        Create configuration from command-line arguments.

        Args:
            args: Parsed command-line arguments

        Returns:
            RebootConfig instance
        """
        symbols = args.symbols.split(",") if isinstance(args.symbols, str) else args.symbols
        return cls(
            symbols=symbols,
            timeframe=args.timeframe,
            initial_candles=args.initial_candles,
            api_url=args.api_url,
            ws_url=args.ws_url,
            api_key=args.api_key,
            api_secret=args.api_secret,
            cache_dir=args.cache_dir,
            log_file=args.log_file,
        )


# ============================================================================
# Reboot Orchestrator
# ============================================================================


class DailyRebootOrchestrator:
    """
    Orchestrates daily system reboot process.

    Performs:
    1. Setup logging
    2. Check cache freshness
    3. Fetch historical data (500 candles)
    4. Initialize WebSocket connections
    5. Populate candle buffers
    6. Verify system readiness
    """

    def __init__(self, config: RebootConfig):
        """
        Initialize reboot orchestrator.

        Args:
            config: Reboot configuration
        """
        self.config = config
        self.logger = self._setup_logging()
        self.parquet_store = ParquetStore(cache_dir=config.cache_dir)
        self.candle_buffers: dict[str, CandleBuffer] = {}
        self.rest_client: AsyncAPIClient | None = None
        self.ws_feeds: dict[str, WebSocketFeed] = {}
        self.pipelines: dict[str, DataPipeline] = {}

    def _setup_logging(self) -> logging.Logger:
        """
        Setup logging configuration.

        Returns:
            Configured logger instance
        """
        # Create log directory if needed
        log_path = Path(self.config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(self.config.log_file),
                logging.StreamHandler(sys.stdout),
            ],
        )
        return logging.getLogger("DailyReboot")

    async def run(self) -> dict[str, Any]:
        """
        Execute daily reboot sequence.

        Returns:
            Dictionary with reboot status and metrics

        Raises:
            Exception: If critical reboot step fails
        """
        self.logger.info("=" * 80)
        self.logger.info("FluxHero Daily Reboot - Starting")
        self.logger.info(f"Timestamp: {datetime.now(UTC).isoformat()}")
        self.logger.info(f"Symbols: {self.config.symbols}")
        self.logger.info(f"Timeframe: {self.config.timeframe}")
        self.logger.info("=" * 80)

        start_time = datetime.now()
        results = {
            "success": False,
            "timestamp": start_time.isoformat(),
            "symbols": self.config.symbols,
            "metrics": {},
            "errors": [],
        }

        try:
            # Step 1: Initialize REST client
            self.logger.info("Step 1/5: Initializing REST API client...")
            self.rest_client = AsyncAPIClient(
                base_url=self.config.api_url,
                api_key=self.config.api_key,
                api_secret=self.config.api_secret,
            )
            self.logger.info("REST client initialized")

            # Step 2: Fetch historical data for each symbol
            self.logger.info("Step 2/5: Fetching historical candle data...")
            fetch_start = datetime.now()

            for symbol in self.config.symbols:
                await self._fetch_and_cache_candles(symbol)

            fetch_duration = (datetime.now() - fetch_start).total_seconds()
            results["metrics"]["fetch_duration_seconds"] = fetch_duration
            self.logger.info(f"Historical data fetched in {fetch_duration:.2f}s")

            # Step 3: Initialize WebSocket connections
            self.logger.info("Step 3/5: Initializing WebSocket connections...")
            ws_start = datetime.now()

            for symbol in self.config.symbols:
                await self._initialize_websocket(symbol)

            ws_duration = (datetime.now() - ws_start).total_seconds()
            results["metrics"]["websocket_init_duration_seconds"] = ws_duration
            self.logger.info(f"WebSocket connections established in {ws_duration:.2f}s")

            # Step 4: Verify system readiness
            self.logger.info("Step 4/5: Verifying system readiness...")
            await self._verify_system_readiness()
            self.logger.info("System verification complete")

            # Step 5: Calculate final metrics
            total_duration = (datetime.now() - start_time).total_seconds()
            results["metrics"]["total_duration_seconds"] = total_duration
            results["metrics"]["total_candles_loaded"] = sum(
                buf.size() for buf in self.candle_buffers.values()
            )
            results["success"] = True

            self.logger.info("=" * 80)
            self.logger.info("Daily Reboot SUCCESSFUL")
            self.logger.info(f"Total Duration: {total_duration:.2f}s")
            self.logger.info(f"Candles Loaded: {results['metrics']['total_candles_loaded']}")
            self.logger.info("=" * 80)

        except Exception as e:
            self.logger.error(f"Daily reboot FAILED: {e}", exc_info=True)
            results["errors"].append(str(e))
            results["success"] = False
            raise

        finally:
            # Cleanup
            if self.rest_client:
                await self.rest_client.close()

        return results

    async def _fetch_and_cache_candles(self, symbol: str) -> None:
        """
        Fetch historical candles and update cache.

        Args:
            symbol: Symbol to fetch (e.g., "SPY")
        """
        self.logger.info(f"  Fetching {self.config.initial_candles} candles for {symbol}...")

        # Check cache freshness
        if self.parquet_store.is_cache_fresh(symbol, self.config.timeframe):
            self.logger.info(f"  Cache hit for {symbol} (fresh)")
            candle_data = self.parquet_store.load_candles(symbol, self.config.timeframe)

            if candle_data and len(candle_data.close) >= self.config.initial_candles:
                self.logger.info(f"  Loaded {len(candle_data.close)} candles from cache")
                self.candle_buffers[symbol] = self._populate_buffer(candle_data)
                return

        # Cache miss or stale - fetch from API
        self.logger.info(f"  Cache miss/stale for {symbol}, fetching from API...")

        if not self.rest_client:
            raise RuntimeError("REST client not initialized")

        candles = await self.rest_client.fetch_candles(
            symbol=symbol,
            timeframe=self.config.timeframe,
            limit=self.config.initial_candles,
        )

        if not candles:
            raise ValueError(f"No candles returned for {symbol}")

        self.logger.info(f"  Fetched {len(candles)} candles from API")

        # Save to cache
        import numpy as np

        from backend.storage.parquet_store import CandleData

        cache_data = CandleData(
            symbol=symbol,
            timeframe=self.config.timeframe,
            timestamp=np.array([c.timestamp for c in candles], dtype="datetime64[ns]"),
            open=np.array([c.open for c in candles], dtype=np.float64),
            high=np.array([c.high for c in candles], dtype=np.float64),
            low=np.array([c.low for c in candles], dtype=np.float64),
            close=np.array([c.close for c in candles], dtype=np.float64),
            volume=np.array([c.volume for c in candles], dtype=np.float64),
        )
        self.parquet_store.save_candles(cache_data)
        self.logger.info(f"  Cached {len(candles)} candles to Parquet")

        # Populate buffer
        self.candle_buffers[symbol] = self._populate_buffer(cache_data)

    def _populate_buffer(self, candle_data) -> CandleBuffer:
        """
        Populate CandleBuffer from CandleData.

        Args:
            candle_data: CandleData instance

        Returns:
            Populated CandleBuffer
        """
        buffer = CandleBuffer(max_size=self.config.initial_candles)

        for i in range(len(candle_data.close)):
            buffer.add_candle(
                timestamp=candle_data.timestamp[i],
                open=float(candle_data.open[i]),
                high=float(candle_data.high[i]),
                low=float(candle_data.low[i]),
                close=float(candle_data.close[i]),
                volume=float(candle_data.volume[i]),
            )

        return buffer

    async def _initialize_websocket(self, symbol: str) -> None:
        """
        Initialize WebSocket feed for symbol.

        Args:
            symbol: Symbol to subscribe to
        """
        self.logger.info(f"  Connecting WebSocket for {symbol}...")

        ws_feed = WebSocketFeed(
            ws_url=self.config.ws_url,
            api_key=self.config.api_key,
            api_secret=self.config.api_secret,
        )

        await ws_feed.connect()
        await ws_feed.subscribe([symbol])

        self.ws_feeds[symbol] = ws_feed
        self.logger.info(f"  WebSocket connected and subscribed to {symbol}")

        # Create data pipeline
        if self.rest_client:
            pipeline = DataPipeline(self.rest_client, ws_feed)
            self.pipelines[symbol] = pipeline

    async def _verify_system_readiness(self) -> None:
        """
        Verify system is ready for trading.

        Raises:
            RuntimeError: If system verification fails
        """
        # Check all buffers populated
        for symbol in self.config.symbols:
            if symbol not in self.candle_buffers:
                raise RuntimeError(f"Missing candle buffer for {symbol}")

            buffer = self.candle_buffers[symbol]
            if buffer.size() < self.config.initial_candles:
                raise RuntimeError(
                    f"Insufficient candles for {symbol}: "
                    f"{buffer.size()} < {self.config.initial_candles}"
                )

            self.logger.info(f"  {symbol}: {buffer.size()} candles loaded")

        # Check all WebSocket connections
        for symbol in self.config.symbols:
            if symbol not in self.ws_feeds:
                raise RuntimeError(f"Missing WebSocket feed for {symbol}")

            ws_feed = self.ws_feeds[symbol]
            if ws_feed.is_stale():
                raise RuntimeError(f"WebSocket connection stale for {symbol}")

            self.logger.info(f"  {symbol}: WebSocket connection healthy")

    async def cleanup(self) -> None:
        """
        Cleanup resources (WebSocket connections, clients).
        """
        self.logger.info("Cleaning up resources...")

        # Close WebSocket feeds
        for symbol, ws_feed in self.ws_feeds.items():
            try:
                await ws_feed.disconnect()
                self.logger.info(f"  Closed WebSocket for {symbol}")
            except Exception as e:
                self.logger.warning(f"  Error closing WebSocket for {symbol}: {e}")

        # Close REST client
        if self.rest_client:
            try:
                await self.rest_client.close()
                self.logger.info("  Closed REST client")
            except Exception as e:
                self.logger.warning(f"  Error closing REST client: {e}")


# ============================================================================
# CLI Entry Point
# ============================================================================


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="FluxHero Daily Reboot Script - 9:00 AM EST System Initialization"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON configuration file",
    )

    parser.add_argument(
        "--symbols",
        type=str,
        default="SPY",
        help="Comma-separated list of symbols (default: SPY)",
    )

    parser.add_argument(
        "--timeframe",
        type=str,
        default=None,
        help="Candle timeframe (default: from config)",
    )

    parser.add_argument(
        "--initial-candles",
        type=int,
        default=None,
        help="Number of candles to fetch (default: from config)",
    )

    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="REST API base URL (default: from config)",
    )

    parser.add_argument(
        "--ws-url",
        type=str,
        default=None,
        help="WebSocket feed URL (default: from config)",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication (default: from config)",
    )

    parser.add_argument(
        "--api-secret",
        type=str,
        default=None,
        help="API secret for authentication (default: from config)",
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for Parquet files (default: from config)",
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (default: from config)",
    )

    return parser.parse_args()


async def main() -> int:
    """
    Main entry point for daily reboot script.

    Returns:
        Exit code (0 = success, 1 = failure)
    """
    args = parse_args()

    # Load configuration
    if args.config:
        config = RebootConfig.from_file(args.config)
    else:
        config = RebootConfig.from_args(args)

    # Run reboot
    orchestrator = DailyRebootOrchestrator(config)

    try:
        results = await orchestrator.run()

        if results["success"]:
            print("\n✓ Daily reboot completed successfully")
            return 0
        else:
            print("\n✗ Daily reboot failed")
            return 1

    except Exception as e:
        print(f"\n✗ Daily reboot failed with error: {e}")
        return 1

    finally:
        await orchestrator.cleanup()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
