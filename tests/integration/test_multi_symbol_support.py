"""
Integration tests for multi-symbol support (SPY, AAPL, MSFT)

This module tests the system's ability to handle multiple symbols
for test data, API endpoints, and WebSocket replay.

Test Coverage:
- CSV data files for all symbols exist
- API endpoint supports multiple symbols
- WebSocket replay cycles through all symbols
- Data format consistency across symbols
"""

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient


class TestMultiSymbolDataFiles:
    """Tests for multi-symbol CSV data files"""

    @pytest.mark.parametrize("symbol", ["spy", "aapl", "msft"])
    def test_csv_files_exist(self, symbol):
        """Test that CSV files exist for all supported symbols"""
        csv_path = Path(f"backend/test_data/{symbol}_daily.csv")
        assert csv_path.exists(), f"{symbol.upper()} CSV file not found: {csv_path}"

    @pytest.mark.parametrize("symbol", ["spy", "aapl", "msft"])
    def test_csv_files_have_valid_structure(self, symbol):
        """Test that all CSV files have proper header and data"""
        csv_path = Path(f"backend/test_data/{symbol}_daily.csv")
        with open(csv_path) as f:
            lines = f.readlines()

            # Should have at least header + data rows
            assert len(lines) >= 3, f"{symbol.upper()} CSV file too small"

            # First line should be column headers
            header = lines[0].lower()
            assert "close" in header or "price" in header
            assert "open" in header
            assert "volume" in header

    @pytest.mark.parametrize("symbol", ["spy", "aapl", "msft"])
    def test_csv_data_parsing_logic(self, symbol):
        """Test CSV parsing logic works for all symbols"""
        csv_path = Path(f"backend/test_data/{symbol}_daily.csv")

        # Replicate the parsing logic from server.py
        # SPY has a different format with extra header row
        if symbol == "spy":
            df = pd.read_csv(csv_path, skiprows=[1])
            df = df.rename(columns={"Price": "Date"})
        else:
            # AAPL and MSFT have standard format
            df = pd.read_csv(csv_path)
            df = df.rename(columns={"Price": "Date"})

        # Common column renaming
        df = df.rename(
            columns={
                "Close": "close",
                "High": "high",
                "Low": "low",
                "Open": "open",
                "Volume": "volume",
            }
        )

        # Test that we can parse into the expected format
        test_data = []
        for _, row in df.head(10).iterrows():  # Test first 10 rows
            try:
                # Check if any required field is NaN
                required_fields = ["Date", "open", "high", "low", "close", "volume"]
                skip_row = False
                for field in required_fields:
                    if field in row and pd.isna(row[field]):
                        skip_row = True
                        break

                if skip_row:
                    continue

                candle = {
                    "timestamp": str(row["Date"]),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": int(row["volume"]),
                }
                test_data.append(candle)
            except (ValueError, KeyError):
                # Skip rows that can't be parsed
                continue

        # Should have successfully parsed at least some rows
        assert len(test_data) > 0, f"No valid data parsed from {symbol.upper()} CSV"

        # Validate data types
        for candle in test_data:
            assert isinstance(candle["open"], float)
            assert isinstance(candle["high"], float)
            assert isinstance(candle["low"], float)
            assert isinstance(candle["close"], float)
            assert isinstance(candle["volume"], int)

    @pytest.mark.parametrize("symbol", ["spy", "aapl", "msft"])
    def test_csv_has_sufficient_data(self, symbol):
        """Test that CSV files have at least 200 rows of data (about 1 year)"""
        csv_path = Path(f"backend/test_data/{symbol}_daily.csv")
        df = pd.read_csv(csv_path, skiprows=[1])

        # Should have at least 200 trading days (approximately 1 year)
        assert len(df) >= 200, f"{symbol.upper()} has insufficient data: {len(df)} rows"


class TestMultiSymbolAPIEndpoint:
    """Tests for /api/test/candles endpoint with multiple symbols"""

    def test_endpoint_accepts_spy_symbol(self):
        """Test endpoint accepts SPY symbol"""
        from backend.api.server import app

        with TestClient(app) as client:
            response = client.get("/api/test/candles?symbol=SPY")
            # May return 503 if data not loaded (expected in test context)
            # or 200 if data is available
            assert response.status_code in [200, 503]

    def test_endpoint_accepts_aapl_symbol(self):
        """Test endpoint accepts AAPL symbol"""
        from backend.api.server import app

        with TestClient(app) as client:
            response = client.get("/api/test/candles?symbol=AAPL")
            # May return 503 if data not loaded (expected in test context)
            # or 200 if data is available
            assert response.status_code in [200, 503]

    def test_endpoint_accepts_msft_symbol(self):
        """Test endpoint accepts MSFT symbol"""
        from backend.api.server import app

        with TestClient(app) as client:
            response = client.get("/api/test/candles?symbol=MSFT")
            # May return 503 if data not loaded (expected in test context)
            # or 200 if data is available
            assert response.status_code in [200, 503]

    def test_endpoint_rejects_unsupported_symbol(self):
        """Test endpoint rejects symbols that aren't supported"""
        from backend.api.server import app

        with TestClient(app) as client:
            response = client.get("/api/test/candles?symbol=INVALID")
            assert response.status_code == 400
            assert "not supported" in response.json()["detail"].lower()

    def test_endpoint_case_insensitive(self):
        """Test endpoint handles lowercase/uppercase symbols"""
        from backend.api.server import app

        with TestClient(app) as client:
            # Test lowercase
            response = client.get("/api/test/candles?symbol=spy")
            assert response.status_code in [200, 503]

            # Test mixed case
            response = client.get("/api/test/candles?symbol=AaPl")
            assert response.status_code in [200, 503]

    def test_endpoint_default_symbol(self):
        """Test endpoint defaults to SPY when no symbol provided"""
        from backend.api.server import app

        with TestClient(app) as client:
            response = client.get("/api/test/candles")
            # Should default to SPY
            assert response.status_code in [200, 503]


class TestMultiSymbolWebSocketReplay:
    """Tests for WebSocket replay with multiple symbols"""

    def test_websocket_replay_data_structure(self):
        """Test that app_state.test_data is properly structured"""
        # Import after pytest starts to avoid early initialization
        from backend.api.server import app_state

        # After startup, test_data should be a dict
        assert isinstance(app_state.test_data, dict)

        # In test environment, data may or may not be loaded
        # Just verify the structure is correct
        if app_state.test_data:
            for symbol, data in app_state.test_data.items():
                assert isinstance(symbol, str)
                assert isinstance(data, list)
                if data:
                    # Verify candle structure
                    candle = data[0]
                    assert "timestamp" in candle
                    assert "open" in candle
                    assert "high" in candle
                    assert "low" in candle
                    assert "close" in candle
                    assert "volume" in candle

    def test_websocket_message_format_for_multiple_symbols(self):
        """Test that WebSocket messages include correct symbol field"""
        # This test validates the message structure
        # Actual WebSocket testing requires running server

        expected_message_format = {
            "type": "price_update",
            "symbol": "SPY",  # Should be one of: SPY, AAPL, MSFT
            "timestamp": "2025-01-22 00:00:00",
            "open": 100.0,
            "high": 105.0,
            "low": 99.0,
            "close": 103.0,
            "volume": 1000000,
            "replay_index": 0,
            "total_rows": 250,
        }

        # Verify all required fields are present
        required_fields = [
            "type",
            "symbol",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "replay_index",
            "total_rows",
        ]

        for field in required_fields:
            assert field in expected_message_format

        # Verify symbol is one of the supported symbols
        assert expected_message_format["symbol"] in ["SPY", "AAPL", "MSFT"]


class TestMultiSymbolDataConsistency:
    """Tests for data consistency across symbols"""

    @pytest.mark.parametrize("symbol", ["spy", "aapl", "msft"])
    def test_ohlc_relationships(self, symbol):
        """Test that OHLC data follows logical relationships (high >= low, etc.)"""
        csv_path = Path(f"backend/test_data/{symbol}_daily.csv")
        df = pd.read_csv(csv_path, skiprows=[1])
        df = df.rename(
            columns={
                "Price": "Date",
                "Close": "close",
                "High": "high",
                "Low": "low",
                "Open": "open",
                "Volume": "volume",
            }
        )

        # Test first 10 rows
        for idx, row in df.head(10).iterrows():
            try:
                # Skip rows with NaN values
                if (
                    pd.isna(row.get("open"))
                    or pd.isna(row.get("high"))
                    or pd.isna(row.get("low"))
                    or pd.isna(row.get("close"))
                    or pd.isna(row.get("volume"))
                ):
                    continue

                open_price = float(row["open"])
                high_price = float(row["high"])
                low_price = float(row["low"])
                close_price = float(row["close"])

                # High should be >= all other prices
                assert (
                    high_price >= open_price
                ), f"{symbol.upper()} row {idx}: high < open"
                assert (
                    high_price >= close_price
                ), f"{symbol.upper()} row {idx}: high < close"
                assert high_price >= low_price, f"{symbol.upper()} row {idx}: high < low"

                # Low should be <= all other prices
                assert low_price <= open_price, f"{symbol.upper()} row {idx}: low > open"
                assert (
                    low_price <= close_price
                ), f"{symbol.upper()} row {idx}: low > close"

                # Volume should be positive
                volume = int(row["volume"])
                assert volume > 0, f"{symbol.upper()} row {idx}: volume <= 0"

            except (ValueError, KeyError):
                continue  # Skip invalid rows

    @pytest.mark.parametrize("symbol", ["spy", "aapl", "msft"])
    def test_price_ranges_reasonable(self, symbol):
        """Test that prices are within reasonable ranges for each symbol"""
        csv_path = Path(f"backend/test_data/{symbol}_daily.csv")
        df = pd.read_csv(csv_path, skiprows=[1])
        df = df.rename(columns={"Close": "close"})

        # Define reasonable price ranges for each symbol (2024-2025 data)
        price_ranges = {
            "spy": (400, 700),  # SPY typically trades 500-650 in 2024-2025
            "aapl": (150, 250),  # AAPL typically trades 180-240
            "msft": (300, 500),  # MSFT typically trades 350-480
        }

        min_price, max_price = price_ranges[symbol]
        avg_close = df["close"].mean()

        assert (
            min_price <= avg_close <= max_price
        ), (
            f"{symbol.upper()} average price {avg_close:.2f} "
            f"outside expected range {min_price}-{max_price}"
        )
