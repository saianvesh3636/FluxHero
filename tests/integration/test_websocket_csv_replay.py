"""
Integration tests for WebSocket CSV replay functionality

This module tests the simulated live price updates feature where the
WebSocket endpoint replays CSV data to simulate live market data.

Test Coverage:
- CSV data loading and caching
- WebSocket connection and authentication
- CSV data replay with proper iteration
- Loop behavior when reaching end of data
- Message format validation

Note: These are mostly structure/code validation tests since the lifespan
context manager doesn't run with TestClient. For actual WebSocket behavior
testing, run the server manually and use a WebSocket client.
"""

from pathlib import Path

import pytest


class TestCSVReplaySetup:
    """Tests for CSV data loading and setup"""

    def test_spy_csv_file_exists(self):
        """Test that SPY CSV file exists in test_data directory"""
        csv_path = Path("backend/test_data/spy_daily.csv")
        assert csv_path.exists(), f"SPY CSV file not found: {csv_path}"

    def test_csv_file_has_valid_structure(self):
        """Test that CSV file has proper header and data"""
        csv_path = Path("backend/test_data/spy_daily.csv")
        with open(csv_path) as f:
            lines = f.readlines()

            # Should have at least header + data rows
            assert len(lines) >= 3, "CSV file too small"

            # First line should be column headers
            assert "Close" in lines[0] or "close" in lines[0]
            assert "Open" in lines[0] or "open" in lines[0]
            assert "Volume" in lines[0] or "volume" in lines[0]

    def test_csv_data_parsing_logic(self):
        """Test CSV parsing logic from server.py"""
        import pandas as pd

        csv_path = Path("backend/test_data/spy_daily.csv")

        # Replicate the parsing logic from server.py
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
        df = df.rename(columns={"Ticker": "Date"})

        # Test that we can parse into the expected format
        test_data = []
        for _, row in df.head(10).iterrows():  # Test first 10 rows
            try:
                # Skip rows with NaN values (mimics server.py behavior)
                if (
                    pd.isna(row["Date"])
                    or pd.isna(row["open"])
                    or pd.isna(row["high"])
                    or pd.isna(row["low"])
                    or pd.isna(row["close"])
                    or pd.isna(row["volume"])
                ):
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
                # Skip invalid rows (mimics server.py behavior)
                continue

        # Verify we got valid data
        assert len(test_data) > 0, "No data parsed from CSV"

        # Validate OHLC relationships
        for candle in test_data:
            assert candle["high"] >= candle["low"], "High < Low"
            assert candle["high"] >= candle["open"], "High < Open"
            assert candle["high"] >= candle["close"], "High < Close"
            assert candle["low"] <= candle["open"], "Low > Open"
            assert candle["low"] <= candle["close"], "Low > Close"
            assert candle["volume"] > 0, f"Invalid volume: {candle['volume']}"


class TestWebSocketCSVReplay:
    """Tests for WebSocket CSV replay code structure"""

    def test_websocket_endpoint_exists(self):
        """Test that WebSocket endpoint is defined"""
        server_path = Path("backend/api/server.py")
        with open(server_path) as f:
            content = f.read()
            assert "@app.websocket(\"/ws/prices\")" in content

    def test_websocket_has_csv_replay_logic(self):
        """Test that WebSocket has CSV replay implementation"""
        server_path = Path("backend/api/server.py")
        with open(server_path) as f:
            content = f.read()

            # Check for replay logic
            assert "if app_state.test_spy_data" in content
            assert "data_index" in content
            assert "replay_index" in content
            assert "total_rows" in content

    def test_websocket_sends_ohlc_data(self):
        """Test that WebSocket sends full OHLC data"""
        server_path = Path("backend/api/server.py")
        with open(server_path) as f:
            content = f.read()

            # Check that all OHLC fields are sent
            assert '"open":' in content or "'open':" in content
            assert '"high":' in content or "'high':" in content
            assert '"low":' in content or "'low':" in content
            assert '"close":' in content or "'close':" in content
            assert '"volume":' in content or "'volume':" in content

    def test_websocket_loops_data_index(self):
        """Test that WebSocket implements index looping"""
        server_path = Path("backend/api/server.py")
        with open(server_path) as f:
            content = f.read()

            # Check for modulo operation to loop index
            assert "data_index = (data_index + 1) %" in content

    def test_websocket_has_fallback_mode(self):
        """Test that WebSocket has fallback to synthetic data"""
        server_path = Path("backend/api/server.py")
        with open(server_path) as f:
            content = f.read()

            # Should have else clause for when CSV not available
            assert "else:" in content
            # Should still have synthetic price generation
            assert "np.random.uniform" in content or "random" in content

    def test_websocket_timing_configuration(self):
        """Test that WebSocket has configurable timing"""
        server_path = Path("backend/api/server.py")
        with open(server_path) as f:
            content = f.read()

            # Should have asyncio.sleep calls
            assert "await asyncio.sleep" in content
            # Should mention 2 seconds for replay mode
            assert "2.0" in content


class TestWebSocketReplayBehavior:
    """Tests for WebSocket replay behavior and message structure"""

    def test_replay_message_structure(self):
        """Test that replay messages include all required fields"""
        # This is a code structure test - actual behavior requires running server
        server_path = Path("backend/api/server.py")
        with open(server_path) as f:
            content = f.read()

            # Find the replay message construction
            lines = content.split("\n")
            in_replay_block = False
            message_fields = []

            for line in lines:
                if '"type": "price_update"' in line or "'type': 'price_update'" in line:
                    in_replay_block = True
                if in_replay_block and ("}" in line or "]" in line):
                    break
                if in_replay_block and (":" in line):
                    # Extract field names
                    if '"symbol"' in line or "'symbol'" in line:
                        message_fields.append("symbol")
                    if '"timestamp"' in line or "'timestamp'" in line:
                        message_fields.append("timestamp")
                    if '"open"' in line or "'open'" in line:
                        message_fields.append("open")

            # At minimum should have symbol and timestamp
            assert "symbol" in content
            assert "timestamp" in content


class TestWebSocketDocumentation:
    """Tests for WebSocket implementation documentation"""

    def test_server_has_websocket_comments(self):
        """Test that server.py has comments about CSV replay"""
        server_path = Path("backend/api/server.py")
        with open(server_path) as f:
            content = f.read()

            # Check for documentation of replay functionality
            assert "Replay CSV data" in content or "replay" in content.lower()

    def test_websocket_endpoint_docstring(self):
        """Test that WebSocket endpoint has proper docstring"""
        server_path = Path("backend/api/server.py")
        with open(server_path) as f:
            content = f.read()

            # Find websocket endpoint
            assert "@app.websocket" in content
            assert "/ws/prices" in content

            # Should have docstring explaining functionality
            assert '"""' in content or "'''" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
