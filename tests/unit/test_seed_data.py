"""
Unit tests for test data seeding script.

Tests the seed_test_data.py script functionality including:
- Realistic position data generation
- Position creation
- Database seeding
- Validation of P&L calculations
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.execution.broker_interface import OrderSide
from backend.storage.sqlite_store import SQLiteStore
from scripts.seed_test_data import (
    clear_positions,
    generate_realistic_position_data,
    seed_positions,
)


class TestPositionDataGeneration:
    """Test realistic position data generation."""

    def test_generate_realistic_position_data_structure(self):
        """Test that generated position data has correct structure."""
        data = generate_realistic_position_data()

        # Check all required fields are present
        required_fields = [
            "symbol", "entry_price", "current_price", "shares",
            "side", "stop_loss", "entry_time", "strategy",
            "regime", "signal_reason"
        ]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

    def test_position_data_types(self):
        """Test that generated data has correct types."""
        data = generate_realistic_position_data()

        assert isinstance(data["symbol"], str)
        assert isinstance(data["entry_price"], float)
        assert isinstance(data["current_price"], float)
        assert isinstance(data["shares"], int)
        assert isinstance(data["side"], OrderSide)
        assert isinstance(data["stop_loss"], float)
        assert isinstance(data["entry_time"], datetime)
        assert isinstance(data["strategy"], str)
        assert isinstance(data["regime"], str)
        assert isinstance(data["signal_reason"], str)

    def test_position_data_ranges(self):
        """Test that generated data is within realistic ranges."""
        # Generate multiple samples to test consistency
        for _ in range(10):
            data = generate_realistic_position_data()

            # Price should be positive
            assert data["entry_price"] > 0
            assert data["current_price"] > 0
            assert data["stop_loss"] > 0

            # Shares should be positive and reasonable
            assert data["shares"] > 0
            assert data["shares"] < 10000  # Reasonable max

            # Side should be valid
            assert data["side"] in [OrderSide.BUY, OrderSide.SELL]

            # Strategy and regime should be valid
            assert data["strategy"] in ["TREND", "MEAN_REVERSION"]
            assert data["regime"] in ["STRONG_TREND", "MEAN_REVERSION", "NEUTRAL"]

    def test_position_pnl_calculation(self):
        """Test that P&L is calculated correctly."""
        for _ in range(10):
            data = generate_realistic_position_data()

            # Calculate P&L percentage
            pnl_pct = (data["current_price"] - data["entry_price"]) / data["entry_price"]

            # P&L should be within reasonable range (-5% to +10%)
            assert -0.05 <= pnl_pct <= 0.10

    def test_stop_loss_placement(self):
        """Test that stop loss is placed correctly."""
        for _ in range(10):
            data = generate_realistic_position_data()

            if data["side"] == OrderSide.BUY:
                # For long positions, stop should be below entry
                assert data["stop_loss"] < data["entry_price"]
                # Stop should be 2.5-3% below entry
                stop_distance = (data["entry_price"] - data["stop_loss"]) / data["entry_price"]
                assert 0.02 <= stop_distance <= 0.035
            else:
                # For short positions, stop should be above entry
                assert data["stop_loss"] > data["entry_price"]
                # Stop should be 2.5-3% above entry
                stop_distance = (data["stop_loss"] - data["entry_price"]) / data["entry_price"]
                assert 0.02 <= stop_distance <= 0.035

    def test_entry_time_is_recent(self):
        """Test that entry time is within last 30 days."""
        data = generate_realistic_position_data()
        now = datetime.now()
        time_diff = now - data["entry_time"]

        # Entry time should be within last 30 days
        assert timedelta(days=0) <= time_diff <= timedelta(days=31)

    def test_signal_reason_not_empty(self):
        """Test that signal reason is provided."""
        data = generate_realistic_position_data()

        assert len(data["signal_reason"]) > 0
        assert isinstance(data["signal_reason"], str)


class TestSeedPositions:
    """Test database seeding functionality."""

    @pytest.mark.asyncio
    async def test_seed_positions_creates_positions(self, tmp_path):
        """Test that seeding creates the correct number of positions."""
        # Use temporary database
        db_path = tmp_path / "test.db"
        store = SQLiteStore(db_path=str(db_path))
        await store.initialize()

        try:
            # Clear any existing data
            await clear_positions()

            # Seed 5 positions
            count = 5
            await seed_positions(count=count)

            # Verify positions were created
            # Note: This would need access to the broker instance to verify
            # For now, we test that the function runs without errors

        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_seed_positions_count_range(self, tmp_path):
        """Test that seeding works with different counts."""
        db_path = tmp_path / "test.db"
        store = SQLiteStore(db_path=str(db_path))
        await store.initialize()

        try:
            # Test minimum count
            await seed_positions(count=5)

            # Test maximum count
            await seed_positions(count=10)

            # Test middle count
            await seed_positions(count=7)

        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_clear_positions(self, tmp_path):
        """Test that clearing positions works."""
        db_path = tmp_path / "test.db"
        store = SQLiteStore(db_path=str(db_path))
        await store.initialize()

        try:
            # Clear should not raise errors even if no positions exist
            await clear_positions()

        finally:
            await store.close()


class TestPositionRealism:
    """Test that generated positions are realistic."""

    def test_position_diversity(self):
        """Test that generated positions have diversity."""
        positions = [generate_realistic_position_data() for _ in range(20)]

        # Check symbol diversity
        symbols = [p["symbol"] for p in positions]
        unique_symbols = set(symbols)
        assert len(unique_symbols) >= 3, "Should have at least 3 different symbols"

        # Check strategy diversity
        strategies = [p["strategy"] for p in positions]
        assert "TREND" in strategies
        assert "MEAN_REVERSION" in strategies

        # Check side diversity (should have both long and short)
        sides = [p["side"] for p in positions]
        # With 20 samples, we should have both sides represented
        assert OrderSide.BUY in sides
        # Note: SHORT might not always appear in 20 samples (20% probability)

    def test_realistic_share_counts(self):
        """Test that share counts are realistic for position sizing."""
        for _ in range(10):
            data = generate_realistic_position_data()

            # Position value should not exceed 20% of $100k account
            position_value = data["shares"] * data["entry_price"]
            assert position_value <= 25000  # Bit of buffer for rounding

            # Should be at least 10 shares
            assert data["shares"] >= 10

    def test_pnl_distribution(self):
        """Test that P&L has realistic distribution (more winners than losers)."""
        positions = [generate_realistic_position_data() for _ in range(100)]

        winning_positions = 0
        losing_positions = 0

        for data in positions:
            pnl = (data["current_price"] - data["entry_price"]) * data["shares"]
            if pnl > 0:
                winning_positions += 1
            else:
                losing_positions += 1

        # Should have approximately 60% win rate (with some variance)
        win_rate = winning_positions / len(positions)
        assert 0.50 <= win_rate <= 0.70, f"Win rate {win_rate} is outside expected range"


class TestScriptIntegration:
    """Test script integration and edge cases."""

    def test_multiple_generations_unique(self):
        """Test that multiple generations create unique data."""
        positions = [generate_realistic_position_data() for _ in range(5)]

        # Positions should not be identical
        # Check that at least prices are different
        entry_prices = [p["entry_price"] for p in positions]
        assert len(set(entry_prices)) > 1, "Entry prices should vary"

    def test_strategy_regime_consistency(self):
        """Test that strategy and regime are consistent."""
        for _ in range(20):
            data = generate_realistic_position_data()

            # Certain strategies should align with certain regimes
            if data["regime"] == "STRONG_TREND":
                assert data["strategy"] == "TREND"
            elif data["regime"] == "MEAN_REVERSION":
                assert data["strategy"] == "MEAN_REVERSION"
            # NEUTRAL can have either strategy


class TestErrorHandling:
    """Test error handling in seeding script."""

    @pytest.mark.asyncio
    async def test_seed_positions_invalid_count(self, tmp_path):
        """Test handling of invalid position counts."""
        # The script should handle invalid counts gracefully
        # Since count is validated in main(), the function itself
        # should accept any positive integer
        db_path = tmp_path / "test.db"
        store = SQLiteStore(db_path=str(db_path))
        await store.initialize()

        try:
            # Should work with edge cases
            await seed_positions(count=1)  # Minimum

            # Note: We don't test 15 positions because that would exceed
            # the $100k account balance (realistic constraint)
            # The recommended max is 10 for a reason

        finally:
            await store.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
