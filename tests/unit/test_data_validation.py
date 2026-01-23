"""
Unit tests for OHLCV data validation.

Tests the validate_ohlcv_data function in yahoo_provider.py that checks for:
- NaN values in OHLCV columns
- Negative prices (Open, High, Low, Close)
- Zero volume
- Invalid OHLC relationships (High < Low)
- Large gaps in data (> max_gap_days)

Reference: enhancement_tasks.md Phase 24 - Quality Control & Validation Framework
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from backend.data.provider import DataValidationError
from backend.data.yahoo_provider import validate_ohlcv_data


def create_test_df(
    n_rows: int = 10,
    start_date: str = "2024-01-01",
    include_weekend_gaps: bool = False,
) -> pd.DataFrame:
    """Create a valid test DataFrame with OHLCV data."""
    dates = pd.date_range(start=start_date, periods=n_rows, freq="B")  # Business days

    # Generate realistic OHLCV data
    np.random.seed(42)
    base_price = 100.0

    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []

    for i in range(n_rows):
        daily_change = np.random.uniform(-0.02, 0.02)
        open_price = base_price * (1 + daily_change)
        close_price = open_price * (1 + np.random.uniform(-0.01, 0.01))
        high_price = max(open_price, close_price) * (1 + np.random.uniform(0, 0.005))
        low_price = min(open_price, close_price) * (1 - np.random.uniform(0, 0.005))
        volume = np.random.randint(1000000, 5000000)

        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
        closes.append(close_price)
        volumes.append(volume)

        base_price = close_price

    return pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes,
    }, index=dates)


class TestValidateOhlcvDataCleanData:
    """Test validation with clean, valid data."""

    def test_valid_data_returns_empty_list(self):
        """Valid OHLCV data should return no issues."""
        df = create_test_df()
        issues = validate_ohlcv_data(df, "TEST")
        assert issues == []

    def test_valid_data_with_various_lengths(self):
        """Valid data of various lengths should return no issues."""
        for n_rows in [1, 10, 100, 252]:  # 1 day, 2 weeks, ~5 months, 1 year
            df = create_test_df(n_rows=n_rows)
            issues = validate_ohlcv_data(df, "TEST")
            assert issues == [], f"Failed for n_rows={n_rows}"


class TestNaNValidation:
    """Test NaN detection in OHLCV columns."""

    def test_nan_in_open_detected(self):
        """NaN values in Open column should be detected."""
        df = create_test_df()
        df.loc[df.index[2], "Open"] = np.nan

        issues = validate_ohlcv_data(df, "TEST")

        assert len(issues) == 1
        assert "NaN values" in issues[0]
        assert "Open" in issues[0]

    def test_nan_in_close_detected(self):
        """NaN values in Close column should be detected."""
        df = create_test_df()
        df.loc[df.index[5], "Close"] = np.nan

        issues = validate_ohlcv_data(df, "TEST")

        assert len(issues) == 1
        assert "NaN values" in issues[0]
        assert "Close" in issues[0]

    def test_nan_in_multiple_columns_detected(self):
        """NaN values in multiple columns should all be reported."""
        df = create_test_df()
        df.loc[df.index[2], "Open"] = np.nan
        df.loc[df.index[3], "High"] = np.nan
        df.loc[df.index[4], "Low"] = np.nan

        issues = validate_ohlcv_data(df, "TEST")

        assert len(issues) == 1  # All NaN issues in one message
        assert "Open" in issues[0]
        assert "High" in issues[0]
        assert "Low" in issues[0]

    def test_multiple_nan_same_column_counted(self):
        """Multiple NaN values in same column should show count."""
        df = create_test_df()
        df.loc[df.index[2], "Volume"] = np.nan
        df.loc[df.index[5], "Volume"] = np.nan
        df.loc[df.index[7], "Volume"] = np.nan

        issues = validate_ohlcv_data(df, "TEST")

        assert len(issues) == 1
        assert "Volume(3)" in issues[0]


class TestNegativePriceValidation:
    """Test negative price detection."""

    def test_negative_open_detected(self):
        """Negative Open price should be detected."""
        df = create_test_df()
        df.loc[df.index[3], "Open"] = -50.0

        issues = validate_ohlcv_data(df, "TEST")

        # Will also trigger High < Low if not adjusted
        neg_issue = [i for i in issues if "Negative prices" in i]
        assert len(neg_issue) == 1
        assert "Open" in neg_issue[0]

    def test_negative_close_detected(self):
        """Negative Close price should be detected."""
        df = create_test_df()
        df.loc[df.index[3], "Close"] = -25.0

        issues = validate_ohlcv_data(df, "TEST")

        neg_issue = [i for i in issues if "Negative prices" in i]
        assert len(neg_issue) == 1
        assert "Close" in neg_issue[0]

    def test_negative_high_low_detected(self):
        """Negative High and Low prices should be detected."""
        df = create_test_df()
        df.loc[df.index[3], "High"] = -10.0
        df.loc[df.index[3], "Low"] = -20.0

        issues = validate_ohlcv_data(df, "TEST")

        neg_issue = [i for i in issues if "Negative prices" in i]
        assert len(neg_issue) == 1
        assert "High" in neg_issue[0]
        assert "Low" in neg_issue[0]

    def test_zero_price_not_flagged_as_negative(self):
        """Zero price should not trigger negative price warning."""
        df = create_test_df()
        df.loc[df.index[3], "Open"] = 0.0  # Zero is suspicious but not negative

        issues = validate_ohlcv_data(df, "TEST")

        neg_issue = [i for i in issues if "Negative prices" in i]
        assert len(neg_issue) == 0


class TestZeroVolumeValidation:
    """Test zero volume detection."""

    def test_zero_volume_detected(self):
        """Zero volume should be detected and reported."""
        df = create_test_df()
        df.loc[df.index[3], "Volume"] = 0

        issues = validate_ohlcv_data(df, "TEST")

        vol_issue = [i for i in issues if "Zero volume" in i]
        assert len(vol_issue) == 1
        assert "1 bars" in vol_issue[0]

    def test_multiple_zero_volumes_counted(self):
        """Multiple zero volume bars should be counted."""
        df = create_test_df()
        df.loc[df.index[2], "Volume"] = 0
        df.loc[df.index[5], "Volume"] = 0
        df.loc[df.index[8], "Volume"] = 0

        issues = validate_ohlcv_data(df, "TEST")

        vol_issue = [i for i in issues if "Zero volume" in i]
        assert len(vol_issue) == 1
        assert "3 bars" in vol_issue[0]
        assert "30.0%" in vol_issue[0]  # 3/10 = 30%


class TestHighLowValidation:
    """Test High < Low (invalid OHLC relationship) detection."""

    def test_high_less_than_low_detected(self):
        """High < Low should be detected as invalid."""
        df = create_test_df()
        # Swap high and low to create invalid bar
        df.loc[df.index[4], "High"] = 95.0
        df.loc[df.index[4], "Low"] = 105.0  # Low > High is invalid

        issues = validate_ohlcv_data(df, "TEST")

        hl_issue = [i for i in issues if "High < Low" in i]
        assert len(hl_issue) == 1
        assert "1 bars" in hl_issue[0]

    def test_multiple_invalid_high_low_counted(self):
        """Multiple invalid High/Low bars should be counted."""
        df = create_test_df()
        df.loc[df.index[2], "High"] = 90.0
        df.loc[df.index[2], "Low"] = 100.0
        df.loc[df.index[6], "High"] = 85.0
        df.loc[df.index[6], "Low"] = 95.0

        issues = validate_ohlcv_data(df, "TEST")

        hl_issue = [i for i in issues if "High < Low" in i]
        assert len(hl_issue) == 1
        assert "2 bars" in hl_issue[0]

    def test_equal_high_low_not_flagged(self):
        """High == Low (doji) should not be flagged as invalid."""
        df = create_test_df()
        df.loc[df.index[4], "High"] = 100.0
        df.loc[df.index[4], "Low"] = 100.0

        issues = validate_ohlcv_data(df, "TEST")

        hl_issue = [i for i in issues if "High < Low" in i]
        assert len(hl_issue) == 0


class TestDataGapValidation:
    """Test large data gap detection."""

    def test_large_gap_detected(self):
        """Gap > max_gap_days should be detected."""
        # Create data with a 10-day gap
        dates = [
            datetime(2024, 1, 2),
            datetime(2024, 1, 3),
            datetime(2024, 1, 4),
            datetime(2024, 1, 18),  # 14-day gap (> default 5 days)
            datetime(2024, 1, 19),
        ]

        df = pd.DataFrame({
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "High": [101.0, 102.0, 103.0, 104.0, 105.0],
            "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "Volume": [1000000, 1000000, 1000000, 1000000, 1000000],
        }, index=pd.DatetimeIndex(dates))

        issues = validate_ohlcv_data(df, "TEST", max_gap_days=5)

        gap_issue = [i for i in issues if "Data gap" in i]
        assert len(gap_issue) == 1
        assert "14 days" in gap_issue[0]
        assert "2024-01-04" in gap_issue[0]
        assert "2024-01-18" in gap_issue[0]

    def test_weekend_gap_not_flagged(self):
        """Normal weekend gap (3 days Fri-Mon) should not be flagged."""
        dates = [
            datetime(2024, 1, 5),   # Friday
            datetime(2024, 1, 8),   # Monday (3-day gap is normal)
            datetime(2024, 1, 9),
        ]

        df = pd.DataFrame({
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.5, 101.5, 102.5],
            "Volume": [1000000, 1000000, 1000000],
        }, index=pd.DatetimeIndex(dates))

        issues = validate_ohlcv_data(df, "TEST", max_gap_days=5)

        gap_issue = [i for i in issues if "Data gap" in i]
        assert len(gap_issue) == 0

    def test_holiday_gap_configurable(self):
        """Gap threshold should be configurable."""
        dates = [
            datetime(2024, 1, 2),
            datetime(2024, 1, 10),  # 8-day gap
            datetime(2024, 1, 11),
        ]

        df = pd.DataFrame({
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.5, 101.5, 102.5],
            "Volume": [1000000, 1000000, 1000000],
        }, index=pd.DatetimeIndex(dates))

        # With default 5 days, should flag
        issues = validate_ohlcv_data(df, "TEST", max_gap_days=5)
        assert any("Data gap" in i for i in issues)

        # With 10 days, should not flag
        issues = validate_ohlcv_data(df, "TEST", max_gap_days=10)
        assert not any("Data gap" in i for i in issues)

    def test_multiple_gaps_all_reported(self):
        """Multiple large gaps should all be reported."""
        dates = [
            datetime(2024, 1, 2),
            datetime(2024, 1, 15),  # Gap 1: 13 days
            datetime(2024, 1, 16),
            datetime(2024, 2, 1),   # Gap 2: 16 days
            datetime(2024, 2, 2),
        ]

        df = pd.DataFrame({
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "High": [101.0, 102.0, 103.0, 104.0, 105.0],
            "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "Volume": [1000000, 1000000, 1000000, 1000000, 1000000],
        }, index=pd.DatetimeIndex(dates))

        issues = validate_ohlcv_data(df, "TEST", max_gap_days=5)

        gap_issues = [i for i in issues if "Data gap" in i]
        assert len(gap_issues) == 2


class TestMultipleIssuesCombined:
    """Test that multiple different issues are all detected."""

    def test_multiple_issue_types_detected(self):
        """Multiple different issues should all be reported."""
        df = create_test_df()

        # Add NaN
        df.loc[df.index[1], "Open"] = np.nan

        # Add negative price
        df.loc[df.index[3], "Close"] = -10.0

        # Add zero volume
        df.loc[df.index[5], "Volume"] = 0

        # Add High < Low
        df.loc[df.index[7], "High"] = 90.0
        df.loc[df.index[7], "Low"] = 100.0

        issues = validate_ohlcv_data(df, "TEST")

        # Should have 4 different types of issues
        assert any("NaN values" in i for i in issues)
        assert any("Negative prices" in i for i in issues)
        assert any("Zero volume" in i for i in issues)
        assert any("High < Low" in i for i in issues)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_dataframe(self):
        """Empty DataFrame should return no issues."""
        df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        df.index = pd.DatetimeIndex([])

        issues = validate_ohlcv_data(df, "TEST")

        assert issues == []

    def test_single_row_dataframe(self):
        """Single row DataFrame should work correctly."""
        dates = [datetime(2024, 1, 2)]
        df = pd.DataFrame({
            "Open": [100.0],
            "High": [101.0],
            "Low": [99.0],
            "Close": [100.5],
            "Volume": [1000000],
        }, index=pd.DatetimeIndex(dates))

        issues = validate_ohlcv_data(df, "TEST")

        assert issues == []

    def test_missing_columns_handled(self):
        """Missing columns should be handled gracefully."""
        dates = pd.date_range(start="2024-01-01", periods=5, freq="B")
        df = pd.DataFrame({
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
            # Missing High, Low, Volume
        }, index=dates)

        # Should not raise, just check available columns
        issues = validate_ohlcv_data(df, "TEST")
        # No issues if columns are missing (they just aren't checked)
        assert isinstance(issues, list)


class TestDataValidationErrorIntegration:
    """Test DataValidationError exception class."""

    def test_error_creation(self):
        """DataValidationError should be created correctly."""
        issues = ["NaN values in: Open(2)", "Negative prices in: Close(1)"]
        error = DataValidationError("TEST", issues)

        assert error.symbol == "TEST"
        assert error.issues == issues
        assert "TEST" in str(error)
        assert "NaN values" in str(error)
        assert "Negative prices" in str(error)

    def test_error_inheritance(self):
        """DataValidationError should inherit from DataProviderError."""
        from backend.data.provider import DataProviderError

        error = DataValidationError("TEST", ["Some issue"])
        assert isinstance(error, DataProviderError)
        assert isinstance(error, Exception)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
