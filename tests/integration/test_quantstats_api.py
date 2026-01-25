"""
Integration tests for QuantStats API endpoints.

Tests the /api/reports endpoints including:
- Report generation
- Report download
- Enhanced metrics
- Report listing and deletion

Reference: /Users/anvesh/.claude/plans/swirling-tumbling-cloud.md
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import shutil


@pytest.fixture(scope="module")
def client():
    """Create test client for the API."""
    from backend.api.server import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def temp_report_dir():
    """Create temporary directory for reports."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestEnhancedMetricsEndpoint:
    """Tests for GET /api/reports/metrics endpoint."""

    def test_get_metrics_default(self, client):
        """Should return enhanced metrics with default parameters."""
        response = client.get("/api/reports/metrics")

        assert response.status_code == 200
        data = response.json()

        # Check Tier 1 metrics exist
        assert "sortino_ratio" in data
        assert "calmar_ratio" in data
        assert "profit_factor" in data
        assert "value_at_risk_95" in data
        assert "cvar_95" in data
        assert "alpha" in data
        assert "beta" in data
        assert "kelly_criterion" in data
        assert "recovery_factor" in data
        assert "ulcer_index" in data

        # Check Tier 2 metrics exist
        assert "max_consecutive_wins" in data
        assert "max_consecutive_losses" in data
        assert "skewness" in data
        assert "kurtosis" in data
        assert "tail_ratio" in data

        # Check standard metrics exist
        assert "sharpe_ratio" in data
        assert "max_drawdown_pct" in data
        assert "win_rate" in data
        assert "total_return_pct" in data

        # Check metadata
        assert "periods_analyzed" in data
        assert "benchmark_symbol" in data

    def test_get_metrics_with_benchmark(self, client):
        """Should accept benchmark parameter."""
        response = client.get("/api/reports/metrics?benchmark=QQQ")

        assert response.status_code == 200
        data = response.json()
        assert data["benchmark_symbol"] == "QQQ"

    def test_get_metrics_with_mode(self, client):
        """Should accept mode parameter."""
        response = client.get("/api/reports/metrics?mode=live")

        assert response.status_code == 200


class TestReportGenerationEndpoint:
    """Tests for POST /api/reports/generate endpoint."""

    def test_generate_report_default(self, client):
        """Should generate report with default parameters."""
        response = client.post(
            "/api/reports/generate",
            json={"mode": "paper", "benchmark": "SPY"},
        )

        assert response.status_code == 200
        data = response.json()

        assert "report_id" in data
        assert "download_url" in data
        assert "generated_at" in data
        assert "expires_at" in data
        assert "title" in data

        # Download URL should be properly formatted
        assert data["download_url"].startswith("/api/reports/download/")

    def test_generate_report_with_title(self, client):
        """Should accept custom title."""
        response = client.post(
            "/api/reports/generate",
            json={"title": "Custom Report Title", "benchmark": "SPY"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "Custom Report Title" in data["title"]


class TestReportDownloadEndpoint:
    """Tests for GET /api/reports/download/{report_id} endpoint."""

    def test_download_nonexistent_report(self, client):
        """Should return 404 for nonexistent report."""
        response = client.get("/api/reports/download/nonexistent-report-id")

        assert response.status_code == 404

    def test_download_generated_report(self, client):
        """Should be able to download a generated report."""
        # First generate a report
        gen_response = client.post(
            "/api/reports/generate",
            json={"mode": "paper", "benchmark": "SPY"},
        )

        assert gen_response.status_code == 200
        report_id = gen_response.json()["report_id"]

        # Then download it
        download_response = client.get(f"/api/reports/download/{report_id}")

        assert download_response.status_code == 200
        assert "text/html" in download_response.headers.get("content-type", "")


class TestReportListEndpoint:
    """Tests for GET /api/reports/list endpoint."""

    def test_list_reports(self, client):
        """Should return list of reports."""
        response = client.get("/api/reports/list")

        assert response.status_code == 200
        data = response.json()

        assert "reports" in data
        assert "total_count" in data
        assert isinstance(data["reports"], list)


class TestReportDeleteEndpoint:
    """Tests for DELETE /api/reports/{report_id} endpoint."""

    def test_delete_nonexistent_report(self, client):
        """Should return 404 for nonexistent report."""
        response = client.delete("/api/reports/nonexistent-report-id")

        assert response.status_code == 404

    def test_delete_generated_report(self, client):
        """Should be able to delete a generated report."""
        # First generate a report
        gen_response = client.post(
            "/api/reports/generate",
            json={"mode": "paper", "benchmark": "SPY"},
        )

        assert gen_response.status_code == 200
        report_id = gen_response.json()["report_id"]

        # Then delete it
        delete_response = client.delete(f"/api/reports/{report_id}")

        assert delete_response.status_code == 200
        data = delete_response.json()
        assert data["status"] == "deleted"

        # Verify it's gone
        download_response = client.get(f"/api/reports/download/{report_id}")
        assert download_response.status_code == 404


class TestReportCleanupEndpoint:
    """Tests for POST /api/reports/cleanup endpoint."""

    def test_cleanup_reports(self, client):
        """Should cleanup old reports."""
        response = client.post("/api/reports/cleanup?max_age_hours=0")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert data["status"] == "completed"
        assert "deleted_count" in data


class TestMetricsValues:
    """Tests to verify metrics values are reasonable."""

    def test_metrics_are_finite(self, client):
        """All metrics should be finite numbers."""
        response = client.get("/api/reports/metrics")

        assert response.status_code == 200
        data = response.json()

        numeric_fields = [
            "sortino_ratio",
            "calmar_ratio",
            "profit_factor",
            "value_at_risk_95",
            "cvar_95",
            "alpha",
            "beta",
            "kelly_criterion",
            "recovery_factor",
            "ulcer_index",
            "skewness",
            "kurtosis",
            "tail_ratio",
            "sharpe_ratio",
            "max_drawdown_pct",
            "win_rate",
            "avg_win_loss_ratio",
            "total_return_pct",
            "annualized_return_pct",
        ]

        import math

        for field in numeric_fields:
            value = data.get(field)
            assert value is not None, f"{field} should not be None"
            assert math.isfinite(value), f"{field} should be finite, got {value}"

    def test_win_rate_in_valid_range(self, client):
        """Win rate should be between 0 and 1."""
        response = client.get("/api/reports/metrics")

        assert response.status_code == 200
        data = response.json()

        win_rate = data["win_rate"]
        assert 0 <= win_rate <= 1, f"Win rate should be 0-1, got {win_rate}"

    def test_beta_reasonable(self, client):
        """Beta should be in a reasonable range."""
        response = client.get("/api/reports/metrics")

        assert response.status_code == 200
        data = response.json()

        beta = data["beta"]
        # Beta typically ranges from -2 to 3 for most strategies
        assert -3 <= beta <= 5, f"Beta seems unreasonable: {beta}"
