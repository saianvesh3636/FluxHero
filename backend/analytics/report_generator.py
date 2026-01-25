"""
HTML Tearsheet Report Generator using QuantStats

Generates comprehensive visual reports for backtest and live trading results.

Features:
- Full QuantStats HTML tearsheets with charts
- Benchmark comparison (SPY, QQQ, etc.)
- Temporary file storage with auto-cleanup
- Download URL generation for API access

Usage:
    generator = TearsheetGenerator()

    # Generate report from returns
    report_path = generator.generate_tearsheet(
        returns=returns_array,
        timestamps=timestamps_array,
        benchmark_symbol="SPY",
        title="My Strategy Report"
    )

    # Get download URL
    url = generator.get_report_url(report_path.stem)

    # Cleanup old reports
    generator.cleanup_old_reports(max_age_hours=24)

Reference: /Users/anvesh/.claude/plans/swirling-tumbling-cloud.md
"""

import logging
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Set matplotlib backend before importing quantstats to avoid GUI issues in threads
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import quantstats as qs

from backend.analytics.quantstats_wrapper import QuantStatsAdapter

logger = logging.getLogger(__name__)


class TearsheetGenerator:
    """
    Generates QuantStats HTML tearsheet reports.

    Reports are stored in a configurable directory with automatic
    cleanup of old files.
    """

    # Default report storage directory
    DEFAULT_REPORT_DIR = Path(__file__).parent.parent.parent / "data" / "reports"

    def __init__(self, report_dir: Path | str | None = None):
        """
        Initialize generator.

        Args:
            report_dir: Directory to store reports (default: data/reports)
        """
        self.report_dir = Path(report_dir) if report_dir else self.DEFAULT_REPORT_DIR
        self.report_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"TearsheetGenerator initialized with report_dir: {self.report_dir}")

    def generate_tearsheet(
        self,
        returns: np.ndarray,
        timestamps: np.ndarray | list[str] | None = None,
        benchmark_symbol: str = "SPY",
        title: str = "FluxHero Strategy Report",
        output_filename: str | None = None,
    ) -> Path:
        """
        Generate HTML tearsheet report.

        Args:
            returns: Array of period returns (decimals)
            timestamps: Array of timestamps (datetime or string)
            benchmark_symbol: Benchmark symbol for comparison (e.g., "SPY")
            title: Report title
            output_filename: Custom filename (auto-generated if not provided)

        Returns:
            Path to generated HTML file

        Example:
            >>> returns = np.array([0.01, -0.005, 0.02, 0.015, -0.01])
            >>> path = generator.generate_tearsheet(returns, benchmark_symbol="SPY")
        """
        # Generate unique filename if not provided
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_id = str(uuid.uuid4())[:8]
            output_filename = f"report_{timestamp}_{report_id}"

        # Ensure .html extension
        if not output_filename.endswith(".html"):
            output_filename = f"{output_filename}.html"

        output_path = self.report_dir / output_filename

        # Create pandas Series with datetime index
        if timestamps is not None:
            try:
                if isinstance(timestamps, np.ndarray):
                    index = pd.to_datetime(timestamps)
                else:
                    index = pd.to_datetime(timestamps)
            except Exception as e:
                logger.warning(f"Could not parse timestamps: {e}, using default index")
                index = pd.date_range(end=pd.Timestamp.now(), periods=len(returns), freq="D")
        else:
            index = pd.date_range(end=pd.Timestamp.now(), periods=len(returns), freq="D")

        returns_series = pd.Series(returns, index=index, name="Strategy")

        # Handle duplicate timestamps by grouping by date and compounding returns
        # Multiple trades on same day: (1+r1)*(1+r2) - 1
        if returns_series.index.duplicated().any():
            logger.info("Aggregating duplicate timestamps...")
            # Group by date and compound returns
            returns_series = returns_series.groupby(returns_series.index.date).apply(
                lambda x: (1 + x).prod() - 1
            )
            returns_series.index = pd.to_datetime(returns_series.index)
            returns_series.name = "Strategy"

        # Clean data - remove NaN/inf
        returns_series = returns_series.replace([np.inf, -np.inf], np.nan).dropna()

        if len(returns_series) < 5:
            raise ValueError("Insufficient data for report generation (need at least 5 periods)")

        try:
            # Generate HTML report using QuantStats
            # Note: qs.reports.html() saves directly to file
            logger.info(f"Generating tearsheet for {len(returns_series)} periods vs {benchmark_symbol}")

            qs.reports.html(
                returns_series,
                benchmark=benchmark_symbol,
                output=str(output_path),
                title=title,
                download_filename=output_filename,
            )

            logger.info(f"Tearsheet generated: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error generating tearsheet: {e}")
            # Try without benchmark
            try:
                logger.info("Retrying without benchmark...")
                qs.reports.html(
                    returns_series,
                    output=str(output_path),
                    title=title,
                    download_filename=output_filename,
                )
                return output_path
            except Exception as e2:
                logger.error(f"Error generating tearsheet without benchmark: {e2}")
                raise

    def generate_from_backtest(
        self,
        equity_curve: list[float] | np.ndarray,
        timestamps: list[str] | np.ndarray,
        trades: list[dict] | None = None,
        config: dict[str, Any] | None = None,
        benchmark_symbol: str = "SPY",
    ) -> Path:
        """
        Generate report from backtest results.

        Args:
            equity_curve: Array of equity values
            timestamps: Array of timestamps (ISO format strings)
            trades: Optional list of trade dictionaries
            config: Optional backtest configuration
            benchmark_symbol: Benchmark for comparison

        Returns:
            Path to generated HTML file
        """
        equity = np.array(equity_curve, dtype=np.float64)

        # Calculate returns from equity curve
        returns = np.diff(equity) / equity[:-1]
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

        # Parse timestamps (skip first since returns are n-1)
        ts = timestamps[1:] if len(timestamps) > len(returns) else timestamps

        # Build title
        title = "FluxHero Backtest Report"
        if config:
            symbol = config.get("symbol", "")
            strategy = config.get("strategy_mode", "")
            if symbol:
                title = f"{title} - {symbol}"
            if strategy:
                title = f"{title} ({strategy})"

        return self.generate_tearsheet(
            returns=returns,
            timestamps=ts,
            benchmark_symbol=benchmark_symbol,
            title=title,
        )

    def generate_from_adapter(
        self,
        adapter: QuantStatsAdapter,
        title: str = "FluxHero Strategy Report",
        benchmark_symbol: str = "SPY",
    ) -> Path:
        """
        Generate report from QuantStatsAdapter.

        Args:
            adapter: Configured QuantStatsAdapter instance
            title: Report title
            benchmark_symbol: Benchmark symbol

        Returns:
            Path to generated HTML file
        """
        return self.generate_tearsheet(
            returns=adapter._returns,
            timestamps=None,  # Adapter already has timestamps
            benchmark_symbol=benchmark_symbol,
            title=title,
        )

    def generate_metrics_report(
        self,
        adapter: QuantStatsAdapter,
        output_filename: str | None = None,
    ) -> Path:
        """
        Generate a simple metrics-only HTML report.

        This is faster than full tearsheet as it doesn't include plots.

        Args:
            adapter: Configured QuantStatsAdapter instance
            output_filename: Custom filename

        Returns:
            Path to generated HTML file
        """
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_id = str(uuid.uuid4())[:8]
            output_filename = f"metrics_{timestamp}_{report_id}.html"

        if not output_filename.endswith(".html"):
            output_filename = f"{output_filename}.html"

        output_path = self.report_dir / output_filename

        # Get all metrics
        metrics = adapter.get_all_metrics()
        tier1 = adapter.get_tier1_metrics()

        # Generate simple HTML
        html = self._generate_metrics_html(metrics, tier1)

        with open(output_path, "w") as f:
            f.write(html)

        return output_path

    def _generate_metrics_html(
        self,
        metrics: dict[str, float],
        tier1: dict[str, float],
    ) -> str:
        """Generate simple HTML for metrics display."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>FluxHero Metrics Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #666; margin-top: 30px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 15px; }}
        .metric {{ background: #f9f9f9; padding: 15px; border-radius: 5px; border-left: 4px solid #4CAF50; }}
        .metric-name {{ font-size: 12px; color: #666; text-transform: uppercase; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
        .positive {{ color: #4CAF50; }}
        .negative {{ color: #f44336; }}
        .timestamp {{ color: #999; font-size: 12px; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>FluxHero Performance Metrics</h1>
        <p class="timestamp">Generated: {timestamp}</p>

        <h2>Tier 1 Metrics (Numba-Optimized)</h2>
        <div class="metrics-grid">
"""

        # Add Tier 1 metrics
        for name, value in tier1.items():
            value_class = "positive" if isinstance(value, (int, float)) and value > 0 else ""
            if isinstance(value, float):
                formatted = f"{value:.4f}"
            else:
                formatted = str(value)

            html += f"""
            <div class="metric">
                <div class="metric-name">{name.replace('_', ' ').title()}</div>
                <div class="metric-value {value_class}">{formatted}</div>
            </div>
"""

        html += """
        </div>

        <h2>QuantStats Metrics</h2>
        <div class="metrics-grid">
"""

        # Add QuantStats metrics (excluding those in tier1)
        tier1_keys = set(tier1.keys())
        for name, value in metrics.items():
            if name in tier1_keys:
                continue
            if value is None:
                continue

            value_class = ""
            if isinstance(value, (int, float)):
                if "ratio" in name.lower() or "return" in name.lower():
                    value_class = "positive" if value > 0 else "negative"
                formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
            else:
                formatted = str(value)

            html += f"""
            <div class="metric">
                <div class="metric-name">{name.replace('_', ' ').title()}</div>
                <div class="metric-value {value_class}">{formatted}</div>
            </div>
"""

        html += """
        </div>
    </div>
</body>
</html>
"""
        return html

    def get_report_path(self, report_id: str) -> Path | None:
        """
        Get full path for a report by ID.

        Args:
            report_id: Report identifier (filename without extension)

        Returns:
            Path to report file, or None if not found
        """
        # Try with .html extension
        path = self.report_dir / f"{report_id}.html"
        if path.exists():
            return path

        # Try exact match
        path = self.report_dir / report_id
        if path.exists():
            return path

        # Search for partial match
        for file in self.report_dir.glob(f"*{report_id}*.html"):
            return file

        return None

    def get_report_url(self, report_id: str) -> str:
        """
        Get download URL for a report.

        Args:
            report_id: Report identifier

        Returns:
            API URL for downloading the report
        """
        return f"/api/reports/download/{report_id}"

    def list_reports(self) -> list[dict[str, Any]]:
        """
        List all available reports.

        Returns:
            List of report metadata dictionaries
        """
        reports = []

        for file in self.report_dir.glob("*.html"):
            stat = file.stat()
            reports.append({
                "report_id": file.stem,
                "filename": file.name,
                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "size_bytes": stat.st_size,
                "download_url": self.get_report_url(file.stem),
            })

        # Sort by creation time (newest first)
        reports.sort(key=lambda x: x["created_at"], reverse=True)
        return reports

    def cleanup_old_reports(self, max_age_hours: int = 24) -> int:
        """
        Remove reports older than specified hours.

        Args:
            max_age_hours: Maximum age in hours (default: 24)

        Returns:
            Number of reports deleted
        """
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        deleted = 0

        for file in self.report_dir.glob("*.html"):
            stat = file.stat()
            created = datetime.fromtimestamp(stat.st_ctime)

            if created < cutoff:
                try:
                    file.unlink()
                    deleted += 1
                    logger.info(f"Deleted old report: {file.name}")
                except Exception as e:
                    logger.error(f"Error deleting report {file.name}: {e}")

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old reports")

        return deleted

    def delete_report(self, report_id: str) -> bool:
        """
        Delete a specific report.

        Args:
            report_id: Report identifier

        Returns:
            True if deleted, False if not found
        """
        path = self.get_report_path(report_id)
        if path and path.exists():
            try:
                path.unlink()
                logger.info(f"Deleted report: {report_id}")
                return True
            except Exception as e:
                logger.error(f"Error deleting report {report_id}: {e}")
                return False
        return False


# Module-level instance for convenience
_default_generator: TearsheetGenerator | None = None


def get_generator() -> TearsheetGenerator:
    """Get or create the default TearsheetGenerator instance."""
    global _default_generator
    if _default_generator is None:
        _default_generator = TearsheetGenerator()
    return _default_generator
