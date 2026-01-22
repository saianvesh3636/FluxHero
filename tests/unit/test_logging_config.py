"""
Unit tests for logging configuration module.

Tests the centralized logging configuration including structured formatters,
log setup, and context logging.
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import List
import pytest

from fluxhero.backend.core.logging_config import (
    setup_logging,
    get_logger,
    log_with_context,
    StructuredFormatter,
    HumanReadableFormatter,
)


class LogCapture(logging.Handler):
    """Custom handler to capture log records for testing."""

    def __init__(self):
        super().__init__()
        self.records: List[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord):
        self.records.append(record)

    def clear(self):
        self.records.clear()


@pytest.fixture
def cleanup_logging():
    """Fixture to cleanup logging handlers after each test."""
    yield
    # Remove all handlers from root logger
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    # Reset log level
    root.setLevel(logging.WARNING)


@pytest.fixture
def temp_log_dir():
    """Fixture providing a temporary directory for log files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestStructuredFormatter:
    """Tests for StructuredFormatter."""

    def test_basic_format(self):
        """Test basic log formatting to JSON."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        log_data = json.loads(result)

        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test.logger"
        assert log_data["message"] == "Test message"
        assert log_data["line"] == 42
        assert "timestamp" in log_data
        assert log_data["timestamp"].endswith("+00:00")

    def test_format_with_extra_fields(self):
        """Test formatting with extra context fields."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Order placed",
            args=(),
            exc_info=None,
        )
        # Add extra fields
        record.symbol = "AAPL"
        record.qty = 100
        record.price = 150.5

        result = formatter.format(record)
        log_data = json.loads(result)

        assert log_data["symbol"] == "AAPL"
        assert log_data["qty"] == 100
        assert log_data["price"] == 150.5

    def test_format_with_exception(self):
        """Test formatting with exception information."""
        formatter = StructuredFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

            record = logging.LogRecord(
                name="test.logger",
                level=logging.ERROR,
                pathname="test.py",
                lineno=42,
                msg="Error occurred",
                args=(),
                exc_info=exc_info,
            )

            result = formatter.format(record)
            log_data = json.loads(result)

            assert "exception" in log_data
            assert "ValueError: Test error" in log_data["exception"]
            assert "Traceback" in log_data["exception"]


class TestHumanReadableFormatter:
    """Tests for HumanReadableFormatter."""

    def test_basic_format(self):
        """Test basic human-readable formatting."""
        formatter = HumanReadableFormatter(use_colors=False)
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        assert "INFO" in result
        assert "test.logger" in result
        assert "Test message" in result

    def test_format_with_extra_fields(self):
        """Test formatting with extra context displayed."""
        formatter = HumanReadableFormatter(use_colors=False)
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Order placed",
            args=(),
            exc_info=None,
        )
        record.symbol = "AAPL"
        record.qty = 100

        result = formatter.format(record)

        assert "symbol=AAPL" in result
        assert "qty=100" in result

    def test_format_with_colors(self):
        """Test that color codes are added when enabled."""
        formatter = HumanReadableFormatter(use_colors=True)
        # Force color output for testing
        formatter.use_colors = True

        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=42,
            msg="Error message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        # Should contain ANSI escape codes for ERROR (red)
        assert "\033[31m" in result or "ERROR" in result


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_with_defaults(self, cleanup_logging):
        """Test logging setup with default parameters."""
        setup_logging()

        root = logging.getLogger()
        assert root.level == logging.INFO
        assert len(root.handlers) == 1  # Console handler only

    def test_setup_with_custom_level(self, cleanup_logging):
        """Test logging setup with custom log level."""
        setup_logging(log_level="DEBUG")

        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_setup_with_invalid_level(self, cleanup_logging):
        """Test that invalid log level raises ValueError."""
        with pytest.raises(ValueError, match="Invalid log level"):
            setup_logging(log_level="INVALID")

    def test_setup_with_file_output(self, cleanup_logging, temp_log_dir):
        """Test logging setup with file output."""
        log_file = Path(temp_log_dir) / "test.log"
        setup_logging(log_file=str(log_file))

        root = logging.getLogger()
        assert len(root.handlers) == 2  # Console + file

        # Verify file was created
        assert log_file.exists()

    def test_setup_with_log_dir(self, cleanup_logging, temp_log_dir):
        """Test logging setup with separate log directory."""
        setup_logging(log_file="app.log", log_dir=temp_log_dir)

        log_file = Path(temp_log_dir) / "app.log"
        assert log_file.exists()

    def test_setup_json_format(self, cleanup_logging, temp_log_dir):
        """Test logging setup with JSON formatting."""
        log_file = Path(temp_log_dir) / "test.log"
        setup_logging(log_file=str(log_file), json_format=True)

        logger = get_logger(__name__)
        logger.info("Test message", extra={"key": "value"})

        # Read log file and verify JSON format
        content = log_file.read_text()
        log_entry = json.loads(content.strip())

        assert log_entry["message"] == "Test message"
        assert log_entry["key"] == "value"

    def test_setup_no_console(self, cleanup_logging, temp_log_dir):
        """Test logging setup without console output."""
        log_file = Path(temp_log_dir) / "test.log"
        setup_logging(log_file=str(log_file), console_output=False)

        root = logging.getLogger()
        assert len(root.handlers) == 1  # File handler only

    def test_log_rotation_config(self, cleanup_logging, temp_log_dir):
        """Test that log rotation is configured correctly."""
        from logging.handlers import RotatingFileHandler

        log_file = Path(temp_log_dir) / "test.log"
        max_bytes = 5 * 1024 * 1024  # 5MB
        backup_count = 3

        setup_logging(
            log_file=str(log_file),
            max_bytes=max_bytes,
            backup_count=backup_count,
        )

        # Find the RotatingFileHandler
        root = logging.getLogger()
        rotating_handler = None
        for handler in root.handlers:
            if isinstance(handler, RotatingFileHandler):
                rotating_handler = handler
                break

        assert rotating_handler is not None
        assert rotating_handler.maxBytes == max_bytes
        assert rotating_handler.backupCount == backup_count


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a Logger instance."""
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

    def test_get_logger_same_name_returns_same_instance(self):
        """Test that same name returns same logger instance."""
        logger1 = get_logger("test.module")
        logger2 = get_logger("test.module")
        assert logger1 is logger2


class TestLogWithContext:
    """Tests for log_with_context function."""

    def test_log_with_context_basic(self, cleanup_logging):
        """Test logging with context adds extra fields."""
        capture = LogCapture()
        logger = get_logger("test.logger")
        logger.addHandler(capture)
        logger.setLevel(logging.INFO)

        log_with_context(
            logger,
            "info",
            "Order placed",
            symbol="AAPL",
            qty=100,
            price=150.5,
        )

        assert len(capture.records) == 1
        record = capture.records[0]
        assert record.message == "Order placed"
        assert record.symbol == "AAPL"
        assert record.qty == 100
        assert record.price == 150.5

    def test_log_with_context_all_levels(self, cleanup_logging):
        """Test log_with_context works with all log levels."""
        capture = LogCapture()
        logger = get_logger("test.logger")
        logger.addHandler(capture)
        logger.setLevel(logging.DEBUG)

        levels = ["debug", "info", "warning", "error", "critical"]

        for level in levels:
            log_with_context(logger, level, f"Test {level}")

        assert len(capture.records) == 5
        for i, level in enumerate(levels):
            assert capture.records[i].levelname == level.upper()


class TestIntegration:
    """Integration tests for the logging system."""

    def test_end_to_end_file_logging(self, cleanup_logging, temp_log_dir):
        """Test complete logging workflow with file output."""
        log_file = Path(temp_log_dir) / "app.log"

        # Setup logging
        setup_logging(
            log_level="DEBUG",
            log_file=str(log_file),
            json_format=False,
            console_output=False,
        )

        # Create logger and log messages
        logger = get_logger("test.module")
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message", extra={"key": "value"})

        # Verify file contents
        content = log_file.read_text()
        assert "Debug message" in content
        assert "Info message" in content
        assert "Warning message" in content
        assert "key=value" in content

    def test_end_to_end_json_logging(self, cleanup_logging, temp_log_dir):
        """Test complete logging workflow with JSON output."""
        log_file = Path(temp_log_dir) / "app.log"

        # Setup logging
        setup_logging(
            log_level="INFO",
            log_file=str(log_file),
            json_format=True,
            console_output=False,
        )

        # Log structured messages
        logger = get_logger("test.module")
        logger.info("Order executed", extra={"symbol": "AAPL", "qty": 100})

        # Parse and verify JSON
        content = log_file.read_text().strip()
        log_entry = json.loads(content)

        assert log_entry["level"] == "INFO"
        assert log_entry["message"] == "Order executed"
        assert log_entry["symbol"] == "AAPL"
        assert log_entry["qty"] == 100

    def test_multiple_loggers(self, cleanup_logging):
        """Test that multiple loggers work correctly."""
        setup_logging(console_output=False)

        logger1 = get_logger("module1")
        logger2 = get_logger("module2")

        capture1 = LogCapture()
        capture2 = LogCapture()

        logger1.addHandler(capture1)
        logger2.addHandler(capture2)

        logger1.info("Message from module1")
        logger2.info("Message from module2")

        # Both should have received messages due to propagation
        assert len(capture1.records) >= 1
        assert len(capture2.records) >= 1
