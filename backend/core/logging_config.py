"""
Centralized Logging Configuration for FluxHero

This module provides structured logging configuration using Python's standard
library logging. It supports both console and file output with structured
formatting for better observability.

Usage:
    from backend.core.logging_config import setup_logging, get_logger

    # Setup logging once at application startup
    setup_logging(log_level="INFO", log_file="app.log")

    # Get logger in each module
    logger = get_logger(__name__)
    logger.info("Message", extra={"symbol": "AAPL", "price": 150.0})
"""

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that outputs structured JSON logs.

    This formatter converts log records into JSON format with consistent
    structure for easier parsing by log aggregation tools.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON string.

        Args:
            record: Log record to format

        Returns:
            JSON-formatted log string
        """
        log_data: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add custom fields from extra parameter
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        # Check for any extra attributes added via extra={}
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
                "extra_fields",
                "getMessage",
                "asctime",
            ]:
                log_data[key] = value

        return json.dumps(log_data)


class HumanReadableFormatter(logging.Formatter):
    """
    Custom formatter for human-readable console output.

    Provides colored output (when supported) and structured but readable format.
    """

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",
    }

    def __init__(self, use_colors: bool = True):
        """
        Initialize formatter.

        Args:
            use_colors: Whether to use ANSI colors in output
        """
        super().__init__()
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as human-readable string.

        Args:
            record: Log record to format

        Returns:
            Formatted log string
        """
        # Base format: timestamp [LEVEL] logger - message
        timestamp = datetime.fromtimestamp(record.created, UTC).strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname

        if self.use_colors:
            color = self.COLORS.get(level, "")
            reset = self.COLORS["RESET"]
            level_str = f"{color}{level:8s}{reset}"
        else:
            level_str = f"{level:8s}"

        base_msg = f"{timestamp} [{level_str}] {record.name} - {record.getMessage()}"

        # Add extra fields if present
        extra_parts = []
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
                "getMessage",
                "asctime",
            ]:
                extra_parts.append(f"{key}={value}")

        if extra_parts:
            base_msg += f" | {' '.join(extra_parts)}"

        # Add exception if present
        if record.exc_info:
            base_msg += f"\n{self.formatException(record.exc_info)}"

        return base_msg


def setup_logging(
    log_level: str = "INFO",
    log_file: str | None = None,
    log_dir: str | None = None,
    json_format: bool = False,
    console_output: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> None:
    """
    Configure application-wide logging settings.

    This function should be called once at application startup to configure
    all logging handlers and formatters.

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional). If log_dir is also provided,
                 log_file is treated as filename only.
        log_dir: Directory for log files (optional)
        json_format: If True, use JSON structured format for file output
        console_output: If True, add console (stdout) handler
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep

    Raises:
        ValueError: If log_level is invalid
    """
    # Validate log level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Setup console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(HumanReadableFormatter(use_colors=True))
        root_logger.addHandler(console_handler)

    # Setup file handler if log file specified
    if log_file:
        # Determine full log file path
        if log_dir:
            log_path = Path(log_dir) / log_file
            log_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

        # Use rotating file handler to prevent unbounded growth
        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(
            str(log_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_handler.setLevel(numeric_level)

        # Use JSON format for files if requested, otherwise human-readable
        if json_format:
            file_handler.setFormatter(StructuredFormatter())
        else:
            file_handler.setFormatter(HumanReadableFormatter(use_colors=False))

        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the specified name.

    This is a convenience wrapper around logging.getLogger() that ensures
    consistent logger naming across the application.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        Logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Order placed", extra={"symbol": "AAPL", "qty": 100})
    """
    return logging.getLogger(name)


def log_with_context(logger: logging.Logger, level: str, message: str, **context: Any) -> None:
    """
    Log a message with structured context.

    This is a convenience function for adding structured context to log messages.

    Args:
        logger: Logger instance to use
        level: Log level (debug, info, warning, error, critical)
        message: Log message
        **context: Additional context as keyword arguments

    Example:
        log_with_context(
            logger,
            "info",
            "Order executed",
            symbol="AAPL",
            side="BUY",
            qty=100,
            price=150.0
        )
    """
    log_method = getattr(logger, level.lower())
    log_method(message, extra=context)
