"""
Structured logging configuration.

Provides consistent logging across the application with JSON formatting.
"""

import logging
import sys
from datetime import datetime
from typing import Optional

import structlog
from structlog.types import Processor


def setup_logging(
    level: str = "INFO",
    json_format: bool = True,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure application-wide structured logging.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_format: Whether to output logs in JSON format.
        log_file: Optional file path for logging output.
    """
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.format_exc_info,
    ]
    
    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger for the given module name.
    
    Args:
        name: Logger name, typically __name__.
        
    Returns:
        Configured structlog logger instance.
    """
    return structlog.get_logger(name)


class LogContext:
    """Context manager for adding temporary log context."""
    
    def __init__(self, **kwargs):
        self.context = kwargs
        self._token = None
    
    def __enter__(self):
        self._token = structlog.contextvars.bind_contextvars(**self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._token:
            structlog.contextvars.unbind_contextvars(*self.context.keys())
