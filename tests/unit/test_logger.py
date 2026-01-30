import pytest
import structlog
import logging
from src.utils.logger import setup_logging, get_logger, LogContext

def test_setup_logging():
    # Calling it shouldn't crash
    setup_logging(level="DEBUG", json_format=False)
    setup_logging(level="INFO", json_format=True)
    
def test_get_logger():
    logger = get_logger("test_module")
    assert logger is not None
    # Test logging doesn't crash
    logger.info("test message", key="value")

def test_log_context():
    logger = get_logger("test_module")
    with LogContext(request_id="123"):
        # We can't easily verify contextvars state without internal structlog inspection,
        # but we can ensure it runs.
        logger.info("with context")
    logger.info("without context")
