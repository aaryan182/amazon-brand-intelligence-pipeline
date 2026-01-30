"""
Resilient error handling utilities.

Provides retry logic, circuit breaker pattern, and centralized error handling strategies.
"""

import asyncio
import logging
from datetime import datetime
from functools import wraps
from typing import Callable, Optional, Type

# Use standard logging
logger = logging.getLogger(__name__)

# =============================================================================
# Custom Exceptions
# =============================================================================

class AppError(Exception):
    """Base application exception."""
    pass

class NetworkError(AppError):
    pass

class RateLimitError(AppError):
    pass

class ValidationError(AppError):
    pass

class APIKeyError(AppError):
    pass

class AppTimeoutError(AppError):
    pass

class UnknownBrandError(AppError):
    pass

class ServiceUnavailableError(AppError):
    pass

# =============================================================================
# Retry Decorator
# =============================================================================

def async_retry(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None
):
    """
    Retry decorator with exponential backoff.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            wait_time = 1.0  # Initial wait time
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts:
                        logger.warning(f"Final attempt {attempt} failed for {func.__name__}: {e}")
                        break
                    
                    # Backoff
                    sleep_time = wait_time * (backoff_factor ** (attempt - 1))
                    
                    if on_retry:
                        try:
                            on_retry(attempt, e)
                        except Exception:
                            pass # suppress callback errors
                            
                    logger.warning(
                        f"Attempt {attempt} failed for {func.__name__}: {e}. "
                        f"Retrying in {sleep_time:.2f}s..."
                    )
                    
                    await asyncio.sleep(sleep_time)
            
            if last_exception:
                raise last_exception
        return wrapper
    return decorator

# =============================================================================
# Circuit Breaker
# =============================================================================

class CircuitBreaker:
    """
    Circuit breaker pattern for external services.
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, name: str = "CircuitBreaker"):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name
        
        self.failures = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half-open
    
    def _should_attempt_reset(self) -> bool:
        if self.state != "open" or not self.last_failure_time:
            return False
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed > self.recovery_timeout

    def _on_success(self):
        if self.state != "closed":
            logger.info(f"Circuit {self.name} recovering. State: Closed.")
        self.failures = 0
        self.state = "closed"
        self.last_failure_time = None

    def _on_failure(self):
        self.failures += 1
        self.last_failure_time = datetime.now()
        
        if self.state == "half-open":
            self.state = "open"
            logger.warning(f"Circuit {self.name} trial failed. State: Re-Opened.")
        elif self.failures >= self.failure_threshold and self.state == "closed":
            self.state = "open"
            logger.error(f"Circuit {self.name} threshold reached. State: Open.")

    async def call(self, func: Callable, *args, **kwargs):
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
                logger.info(f"Circuit {self.name} attempting reset. State: Half-Open.")
            else:
                remaining = self.recovery_timeout - (datetime.now() - self.last_failure_time).total_seconds()
                raise ServiceUnavailableError(f"Circuit {self.name} is OPEN. Retry in {remaining:.1f}s")
        
        try:
            result = await func(*args, **kwargs)
            if self.state == "half-open":
                self._on_success()
            else:
                self.failures = 0
            return result
        except Exception as e:
            if self.state == "half-open":
                self.state = "open"
                self.last_failure_time = datetime.now()
            else:
                self._on_failure()
            raise e

# =============================================================================
# Error Handler
# =============================================================================

class ErrorHandler:
    """Centralized error handling and categorization."""
    
    @staticmethod
    def categorize_error(error: Exception) -> str:
        """Categorize errors for appropriate handling."""
        if isinstance(error, (NetworkError, ConnectionError, OSError)):
            return "NETWORK_ERROR"
        if isinstance(error, RateLimitError):
            return "RATE_LIMIT_ERROR"
        if isinstance(error, (ValidationError, ValueError, TypeError)):
            return "VALIDATION_ERROR"
        if isinstance(error, APIKeyError):
            return "API_KEY_ERROR"
        if isinstance(error, (AppTimeoutError, asyncio.TimeoutError)):
            return "TIMEOUT_ERROR"
        if isinstance(error, UnknownBrandError):
            return "UNKNOWN_BRAND"
        
        # Check string content
        err_str = str(error).lower()
        if "rate limit" in err_str: return "RATE_LIMIT_ERROR"
        if "timeout" in err_str: return "TIMEOUT_ERROR"
        if "api key" in err_str or "unauthorized" in err_str: return "API_KEY_ERROR"
        if "connection" in err_str: return "NETWORK_ERROR"
        
        return "UNKNOWN_ERROR"
    
    @staticmethod
    def get_fallback_strategy(error_type: str) -> Callable:
        """Get fallback strategy for error type."""
        strategies = {
            "NETWORK_ERROR": lambda: {"status": "failed", "retryable": True, "action": "retry_with_backoff"},
            "RATE_LIMIT_ERROR": lambda: {"status": "failed", "retryable": True, "action": "wait_and_retry"},
            "VALIDATION_ERROR": lambda: {"status": "failed", "retryable": False, "action": "return_error"},
            "API_KEY_ERROR": lambda: {"status": "failed", "retryable": False, "action": "fail_fast"},
            "TIMEOUT_ERROR": lambda: {"status": "failed", "retryable": True, "action": "retry_extended_timeout"},
            "UNKNOWN_BRAND": lambda: {"status": "completed", "result": "not_found", "action": "return_default"},
        }
        return strategies.get(error_type, lambda: {"status": "failed", "action": "log_and_raise"})
