import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from src.utils.retry import async_retry, CircuitBreaker, ErrorHandler, ServiceUnavailableError, RateLimitError

@pytest.mark.asyncio
async def test_async_retry_success():
    mock_func = AsyncMock(return_value="success")
    
    result = await async_retry(max_attempts=3)(mock_func)()
    
    assert result == "success"
    assert mock_func.call_count == 1

@pytest.mark.asyncio
async def test_async_retry_fail_then_success():
    mock_func = AsyncMock()
    mock_func.side_effect = [ValueError("Fail"), "success"]
    
    # Use small backoff for testing
    result = await async_retry(max_attempts=3, backoff_factor=0.1)(mock_func)()
    
    assert result == "success"
    assert mock_func.call_count == 2

@pytest.mark.asyncio
async def test_async_retry_exhausted():
    mock_func = AsyncMock(side_effect=ValueError("Permanent Fail"))
    
    with pytest.raises(ValueError, match="Permanent Fail"):
        await async_retry(max_attempts=2, backoff_factor=0.1)(mock_func)()
        
    assert mock_func.call_count == 2

@pytest.mark.asyncio
async def test_circuit_breaker_closed():
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
    mock_func = AsyncMock(return_value="ok")
    
    assert await cb.call(mock_func) == "ok"
    assert cb.state == "closed"

@pytest.mark.asyncio
async def test_circuit_breaker_opens():
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
    mock_func = AsyncMock(side_effect=ValueError("Fail"))
    
    # 1st fail
    with pytest.raises(ValueError):
        await cb.call(mock_func)
    assert cb.state == "closed"
    
    # 2nd fail -> Open
    with pytest.raises(ValueError):
        await cb.call(mock_func)
    assert cb.state == "open"
    
    # Subsequent calls fail immediately
    with pytest.raises(ServiceUnavailableError):
        await cb.call(mock_func)

@pytest.mark.asyncio
async def test_circuit_breaker_half_open():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
    mock_func = AsyncMock(side_effect=ValueError("Fail"))
    
    # Open it
    with pytest.raises(ValueError):
        await cb.call(mock_func)
    assert cb.state == "open"
    
    await asyncio.sleep(0.2)
    
    # Now it should be half-open and attempt reset
    mock_func.side_effect = None
    mock_func.return_value = "recovered"
    
    assert await cb.call(mock_func) == "recovered"
    assert cb.state == "closed"

def test_error_handler_categorization():
    assert ErrorHandler.categorize_error(RateLimitError("too many requests")) == "RATE_LIMIT_ERROR"
    assert ErrorHandler.categorize_error(ValueError("bad data")) == "VALIDATION_ERROR"
    assert ErrorHandler.categorize_error(Exception("timeout happened")) == "TIMEOUT_ERROR"

def test_error_handler_fallback_strategy():
    strategy = ErrorHandler.get_fallback_strategy("NETWORK_ERROR")
    assert strategy() == {"status": "failed", "retryable": True, "action": "retry_with_backoff"}
    
    strategy = ErrorHandler.get_fallback_strategy("UNKNOWN")
    assert strategy()["action"] == "log_and_raise"
