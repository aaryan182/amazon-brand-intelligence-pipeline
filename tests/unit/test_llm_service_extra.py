import pytest
import asyncio
import time
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock
from src.services.llm_service import ClaudeService, RateLimiter, TokenUsage, CacheEntry, TaskType

@pytest.fixture
def service(mock_settings):
    return ClaudeService(settings=mock_settings)

@pytest.mark.asyncio
async def test_rate_limiter_acquire():
    # Test with very low limit to trigger wait
    limiter = RateLimiter(max_requests=1, window_seconds=1)
    await limiter.acquire()
    
    # Second acquire should wait almost 1 second
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        await limiter.acquire()
        assert mock_sleep.called

@pytest.mark.asyncio
async def test_token_usage_cost():
    # Test opus model
    usage = TokenUsage(input_tokens=1000, output_tokens=1000)
    usage.calculate_cost("claude-3-opus-20240229")
    assert usage.estimated_cost > 0
    
    # Test sonnet model - use exact model name from TOKEN_COSTS
    usage2 = TokenUsage(input_tokens=1000, output_tokens=1000)
    usage2.calculate_cost("claude-3-5-sonnet-20241022")
    assert usage2.estimated_cost > 0
    
    # unknown model doesn't change cost from 0.0
    usage3 = TokenUsage(input_tokens=1000, output_tokens=1000)
    usage3.calculate_cost("unknown-model")
    assert usage3.estimated_cost == 0.0

@pytest.mark.asyncio
async def test_cache_entry_expiry():
    entry = CacheEntry(response="test", timestamp=datetime.utcnow(), tokens_used=10)
    assert entry.is_expired(3600) is False
    assert entry.is_expired(-1) is True

@pytest.mark.asyncio
async def test_llm_service_call_api_rate_limit_retry(service):
    # Mock anthropic client to raise RateLimitError then succeed
    from anthropic import RateLimitError
    mock_response = MagicMock()
    mock_response.status_code = 429
    
    messages = [{"role": "user", "content": "test"}]
    
    with patch.object(service.client.messages, "create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = [
            RateLimitError(message="Rate limited", response=mock_response, body={}),
            MagicMock(content=[MagicMock(text="Success")], usage=MagicMock(input_tokens=1, output_tokens=1))
        ]
        
        with patch("asyncio.sleep", new_callable=AsyncMock):
            res, usage = await service._call_api(messages, task_type=TaskType.EXTRACTION)
            assert res == "Success"
            assert mock_create.call_count == 2
