import pytest
import httpx
import json
from unittest.mock import AsyncMock, patch, MagicMock
from src.services.search_service import SerpAPIProvider, ExaAIProvider, SearchCache, RateLimiter, SearchResults
from src.models.schemas import ConfidenceLevel

@pytest.fixture
def mock_httpx_client():
    client = AsyncMock(spec=httpx.AsyncClient)
    # Mocking common response behavior
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {}
    client.get.return_value = mock_response
    client.post.return_value = mock_response
    return client

@pytest.mark.asyncio
async def test_search_cache():
    cache = SearchCache(max_size=2, default_ttl=1)
    await cache.set("k1", "v1")
    await cache.set("k2", "v2")
    
    assert await cache.get("k1") == "v1"
    
    # Trigger eviction
    await cache.set("k3", "v3")
    # max_size=2, so one should be evicted
    stats = cache.get_stats()
    # It evicts 1/4 of cache if full, so 2 -> 1 entry left before adding k3? 
    # Logic is: if len >= max_size: evict. 2 >= 2: evict.
    assert stats["size"] <= 2

@pytest.mark.asyncio
async def test_rate_limiter():
    rl = RateLimiter(requests_per_second=10)
    wait = await rl.acquire()
    assert wait == 0.0
    
    rl_slow = RateLimiter(requests_per_second=0.1)
    # First one immediate
    await rl_slow.acquire()
    # Second one should have wait
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        await rl_slow.acquire()
        assert mock_sleep.called

@pytest.mark.asyncio
async def test_serpapi_provider_search(mock_settings, mock_httpx_client):
    provider = SerpAPIProvider(settings=mock_settings)
    provider._client = mock_httpx_client
    
    mock_httpx_client.get.return_value.json.return_value = {
        "organic_results": [{"title": "P1", "asin": "B001", "link": "http", "price": "$10"}]
    }
    
    res = await provider.search_amazon("test")
    assert isinstance(res, SearchResults)
    assert len(res.items) == 1
    assert res.items[0].asin == "B001"

@pytest.mark.asyncio
async def test_serpapi_provider_verify_presence(mock_settings, mock_httpx_client):
    provider = SerpAPIProvider(settings=mock_settings)
    provider._client = mock_httpx_client
    
    # Mock search response inside verify_brand_presence
    mock_httpx_client.get.return_value.json.return_value = {
        "organic_results": [
            {"title": "P1", "asin": "B001", "brand": "BrandX", "link": "http", "price": "$10"}
        ] * 10
    }
    
    presence = await provider.verify_brand_presence("brandx.com", "BrandX")
    assert presence.is_present is True
    assert presence.confidence == ConfidenceLevel.HIGH

@pytest.mark.asyncio
async def test_exa_provider_is_configured(mock_settings):
    # Test with None
    mock_settings.exa_api_key = None
    provider = ExaAIProvider(settings=mock_settings)
    assert provider.is_configured is False
    
    # Test with Mock SecretStr
    mock_key = MagicMock()
    mock_key.get_secret_value.return_value = "key"
    mock_settings.exa_api_key = mock_key
    assert provider.is_configured is True

def test_serpapi_parsing():
    # Test internal static methods
    assert SerpAPIProvider._parse_price("$19.99") == 19.99
    assert SerpAPIProvider._parse_price("1,234.56") == 1234.56
    assert SerpAPIProvider._parse_review_count("1,234 reviews") == 1234
    assert SerpAPIProvider._extract_asin("https://amazon.com/dp/B001234567") == "B001234567"
