import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock
from src.services.search_service import (
    PerplexityProvider, 
    ExaAIProvider, 
    SerpAPIProvider,
    SearchResults, 
    SearchCache, 
    CacheEntry,
    ProductDetails,
    PresenceConfidence
)

@pytest.fixture
def mock_cache():
    c = MagicMock(spec=SearchCache)
    c.get = AsyncMock(return_value=None)
    c.set = AsyncMock()
    return c

@pytest.mark.asyncio
async def test_perplexity_provider_extra(mock_settings, mock_cache):
    # Setup mock key
    mock_key = MagicMock()
    mock_key.get_secret_value.return_value = "key"
    mock_settings.perplexity_api_key = mock_key
    
    provider = PerplexityProvider(settings=mock_settings, cache=mock_cache)
    
    # Test verify_brand_presence
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Yes, they sell on Amazon."}}]
        }
        mock_post.return_value = mock_response
        
        res = await provider.verify_brand_presence("test.com", "Test")
        assert isinstance(res, PresenceConfidence)
        assert res.is_present is True

@pytest.mark.asyncio
async def test_exa_provider_get_details(mock_settings, mock_cache):
    mock_key = MagicMock()
    mock_key.get_secret_value.return_value = "key"
    mock_settings.exa_api_key = mock_key
    
    provider = ExaAIProvider(settings=mock_settings, cache=mock_cache)
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [{"title": "Details", "text": "Product content"}]
        }
        mock_post.return_value = mock_response
        
        res = await provider.get_product_details("B001234567")
        assert isinstance(res, ProductDetails)
        assert res.asin == "B001234567"

@pytest.mark.asyncio
async def test_serpapi_google_shopping(mock_settings, mock_cache):
    mock_key = MagicMock()
    mock_key.get_secret_value.return_value = "key"
    mock_settings.serpapi_api_key = mock_key
    
    provider = SerpAPIProvider(settings=mock_settings, cache=mock_cache)
    
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "shopping_results": [{"title": "S1", "link": "https://amazon.com/dp/B001234567"}]
        }
        mock_get.return_value = mock_response
        
        res = await provider.search_google_shopping("query")
        assert isinstance(res, SearchResults)
        assert len(res.items) == 1

@pytest.mark.asyncio
async def test_search_cache_expiry():
    cache = SearchCache(max_size=2)
    entry = CacheEntry(data="val", created_at=datetime.utcnow() - timedelta(seconds=10), ttl_seconds=5)
    cache._cache["key"] = entry
    
    res = await cache.get("key")
    assert res is None
    assert "key" not in cache._cache

@pytest.mark.asyncio
async def test_search_cache_eviction():
    cache = SearchCache(max_size=2)
    await cache.set("k1", "v1")
    await cache.set("k2", "v2")
    # This should trigger eviction
    await cache.set("k3", "v3")
    
    assert len(cache._cache) < 3
    assert cache.get_stats()["size"] < 3
