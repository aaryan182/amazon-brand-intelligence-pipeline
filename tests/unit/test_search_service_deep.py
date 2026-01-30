import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from src.services.search_service import (
    SearchService, 
    SerpAPIProvider, 
    ExaAIProvider, 
    PerplexityProvider,
    ProviderError,
    RateLimitError,
    ConfigurationError
)

@pytest.fixture
def mock_provider():
    p = MagicMock()
    p.name = "mock"
    p.is_configured = True
    p.connect = AsyncMock()
    p.search_amazon = AsyncMock()
    return p

@pytest.mark.asyncio
async def test_search_service_init_no_providers(mock_settings):
    # Ensure no providers are configured
    mock_settings.serpapi_api_key = None
    mock_settings.exa_api_key = None
    mock_settings.perplexity_api_key = None
    
    # We need to ensure SerpAPIProvider.is_configured returns False
    with patch("src.services.search_service.SerpAPIProvider.is_configured", False):
        with patch("src.services.search_service.ExaAIProvider.is_configured", False):
            with patch("src.services.search_service.PerplexityProvider.is_configured", False):
                service = SearchService(settings=mock_settings)
                with pytest.raises(ConfigurationError):
                    await service._init_providers()

@pytest.mark.asyncio
async def test_search_service_failover(mock_settings):
    p1 = MagicMock()
    p1.name = "p1"
    p1.search_amazon = AsyncMock(side_effect=ProviderError("Fail"))
    
    p2 = MagicMock()
    p2.name = "p2"
    p2.search_amazon = AsyncMock(return_value={"data": "success"})
    
    service = SearchService(settings=mock_settings)
    service._initialized = True
    service._providers = [p1, p2]
    
    with patch.object(service, "_get_available_providers", return_value=[p1, p2]):
        result = await service._try_providers("search_amazon", "query")
        assert result == {"data": "success"}
        assert p1.search_amazon.called
        assert p2.search_amazon.called

@pytest.mark.asyncio
async def test_search_service_all_fail(mock_settings):
    p1 = MagicMock()
    p1.name = "p1"
    p1.search_amazon = AsyncMock(side_effect=Exception("Unexpected"))
    
    service = SearchService(settings=mock_settings)
    service._initialized = True
    service._providers = [p1]
    
    with patch.object(service, "_get_available_providers", return_value=[p1]):
        with pytest.raises(ProviderError) as exc:
            await service._try_providers("search_amazon", "query")
        assert "All providers failed" in str(exc.value)

@pytest.mark.asyncio
async def test_search_service_rate_limit_failover(mock_settings):
    p1 = MagicMock()
    p1.name = "p1"
    p1.search_amazon = AsyncMock(side_effect=RateLimitError("Rate limit exceeded"))
    
    p2 = MagicMock()
    p2.name = "p2"
    p2.search_amazon = AsyncMock(return_value={"data": "success"})
    
    service = SearchService(settings=mock_settings)
    service._initialized = True
    service._providers = [p1, p2]
    
    with patch.object(service, "_get_available_providers", return_value=[p1, p2]):
        result = await service._try_providers("search_amazon", "query")
        assert result == {"data": "success"}

@pytest.mark.asyncio
async def test_search_service_context_manager(mock_settings):
    with patch("src.services.search_service.SearchService._init_providers", new_callable=AsyncMock) as mock_init:
        async with SearchService(settings=mock_settings) as service:
            assert mock_init.called
            # mock close
            service.close = AsyncMock()
        assert service.close.called
