import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.services.search_service import SearchService, SearchResults, SearchResultItem, PresenceConfidence
from src.models.schemas import ConfidenceLevel, AmazonProduct

@pytest.mark.asyncio
async def test_search_service_init(mock_settings):
    service = SearchService(settings=mock_settings)
    assert service.settings == mock_settings
    assert service.cache is not None

@pytest.mark.asyncio
async def test_search_brand_on_amazon(mock_settings):
    service = SearchService(settings=mock_settings)
    
    # Mocking _try_providers to return mock results
    mock_results = SearchResults(
        query="Test Brand",
        provider="serpapi",
        items=[
            SearchResultItem(
                title="Product 1",
                url="https://amazon.com/dp/B001",
                asin="B001",
                price=10.0
            )
        ]
    )
    
    with patch.object(service, "_try_providers", new_callable=AsyncMock) as mock_try:
        mock_try.return_value = mock_results
        
        results = await service.search_brand_on_amazon("testbrand.com", "Test Brand")
        
        assert isinstance(results, SearchResults)
        assert results.items[0].title == "Product 1"
        mock_try.assert_called_once()

@pytest.mark.asyncio
async def test_verify_brand_presence(mock_settings):
    service = SearchService(settings=mock_settings)
    
    mock_presence = PresenceConfidence(
        is_present=True,
        confidence=ConfidenceLevel.HIGH,
        confidence_score=0.95,
        evidence=["Storefront verified"],
        provider="serpapi"
    )
    
    with patch.object(service, "_try_providers", new_callable=AsyncMock) as mock_try:
        mock_try.return_value = mock_presence
        
        presence = await service.verify_brand_presence("testbrand.com", "Test Brand")
        
        assert presence.is_present is True
        assert presence.confidence == ConfidenceLevel.HIGH
        mock_try.assert_called_once()

@pytest.mark.asyncio
async def test_search_brand_products_conversion(mock_settings):
    service = SearchService(settings=mock_settings)
    
    # Mock search_brand_on_amazon to return SearchResults
    mock_results = SearchResults(
        query="Test Brand",
        provider="serpapi",
        items=[
            SearchResultItem(
                asin="B001234567",
                title="Product 1",
                url="https://amazon.com/dp/B001",
                price=10.0,
                rating=4.5,
                reviews=100
            )
        ]
    )
    
    with patch.object(service, "_try_providers", new_callable=AsyncMock) as mock_try:
        mock_try.return_value = mock_results
        
        products = await service.search_brand_products("Test Brand")
        
        assert len(products) == 1
        assert isinstance(products[0], AmazonProduct)
        assert products[0].asin == "B001234567"
        assert products[0].price == 10.0
