import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.extractors.brand_extractor import BrandExtractor
from src.models.schemas import ExtractionResult, ConfidenceLevel

@pytest.mark.asyncio
async def test_brand_extractor_full_flow(mock_search_service, mock_llm_service, mock_settings):
    extractor = BrandExtractor(
        search_service=mock_search_service,
        llm_service=mock_llm_service,
        settings=mock_settings
    )
    
    # Mock search service response
    from src.services.search_service import SearchResults, SearchResultItem
    mock_search_service.search_brand_on_amazon.return_value = SearchResults(
        query="Test Brand",
        provider="serpapi",
        items=[
            SearchResultItem(
                asin="B00TEST001",
                title="Test Product",
                url="https://amazon.com/dp/B00TEST001",
                price=29.99
            )
        ]
    )
    
    # Mock LLM service response for structuring
    mock_llm_service.extract_structured_data.return_value = MagicMock(
        products=[{
            "asin": "B00TEST001",
            "title": "Test Product",
            "price": 29.99,
            "brand": "Test Brand"
        }],
        confidence_reasoning="Strong match",
        has_official_store=True,
        verified_seller=True
    )
    
    # We need to mock validate_and_retry behavior or just the final ExtractionResult
    # Actually, BrandExtractor._structure_with_llm calls llm_service.extract_structured_data
    # which returns an ExtractionSchema (local class).
    
    from src.models.schemas import AmazonPresence, ConfidenceLevel, ExtractionMetadata
    
    result = await extractor.extract_brand_data("testbrand.com")
    
    assert isinstance(result, ExtractionResult)
    assert result.brand_name == "Testbrand"
    assert result.amazon_presence.found is True
    assert len(result.top_products) >= 1
    assert result.metadata.data_source == "serpapi"

@pytest.mark.asyncio
async def test_brand_extractor_invalid_domain(mock_search_service, mock_llm_service, mock_settings):
    extractor = BrandExtractor(mock_search_service, mock_llm_service, mock_settings)
    
    with pytest.raises(ValueError, match="Invalid domain"):
        await extractor.extract_brand_data("invalid-domain")
