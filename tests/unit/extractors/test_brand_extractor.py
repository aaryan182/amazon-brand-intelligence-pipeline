"""
Unit tests for the BrandExtractor component.

Tests cover:
- Domain processing and validation
- Multi-query search strategy execution
- LLM data structuring and fallback
- Product enrichment logic
- Confidence scoring
- Category detection
- End-to-end extraction flow
- Error handling scenarios
"""

import asyncio
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.extractors.brand_extractor import (
    BrandExtractor,
    BrandNameVariation,
    ExtractionPrompts,
    ExtractionResult,
    ExtractionMetrics,
    ExtractionStage,
    StageResult,
)
from src.models.schemas import AmazonProduct, AmazonPresence, ConfidenceLevel, AmazonCategory
from src.services.llm_service import ClaudeService
from src.services.search_service import (
    SearchService,
    SearchResults,
    SearchResultItem,
    ProductDetails,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def mock_env():
    """Set required environment variables to satisfy Settings validation."""
    with patch.dict(os.environ, {
        "ANTHROPIC_API_KEY": "sk-test-key",
        "SERPAPI_API_KEY": "test-key",
    }):
        yield

@pytest.fixture
def mock_settings():
    """Mock application settings."""
    settings = MagicMock()
    settings.anthropic_api_key = "test-key"
    settings.serpapi_api_key = "test-key"
    return settings


@pytest.fixture
def mock_search_service():
    """Mock SearchService."""
    service = MagicMock(spec=SearchService)
    service.search_brand_on_amazon = AsyncMock()
    service.get_product_details = AsyncMock()
    return service


@pytest.fixture
def mock_llm_service():
    """Mock ClaudeService."""
    service = MagicMock(spec=ClaudeService)
    service.extract_structured_data = AsyncMock()
    return service


@pytest.fixture
def brand_extractor(mock_search_service, mock_llm_service, mock_settings):
    """Create BrandExtractor instance."""
    extractor = BrandExtractor(
        search_service=mock_search_service,
        llm_service=mock_llm_service,
        settings=mock_settings,
        max_products=10,
        enrich_products=True,
    )
    # Initialize metrics specifically for component-level testing
    extractor._metrics = ExtractionMetrics()
    return extractor


@pytest.fixture
def sample_search_results():
    """Create sample search results with valid ASINs (10 chars)."""
    return SearchResults(
        query="Nike",
        provider="serpapi",
        items=[
            SearchResultItem(
                asin="B0000NIKE1",
                title="Nike Air Max",
                url="https://amazon.com/dp/B0000NIKE1",
                price=120.0,
                rating=4.5,
                review_count=1000,
                brand="Nike",
                category="Shoes",
                position=1,
            ),
            SearchResultItem(
                asin="B0000NIKE2",
                title="Nike Running Shirt",
                url="https://amazon.com/dp/B0000NIKE2",
                price=40.0,
                rating=4.2,
                review_count=500,
                brand="Nike",
                category="Clothing",
                position=2,
            ),
        ],
    )


@pytest.fixture
def sample_llm_extraction_response():
    """Create sample LLM extraction object with valid ASINs."""
    class MockExtraction:
        products = [
            {
                "asin": "B0000NIKE1",
                "title": "Nike Air Max",
                "price": 120.0,
                "rating": 4.5,
                "review_count": 1000,
                "brand": "Nike",
                "category": "Clothing, Shoes & Jewelry",
                "rank": 1,
            },
            {
                "asin": "B0000NIKE2",
                "title": "Nike Running Shirt",
                "price": 40.0,
                "rating": 4.2,
                "review_count": 500,
                "brand": "Nike",
                "category": "Clothing, Shoes & Jewelry",
                "rank": 2,
            },
        ]
        confidence_reasoning = "Official store and products found"
        has_official_store = True
        verified_seller = True
        
    return MockExtraction()


# =============================================================================
# Domain Processing Tests
# =============================================================================

class TestDomainProcessing:
    """Tests for domain validation and processing."""
    
    @pytest.mark.asyncio
    async def test_validate_domain_success(self, brand_extractor):
        """Test successful domain validation."""
        clean = await brand_extractor._validate_domain("https://www.Nike.com/us/")
        assert clean == "www.nike.com"
        
        clean = await brand_extractor._validate_domain("store.brand.io")
        assert clean == "store.brand.io"
    
    @pytest.mark.asyncio
    async def test_validate_domain_failure(self, brand_extractor):
        """Test domain validation failure."""
        with pytest.raises(ValueError, match="Invalid domain"):
            await brand_extractor._validate_domain("invalid-domain")
            
        with pytest.raises(ValueError, match="cannot be empty"):
            await brand_extractor._validate_domain("")
    
    @pytest.mark.asyncio
    async def test_extract_brand_name(self, brand_extractor):
        """Test brand name extraction."""
        # Simple
        brand = await brand_extractor._extract_brand_name("nike.com")
        assert brand.primary == "Nike"
        
        # Compound
        brand = await brand_extractor._extract_brand_name("cocacola.com")
        assert brand.primary == "Cocacola"
        
        # Hyphenated
        brand = await brand_extractor._extract_brand_name("go-pro.com")
        assert brand.primary == "Go Pro"
        assert "gopro" in brand.variations


# =============================================================================
# Amazon Search Tests
# =============================================================================

class TestAmazonSearch:
    """Tests for Amazon search strategy."""
    
    @pytest.mark.asyncio
    async def test_search_execution(self, brand_extractor, mock_search_service, sample_search_results):
        """Test multi-query search optimization."""
        mock_search_service.search_brand_on_amazon.return_value = sample_search_results
        
        brand = BrandNameVariation(primary="Nike", domain_based="nike", variations=[])
        
        results = await brand_extractor._search_amazon(brand, "nike.com")
        
        # Should persist across multiple queries
        assert len(results) >= 1
        assert mock_search_service.search_brand_on_amazon.called
        
        # Check metrics
        metrics = brand_extractor.get_metrics()
        assert metrics is not None
        assert metrics.search_queries > 0
        assert metrics.products_found == 2
    
    @pytest.mark.asyncio
    async def test_early_termination(self, brand_extractor, mock_search_service):
        """Test search stops early if enough products found."""
        # Mock a result with many items
        many_items = [
            SearchResultItem(
                asin=f"B0000000{i:02d}", title=f"Item {i}", url="https://amazon.com", brand="Nike", price=10.0,
                rating=4.5, review_count=10, position=i
            ) for i in range(25)
        ]
        
        result = SearchResults(query="Nike", provider="serpapi", items=many_items)
        mock_search_service.search_brand_on_amazon.return_value = result
        
        brand = BrandNameVariation(primary="Nike", domain_based="nike", variations=[])
        
        results = await brand_extractor._search_amazon(brand, "nike.com")
        
        # Should only execute 1 query because >20 items found
        assert mock_search_service.search_brand_on_amazon.call_count == 1
        assert len(results) == 1


# =============================================================================
# LLM Structuring Tests
# =============================================================================

class TestLLMStructuring:
    """Tests for LLM data structuring."""
    
    @pytest.mark.asyncio
    async def test_structure_success(
        self,
        brand_extractor,
        mock_llm_service,
        sample_search_results,
        sample_llm_extraction_response
    ):
        """Test successful LLM structuring."""
        mock_llm_service.extract_structured_data.return_value = sample_llm_extraction_response
        
        brand = BrandNameVariation(primary="Nike", domain_based="nike", variations=[])
        
        structured = await brand_extractor._structure_with_llm(
            [sample_search_results], brand, "nike.com"
        )
        
        assert structured["has_official_store"] is True
        assert structured["verified_seller"] is True
        assert len(structured["products"]) == 2
        assert structured["products"][0]["asin"] == "B0000NIKE1"
    
    @pytest.mark.asyncio
    async def test_structure_fallback(self, brand_extractor, mock_llm_service, sample_search_results):
        """Test fallback when LLM fails."""
        mock_llm_service.extract_structured_data.side_effect = Exception("LLM Error")
        
        brand = BrandNameVariation(primary="Nike", domain_based="nike", variations=[])
        
        structured = await brand_extractor._structure_with_llm(
            [sample_search_results], brand, "nike.com"
        )
        
        # Should fall back to raw results
        assert len(structured["products"]) == 2
        assert structured["products"][0]["asin"] == "B0000NIKE1"
        assert structured["has_official_store"] is False  # Default


# =============================================================================
# Product Enrichment Tests
# =============================================================================

class TestProductEnrichment:
    """Tests for product enrichment."""
    
    @pytest.mark.asyncio
    async def test_enrich_products(self, brand_extractor, mock_search_service):
        """Test enriching products with details."""
        structured_data = {
            "products": [
                {"asin": "B0000NIKE1", "title": "Basic Title", "price": 100.0}
            ],
            "has_official_store": True,
        }
        
        # Mock details response
        details = ProductDetails(
            asin="B0000NIKE1",
                title="Enriched Title",
                url="https://amazon.com/dp/B0000NIKE1",
                price=100.0,
                rating=4.5,
            review_count=1000,
            brand="Nike",
            categories=["Shoes"],
            images=["img.jpg"],
            feature_bullets=["Feature 1"],
            description="Desc",
        )
        mock_search_service.get_product_details.return_value = details
        
        enriched = await brand_extractor._enrich_products(structured_data)
        
        assert len(enriched) == 1
        assert enriched[0].title == "Enriched Title"
        assert enriched[0].image_url == "img.jpg"
        
        metrics = brand_extractor.get_metrics()
        assert metrics is not None
        assert metrics.products_enriched == 1


# =============================================================================
# Confidence Scoring Tests
# =============================================================================

class TestConfidenceScoring:
    """Tests for confidence scoring."""
    
    @pytest.mark.asyncio
    async def test_calculate_confidence_high(self, brand_extractor):
        """Test high confidence scenario."""
        products = [
            AmazonProduct(
                asin=f"B0000000{i:02d}", title=f"Nike Item {i}",
                brand="Nike", rating=4.5, review_count=1000
            ) for i in range(20)
        ]
        
        brand = BrandNameVariation(primary="Nike", domain_based="nike", variations=[])
        structured = {"has_official_store": True}
        
        level, score, evidence = await brand_extractor._calculate_confidence(
            products, brand, structured
        )
        
        assert level == ConfidenceLevel.HIGH
        assert score > 0.8
        assert "Official Amazon store found" in evidence
    
    @pytest.mark.asyncio
    async def test_calculate_confidence_low(self, brand_extractor):
        """Test low confidence scenario."""
        products = [
            AmazonProduct(
                asin="B000000001", title="Random Item",
                brand="Other", rating=None, review_count=0
            )
        ]
        
        brand = BrandNameVariation(primary="Nike", domain_based="nike", variations=[])
        structured = {"has_official_store": False}
        
        level, score, evidence = await brand_extractor._calculate_confidence(
            products, brand, structured
        )
        
        assert level == ConfidenceLevel.LOW
        assert score < 0.4
        assert "Minimal product presence: 1 product" in evidence


# =============================================================================
# End-to-End Tests
# =============================================================================

class TestEndToEnd:
    """End-to-end extraction tests."""
    
    def test_initialization(self, brand_extractor):
        """Test component initialization."""
        assert brand_extractor.search_service is not None
        assert brand_extractor.llm_service is not None
        assert brand_extractor.max_products == 10

    @pytest.mark.asyncio
    async def test_end_to_end_extraction(
        self,
        brand_extractor,
        mock_search_service,
        mock_llm_service,
        sample_search_results,
        sample_llm_extraction_response,
    ):
        """Test complete extraction flow."""
        # Setup mocks
        mock_search_service.search_brand_on_amazon.return_value = sample_search_results
        mock_search_service.get_product_details.return_value = ProductDetails(
            asin="B0000NIKE1", title="Title", url="https://amazon.com/dp/B0000NIKE1", price=10.0, rating=4.0, review_count=100,
            brand="Nike", categories=["Shoes"], images=["img.jpg"]
        )
        mock_llm_service.extract_structured_data.return_value = sample_llm_extraction_response
        
        # Execute
        result = await brand_extractor.extract_brand_data("nike.com")
        
        # Verify
        assert isinstance(result, ExtractionResult)
        assert result.brand_name == "Nike"
        assert result.domain == "nike.com"
        assert len(result.top_products) == 2
        assert result.amazon_presence.found is True
        assert result.amazon_presence.confidence == ConfidenceLevel.HIGH
        assert "Shoes" in str(result.primary_category) or "Clothing" in str(result.primary_category)
        
        # Check metrics
        metrics = brand_extractor.get_metrics()
        assert metrics is not None
        assert metrics.total_duration_ms > 0
        assert metrics.products_found == 2
        
    @pytest.mark.asyncio
    async def test_extraction_no_results(
        self,
        brand_extractor,
        mock_search_service,
        mock_llm_service
    ):
        """Test extraction when no products are found."""
        # Empty search results
        empty_result = SearchResults(query="Unknown", provider="serpapi", items=[])
        mock_search_service.search_brand_on_amazon.return_value = empty_result
        
        # Simple extraction response
        class EmptyExtraction:
            products = []
            confidence_reasoning = "None"
            has_official_store = False
            verified_seller = False
            
        mock_llm_service.extract_structured_data.return_value = EmptyExtraction()
        
        result = await brand_extractor.extract_brand_data("unknown-brand.com")
        
        assert result.amazon_presence.found is False
        assert result.amazon_presence.confidence == ConfidenceLevel.LOW
        assert len(result.top_products) == 0
