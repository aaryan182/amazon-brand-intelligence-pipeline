import pytest
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from src.models.schemas import (
    ExtractionResult, 
    StrategicInsight, 
    AmazonProduct, 
    BrandInput,
    AmazonPresence,
    ExtractionMetadata,
    SWOTAnalysis,
    Recommendation,
    ConfidenceLevel,
    MarketPosition
)
from src.config.settings import Settings

@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = MagicMock()
    # Anthropic key must start with sk-ant-api-
    settings.anthropic_api_key.get_secret_value.return_value = "sk-ant-api-mock-key"
    settings.serpapi_api_key.get_secret_value.return_value = "serpapi-test-key"
    settings.exa_api_key.get_secret_value.return_value = "exa-test-key"
    settings.perplexity_api_key.get_secret_value.return_value = "perplexity-test-key"
    
    settings.claude_model = "claude-3-5-sonnet-20241022"
    settings.claude_max_tokens = 4096
    settings.claude_temperature = 0.7
    settings.request_timeout = 30
    settings.max_retries = 3
    settings.max_concurrent_requests = 5
    settings.output_dir = Path("outputs/reports")
    settings.log_dir = Path("logs")
    settings.report_format = "json"
    
    settings.get_search_provider.return_value = "serpapi"
    
    return settings

@pytest.fixture(autouse=True)
def patch_get_settings(mock_settings):
    """Globally patch get_settings to return mock_settings."""
    with patch("src.config.settings.get_settings", return_value=mock_settings):
        # Also patch other common places where get_settings might be imported directly
        with patch("src.pipeline.orchestrator.get_settings", return_value=mock_settings):
            with patch("src.utils.formatters.get_settings", return_value=mock_settings):
                yield mock_settings

@pytest.fixture
def mock_logger():
    return MagicMock()

@pytest.fixture
def sample_products():
    return [
        AmazonProduct(
            title="Test Product 1",
            asin="B00TEST001",
            price=29.99,
            rating=4.5,
            review_count=100,
            url="https://amazon.com/dp/B00TEST001",
            description="A great test product"
        ),
        AmazonProduct(
            title="Test Product 2",
            asin="B00TEST002",
            price=49.99,
            rating=4.8,
            review_count=50,
            url="https://amazon.com/dp/B00TEST002",
            description="Another great test product"
        ),
        AmazonProduct(
            title="Test Product 3",
            asin="B00TEST003",
            price=19.99,
            rating=4.0,
            review_count=10,
            url="https://amazon.com/dp/B00TEST003",
            description="A third test product"
        )
    ]

@pytest.fixture
def sample_extraction_result(sample_products):
    return ExtractionResult(
        brand_name="Test Brand",
        domain="testbrand.com",
        amazon_presence=AmazonPresence(
            found=True,
            confidence=ConfidenceLevel.HIGH,
            evidence=["Storefront found"]
        ),
        primary_category="Electronics",
        estimated_product_count=3,
        top_products=sample_products,
        all_categories=["Electronics", "Computers"],
        competitors_found=["CompBrand A", "CompBrand B"],
        metadata=ExtractionMetadata(
            search_queries_used=["test brand amazon"],
            extraction_duration_seconds=1.5,
            api_calls_made=1,
            data_source="serpapi"
        )
    )

@pytest.fixture
def sample_strategic_insight():
    return StrategicInsight(
        market_position=MarketPosition.CHALLENGER,
        market_position_rationale="Test Brand shows promise but faces stiff competition and needs better PPC strategy.",
        swot=SWOTAnalysis(
            strengths=["Agile", "Low Cost"],
            weaknesses=["Low Brand Awareness"],
            opportunities=["Global Expansion"],
            threats=["Larger Incumbents"]
        ),
        competitive_advantages=["Low price", "Fast shipping"],
        risk_factors=["Supply chain issues"],
        growth_recommendations=["International Markets", "Bundle deals"],
        recommendations=[
            Recommendation(
                priority=1,
                title="Boost Ad Spend",
                description="Increase visibility through PPC campaigns on high-intent keywords.",
                impact="High",
                difficulty="medium",
                timeline="Immediate"
            )
        ],
        brand_health_score=75.0,
        confidence_score=0.9,
        executive_summary="Test Brand shows promise but faces stiff competition. They have a strong price advantage but lack overall brand awareness on the platform."
    )

@pytest.fixture
def mock_search_service():
    """Mock search service."""
    service = AsyncMock()
    service.search_brand_on_amazon = AsyncMock()
    return service

@pytest.fixture
def mock_llm_service():
    """Mock LLM service."""
    service = AsyncMock()
    service.extract_structured_data.return_value = {"mock": "data"}
    service.generate_analysis.return_value = {"mock": "analysis"}
    service.generate_content.return_value = "Mock content"
    return service

@pytest.fixture
def fixtures_dir():
    return Path(__file__).parent / "fixtures"

@pytest.fixture
def mock_search_data(fixtures_dir):
    with open(fixtures_dir / "mock_search_results.json") as f:
        return json.load(f)

@pytest.fixture
def mock_extraction_data(fixtures_dir):
    with open(fixtures_dir / "mock_extraction.json") as f:
        return json.load(f)
