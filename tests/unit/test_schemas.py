import pytest
from src.models.schemas import (
    BrandInput,
    AmazonProduct,
    ExtractionResult,
    StrategicInsight,
    FinalReport,
    AmazonCategory,
    MarketPosition
)
from pydantic import ValidationError

def test_brand_input_validation():
    # Valid cases
    assert BrandInput(domain="patagonia.com").domain == "patagonia.com"
    assert BrandInput(domain="example.co.uk").domain == "example.co.uk"
    
    # Protocol stripping
    assert BrandInput(domain="https://patagonia.com").domain == "patagonia.com"
    assert BrandInput(domain="www.patagonia.com").domain == "patagonia.com"
    
    # Invalid cases
    with pytest.raises(ValidationError):
        # validate_domain raises ValueError which Pydantic wraps in ValidationError
        BrandInput(domain="not-a-domain")
    
    # Empty domain is too short (min_length=4)
    with pytest.raises(ValidationError):
        BrandInput(domain="")

def test_amazon_product_schema():
    product = AmazonProduct(
        title="Test Product",
        asin="B001234567",
        price=10.0,
        rating=4.5,
        review_count=100,
        url="https://amazon.com/dp/B001234567"
    )
    assert product.asin == "B001234567"
    assert product.price == 10.0
    
    # Invalid ASIN (must be 10 chars)
    with pytest.raises(ValidationError):
        AmazonProduct(
            title="T", 
            asin="INVALID", 
            url="https://amazon.com/dp/INVALID", 
            price=1.0,
            review_count=1
        )

def test_extraction_result_integrity(sample_extraction_result):
    json_output = sample_extraction_result.model_dump_json()
    assert "Test Brand" in json_output
    assert "metadata" in json_output
    
    reconstituted = ExtractionResult.model_validate_json(json_output)
    assert reconstituted.brand_name == sample_extraction_result.brand_name
    assert reconstituted.amazon_presence.found is True

def test_strategic_insight_requirements(sample_strategic_insight):
    # Ensure all required fields for logic are present
    assert sample_strategic_insight.market_position == MarketPosition.CHALLENGER
    assert len(sample_strategic_insight.recommendations) > 0
    assert len(sample_strategic_insight.executive_summary) >= 50
