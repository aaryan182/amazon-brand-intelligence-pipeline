"""Additional schema tests for improved coverage."""
import pytest
from pydantic import ValidationError
from src.models.schemas import (
    BrandInput, AmazonProduct, ExtractionResult, AmazonPresence, 
    ConfidenceLevel, SWOTAnalysis, Recommendation, StrategicInsight, FinalReport
)
from datetime import datetime

def test_brand_input_with_all_fields():
    brand = BrandInput(
        domain="example.com",
        brand_name="Example Brand",
        industry="Technology",
        target_markets=["USA", "Europe"],
        additional_context="Premium brand"
    )
    assert brand.domain == "example.com"
    assert brand.brand_name == "Example Brand"

def test_amazon_presence_high_confidence():
    presence = AmazonPresence(
        found=True,
        confidence=ConfidenceLevel.HIGH,
        product_count=100,
        verification_sources=["Amazon Search", "Product Listings"]
    )
    assert presence.found is True
    assert presence.confidence == ConfidenceLevel.HIGH

def test_amazon_presence_not_found():
    presence = AmazonPresence(
        found=False,
        confidence=ConfidenceLevel.LOW,
        product_count=0
    )
    assert presence.found is False

def test_amazon_product_full():
    # Use a valid 10-character ASIN
    product = AmazonProduct(
        asin="B0ABCD1234",
        title="Test Product Title",
        url="https://amazon.com/dp/B0ABCD1234",
        price=29.99,
        original_price=49.99,
        currency="USD",
        rating=4.5,
        review_count=1000,
        is_amazon_choice=True,
        is_prime=True
    )
    assert product.asin == "B0ABCD1234"
    assert product.is_amazon_choice is True

def test_swot_analysis():
    swot = SWOTAnalysis(
        strengths=["Strong brand"],
        weaknesses=["Limited distribution"],
        opportunities=["New markets"],
        threats=["Competition"]
    )
    assert len(swot.strengths) == 1

def test_recommendation_full():
    rec = Recommendation(
        priority=1,
        title="Expand product line",
        description="Add new products to the existing lineup",
        rationale="Market demand analysis",
        impact="High",
        difficulty="Medium",
        timeline="Q2 2026"
    )
    assert rec.priority == 1
    assert rec.impact == "High"

def test_confidence_level_values():
    assert ConfidenceLevel.HIGH.value == "high"
    assert ConfidenceLevel.MEDIUM.value == "medium"
    assert ConfidenceLevel.LOW.value == "low"

def test_brand_input_auto_brand_name():
    """Brand name is derived from domain if not provided."""
    brand = BrandInput(domain="example.com")
    assert brand.domain == "example.com"
    # Brand name is auto-derived from domain
    assert brand.brand_name == "Example"
