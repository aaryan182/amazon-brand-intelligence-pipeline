import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from src.services.search_service import (
    SearchService, SearchResultItem, SearchResults, ProviderStatus
)
from src.models.schemas import ConfidenceLevel

def test_search_result_item():
    item = SearchResultItem(
        title="Test Product",
        url="https://amazon.com/dp/B12345",
        price=19.99,
        rating=4.5,
        review_count=100
    )
    assert item.title == "Test Product"
    assert item.price == 19.99
    assert item.is_prime is False

def test_search_result_item_defaults():
    item = SearchResultItem(
        title="Minimal Product",
        url="https://amazon.com/dp/B12345"
    )
    assert item.asin is None
    assert item.price is None
    assert item.rating is None
    assert item.review_count == 0
    assert item.is_sponsored is False
    assert item.is_amazon_choice is False
    assert item.is_best_seller is False

def test_search_results():
    results = SearchResults(
        query="test query",
        provider="serpapi",
        items=[SearchResultItem(title="P1", url="http://amazon.com/p1")],
        total_results=1
    )
    assert results.query == "test query"
    assert len(results.items) == 1

def test_search_results_defaults():
    results = SearchResults(
        query="my query",
        provider="exa"
    )
    assert results.items == []
    assert results.total_results == 0
    assert results.page == 1
    assert results.has_more is False
    assert results.cache_hit is False

def test_provider_status():
    assert ProviderStatus.AVAILABLE.value == "available"
    assert ProviderStatus.RATE_LIMITED.value == "rate_limited"
    assert ProviderStatus.ERROR.value == "error"
    assert ProviderStatus.DISABLED.value == "disabled"

def test_search_result_item_with_features():
    item = SearchResultItem(
        title="Product with Features",
        url="https://amazon.com/dp/B99999",
        features=["Feature 1", "Feature 2", "Feature 3"]
    )
    assert len(item.features) == 3
    assert "Feature 1" in item.features
