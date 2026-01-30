import pytest
from unittest.mock import patch, MagicMock
from src.utils.formatters import format_extraction_table, format_top_products_table, format_recommendations_table
from src.models.schemas import ExtractionResult, AmazonPresence, ConfidenceLevel, AmazonProduct, Recommendation

def test_format_extraction_table_no_rating(sample_extraction_result):
    sample_extraction_result.average_rating = 0.0
    result = format_extraction_table(sample_extraction_result)
    assert "N/A" in result

def test_format_top_products_table_empty():
    result = format_top_products_table([])
    assert "*No products found.*" in result

def test_format_recommendations_table_empty():
    result = format_recommendations_table([])
    assert "*No recommendations available.*" in result

def test_format_recommendations_table_with_data():
    recs = [
        Recommendation(
            priority=1,
            title="Test Recommendation",
            description="This is a detailed test recommendation description.",
            rationale="Because it works",
            impact="High revenue growth",
            difficulty="Medium",
            timeline="Q1 2026"
        )
    ]
    result = format_recommendations_table(recs)
    assert "Test Recommendation" in result
    assert "1" in result
