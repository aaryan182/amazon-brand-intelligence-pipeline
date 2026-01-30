import pytest
import json
from unittest.mock import MagicMock, AsyncMock
from src.analyzers.strategic_analyzer import StrategicAnalyzer
from src.models.schemas import MarketPosition, ConfidenceLevel

@pytest.fixture
def analyzer(mock_llm_service, mock_settings):
    return StrategicAnalyzer(llm_service=mock_llm_service, settings=mock_settings)

def test_calculate_brand_health(analyzer, sample_extraction_result):
    # Test with high performing brand
    positioning = {"market_position": "leader"}
    score = analyzer._calculate_brand_health(sample_extraction_result, positioning)
    assert 60 <= score <= 100
    
    # Test with lower rating
    sample_extraction_result.top_products[0].rating = 2.0
    score = analyzer._calculate_brand_health(sample_extraction_result, positioning)
    assert score < 70

def test_calculate_confidence(analyzer, sample_extraction_result):
    positioning = {"market_position": "leader"}
    conf = analyzer._calculate_confidence(sample_extraction_result, positioning)
    assert 0.7 <= conf <= 1.0
    
    # Fewer products -> lower confidence
    sample_extraction_result.top_products = sample_extraction_result.top_products[:1]
    conf = analyzer._calculate_confidence(sample_extraction_result, positioning)
    assert conf <= 0.7

def test_parse_json_response_clean(analyzer):
    response = '{"key": "value"}'
    result = analyzer._parse_json_response(response, "test")
    assert result == {"key": "value"}

def test_parse_json_response_with_markdown(analyzer):
    response = '```json\n{"key": "value"}\n```'
    result = analyzer._parse_json_response(response, "test")
    assert result == {"key": "value"}

def test_build_strategic_insight(analyzer, sample_extraction_result):
    positioning = {
        "market_position": "challenger",
        "position_rationale": "Strong growth",
        "competitive_advantages": ["Price"],
        "weaknesses": ["Awareness"]
    }
    opportunities = {"high_priority_opportunities": ["New market"]}
    risks = {"detailed_risks": ["Competition"]}
    recommendations = {
        "recommendations": [
            {"priority": 1, "title": "Test Rec", "description": "Desc", "expected_impact": {"estimate": "High"}, "effort": {"investment": "Low", "time": "1 month"}}
        ]
    }
    
    insight = analyzer._build_strategic_insight(
        sample_extraction_result,
        positioning,
        opportunities,
        risks,
        recommendations,
        "This is a long enough executive summary to pass the validation check of fifty characters."
    )
    
    assert insight.market_position == MarketPosition.CHALLENGER
    assert insight.swot.strengths == ["Price"]
    assert insight.swot.opportunities == ["New market"]
    assert len(insight.recommendations) == 1
    assert len(insight.executive_summary) >= 50
