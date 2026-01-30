import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from src.analyzers.strategic_analyzer import StrategicAnalyzer
from src.models.schemas import ExtractionResult, AmazonPresence, ConfidenceLevel, AmazonProduct

@pytest.fixture
def analyzer(mock_settings):
    mock_llm = MagicMock()
    # Mock _call_api to return valid data
    mock_llm._call_api = AsyncMock(return_value=(
        "This is a response that is more than fifty characters long to pass schema validation.",
        MagicMock(total_tokens=100)
    ))
    return StrategicAnalyzer(settings=mock_settings, llm_service=mock_llm)

@pytest.mark.asyncio
async def test_ensure_minimum_recommendations(analyzer, sample_extraction_result):
    recs = {"recommendations": [{"title": "Exisiting"}]}
    result = analyzer._ensure_minimum_recommendations(recs, sample_extraction_result)
    assert len(result["recommendations"]) >= 3
    assert result["recommendations"][0]["title"] == "Exisiting"

@pytest.mark.asyncio
async def test_generate_executive_summary_complex_structures(analyzer, sample_extraction_result):
    positioning = {
        "competitive_advantages": [{"advantage": "Adv 1"}],
        "market_position": {"classification": "leader"}
    }
    opportunities = {
        "high_priority_opportunities": [{"opportunity": "Opp 1"}],
        "underserved_categories": [],
        "product_expansion": []
    }
    risks = {
        "detailed_risks": [{"severity": "high", "risk_name": "Risk 1"}]
    }
    recommendations = {
        "recommendations": [{"title": "Rec 1"}]
    }
    
    # We don't need to patch _call_api here as it's already an AsyncMock on analyzer.llm_service
    summary = await analyzer._generate_executive_summary(
        sample_extraction_result,
        positioning,
        opportunities,
        risks,
        recommendations
    )
    assert "This is a response" in summary

@pytest.mark.asyncio
async def test_generate_fallback_summary(analyzer, sample_extraction_result):
    positioning = {"market_position": "pioneer"}
    summary = analyzer._generate_fallback_summary(sample_extraction_result, positioning)
    assert sample_extraction_result.brand_name in summary
    assert "pioneer" in summary

@pytest.mark.asyncio
async def test_strategic_analyzer_analyze_error_handling(analyzer, sample_extraction_result):
    # Test top-level analyze error handling
    with patch.object(analyzer, "_analyze_competitive_positioning", side_effect=Exception("Failed")):
        with pytest.raises(Exception) as exc:
            await analyzer.analyze(sample_extraction_result)
        assert "Failed" in str(exc.value)
