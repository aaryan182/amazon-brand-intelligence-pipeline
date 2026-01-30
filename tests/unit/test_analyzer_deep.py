import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from src.analyzers.strategic_analyzer import StrategicAnalyzer
from src.models.schemas import ExtractionResult, StrategicInsight

@pytest.fixture
def analyzer(mock_llm_service, mock_settings):
    return StrategicAnalyzer(llm_service=mock_llm_service, settings=mock_settings)

@pytest.mark.asyncio
async def test_analyze_individual_methods(analyzer, sample_extraction_result):
    # Test _analyze_competitive_positioning
    with patch.object(analyzer, "_call_llm", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = json.dumps({"market_position": {"classification": "leader", "rationale": "test"}})
        res = await analyzer._analyze_competitive_positioning(sample_extraction_result)
        assert res["market_position"]["classification"] == "leader"

    # Test _analyze_market_opportunities
    with patch.object(analyzer, "_call_llm", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = json.dumps({"high_priority_opportunities": ["opp1"]})
        res = await analyzer._analyze_market_opportunities(sample_extraction_result, "summary")
        assert "opp1" in res["high_priority_opportunities"]

    # Test _analyze_risks
    with patch.object(analyzer, "_call_llm", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = json.dumps({"detailed_risks": ["risk1"]})
        res = await analyzer._analyze_risks(sample_extraction_result)
        assert "risk1" in res["detailed_risks"]

    # Test _generate_recommendations
    with patch.object(analyzer, "_call_llm", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = json.dumps({"recommendations": []})
        res = await analyzer._generate_recommendations(sample_extraction_result, {}, {})
        assert "recommendations" in res

@pytest.mark.asyncio
async def test_error_handling_in_analyze(analyzer, sample_extraction_result):
    # Test LLM failure
    with patch.object(analyzer.llm_service, "generate_analysis", side_effect=Exception("LLM Fail")):
        with pytest.raises(Exception):
            await analyzer.analyze(sample_extraction_result)
