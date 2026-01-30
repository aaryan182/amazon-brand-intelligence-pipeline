import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from src.analyzers.strategic_analyzer import StrategicAnalyzer
from src.models.schemas import ExtractionResult, StrategicInsight, MarketPosition

@pytest.fixture
def analyzer(mock_llm_service, mock_settings):
    return StrategicAnalyzer(llm_service=mock_llm_service, settings=mock_settings)

@pytest.mark.asyncio
async def test_analyze_full_flow(analyzer, sample_extraction_result):
    # Mock LLM internal calls
    # StrategicAnalyzer.analyze calls:
    # 1. _analyze_competitive_positioning
    # 2. _analyze_market_opportunities
    # 3. _analyze_risks
    # 4. _generate_recommendations
    # 5. _generate_executive_summary
    
    mock_responses = {
        "competitive_positioning": '{"market_position": "leader", "position_rationale": "Strong presence", "competitive_advantages": ["Scale"], "weaknesses": ["Agility"]}',
        "market_opportunities": '{"high_priority_opportunities": ["Expanding to new niches"]}',
        "risk_assessment": '{"detailed_risks": ["Supply chain"]}',
        "recommendations": '{"recommendations": [{"priority": 1, "title": "Buy more ads", "description": "Ads help", "expected_impact": {"estimate": "High"}, "effort": {"investment": "Medium", "time": "1 month"}}]}',
        "executive_summary": "This is a very long executive summary that covers all the important points about this brand's performance on Amazon and should be at least fifty characters long."
    }
    
    # Mock _call_llm instead of individual methods to cover more of analyze()
    async def mock_call_llm(prompt, analysis_type):
        return mock_responses[analysis_type]
    
    with patch.object(analyzer, "_call_llm", side_effect=mock_call_llm):
        result = await analyzer.analyze(sample_extraction_result)
        
        assert isinstance(result, StrategicInsight)
        assert result.market_position == MarketPosition.LEADER
        # Analyzer adds defaults if < 3 recs provided
        assert len(result.recommendations) >= 3
        assert len(result.executive_summary) >= 50

@pytest.mark.asyncio
async def test_analyze_no_presence(analyzer, sample_extraction_result):
    # Set found=False
    sample_extraction_result.amazon_presence.found = False
    
    mock_response = '{"market_position": "emerging", "position_rationale": "No presence yet", "market_entry_strategy": "Start with PPC", "swot": {"strengths": [], "weaknesses": [], "opportunities": [], "threats": []}, "recommendations": []}'
    
    with patch.object(analyzer, "_call_llm", return_value=mock_response):
        # We need to mock _generate_market_entry_strategy or the LLM call inside it
        # _generate_market_entry_strategy is called by analyze() if found is False
        
        # Adjust mock_responses for no-presence flow
        async def mock_call_llm_no_presence(prompt, analysis_type):
            if analysis_type == "market_opportunities": # Actually it might be another type
                 return mock_response
            return mock_response
            
        result = await analyzer.analyze(sample_extraction_result)
        assert result.market_position == MarketPosition.EMERGING
