import pytest
from unittest.mock import AsyncMock, patch
from src.analyzers.strategic_analyzer import StrategicAnalyzer
from src.models.schemas import StrategicInsight, MarketPosition

@pytest.mark.asyncio
async def test_strategic_analyzer_flow(mock_llm_service, mock_settings, sample_extraction_result, sample_strategic_insight):
    analyzer = StrategicAnalyzer(
        llm_service=mock_llm_service,
        settings=mock_settings
    )
    
    # Mock LLM calls for different stages
    # StrategicAnalyzer.analyze calls multiple internal methods that call _call_llm
    # We can mock _call_llm to return appropriate JSON strings
    
    import json
    with patch.object(analyzer, "_call_llm", new_callable=AsyncMock) as mock_llm_call:
        # 1. Competitive Positioning
        # 2. Market Opportunities
        # 3. Risks
        # 4. Recommendations
        # 5. Executive Summary
        
        mock_llm_call.side_effect = [
            json.dumps({"market_position": "challenger", "rationale": "Strong growth"}),
            json.dumps({"opportunities": ["New markets"]}),
            json.dumps({"risks": ["Price wars"]}),
            json.dumps({"recommendations": [{"priority":1, "title":"Test", "description":"Test", "impact":"High", "difficulty":"low", "timeline":"Short"}]}),
            "This is a comprehensive executive summary of the brand's performance and potential."
        ]
        
        insight = await analyzer.analyze(sample_extraction_result)
        
        assert isinstance(insight, StrategicInsight)
        assert insight.market_position == MarketPosition.CHALLENGER
        assert len(insight.executive_summary) >= 50
        assert mock_llm_call.call_count == 5


