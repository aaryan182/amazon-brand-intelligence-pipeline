import pytest
from unittest.mock import AsyncMock, patch
from src.pipeline.orchestrator import BrandIntelligencePipeline
from src.models.schemas import FinalReport

@pytest.mark.asyncio
async def test_pipeline_full_run(mock_settings, sample_extraction_result, sample_strategic_insight):
    # Mock services
    mock_extractor = AsyncMock()
    mock_extractor.extract_brand_data.return_value = sample_extraction_result
    
    mock_analyzer = AsyncMock()
    mock_analyzer.analyze.return_value = sample_strategic_insight
    
    pipeline = BrandIntelligencePipeline(
        settings=mock_settings,
        extractor=mock_extractor,
        analyzer=mock_analyzer
    )
    
    # We need to mock the report formatter as well if we don't want real file IO
    from src.utils.formatters import ReportFormatter
    mock_formatter = AsyncMock(spec=ReportFormatter)
    mock_formatter.generate_report.return_value = MagicMock() # returns a Path
    
    with patch("src.pipeline.orchestrator.ReportFormatter", return_value=mock_formatter):
        result = await pipeline.run("patagonia.com")
        
        assert result.brand_name is not None
        assert result.strategic_insight is not None
        mock_extractor.extract_brand_data.assert_called_once()
        mock_analyzer.analyze.assert_called_once()

from unittest.mock import MagicMock
