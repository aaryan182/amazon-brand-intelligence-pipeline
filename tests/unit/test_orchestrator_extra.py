import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.pipeline.orchestrator import BrandIntelligencePipeline, PipelineStateDict, InMemoryStatePersistence
from src.models.schemas import ExtractionResult, AmazonPresence, ConfidenceLevel

@pytest.fixture
def mock_brand_extractor(): return MagicMock()
@pytest.fixture
def mock_strategic_analyzer(): return MagicMock()

@pytest.fixture
def pipeline(mock_settings, mock_brand_extractor, mock_strategic_analyzer):
    return BrandIntelligencePipeline(
        settings=mock_settings,
        extractor=mock_brand_extractor,
        analyzer=mock_strategic_analyzer
    )

@pytest.mark.asyncio
async def test_pipeline_nodes_error_cases(pipeline):
    state: PipelineStateDict = {
        "run_id": "test",
        "domain": "test.com",
        "brand_name": "test",
        "errors": ["Error 1"],
        "metadata": {},
        "step_timings": {},
        "retry_counts": {},
        "progress_percent": 0
    }
    result = await pipeline._handle_error_node(state)
    assert result["status"] == "failed"
    assert "error_summary" in result["metadata"]
    assert "recovery_suggestions" in result["metadata"]["error_summary"]

@pytest.mark.asyncio
async def test_pipeline_progress_callback(pipeline):
    callback = MagicMock()
    pipeline.progress_callback = callback
    
    state: PipelineStateDict = {
        "run_id": "test",
        "domain": "test.com",
        "brand_name": "test",
        "errors": [],
        "metadata": {},
        "step_timings": {},
        "retry_counts": {},
        "progress_percent": 0
    }
    
    # Use a real model or a very robust mock
    mock_res = MagicMock(spec=ExtractionResult)
    mock_res.brand_name = "Mock Brand"
    mock_res.top_products = []
    mock_res.amazon_presence = MagicMock(spec=AmazonPresence)
    mock_res.amazon_presence.confidence = ConfidenceLevel.HIGH
    mock_res.amazon_presence.found = True
    
    with patch.object(pipeline.extractor, "extract_brand_data", new_callable=AsyncMock) as mock_ext:
        mock_ext.return_value = mock_res
        await pipeline._extract_data_node(state)
        assert callback.called

@pytest.mark.asyncio
async def test_in_memory_persistence():
    persist = InMemoryStatePersistence()
    state: PipelineStateDict = {"domain": "test.com", "run_id": "test"} 
    run_id = "test_run"
    
    await persist.save_state(run_id, state)
    loaded = await persist.load_state(run_id)
    assert loaded["domain"] == "test.com"
    
    await persist.delete_state(run_id)
    assert await persist.load_state(run_id) is None
