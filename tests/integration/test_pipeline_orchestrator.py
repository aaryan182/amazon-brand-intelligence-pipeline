"""
Integration tests for the LangGraph Pipeline Orchestrator.

Tests the complete pipeline workflow including:
- State management
- Node execution
- Conditional routing
- Error handling
- Retry logic
- Progress tracking
"""

import os
from pathlib import Path
import pytest
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

# Patch environment variables BEFORE importing application modules
# to prevent Settings validation errors during import or initialization
os.environ["ANTHROPIC_API_KEY"] = "sk-mock-key-for-testing-12345"

from src.pipeline.orchestrator import (
    BrandIntelligencePipeline,
    PipelineStateDict,
    PipelineError,
    ValidationError,
    ExtractionError,
    AnalysisError,
    ValidationDecision,
    ProgressTracker,
    InMemoryStatePersistence,
    analyze_brand,
)
from src.models.schemas import (
    AmazonPresence,
    AmazonProduct,
    ConfidenceLevel,
    ExtractionMetadata,
    ExtractionResult,
    MarketPosition,
    StrategicInsight,
    SWOTAnalysis,
    FinalReport,
    BrandInput,
    CompetitorInsight,
    AnalysisStatus,
    PipelineStep,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_settings():
    """Create mock settings."""
    mock = MagicMock()
    mock.report_format = "json"
    mock.output_dir = Path("outputs/test")
    mock.max_retries = 3
    mock.request_timeout = 30
    mock.max_concurrent_requests = 5
    mock.claude_max_tokens = 4096
    mock.claude_temperature = 0.7
    mock.claude_model = "claude-test"
    mock.app_name = "Test Pipeline"
    # Mock validator methods if called directly (though class methods usually)
    return mock


@pytest.fixture
def mock_extraction_result():
    """Create a mock ExtractionResult."""
    return ExtractionResult(
        brand_name="TestBrand",
        domain="testbrand.com",
        amazon_presence=AmazonPresence(
            found=True,
            confidence=ConfidenceLevel.HIGH,
            evidence=["Official store found"],
        ),
        primary_category="Electronics",
        estimated_product_count=10,
        top_products=[
            AmazonProduct(
                asin=f"B0TEST{i:04d}",
                title=f"Test Product {i}",
                price=99.99 + i * 10,
                rating=4.5,
                review_count=100 * (i + 1),
            )
            for i in range(10)
        ],
        all_categories=["Electronics", "Computers"],
        competitors_found=["Competitor1", "Competitor2"],
        metadata=ExtractionMetadata(
            extraction_duration_seconds=5.0,
            api_calls_made=10,
            data_source="serpapi",
        ),
    )


@pytest.fixture
def mock_strategic_insight():
    """Create a mock StrategicInsight."""
    return StrategicInsight(
        market_position=MarketPosition.CHALLENGER,
        market_position_rationale="TestBrand has a strong challenger position with quality products and growing market share. The brand shows potential for becoming a market leader.",
        swot=SWOTAnalysis(
            strengths=["Strong product quality", "Competitive pricing"],
            weaknesses=["Limited brand awareness"],
            opportunities=["Market expansion", "New product categories"],
            threats=["Strong competition"],
        ),
        competitive_advantages=["Quality products", "Good customer service"],
        risk_factors=["Market volatility", "Competition"],
        growth_recommendations=[
            "Expand product line",
            "Increase marketing spend",
            "Improve customer engagement",
        ],
        competitor_insights=[
            CompetitorInsight(
                name="Competitor A",
                price_positioning="higher",
                key_differentiators=["Premium branding"],
            )
        ],
        brand_health_score=75.0,
        confidence_score=0.85,
        executive_summary="TestBrand shows strong potential for growth with quality products and competitive positioning. The brand should focus on expanding market awareness and product line diversification.",
    )


@pytest.fixture
def mock_extractor(mock_extraction_result):
    """Create a mock BrandExtractor."""
    mock = MagicMock()
    mock.extract_brand_data = AsyncMock(return_value=mock_extraction_result)
    return mock


@pytest.fixture
def mock_analyzer(mock_strategic_insight):
    """Create a mock StrategicAnalyzer."""
    mock = MagicMock()
    mock.analyze = AsyncMock(return_value=mock_strategic_insight)
    return mock


@pytest.fixture
def pipeline(mock_settings, mock_extractor, mock_analyzer):
    """Create a pipeline with mocked dependencies."""
    pipeline = BrandIntelligencePipeline(
        settings=mock_settings,
        extractor=mock_extractor,
        analyzer=mock_analyzer,
        enable_retries=False,
    )
    # Pre-set the initialized services
    pipeline._extractor = mock_extractor
    pipeline._analyzer = mock_analyzer
    return pipeline


@pytest.fixture
def initial_state():
    """Create initial pipeline state."""
    return PipelineStateDict(
        run_id=str(uuid4()),
        domain="testbrand.com",
        brand_name=None,  # Set to None to trigger auto-extraction
        brand_input={},
        extraction_result=None,
        strategic_insight=None,
        final_report=None,
        current_step=PipelineStep.INITIALIZATION.value,
        status=AnalysisStatus.PENDING.value,
        validation_decision=ValidationDecision.PROCEED.value,
        errors=[],
        metadata={},
        step_timings={},
        retry_counts={},
        progress_percent=0,
        started_at=datetime.utcnow().isoformat(),
        completed_at=None,
    )


# =============================================================================
# Test Classes
# =============================================================================

class TestPipelineInitialization:
    """Tests for pipeline initialization."""
    
    def test_pipeline_creates_with_defaults(self, mock_settings):
        """Test pipeline creates with default settings."""
        pipeline = BrandIntelligencePipeline(settings=mock_settings)
        assert pipeline.settings == mock_settings
        assert pipeline.enable_retries is True
        assert pipeline.max_retries == 3
    
    def test_pipeline_accepts_custom_persistence(self, mock_settings):
        """Test pipeline accepts custom state persistence."""
        persistence = InMemoryStatePersistence()
        pipeline = BrandIntelligencePipeline(
            settings=mock_settings,
            persistence=persistence,
        )
        assert pipeline.persistence is persistence
    
    def test_pipeline_has_graph(self, pipeline):
        """Test pipeline has compiled graph."""
        assert pipeline._graph is not None


class TestValidateInputNode:
    """Tests for the validate_input node."""
    
    @pytest.mark.asyncio
    async def test_validate_valid_domain(self, pipeline, initial_state):
        """Test validation with valid domain."""
        result = await pipeline.run_step("validate_input", initial_state)
        
        assert result["brand_input"] is not None
        assert result["brand_name"] != ""
        assert result["status"] == AnalysisStatus.IN_PROGRESS.value
    
    @pytest.mark.asyncio
    async def test_validate_extracts_brand_name(self, pipeline, initial_state):
        """Test brand name extraction from domain."""
        result = await pipeline.run_step("validate_input", initial_state)
        
        # Brand name should be extracted from domain
        assert "testbrand" in result["brand_name"].lower()
    
    @pytest.mark.asyncio
    async def test_validate_invalid_domain_sets_error(self, pipeline, initial_state):
        """Test validation with invalid domain."""
        initial_state["domain"] = "invalid"
        
        result = await pipeline.run_step("validate_input", initial_state)
        
        assert len(result.get("errors", [])) > 0


class TestExtractDataNode:
    """Tests for the extract_data node."""
    
    @pytest.mark.asyncio
    async def test_extract_calls_extractor(self, pipeline, initial_state, mock_extractor):
        """Test extraction calls the extractor service."""
        # First validate
        state = await pipeline.run_step("validate_input", initial_state)
        
        # Then extract
        result = await pipeline.run_step("extract_data", state)
        
        mock_extractor.extract_brand_data.assert_called_once()
        assert result["extraction_result"] is not None
    
    @pytest.mark.asyncio
    async def test_extract_updates_progress(self, pipeline, initial_state):
        """Test extraction updates progress percentage."""
        state = await pipeline.run_step("validate_input", initial_state)
        result = await pipeline.run_step("extract_data", state)
        
        # Progress should be > 40% after extraction
        assert result["progress_percent"] >= 40
    
    @pytest.mark.asyncio
    async def test_extract_handles_failure(self, pipeline, initial_state, mock_extractor):
        """Test extraction handles failures gracefully."""
        mock_extractor.extract_brand_data.side_effect = Exception("API Error")
        
        state = await pipeline.run_step("validate_input", initial_state)
        result = await pipeline.run_step("extract_data", state)
        
        assert len(result.get("errors", [])) > 0


class TestValidateExtractionNode:
    """Tests for the validate_extraction node."""
    
    @pytest.mark.asyncio
    async def test_validation_proceeds_with_valid_data(
        self, pipeline, initial_state, mock_extraction_result
    ):
        """Test validation proceeds with valid extraction data."""
        state = await pipeline.run_step("validate_input", initial_state)
        state["extraction_result"] = mock_extraction_result.model_dump()
        
        result = await pipeline.run_step("validate_extraction", state)
        
        assert result["validation_decision"] == ValidationDecision.PROCEED.value
    
    @pytest.mark.asyncio
    async def test_validation_handles_no_amazon_presence(self, pipeline, initial_state):
        """Test validation handles no Amazon presence."""
        no_presence = ExtractionResult(
            brand_name="TestBrand",
            domain="testbrand.com",
            amazon_presence=AmazonPresence(
                found=False,
                confidence=ConfidenceLevel.HIGH,
            ),
            top_products=[],
            metadata=ExtractionMetadata(extraction_duration_seconds=1.0),
        )
        
        state = await pipeline.run_step("validate_input", initial_state)
        state["extraction_result"] = no_presence.model_dump()
        
        result = await pipeline.run_step("validate_extraction", state)
        
        # Should still proceed (analyzer handles no-presence case)
        assert result["validation_decision"] == ValidationDecision.PROCEED.value
    
    @pytest.mark.asyncio
    async def test_validation_errors_on_missing_data(self, pipeline, initial_state):
        """Test validation errors when extraction data is missing."""
        state = await pipeline.run_step("validate_input", initial_state)
        state["extraction_result"] = None
        
        result = await pipeline.run_step("validate_extraction", state)
        
        assert result["validation_decision"] == ValidationDecision.ERROR.value


class TestAnalyzeDataNode:
    """Tests for the analyze_data node."""
    
    @pytest.mark.asyncio
    async def test_analyze_calls_analyzer(
        self, pipeline, initial_state, mock_extraction_result, mock_analyzer
    ):
        """Test analysis calls the analyzer service."""
        state = await pipeline.run_step("validate_input", initial_state)
        state["extraction_result"] = mock_extraction_result.model_dump()
        
        result = await pipeline.run_step("analyze_data", state)
        
        mock_analyzer.analyze.assert_called_once()
        assert result["strategic_insight"] is not None
    
    @pytest.mark.asyncio
    async def test_analyze_updates_progress(
        self, pipeline, initial_state, mock_extraction_result
    ):
        """Test analysis updates progress percentage."""
        state = await pipeline.run_step("validate_input", initial_state)
        state["extraction_result"] = mock_extraction_result.model_dump()
        
        result = await pipeline.run_step("analyze_data", state)
        
        # Progress should be > 80% after analysis
        assert result["progress_percent"] >= 80


class TestGenerateReportNode:
    """Tests for the generate_report node."""
    
    @pytest.mark.asyncio
    async def test_report_generation(
        self, pipeline, initial_state, mock_extraction_result, mock_strategic_insight
    ):
        """Test report generation creates FinalReport."""
        state = await pipeline.run_step("validate_input", initial_state)
        state["extraction_result"] = mock_extraction_result.model_dump()
        state["strategic_insight"] = mock_strategic_insight.model_dump()
        
        result = await pipeline.run_step("generate_report", state)
        
        assert result["final_report"] is not None
        assert result["status"] == AnalysisStatus.COMPLETED.value
        assert result["progress_percent"] == 100
    
    @pytest.mark.asyncio
    async def test_report_has_sections(
        self, pipeline, initial_state, mock_extraction_result, mock_strategic_insight
    ):
        """Test generated report has expected sections."""
        state = await pipeline.run_step("validate_input", initial_state)
        state["extraction_result"] = mock_extraction_result.model_dump()
        state["strategic_insight"] = mock_strategic_insight.model_dump()
        
        result = await pipeline.run_step("generate_report", state)
        
        report = FinalReport(**result["final_report"])
        assert len(report.sections) > 0
        
        section_titles = [s.title for s in report.sections]
        assert "Executive Summary" in section_titles
        assert "Market Position" in section_titles


class TestHandleErrorNode:
    """Tests for the handle_error node."""
    
    @pytest.mark.asyncio
    async def test_error_aggregation(self, pipeline, initial_state):
        """Test error node aggregates errors."""
        initial_state["errors"] = ["Error 1", "Error 2"]
        
        result = await pipeline.run_step("handle_error", initial_state)
        
        assert result["status"] == AnalysisStatus.FAILED.value
        assert "error_summary" in result["metadata"]
    
    @pytest.mark.asyncio
    async def test_error_provides_recovery_suggestions(self, pipeline, initial_state):
        """Test error node provides recovery suggestions."""
        initial_state["errors"] = ["Extraction failed: API Error"]
        
        result = await pipeline.run_step("handle_error", initial_state)
        
        error_summary = result["metadata"]["error_summary"]
        assert len(error_summary["recovery_suggestions"]) > 0


class TestFullPipelineExecution:
    """Tests for full pipeline execution."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_success(self, pipeline):
        """Test complete pipeline execution."""
        report = await pipeline.run("testbrand.com")
        
        assert isinstance(report, FinalReport)
        assert report.brand_name == "TestBrand"
        assert report.strategic_insight is not None
    
    @pytest.mark.asyncio
    async def test_pipeline_with_custom_run_id(self, pipeline):
        """Test pipeline with custom run ID."""
        custom_id = str(uuid4())
        report = await pipeline.run("testbrand.com", run_id=custom_id)
        
        assert str(report.pipeline_run_id) == custom_id
    
    @pytest.mark.asyncio
    async def test_pipeline_saves_state(self, mock_settings, mock_extractor, mock_analyzer):
        """Test pipeline saves state to persistence."""
        persistence = InMemoryStatePersistence()
        pipeline = BrandIntelligencePipeline(
            settings=mock_settings,
            extractor=mock_extractor,
            analyzer=mock_analyzer,
            persistence=persistence,
            enable_retries=False,
        )
        pipeline._extractor = mock_extractor
        pipeline._analyzer = mock_analyzer
        
        run_id = str(uuid4())
        await pipeline.run("testbrand.com", run_id=run_id)
        
        saved_state = await persistence.load_state(run_id)
        assert saved_state is not None
        assert saved_state["status"] == AnalysisStatus.COMPLETED.value


class TestProgressTracking:
    """Tests for progress tracking."""
    
    def test_progress_tracker_tracks_steps(self):
        """Test progress tracker tracks completed steps."""
        tracker = ProgressTracker()
        
        progress = tracker.mark_complete("validate_input")
        assert progress == 5
        
        progress = tracker.mark_complete("extract_data")
        assert progress == 45
    
    def test_progress_tracker_callback(self):
        """Test progress tracker calls callback."""
        callback = MagicMock()
        tracker = ProgressTracker(callback=callback)
        
        tracker.mark_complete("validate_input")
        
        callback.assert_called_once()


class TestStatePersistence:
    """Tests for state persistence."""
    
    @pytest.mark.asyncio
    async def test_in_memory_persistence_save_and_load(self):
        """Test in-memory persistence save and load."""
        persistence = InMemoryStatePersistence()
        
        state = {"run_id": "test-123", "status": "completed"}
        await persistence.save_state("test-123", state)
        
        loaded = await persistence.load_state("test-123")
        assert loaded == state
    
    @pytest.mark.asyncio
    async def test_in_memory_persistence_delete(self):
        """Test in-memory persistence delete."""
        persistence = InMemoryStatePersistence()
        
        await persistence.save_state("test-123", {"status": "completed"})
        await persistence.delete_state("test-123")
        
        loaded = await persistence.load_state("test-123")
        assert loaded is None


class TestMockingSupport:
    """Tests for testing hooks and mocking."""
    
    @pytest.mark.asyncio
    async def test_mock_node_function(self, pipeline, initial_state, mock_extraction_result):
        """Test mocking a node function."""
        custom_result = mock_extraction_result.model_dump()
        custom_result["brand_name"] = "MockedBrand"
        
        async def mock_extract(state):
            return ExtractionResult(**custom_result)
        
        pipeline.mock_node("extract_data", mock_extract)
        
        state = await pipeline.run_step("validate_input", initial_state)
        result = await pipeline.run_step("extract_data", state)
        
        assert result["extraction_result"]["brand_name"] == "MockedBrand"
    
    def test_clear_mocks(self, pipeline):
        """Test clearing mocks."""
        pipeline.mock_node("extract_data", lambda x: x)
        pipeline.clear_mocks()
        
        assert len(pipeline._mock_nodes) == 0


class TestErrorHandling:
    """Tests for error handling throughout the pipeline."""
    
    @pytest.mark.asyncio
    async def test_pipeline_raises_on_validation_failure(
        self, mock_settings, mock_extractor, mock_analyzer
    ):
        """Test pipeline raises PipelineError on validation failure."""
        mock_extractor.extract_brand_data = AsyncMock(
            side_effect=Exception("Extraction failed")
        )
        
        pipeline = BrandIntelligencePipeline(
            settings=mock_settings,
            extractor=mock_extractor,
            analyzer=mock_analyzer,
            enable_retries=False,
        )
        pipeline._extractor = mock_extractor
        pipeline._analyzer = mock_analyzer
        
        with pytest.raises(PipelineError) as exc_info:
            await pipeline.run("testbrand.com")
        
        assert "failed" in str(exc_info.value).lower()
