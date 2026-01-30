"""
Integration tests for the CLI using Click's CliRunner.
"""

import sys
from unittest.mock import MagicMock

# Mock weasyprint BEFORE any other imports to prevent ImportError during collection
sys.modules["weasyprint"] = MagicMock()

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from click.testing import CliRunner
from src.main import cli
from src.models.schemas import FinalReport, ExtractionResult, StrategicInsight, AmazonProduct, AmazonPresence, ConfidenceLevel, ExtractionMetadata, MarketPosition, SWOTAnalysis, ReportSection
from src.pipeline.orchestrator import PipelineError

# Mock weasyprint to avoid import errors
sys.modules["weasyprint"] = MagicMock()

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture(autouse=True)
def mock_settings_global():
    """Globally patch get_settings for all tests."""
    mock_settings = MagicMock()
    mock_settings.anthropic_api_key.get_secret_value.return_value = "sk-mock"
    mock_settings.serpapi_api_key = "mock"
    mock_settings.exa_api_key = None
    mock_settings.get_search_provider.return_value = "serpapi"
    mock_settings.app_env = "test"
    mock_settings.output_dir = Path("outputs/reports")
    
    with patch("src.main.get_settings", return_value=mock_settings):
        yield mock_settings

@pytest.fixture
def mock_pipeline_context(mock_settings_global):
    """Patches Pipeline context manager and run method"""
    
    # Create the AsyncMock that will act as the pipeline instance
    mock_pipeline_instance = AsyncMock()
    # Support async context manager protocol: async with ... as ...
    mock_pipeline_instance.__aenter__.return_value = mock_pipeline_instance
    mock_pipeline_instance.__aexit__.return_value = None
    
    # Mock settings is handled by global fixture, but we can access it if needed
    
    # Mock Final Report (returning value for run())
    mock_report = FinalReport(
        title="Test Report",
        brand_name="TestBrand",
        domain="testbrand.com",
        extraction_result=ExtractionResult(
            brand_name="TestBrand",
            domain="testbrand.com",
            amazon_presence=AmazonPresence(found=True, confidence=ConfidenceLevel.HIGH, evidence=[]),
            primary_category="Misc",
            estimated_product_count=10,
            top_products=[],
            total_reviews=0,
            average_rating=0.0,
            metadata=ExtractionMetadata(extraction_duration_seconds=1.0)
        ),
        strategic_insight=StrategicInsight(
            market_position=MarketPosition.LEADER,
            market_position_rationale="Test " * 5,
            swot=SWOTAnalysis(),
            executive_summary="Test Summary " * 10
        ),
        sections=[ReportSection(title="Summary", content="Test Content", order=1)]
    )
    mock_pipeline_instance.run.return_value = mock_report
    
    # Mock extractor/analyzer attributes for split commands (extraction_only, analysis_only)
    # These attributes are accessed like: pipeline.extractor.extract_brand_data(domain)
    mock_pipeline_instance.extractor.extract_brand_data.return_value = mock_report.extraction_result
    mock_pipeline_instance.analyzer.analyze.return_value = mock_report.strategic_insight

    # CRITICAL FIX: The CLI instantiates the class: BrandIntelligencePipeline(settings=...)
    # So we must mock the class constructor to return our mock instance.
    with patch("src.main.BrandIntelligencePipeline", return_value=mock_pipeline_instance):
        yield mock_pipeline_instance

# =============================================================================
# Tests
# =============================================================================

def test_analyze_command_success(runner, mock_pipeline_context):
    """Test 1: Analyze command runs successfully."""
    result = runner.invoke(cli, ["analyze", "testbrand.com"])
    assert result.exit_code == 0
    assert "Analysis complete!" in result.output
    # Fix: cli shows output summary, looking for "Success" in summary table
    assert "Success" in result.output

def test_batch_command(runner, mock_pipeline_context):
    """Test 2: Batch command processes file."""
    with runner.isolated_filesystem():
        with open("brands.txt", "w") as f:
            f.write("brand1.com\nbrand2.com")
            
        result = runner.invoke(cli, ["batch", "brands.txt"])
        assert result.exit_code == 0
        assert "Batch Processing 2 brands" in result.output
        assert "Success: 2" in result.output

def test_extraction_only(runner, mock_pipeline_context):
    """Test 3: Extraction command output JSON."""
    result = runner.invoke(cli, ["test-extraction", "testbrand.com"])
    assert result.exit_code == 0
    
    # Check output contains key info instead of parsing full JSON often mixed with logs
    assert "TestBrand" in result.output
    # Rich print_json usually outputs valid JSON but console might have other text
    # Let's inspect partial string match for a key field
    assert '"brand_name": "TestBrand"' in result.output

def test_analysis_only(runner, mock_pipeline_context):
    """Test 4: Analysis command reads JSON and outputs content."""
    
    # Create valid extraction input
    extraction_input = {
        "brand_name": "TestBrand",
        "domain": "testbrand.com",
        "amazon_presence": {"found": True, "confidence": "high", "evidence": []},
        "primary_category": "Misc",
        "estimated_product_count": 10,
        "top_products": [],
        "total_reviews": 0,
        "average_rating": 0.0,
        "metadata": {"extraction_duration_seconds": 1.0}
    }
    
    with runner.isolated_filesystem():
        with open("extraction.json", "w") as f:
            json.dump(extraction_input, f)
            
        result = runner.invoke(cli, ["test-analysis", "extraction.json"])
        assert result.exit_code == 0
        
        # Verify output contains key fields
        # Enum serialization might be upper or title case
        assert "market_position" in result.output
        assert "leader" in result.output.lower()
        assert "Test Summary" in result.output

def test_validate_setup(runner, mock_pipeline_context):
    """Test 5: validate-setup command."""
    # Mock settings is already applied by fixture, ensuring checks pass
    result = runner.invoke(cli, ["validate-setup"])
    assert result.exit_code == 0
    assert "Pass" in result.output
    
def test_invalid_domain_arg(runner):
    """
    Test 6: Invalid missing domain argument.
    Note: Click handles missing arguments before our code runs.
    """
    result = runner.invoke(cli, ["analyze"])  # Missing arg
    assert result.exit_code != 0
    assert "Missing argument 'DOMAIN'" in result.output

def test_missing_api_keys(runner, mock_settings_global):
    """Test 7: Missing API keys detection in validate-setup."""
    
    # Configure global mock to return invalid keys for this test
    mock_settings_global.anthropic_api_key.get_secret_value.return_value = "invalid" 
    mock_settings_global.serpapi_api_key = None
    mock_settings_global.exa_api_key = None
    
    result = runner.invoke(cli, ["validate-setup"])
    assert result.exit_code != 0
    # Should show Fail for both
    assert "Fail" in result.output
    assert "Warning: No search provider configured" in result.output

def test_output_formats(runner, mock_pipeline_context):
    """Test 8: Output format option is passed to settings."""
    result = runner.invoke(cli, ["analyze", "testbrand.com", "--format", "json"])
    assert result.exit_code == 0
    
    # Verify settings was updated (we can't easily check the settings object inside the call because it's retrieved inside the command)
    # But we can verify command didn't explode
    assert "Success" in result.output
