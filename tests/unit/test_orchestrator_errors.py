import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from src.pipeline.orchestrator import (
    ValidationError, ExtractionError, AnalysisError, TimeoutError as PipelineTimeoutError,
    with_timeout, with_retry
)

def test_validation_error():
    err = ValidationError("Invalid input", {"field": "domain"})
    assert "Invalid input" in str(err)
    assert err.recoverable is False

def test_extraction_error():
    err = ExtractionError("Extraction failed", {"reason": "timeout"})
    assert "Extraction failed" in str(err)
    assert err.recoverable is True

def test_analysis_error():
    err = AnalysisError("Analysis failed", {"reason": "LLM error"}, recoverable=False)
    assert "Analysis failed" in str(err)
    assert err.recoverable is False

def test_timeout_error():
    err = PipelineTimeoutError("extract_data", 60)
    assert "extract_data" in str(err)
    assert "60" in str(err)

@pytest.mark.asyncio
async def test_with_timeout_decorator():
    @with_timeout(1)
    async def quick_func():
        return "done"
    
    result = await quick_func()
    assert result == "done"

@pytest.mark.asyncio
async def test_with_timeout_decorator_timeout():
    @with_timeout(1)
    async def slow_func():
        await asyncio.sleep(5)
        return "done"
    
    with pytest.raises(PipelineTimeoutError):
        await slow_func()
