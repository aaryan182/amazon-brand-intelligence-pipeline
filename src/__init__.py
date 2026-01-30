"""
Amazon Brand Intelligence Pipeline.

A production-grade AI-powered pipeline for extracting and analyzing
Amazon brand data using LangGraph, Claude, and SerpAPI/Exa.
"""

__version__ = "1.0.0"
__author__ = "Amazon Brand Intelligence Team"

# Lazy imports to avoid circular dependencies
def get_pipeline():
    """Get the BrandIntelligencePipeline class (lazy import)."""
    from src.pipeline.orchestrator import BrandIntelligencePipeline
    return BrandIntelligencePipeline

__all__ = ["get_pipeline", "__version__"]
