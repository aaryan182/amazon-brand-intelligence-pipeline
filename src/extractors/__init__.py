"""
Extractors module for Amazon Brand Intelligence Pipeline.

This module provides data extraction components for the first step of the
brand intelligence pipeline, handling domain processing, Amazon search,
and product data extraction.

Components:
    - BrandExtractor: Main extraction orchestrator
    - DomainProcessor: Domain validation and brand name extraction
    - ConfidenceScorer: Confidence scoring calculations
    - CategoryDetector: Product category detection and normalization
    - ExtractionPrompts: LLM prompt templates (legacy, in brand_extractor)
    - Prompts: Optimized Claude prompt templates module
"""

from src.extractors.brand_extractor import (
    # Main class
    BrandExtractor,
    # Convenience function
    extract_brand_data,
    # Supporting classes
    DomainProcessor,
    ConfidenceScorer,
    CategoryDetector,
    ExtractionPrompts,
    # Data models
    ExtractionStage,
    StageResult,
    ExtractionMetrics,
    BrandNameVariation,
    SearchQuery,
)

from src.extractors.prompts import (
    # Configs
    PromptConfig,
    BRAND_EXTRACTION_CONFIG,
    PRODUCT_ENRICHMENT_CONFIG,
    CATEGORY_CLASSIFICATION_CONFIG,
    CONFIDENCE_ASSESSMENT_CONFIG,
    VALIDATION_RETRY_CONFIG,
    # Formatters
    format_brand_extraction_prompt,
    format_product_enrichment_prompt,
    format_category_classification_prompt,
    format_confidence_assessment_prompt,
    format_validation_retry_prompt,
    # Registry
    PROMPT_REGISTRY,
    get_prompt,
)

__all__ = [
    # Main class
    "BrandExtractor",
    # Convenience function
    "extract_brand_data",
    # Supporting classes
    "DomainProcessor",
    "ConfidenceScorer",
    "CategoryDetector",
    "ExtractionPrompts",
    # Data models
    "ExtractionStage",
    "StageResult",
    "ExtractionMetrics",
    "BrandNameVariation",
    "SearchQuery",
    # Prompts module
    "PromptConfig",
    "BRAND_EXTRACTION_CONFIG",
    "PRODUCT_ENRICHMENT_CONFIG",
    "CATEGORY_CLASSIFICATION_CONFIG",
    "CONFIDENCE_ASSESSMENT_CONFIG",
    "VALIDATION_RETRY_CONFIG",
    "format_brand_extraction_prompt",
    "format_product_enrichment_prompt",
    "format_category_classification_prompt",
    "format_confidence_assessment_prompt",
    "format_validation_retry_prompt",
    "PROMPT_REGISTRY",
    "get_prompt",
]
