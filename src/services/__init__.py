"""
Services package for the Amazon Brand Intelligence Pipeline.

This package contains all service classes for external API integrations
and core business logic.

Services:
    - ClaudeService: LLM-powered analysis using Anthropic Claude
    - SearchService: Multi-provider Amazon product search
    - ValidationService: Data validation and sanitization

Providers:
    - SerpAPIProvider: Primary search provider using SerpAPI
    - ExaAIProvider: Neural search and entity extraction
    - PerplexityProvider: AI-powered contextual search
"""

from src.services.llm_service import (
    ClaudeService,
    ClaudeServiceError,
    MaxRetriesExceededError,
    SchemaValidationError,
    TaskType,
    TokenLimitExceededError,
    TokenUsage,
    create_claude_service,
)
from src.services.search_service import (
    # Service
    SearchService,
    create_search_service,
    # Providers
    SearchProvider,
    SerpAPIProvider,
    ExaAIProvider,
    PerplexityProvider,
    # Models
    SearchResults,
    SearchResultItem,
    ProductDetails,
    PresenceConfidence,
    ProviderStatus,
    # Utilities
    SearchCache,
    RateLimiter,
    # Exceptions
    ProviderError,
    RateLimitError,
    NoResultsError,
    ConfigurationError,
)
from src.services.validation_service import ValidationService

__all__ = [
    # LLM Service
    "ClaudeService",
    "create_claude_service",
    "TaskType",
    "TokenUsage",
    "ClaudeServiceError",
    "TokenLimitExceededError",
    "SchemaValidationError",
    "MaxRetriesExceededError",
    # Search Service
    "SearchService",
    "create_search_service",
    # Search Providers
    "SearchProvider",
    "SerpAPIProvider",
    "ExaAIProvider",
    "PerplexityProvider",
    # Search Models
    "SearchResults",
    "SearchResultItem",
    "ProductDetails",
    "PresenceConfidence",
    "ProviderStatus",
    # Search Utilities
    "SearchCache",
    "RateLimiter",
    # Search Exceptions
    "ProviderError",
    "RateLimitError",
    "NoResultsError",
    "ConfigurationError",
    # Validation
    "ValidationService",
]
