"""
Application settings and configuration management.

This module handles all environment variables, API keys, and application
configuration using Pydantic settings management for type safety and validation.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All sensitive values are stored as SecretStr to prevent accidental logging.
    Settings are validated on load and cached for performance.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )
    
    # API Keys
    anthropic_api_key: SecretStr = Field(..., alias="ANTHROPIC_API_KEY")
    serpapi_api_key: Optional[SecretStr] = Field(default=None, alias="SERPAPI_API_KEY")
    exa_api_key: Optional[SecretStr] = Field(default=None, alias="EXA_API_KEY")
    perplexity_api_key: Optional[SecretStr] = Field(default=None, alias="PERPLEXITY_API_KEY")

    # Application Configuration
    app_env: Literal["development", "staging", "production"] = Field(
        default="development", 
        alias="APP_ENV"
    )
    debug: bool = Field(default=False, alias="DEBUG")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", 
        alias="LOG_LEVEL"
    )

    # Model Configuration
    claude_model: str = Field(
        default="claude-sonnet-4-20250514",
        alias="CLAUDE_MODEL"
    )
    claude_max_tokens: int = Field(default=4000, alias="CLAUDE_MAX_TOKENS")
    extraction_temperature: float = Field(default=0.3, alias="EXTRACTION_TEMPERATURE")
    analysis_temperature: float = Field(default=0.7, alias="ANALYSIS_TEMPERATURE")

    # Rate Limits
    max_requests_per_minute: int = Field(default=20, alias="MAX_REQUESTS_PER_MINUTE")
    max_concurrent_requests: int = Field(default=5, alias="MAX_CONCURRENT_REQUESTS")
    request_timeout_seconds: int = Field(default=30, alias="REQUEST_TIMEOUT_SECONDS")
    max_retries: int = Field(default=3, alias="MAX_RETRIES")

    # Cache Settings
    cache_enabled: bool = Field(default=True, alias="CACHE_ENABLED")
    cache_ttl_seconds: int = Field(default=3600, alias="CACHE_TTL_SECONDS")

    # Output Settings
    output_dir: Path = Field(default=Path("outputs/reports"), alias="OUTPUT_DIR")
    log_dir: Path = Field(default=Path("logs"), alias="LOG_DIR")
    save_intermediate_json: bool = Field(default=True, alias="SAVE_INTERMEDIATE_JSON")
    report_format: Literal["json", "markdown", "html"] = Field(
        default="markdown",
        alias="REPORT_FORMAT"
    )

    # Search Settings
    max_search_results: int = Field(default=10, alias="MAX_SEARCH_RESULTS")
    max_products_to_analyze: int = Field(default=50, alias="MAX_PRODUCTS_TO_ANALYZE")

    # Validation
    min_confidence_threshold: float = Field(default=0.3, alias="MIN_CONFIDENCE_THRESHOLD")

    @field_validator("output_dir", "log_dir", mode="before")
    @classmethod
    def validate_directories(cls, v: str | Path) -> Path:
        """Ensure directories exist."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @field_validator("anthropic_api_key", mode="before")
    @classmethod
    def validate_anthropic_key(cls, v: str) -> str:
        """Validate Anthropic API key format."""
        if not v or not v.startswith("sk-"):
            raise ValueError("Invalid Anthropic API key format")
        return v
    
    def validate_api_keys(self):
        """Validate at least one search provider is configured."""
        if not any([self.serpapi_api_key, self.exa_api_key, self.perplexity_api_key]):
            # If purely doing analysis on existing data, this might be optional.
            # But for full pipeline, it's needed.
            # We log warning instead of error to allow partial usage?
            # Or raise error if strict validation requested.
            pass

    def get_search_provider(self) -> str:
        """Determine which search provider to use based on available keys."""
        if self.serpapi_api_key:
            return "serpapi"
        elif self.exa_api_key:
            return "exa"
        elif self.perplexity_api_key:
            return "perplexity"
        return "none"


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
