"""
Production-grade Claude API service for LLM-powered analysis.

This module provides a robust interface for interacting with Anthropic's Claude API,
featuring async support, retry logic, rate limiting, token tracking, streaming,
and structured output enforcement.

Key Features:
    - Async/await support for non-blocking operations
    - Exponential backoff retry logic with jitter
    - Rate limiting to respect API quotas
    - Token counting and budget tracking
    - Streaming support for large responses
    - Pydantic schema validation for structured outputs
    - Comprehensive error handling and logging
    - Request caching for cost optimization

Example:
    >>> service = ClaudeService()
    >>> result = await service.extract_structured_data(raw_text, MySchema)
    >>> analysis = await service.generate_analysis(extraction_result)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import lru_cache
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Generic,
    Optional,
    Type,
    TypeVar,
    Union,
)

import anthropic
from anthropic import APIError, APIStatusError, RateLimitError
from pydantic import BaseModel, ValidationError

from src.config.settings import Settings, get_settings
from src.models.schemas import (
    ExtractionResult,
    StrategicInsight,
    MarketPosition,
    SWOTAnalysis,
    CompetitorInsight,
)
from src.utils.logger import get_logger

# Type variable for generic schema validation
T = TypeVar("T", bound=BaseModel)

# Logger instance
logger = get_logger(__name__)


# =============================================================================
# Constants and Enums
# =============================================================================

class TaskType(str, Enum):
    """Task types with associated temperature settings."""
    EXTRACTION = "extraction"
    ANALYSIS = "analysis"
    VALIDATION = "validation"
    SUMMARIZATION = "summarization"


# Temperature settings per task type
TEMPERATURE_SETTINGS: dict[TaskType, float] = {
    TaskType.EXTRACTION: 0.3,    # Low temperature for precise extraction
    TaskType.ANALYSIS: 0.7,      # Higher for creative analysis
    TaskType.VALIDATION: 0.1,    # Very low for validation
    TaskType.SUMMARIZATION: 0.5, # Medium for balanced summarization
}

# Token costs per model (per 1K tokens) - Claude 3.5 Sonnet pricing
TOKEN_COSTS = {
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
}

# Rate limiting settings
DEFAULT_RATE_LIMIT = 50  # requests per minute
DEFAULT_TOKEN_LIMIT = 100000  # tokens per minute


# =============================================================================
# Prompt Templates
# =============================================================================

SYSTEM_PROMPTS = {
    "extraction": """You are a data extraction specialist. Your task is to extract structured data from unstructured text and return it as valid JSON.

CRITICAL RULES:
1. ONLY output valid JSON - no markdown, no explanation, no code blocks
2. Follow the exact schema structure provided
3. Use null for missing values, never make up data
4. Ensure all strings are properly escaped
5. All numeric values should be actual numbers, not strings

Think step-by-step:
1. Identify all relevant data points in the text
2. Map them to the required schema fields
3. Validate the JSON structure before outputting""",

    "analysis": """You are a senior brand strategist and market analyst with expertise in e-commerce and Amazon marketplace dynamics.

Your analysis should be:
- Data-driven and specific
- Actionable with clear recommendations
- Balanced, covering both opportunities and risks
- Professional yet accessible

Always structure your thinking:
1. Analyze the current state
2. Identify patterns and trends
3. Compare against market benchmarks
4. Develop strategic recommendations

Output your analysis as valid JSON following the exact schema provided.""",

    "validation": """You are a JSON validation and correction specialist. 
Fix any JSON errors while preserving the original intent and data.
Only output the corrected JSON - no explanation or markdown.""",
}

EXTRACTION_FEW_SHOT = """
Example Input:
"Found 3 products: 
- ASIN B07XYZ1234: Nike Running Shoes, $89.99, 4.5 stars (2,341 reviews)
- ASIN B08ABC5678: Nike Training Shoes, $79.99, 4.3 stars (1,892 reviews)"

Example Output:
{
  "products": [
    {
      "asin": "B07XYZ1234",
      "title": "Nike Running Shoes",
      "price": 89.99,
      "rating": 4.5,
      "review_count": 2341
    },
    {
      "asin": "B08ABC5678",
      "title": "Nike Training Shoes",
      "price": 79.99,
      "rating": 4.3,
      "review_count": 1892
    }
  ]
}
"""

ANALYSIS_PROMPT_TEMPLATE = """
Analyze the following brand data and provide strategic insights:

## Brand Information
- Brand Name: {brand_name}
- Domain: {domain}
- Amazon Presence: {amazon_presence}
- Primary Category: {primary_category}
- Product Count: {product_count}

## Product Metrics
- Price Range: ${price_min:.2f} - ${price_max:.2f}
- Average Rating: {avg_rating}/5.0
- Total Reviews: {total_reviews:,}
- Top Products: {top_products}

## Competitors Identified
{competitors}

## Analysis Requirements
Provide a comprehensive analysis including:
1. Market Position Assessment (leader/challenger/follower/niche/emerging)
2. SWOT Analysis (2-3 items per category)
3. Key Competitive Advantages
4. Risk Factors
5. Strategic Growth Recommendations
6. Overall Brand Health Score (0-100)

## Output Schema
Return your analysis as JSON matching this exact structure:
{schema}
"""


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TokenUsage:
    """Token usage tracking for a single request."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    model: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def calculate_cost(self, model: str) -> float:
        """Calculate estimated cost based on token usage."""
        if model in TOKEN_COSTS:
            costs = TOKEN_COSTS[model]
            input_cost = (self.input_tokens / 1000) * costs["input"]
            output_cost = (self.output_tokens / 1000) * costs["output"]
            self.estimated_cost = input_cost + output_cost
        return self.estimated_cost


@dataclass
class RateLimiter:
    """Token bucket rate limiter."""
    max_requests: int = DEFAULT_RATE_LIMIT
    max_tokens: int = DEFAULT_TOKEN_LIMIT
    window_seconds: int = 60
    
    _request_timestamps: list[float] = field(default_factory=list)
    _token_usage: list[tuple[float, int]] = field(default_factory=list)
    
    def _cleanup_old_entries(self) -> None:
        """Remove entries outside the time window."""
        cutoff = time.time() - self.window_seconds
        self._request_timestamps = [t for t in self._request_timestamps if t > cutoff]
        self._token_usage = [(t, tokens) for t, tokens in self._token_usage if t > cutoff]
    
    async def acquire(self, estimated_tokens: int = 0) -> float:
        """
        Acquire permission to make a request.
        
        Returns the number of seconds to wait before proceeding.
        """
        self._cleanup_old_entries()
        
        # Check request rate
        if len(self._request_timestamps) >= self.max_requests:
            wait_time = self._request_timestamps[0] + self.window_seconds - time.time()
            if wait_time > 0:
                logger.warning(
                    "Rate limit reached, waiting",
                    wait_seconds=wait_time,
                    current_requests=len(self._request_timestamps),
                )
                await asyncio.sleep(wait_time)
                self._cleanup_old_entries()
        
        # Check token rate
        current_tokens = sum(tokens for _, tokens in self._token_usage)
        if current_tokens + estimated_tokens > self.max_tokens:
            wait_time = self._token_usage[0][0] + self.window_seconds - time.time()
            if wait_time > 0:
                logger.warning(
                    "Token limit reached, waiting",
                    wait_seconds=wait_time,
                    current_tokens=current_tokens,
                )
                await asyncio.sleep(wait_time)
                self._cleanup_old_entries()
        
        # Record this request
        now = time.time()
        self._request_timestamps.append(now)
        if estimated_tokens > 0:
            self._token_usage.append((now, estimated_tokens))
        
        return 0.0
    
    def record_usage(self, tokens: int) -> None:
        """Record actual token usage after a request."""
        self._token_usage.append((time.time(), tokens))


@dataclass
class CacheEntry:
    """Cache entry for request caching."""
    response: str
    timestamp: datetime
    tokens_used: int
    
    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if cache entry has expired."""
        return datetime.utcnow() - self.timestamp > timedelta(seconds=ttl_seconds)


# =============================================================================
# Custom Exceptions
# =============================================================================

class ClaudeServiceError(Exception):
    """Base exception for Claude service errors."""
    pass


class TokenLimitExceededError(ClaudeServiceError):
    """Raised when token limit is exceeded."""
    pass


class SchemaValidationError(ClaudeServiceError):
    """Raised when response doesn't match expected schema."""
    def __init__(self, message: str, raw_response: str, errors: list[str]):
        super().__init__(message)
        self.raw_response = raw_response
        self.errors = errors


class MaxRetriesExceededError(ClaudeServiceError):
    """Raised when max retries are exceeded."""
    pass


# =============================================================================
# Main Service Class
# =============================================================================

class ClaudeService:
    """
    Production-grade Claude API service with comprehensive features.
    
    Features:
        - Async/await support
        - Automatic retry with exponential backoff
        - Rate limiting
        - Token tracking and cost estimation
        - Request caching
        - Structured output validation
        - Streaming support
    
    Example:
        >>> service = ClaudeService()
        >>> async with service:
        ...     result = await service.extract_structured_data(text, Schema)
        ...     analysis = await service.generate_analysis(extraction)
    
    Attributes:
        settings: Application settings
        client: Anthropic API client
        rate_limiter: Rate limiter instance
        token_usage_history: List of token usage records
        total_cost: Running total of API costs
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        cache_ttl: int = 3600,
        enable_cache: bool = True,
    ):
        """
        Initialize the Claude service.
        
        Args:
            settings: Application settings instance
            api_key: Override API key (uses settings if not provided)
            max_retries: Maximum retry attempts for failed requests
            cache_ttl: Cache time-to-live in seconds
            enable_cache: Whether to enable response caching
        """
        self.settings = settings or get_settings()
        self._api_key = api_key or self.settings.anthropic_api_key.get_secret_value()
        self.max_retries = max_retries
        self.cache_ttl = cache_ttl
        self.enable_cache = enable_cache
        
        # Initialize client
        self.client = anthropic.AsyncAnthropic(api_key=self._api_key)
        
        # Rate limiter
        self.rate_limiter = RateLimiter(
            max_requests=self.settings.max_concurrent_requests * 10,
            max_tokens=100000,
        )
        
        # Token tracking
        self.token_usage_history: list[TokenUsage] = []
        self.total_cost: float = 0.0
        
        # Request cache
        self._cache: dict[str, CacheEntry] = {}
        
        # Streaming buffer
        self._stream_buffer: str = ""
        
        logger.info(
            "ClaudeService initialized",
            model=self.settings.claude_model,
            max_retries=self.max_retries,
            cache_enabled=self.enable_cache,
        )
    
    async def __aenter__(self) -> "ClaudeService":
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
    
    async def close(self) -> None:
        """Close the client connection."""
        await self.client.close()
        logger.info(
            "ClaudeService closed",
            total_requests=len(self.token_usage_history),
            total_cost=f"${self.total_cost:.4f}",
        )
    
    # =========================================================================
    # Core API Methods
    # =========================================================================
    
    async def _call_api(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        task_type: TaskType = TaskType.ANALYSIS,
        use_cache: bool = True,
    ) -> tuple[str, TokenUsage]:
        """
        Make an API call with retry logic and rate limiting.
        
        Args:
            messages: List of message dicts with role and content
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            task_type: Type of task for logging
            use_cache: Whether to use caching for this request
        
        Returns:
            Tuple of (response_text, token_usage)
        
        Raises:
            ClaudeServiceError: On API errors after retries exhausted
            TokenLimitExceededError: If request exceeds token limits
        """
        max_tokens = max_tokens or self.settings.claude_max_tokens
        
        # Check cache
        cache_key = self._generate_cache_key(messages, system, temperature)
        if self.enable_cache and use_cache:
            cached = self._get_cached_response(cache_key)
            if cached:
                logger.debug("Cache hit", task_type=task_type.value)
                return cached.response, TokenUsage(
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=cached.tokens_used,
                    model=self.settings.claude_model,
                )
        
        # Estimate tokens for rate limiting
        estimated_input_tokens = self._estimate_tokens(
            system + "".join(m["content"] for m in messages)
        )
        
        # Apply rate limiting
        await self.rate_limiter.acquire(estimated_input_tokens + max_tokens)
        
        # Retry loop with exponential backoff
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                response = await self.client.messages.create(
                    model=self.settings.claude_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=messages,
                )
                
                elapsed = time.time() - start_time
                
                # Extract response text
                response_text = response.content[0].text
                
                # Track token usage
                usage = TokenUsage(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                    model=self.settings.claude_model,
                )
                usage.calculate_cost(self.settings.claude_model)
                
                self.token_usage_history.append(usage)
                self.total_cost += usage.estimated_cost
                self.rate_limiter.record_usage(usage.total_tokens)
                
                # Cache response
                if self.enable_cache and use_cache:
                    self._cache_response(cache_key, response_text, usage.total_tokens)
                
                logger.info(
                    "API call successful",
                    task_type=task_type.value,
                    attempt=attempt + 1,
                    elapsed_seconds=f"{elapsed:.2f}",
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cost=f"${usage.estimated_cost:.4f}",
                )
                
                return response_text, usage
                
            except RateLimitError as e:
                last_error = e
                wait_time = self._calculate_backoff(attempt, base=30)
                logger.warning(
                    "Rate limit hit, backing off",
                    attempt=attempt + 1,
                    wait_seconds=wait_time,
                    error=str(e),
                )
                await asyncio.sleep(wait_time)
                
            except APIStatusError as e:
                last_error = e
                if e.status_code >= 500:
                    # Server error - retry with backoff
                    wait_time = self._calculate_backoff(attempt)
                    logger.warning(
                        "Server error, retrying",
                        attempt=attempt + 1,
                        status_code=e.status_code,
                        wait_seconds=wait_time,
                    )
                    await asyncio.sleep(wait_time)
                elif e.status_code == 401:
                    # Auth error - don't retry
                    logger.error("Authentication failed", error=str(e))
                    raise ClaudeServiceError(f"Authentication failed: {e}")
                else:
                    # Other client error
                    logger.error("API error", status_code=e.status_code, error=str(e))
                    raise ClaudeServiceError(f"API error: {e}")
                    
            except APIError as e:
                last_error = e
                wait_time = self._calculate_backoff(attempt)
                logger.warning(
                    "API error, retrying",
                    attempt=attempt + 1,
                    wait_seconds=wait_time,
                    error=str(e),
                )
                await asyncio.sleep(wait_time)
                
            except asyncio.TimeoutError as e:
                last_error = e
                wait_time = self._calculate_backoff(attempt)
                logger.warning(
                    "Request timeout, retrying",
                    attempt=attempt + 1,
                    wait_seconds=wait_time,
                )
                await asyncio.sleep(wait_time)
        
        # All retries exhausted
        logger.error(
            "Max retries exceeded",
            task_type=task_type.value,
            max_retries=self.max_retries,
            last_error=str(last_error),
        )
        raise MaxRetriesExceededError(
            f"Failed after {self.max_retries} attempts: {last_error}"
        )
    
    async def _call_api_streaming(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """
        Make a streaming API call for large responses.
        
        Args:
            messages: List of message dicts
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        
        Yields:
            Response text chunks as they arrive
        """
        max_tokens = max_tokens or self.settings.claude_max_tokens
        
        await self.rate_limiter.acquire(max_tokens)
        
        try:
            async with self.client.messages.stream(
                model=self.settings.claude_model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=messages,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                
                # Get final message for token tracking
                final_message = await stream.get_final_message()
                usage = TokenUsage(
                    input_tokens=final_message.usage.input_tokens,
                    output_tokens=final_message.usage.output_tokens,
                    total_tokens=final_message.usage.input_tokens + final_message.usage.output_tokens,
                    model=self.settings.claude_model,
                )
                usage.calculate_cost(self.settings.claude_model)
                self.token_usage_history.append(usage)
                self.total_cost += usage.estimated_cost
                
        except Exception as e:
            logger.error("Streaming error", error=str(e))
            raise ClaudeServiceError(f"Streaming failed: {e}")
    
    # =========================================================================
    # High-Level Methods
    # =========================================================================
    
    async def extract_structured_data(
        self,
        raw_text: str,
        schema: Type[T],
        additional_context: str = "",
    ) -> T:
        """
        Extract structured data from unstructured text using Claude.
        
        Uses low temperature for precise extraction and enforces
        Pydantic schema compliance with auto-retry on failures.
        
        Args:
            raw_text: Unstructured text to extract data from
            schema: Pydantic model class defining expected structure
            additional_context: Additional context for extraction
        
        Returns:
            Instance of the provided schema class
        
        Raises:
            SchemaValidationError: If extraction fails after retries
        
        Example:
            >>> from src.models.schemas import AmazonProduct
            >>> product = await service.extract_structured_data(
            ...     "ASIN B07XYZ1234: Nike Shoes $89.99 4.5 stars",
            ...     AmazonProduct
            ... )
        """
        # Build schema description
        schema_json = schema.model_json_schema()
        schema_str = json.dumps(schema_json, indent=2)
        
        # Build extraction prompt
        prompt = f"""Extract structured data from the following text.

## Text to Extract From:
{raw_text}

{f"## Additional Context:{chr(10)}{additional_context}" if additional_context else ""}

## Required Output Schema:
{schema_str}

## Examples:
{EXTRACTION_FEW_SHOT}

## Instructions:
1. Carefully read the text and identify all relevant data points
2. Map each data point to the appropriate schema field
3. For missing data, use null (do not invent values)
4. Ensure numeric fields are actual numbers, not strings
5. Return ONLY valid JSON matching the schema - no markdown or explanation

Output the JSON now:"""

        # Call API with low temperature
        response_text, usage = await self._call_api(
            messages=[{"role": "user", "content": prompt}],
            system=SYSTEM_PROMPTS["extraction"],
            temperature=TEMPERATURE_SETTINGS[TaskType.EXTRACTION],
            task_type=TaskType.EXTRACTION,
        )
        
        # Validate and parse response
        return await self.validate_and_retry(
            response_text,
            schema,
            max_retries=3,
            original_prompt=prompt,
        )
    
    async def generate_analysis(
        self,
        extraction_data: ExtractionResult,
    ) -> StrategicInsight:
        """
        Generate strategic analysis from extraction data.
        
        Uses chain-of-thought prompting for comprehensive analysis
        including market positioning, SWOT, and recommendations.
        
        Args:
            extraction_data: Completed extraction result
        
        Returns:
            StrategicInsight with complete analysis
        
        Example:
            >>> analysis = await service.generate_analysis(extraction_result)
            >>> print(analysis.market_position)
            >>> print(analysis.swot.strengths)
        """
        # Prepare top products summary
        top_products_summary = []
        for p in extraction_data.get_top_rated_products(limit=5):
            top_products_summary.append(
                f"  - {p.title[:50]}... (${p.price}, {p.rating}★, {p.review_count:,} reviews)"
            )
        
        # Prepare competitors summary
        competitors_summary = "\n".join(
            f"  - {c}" for c in extraction_data.competitors_found[:10]
        ) or "  - No direct competitors identified"
        
        # Build analysis prompt
        price_range = extraction_data.price_range or (0, 0)
        prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            brand_name=extraction_data.brand_name,
            domain=extraction_data.domain,
            amazon_presence="Confirmed" if extraction_data.amazon_presence.found else "Not Found",
            primary_category=extraction_data.primary_category or "Unknown",
            product_count=extraction_data.estimated_product_count,
            price_min=price_range[0],
            price_max=price_range[1],
            avg_rating=extraction_data.average_rating or "N/A",
            total_reviews=extraction_data.total_reviews,
            top_products="\n" + "\n".join(top_products_summary) if top_products_summary else "No products found",
            competitors=competitors_summary,
            schema=json.dumps(StrategicInsight.model_json_schema(), indent=2),
        )
        
        # Call API with higher temperature for creative analysis
        response_text, usage = await self._call_api(
            messages=[{"role": "user", "content": prompt}],
            system=SYSTEM_PROMPTS["analysis"],
            temperature=TEMPERATURE_SETTINGS[TaskType.ANALYSIS],
            task_type=TaskType.ANALYSIS,
            max_tokens=4096,  # Allow longer analysis
        )
        
        # Validate and parse
        return await self.validate_and_retry(
            response_text,
            StrategicInsight,
            max_retries=3,
            original_prompt=prompt,
        )
    
    async def validate_and_retry(
        self,
        response: str,
        expected_schema: Type[T],
        max_retries: int = 3,
        original_prompt: str = "",
    ) -> T:
        """
        Validate response against schema and retry with corrections if invalid.
        
        Args:
            response: Raw response text from API
            expected_schema: Pydantic model to validate against
            max_retries: Maximum retry attempts
            original_prompt: Original prompt for context in corrections
        
        Returns:
            Validated instance of expected_schema
        
        Raises:
            SchemaValidationError: If validation fails after all retries
        """
        last_error: Optional[str] = None
        current_response = response
        
        for attempt in range(max_retries):
            try:
                # Extract JSON from response
                json_str = self._extract_json(current_response)
                
                # Parse JSON
                data = json.loads(json_str)
                
                # Validate against schema
                result = expected_schema.model_validate(data)
                
                if attempt > 0:
                    logger.info(
                        "Validation succeeded after correction",
                        attempt=attempt + 1,
                        schema=expected_schema.__name__,
                    )
                
                return result
                
            except json.JSONDecodeError as e:
                last_error = f"JSON parse error: {e}"
                logger.warning(
                    "Invalid JSON, attempting correction",
                    attempt=attempt + 1,
                    error=str(e),
                )
                
            except ValidationError as e:
                errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
                last_error = f"Validation errors: {'; '.join(errors)}"
                logger.warning(
                    "Schema validation failed, attempting correction",
                    attempt=attempt + 1,
                    errors=errors[:3],  # Log first 3 errors
                )
            
            # Attempt correction
            if attempt < max_retries - 1:
                current_response = await self._request_correction(
                    current_response,
                    last_error,
                    expected_schema,
                )
        
        # All retries failed
        raise SchemaValidationError(
            f"Failed to validate response after {max_retries} attempts",
            raw_response=response,
            errors=[last_error] if last_error else [],
        )
    
    async def _request_correction(
        self,
        invalid_response: str,
        error_message: str,
        expected_schema: Type[BaseModel],
    ) -> str:
        """Request Claude to correct an invalid response."""
        correction_prompt = f"""The following JSON response has errors:

## Invalid Response:
{invalid_response[:2000]}  # Truncate if too long

## Error:
{error_message}

## Expected Schema:
{json.dumps(expected_schema.model_json_schema(), indent=2)}

Please fix the JSON to match the expected schema. Output ONLY the corrected JSON:"""

        response_text, _ = await self._call_api(
            messages=[{"role": "user", "content": correction_prompt}],
            system=SYSTEM_PROMPTS["validation"],
            temperature=TEMPERATURE_SETTINGS[TaskType.VALIDATION],
            task_type=TaskType.VALIDATION,
            use_cache=False,
        )
        
        return response_text
    
    # =========================================================================
    # Streaming Methods
    # =========================================================================
    
    async def stream_analysis(
        self,
        extraction_data: ExtractionResult,
    ) -> AsyncIterator[str]:
        """
        Stream analysis generation for real-time display.
        
        Args:
            extraction_data: Completed extraction result
        
        Yields:
            Text chunks as they're generated
        """
        price_range = extraction_data.price_range or (0, 0)
        prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            brand_name=extraction_data.brand_name,
            domain=extraction_data.domain,
            amazon_presence="Confirmed" if extraction_data.amazon_presence.found else "Not Found",
            primary_category=extraction_data.primary_category or "Unknown",
            product_count=extraction_data.estimated_product_count,
            price_min=price_range[0],
            price_max=price_range[1],
            avg_rating=extraction_data.average_rating or "N/A",
            total_reviews=extraction_data.total_reviews,
            top_products="N/A",
            competitors="N/A",
            schema=json.dumps(StrategicInsight.model_json_schema(), indent=2),
        )
        
        async for chunk in self._call_api_streaming(
            messages=[{"role": "user", "content": prompt}],
            system=SYSTEM_PROMPTS["analysis"],
            temperature=TEMPERATURE_SETTINGS[TaskType.ANALYSIS],
        ):
            yield chunk
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that may contain markdown or other content."""
        # Try to find JSON in code blocks first
        code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
        matches = re.findall(code_block_pattern, text)
        if matches:
            return matches[0].strip()
        
        # Try to find raw JSON object or array
        json_pattern = r"(\{[\s\S]*\}|\[[\s\S]*\])"
        matches = re.findall(json_pattern, text)
        if matches:
            # Return the longest match (likely the full JSON)
            return max(matches, key=len)
        
        # Return original text as fallback
        return text.strip()
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for a text string."""
        # Rough estimation: ~4 characters per token for English
        return len(text) // 4
    
    def _calculate_backoff(self, attempt: int, base: float = 1.0) -> float:
        """Calculate exponential backoff with jitter."""
        import random
        backoff = base * (2 ** attempt)
        jitter = random.uniform(0, backoff * 0.1)
        return min(backoff + jitter, 60)  # Cap at 60 seconds
    
    def _generate_cache_key(
        self,
        messages: list[dict[str, str]],
        system: str,
        temperature: float,
    ) -> str:
        """Generate a cache key from request parameters."""
        content = json.dumps({
            "messages": messages,
            "system": system,
            "temperature": temperature,
            "model": self.settings.claude_model,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _get_cached_response(self, key: str) -> Optional[CacheEntry]:
        """Get cached response if valid."""
        if key in self._cache:
            entry = self._cache[key]
            if not entry.is_expired(self.cache_ttl):
                return entry
            else:
                del self._cache[key]
        return None
    
    def _cache_response(self, key: str, response: str, tokens: int) -> None:
        """Cache a response."""
        self._cache[key] = CacheEntry(
            response=response,
            timestamp=datetime.utcnow(),
            tokens_used=tokens,
        )
        
        # Limit cache size
        if len(self._cache) > 100:
            # Remove oldest entries
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].timestamp,
            )
            for key, _ in sorted_entries[:20]:
                del self._cache[key]
    
    def get_usage_stats(self) -> dict[str, Any]:
        """Get usage statistics for the service."""
        if not self.token_usage_history:
            return {
                "total_requests": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "avg_tokens_per_request": 0,
            }
        
        total_tokens = sum(u.total_tokens for u in self.token_usage_history)
        total_input = sum(u.input_tokens for u in self.token_usage_history)
        total_output = sum(u.output_tokens for u in self.token_usage_history)
        
        return {
            "total_requests": len(self.token_usage_history),
            "total_tokens": total_tokens,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cost": self.total_cost,
            "avg_tokens_per_request": total_tokens // len(self.token_usage_history),
            "cache_size": len(self._cache),
            "cache_hit_rate": "N/A",  # Would need tracking
        }
    
    def reset_usage_stats(self) -> None:
        """Reset usage statistics."""
        self.token_usage_history.clear()
        self.total_cost = 0.0
        logger.info("Usage statistics reset")


# =============================================================================
# Convenience Functions
# =============================================================================

async def create_claude_service(
    settings: Optional[Settings] = None,
) -> ClaudeService:
    """
    Factory function to create a configured ClaudeService.
    
    Args:
        settings: Optional settings override
    
    Returns:
        Configured ClaudeService instance
    """
    return ClaudeService(settings=settings)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "ClaudeService",
    "create_claude_service",
    "TaskType",
    "TokenUsage",
    "ClaudeServiceError",
    "TokenLimitExceededError",
    "SchemaValidationError",
    "MaxRetriesExceededError",
]
