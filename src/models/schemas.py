"""
Pydantic models and schemas for the Amazon Brand Intelligence Pipeline.

This module defines all data structures used throughout the pipeline,
ensuring type safety, validation, and serialization consistency.

Models:
    - BrandInput: Domain input validation
    - AmazonProduct: Individual product structure
    - ExtractionResult: Step 1 output
    - StrategicInsight: Step 2 analysis components
    - FinalReport: Complete report structure
    - ErrorResponse: Standardized error handling
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Self
from uuid import UUID, uuid4

from pydantic import (
    BaseModel as PydanticBaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_serializer,
    field_validator,
    model_validator,
)


# =============================================================================
# Base Configuration
# =============================================================================

class BaseModel(PydanticBaseModel):
    """Base model with common configuration for all schemas."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=False,  # Disable to prevent recursion with model_validators
        populate_by_name=True,
        use_enum_values=True,
        json_schema_extra={"examples": []},
        ser_json_timedelta="iso8601",
        ser_json_bytes="base64",
    )
    
    def to_json(self, **kwargs) -> str:
        """Serialize model to JSON string."""
        return self.model_dump_json(indent=2, **kwargs)
    
    def to_dict(self, **kwargs) -> dict[str, Any]:
        """Serialize model to dictionary."""
        return self.model_dump(**kwargs)
    
    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Deserialize model from JSON string."""
        return cls.model_validate_json(json_str)


class TimestampMixin(BaseModel):
    """Mixin for models that need timestamp tracking."""
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Record creation timestamp in ISO 8601 format",
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        description="Last update timestamp in ISO 8601 format",
    )
    
    @field_serializer("created_at", "updated_at")
    def serialize_datetime(self, value: Optional[datetime]) -> Optional[str]:
        """Serialize datetime to ISO format string."""
        if not value:
            return None
        # If naive, assume UTC and append Z
        if value.tzinfo is None:
            return value.isoformat() + "Z"
        # If aware, use standard isoformat (includes offset)
        return value.isoformat()


# =============================================================================
# Enums
# =============================================================================

class ConfidenceLevel(str, Enum):
    """Confidence level for analysis results."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class MarketPosition(str, Enum):
    """Brand market position classification."""
    LEADER = "leader"
    CHALLENGER = "challenger"
    FOLLOWER = "follower"
    NICHE = "niche"
    EMERGING = "emerging"


class AnalysisStatus(str, Enum):
    """Status of analysis operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PipelineStep(str, Enum):
    """Pipeline execution steps."""
    INITIALIZATION = "initialization"
    DATA_EXTRACTION = "data_extraction"
    DATA_VALIDATION = "data_validation"
    STRATEGIC_ANALYSIS = "strategic_analysis"
    REPORT_GENERATION = "report_generation"
    FINALIZATION = "finalization"


class AmazonCategory(str, Enum):
    """Amazon product category enumeration."""
    ELECTRONICS = "Electronics"
    COMPUTERS = "Computers & Accessories"
    HOME_KITCHEN = "Home & Kitchen"
    BEAUTY = "Beauty & Personal Care"
    HEALTH = "Health & Personal Care"
    GROCERY = "Grocery & Gourmet Food"
    TOYS = "Toys & Games"
    SPORTS = "Sports & Outdoors"
    CLOTHING = "Clothing, Shoes & Jewelry"
    AUTOMOTIVE = "Automotive"
    PET_SUPPLIES = "Pet Supplies"
    OFFICE = "Office Products"
    TOOLS = "Tools & Home Improvement"
    GARDEN = "Garden & Outdoor"
    BABY = "Baby"
    BOOKS = "Books"
    MUSIC = "Music"
    MOVIES_TV = "Movies & TV"
    VIDEO_GAMES = "Video Games"
    SOFTWARE = "Software"


class ErrorType(str, Enum):
    """Error type classification."""
    VALIDATION_ERROR = "validation_error"
    API_ERROR = "api_error"
    EXTRACTION_ERROR = "extraction_error"
    ANALYSIS_ERROR = "analysis_error"
    TIMEOUT_ERROR = "timeout_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    NOT_FOUND_ERROR = "not_found_error"
    INTERNAL_ERROR = "internal_error"


# =============================================================================
# Validators (Reusable)
# =============================================================================

# Domain validation pattern: valid TLD, no path/query
DOMAIN_PATTERN = re.compile(
    r"^(?!-)[A-Za-z0-9-]{1,63}(?<!-)(\.[A-Za-z0-9-]{1,63})*\.[A-Za-z]{2,}$"
)

# ASIN pattern: B0 + 8 alphanumeric characters
ASIN_PATTERN = re.compile(r"^[A-Z0-9]{10}$")

# Amazon URL pattern
AMAZON_URL_PATTERN = re.compile(
    r"^https?://(www\.)?amazon\.(com|co\.uk|de|fr|it|es|ca|com\.au|co\.jp|in)/.*$"
)


def validate_domain(domain: str) -> str:
    """Validate domain format - valid TLD, no path/query."""
    domain = domain.lower().strip()
    
    # Remove protocol if present
    if domain.startswith(("http://", "https://")):
        domain = domain.split("://", 1)[1]
    
    # Remove path and query string
    domain = domain.split("/")[0].split("?")[0]
    
    # Remove www prefix
    if domain.startswith("www."):
        domain = domain[4:]
    
    if not DOMAIN_PATTERN.match(domain):
        raise ValueError(
            f"Invalid domain format: '{domain}'. "
            "Must be a valid domain with TLD (e.g., 'example.com')"
        )
    
    return domain


def validate_asin(asin: str) -> str:
    """Validate Amazon ASIN format - 10 alphanumeric characters."""
    asin = asin.upper().strip()
    
    if not ASIN_PATTERN.match(asin):
        raise ValueError(
            f"Invalid ASIN format: '{asin}'. "
            "Must be 10 alphanumeric characters (e.g., 'B07XYZ1234')"
        )
    
    return asin


def validate_amazon_url(url: str) -> str:
    """Validate Amazon product URL format."""
    if not AMAZON_URL_PATTERN.match(url):
        raise ValueError(
            f"Invalid Amazon URL: '{url}'. "
            "Must be a valid Amazon product URL"
        )
    return url


# =============================================================================
# Input Models
# =============================================================================

class BrandInput(BaseModel):
    """
    Input model for brand analysis request.
    
    Validates the domain format and provides defaults for optional fields.
    
    Example:
        >>> brand = BrandInput(domain="patagonia.com")
        >>> brand.domain
        'patagonia.com'
    """
    
    domain: str = Field(
        ...,
        min_length=4,
        max_length=253,
        description="Brand's domain name without protocol (e.g., 'patagonia.com')",
        examples=["patagonia.com", "nike.com", "apple.com"],
    )
    brand_name: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Brand name (auto-extracted from domain if not provided)",
        examples=["Patagonia", "Nike", "Apple"],
    )
    category: Optional[AmazonCategory] = Field(
        default=None,
        description="Amazon category to search within",
    )
    marketplace: str = Field(
        default="amazon.com",
        description="Amazon marketplace domain",
        examples=["amazon.com", "amazon.co.uk", "amazon.de"],
    )
    include_competitors: bool = Field(
        default=True,
        description="Whether to include competitor analysis",
    )
    max_products: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum products to analyze per brand",
    )
    
    @field_validator("domain", mode="before")
    @classmethod
    def validate_domain_format(cls, v: str) -> str:
        """Validate and normalize domain format."""
        return validate_domain(v)
    
    @model_validator(mode="after")
    def set_brand_name_from_domain(self) -> Self:
        """Extract brand name from domain if not provided."""
        if not self.brand_name:  # Handles None and empty string
            # Extract brand name from domain (first part, capitalized)
            domain_parts = self.domain.split(".")
            self.brand_name = domain_parts[0].replace("-", " ").title()
        return self


# =============================================================================
# Product Models
# =============================================================================

class PriceInfo(BaseModel):
    """Product pricing information with validation."""
    
    current_price: float = Field(
        ...,
        ge=0.01,
        le=1000000,
        description="Current product price in USD",
        examples=[89.99, 149.00, 24.99],
    )
    original_price: Optional[float] = Field(
        default=None,
        ge=0.01,
        le=1000000,
        description="Original/list price before discount",
        examples=[109.99, 199.00],
    )
    currency: str = Field(
        default="USD",
        min_length=3,
        max_length=3,
        description="ISO 4217 currency code",
        examples=["USD", "EUR", "GBP"],
    )
    discount_percentage: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Discount percentage if applicable",
    )
    
    @model_validator(mode="after")
    def calculate_discount(self) -> Self:
        """Calculate discount percentage if not provided."""
        if self.original_price and self.original_price > self.current_price:
            if self.discount_percentage is None:
                discount = (
                    (self.original_price - self.current_price) / self.original_price
                ) * 100
                self.discount_percentage = round(discount, 1)
        return self


class AmazonProduct(BaseModel):
    """
    Individual Amazon product structure with comprehensive validation.
    
    Represents a single product from Amazon search results.
    
    Example:
        >>> product = AmazonProduct(
        ...     asin="B07XYZ1234",
        ...     title="Patagonia Down Sweater Jacket",
        ...     price=189.99,
        ...     rating=4.5,
        ...     review_count=1234
        ... )
    """
    
    asin: str = Field(
        ...,
        min_length=10,
        max_length=10,
        description="Amazon Standard Identification Number (10 chars)",
        examples=["B07XYZ1234", "B09ABC5678"],
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Product title",
        examples=["Patagonia Down Sweater Jacket - Men's"],
    )
    price: Optional[float] = Field(
        default=None,
        ge=0.01,
        le=1000000,
        description="Current product price in USD",
        examples=[89.99, 149.00],
    )
    price_info: Optional[PriceInfo] = Field(
        default=None,
        description="Detailed pricing information",
    )
    rating: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=5.0,
        description="Average customer rating (0-5 stars)",
        examples=[4.5, 4.8, 3.9],
    )
    review_count: int = Field(
        default=0,
        ge=0,
        description="Total number of customer reviews",
        examples=[1234, 567, 89],
    )
    url: Optional[str] = Field(
        default=None,
        description="Full Amazon product URL",
        examples=["https://www.amazon.com/dp/B07XYZ1234"],
    )
    image_url: Optional[str] = Field(
        default=None,
        description="Product image URL",
    )
    rank: Optional[int] = Field(
        default=None,
        ge=1,
        description="Search result position/rank",
        examples=[1, 5, 10],
    )
    best_seller_rank: Optional[int] = Field(
        default=None,
        ge=1,
        description="Best seller rank in category",
    )
    category: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Product category path",
    )
    brand: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Product brand name",
    )
    is_prime: bool = Field(
        default=False,
        description="Prime eligibility status",
    )
    is_sponsored: bool = Field(
        default=False,
        description="Whether product is a sponsored listing",
    )
    is_amazon_choice: bool = Field(
        default=False,
        description="Amazon's Choice badge",
    )
    is_best_seller: bool = Field(
        default=False,
        description="Best Seller badge",
    )
    features: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Product feature bullet points",
    )
    
    @field_validator("asin", mode="before")
    @classmethod
    def validate_asin_format(cls, v: str) -> str:
        """Validate ASIN format."""
        return validate_asin(v)
    
    @field_validator("url", mode="before")
    @classmethod
    def validate_product_url(cls, v: Optional[str]) -> Optional[str]:
        """Validate Amazon URL if provided."""
        if v is None:
            return None
        return validate_amazon_url(v)
    
    @model_validator(mode="after")
    def generate_url_if_missing(self) -> Self:
        """Generate URL from ASIN if not provided."""
        if self.url is None:
            self.url = f"https://www.amazon.com/dp/{self.asin}"
        return self


# =============================================================================
# Extraction Models (Step 1 Output)
# =============================================================================

class AmazonPresence(BaseModel):
    """Amazon presence detection result."""
    
    found: bool = Field(
        ...,
        description="Whether the brand was found on Amazon",
    )
    confidence: ConfidenceLevel = Field(
        ...,
        description="Confidence level of the detection",
    )
    evidence: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Evidence supporting the detection",
        examples=[["Official store found", "Multiple verified products"]],
    )
    official_store_url: Optional[str] = Field(
        default=None,
        description="URL to official brand store if found",
    )
    verified_seller: bool = Field(
        default=False,
        description="Whether brand is a verified Amazon seller",
    )


class ExtractionMetadata(BaseModel):
    """Metadata about the extraction process."""
    
    search_queries_used: list[str] = Field(
        default_factory=list,
        description="Search queries executed during extraction",
        examples=[["Patagonia jacket", "Patagonia outdoor gear"]],
    )
    extraction_duration_seconds: float = Field(
        ...,
        ge=0,
        description="Total extraction time in seconds",
        examples=[45.2, 30.5],
    )
    api_calls_made: int = Field(
        default=0,
        ge=0,
        description="Number of API calls made during extraction",
        examples=[3, 5],
    )
    data_source: str = Field(
        default="serpapi",
        description="Primary data source used",
        examples=["serpapi", "exa"],
    )
    cache_hit: bool = Field(
        default=False,
        description="Whether results were served from cache",
    )


class ExtractionResult(TimestampMixin):
    """
    Step 1 output - Complete extraction result.
    
    Contains all data extracted from Amazon for a brand, including
    products, presence analysis, and metadata.
    
    Example:
        >>> result = ExtractionResult(
        ...     brand_name="Patagonia",
        ...     domain="patagonia.com",
        ...     amazon_presence=AmazonPresence(found=True, confidence="high"),
        ...     primary_category="Sports & Outdoors",
        ...     estimated_product_count=150,
        ...     metadata=ExtractionMetadata(extraction_duration_seconds=45.2)
        ... )
    """
    
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique extraction result identifier",
    )
    brand_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Brand name being analyzed",
        examples=["Patagonia", "Nike", "Apple"],
    )
    domain: str = Field(
        ...,
        description="Brand's domain name",
        examples=["patagonia.com"],
    )
    amazon_presence: AmazonPresence = Field(
        ...,
        description="Amazon presence detection results",
    )
    primary_category: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Primary Amazon category for brand products",
        examples=["Sports & Outdoors", "Electronics"],
    )
    estimated_product_count: int = Field(
        default=0,
        ge=0,
        description="Estimated total products on Amazon",
        examples=[150, 500],
    )
    top_products: list[AmazonProduct] = Field(
        default_factory=list,
        max_length=100,
        description="Top products found for the brand",
    )
    all_categories: list[str] = Field(
        default_factory=list,
        description="All categories where brand products were found",
    )
    price_range: Optional[tuple[float, float]] = Field(
        default=None,
        description="Price range (min, max) in USD",
        examples=[(24.99, 299.99)],
    )
    average_rating: Optional[float] = Field(
        default=None,
        ge=0,
        le=5,
        description="Average rating across all products",
    )
    total_reviews: int = Field(
        default=0,
        ge=0,
        description="Total reviews across all products",
    )
    competitors_found: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Competitor brands identified",
    )
    metadata: ExtractionMetadata = Field(
        ...,
        description="Extraction process metadata",
    )
    raw_data: Optional[dict[str, Any]] = Field(
        default=None,
        description="Raw API response data (for debugging)",
        exclude=True,  # Exclude from serialization by default
    )
    
    @field_validator("domain", mode="before")
    @classmethod
    def validate_domain_format(cls, v: str) -> str:
        """Validate domain format."""
        return validate_domain(v)
    
    @model_validator(mode="after")
    def calculate_aggregates(self) -> Self:
        """Calculate aggregate metrics from products."""
        if self.top_products:
            # Calculate price range
            prices = [p.price for p in self.top_products if p.price]
            if prices:
                self.price_range = (min(prices), max(prices))
            
            # Calculate average rating
            ratings = [p.rating for p in self.top_products if p.rating]
            if ratings:
                self.average_rating = round(sum(ratings) / len(ratings), 2)
            
            # Calculate total reviews
            self.total_reviews = sum(p.review_count for p in self.top_products)
        
        return self
    
    def get_top_rated_products(self, limit: int = 5) -> list[AmazonProduct]:
        """Get top-rated products."""
        return sorted(
            [p for p in self.top_products if p.rating],
            key=lambda p: (p.rating or 0, p.review_count),
            reverse=True,
        )[:limit]
    
    def get_best_sellers(self, limit: int = 5) -> list[AmazonProduct]:
        """Get products with best seller rank."""
        return sorted(
            [p for p in self.top_products if p.best_seller_rank],
            key=lambda p: p.best_seller_rank or float("inf"),
        )[:limit]


# =============================================================================
# Analysis Models (Step 2 Output)
# =============================================================================

class SWOTAnalysis(BaseModel):
    """SWOT analysis structure."""
    
    strengths: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Brand strengths identified",
        examples=[["Strong brand recognition", "Premium product quality"]],
    )
    weaknesses: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Brand weaknesses identified",
        examples=[["Higher price point", "Limited product range"]],
    )
    opportunities: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Growth opportunities",
        examples=[["Expanding direct-to-consumer", "New market segments"]],
    )
    threats: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Market threats",
        examples=[["Intense competition", "Economic downturn"]],
    )


class CompetitorInsight(BaseModel):
    """Insight about a competitor brand."""
    
    name: str = Field(..., description="Competitor brand name")
    estimated_market_share: Optional[float] = Field(
        default=None, ge=0, le=100, description="Estimated market share %"
    )
    price_positioning: str = Field(
        default="similar",
        description="Price positioning relative to analyzed brand",
        examples=["lower", "similar", "higher"],
    )
    key_differentiators: list[str] = Field(
        default_factory=list,
        description="Key differentiators from analyzed brand",
    )


class Recommendation(BaseModel):
    """
    Actionable strategic recommendation.
    """
    
    priority: int = Field(..., description="Priority level (1-5)")
    title: str = Field(..., description="Recommendation title")
    description: str = Field(..., description="Detailed description")
    impact: str = Field(..., description="Expected impact description")
    difficulty: str = Field(..., description="Implementation difficulty/effort")
    timeline: str = Field(..., description="Implementation timeline")


class StrategicInsight(BaseModel):
    """
    Step 2 analysis components - Strategic insights.
    
    Contains LLM-generated strategic analysis of the brand.
    """
    
    market_position: MarketPosition = Field(
        ...,
        description="Brand's market position classification",
    )
    market_position_rationale: str = Field(
        ...,
        min_length=10,
        description="Explanation for market position classification",
    )
    swot: SWOTAnalysis = Field(
        default_factory=SWOTAnalysis,
        description="SWOT analysis",
    )
    competitive_advantages: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Key competitive advantages",
    )
    risk_factors: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Identified risk factors",
    )
    growth_recommendations: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Strategic growth recommendations (simple list)",
    )
    recommendations: list[Recommendation] = Field(
        default_factory=list,
        description="Detailed strategic recommendations",
    )
    competitor_insights: list[CompetitorInsight] = Field(
        default_factory=list,
        max_length=10,
        description="Insights about key competitors",
    )
    estimated_market_share: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Estimated market share percentage in category",
    )
    brand_health_score: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Overall brand health score (0-100)",
    )
    confidence_score: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Confidence in analysis (0-1)",
    )
    executive_summary: str = Field(
        ...,
        min_length=50,
        max_length=2000,
        description="Executive summary of the analysis",
    )


# =============================================================================
# Pipeline State Models
# =============================================================================

class StepResult(BaseModel):
    """Result of a single pipeline step."""
    
    step: PipelineStep = Field(..., description="Pipeline step identifier")
    status: AnalysisStatus = Field(..., description="Step execution status")
    started_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Step start timestamp",
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Step completion timestamp",
    )
    duration_ms: Optional[int] = Field(
        default=None,
        ge=0,
        description="Step duration in milliseconds",
    )
    data: Optional[dict[str, Any]] = Field(
        default=None,
        description="Step output data",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if step failed",
    )
    retries: int = Field(default=0, ge=0, description="Number of retry attempts")
    
    @field_serializer("started_at", "completed_at")
    def serialize_datetime(self, value: Optional[datetime]) -> Optional[str]:
        """Serialize datetime to ISO format."""
        return value.isoformat() + "Z" if value else None


class PipelineState(BaseModel):
    """Current state of pipeline execution."""
    
    run_id: UUID = Field(
        default_factory=uuid4,
        description="Unique pipeline run identifier",
    )
    current_step: PipelineStep = Field(
        default=PipelineStep.INITIALIZATION,
        description="Current execution step",
    )
    status: AnalysisStatus = Field(
        default=AnalysisStatus.PENDING,
        description="Overall pipeline status",
    )
    brand_input: Optional[BrandInput] = Field(
        default=None,
        description="Input brand data",
    )
    extraction_result: Optional[ExtractionResult] = Field(
        default=None,
        description="Step 1: Extraction result",
    )
    strategic_insight: Optional[StrategicInsight] = Field(
        default=None,
        description="Step 2: Strategic analysis result",
    )
    step_results: list[StepResult] = Field(
        default_factory=list,
        description="Results from completed steps",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Accumulated error messages",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional pipeline metadata",
    )


# =============================================================================
# Report Models
# =============================================================================

class ReportSection(BaseModel):
    """Individual report section."""
    
    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Section content (markdown)")
    order: int = Field(default=0, description="Section display order")
    subsections: list["ReportSection"] = Field(
        default_factory=list,
        description="Nested subsections",
    )


class FinalReport(TimestampMixin):
    """
    Complete report structure - Final pipeline output.
    
    Contains all analysis results in a presentable format.
    """
    
    report_id: UUID = Field(
        default_factory=uuid4,
        description="Unique report identifier",
    )
    title: str = Field(
        ...,
        max_length=200,
        description="Report title",
        examples=["Brand Intelligence Report: Patagonia"],
    )
    brand_name: str = Field(..., description="Analyzed brand name")
    domain: str = Field(..., description="Brand domain")
    
    # Core data
    extraction_result: ExtractionResult = Field(
        ...,
        description="Complete extraction data",
    )
    strategic_insight: StrategicInsight = Field(
        ...,
        description="Strategic analysis",
    )
    
    # Report metadata
    report_format: str = Field(
        default="json",
        description="Report output format",
        examples=["json", "markdown", "html"],
    )
    generated_by: str = Field(
        default="Amazon Brand Intelligence Pipeline",
        description="Report generator identifier",
    )
    pipeline_version: str = Field(
        default="1.0.0",
        description="Pipeline version used",
    )
    
    # Execution info
    pipeline_run_id: Optional[UUID] = Field(
        default=None,
        description="Associated pipeline run ID",
    )
    total_duration_ms: int = Field(
        default=0,
        ge=0,
        description="Total pipeline execution time",
    )
    step_results: list[StepResult] = Field(
        default_factory=list,
        description="All pipeline step results",
    )
    
    # Content sections
    sections: list[ReportSection] = Field(
        default_factory=list,
        description="Report sections for rendering",
    )
    
    def generate_executive_summary(self) -> str:
        """Generate executive summary from analysis data."""
        insight = self.strategic_insight
        extraction = self.extraction_result
        
        # Handle market_position which may be string or enum
        market_pos = insight.market_position
        if hasattr(market_pos, 'value'):
            market_pos = market_pos.value
        
        # Handle confidence which may be string or enum
        confidence = extraction.amazon_presence.confidence
        if hasattr(confidence, 'value'):
            confidence = confidence.value
        
        # Handle optional price_range
        if extraction.price_range:
            price_range_str = f"${extraction.price_range[0]:.2f} - ${extraction.price_range[1]:.2f}"
        else:
            price_range_str = "N/A"
        
        return f"""
## Executive Summary

**{self.brand_name}** ({self.domain}) analysis completed on {self.created_at.strftime('%Y-%m-%d')}.

### Key Findings
- **Market Position:** {str(market_pos).title()}
- **Amazon Presence:** {'✓ Confirmed' if extraction.amazon_presence.found else '✗ Not Found'} ({confidence} confidence)
- **Products Found:** {extraction.estimated_product_count}
- **Average Rating:** {extraction.average_rating or 'N/A'}/5.0
- **Brand Health Score:** {insight.brand_health_score}/100

### Quick Stats
| Metric | Value |
|--------|-------|
| Total Reviews | {extraction.total_reviews:,} |
| Price Range | {price_range_str} |
| Competitors | {len(extraction.competitors_found)} identified |

{insight.executive_summary}
"""


# =============================================================================
# Error Models
# =============================================================================

class ErrorDetail(BaseModel):
    """Detailed error information."""
    
    field: Optional[str] = Field(
        default=None,
        description="Field that caused the error",
    )
    message: str = Field(..., description="Error message")
    code: Optional[str] = Field(
        default=None,
        description="Error code for programmatic handling",
    )


class ErrorResponse(BaseModel):
    """
    Standardized error response structure.
    
    Provides consistent error handling across the pipeline.
    
    Example:
        >>> error = ErrorResponse(
        ...     error_type=ErrorType.VALIDATION_ERROR,
        ...     message="Invalid domain format",
        ...     details=[ErrorDetail(field="domain", message="Must be valid TLD")]
        ... )
    """
    
    error_id: UUID = Field(
        default_factory=uuid4,
        description="Unique error identifier for tracking",
    )
    error_type: ErrorType = Field(..., description="Error classification")
    message: str = Field(..., description="Human-readable error message")
    details: list[ErrorDetail] = Field(
        default_factory=list,
        description="Detailed error information",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the error occurred",
    )
    request_id: Optional[UUID] = Field(
        default=None,
        description="Associated request/run ID",
    )
    recoverable: bool = Field(
        default=False,
        description="Whether the error is recoverable",
    )
    retry_after: Optional[int] = Field(
        default=None,
        ge=0,
        description="Seconds to wait before retry (for rate limits)",
    )
    documentation_url: Optional[str] = Field(
        default=None,
        description="URL to relevant documentation",
    )
    
    @field_serializer("timestamp")
    def serialize_timestamp(self, value: datetime) -> str:
        """Serialize timestamp to ISO format."""
        return value.isoformat() + "Z"
    
    def to_dict_safe(self) -> dict[str, Any]:
        """Return error as dict, safe for logging (no sensitive data)."""
        return self.model_dump(exclude={"details"})


# =============================================================================
# Convenience Type Aliases
# =============================================================================

ProductList = list[AmazonProduct]
CategoryList = list[AmazonCategory]


# =============================================================================
# Export All Models
# =============================================================================

__all__ = [
    # Base
    "BaseModel",
    "TimestampMixin",
    
    # Enums
    "ConfidenceLevel",
    "MarketPosition",
    "AnalysisStatus",
    "PipelineStep",
    "AmazonCategory",
    "ErrorType",
    
    # Input
    "BrandInput",
    
    # Products
    "PriceInfo",
    "AmazonProduct",
    
    # Extraction (Step 1)
    "AmazonPresence",
    "ExtractionMetadata",
    "ExtractionResult",
    
    # Analysis (Step 2)
    "SWOTAnalysis",
    "CompetitorInsight",
    "StrategicInsight",
    
    # Pipeline
    "StepResult",
    "PipelineState",
    
    # Report
    "ReportSection",
    "FinalReport",
    
    # Errors
    "ErrorDetail",
    "ErrorResponse",
    
    # Validators
    "validate_domain",
    "validate_asin",
    "validate_amazon_url",
]
