"""
Brand data extraction component for the Amazon Intelligence Pipeline.

This module provides the BrandExtractor class which orchestrates Step 1 of the
pipeline: extracting structured brand presence data from Amazon based on a domain.

Features:
    - Domain validation and brand name extraction
    - Multi-query search strategy for comprehensive coverage
    - LLM-powered data structuring and validation
    - Confidence scoring with evidence-based reasoning
    - Product enrichment with detailed information
    - Comprehensive error handling and logging

Example:
    >>> extractor = BrandExtractor(search_service, llm_service)
    >>> result = await extractor.extract_brand_data("patagonia.com")
    >>> print(f"Found {len(result.top_products)} products with {result.amazon_presence.confidence} confidence")
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from urllib.parse import quote_plus

from pydantic import BaseModel, Field

from src.config.settings import Settings, get_settings
from src.models.schemas import (
    AmazonPresence,
    AmazonProduct,
    BrandInput,
    ConfidenceLevel,
    ExtractionResult,
    ExtractionMetadata,
)
from src.services.llm_service import ClaudeService, TaskType
from src.services.search_service import (
    SearchService,
    SearchResults,
    SearchResultItem,
    ProductDetails,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Data Models
# =============================================================================

class ExtractionStage(str, Enum):
    """Extraction pipeline stages."""
    DOMAIN_VALIDATION = "domain_validation"
    BRAND_EXTRACTION = "brand_extraction"
    AMAZON_SEARCH = "amazon_search"
    LLM_STRUCTURING = "llm_structuring"
    PRODUCT_ENRICHMENT = "product_enrichment"
    CONFIDENCE_SCORING = "confidence_scoring"
    CATEGORY_DETECTION = "category_detection"
    FINALIZATION = "finalization"


@dataclass
class StageResult:
    """Result of a pipeline stage."""
    stage: ExtractionStage
    success: bool
    duration_ms: int
    data: Any = None
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionMetrics:
    """Metrics collected during extraction."""
    total_duration_ms: int = 0
    search_queries: int = 0
    queries_executed: list[str] = field(default_factory=list)
    api_calls: int = 0
    tokens_used: int = 0
    products_found: int = 0
    products_enriched: int = 0
    cache_hits: int = 0
    retries: int = 0
    stages: list[StageResult] = field(default_factory=list)


class BrandNameVariation(BaseModel):
    """Brand name variations for search."""
    primary: str
    variations: list[str] = Field(default_factory=list)
    domain_based: str
    
    def all_variations(self) -> list[str]:
        """Get all unique variations."""
        return list(dict.fromkeys([self.primary, self.domain_based] + self.variations))


class SearchQuery(BaseModel):
    """Search query with metadata."""
    query: str
    priority: int = Field(ge=1, le=5, default=3)
    description: str = ""


class ProductExtractionPrompt(BaseModel):
    """Schema for LLM product extraction prompt."""
    products: list[dict[str, Any]]
    brand_name: str
    confidence_reasoning: str
    
    
# =============================================================================
# Domain Processing
# =============================================================================

class DomainProcessor:
    """Handles domain validation and brand name extraction."""
    
    # Common TLDs to strip
    TLDS = {
        ".com", ".net", ".org", ".io", ".co", ".ai", ".app", ".dev",
        ".us", ".uk", ".de", ".fr", ".jp", ".cn", ".in", ".au",
        ".com.au", ".co.uk", ".co.in", ".com.br",
    }
    
    # Common prefixes to strip
    PREFIXES = {"www.", "shop.", "store.", "buy.", "get."}
    
    # Stop words that don't help brand identification
    STOP_WORDS = {"the", "inc", "llc", "ltd", "corp", "company", "co"}
    
    # Common misspelling patterns
    TYPO_PATTERNS = [
        (r"(\w)\1+", r"\1"),  # Double letters: niike -> nike
        (r"([aeiou])([aeiou])+", r"\1"),  # Double vowels
    ]
    
    @classmethod
    def validate_domain(cls, domain: str) -> tuple[bool, Optional[str]]:
        """
        Validate domain format.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not domain:
            return False, "Domain cannot be empty"
        
        # Clean the domain
        domain = domain.strip().lower()
        
        # Remove protocol if present
        domain = re.sub(r'^https?://', '', domain)
        
        # Remove trailing slash and path
        domain = domain.split('/')[0]
        
        # Basic format validation
        domain_pattern = r'^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z]{2,})+$'
        if not re.match(domain_pattern, domain):
            return False, f"Invalid domain format: {domain}"
        
        # Check for common invalid patterns
        if '..' in domain:
            return False, "Domain contains consecutive dots"
        
        if domain.startswith('-') or domain.endswith('-'):
            return False, "Domain cannot start or end with hyphen"
        
        return True, None
    
    @classmethod
    def extract_brand_name(cls, domain: str) -> BrandNameVariation:
        """
        Extract brand name from domain.
        
        Args:
            domain: The domain to process
            
        Returns:
            BrandNameVariation with primary and alternative names
        """
        # Clean domain
        clean_domain = domain.strip().lower()
        clean_domain = re.sub(r'^https?://', '', clean_domain)
        clean_domain = clean_domain.split('/')[0]
        
        # Remove common prefixes
        for prefix in cls.PREFIXES:
            if clean_domain.startswith(prefix):
                clean_domain = clean_domain[len(prefix):]
                break
        
        # Remove TLD
        domain_base = clean_domain
        for tld in sorted(cls.TLDS, key=len, reverse=True):
            if clean_domain.endswith(tld):
                domain_base = clean_domain[:-len(tld)]
                break
        
        # Handle special cases
        # Convert hyphens to spaces for compound names
        brand_with_spaces = domain_base.replace('-', ' ')
        
        # Title case for display
        primary_name = brand_with_spaces.title()
        
        # Generate variations
        variations = []
        
        # Without spaces/hyphens
        if ' ' in brand_with_spaces:
            variations.append(brand_with_spaces.replace(' ', ''))
        
        # With different casing
        variations.append(brand_with_spaces.upper())
        variations.append(brand_with_spaces.lower())
        
        # Remove stop words for cleaner version
        words = brand_with_spaces.split()
        clean_words = [w for w in words if w.lower() not in cls.STOP_WORDS]
        if clean_words and clean_words != words:
            variations.append(' '.join(clean_words).title())
        
        # Handle camelCase or compound names (e.g., GoPro -> Go Pro)
        camel_split = re.sub(r'([a-z])([A-Z])', r'\1 \2', primary_name)
        if camel_split != primary_name:
            variations.append(camel_split)
        
        # Remove duplicates while preserving order
        unique_variations = list(dict.fromkeys(variations))
        
        return BrandNameVariation(
            primary=primary_name,
            variations=unique_variations,
            domain_based=domain_base,
        )
    
    @classmethod
    def generate_search_queries(cls, brand: BrandNameVariation) -> list[SearchQuery]:
        """
        Generate multi-query search strategy.
        
        Args:
            brand: Brand name variations
            
        Returns:
            List of prioritized search queries
        """
        queries = []
        primary = brand.primary
        
        # Query 1: Exact brand name search (highest priority)
        queries.append(SearchQuery(
            query=f'"{primary}"',
            priority=1,
            description="Exact brand name match",
        ))
        
        # Query 2: Brand + amazon store
        queries.append(SearchQuery(
            query=f"{primary} amazon store",
            priority=2,
            description="Brand official store search",
        ))
        
        # Query 3: Brand products
        queries.append(SearchQuery(
            query=f"{primary} products",
            priority=2,
            description="Brand products search",
        ))
        
        # Query 4: Site-specific search
        queries.append(SearchQuery(
            query=f"site:amazon.com {primary}",
            priority=3,
            description="Site-specific brand search",
        ))
        
        # Query 5: Domain-based search (if different from primary)
        if brand.domain_based != primary.lower():
            queries.append(SearchQuery(
                query=brand.domain_based,
                priority=4,
                description="Domain-based search",
            ))
        
        # Query 6: Variations search
        for variation in brand.variations[:2]:  # Only top 2 variations
            if variation.lower() != primary.lower():
                queries.append(SearchQuery(
                    query=variation,
                    priority=5,
                    description=f"Variation: {variation}",
                ))
        
        # Sort by priority
        queries.sort(key=lambda q: q.priority)
        
        return queries


# =============================================================================
# Confidence Scoring
# =============================================================================

class ConfidenceScorer:
    """Calculates confidence scores for brand presence."""
    
    # Scoring weights
    WEIGHTS = {
        "official_store": 0.30,
        "product_count": 0.25,
        "brand_match": 0.20,
        "rating_quality": 0.15,
        "review_volume": 0.10,
    }
    
    # Thresholds
    THRESHOLDS = {
        "high": 0.75,
        "medium": 0.45,
        "low": 0.20,
    }
    
    @classmethod
    def calculate_score(
        cls,
        products: list[AmazonProduct],
        brand_name: str,
        has_official_store: bool = False,
        verified_seller: bool = False,
    ) -> tuple[ConfidenceLevel, float, list[str]]:
        """
        Calculate confidence score for brand presence.
        
        Args:
            products: List of found products
            brand_name: Brand name to match
            has_official_store: Whether official store was found
            verified_seller: Whether verified seller was found
            
        Returns:
            Tuple of (confidence_level, score, evidence)
        """
        evidence = []
        scores = {}
        
        # 1. Official store score
        if has_official_store:
            scores["official_store"] = 1.0
            evidence.append("Official Amazon store found")
        elif verified_seller:
            scores["official_store"] = 0.7
            evidence.append("Verified seller presence detected")
        else:
            scores["official_store"] = 0.0
        
        # 2. Product count score
        product_count = len(products)
        if product_count >= 20:
            scores["product_count"] = 1.0
            evidence.append(f"Strong product presence: {product_count} products")
        elif product_count >= 10:
            scores["product_count"] = 0.8
            evidence.append(f"Good product presence: {product_count} products")
        elif product_count >= 5:
            scores["product_count"] = 0.6
            evidence.append(f"Moderate product presence: {product_count} products")
        elif product_count >= 2:
            scores["product_count"] = 0.4
            evidence.append(f"Limited product presence: {product_count} products")
        elif product_count == 1:
            scores["product_count"] = 0.2
            evidence.append("Minimal product presence: 1 product")
        else:
            scores["product_count"] = 0.0
            evidence.append("No products found")
        
        # 3. Brand match score
        brand_lower = brand_name.lower()
        brand_matches = sum(
            1 for p in products
            if p.brand and brand_lower in p.brand.lower()
        )
        
        if product_count > 0:
            match_ratio = brand_matches / product_count
            scores["brand_match"] = match_ratio
            if match_ratio > 0.8:
                evidence.append(f"Strong brand match: {brand_matches}/{product_count} products")
            elif match_ratio > 0.5:
                evidence.append(f"Moderate brand match: {brand_matches}/{product_count} products")
        else:
            scores["brand_match"] = 0.0
        
        # 4. Rating quality score
        ratings = [p.rating for p in products if p.rating]
        if ratings:
            avg_rating = sum(ratings) / len(ratings)
            scores["rating_quality"] = avg_rating / 5.0
            evidence.append(f"Average rating: {avg_rating:.1f}/5.0")
        else:
            scores["rating_quality"] = 0.5  # Neutral if no ratings
        
        # 5. Review volume score
        total_reviews = sum(p.review_count for p in products)
        if total_reviews >= 10000:
            scores["review_volume"] = 1.0
            evidence.append(f"High review volume: {total_reviews:,} reviews")
        elif total_reviews >= 1000:
            scores["review_volume"] = 0.7
            evidence.append(f"Good review volume: {total_reviews:,} reviews")
        elif total_reviews >= 100:
            scores["review_volume"] = 0.4
            evidence.append(f"Moderate review volume: {total_reviews:,} reviews")
        else:
            scores["review_volume"] = 0.2
            evidence.append(f"Low review volume: {total_reviews:,} reviews")
        
        # Calculate weighted score
        final_score = sum(
            scores.get(key, 0) * weight
            for key, weight in cls.WEIGHTS.items()
        )
        
        # Determine confidence level
        if final_score >= cls.THRESHOLDS["high"]:
            level = ConfidenceLevel.HIGH
        elif final_score >= cls.THRESHOLDS["medium"]:
            level = ConfidenceLevel.MEDIUM
        elif final_score >= cls.THRESHOLDS["low"]:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.LOW
        
        return level, round(final_score, 3), evidence


# =============================================================================
# Category Detection
# =============================================================================

class CategoryDetector:
    """Detects and normalizes product categories."""
    
    # Amazon category taxonomy mapping
    CATEGORY_MAPPING = {
        # Apparel
        "clothing": "Clothing, Shoes & Jewelry",
        "shoes": "Clothing, Shoes & Jewelry",
        "jewelry": "Clothing, Shoes & Jewelry",
        "fashion": "Clothing, Shoes & Jewelry",
        "apparel": "Clothing, Shoes & Jewelry",
        
        # Electronics
        "electronics": "Electronics",
        "computers": "Electronics",
        "phones": "Electronics",
        "tablets": "Electronics",
        "cameras": "Electronics",
        "audio": "Electronics",
        
        # Home
        "home": "Home & Kitchen",
        "kitchen": "Home & Kitchen",
        "furniture": "Home & Kitchen",
        "garden": "Patio, Lawn & Garden",
        "outdoor": "Patio, Lawn & Garden",
        
        # Sports
        "sports": "Sports & Outdoors",
        "fitness": "Sports & Outdoors",
        "exercise": "Sports & Outdoors",
        
        # Beauty
        "beauty": "Beauty & Personal Care",
        "skincare": "Beauty & Personal Care",
        "makeup": "Beauty & Personal Care",
        "cosmetics": "Beauty & Personal Care",
        
        # Health
        "health": "Health & Household",
        "vitamins": "Health & Household",
        "supplements": "Health & Household",
        
        # Toys
        "toys": "Toys & Games",
        "games": "Toys & Games",
        
        # Automotive
        "automotive": "Automotive",
        "car": "Automotive",
        "vehicle": "Automotive",
        
        # Pet
        "pet": "Pet Supplies",
        "dog": "Pet Supplies",
        "cat": "Pet Supplies",
        
        # Baby
        "baby": "Baby",
        "infant": "Baby",
        "toddler": "Baby",
        
        # Grocery
        "grocery": "Grocery & Gourmet Food",
        "food": "Grocery & Gourmet Food",
        "snacks": "Grocery & Gourmet Food",
        
        # Books
        "books": "Books",
        "kindle": "Kindle Store",
        
        # Tools
        "tools": "Tools & Home Improvement",
        "hardware": "Tools & Home Improvement",
    }
    
    @classmethod
    def detect_primary_category(
        cls,
        products: list[AmazonProduct],
    ) -> tuple[Optional[str], dict[str, int]]:
        """
        Detect the primary category from products.
        
        Args:
            products: List of products to analyze
            
        Returns:
            Tuple of (primary_category, category_distribution)
        """
        category_counts: dict[str, int] = {}
        
        for product in products:
            if product.category:
                # Normalize category
                normalized = cls._normalize_category(product.category)
                category_counts[normalized] = category_counts.get(normalized, 0) + 1
        
        if not category_counts:
            return None, {}
        
        # Find the most common category
        primary = max(category_counts.items(), key=lambda x: x[1])[0]
        
        return primary, category_counts
    
    @classmethod
    def _normalize_category(cls, category: str) -> str:
        """Normalize a category string to standard taxonomy."""
        category_lower = category.lower()
        
        # Check for keyword matches
        for keyword, normalized in cls.CATEGORY_MAPPING.items():
            if keyword in category_lower:
                return normalized
        
        # If no match, return the original category cleaned up
        return category.strip().title()


# =============================================================================
# LLM Extraction Prompts
# =============================================================================

class ExtractionPrompts:
    """Prompt templates for LLM-powered extraction."""
    
    PRODUCT_EXTRACTION = """Analyze these Amazon search results and extract structured product information.

Brand Name: {brand_name}
Domain: {domain}

Search Results:
{search_results}

Extract the following information for each product that appears to be from the brand "{brand_name}":

For each product, provide:
1. title: The full product title
2. asin: The Amazon Standard Identification Number (10 characters, starts with B usually)
3. price: The current price (number only)
4. rating: The star rating (0-5)
5. review_count: Number of reviews (number only)
6. brand: The brand name as shown
7. is_prime: Boolean if Prime eligible
8. is_sponsored: Boolean if a sponsored listing
9. is_amazon_choice: Boolean if has Amazon's Choice badge
10. is_best_seller: Boolean if has Best Seller badge
11. category: The product category
12. rank: The position in search results

Also provide:
- confidence_reasoning: Explain why these products match or don't match the brand
- has_official_store: Boolean if an official brand store was detected
- verified_seller: Boolean if the brand appears as a verified seller

Return valid JSON matching this structure:
{{
    "products": [...],
    "confidence_reasoning": "...",
    "has_official_store": false,
    "verified_seller": false
}}

Important:
- Only include products that genuinely appear to be from or related to {brand_name}
- Exclude third-party sellers or knockoffs
- If unsure, include with a note in the product data
- Extract actual data; don't fabricate information"""

    CATEGORY_VALIDATION = """Given these product categories from an Amazon brand search, determine the primary category.

Brand: {brand_name}
Categories found: {categories}

Determine:
1. The primary product category for this brand
2. Whether the brand spans multiple categories
3. The confidence level (high/medium/low) in this categorization

Return JSON:
{{
    "primary_category": "...",
    "secondary_categories": [...],
    "is_multi_category": false,
    "confidence": "high"
}}"""

    AMBIGUITY_RESOLUTION = """There are multiple products/brands that could match "{brand_name}" from domain {domain}.

Candidates:
{candidates}

Based on the domain {domain}, determine:
1. Which products are most likely from the actual brand
2. Which are third-party or resellers
3. Which are unrelated/coincidental matches

Return JSON:
{{
    "likely_brand_products": [...],
    "third_party_products": [...],
    "unrelated_products": [...],
    "reasoning": "..."
}}"""


# =============================================================================
# Main BrandExtractor Class
# =============================================================================

class BrandExtractor:
    """
    Orchestrates brand data extraction from Amazon.
    
    This class implements Step 1 of the brand intelligence pipeline,
    combining search services and LLM analysis to extract structured
    brand presence data from Amazon.
    
    Attributes:
        search_service: Service for Amazon search operations
        llm_service: Service for LLM-powered analysis
        settings: Application settings
        
    Example:
        >>> async with SearchService() as search, ClaudeService() as llm:
        ...     extractor = BrandExtractor(search, llm)
        ...     result = await extractor.extract_brand_data("nike.com")
        ...     print(f"Found {len(result.top_products)} products")
    """
    
    def __init__(
        self,
        search_service: SearchService,
        llm_service: ClaudeService,
        settings: Optional[Settings] = None,
        max_products: int = 50,
        enrich_products: bool = True,
        parallel_enrichment: int = 5,
    ):
        """
        Initialize the BrandExtractor.
        
        Args:
            search_service: Initialized SearchService instance
            llm_service: Initialized ClaudeService instance
            settings: Optional application settings
            max_products: Maximum products to extract
            enrich_products: Whether to fetch detailed product info
            parallel_enrichment: Max concurrent enrichment requests
        """
        self.search_service = search_service
        self.llm_service = llm_service
        self.settings = settings or get_settings()
        self.max_products = max_products
        self.enrich_products = enrich_products
        self.parallel_enrichment = parallel_enrichment
        
        # Processing components
        self.domain_processor = DomainProcessor()
        self.confidence_scorer = ConfidenceScorer()
        self.category_detector = CategoryDetector()
        
        # Metrics tracking
        self._metrics: Optional[ExtractionMetrics] = None
    
    async def extract_brand_data(self, domain: str) -> ExtractionResult:
        """
        Main extraction pipeline.
        
        Args:
            domain: The brand's domain (e.g., "nike.com")
            
        Returns:
            ExtractionResult with structured brand presence data
            
        Raises:
            ValueError: If domain is invalid
            ExtractionError: If extraction fails
        """
        self._metrics = ExtractionMetrics()
        start_time = time.time()
        
        logger.info("Starting brand extraction", domain=domain)
        
        try:
            # Stage 1: Validate domain
            validated_domain = await self._validate_domain(domain)
            
            # Stage 2: Extract brand name
            brand = await self._extract_brand_name(validated_domain)
            
            # Stage 3: Search Amazon
            search_results = await self._search_amazon(brand, validated_domain)
            
            # Stage 4: Structure with LLM
            structured_data = await self._structure_with_llm(
                search_results, brand, validated_domain
            )
            
            # Stage 5: Enrich products
            enriched_products = await self._enrich_products(structured_data)
            
            # Stage 6: Calculate confidence
            confidence_result = await self._calculate_confidence(
                enriched_products, brand, structured_data
            )
            
            # Stage 7: Detect categories
            category_result = await self._detect_categories(enriched_products, brand)
            
            # Stage 8: Finalize result
            result = await self._finalize_result(
                domain=validated_domain,
                brand=brand,
                products=enriched_products,
                confidence=confidence_result,
                categories=category_result,
            )
            
            self._metrics.total_duration_ms = int((time.time() - start_time) * 1000)
            
            logger.info(
                "Brand extraction completed",
                domain=domain,
                products_found=len(result.top_products),
                confidence=result.amazon_presence.confidence,
                duration_ms=self._metrics.total_duration_ms,
            )
            
            return result
            
        except Exception as e:
            logger.error("Brand extraction failed", domain=domain, error=str(e))
            raise
    
    async def _validate_domain(self, domain: str) -> str:
        """Stage 1: Validate and clean domain."""
        start = time.time()
        
        is_valid, error = DomainProcessor.validate_domain(domain)
        
        if not is_valid:
            self._record_stage(
                ExtractionStage.DOMAIN_VALIDATION,
                success=False,
                duration_ms=int((time.time() - start) * 1000),
                error=error,
            )
            raise ValueError(f"Invalid domain: {error}")
        
        # Clean the domain
        clean_domain = domain.strip().lower()
        clean_domain = re.sub(r'^https?://', '', clean_domain)
        clean_domain = clean_domain.split('/')[0]
        
        self._record_stage(
            ExtractionStage.DOMAIN_VALIDATION,
            success=True,
            duration_ms=int((time.time() - start) * 1000),
            data={"original": domain, "cleaned": clean_domain},
        )
        
        return clean_domain
    
    async def _extract_brand_name(self, domain: str) -> BrandNameVariation:
        """Stage 2: Extract brand name from domain."""
        start = time.time()
        
        brand = DomainProcessor.extract_brand_name(domain)
        queries = DomainProcessor.generate_search_queries(brand)
        
        self._record_stage(
            ExtractionStage.BRAND_EXTRACTION,
            success=True,
            duration_ms=int((time.time() - start) * 1000),
            data={
                "primary": brand.primary,
                "variations": brand.variations,
                "search_queries": len(queries),
            },
        )
        
        logger.info(
            "Brand name extracted",
            primary=brand.primary,
            variations=brand.variations[:3],
        )
        
        return brand
    
    async def _search_amazon(
        self,
        brand: BrandNameVariation,
        domain: str,
    ) -> list[SearchResults]:
        """Stage 3: Execute multi-query search strategy."""
        start = time.time()
        
        queries = DomainProcessor.generate_search_queries(brand)
        all_results: list[SearchResults] = []
        
        # Execute queries in priority order
        for query in queries[:4]:  # Limit to top 4 queries
            try:
                logger.debug("Executing search query", query=query.query)
                
                results = await self.search_service.search_brand_on_amazon(
                    domain=domain,
                    brand_name=query.query.replace('"', ''),  # Remove quotes for API
                    limit=self.max_products,
                )
                
                all_results.append(results)
                if self._metrics:
                    self._metrics.search_queries += 1
                    self._metrics.queries_executed.append(query.query)
                
                # Early termination if high confidence results
                if len(results.items) >= 20:
                    logger.info(
                        "Found sufficient results, stopping early",
                        query=query.query,
                        results=len(results.items),
                    )
                    break
                    
            except Exception as e:
                logger.warning(
                    "Search query failed",
                    query=query.query,
                    error=str(e),
                )
                continue
        
        # Deduplicate results
        seen_asins: set[str] = set()
        unique_items: list[SearchResultItem] = []
        
        for result in all_results:
            for item in result.items:
                if item.asin and item.asin not in seen_asins:
                    seen_asins.add(item.asin)
                    unique_items.append(item)
        
        self._metrics.products_found = len(unique_items)
        
        self._record_stage(
            ExtractionStage.AMAZON_SEARCH,
            success=True,
            duration_ms=int((time.time() - start) * 1000),
            data={
                "queries_executed": len(all_results),
                "total_items": sum(len(r.items) for r in all_results),
                "unique_items": len(unique_items),
            },
        )
        
        logger.info(
            "Amazon search completed",
            queries=len(all_results),
            unique_products=len(unique_items),
        )
        
        return all_results
    
    async def _structure_with_llm(
        self,
        search_results: list[SearchResults],
        brand: BrandNameVariation,
        domain: str,
    ) -> dict[str, Any]:
        """Stage 4: Use LLM to structure raw search results."""
        start = time.time()
        
        # Combine all results for LLM processing
        all_items = []
        for result in search_results:
            for item in result.items:
                all_items.append({
                    "asin": item.asin,
                    "title": item.title,
                    "price": item.price,
                    "rating": item.rating,
                    "review_count": item.review_count,
                    "brand": item.brand,
                    "is_prime": item.is_prime,
                    "is_sponsored": item.is_sponsored,
                    "is_amazon_choice": item.is_amazon_choice,
                    "is_best_seller": item.is_best_seller,
                    "category": item.category,
                    "position": item.position,
                })
        
        if not all_items:
            self._record_stage(
                ExtractionStage.LLM_STRUCTURING,
                success=True,
                duration_ms=int((time.time() - start) * 1000),
                data={"products": [], "has_official_store": False, "verified_seller": False},
            )
            return {"products": [], "has_official_store": False, "verified_seller": False}
        
        # Prepare prompt
        prompt = ExtractionPrompts.PRODUCT_EXTRACTION.format(
            brand_name=brand.primary,
            domain=domain,
            search_results=all_items[:30],  # Limit to avoid token limits
        )
        
        try:
            # Use LLM to structure results
            class ExtractionSchema(BaseModel):
                products: list[dict[str, Any]]
                confidence_reasoning: str
                has_official_store: bool = False
                verified_seller: bool = False
            
            result = await self.llm_service.extract_structured_data(
                raw_text=prompt,
                schema=ExtractionSchema,
                additional_context=f"Extracting products for brand: {brand.primary}",
            )
            
            self._metrics.api_calls += 1
            
            structured = {
                "products": result.products,
                "has_official_store": result.has_official_store,
                "verified_seller": result.verified_seller,
                "confidence_reasoning": result.confidence_reasoning,
            }
            
            self._record_stage(
                ExtractionStage.LLM_STRUCTURING,
                success=True,
                duration_ms=int((time.time() - start) * 1000),
                data={
                    "products_structured": len(result.products),
                    "has_official_store": result.has_official_store,
                },
            )
            
            return structured
            
        except Exception as e:
            logger.warning(
                "LLM structuring failed, using raw results",
                error=str(e),
            )
            
            # Fallback to raw results
            self._record_stage(
                ExtractionStage.LLM_STRUCTURING,
                success=False,
                duration_ms=int((time.time() - start) * 1000),
                error=str(e),
            )
            
            return {
                "products": all_items,
                "has_official_store": False,
                "verified_seller": False,
            }
    
    async def _enrich_products(
        self,
        structured_data: dict[str, Any],
    ) -> list[AmazonProduct]:
        """Stage 5: Enrich products with detailed information."""
        start = time.time()
        
        raw_products = structured_data.get("products", [])
        enriched: list[AmazonProduct] = []
        
        # Convert to AmazonProduct models
        for i, p in enumerate(raw_products[:self.max_products]):
            try:
                product = AmazonProduct(
                    asin=p.get("asin") or f"UNKNOWN_{i}",
                    title=p.get("title") or "Unknown Product",
                    price=p.get("price"),
                    rating=p.get("rating"),
                    review_count=p.get("review_count") or 0,
                    brand=p.get("brand"),
                    is_prime=p.get("is_prime", False),
                    is_sponsored=p.get("is_sponsored", False),
                    is_amazon_choice=p.get("is_amazon_choice", False),
                    is_best_seller=p.get("is_best_seller", False),
                    category=p.get("category"),
                    rank=p.get("position") or p.get("rank"),
                )
                enriched.append(product)
            except Exception as e:
                logger.debug("Failed to create product model", error=str(e))
                continue
        
        # Optionally fetch detailed product info
        if self.enrich_products and enriched:
            # Filter products with valid ASINs for enrichment
            products_to_enrich = [
                p for p in enriched
                if p.asin and not p.asin.startswith("UNKNOWN")
            ][:self.parallel_enrichment]
            
            if products_to_enrich:
                async def enrich_one(product: AmazonProduct) -> Optional[AmazonProduct]:
                    try:
                        details = await self.search_service.get_product_details(product.asin)
                        if details:
                            # Update product with enriched data
                            return AmazonProduct(
                                asin=product.asin,
                                title=details.title or product.title,
                                price=details.price or product.price,
                                rating=details.rating or product.rating,
                                review_count=details.review_count or product.review_count,
                                brand=details.brand or product.brand,
                                is_prime=details.is_prime or product.is_prime,
                                is_sponsored=product.is_sponsored,
                                is_amazon_choice=details.is_amazon_choice or product.is_amazon_choice,
                                is_best_seller=details.is_best_seller or product.is_best_seller,
                                category=details.categories[0] if details.categories else product.category,
                                rank=product.rank,
                                image_url=details.images[0] if details.images else None,
                            )
                        return product
                    except Exception as e:
                        logger.debug("Product enrichment failed", asin=product.asin, error=str(e))
                        return product
                
                # Run enrichment in parallel with concurrency limit
                tasks = [enrich_one(p) for p in products_to_enrich]
                enriched_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Update enriched products
                enriched_dict = {p.asin: p for p in enriched}
                for result in enriched_results:
                    if isinstance(result, AmazonProduct):
                        enriched_dict[result.asin] = result
                        self._metrics.products_enriched += 1
                
                enriched = list(enriched_dict.values())
        
        self._record_stage(
            ExtractionStage.PRODUCT_ENRICHMENT,
            success=True,
            duration_ms=int((time.time() - start) * 1000),
            data={
                "products_processed": len(enriched),
                "products_enriched": self._metrics.products_enriched,
            },
        )
        
        return enriched
    
    async def _calculate_confidence(
        self,
        products: list[AmazonProduct],
        brand: BrandNameVariation,
        structured_data: dict[str, Any],
    ) -> tuple[ConfidenceLevel, float, list[str]]:
        """Stage 6: Calculate confidence scores."""
        start = time.time()
        
        level, score, evidence = ConfidenceScorer.calculate_score(
            products=products,
            brand_name=brand.primary,
            has_official_store=structured_data.get("has_official_store", False),
            verified_seller=structured_data.get("verified_seller", False),
        )
        
        self._record_stage(
            ExtractionStage.CONFIDENCE_SCORING,
            success=True,
            duration_ms=int((time.time() - start) * 1000),
            data={
                "level": level.value if hasattr(level, 'value') else str(level),
                "score": score,
                "evidence_count": len(evidence),
            },
        )
        
        logger.info(
            "Confidence calculated",
            level=level.value if hasattr(level, 'value') else str(level),
            score=score,
        )
        
        return level, score, evidence
    
    async def _detect_categories(
        self,
        products: list[AmazonProduct],
        brand: BrandNameVariation,
    ) -> tuple[Optional[str], dict[str, int]]:
        """Stage 7: Detect product categories."""
        start = time.time()
        
        primary_category, distribution = CategoryDetector.detect_primary_category(products)
        
        self._record_stage(
            ExtractionStage.CATEGORY_DETECTION,
            success=True,
            duration_ms=int((time.time() - start) * 1000),
            data={
                "primary_category": primary_category,
                "category_distribution": distribution,
            },
        )
        
        return primary_category, distribution
    
    async def _finalize_result(
        self,
        domain: str,
        brand: BrandNameVariation,
        products: list[AmazonProduct],
        confidence: tuple[ConfidenceLevel, float, list[str]],
        categories: tuple[Optional[str], dict[str, int]],
    ) -> ExtractionResult:
        """Stage 8: Finalize and return structured result."""
        start = time.time()
        
        level, score, evidence = confidence
        primary_category, category_distribution = categories
        
        # Create AmazonPresence
        amazon_presence = AmazonPresence(
            found=len(products) > 0,
            confidence=level,
            search_queries=self._metrics.search_queries if self._metrics else 1,
            evidence=evidence,
        )
        
        # Find competitors from products
        competitors = list(set(
            p.brand for p in products
            if p.brand and p.brand.lower() != brand.primary.lower()
        ))[:10]

        # Create ExtractionMetadata
        metadata = ExtractionMetadata(
            search_queries=self._metrics.queries_executed if self._metrics else [],
            extraction_duration_seconds=(time.time() - start) / 1000,
            api_calls_made=self._metrics.api_calls if self._metrics else 0,
            data_source="serpapi",  # Default to serpapi
            cache_hit=False
        )

        # Create ExtractionResult
        result = ExtractionResult(
            brand_name=brand.primary,
            domain=domain,
            amazon_presence=amazon_presence,
            top_products=products[:self.max_products],
            competitors_found=competitors,
            primary_category=primary_category,
            all_categories=list(category_distribution.keys()),
            estimated_product_count=len(products),
            metadata=metadata,
        )
        
        self._record_stage(
            ExtractionStage.FINALIZATION,
            success=True,
            duration_ms=int((time.time() - start) * 1000),
            data={
                "products": len(products),
                "competitors": len(competitors),
                "category": primary_category,
            },
        )
        
        return result
    
    def _record_stage(
        self,
        stage: ExtractionStage,
        success: bool,
        duration_ms: int,
        data: Any = None,
        error: Optional[str] = None,
    ) -> None:
        """Record a pipeline stage result."""
        if self._metrics:
            self._metrics.stages.append(StageResult(
                stage=stage,
                success=success,
                duration_ms=duration_ms,
                data=data,
                error=error,
            ))
    
    def get_metrics(self) -> Optional[ExtractionMetrics]:
        """Get extraction metrics from the last run."""
        return self._metrics


# =============================================================================
# Convenience Function
# =============================================================================

async def extract_brand_data(
    domain: str,
    search_service: Optional[SearchService] = None,
    llm_service: Optional[ClaudeService] = None,
) -> ExtractionResult:
    """
    Convenience function for one-off brand extraction.
    
    Args:
        domain: The brand's domain
        search_service: Optional pre-initialized search service
        llm_service: Optional pre-initialized LLM service
        
    Returns:
        ExtractionResult with brand presence data
    """
    async with SearchService() as search:
        async with ClaudeService() as llm:
            extractor = BrandExtractor(
                search_service=search_service or search,
                llm_service=llm_service or llm,
            )
            return await extractor.extract_brand_data(domain)


# =============================================================================
# Module Exports
# =============================================================================

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
]
