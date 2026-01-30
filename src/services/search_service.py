"""
Multi-provider search service for Amazon product data extraction.

This module provides a unified interface for searching Amazon marketplace data
through multiple providers including SerpAPI, Exa AI, and Perplexity.

Features:
    - Abstract SearchProvider base class for extensibility
    - Multiple provider implementations with automatic failover
    - Caching with configurable TTL
    - Rate limiting and throttling
    - Comprehensive error handling
    - Structured logging

Example:
    >>> service = SearchService()
    >>> async with service:
    ...     results = await service.search_brand_on_amazon("patagonia.com", "Patagonia")
    ...     products = await service.search_brand_products("Nike", limit=20)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional, Type
from urllib.parse import quote_plus

import httpx
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config.settings import Settings, get_settings
from src.models.schemas import AmazonProduct, AmazonPresence, ConfidenceLevel
from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Data Models for Search Results
# =============================================================================

class ProviderStatus(str, Enum):
    """Provider health status."""
    AVAILABLE = "available"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    DISABLED = "disabled"


class SearchResultItem(BaseModel):
    """Individual search result item."""
    asin: Optional[str] = None
    title: str
    url: str
    price: Optional[float] = None
    original_price: Optional[float] = None
    currency: str = "USD"
    rating: Optional[float] = None
    review_count: int = 0
    image_url: Optional[str] = None
    brand: Optional[str] = None
    is_prime: bool = False
    is_sponsored: bool = False
    is_amazon_choice: bool = False
    is_best_seller: bool = False
    position: Optional[int] = None
    category: Optional[str] = None
    seller: Optional[str] = None
    availability: Optional[str] = None
    features: list[str] = Field(default_factory=list)
    raw_data: Optional[dict[str, Any]] = None


class SearchResults(BaseModel):
    """Standardized search results from any provider."""
    query: str
    provider: str
    items: list[SearchResultItem] = Field(default_factory=list)
    total_results: int = 0
    page: int = 1
    has_more: bool = False
    search_duration_ms: int = 0
    cache_hit: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Metadata
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    related_searches: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class ProductDetails(BaseModel):
    """Detailed product information."""
    asin: str
    title: str
    url: str
    brand: Optional[str] = None
    price: Optional[float] = None
    original_price: Optional[float] = None
    currency: str = "USD"
    rating: Optional[float] = None
    rating_breakdown: dict[str, int] = Field(default_factory=dict)
    review_count: int = 0
    images: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)
    description: Optional[str] = None
    specifications: dict[str, str] = Field(default_factory=dict)
    variants: list[dict[str, Any]] = Field(default_factory=list)
    seller_info: dict[str, Any] = Field(default_factory=dict)
    is_prime: bool = False
    is_amazon_choice: bool = False
    is_best_seller: bool = False
    best_seller_rank: Optional[int] = None
    availability: Optional[str] = None
    delivery_info: Optional[str] = None
    provider: str = ""
    fetched_at: datetime = Field(default_factory=datetime.utcnow)


class PresenceConfidence(BaseModel):
    """Brand presence verification result."""
    is_present: bool
    confidence: ConfidenceLevel
    confidence_score: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    official_store_url: Optional[str] = None
    verified_seller: bool = False
    product_count: int = 0
    average_rating: Optional[float] = None
    total_reviews: int = 0
    top_categories: list[str] = Field(default_factory=list)
    competitors: list[str] = Field(default_factory=list)
    provider: str = ""


# =============================================================================
# Cache Implementation
# =============================================================================

@dataclass
class CacheEntry:
    """Cache entry with TTL tracking."""
    data: Any
    created_at: datetime
    ttl_seconds: int
    hits: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return datetime.utcnow() - self.created_at > timedelta(seconds=self.ttl_seconds)
    
    def get(self) -> Any:
        """Get cached data and increment hit counter."""
        self.hits += 1
        return self.data


class SearchCache:
    """Thread-safe LRU cache for search results."""
    
    def __init__(self, max_size: int = 500, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
    
    @staticmethod
    def _generate_key(provider: str, method: str, **kwargs) -> str:
        """Generate a unique cache key."""
        key_data = {"provider": provider, "method": method, **kwargs}
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]
    
    async def get(self, key: str) -> Optional[Any]:
        """Get cached value if exists and not expired."""
        async with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if not entry.is_expired():
                    return entry.get()
                else:
                    del self._cache[key]
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cache value with optional custom TTL."""
        async with self._lock:
            # Evict oldest entries if at capacity
            if len(self._cache) >= self.max_size:
                # Ensure at least one item is evicted
                evict_count = max(1, len(self._cache) // 4)
                oldest = sorted(
                    self._cache.items(),
                    key=lambda x: x[1].created_at
                )[:evict_count]
                for k, _ in oldest:
                    del self._cache[k]
            
            self._cache[key] = CacheEntry(
                data=value,
                created_at=datetime.utcnow(),
                ttl_seconds=ttl or self.default_ttl,
            )
    
    async def clear(self) -> None:
        """Clear all cached entries."""
        async with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_hits = sum(e.hits for e in self._cache.values())
        expired = sum(1 for e in self._cache.values() if e.is_expired())
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "total_hits": total_hits,
            "expired_entries": expired,
        }


# =============================================================================
# Rate Limiter
# =============================================================================

class RateLimiter:
    """Token bucket rate limiter for API requests."""
    
    def __init__(self, requests_per_second: float = 2.0):
        self.rate = requests_per_second
        self.tokens = requests_per_second
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> float:
        """
        Acquire permission to make a request.
        
        Returns wait time in seconds (0 if immediate).
        """
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return 0.0
            else:
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
                return wait_time


# =============================================================================
# Abstract Search Provider
# =============================================================================

class SearchProvider(ABC):
    """
    Abstract base class for search providers.
    
    All search providers must implement these core methods for
    standardized access to search functionality.
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        cache: Optional[SearchCache] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.settings = settings or get_settings()
        self.cache = cache or SearchCache()
        self.rate_limiter = rate_limiter or RateLimiter()
        self._client: Optional[httpx.AsyncClient] = None
        self._status = ProviderStatus.AVAILABLE
        self._last_error: Optional[str] = None
        self._request_count = 0
        self._error_count = 0
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier."""
        pass
    
    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """Check if provider is properly configured with API keys."""
        pass
    
    @property
    def status(self) -> ProviderStatus:
        """Current provider status."""
        return self._status
    
    async def connect(self) -> None:
        """Initialize HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=30.0,
                    write=10.0,
                    pool=5.0,
                ),
                limits=httpx.Limits(
                    max_keepalive_connections=5,
                    max_connections=10,
                ),
            )
    
    async def disconnect(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def __aenter__(self) -> "SearchProvider":
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()
    
    def _cache_key(self, method: str, **kwargs) -> str:
        """Generate cache key for a method call."""
        return SearchCache._generate_key(self.name, method, **kwargs)
    
    async def _with_rate_limit(self) -> None:
        """Apply rate limiting before request."""
        wait = await self.rate_limiter.acquire()
        if wait > 0:
            logger.debug(
                "Rate limit applied",
                provider=self.name,
                wait_seconds=f"{wait:.2f}",
            )
    
    def _update_status(self, success: bool, error: Optional[str] = None) -> None:
        """Update provider status based on request result."""
        self._request_count += 1
        if success:
            self._error_count = 0
            self._status = ProviderStatus.AVAILABLE
        else:
            self._error_count += 1
            self._last_error = error
            if self._error_count >= 3:
                if "rate limit" in (error or "").lower():
                    self._status = ProviderStatus.RATE_LIMITED
                else:
                    self._status = ProviderStatus.ERROR
    
    @abstractmethod
    async def search_amazon(
        self,
        query: str,
        category: Optional[str] = None,
        page: int = 1,
        limit: int = 20,
    ) -> SearchResults:
        """Search Amazon for products."""
        pass
    
    @abstractmethod
    async def get_product_details(self, asin: str) -> Optional[ProductDetails]:
        """Get detailed product information."""
        pass
    
    @abstractmethod
    async def verify_brand_presence(
        self,
        domain: str,
        brand_name: str,
    ) -> PresenceConfidence:
        """Verify brand presence on Amazon."""
        pass
    
    def get_stats(self) -> dict[str, Any]:
        """Get provider statistics."""
        return {
            "name": self.name,
            "status": self._status.value,
            "configured": self.is_configured,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "last_error": self._last_error,
        }


# =============================================================================
# SerpAPI Provider (Primary)
# =============================================================================

class SerpAPIProvider(SearchProvider):
    """
    SerpAPI provider for Amazon search.
    
    Primary provider using SerpAPI's Google and Amazon search endpoints.
    Supports product search, shopping results, and product details.
    """
    
    BASE_URL = "https://serpapi.com/search"
    
    @property
    def name(self) -> str:
        return "serpapi"
    
    @property
    def is_configured(self) -> bool:
        return self.settings.serpapi_api_key is not None
    
    def _get_api_key(self) -> str:
        """Get API key from settings."""
        if not self.settings.serpapi_api_key:
            raise ValueError("SerpAPI API key not configured")
        return self.settings.serpapi_api_key.get_secret_value()
    
    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def search_amazon(
        self,
        query: str,
        category: Optional[str] = None,
        page: int = 1,
        limit: int = 20,
    ) -> SearchResults:
        """
        Search Amazon products via SerpAPI.
        
        Args:
            query: Search query string
            category: Amazon category filter
            page: Page number for pagination
            limit: Maximum results per page
        
        Returns:
            SearchResults with matching products
        """
        # Check cache
        cache_key = self._cache_key(
            "search_amazon", query=query, category=category, page=page
        )
        cached = await self.cache.get(cache_key)
        if cached:
            cached.cache_hit = True
            return cached
        
        await self._with_rate_limit()
        
        if not self._client:
            await self.connect()
        
        start_time = time.time()
        
        params = {
            "engine": "amazon",
            "amazon_domain": "amazon.com",
            "k": query,
            "page": page,
            "api_key": self._get_api_key(),
        }
        
        if category:
            params["search_alias"] = category
        
        try:
            response = await self._client.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            items = self._parse_amazon_results(data.get("organic_results", []))
            
            result = SearchResults(
                query=query,
                provider=self.name,
                items=items[:limit],
                total_results=data.get("search_information", {}).get("total_results", len(items)),
                page=page,
                has_more=len(items) >= 20,
                search_duration_ms=int((time.time() - start_time) * 1000),
                related_searches=[
                    r.get("query", "") 
                    for r in data.get("related_searches", [])[:5]
                ],
            )
            
            self._update_status(True)
            await self.cache.set(cache_key, result)
            
            logger.info(
                "SerpAPI Amazon search completed",
                query=query,
                results_count=len(items),
                duration_ms=result.search_duration_ms,
            )
            
            return result
            
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            self._update_status(False, error_msg)
            logger.error("SerpAPI request failed", error=error_msg)
            
            if e.response.status_code == 429:
                raise RateLimitError(f"SerpAPI rate limit exceeded: {error_msg}")
            raise ProviderError(f"SerpAPI error: {error_msg}")
            
        except Exception as e:
            self._update_status(False, str(e))
            logger.error("SerpAPI unexpected error", error=str(e))
            raise ProviderError(f"SerpAPI error: {e}")
    
    async def search_google_shopping(
        self,
        query: str,
        limit: int = 20,
    ) -> SearchResults:
        """Search Google Shopping for Amazon products."""
        cache_key = self._cache_key("google_shopping", query=query)
        cached = await self.cache.get(cache_key)
        if cached:
            cached.cache_hit = True
            return cached
        
        await self._with_rate_limit()
        
        if not self._client:
            await self.connect()
        
        start_time = time.time()
        
        params = {
            "engine": "google_shopping",
            "q": f"site:amazon.com {query}",
            "api_key": self._get_api_key(),
        }
        
        try:
            response = await self._client.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            items = self._parse_shopping_results(data.get("shopping_results", []))
            
            result = SearchResults(
                query=query,
                provider=f"{self.name}_shopping",
                items=items[:limit],
                total_results=len(items),
                search_duration_ms=int((time.time() - start_time) * 1000),
            )
            
            await self.cache.set(cache_key, result)
            return result
            
        except Exception as e:
            logger.error("Google Shopping search failed", error=str(e))
            return SearchResults(
                query=query,
                provider=f"{self.name}_shopping",
                errors=[str(e)],
            )
    
    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def get_product_details(self, asin: str) -> Optional[ProductDetails]:
        """Get detailed product information from Amazon."""
        cache_key = self._cache_key("product_details", asin=asin)
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
        
        await self._with_rate_limit()
        
        if not self._client:
            await self.connect()
        
        params = {
            "engine": "amazon_product",
            "amazon_domain": "amazon.com",
            "asin": asin,
            "api_key": self._get_api_key(),
        }
        
        try:
            response = await self._client.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "error" in data:
                logger.warning("Product not found", asin=asin)
                return None
            
            product = self._parse_product_details(data)
            self._update_status(True)
            await self.cache.set(cache_key, product)
            
            return product
            
        except Exception as e:
            self._update_status(False, str(e))
            logger.error("Product details fetch failed", asin=asin, error=str(e))
            return None
    
    async def verify_brand_presence(
        self,
        domain: str,
        brand_name: str,
    ) -> PresenceConfidence:
        """Verify brand presence on Amazon via search."""
        # Search for the brand
        search_results = await self.search_amazon(f'"{brand_name}"', limit=50)
        
        evidence = []
        product_count = 0
        ratings = []
        reviews = []
        categories = []
        official_store = None
        verified_seller = False
        
        for item in search_results.items:
            # Check if product is from the brand
            if item.brand and brand_name.lower() in item.brand.lower():
                product_count += 1
                if item.rating:
                    ratings.append(item.rating)
                if item.review_count:
                    reviews.append(item.review_count)
                if item.category:
                    categories.append(item.category)
                
                # Look for official store indicator
                if item.seller and "official" in item.seller.lower():
                    official_store = f"https://www.amazon.com/stores/{quote_plus(brand_name)}"
                    verified_seller = True
        
        # Calculate confidence
        if product_count >= 10:
            confidence = ConfidenceLevel.HIGH
            confidence_score = min(0.95, 0.6 + (product_count / 100))
            evidence.append(f"Found {product_count} products from {brand_name}")
        elif product_count >= 3:
            confidence = ConfidenceLevel.MEDIUM
            confidence_score = 0.4 + (product_count / 30)
            evidence.append(f"Found {product_count} products from {brand_name}")
        elif product_count >= 1:
            confidence = ConfidenceLevel.LOW
            confidence_score = 0.2 + (product_count / 10)
            evidence.append(f"Found {product_count} product(s) from {brand_name}")
        else:
            confidence = ConfidenceLevel.LOW
            confidence_score = 0.1
            evidence.append(f"No products found for {brand_name}")
        
        if verified_seller:
            evidence.append("Official store detected")
            confidence_score = min(1.0, confidence_score + 0.1)
        
        avg_rating = sum(ratings) / len(ratings) if ratings else None
        total_reviews = sum(reviews)
        
        # Get unique top categories
        top_cats = list(dict.fromkeys(categories))[:5]
        
        return PresenceConfidence(
            is_present=product_count > 0,
            confidence=confidence,
            confidence_score=round(confidence_score, 2),
            evidence=evidence,
            official_store_url=official_store,
            verified_seller=verified_seller,
            product_count=product_count,
            average_rating=round(avg_rating, 2) if avg_rating else None,
            total_reviews=total_reviews,
            top_categories=top_cats,
            provider=self.name,
        )
    
    def _parse_amazon_results(self, results: list[dict]) -> list[SearchResultItem]:
        """Parse SerpAPI Amazon results into standardized format."""
        items = []
        for idx, r in enumerate(results):
            try:
                # Parse price
                price = None
                price_str = r.get("price", {})
                if isinstance(price_str, dict):
                    price_str = price_str.get("value") or price_str.get("raw", "")
                if price_str:
                    price = self._parse_price(str(price_str))
                
                item = SearchResultItem(
                    asin=r.get("asin"),
                    title=r.get("title", ""),
                    url=r.get("link", ""),
                    price=price,
                    rating=r.get("rating"),
                    review_count=self._parse_review_count(r.get("reviews", 0)),
                    image_url=r.get("thumbnail"),
                    brand=r.get("brand"),
                    is_prime=r.get("is_prime", False),
                    is_sponsored=r.get("is_sponsored", False),
                    is_amazon_choice=r.get("amazon_choice", False),
                    is_best_seller=r.get("best_seller", False),
                    position=idx + 1,
                    category=r.get("category"),
                    raw_data=r,
                )
                items.append(item)
            except Exception as e:
                logger.debug("Failed to parse result", error=str(e))
        
        return items
    
    def _parse_shopping_results(self, results: list[dict]) -> list[SearchResultItem]:
        """Parse Google Shopping results."""
        items = []
        for idx, r in enumerate(results):
            try:
                # Only include Amazon results
                source = r.get("source", "").lower()
                link = r.get("link", "")
                if "amazon" not in source and "amazon.com" not in link:
                    continue
                
                # Extract ASIN from URL
                asin = self._extract_asin(link)
                
                item = SearchResultItem(
                    asin=asin,
                    title=r.get("title", ""),
                    url=link,
                    price=self._parse_price(r.get("price", "")),
                    rating=r.get("rating"),
                    review_count=r.get("reviews", 0),
                    image_url=r.get("thumbnail"),
                    position=idx + 1,
                    seller=source,
                )
                items.append(item)
            except Exception as e:
                logger.debug("Failed to parse shopping result", error=str(e))
        
        return items
    
    def _parse_product_details(self, data: dict) -> ProductDetails:
        """Parse SerpAPI product details response."""
        product_info = data.get("product_results", {})
        
        # Parse rating breakdown
        rating_breakdown = {}
        for r in data.get("reviews", {}).get("ratings", []):
            stars = r.get("stars", 0)
            count = r.get("count", 0)
            rating_breakdown[f"{stars}_star"] = count
        
        return ProductDetails(
            asin=product_info.get("asin", ""),
            title=product_info.get("title", ""),
            url=product_info.get("link", ""),
            brand=product_info.get("brand"),
            price=self._parse_price(product_info.get("price", "")),
            original_price=self._parse_price(product_info.get("list_price", "")),
            rating=product_info.get("rating"),
            rating_breakdown=rating_breakdown,
            review_count=self._parse_review_count(product_info.get("reviews_total", 0)),
            images=[img.get("link", "") for img in data.get("images", [])],
            categories=[c.get("name", "") for c in data.get("categories", [])],
            features=product_info.get("feature_bullets", []),
            description=product_info.get("description"),
            specifications={
                s.get("name", ""): s.get("value", "")
                for s in data.get("specifications", [])
            },
            is_prime=product_info.get("is_prime", False),
            is_amazon_choice=product_info.get("amazon_choice", False),
            is_best_seller=product_info.get("best_seller", False),
            best_seller_rank=data.get("bestsellers_rank", [{}])[0].get("rank") if data.get("bestsellers_rank") else None,
            availability=product_info.get("availability", {}).get("raw"),
            delivery_info=product_info.get("delivery"),
            provider=self.name,
        )
    
    @staticmethod
    def _parse_price(price_str: str) -> Optional[float]:
        """Parse price string to float."""
        if not price_str:
            return None
        try:
            # Remove currency symbols and commas
            cleaned = re.sub(r"[^\d.]", "", str(price_str))
            return float(cleaned) if cleaned else None
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def _parse_review_count(count: Any) -> int:
        """Parse review count to integer."""
        if isinstance(count, int):
            return count
        if isinstance(count, str):
            try:
                cleaned = re.sub(r"[^\d]", "", count)
                return int(cleaned) if cleaned else 0
            except ValueError:
                return 0
        return 0
    
    @staticmethod
    def _extract_asin(url: str) -> Optional[str]:
        """Extract ASIN from Amazon URL."""
        patterns = [
            r"/dp/([A-Z0-9]{10})",
            r"/gp/product/([A-Z0-9]{10})",
            r"/product/([A-Z0-9]{10})",
            r"ASIN=([A-Z0-9]{10})",
        ]
        for pattern in patterns:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        return None


# =============================================================================
# Exa AI Provider (Fallback/Supplementary)
# =============================================================================

class ExaAIProvider(SearchProvider):
    """
    Exa AI provider for neural search and entity extraction.
    
    Provides supplementary information including:
    - Neural search for brand context
    - Entity extraction
    - Similar company discovery
    """
    
    BASE_URL = "https://api.exa.ai"
    
    @property
    def name(self) -> str:
        return "exa"
    
    @property
    def is_configured(self) -> bool:
        return self.settings.exa_api_key is not None
    
    def _get_headers(self) -> dict[str, str]:
        """Get API headers."""
        if not self.settings.exa_api_key:
            raise ValueError("Exa API key not configured")
        return {
            "Authorization": f"Bearer {self.settings.exa_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
    
    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def search_amazon(
        self,
        query: str,
        category: Optional[str] = None,
        page: int = 1,
        limit: int = 20,
    ) -> SearchResults:
        """Search Amazon via Exa's neural search."""
        cache_key = self._cache_key("search_amazon", query=query, page=page)
        cached = await self.cache.get(cache_key)
        if cached:
            cached.cache_hit = True
            return cached
        
        await self._with_rate_limit()
        
        if not self._client:
            await self.connect()
        
        start_time = time.time()
        
        payload = {
            "query": f"site:amazon.com {query}",
            "numResults": limit,
            "type": "neural",
            "useAutoprompt": True,
            "contents": {
                "text": True,
                "highlights": True,
            },
        }
        
        if category:
            payload["query"] = f"site:amazon.com {category} {query}"
        
        try:
            response = await self._client.post(
                f"{self.BASE_URL}/search",
                headers=self._get_headers(),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            
            items = self._parse_exa_results(data.get("results", []))
            
            result = SearchResults(
                query=query,
                provider=self.name,
                items=items,
                total_results=len(items),
                search_duration_ms=int((time.time() - start_time) * 1000),
            )
            
            self._update_status(True)
            await self.cache.set(cache_key, result)
            
            return result
            
        except Exception as e:
            self._update_status(False, str(e))
            logger.error("Exa search failed", error=str(e))
            raise ProviderError(f"Exa error: {e}")
    
    async def get_product_details(self, asin: str) -> Optional[ProductDetails]:
        """Get product details via Exa content extraction."""
        cache_key = self._cache_key("product_details", asin=asin)
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
        
        await self._with_rate_limit()
        
        if not self._client:
            await self.connect()
        
        url = f"https://www.amazon.com/dp/{asin}"
        
        payload = {
            "ids": [url],
            "text": True,
            "highlights": True,
        }
        
        try:
            response = await self._client.post(
                f"{self.BASE_URL}/contents",
                headers=self._get_headers(),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            
            results = data.get("results", [])
            if not results:
                return None
            
            content = results[0]
            
            # Parse basic details from content
            product = ProductDetails(
                asin=asin,
                title=content.get("title", ""),
                url=url,
                description=content.get("text", "")[:1000],
                provider=self.name,
            )
            
            await self.cache.set(cache_key, product)
            return product
            
        except Exception as e:
            logger.error("Exa content fetch failed", asin=asin, error=str(e))
            return None
    
    async def verify_brand_presence(
        self,
        domain: str,
        brand_name: str,
    ) -> PresenceConfidence:
        """Verify brand presence using neural search."""
        search_results = await self.search_amazon(brand_name, limit=30)
        
        product_count = len([
            item for item in search_results.items
            if "amazon.com" in item.url
        ])
        
        if product_count >= 10:
            confidence = ConfidenceLevel.HIGH
            confidence_score = 0.8
        elif product_count >= 5:
            confidence = ConfidenceLevel.MEDIUM
            confidence_score = 0.5
        else:
            confidence = ConfidenceLevel.LOW
            confidence_score = 0.2
        
        return PresenceConfidence(
            is_present=product_count > 0,
            confidence=confidence,
            confidence_score=confidence_score,
            evidence=[f"Neural search found {product_count} Amazon results"],
            product_count=product_count,
            provider=self.name,
        )
    
    async def find_similar_companies(
        self,
        domain: str,
        limit: int = 10,
    ) -> list[str]:
        """Find similar companies using Exa's neural search."""
        await self._with_rate_limit()
        
        if not self._client:
            await self.connect()
        
        payload = {
            "url": f"https://{domain}",
            "numResults": limit,
            "excludeSourceDomain": True,
        }
        
        try:
            response = await self._client.post(
                f"{self.BASE_URL}/findSimilar",
                headers=self._get_headers(),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            
            return [
                r.get("url", "").replace("https://", "").replace("www.", "").split("/")[0]
                for r in data.get("results", [])
            ]
            
        except Exception as e:
            logger.error("Similar companies search failed", error=str(e))
            return []
    
    def _parse_exa_results(self, results: list[dict]) -> list[SearchResultItem]:
        """Parse Exa search results."""
        items = []
        for idx, r in enumerate(results):
            try:
                url = r.get("url", "")
                
                # Only include Amazon results
                if "amazon.com" not in url:
                    continue
                
                asin = SerpAPIProvider._extract_asin(url)
                
                item = SearchResultItem(
                    asin=asin,
                    title=r.get("title", ""),
                    url=url,
                    position=idx + 1,
                    raw_data=r,
                )
                items.append(item)
            except Exception as e:
                logger.debug("Failed to parse Exa result", error=str(e))
        
        return items


# =============================================================================
# Perplexity Provider (Optional Enhancement)
# =============================================================================

class PerplexityProvider(SearchProvider):
    """
    Perplexity AI provider for enhanced context and analysis.
    
    Provides AI-powered insights and contextual information.
    Note: Requires Perplexity API key to be configured.
    """
    
    BASE_URL = "https://api.perplexity.ai"
    
    @property
    def name(self) -> str:
        return "perplexity"
    
    @property
    def is_configured(self) -> bool:
        # Check if perplexity API key exists in settings
        return hasattr(self.settings, "perplexity_api_key") and self.settings.perplexity_api_key is not None
    
    def _get_headers(self) -> dict[str, str]:
        """Get API headers."""
        if not self.is_configured:
            raise ValueError("Perplexity API key not configured")
        return {
            "Authorization": f"Bearer {self.settings.perplexity_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
    
    async def search_amazon(
        self,
        query: str,
        category: Optional[str] = None,
        page: int = 1,
        limit: int = 20,
    ) -> SearchResults:
        """
        Search via Perplexity's AI-powered search.
        
        Note: Returns contextual results, not raw product data.
        """
        if not self.is_configured:
            return SearchResults(
                query=query,
                provider=self.name,
                errors=["Perplexity API not configured"],
            )
        
        cache_key = self._cache_key("search", query=query)
        cached = await self.cache.get(cache_key)
        if cached:
            cached.cache_hit = True
            return cached
        
        await self._with_rate_limit()
        
        if not self._client:
            await self.connect()
        
        start_time = time.time()
        
        payload = {
            "model": "pplx-7b-online",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides information about products on Amazon.",
                },
                {
                    "role": "user",
                    "content": f"Find the top Amazon products for: {query}. Include product names, prices, and ratings if available.",
                },
            ],
        }
        
        try:
            response = await self._client.post(
                f"{self.BASE_URL}/chat/completions",
                headers=self._get_headers(),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            
            # Perplexity returns text, not structured data
            # Store as raw context for analysis
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            result = SearchResults(
                query=query,
                provider=self.name,
                search_duration_ms=int((time.time() - start_time) * 1000),
                related_searches=[content[:500]],  # Store context
            )
            
            await self.cache.set(cache_key, result)
            return result
            
        except Exception as e:
            logger.error("Perplexity search failed", error=str(e))
            return SearchResults(
                query=query,
                provider=self.name,
                errors=[str(e)],
            )
    
    async def get_product_details(self, asin: str) -> Optional[ProductDetails]:
        """Perplexity doesn't support direct product lookup."""
        return None
    
    async def verify_brand_presence(
        self,
        domain: str,
        brand_name: str,
    ) -> PresenceConfidence:
        """Verify brand presence using Perplexity AI."""
        if not self.is_configured:
            return PresenceConfidence(
                is_present=False,
                confidence=ConfidenceLevel.LOW,
                confidence_score=0.0,
                evidence=["Perplexity API not configured"],
                provider=self.name,
            )
        
        await self._with_rate_limit()
        
        if not self._client:
            await self.connect()
        
        payload = {
            "model": "pplx-7b-online",
            "messages": [
                {
                    "role": "user",
                    "content": f"Is {brand_name} ({domain}) selling products on Amazon? Provide a brief yes/no answer with evidence.",
                },
            ],
        }
        
        try:
            response = await self._client.post(
                f"{self.BASE_URL}/chat/completions",
                headers=self._get_headers(),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "").lower()
            
            is_present = "yes" in content[:100]
            
            return PresenceConfidence(
                is_present=is_present,
                confidence=ConfidenceLevel.MEDIUM,
                confidence_score=0.6 if is_present else 0.4,
                evidence=[content[:300]],
                provider=self.name,
            )
            
        except Exception as e:
            logger.error("Perplexity verification failed", error=str(e))
            return PresenceConfidence(
                is_present=False,
                confidence=ConfidenceLevel.LOW,
                confidence_score=0.0,
                evidence=[str(e)],
                provider=self.name,
            )


# =============================================================================
# Custom Exceptions
# =============================================================================

class ProviderError(Exception):
    """Base exception for provider errors."""
    pass


class RateLimitError(ProviderError):
    """Raised when rate limit is exceeded."""
    pass


class NoResultsError(ProviderError):
    """Raised when no results are found."""
    pass


class ConfigurationError(ProviderError):
    """Raised when provider is not properly configured."""
    pass


# =============================================================================
# Unified Search Service Orchestrator
# =============================================================================

class SearchService:
    """
    Unified search service orchestrating multiple providers.
    
    Features:
        - Automatic failover between providers
        - Result aggregation and deduplication
        - Caching across providers
        - Rate limiting and throttling
        - Comprehensive error handling
    
    Example:
        >>> async with SearchService() as service:
        ...     results = await service.search_brand_on_amazon("patagonia.com", "Patagonia")
        ...     products = await service.search_brand_products("Nike", limit=20)
        ...     presence = await service.verify_brand_presence("apple.com")
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        enable_cache: bool = True,
        cache_ttl: int = 3600,
    ):
        """
        Initialize the search service.
        
        Args:
            settings: Application settings
            enable_cache: Whether to enable response caching
            cache_ttl: Cache time-to-live in seconds
        """
        self.settings = settings or get_settings()
        self.cache = SearchCache(default_ttl=cache_ttl) if enable_cache else None
        
        # Initialize providers
        self._providers: list[SearchProvider] = []
        self._primary_provider: Optional[SearchProvider] = None
        
        # Shared rate limiter
        self._rate_limiter = RateLimiter(requests_per_second=2.0)
        
        self._initialized = False
    
    async def _init_providers(self) -> None:
        """Initialize and configure providers."""
        if self._initialized:
            return
        
        # SerpAPI (Primary)
        serpapi = SerpAPIProvider(
            settings=self.settings,
            cache=self.cache,
            rate_limiter=self._rate_limiter,
        )
        if serpapi.is_configured:
            await serpapi.connect()
            self._providers.append(serpapi)
            self._primary_provider = serpapi
            logger.info("SerpAPI provider initialized (primary)")
        
        # Exa AI (Fallback)
        exa = ExaAIProvider(
            settings=self.settings,
            cache=self.cache,
            rate_limiter=self._rate_limiter,
        )
        if exa.is_configured:
            await exa.connect()
            self._providers.append(exa)
            logger.info("Exa AI provider initialized (fallback)")
        
        # Perplexity (Optional)
        perplexity = PerplexityProvider(
            settings=self.settings,
            cache=self.cache,
            rate_limiter=self._rate_limiter,
        )
        if perplexity.is_configured:
            await perplexity.connect()
            self._providers.append(perplexity)
            logger.info("Perplexity provider initialized (optional)")
        
        if not self._providers:
            raise ConfigurationError(
                "No search providers configured. Please set at least one API key "
                "(SERPAPI_API_KEY, EXA_API_KEY, or PERPLEXITY_API_KEY)"
            )
        
        self._initialized = True
    
    async def __aenter__(self) -> "SearchService":
        await self._init_providers()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
    
    async def close(self) -> None:
        """Close all provider connections."""
        for provider in self._providers:
            await provider.disconnect()
        self._providers.clear()
        self._primary_provider = None
        self._initialized = False
    
    def _get_available_providers(self) -> list[SearchProvider]:
        """Get list of available providers sorted by priority."""
        available = [
            p for p in self._providers
            if p.status == ProviderStatus.AVAILABLE
        ]
        # Primary provider first
        if self._primary_provider in available:
            available.remove(self._primary_provider)
            available.insert(0, self._primary_provider)
        return available
    
    async def _try_providers(
        self,
        method_name: str,
        *args,
        **kwargs,
    ) -> Any:
        """Try method across providers with failover."""
        last_error = None
        
        for provider in self._get_available_providers():
            try:
                method = getattr(provider, method_name)
                result = await method(*args, **kwargs)
                
                # Check for valid result
                if result is not None:
                    return result
                    
            except RateLimitError as e:
                logger.warning(
                    "Provider rate limited, trying next",
                    provider=provider.name,
                    error=str(e),
                )
                last_error = e
                continue
                
            except ProviderError as e:
                logger.warning(
                    "Provider error, trying next",
                    provider=provider.name,
                    error=str(e),
                )
                last_error = e
                continue
                
            except Exception as e:
                logger.error(
                    "Unexpected provider error",
                    provider=provider.name,
                    error=str(e),
                )
                last_error = e
                continue
        
        # All providers failed
        raise ProviderError(f"All providers failed: {last_error}")
    
    # =========================================================================
    # Public API Methods
    # =========================================================================
    
    async def search_brand_on_amazon(
        self,
        domain: str,
        brand_name: str,
        limit: int = 50,
    ) -> SearchResults:
        """
        Search for a brand's products on Amazon.
        
        Args:
            domain: Brand's domain for context
            brand_name: Brand name to search
            limit: Maximum results to return
        
        Returns:
            SearchResults with matching products
        """
        await self._init_providers()
        
        # Build search query
        query = f'"{brand_name}"'
        
        return await self._try_providers(
            "search_amazon",
            query=query,
            limit=limit,
        )
    
    async def search_brand_products(
        self,
        brand_name: str,
        category: Optional[str] = None,
        limit: int = 10,
    ) -> list[AmazonProduct]:
        """
        Search for products from a specific brand.
        
        Args:
            brand_name: Brand name to search
            category: Optional Amazon category filter
            limit: Maximum products to return
        
        Returns:
            List of AmazonProduct models
        """
        await self._init_providers()
        
        results = await self._try_providers(
            "search_amazon",
            query=f'"{brand_name}"',
            category=category,
            limit=limit,
        )
        
        # Convert to AmazonProduct models
        products = []
        for item in results.items:
            if not item.asin:
                continue
            try:
                product = AmazonProduct(
                    asin=item.asin,
                    title=item.title,
                    price=item.price,
                    rating=item.rating,
                    review_count=item.review_count,
                    brand=item.brand or brand_name,
                    is_prime=item.is_prime,
                    is_sponsored=item.is_sponsored,
                    is_amazon_choice=item.is_amazon_choice,
                    is_best_seller=item.is_best_seller,
                    rank=item.position,
                    category=item.category,
                    image_url=item.image_url,
                )
                products.append(product)
            except Exception as e:
                logger.debug("Failed to create product model", error=str(e))
        
        return products
    
    async def get_product_details(self, asin: str) -> Optional[ProductDetails]:
        """
        Get detailed product information.
        
        Args:
            asin: Amazon Standard Identification Number
        
        Returns:
            ProductDetails or None if not found
        """
        await self._init_providers()
        
        return await self._try_providers(
            "get_product_details",
            asin=asin,
        )
    
    async def verify_brand_presence(
        self,
        domain: str,
        brand_name: Optional[str] = None,
    ) -> PresenceConfidence:
        """
        Verify brand presence on Amazon.
        
        Args:
            domain: Brand's domain
            brand_name: Brand name (extracted from domain if not provided)
        
        Returns:
            PresenceConfidence with verification results
        """
        await self._init_providers()
        
        # Extract brand name from domain if not provided
        if not brand_name:
            brand_name = domain.split(".")[0].replace("-", " ").title()
        
        return await self._try_providers(
            "verify_brand_presence",
            domain=domain,
            brand_name=brand_name,
        )
    
    async def search_with_all_providers(
        self,
        query: str,
        limit: int = 20,
    ) -> dict[str, SearchResults]:
        """
        Search using all available providers and aggregate results.
        
        Args:
            query: Search query
            limit: Results per provider
        
        Returns:
            Dict mapping provider name to results
        """
        await self._init_providers()
        
        results = {}
        
        async def search(provider: SearchProvider) -> tuple[str, SearchResults]:
            try:
                result = await provider.search_amazon(query, limit=limit)
                return provider.name, result
            except Exception as e:
                return provider.name, SearchResults(
                    query=query,
                    provider=provider.name,
                    errors=[str(e)],
                )
        
        # Run searches concurrently
        tasks = [search(p) for p in self._providers]
        completed = await asyncio.gather(*tasks)
        
        for name, result in completed:
            results[name] = result
        
        return results
    
    async def find_competitors(
        self,
        domain: str,
        brand_name: str,
        search_results: Optional[SearchResults] = None,
    ) -> list[str]:
        """
        Find competitor brands based on search results.
        
        Args:
            domain: Brand's domain
            brand_name: Brand name
            search_results: Optional pre-fetched search results
        
        Returns:
            List of competitor brand names
        """
        await self._init_providers()
        
        competitors = set()
        
        # Get competitors from search results
        if not search_results:
            search_results = await self.search_brand_on_amazon(domain, brand_name, limit=50)
        
        for item in search_results.items:
            if item.brand and item.brand.lower() != brand_name.lower():
                competitors.add(item.brand)
        
        # Try to get similar companies from Exa
        for provider in self._providers:
            if isinstance(provider, ExaAIProvider):
                try:
                    similar = await provider.find_similar_companies(domain, limit=10)
                    competitors.update(similar)
                except Exception as e:
                    logger.debug("Similar companies search failed", error=str(e))
        
        return list(competitors)[:20]
    
    def get_provider_stats(self) -> dict[str, Any]:
        """Get statistics for all providers."""
        return {
            "providers": [p.get_stats() for p in self._providers],
            "cache": self.cache.get_stats() if self.cache else None,
            "primary": self._primary_provider.name if self._primary_provider else None,
        }
    
    async def clear_cache(self) -> None:
        """Clear all cached data."""
        if self.cache:
            await self.cache.clear()
            logger.info("Search cache cleared")


# =============================================================================
# Convenience Functions
# =============================================================================

async def create_search_service(
    settings: Optional[Settings] = None,
) -> SearchService:
    """Create and initialize a SearchService."""
    service = SearchService(settings=settings)
    await service._init_providers()
    return service


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Service
    "SearchService",
    "create_search_service",
    # Providers
    "SearchProvider",
    "SerpAPIProvider",
    "ExaAIProvider",
    "PerplexityProvider",
    # Models
    "SearchResults",
    "SearchResultItem",
    "ProductDetails",
    "PresenceConfidence",
    "ProviderStatus",
    # Utilities
    "SearchCache",
    "RateLimiter",
    # Exceptions
    "ProviderError",
    "RateLimitError",
    "NoResultsError",
    "ConfigurationError",
]
