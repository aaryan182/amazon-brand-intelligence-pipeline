"""
Optimized Claude prompts for the Amazon Brand Intelligence Pipeline.

This module contains structured prompt templates for various LLM-powered
extraction and analysis tasks. All prompts are designed for Claude 3.5 Sonnet
with recommended temperature settings and best practices.

Prompt Categories:
    1. Brand Data Extraction - Extract structured data from search results
    2. Product Detail Enrichment - Enhance partial product information
    3. Category Classification - Determine primary product categories
    4. Confidence Assessment - Score brand presence confidence
    5. Validation & Retry - Handle extraction failures gracefully

Best Practices Applied:
    - XML tags for clear structure
    - Chain-of-thought reasoning
    - 2-3 few-shot examples per task
    - Explicit JSON schema specifications
    - Edge case handling in instructions
    - Temperature=0.3 for extraction tasks
"""

from dataclasses import dataclass
from typing import Any, Optional
import json


# =============================================================================
# Prompt Configuration
# =============================================================================

@dataclass
class PromptConfig:
    """Configuration for a prompt template."""
    name: str
    description: str
    recommended_temperature: float
    recommended_max_tokens: int
    model: str = "claude-sonnet-4-20250514"
    
    def __repr__(self) -> str:
        return f"PromptConfig({self.name}, temp={self.recommended_temperature})"


# =============================================================================
# PROMPT 1: Brand Data Extraction
# =============================================================================

BRAND_EXTRACTION_CONFIG = PromptConfig(
    name="brand_data_extraction",
    description="Extract structured Amazon brand presence data from search results",
    recommended_temperature=0.3,
    recommended_max_tokens=4096,
)

BRAND_EXTRACTION_SYSTEM = """You are an expert Amazon marketplace analyst specializing in brand presence detection and product data extraction. Your task is to analyze raw search results and extract structured information about a brand's Amazon presence.

<role>
You have deep expertise in:
- Amazon product catalog structure and ASIN formats
- Brand verification and official seller detection
- Price analysis and competitive positioning
- Review sentiment and rating interpretation
- Category taxonomy and product classification
</role>

<guidelines>
1. ACCURACY: Only extract information explicitly present in the data. Never fabricate ASINs, prices, or ratings.
2. COMPLETENESS: Extract all relevant products, even if some fields are missing.
3. CONFIDENCE: Provide honest assessment of data quality and brand match certainty.
4. FORMAT: Output strictly valid JSON matching the specified schema.
5. REASONING: Include brief reasoning for confidence assessments.
</guidelines>

<edge_cases>
- Missing prices: Set to null, note in confidence reasoning
- Ambiguous brand matches: Include with lower confidence, explain uncertainty
- Duplicate products: Keep the most complete version
- Third-party sellers: Distinguish from official brand presence
- Sponsored listings: Flag but include in analysis
</edge_cases>"""

BRAND_EXTRACTION_USER = """<task>
Extract structured Amazon presence data for the brand "{brand_name}" (domain: {domain}) from the following search results.
</task>

<search_results>
{search_results}
</search_results>

<instructions>
Analyze the search results step by step:

1. IDENTIFY BRAND PRODUCTS
   - Find products that genuinely belong to or are sold by "{brand_name}"
   - Distinguish official products from third-party resellers
   - Note any official Amazon store presence

2. EXTRACT PRODUCT DATA
   For each identified product, extract:
   - ASIN (10-character Amazon identifier, starts with 'B' typically)
   - Title (full product title as shown)
   - Price (current price in USD, null if unavailable)
   - Rating (0-5 stars, null if unavailable)
   - Review count (integer, 0 if unavailable)
   - Brand name as displayed
   - Category if identifiable
   - Prime eligibility, Best Seller, Amazon's Choice badges

3. ASSESS CONFIDENCE
   Consider:
   - How many products clearly match the brand?
   - Is there an official store presence?
   - Are prices and ratings consistent with a legitimate brand?
   - Any signs of counterfeits or unauthorized sellers?

4. OUTPUT STRUCTURED JSON
</instructions>

<examples>
Example 1 - Strong Brand Presence:
Input: Search results showing "Nike" with official store, 15+ products, verified seller badges
Output:
{{
  "products": [
    {{
      "asin": "B07XYZ1234",
      "title": "Nike Air Max 270 Men's Running Shoes",
      "price": 150.00,
      "rating": 4.5,
      "review_count": 12543,
      "brand": "Nike",
      "category": "Clothing, Shoes & Jewelry",
      "is_prime": true,
      "is_amazon_choice": true,
      "is_best_seller": false,
      "is_sponsored": false,
      "rank": 1
    }}
  ],
  "has_official_store": true,
  "verified_seller": true,
  "confidence_reasoning": "Strong brand presence confirmed: Official Nike store found, 15+ products with consistent branding, high review volumes, and Amazon's Choice badges on multiple items."
}}

Example 2 - Weak/Uncertain Presence:
Input: Search results for "SmallBrandXYZ" with 2 products, no official store
Output:
{{
  "products": [
    {{
      "asin": "B09ABC5678",
      "title": "SmallBrandXYZ Wireless Earbuds",
      "price": 29.99,
      "rating": 3.8,
      "review_count": 142,
      "brand": "SmallBrandXYZ",
      "category": "Electronics",
      "is_prime": false,
      "is_amazon_choice": false,
      "is_best_seller": false,
      "is_sponsored": true,
      "rank": 1
    }}
  ],
  "has_official_store": false,
  "verified_seller": false,
  "confidence_reasoning": "Limited brand presence: Only 2 products found, no official store, low review counts. Products appear to be from third-party sellers. Brand legitimacy uncertain."
}}

Example 3 - No Brand Found:
Input: Search results with no matching products for "NonExistentBrand123"
Output:
{{
  "products": [],
  "has_official_store": false,
  "verified_seller": false,
  "confidence_reasoning": "No brand presence detected: Search returned no products matching 'NonExistentBrand123'. The brand may not sell on Amazon or uses a different name."
}}
</examples>

<output_schema>
{{
  "products": [
    {{
      "asin": "string (10 chars, required)",
      "title": "string (required)",
      "price": "number or null",
      "rating": "number 0-5 or null",
      "review_count": "integer >= 0",
      "brand": "string or null",
      "category": "string or null",
      "is_prime": "boolean",
      "is_amazon_choice": "boolean",
      "is_best_seller": "boolean",
      "is_sponsored": "boolean",
      "rank": "integer position in results"
    }}
  ],
  "has_official_store": "boolean",
  "verified_seller": "boolean",
  "confidence_reasoning": "string explaining confidence assessment"
}}
</output_schema>

<output_format>
Respond with ONLY valid JSON matching the schema above. No markdown code blocks, no explanation text, just the JSON object.
</output_format>"""


# =============================================================================
# PROMPT 2: Product Detail Enrichment
# =============================================================================

PRODUCT_ENRICHMENT_CONFIG = PromptConfig(
    name="product_detail_enrichment",
    description="Enhance partial product information with additional context",
    recommended_temperature=0.2,
    recommended_max_tokens=2048,
)

PRODUCT_ENRICHMENT_SYSTEM = """You are a product data specialist focused on enriching and completing partial Amazon product information. Your role is to analyze incomplete product data and infer missing fields where reasonable, while clearly marking inferred vs. confirmed data.

<capabilities>
- Infer product categories from titles and descriptions
- Estimate price ranges based on product type
- Identify brand from product titles when not explicitly stated
- Detect product variants and relationships
- Normalize inconsistent data formats
</capabilities>

<rules>
1. Never fabricate specific values (exact prices, ASINs, review counts)
2. Mark all inferred data with confidence levels
3. Preserve original data when available
4. Flag data quality issues
5. Maintain JSON validity
</rules>"""

PRODUCT_ENRICHMENT_USER = """<task>
Enrich the following partial product data for brand "{brand_name}". Fill in missing fields where inferable, and assess data quality.
</task>

<partial_data>
{product_data}
</partial_data>

<enrichment_instructions>
For each product:
1. If category is missing, infer from title keywords
2. If brand is missing but inferable from title, add it
3. Identify any data quality issues (inconsistent formats, suspicious values)
4. Add enrichment metadata showing what was inferred vs. original

Do NOT:
- Invent specific prices, ratings, or review counts
- Create fake ASINs
- Assume Prime eligibility without evidence
</enrichment_instructions>

<examples>
Input:
{{"asin": "B07XYZ1234", "title": "Nike Air Max 270 Running Shoes Men's Size 10", "price": null}}

Output:
{{
  "asin": "B07XYZ1234",
  "title": "Nike Air Max 270 Running Shoes Men's Size 10",
  "price": null,
  "brand": "Nike",
  "category": "Clothing, Shoes & Jewelry",
  "enrichment_metadata": {{
    "brand_source": "inferred_from_title",
    "category_source": "inferred_from_title",
    "confidence": "high",
    "quality_issues": ["missing_price"]
  }}
}}
</examples>

<output_format>
Return a JSON object with:
- All original fields preserved
- Missing fields filled where inferable (with null if not)
- "enrichment_metadata" object describing changes and confidence
</output_format>

Respond with ONLY valid JSON. No markdown, no explanation."""


# =============================================================================
# PROMPT 3: Category Classification
# =============================================================================

CATEGORY_CLASSIFICATION_CONFIG = PromptConfig(
    name="category_classification",
    description="Determine primary product category with reasoning",
    recommended_temperature=0.3,
    recommended_max_tokens=1024,
)

CATEGORY_CLASSIFICATION_SYSTEM = """You are an Amazon category taxonomy expert. Your task is to analyze product data and determine the most appropriate Amazon category classification with clear reasoning.

<amazon_categories>
The main Amazon product categories are:
- Electronics
- Computers & Accessories
- Home & Kitchen
- Clothing, Shoes & Jewelry
- Sports & Outdoors
- Beauty & Personal Care
- Health & Household
- Toys & Games
- Automotive
- Pet Supplies
- Baby
- Grocery & Gourmet Food
- Tools & Home Improvement
- Patio, Lawn & Garden
- Office Products
- Arts, Crafts & Sewing
- Industrial & Scientific
- Musical Instruments
- Books
- Movies & TV
- Video Games
</amazon_categories>

<classification_rules>
1. Choose the MOST SPECIFIC category that accurately describes the majority of products
2. Consider product titles, descriptions, and any existing category data
3. If products span multiple categories, identify the PRIMARY one
4. Provide confidence level and reasoning for your classification
</classification_rules>"""

CATEGORY_CLASSIFICATION_USER = """<task>
Classify the primary Amazon category for brand "{brand_name}" based on the following product data.
</task>

<products>
{products_data}
</products>

<chain_of_thought>
Think through this step by step:
1. What product types are represented?
2. Which categories could these products belong to?
3. Is there a dominant category (>50% of products)?
4. What is the confidence level of this classification?
</chain_of_thought>

<examples>
Example 1:
Products: ["Nike Running Shoes", "Nike Basketball Sneakers", "Nike Athletic Socks", "Nike Sports Bra"]
Classification:
{{
  "primary_category": "Clothing, Shoes & Jewelry",
  "secondary_categories": ["Sports & Outdoors"],
  "is_multi_category": false,
  "confidence": "high",
  "reasoning": "All 4 products are athletic apparel/footwear items. While sports-related, these are primarily clothing/shoes which map to 'Clothing, Shoes & Jewelry' on Amazon."
}}

Example 2:
Products: ["TechBrand Bluetooth Speaker", "TechBrand Phone Case", "TechBrand USB Cable", "TechBrand Desk Lamp"]
Classification:
{{
  "primary_category": "Electronics",
  "secondary_categories": ["Cell Phones & Accessories", "Home & Kitchen"],
  "is_multi_category": true,
  "confidence": "medium",
  "reasoning": "Products span multiple categories: speaker (Electronics), phone case (Cell Phones), USB cable (Electronics), desk lamp (Home). Primary category is Electronics as 3/4 items fit there, but brand is genuinely multi-category."
}}
</examples>

<output_schema>
{{
  "primary_category": "string (from Amazon categories list)",
  "secondary_categories": ["array of other relevant categories"],
  "is_multi_category": "boolean",
  "confidence": "high | medium | low",
  "reasoning": "string explaining the classification decision"
}}
</output_schema>

Respond with ONLY valid JSON matching the schema."""


# =============================================================================
# PROMPT 4: Confidence Assessment
# =============================================================================

CONFIDENCE_ASSESSMENT_CONFIG = PromptConfig(
    name="confidence_assessment",
    description="Score brand presence confidence with evidence",
    recommended_temperature=0.2,
    recommended_max_tokens=1024,
)

CONFIDENCE_ASSESSMENT_SYSTEM = """You are a brand verification specialist. Your role is to assess the confidence level of a brand's Amazon presence based on available evidence. You must provide objective, evidence-based confidence scoring.

<scoring_factors>
Weight these factors in your assessment:
1. Official Store Presence (30%): Does an official brand store exist?
2. Product Volume (25%): How many products are listed?
3. Brand Consistency (20%): Do products consistently show the brand name?
4. Review Quality (15%): Are reviews genuine and substantial?
5. Seller Verification (10%): Are sellers verified/authorized?
</scoring_factors>

<confidence_levels>
- HIGH (0.75-1.0): Official store, 10+ products, consistent branding, verified seller
- MEDIUM (0.45-0.74): Some products found, partial branding, no official store
- LOW (0.20-0.44): Few products, inconsistent presence, unverified sellers
- NONE (0.0-0.19): No products found or all third-party
</confidence_levels>"""

CONFIDENCE_ASSESSMENT_USER = """<task>
Assess the Amazon presence confidence for brand "{brand_name}" based on the extracted data.
</task>

<brand_data>
{brand_data}
</brand_data>

<assessment_process>
Evaluate each scoring factor:
1. Official Store: Is there evidence of an official brand store?
2. Product Volume: How many products were found?
3. Brand Match: What percentage of products clearly match the brand?
4. Review Analysis: What's the total review volume and average rating?
5. Seller Status: Any verified or official seller indicators?

Calculate a weighted score and determine confidence level.
</assessment_process>

<examples>
Example - High Confidence:
{{
  "confidence_level": "high",
  "confidence_score": 0.87,
  "evidence": [
    "Official Amazon store found for brand",
    "23 products with consistent Nike branding",
    "Average rating 4.4/5 across 45,000+ reviews",
    "Multiple Amazon's Choice badges",
    "Prime eligibility on 90% of products"
  ],
  "concerns": [],
  "recommendation": "Brand has strong, verified Amazon presence. Safe to classify as established seller."
}}

Example - Low Confidence:
{{
  "confidence_level": "low",
  "confidence_score": 0.32,
  "evidence": [
    "3 products found matching brand name",
    "No official store presence"
  ],
  "concerns": [
    "Low review counts (under 50 per product)",
    "All products sold by third-party sellers",
    "Inconsistent pricing suggests possible counterfeits",
    "Brand name variations in listings"
  ],
  "recommendation": "Brand presence is weak and unverified. Recommend manual review before classification."
}}
</examples>

<output_schema>
{{
  "confidence_level": "high | medium | low | none",
  "confidence_score": "float 0.0-1.0",
  "evidence": ["array of positive indicators"],
  "concerns": ["array of negative indicators or risks"],
  "recommendation": "string with actionable recommendation"
}}
</output_schema>

Respond with ONLY valid JSON matching the schema."""


# =============================================================================
# PROMPT 5: Validation & Retry
# =============================================================================

VALIDATION_RETRY_CONFIG = PromptConfig(
    name="validation_retry",
    description="Handle extraction failures and validate/correct JSON output",
    recommended_temperature=0.1,
    recommended_max_tokens=4096,
)

VALIDATION_RETRY_SYSTEM = """You are a JSON validation and correction specialist. Your task is to fix malformed JSON output from previous extraction attempts while preserving all valid data.

<capabilities>
- Fix JSON syntax errors (missing brackets, quotes, commas)
- Correct schema violations (wrong types, missing required fields)
- Preserve original intent and data
- Handle partial or truncated responses
</capabilities>

<rules>
1. Fix syntax errors without changing semantic meaning
2. Add missing required fields with appropriate defaults (null for unknown)
3. Convert wrong types to correct types where possible
4. Never invent data that wasn't in the original
5. Output ONLY the corrected JSON
</rules>"""

VALIDATION_RETRY_USER = """<task>
The previous extraction attempt produced invalid or malformed output. Fix the JSON while preserving all valid data.
</task>

<original_prompt>
Extract Amazon presence data for brand: {brand_name}
</original_prompt>

<failed_output>
{failed_output}
</failed_output>

<validation_errors>
{validation_errors}
</validation_errors>

<expected_schema>
{expected_schema}
</expected_schema>

<correction_instructions>
1. Identify what's wrong with the output
2. Fix JSON syntax errors (brackets, quotes, commas)
3. Ensure all required fields are present (use null for unknown)
4. Convert types as needed (strings to numbers, etc.)
5. Preserve all valid data from the original

If the original is completely unsalvageable, return a minimal valid response:
{{
  "products": [],
  "has_official_store": false,
  "verified_seller": false,
  "confidence_reasoning": "Extraction failed: [brief reason]"
}}
</correction_instructions>

Respond with ONLY the corrected, valid JSON. No explanation."""


# =============================================================================
# Prompt Formatter Functions
# =============================================================================

def format_brand_extraction_prompt(
    brand_name: str,
    domain: str,
    search_results: str | list[dict],
) -> tuple[str, str]:
    """
    Format the brand extraction prompt with provided data.
    
    Args:
        brand_name: Name of the brand to extract
        domain: Brand's domain
        search_results: Raw search results (string or list of dicts)
        
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    if isinstance(search_results, list):
        search_results = json.dumps(search_results, indent=2)
    
    user_prompt = BRAND_EXTRACTION_USER.format(
        brand_name=brand_name,
        domain=domain,
        search_results=search_results,
    )
    
    return BRAND_EXTRACTION_SYSTEM, user_prompt


def format_product_enrichment_prompt(
    brand_name: str,
    product_data: dict | list[dict],
) -> tuple[str, str]:
    """
    Format the product enrichment prompt.
    
    Args:
        brand_name: Name of the brand
        product_data: Partial product data to enrich
        
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    if isinstance(product_data, (dict, list)):
        product_data = json.dumps(product_data, indent=2)
    
    user_prompt = PRODUCT_ENRICHMENT_USER.format(
        brand_name=brand_name,
        product_data=product_data,
    )
    
    return PRODUCT_ENRICHMENT_SYSTEM, user_prompt


def format_category_classification_prompt(
    brand_name: str,
    products_data: list[dict] | str,
) -> tuple[str, str]:
    """
    Format the category classification prompt.
    
    Args:
        brand_name: Name of the brand
        products_data: Product data for classification
        
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    if isinstance(products_data, list):
        # Extract just titles for cleaner prompt
        products_data = json.dumps([p.get("title", str(p)) for p in products_data], indent=2)
    
    user_prompt = CATEGORY_CLASSIFICATION_USER.format(
        brand_name=brand_name,
        products_data=products_data,
    )
    
    return CATEGORY_CLASSIFICATION_SYSTEM, user_prompt


def format_confidence_assessment_prompt(
    brand_name: str,
    brand_data: dict,
) -> tuple[str, str]:
    """
    Format the confidence assessment prompt.
    
    Args:
        brand_name: Name of the brand
        brand_data: Extracted brand data to assess
        
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    user_prompt = CONFIDENCE_ASSESSMENT_USER.format(
        brand_name=brand_name,
        brand_data=json.dumps(brand_data, indent=2),
    )
    
    return CONFIDENCE_ASSESSMENT_SYSTEM, user_prompt


def format_validation_retry_prompt(
    brand_name: str,
    failed_output: str,
    validation_errors: list[str],
    expected_schema: dict,
) -> tuple[str, str]:
    """
    Format the validation retry prompt.
    
    Args:
        brand_name: Name of the brand
        failed_output: The malformed output to fix
        validation_errors: List of validation error messages
        expected_schema: The expected JSON schema
        
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    user_prompt = VALIDATION_RETRY_USER.format(
        brand_name=brand_name,
        failed_output=failed_output,
        validation_errors="\n".join(f"- {e}" for e in validation_errors),
        expected_schema=json.dumps(expected_schema, indent=2),
    )
    
    return VALIDATION_RETRY_SYSTEM, user_prompt


# =============================================================================
# Prompt Registry
# =============================================================================

PROMPT_REGISTRY = {
    "brand_extraction": {
        "config": BRAND_EXTRACTION_CONFIG,
        "system": BRAND_EXTRACTION_SYSTEM,
        "user_template": BRAND_EXTRACTION_USER,
        "formatter": format_brand_extraction_prompt,
    },
    "product_enrichment": {
        "config": PRODUCT_ENRICHMENT_CONFIG,
        "system": PRODUCT_ENRICHMENT_SYSTEM,
        "user_template": PRODUCT_ENRICHMENT_USER,
        "formatter": format_product_enrichment_prompt,
    },
    "category_classification": {
        "config": CATEGORY_CLASSIFICATION_CONFIG,
        "system": CATEGORY_CLASSIFICATION_SYSTEM,
        "user_template": CATEGORY_CLASSIFICATION_USER,
        "formatter": format_category_classification_prompt,
    },
    "confidence_assessment": {
        "config": CONFIDENCE_ASSESSMENT_CONFIG,
        "system": CONFIDENCE_ASSESSMENT_SYSTEM,
        "user_template": CONFIDENCE_ASSESSMENT_USER,
        "formatter": format_confidence_assessment_prompt,
    },
    "validation_retry": {
        "config": VALIDATION_RETRY_CONFIG,
        "system": VALIDATION_RETRY_SYSTEM,
        "user_template": VALIDATION_RETRY_USER,
        "formatter": format_validation_retry_prompt,
    },
}


def get_prompt(prompt_name: str) -> dict:
    """
    Get a prompt configuration by name.
    
    Args:
        prompt_name: Name of the prompt to retrieve
        
    Returns:
        Dict with config, system, user_template, and formatter
        
    Raises:
        KeyError: If prompt name not found
    """
    if prompt_name not in PROMPT_REGISTRY:
        available = ", ".join(PROMPT_REGISTRY.keys())
        raise KeyError(f"Prompt '{prompt_name}' not found. Available: {available}")
    return PROMPT_REGISTRY[prompt_name]


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Configs
    "PromptConfig",
    "BRAND_EXTRACTION_CONFIG",
    "PRODUCT_ENRICHMENT_CONFIG",
    "CATEGORY_CLASSIFICATION_CONFIG",
    "CONFIDENCE_ASSESSMENT_CONFIG",
    "VALIDATION_RETRY_CONFIG",
    # System Prompts
    "BRAND_EXTRACTION_SYSTEM",
    "PRODUCT_ENRICHMENT_SYSTEM",
    "CATEGORY_CLASSIFICATION_SYSTEM",
    "CONFIDENCE_ASSESSMENT_SYSTEM",
    "VALIDATION_RETRY_SYSTEM",
    # User Templates
    "BRAND_EXTRACTION_USER",
    "PRODUCT_ENRICHMENT_USER",
    "CATEGORY_CLASSIFICATION_USER",
    "CONFIDENCE_ASSESSMENT_USER",
    "VALIDATION_RETRY_USER",
    # Formatters
    "format_brand_extraction_prompt",
    "format_product_enrichment_prompt",
    "format_category_classification_prompt",
    "format_confidence_assessment_prompt",
    "format_validation_retry_prompt",
    # Registry
    "PROMPT_REGISTRY",
    "get_prompt",
]
