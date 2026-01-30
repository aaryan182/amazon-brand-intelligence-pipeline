"""
Sophisticated Analysis Prompts for Strategic Analyzer.

This module contains production-grade prompts optimized for Claude 3.5 Sonnet
to generate high-quality strategic analysis of Amazon brand presence data.

Prompts are designed with:
- Chain-of-thought reasoning (Observation → Interpretation → Implication → Recommendation)
- Specific data references required
- Quality controls to avoid generic advice
- XML structure for reliable parsing
- Few-shot examples where appropriate

Prompt Categories:
    1. Competitive Positioning Analysis
    2. Opportunity Identification
    3. Recommendation Generation
    4. Category Trend Analysis
    5. Risk Assessment
    6. Executive Summary Synthesis
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# =============================================================================
# Configuration
# =============================================================================

class AnalysisPromptType(str, Enum):
    """Types of analysis prompts."""
    COMPETITIVE_POSITIONING = "competitive_positioning"
    OPPORTUNITY_IDENTIFICATION = "opportunity_identification"
    RECOMMENDATION_GENERATION = "recommendation_generation"
    CATEGORY_TREND_ANALYSIS = "category_trend_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    EXECUTIVE_SUMMARY = "executive_summary"


@dataclass
class AnalysisPromptConfig:
    """Configuration for analysis prompts."""
    temperature: float = 0.7
    max_tokens: int = 4000
    model: str = "claude-sonnet-4-20250514"
    require_chain_of_thought: bool = True
    require_data_references: bool = True


# Default configuration for all analysis prompts
DEFAULT_ANALYSIS_CONFIG = AnalysisPromptConfig()


# =============================================================================
# System Prompts
# =============================================================================

STRATEGIC_ANALYST_SYSTEM = """You are a senior Amazon marketplace strategist with 10+ years of experience in brand positioning, competitive analysis, and e-commerce growth strategy. Your expertise includes:

<expertise>
- Amazon marketplace dynamics, algorithms, and seller ecosystem
- Brand positioning and competitive intelligence
- Pricing strategy and margin optimization
- Product line architecture and catalog management
- Customer review psychology and social proof
- Category trends, seasonality, and market dynamics
- Risk identification and mitigation strategies
- Data-driven decision making with quantifiable outcomes
</expertise>

<analysis_principles>
1. SPECIFICITY: Every insight MUST reference specific data points (product names, prices, ratings, counts)
2. ACTIONABILITY: Recommendations must be concrete, measurable, and time-bound
3. TAILORING: Avoid generic marketplace advice - all insights specific to this brand
4. QUANTIFICATION: Include numbers, ranges, and benchmarks wherever possible
5. HONESTY: Acknowledge data limitations; don't infer beyond what data supports
6. CHAIN-OF-THOUGHT: Follow Observation → Interpretation → Implication → Recommendation
</analysis_principles>

<quality_standards>
- Reference at least 2-3 specific products by name when discussing product strategy
- Include price points when discussing pricing strategy
- Mention review counts and ratings when discussing brand perception
- Compare to category benchmarks when available
- Flag assumptions explicitly
- Prioritize recommendations by impact and feasibility
</quality_standards>

<output_format>
Respond ONLY with valid JSON matching the specified schema.
Use the chain-of-thought structure internally but output structured results.
Never include markdown code blocks in your response - just the raw JSON.
</output_format>"""


CATEGORY_EXPERT_SYSTEM = """You are an Amazon category analyst specializing in marketplace trends, competitive dynamics, and category-level insights. You combine data analysis with industry knowledge to identify patterns and opportunities.

<expertise>
- Amazon category structures and classification
- Seasonal and cyclical demand patterns
- Competitive intensity metrics
- Category growth trajectories
- Cross-category expansion opportunities
- Category-specific success factors
</expertise>

<analysis_approach>
- Use data patterns to infer category dynamics
- Compare observed metrics to typical category benchmarks
- Identify category-specific opportunities and threats
- Consider category maturity and saturation levels
</analysis_approach>

Respond ONLY with valid JSON. Do not include markdown code blocks."""


# =============================================================================
# Prompt 1: Competitive Positioning Analysis
# =============================================================================

COMPETITIVE_POSITIONING_SYSTEM = STRATEGIC_ANALYST_SYSTEM

COMPETITIVE_POSITIONING_USER = """<task>
Analyze the competitive positioning for {brand_name} on Amazon based on the following extracted data.
</task>

<brand_data>
<amazon_presence>
- Presence Detected: {amazon_presence_found}
- Confidence Level: {amazon_presence_confidence}
- Evidence: {amazon_presence_evidence}
</amazon_presence>

<catalog_metrics>
- Total Products Found: {product_count}
- Primary Category: {primary_category}
- All Categories: {all_categories}
- Estimated Total on Amazon: {estimated_product_count}
</catalog_metrics>

<pricing_data>
- Price Range: ${price_range_low:.2f} - ${price_range_high:.2f}
- Average Price: ${avg_price:.2f}
- Median Price: ${median_price:.2f}
</pricing_data>

<customer_metrics>
- Average Rating: {avg_rating}/5.0
- Total Reviews: {total_reviews:,}
- Review Velocity: {review_velocity} reviews/product
</customer_metrics>

<top_products>
{top_products_formatted}
</top_products>

<competitors_identified>
{competitors_list}
</competitors_identified>
</brand_data>

<analysis_framework>
Apply chain-of-thought reasoning for each section:

1. MARKET POSITION ASSESSMENT
   - Observation: What does the data objectively show?
   - Interpretation: What market position does this indicate?
   - Classification: leader | challenger | niche | emerging | follower
   - Confidence: How certain is this classification given the data?

2. COMPETITIVE ADVANTAGES
   - Observation: What strengths are evident in the data?
   - Interpretation: Why are these competitive advantages?
   - Specificity: Which products exemplify these advantages?
   - Sustainability: How defensible are these advantages?

3. WEAKNESSES AND GAPS
   - Observation: What gaps or weaknesses appear in the data?
   - Interpretation: What do these gaps mean competitively?
   - Risk Level: How significant are these weaknesses?
   - Addressability: Can these be fixed, and how easily?

4. CATEGORY BENCHMARKS
   - How does this brand compare to typical category performance?
   - Is the pricing above/below/at category average?
   - Is the review volume strong/moderate/weak for this category?
   - Is the rating competitive within the category?
</analysis_framework>

<output_schema>
{{
  "market_position": {{
    "classification": "leader|challenger|niche|emerging|follower",
    "confidence": "high|medium|low",
    "rationale": "2-3 sentences explaining classification with specific data references",
    "chain_of_thought": {{
      "observation": "What the data shows",
      "interpretation": "What it means for positioning",
      "implication": "Strategic implications"
    }}
  }},
  "competitive_advantages": [
    {{
      "advantage": "Specific advantage description",
      "evidence": "Data points supporting this (product names, metrics)",
      "defensibility": "high|medium|low",
      "exemplar_products": ["Product names that demonstrate this"]
    }}
  ],
  "weaknesses": [
    {{
      "weakness": "Specific weakness description",
      "evidence": "Data indicating this gap",
      "severity": "critical|moderate|minor",
      "addressability": "easy|moderate|difficult"
    }}
  ],
  "category_benchmarks": {{
    "price_positioning": "premium|above_average|average|below_average|value",
    "review_volume_assessment": "strong|moderate|weak",
    "rating_competitiveness": "leading|competitive|average|below_average",
    "catalog_depth": "extensive|moderate|limited|minimal",
    "overall_category_fit": "string assessment"
  }},
  "unique_selling_propositions": [
    "USP 1 with specific evidence",
    "USP 2 with specific evidence",
    "USP 3 with specific evidence"
  ],
  "data_limitations": ["Any caveats about the analysis due to data constraints"]
}}
</output_schema>

Respond with ONLY valid JSON. Reference specific products, prices, and metrics."""

COMPETITIVE_POSITIONING_EXAMPLE = {
    "input": {
        "brand_name": "Patagonia",
        "product_count": 47,
        "avg_rating": 4.6,
        "total_reviews": 52000,
        "price_range": (45.00, 299.00),
        "primary_category": "Sports & Outdoors",
    },
    "output": {
        "market_position": {
            "classification": "leader",
            "confidence": "high",
            "rationale": "With 47 products, 52,000+ total reviews, and an exceptional 4.6 average rating, Patagonia demonstrates market leader characteristics. The premium pricing ($45-$299) combined with high review volumes indicates strong brand loyalty and customer satisfaction in the outdoor apparel category.",
            "chain_of_thought": {
                "observation": "47 products with 52K reviews and 4.6 rating, premium prices",
                "interpretation": "High volume + high satisfaction + premium pricing = leader position",
                "implication": "Brand can command premium prices while maintaining volume",
            },
        },
        "competitive_advantages": [
            {
                "advantage": "Premium brand positioning with strong customer loyalty",
                "evidence": "4.6 average rating across 52K reviews; prices 20-40% above category average",
                "defensibility": "high",
                "exemplar_products": ["Better Sweater Fleece Jacket", "Nano Puff Jacket"],
            },
        ],
    },
}


# =============================================================================
# Prompt 2: Opportunity Identification
# =============================================================================

OPPORTUNITY_IDENTIFICATION_SYSTEM = STRATEGIC_ANALYST_SYSTEM

OPPORTUNITY_IDENTIFICATION_USER = """<task>
Identify actionable growth opportunities for {brand_name} based on their Amazon presence data and competitive positioning analysis.
</task>

<brand_context>
- Brand: {brand_name}
- Market Position: {market_position}
- Current Product Count: {product_count}
- Primary Category: {primary_category}
- Price Range: ${price_range_low:.2f} - ${price_range_high:.2f}
- Average Rating: {avg_rating}/5.0
- Total Reviews: {total_reviews:,}
</brand_context>

<current_products>
{top_products_formatted}
</current_products>

<category_distribution>
{category_distribution}
</category_distribution>

<positioning_insights>
{positioning_summary}
</positioning_insights>

<opportunity_framework>
Analyze each opportunity area using chain-of-thought:

1. CATALOG EXPANSION OPPORTUNITIES
   - What product gaps exist in current lineup?
   - What complementary products could be added?
   - What variants or bundles would increase capture?
   
2. CATEGORY EXPANSION
   - Are there adjacent categories with low competition?
   - What product adaptations enable category crossover?
   - Which new categories align with brand identity?

3. PRICING OPPORTUNITIES
   - Are there underserved price tiers?
   - Could bundles/multipacks increase AOV?
   - Is there room for a value or premium sub-line?

4. CONVERSION OPTIMIZATION
   - Are there products with high ratings but low reviews?
   - Could better imagery/content improve conversion?
   - Are A+ Content or video opportunities present?

5. SEASONAL/TIMING OPPORTUNITIES
   - What seasonal peaks apply to this category?
   - Are there holiday-specific product opportunities?
   - When should inventory/advertising scale?
</opportunity_framework>

<output_schema>
{{
  "high_priority_opportunities": [
    {{
      "opportunity": "Clear, actionable opportunity description",
      "category": "catalog|pricing|category_expansion|conversion|seasonal",
      "chain_of_thought": {{
        "observation": "What data pattern suggests this opportunity",
        "interpretation": "Why this represents an opportunity",
        "implication": "What success would look like",
        "recommendation": "Specific action to capture this opportunity"
      }},
      "potential_impact": {{
        "type": "revenue|margin|market_share|brand|efficiency",
        "estimate": "Quantified estimate (e.g., '15-25% revenue increase')",
        "confidence": "high|medium|low"
      }},
      "effort_required": {{
        "time": "e.g., '2-4 weeks'",
        "investment": "low|medium|high",
        "complexity": "simple|moderate|complex"
      }},
      "supporting_evidence": "Specific data points backing this opportunity"
    }}
  ],
  "medium_priority_opportunities": [
    {{
      "opportunity": "Description",
      "category": "string",
      "potential_impact": "Brief impact description",
      "effort_required": "Brief effort description"
    }}
  ],
  "long_term_strategic_opportunities": [
    {{
      "opportunity": "Description",
      "timeline": "6-12 months or longer",
      "strategic_rationale": "Why this matters long-term"
    }}
  ],
  "quick_wins": [
    "Immediate action 1 (can be done this week)",
    "Immediate action 2 (can be done this week)"
  ],
  "opportunities_to_avoid": [
    {{
      "trap": "An apparent opportunity that should be avoided",
      "reason": "Why this is actually a trap"
    }}
  ]
}}
</output_schema>

Provide specific, actionable opportunities. Reference actual products and metrics.
Minimum 3 high-priority opportunities required."""


# =============================================================================
# Prompt 3: Recommendation Generation
# =============================================================================

RECOMMENDATION_GENERATION_SYSTEM = STRATEGIC_ANALYST_SYSTEM

RECOMMENDATION_GENERATION_USER = """<task>
Generate 3-5 specific, prioritized strategic recommendations for {brand_name} to improve their Amazon marketplace performance.
</task>

<complete_analysis_context>
<brand_overview>
- Brand: {brand_name}
- Domain: {domain}
- Market Position: {market_position}
- Brand Health Score: {brand_health_score}/100
</brand_overview>

<amazon_presence>
- Products on Amazon: {product_count}
- Primary Category: {primary_category}
- Price Range: ${price_range_low:.2f} - ${price_range_high:.2f}
- Average Rating: {avg_rating}/5.0
- Total Reviews: {total_reviews:,}
</amazon_presence>

<competitive_positioning>
{competitive_positioning_summary}
</competitive_positioning>

<identified_opportunities>
{opportunities_summary}
</identified_opportunities>

<identified_risks>
{risks_summary}
</identified_risks>
</complete_analysis_context>

<recommendation_criteria>
Each recommendation MUST include:
1. SPECIFIC ACTION: Exactly what to do (not vague "improve X")
2. DATA JUSTIFICATION: Why this matters based on the analysis
3. EXPECTED IMPACT: Quantified benefit (range acceptable)
4. IMPLEMENTATION DIFFICULTY: Time, cost, complexity
5. TIMELINE: Realistic implementation phases
6. SUCCESS METRICS: How to measure outcomes
7. RISK FACTORS: What could go wrong
8. DEPENDENCIES: What's needed to execute

Prioritization Matrix:
- Priority 1: High Impact + Low/Medium Effort (Do First)
- Priority 2: High Impact + High Effort (Major Initiative)
- Priority 3: Medium Impact + Low Effort (Quick Wins)
- Skip: Low Impact + High Effort (Avoid)
</recommendation_criteria>

<output_schema>
{{
  "recommendations": [
    {{
      "priority": 1,
      "title": "Action-oriented title (verb + object)",
      "action_item": "Specific, concrete action description (2-3 sentences)",
      "data_justification": "Why the data supports this recommendation",
      "expected_impact": {{
        "primary_benefit": "Main benefit type",
        "quantified_estimate": "e.g., '10-20% increase in conversion'",
        "secondary_benefits": ["Additional benefits"],
        "confidence_level": "high|medium|low"
      }},
      "implementation": {{
        "difficulty": "easy|moderate|challenging",
        "estimated_time": "e.g., '2-4 weeks'",
        "estimated_cost": "e.g., '$5,000-15,000' or 'Minimal/staff time only'",
        "required_resources": ["Resource 1", "Resource 2"],
        "dependencies": ["Prerequisite 1", "Prerequisite 2"]
      }},
      "timeline": {{
        "phase_1": {{
          "name": "Phase name",
          "duration": "e.g., 'Week 1-2'",
          "activities": ["Activity 1", "Activity 2"]
        }},
        "phase_2": {{
          "name": "Phase name",
          "duration": "e.g., 'Week 3-4'",
          "activities": ["Activity 1", "Activity 2"]
        }},
        "phase_3": {{
          "name": "Phase name (if applicable)",
          "duration": "e.g., 'Week 5-8'",
          "activities": ["Activity 1", "Activity 2"]
        }}
      }},
      "success_metrics": [
        {{
          "metric": "Specific KPI",
          "current_baseline": "Current value if known",
          "target": "Target value",
          "measurement_method": "How to track"
        }}
      ],
      "risk_factors": [
        {{
          "risk": "What could go wrong",
          "likelihood": "high|medium|low",
          "mitigation": "How to address"
        }}
      ]
    }}
  ],
  "implementation_sequence": {{
    "immediate_actions": ["Actions to start this week"],
    "short_term_30_days": ["Actions for first month"],
    "medium_term_90_days": ["Actions for first quarter"],
    "long_term_initiatives": ["Strategic initiatives for 6+ months"]
  }},
  "resource_requirements": {{
    "total_estimated_investment": "Range estimate for all recommendations",
    "key_skills_needed": ["Skill 1", "Skill 2"],
    "external_resources": ["Consultants/tools that might be needed"]
  }}
}}
</output_schema>

Requirements:
- Minimum 3 recommendations, maximum 5
- At least 2 must be Priority 1 or 2
- Reference specific products, prices, or metrics from the analysis
- Avoid generic advice like "improve SEO" - be specific about what and how"""

RECOMMENDATION_EXAMPLE = {
    "priority": 1,
    "title": "Launch Product Bundle Strategy for Top 3 SKUs",
    "action_item": "Create curated bundles combining the Better Sweater Fleece ($129) with complementary items like the Synchilla Snap-T ($99) for an 'Outdoor Essentials' bundle at $199 (15% discount vs. separate purchase). Target Prime Day and holiday season.",
    "data_justification": "Analysis shows strong performance of these individual products (combined 12,000+ reviews, 4.7 avg rating) but no bundle options. Category benchmarks show bundles increase AOV by 25-40%.",
    "expected_impact": {
        "primary_benefit": "Revenue per transaction",
        "quantified_estimate": "25-35% AOV increase on bundled purchases",
        "secondary_benefits": ["Reduced comparison shopping", "Inventory velocity increase"],
        "confidence_level": "high",
    },
}


# =============================================================================
# Prompt 4: Category Trend Analysis
# =============================================================================

CATEGORY_TREND_SYSTEM = CATEGORY_EXPERT_SYSTEM

CATEGORY_TREND_USER = """<task>
Analyze category trends and dynamics for {brand_name}'s presence in {primary_category} based on their Amazon product data.
</task>

<brand_category_data>
<primary_category>{primary_category}</primary_category>
<all_categories>
{all_categories_formatted}
</all_categories>

<product_distribution>
{product_category_distribution}
</product_distribution>

<price_by_category>
{price_by_category}
</price_by_category>

<rating_by_category>
{rating_by_category}
</rating_by_category>
</brand_category_data>

<category_analysis_framework>
Infer category dynamics from the product data:

1. CATEGORY POSITION
   - In which categories is the brand strongest?
   - Where is presence weak or absent?
   - What does product distribution suggest about brand focus?

2. CATEGORY TRENDS (inferred)
   - Based on price points, is this category premiumizing or commoditizing?
   - Based on review volumes, is demand strong or stabilizing?
   - What product types suggest emerging sub-categories?

3. COMPETITIVE INTENSITY
   - High ratings + many competitors = intense competition
   - Price compression = competitive pressure
   - What does the data suggest about category competitiveness?

4. SEASONAL PATTERNS
   - What seasonality is typical for these categories?
   - When are peak demand periods?
   - How should inventory and marketing align?

5. CROSS-CATEGORY OPPORTUNITIES
   - What adjacent categories could the brand enter?
   - What products bridge current categories?
   - Where are category whitespaces?
</category_analysis_framework>

<output_schema>
{{
  "category_overview": {{
    "primary_category": "{primary_category}",
    "category_position": "leader|strong|moderate|emerging|weak",
    "concentration": "focused|diversified|fragmented",
    "assessment": "2-3 sentence summary of category position"
  }},
  "category_dynamics": {{
    "trend_direction": "growing|stable|declining",
    "price_trend": "premiumizing|stable|commoditizing",
    "competition_intensity": "high|medium|low",
    "maturity_level": "emerging|growth|mature|saturated",
    "supporting_evidence": "Data points supporting these assessments"
  }},
  "category_performance": [
    {{
      "category": "Category name",
      "brand_strength": "strong|moderate|weak",
      "product_count": 0,
      "avg_rating": 0.0,
      "avg_price": 0.0,
      "strategic_importance": "core|growth|opportunistic"
    }}
  ],
  "seasonal_patterns": {{
    "peak_seasons": ["Season 1", "Season 2"],
    "low_seasons": ["Season 1"],
    "key_events": ["Prime Day", "Black Friday", etc.],
    "recommended_calendar": {{
      "Q1": "Focus area",
      "Q2": "Focus area",
      "Q3": "Focus area",
      "Q4": "Focus area"
    }}
  }},
  "cross_category_opportunities": [
    {{
      "target_category": "Category name",
      "fit_with_brand": "high|medium|low",
      "rationale": "Why this makes sense",
      "entry_products": ["Suggested product types"]
    }}
  ],
  "category_risks": [
    {{
      "risk": "Category-level risk",
      "severity": "high|medium|low",
      "mitigation": "Suggested approach"
    }}
  ]
}}
</output_schema>

Provide category insights grounded in the available data. Flag assumptions clearly."""


# =============================================================================
# Prompt 5: Risk Assessment
# =============================================================================

RISK_ASSESSMENT_SYSTEM = STRATEGIC_ANALYST_SYSTEM

RISK_ASSESSMENT_USER = """<task>
Conduct a comprehensive risk assessment for {brand_name}'s Amazon marketplace presence.
</task>

<brand_profile>
<identity>
- Brand: {brand_name}
- Domain: {domain}
- Market Position: {market_position}
</identity>

<amazon_metrics>
- Product Count: {product_count}
- Categories: {categories}
- Price Range: ${price_range_low:.2f} - ${price_range_high:.2f}
- Average Rating: {avg_rating}/5.0
- Total Reviews: {total_reviews:,}
</amazon_metrics>

<product_details>
{top_products_formatted}
</product_details>

<identified_competitors>
{competitors_list}
</identified_competitors>
</brand_profile>

<risk_assessment_framework>
Analyze risks across five critical dimensions:

1. BRAND PROTECTION RISKS
   - Unauthorized/counterfeit sellers
   - Listing hijacking potential
   - Brand registry considerations
   - IP protection gaps

2. PRICING RISKS
   - Competitor undercutting
   - Race to bottom dynamics
   - MAP policy enforcement
   - Margin erosion indicators

3. REPUTATION RISKS
   - Negative review trends
   - Rating decline indicators
   - Customer service concerns
   - Public relations vulnerabilities

4. OPERATIONAL RISKS
   - Inventory/stockout risks
   - Fulfillment vulnerabilities
   - Supply chain dependencies
   - Amazon policy compliance

5. COMPETITIVE RISKS
   - New entrant threats
   - Market share erosion
   - Feature parity gaps
   - Customer loyalty vulnerabilities

For each risk:
- Observation: What data indicates this risk?
- Severity: How impactful if it occurs?
- Likelihood: How probable is it?
- Mitigation: How to address it?
</risk_assessment_framework>

<output_schema>
{{
  "risk_summary": {{
    "overall_risk_level": "high|medium|low",
    "highest_priority_risks": ["Risk 1", "Risk 2", "Risk 3"],
    "executive_assessment": "2-3 sentence summary of risk posture"
  }},
  "detailed_risks": [
    {{
      "category": "brand_protection|pricing|reputation|operational|competitive",
      "risk_name": "Specific risk title",
      "description": "Detailed description of the risk",
      "chain_of_thought": {{
        "observation": "What data suggests this risk",
        "interpretation": "Why this is concerning",
        "implication": "What happens if not addressed"
      }},
      "severity": "critical|high|medium|low",
      "likelihood": "high|medium|low",
      "risk_score": 1-10,
      "current_evidence": ["Specific data points indicating this risk"],
      "warning_signs": ["What to monitor for this risk"],
      "mitigation_strategy": {{
        "immediate_actions": ["Action 1", "Action 2"],
        "long_term_measures": ["Measure 1", "Measure 2"],
        "resources_needed": "Brief description",
        "timeline": "How long to mitigate"
      }},
      "monitoring_approach": "How to track this risk ongoing"
    }}
  ],
  "risk_matrix": {{
    "critical_immediate": ["Risks requiring immediate action"],
    "high_priority": ["Risks to address in 30 days"],
    "monitor_closely": ["Risks to track but not urgent"],
    "accept_and_monitor": ["Low risks to simply watch"]
  }},
  "risk_mitigation_priorities": [
    {{
      "priority": 1,
      "risk": "Risk name",
      "action": "Specific mitigation action",
      "timeline": "When to complete",
      "owner": "Suggested responsibility"
    }}
  ]
}}
</output_schema>

Identify real risks based on the data. Avoid hypothetical risks not supported by evidence.
Prioritize risks that are both high-severity and high-likelihood."""


# =============================================================================
# Prompt 6: Executive Summary Synthesis
# =============================================================================

EXECUTIVE_SUMMARY_SYSTEM = """You are a senior business strategist creating concise executive communications. You distill complex analysis into clear, actionable summaries for C-suite executives.

<style_guidelines>
- Lead with the most important finding
- Use specific numbers and metrics
- Be concise but complete (2-3 sentences)
- End with a clear call to action
- Avoid jargon and generic statements
- Tailor completely to this specific brand
</style_guidelines>"""

EXECUTIVE_SUMMARY_USER = """<task>
Synthesize the complete analysis for {brand_name} into a 2-3 sentence executive summary.
</task>

<analysis_summary>
<brand>
- Name: {brand_name}
- Market Position: {market_position}
- Brand Health Score: {brand_health_score}/100
</brand>

<key_metrics>
- Products on Amazon: {product_count}
- Average Rating: {avg_rating}/5.0
- Total Reviews: {total_reviews:,}
- Price Range: ${price_range_low:.2f} - ${price_range_high:.2f}
</key_metrics>

<competitive_position>
{competitive_positioning_brief}
</competitive_position>

<top_opportunities>
{top_opportunities}
</top_opportunities>

<key_risks>
{key_risks}
</key_risks>

<priority_recommendation>
{top_recommendation}
</priority_recommendation>
</analysis_summary>

<requirements>
Create an executive summary that:
1. Opens with market position and key strength (1 sentence)
2. Highlights the primary opportunity OR risk (1 sentence)
3. States the #1 recommended action (1 sentence)

Must include:
- At least 2 specific numbers/metrics
- The brand name
- A clear action directive
- Time-bound language where applicable

Avoid:
- Generic statements that could apply to any brand
- Vague recommendations
- Jargon or buzzwords without substance
</requirements>

Respond with ONLY the executive summary text (2-3 sentences). No JSON, no formatting, just the summary."""


# =============================================================================
# Formatter Functions
# =============================================================================

def format_products_for_prompt(products: list[dict], max_products: int = 10) -> str:
    """Format product list for inclusion in prompts."""
    if not products:
        return "No products available"
    
    formatted = []
    for i, p in enumerate(products[:max_products], 1):
        formatted.append(
            f"{i}. {p.get('title', 'Unknown')[:80]}\n"
            f"   ASIN: {p.get('asin', 'N/A')} | "
            f"Price: ${p.get('price', 0):.2f} | "
            f"Rating: {p.get('rating', 0)}/5.0 | "
            f"Reviews: {p.get('review_count', 0):,}"
        )
    
    return "\n".join(formatted)


def format_category_distribution(categories: dict[str, int]) -> str:
    """Format category distribution for prompts."""
    if not categories:
        return "No category data available"
    
    formatted = []
    for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        formatted.append(f"- {category}: {count} products")
    
    return "\n".join(formatted)


def format_competitors_list(competitors: list[str]) -> str:
    """Format competitor list for prompts."""
    if not competitors:
        return "No competitors identified"
    
    return "\n".join(f"- {c}" for c in competitors[:10])


def format_competitive_positioning_prompt(
    brand_name: str,
    amazon_presence_found: bool,
    amazon_presence_confidence: str,
    amazon_presence_evidence: list[str],
    product_count: int,
    primary_category: str,
    all_categories: list[str],
    estimated_product_count: int,
    price_range_low: float,
    price_range_high: float,
    avg_price: float,
    median_price: float,
    avg_rating: float,
    total_reviews: int,
    top_products: list[dict],
    competitors: list[str],
) -> tuple[str, str]:
    """
    Format the competitive positioning prompt with all required data.
    
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    review_velocity = total_reviews / max(product_count, 1)
    
    user_prompt = COMPETITIVE_POSITIONING_USER.format(
        brand_name=brand_name,
        amazon_presence_found=amazon_presence_found,
        amazon_presence_confidence=amazon_presence_confidence,
        amazon_presence_evidence=", ".join(amazon_presence_evidence[:5]),
        product_count=product_count,
        primary_category=primary_category or "Uncategorized",
        all_categories=", ".join(all_categories[:10]) if all_categories else "N/A",
        estimated_product_count=estimated_product_count,
        price_range_low=price_range_low or 0,
        price_range_high=price_range_high or 0,
        avg_price=avg_price or 0,
        median_price=median_price or avg_price or 0,
        avg_rating=avg_rating or 0,
        total_reviews=total_reviews,
        review_velocity=f"{review_velocity:.1f}",
        top_products_formatted=format_products_for_prompt(top_products),
        competitors_list=format_competitors_list(competitors),
    )
    
    return COMPETITIVE_POSITIONING_SYSTEM, user_prompt


def format_opportunity_prompt(
    brand_name: str,
    market_position: str,
    product_count: int,
    primary_category: str,
    price_range_low: float,
    price_range_high: float,
    avg_rating: float,
    total_reviews: int,
    top_products: list[dict],
    category_distribution: dict[str, int],
    positioning_summary: str,
) -> tuple[str, str]:
    """Format the opportunity identification prompt."""
    
    user_prompt = OPPORTUNITY_IDENTIFICATION_USER.format(
        brand_name=brand_name,
        market_position=market_position,
        product_count=product_count,
        primary_category=primary_category or "General",
        price_range_low=price_range_low or 0,
        price_range_high=price_range_high or 0,
        avg_rating=avg_rating or 0,
        total_reviews=total_reviews,
        top_products_formatted=format_products_for_prompt(top_products),
        category_distribution=format_category_distribution(category_distribution),
        positioning_summary=positioning_summary,
    )
    
    return OPPORTUNITY_IDENTIFICATION_SYSTEM, user_prompt


def format_recommendation_prompt(
    brand_name: str,
    domain: str,
    market_position: str,
    brand_health_score: float,
    product_count: int,
    primary_category: str,
    price_range_low: float,
    price_range_high: float,
    avg_rating: float,
    total_reviews: int,
    competitive_positioning_summary: str,
    opportunities_summary: str,
    risks_summary: str,
) -> tuple[str, str]:
    """Format the recommendation generation prompt."""
    
    user_prompt = RECOMMENDATION_GENERATION_USER.format(
        brand_name=brand_name,
        domain=domain,
        market_position=market_position,
        brand_health_score=brand_health_score,
        product_count=product_count,
        primary_category=primary_category or "General",
        price_range_low=price_range_low or 0,
        price_range_high=price_range_high or 0,
        avg_rating=avg_rating or 0,
        total_reviews=total_reviews,
        competitive_positioning_summary=competitive_positioning_summary,
        opportunities_summary=opportunities_summary,
        risks_summary=risks_summary,
    )
    
    return RECOMMENDATION_GENERATION_SYSTEM, user_prompt


def format_category_trend_prompt(
    brand_name: str,
    primary_category: str,
    all_categories: list[str],
    category_product_counts: dict[str, int],
    category_avg_prices: dict[str, float],
    category_avg_ratings: dict[str, float],
) -> tuple[str, str]:
    """Format the category trend analysis prompt."""
    
    # Format category distributions
    all_cats_formatted = "\n".join(f"- {cat}" for cat in all_categories[:15])
    
    product_dist = format_category_distribution(category_product_counts)
    
    price_by_cat = "\n".join(
        f"- {cat}: ${price:.2f} avg"
        for cat, price in category_avg_prices.items()
    ) if category_avg_prices else "No price data by category"
    
    rating_by_cat = "\n".join(
        f"- {cat}: {rating:.1f}/5.0 avg"
        for cat, rating in category_avg_ratings.items()
    ) if category_avg_ratings else "No rating data by category"
    
    user_prompt = CATEGORY_TREND_USER.format(
        brand_name=brand_name,
        primary_category=primary_category or "Uncategorized",
        all_categories_formatted=all_cats_formatted,
        product_category_distribution=product_dist,
        price_by_category=price_by_cat,
        rating_by_category=rating_by_cat,
    )
    
    return CATEGORY_TREND_SYSTEM, user_prompt


def format_risk_assessment_prompt(
    brand_name: str,
    domain: str,
    market_position: str,
    product_count: int,
    categories: str,
    price_range_low: float,
    price_range_high: float,
    avg_rating: float,
    total_reviews: int,
    top_products: list[dict],
    competitors: list[str],
) -> tuple[str, str]:
    """Format the risk assessment prompt."""
    
    user_prompt = RISK_ASSESSMENT_USER.format(
        brand_name=brand_name,
        domain=domain,
        market_position=market_position,
        product_count=product_count,
        categories=categories,
        price_range_low=price_range_low or 0,
        price_range_high=price_range_high or 0,
        avg_rating=avg_rating or 0,
        total_reviews=total_reviews,
        top_products_formatted=format_products_for_prompt(top_products),
        competitors_list=format_competitors_list(competitors),
    )
    
    return RISK_ASSESSMENT_SYSTEM, user_prompt


def format_executive_summary_prompt(
    brand_name: str,
    market_position: str,
    brand_health_score: float,
    product_count: int,
    avg_rating: float,
    total_reviews: int,
    price_range_low: float,
    price_range_high: float,
    competitive_positioning_brief: str,
    top_opportunities: str,
    key_risks: str,
    top_recommendation: str,
) -> tuple[str, str]:
    """Format the executive summary prompt."""
    
    user_prompt = EXECUTIVE_SUMMARY_USER.format(
        brand_name=brand_name,
        market_position=market_position,
        brand_health_score=brand_health_score,
        product_count=product_count,
        avg_rating=avg_rating or 0,
        total_reviews=total_reviews,
        price_range_low=price_range_low or 0,
        price_range_high=price_range_high or 0,
        competitive_positioning_brief=competitive_positioning_brief,
        top_opportunities=top_opportunities,
        key_risks=key_risks,
        top_recommendation=top_recommendation,
    )
    
    return EXECUTIVE_SUMMARY_SYSTEM, user_prompt


# =============================================================================
# Prompt Registry
# =============================================================================

ANALYSIS_PROMPT_REGISTRY = {
    AnalysisPromptType.COMPETITIVE_POSITIONING: {
        "system": COMPETITIVE_POSITIONING_SYSTEM,
        "user_template": COMPETITIVE_POSITIONING_USER,
        "formatter": format_competitive_positioning_prompt,
        "config": AnalysisPromptConfig(temperature=0.7, max_tokens=4000),
        "example": COMPETITIVE_POSITIONING_EXAMPLE,
    },
    AnalysisPromptType.OPPORTUNITY_IDENTIFICATION: {
        "system": OPPORTUNITY_IDENTIFICATION_SYSTEM,
        "user_template": OPPORTUNITY_IDENTIFICATION_USER,
        "formatter": format_opportunity_prompt,
        "config": AnalysisPromptConfig(temperature=0.7, max_tokens=4000),
    },
    AnalysisPromptType.RECOMMENDATION_GENERATION: {
        "system": RECOMMENDATION_GENERATION_SYSTEM,
        "user_template": RECOMMENDATION_GENERATION_USER,
        "formatter": format_recommendation_prompt,
        "config": AnalysisPromptConfig(temperature=0.7, max_tokens=4000),
        "example": RECOMMENDATION_EXAMPLE,
    },
    AnalysisPromptType.CATEGORY_TREND_ANALYSIS: {
        "system": CATEGORY_TREND_SYSTEM,
        "user_template": CATEGORY_TREND_USER,
        "formatter": format_category_trend_prompt,
        "config": AnalysisPromptConfig(temperature=0.6, max_tokens=3000),
    },
    AnalysisPromptType.RISK_ASSESSMENT: {
        "system": RISK_ASSESSMENT_SYSTEM,
        "user_template": RISK_ASSESSMENT_USER,
        "formatter": format_risk_assessment_prompt,
        "config": AnalysisPromptConfig(temperature=0.5, max_tokens=4000),
    },
    AnalysisPromptType.EXECUTIVE_SUMMARY: {
        "system": EXECUTIVE_SUMMARY_SYSTEM,
        "user_template": EXECUTIVE_SUMMARY_USER,
        "formatter": format_executive_summary_prompt,
        "config": AnalysisPromptConfig(temperature=0.6, max_tokens=500),
    },
}


def get_analysis_prompt(prompt_type: AnalysisPromptType) -> dict:
    """
    Get prompt configuration by type.
    
    Args:
        prompt_type: Type of analysis prompt
        
    Returns:
        Dictionary with system, user_template, formatter, and config
        
    Raises:
        KeyError: If prompt type not found
    """
    if prompt_type not in ANALYSIS_PROMPT_REGISTRY:
        raise KeyError(f"Unknown prompt type: {prompt_type}")
    return ANALYSIS_PROMPT_REGISTRY[prompt_type]


def list_analysis_prompts() -> list[str]:
    """List all available analysis prompt types."""
    return [p.value for p in AnalysisPromptType]


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Configuration
    "AnalysisPromptType",
    "AnalysisPromptConfig",
    "DEFAULT_ANALYSIS_CONFIG",
    # System Prompts
    "STRATEGIC_ANALYST_SYSTEM",
    "CATEGORY_EXPERT_SYSTEM",
    # Prompt Templates
    "COMPETITIVE_POSITIONING_SYSTEM",
    "COMPETITIVE_POSITIONING_USER",
    "OPPORTUNITY_IDENTIFICATION_SYSTEM",
    "OPPORTUNITY_IDENTIFICATION_USER",
    "RECOMMENDATION_GENERATION_SYSTEM",
    "RECOMMENDATION_GENERATION_USER",
    "CATEGORY_TREND_SYSTEM",
    "CATEGORY_TREND_USER",
    "RISK_ASSESSMENT_SYSTEM",
    "RISK_ASSESSMENT_USER",
    "EXECUTIVE_SUMMARY_SYSTEM",
    "EXECUTIVE_SUMMARY_USER",
    # Examples
    "COMPETITIVE_POSITIONING_EXAMPLE",
    "RECOMMENDATION_EXAMPLE",
    # Formatters
    "format_products_for_prompt",
    "format_category_distribution",
    "format_competitors_list",
    "format_competitive_positioning_prompt",
    "format_opportunity_prompt",
    "format_recommendation_prompt",
    "format_category_trend_prompt",
    "format_risk_assessment_prompt",
    "format_executive_summary_prompt",
    # Registry
    "ANALYSIS_PROMPT_REGISTRY",
    "get_analysis_prompt",
    "list_analysis_prompts",
]
