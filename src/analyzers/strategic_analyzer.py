"""
Strategic Analyzer for Amazon Brand Intelligence Pipeline.

This module implements comprehensive strategic analysis of brand data,
generating actionable insights, competitive positioning, and recommendations.

The analyzer uses Claude LLM for creative strategic thinking with chain-of-thought
reasoning to produce high-quality business intelligence.

Components:
    1. Competitive Positioning - Market position assessment
    2. Market Opportunities - Category and expansion analysis
    3. Recommendations - Prioritized actionable items
    4. Category Context - Market trends and dynamics
    5. Risk Assessment - Threats and mitigation strategies
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from src.config.settings import Settings, get_settings
from src.models.schemas import (
    CompetitorInsight,
    ConfidenceLevel,
    ExtractionResult,
    MarketPosition,
    Recommendation,
    StrategicInsight,
    SWOTAnalysis,
)
from src.services.llm_service import ClaudeService
from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Analysis Constants
# =============================================================================

class AnalysisType(str, Enum):
    """Types of strategic analysis."""
    COMPETITIVE_POSITIONING = "competitive_positioning"
    MARKET_OPPORTUNITIES = "market_opportunities"
    RECOMMENDATIONS = "recommendations"
    CATEGORY_CONTEXT = "category_context"
    RISK_ASSESSMENT = "risk_assessment"


@dataclass
class AnalysisMetrics:
    """Metrics collected during analysis."""
    total_duration_ms: int = 0
    llm_calls: int = 0
    tokens_used: int = 0
    analysis_stages: list[str] = field(default_factory=list)


# =============================================================================
# Prompt Templates for Strategic Analysis
# =============================================================================

STRATEGIC_ANALYSIS_SYSTEM = """You are a senior Amazon marketplace strategist with 15+ years of experience in brand positioning, competitive analysis, and e-commerce growth strategy. You provide actionable, data-driven insights.

<expertise>
- Amazon marketplace dynamics and seller ecosystem
- Brand positioning and competitive intelligence
- Price strategy and product line optimization
- Review management and customer sentiment
- Category trends and seasonal patterns
- Risk assessment and opportunity identification
</expertise>

<analysis_principles>
1. Every insight must be backed by specific data points from the extraction
2. Recommendations must be specific, measurable, and time-bound
3. Avoid generic advice - tailor everything to this specific brand
4. Consider the brand's current position when suggesting strategies
5. Quantify opportunities and risks where possible
6. Acknowledge data limitations honestly
</analysis_principles>

<output_format>
All responses must be valid JSON matching the specified schema exactly.
Use chain-of-thought reasoning internally but output only the structured result.
</output_format>"""

COMPETITIVE_POSITIONING_PROMPT = """<task>
Analyze the competitive positioning for brand "{brand_name}" based on the following Amazon presence data.
</task>

<extraction_data>
{extraction_data}
</extraction_data>

<analysis_steps>
Follow this reasoning process:
1. Product Count Assessment
   - How many products does the brand have on Amazon?
   - Compare to typical category averages (Electronics: 50+, Apparel: 100+, Specialty: 10-30)
   - Is this a broad catalog or focused selection?

2. Price Positioning
   - What is the price range? What's the average price?
   - Is this premium, mid-market, or value positioning?
   - How does pricing compare to category norms?

3. Review Analysis
   - What's the average rating? Total review volume?
   - High ratings (4.5+) + High volume = Strong reputation
   - Calculate review velocity (reviews/product count)

4. Market Position Classification
   - LEADER: Dominant presence, high reviews, broad catalog, premium pricing ok
   - CHALLENGER: Strong but not dominant, competitive pricing, growing reviews
   - NICHE: Focused catalog, specialized products, loyal customer base
   - EMERGING: New to Amazon, building presence, limited data

5. Unique Selling Propositions
   - What makes this brand stand out from competitors?
   - Any Amazon's Choice or Best Seller badges?
   - Price/quality differentiation?
</analysis_steps>

<output_schema>
{{
  "market_position": "leader|challenger|niche|emerging|follower",
  "position_rationale": "2-3 sentence explanation backed by specific data",
  "product_count_assessment": {{
    "count": number,
    "category_average_comparison": "above|at|below",
    "assessment": "string"
  }},
  "price_positioning": {{
    "range_low": number or null,
    "range_high": number or null,
    "average": number or null,
    "tier": "premium|mid-market|value|mixed",
    "analysis": "string"
  }},
  "review_analysis": {{
    "average_rating": number or null,
    "total_reviews": number,
    "review_velocity": number or null,
    "sentiment": "positive|neutral|negative|mixed",
    "assessment": "string"
  }},
  "unique_selling_propositions": ["string array of 3-5 USPs"],
  "competitive_advantages": ["string array of key advantages"],
  "confidence": "high|medium|low"
}}
</output_schema>

Respond with ONLY valid JSON."""

MARKET_OPPORTUNITIES_PROMPT = """<task>
Identify market opportunities for brand "{brand_name}" based on the Amazon presence data and competitive positioning.
</task>

<extraction_data>
{extraction_data}
</extraction_data>

<positioning_context>
{positioning_data}
</positioning_context>

<opportunity_areas>
Analyze each area for potential:

1. Underserved Categories
   - Look at current category distribution
   - Identify adjacent categories with less brand presence
   - Consider natural product line extensions

2. Pricing Gaps
   - Are there price points not covered?
   - Could a value or premium tier expand market reach?
   - Bundle opportunities?

3. Product Line Expansion
   - What's missing from the current lineup?
   - Complementary products often bought together?
   - Accessories or variants?

4. Seasonal Opportunities
   - Category-specific seasonality (outdoor gear = summer, electronics = holidays)
   - Inventory/pricing optimization timing
   - Marketing calendar alignment

5. Geographic Expansion
   - Currently on which Amazon marketplaces?
   - Which additional markets make sense?
   - Localization requirements?
</opportunity_areas>

<output_schema>
{{
  "underserved_categories": [
    {{
      "category": "string",
      "opportunity_size": "high|medium|low",
      "rationale": "string",
      "suggested_products": ["string array"]
    }}
  ],
  "pricing_gaps": [
    {{
      "gap_description": "string",
      "target_price_range": "string",
      "potential_impact": "string"
    }}
  ],
  "product_expansion": [
    {{
      "product_type": "string",
      "rationale": "string",
      "estimated_demand": "high|medium|low",
      "competition_level": "high|medium|low"
    }}
  ],
  "seasonal_opportunities": [
    {{
      "season": "string",
      "opportunity": "string",
      "timing": "string",
      "action_items": ["string array"]
    }}
  ],
  "geographic_expansion": [
    {{
      "marketplace": "string",
      "priority": "high|medium|low",
      "rationale": "string",
      "considerations": ["string array"]
    }}
  ]
}}
</output_schema>

Respond with ONLY valid JSON."""

RECOMMENDATIONS_PROMPT = """<task>
Generate 3-5 prioritized, actionable recommendations for brand "{brand_name}" based on the complete analysis.
</task>

<extraction_data>
{extraction_data}
</extraction_data>

<positioning>
{positioning_data}
</positioning>

<opportunities>
{opportunities_data}
</opportunities>

<recommendation_framework>
For each recommendation:
1. Be SPECIFIC to this brand - avoid generic advice
2. Reference exact data points from the extraction
3. Estimate impact (revenue potential, efficiency gain)
4. Estimate effort (time, resources, investment)
5. Provide clear implementation timeline
6. Identify risk factors and mitigation
7. Define success metrics
</recommendation_framework>

<impact_effort_matrix>
Priority based on:
- Quick Wins: High Impact, Low Effort → Priority 1
- Major Projects: High Impact, High Effort → Priority 2
- Fill-Ins: Low Impact, Low Effort → Priority 3
- Avoid: Low Impact, High Effort → Don't recommend
</impact_effort_matrix>

<output_schema>
{{
  "recommendations": [
    {{
      "priority": 1-5,
      "title": "Short actionable title",
      "description": "Detailed description (2-3 sentences)",
      "data_support": "Specific data points backing this recommendation",
      "expected_impact": {{
        "type": "revenue|efficiency|brand|risk_reduction",
        "estimate": "string quantifying the impact",
        "confidence": "high|medium|low"
      }},
      "effort": {{
        "time": "string (e.g., '2-4 weeks')",
        "investment": "low|medium|high",
        "resources": "string describing needed resources"
      }},
      "implementation_timeline": {{
        "phase_1": "string",
        "phase_2": "string",
        "phase_3": "string"
      }},
      "risk_factors": ["string array"],
      "success_metrics": ["string array of measurable KPIs"]
    }}
  ],
  "quick_wins": ["1-2 sentence immediate actions"],
  "long_term_initiatives": ["1-2 sentence strategic initiatives"]
}}
</output_schema>

Respond with ONLY valid JSON. Provide minimum 3 recommendations."""

RISK_ASSESSMENT_PROMPT = """<task>
Assess risks and threats for brand "{brand_name}" on Amazon marketplace.
</task>

<extraction_data>
{extraction_data}
</extraction_data>

<risk_categories>
Analyze each risk area:

1. Brand Protection
   - Counterfeit/unauthorized sellers
   - Brand registry status implications
   - Listing hijacking risks

2. Pricing Pressure
   - Competitor undercutting
   - Race to bottom risks
   - Margin erosion indicators

3. Review Management
   - Negative review trends
   - Review velocity concerns
   - Customer satisfaction gaps

4. Inventory Risks
   - Stockout impact (rank loss, customer loss)
   - Overstock risks
   - Supply chain vulnerabilities

5. Competitive Threats
   - Emerging competitors
   - Market share erosion
   - Pricing wars
   - Feature parity loss
</risk_categories>

<output_schema>
{{
  "risk_factors": [
    {{
      "category": "brand_protection|pricing|reviews|inventory|competition",
      "risk": "Description of the risk",
      "severity": "critical|high|medium|low",
      "likelihood": "high|medium|low",
      "current_evidence": "Data points indicating this risk",
      "mitigation": "Recommended mitigation strategy",
      "monitoring": "How to monitor this risk"
    }}
  ],
  "overall_risk_level": "high|medium|low",
  "priority_actions": ["Top 3 risk mitigation actions"]
}}
</output_schema>

Respond with ONLY valid JSON."""

EXECUTIVE_SUMMARY_PROMPT = """<task>
Create an executive summary (2-3 sentences) for brand "{brand_name}" synthesizing all analysis.
</task>

<brand_data>
- Market Position: {market_position}
- Product Count: {product_count}
- Average Rating: {average_rating}
- Price Range: {price_range}
- Total Reviews: {total_reviews}
</brand_data>

<key_findings>
- Competitive Advantages: {advantages}
- Top Opportunities: {opportunities}
- Key Risks: {risks}
- Top Recommendation: {top_recommendation}
</key_findings>

<requirements>
- 2-3 concise sentences
- Lead with market position
- Highlight the single biggest opportunity
- Mention key action item
- Be specific to this brand, avoid generic statements
</requirements>

Respond with ONLY the executive summary text (no JSON, no quotes, just the summary)."""


# =============================================================================
# Strategic Analyzer Implementation
# =============================================================================

class StrategicAnalyzer:
    """
    Performs comprehensive strategic analysis on extracted brand data.
    
    The analyzer uses Claude LLM with chain-of-thought reasoning to generate
    actionable insights across five key dimensions:
    1. Competitive Positioning
    2. Market Opportunities
    3. Recommendations
    4. Category Context
    5. Risk Assessment
    
    Example:
        >>> analyzer = StrategicAnalyzer(llm_service)
        >>> insight = await analyzer.analyze(extraction_result)
        >>> print(insight.executive_summary)
    """
    
    def __init__(
        self,
        llm_service: ClaudeService,
        settings: Optional[Settings] = None,
        temperature: float = 0.7,
        max_retries: int = 2,
    ):
        """
        Initialize the strategic analyzer.
        
        Args:
            llm_service: Claude service for LLM calls
            settings: Application settings (optional)
            temperature: LLM temperature for creative analysis (default 0.7)
            max_retries: Maximum retries for failed LLM calls
        """
        self.llm_service = llm_service
        self.settings = settings or get_settings()
        self.temperature = temperature
        self.max_retries = max_retries
        self._metrics = AnalysisMetrics()
    
    async def analyze(self, extraction_data: ExtractionResult) -> StrategicInsight:
        """
        Generate comprehensive strategic analysis from extraction data.
        
        Performs multi-step analysis:
        1. Competitive positioning assessment
        2. Market opportunity identification
        3. Actionable recommendations
        4. Risk assessment
        5. Executive summary synthesis
        
        Args:
            extraction_data: Extraction result from Step 1
            
        Returns:
            StrategicInsight with all analysis components
            
        Raises:
            ValueError: If extraction data is invalid
            Exception: If analysis fails after retries
        """
        start_time = time.time()
        self._metrics = AnalysisMetrics()
        
        brand_name = extraction_data.brand_name
        logger.info(f"Starting strategic analysis for brand: {brand_name}")
        
        # Handle edge case: No Amazon presence
        if not extraction_data.amazon_presence.found:
            logger.warning(f"No Amazon presence for {brand_name}, generating market entry analysis")
            return await self._generate_market_entry_strategy(extraction_data)
        
        # Handle edge case: Limited data
        if len(extraction_data.top_products) < 3:
            logger.warning(f"Limited data for {brand_name}, generating directional insights")
            return await self._generate_directional_insights(extraction_data)
        
        try:
            # Step 1: Competitive Positioning Analysis
            positioning = await self._analyze_competitive_positioning(extraction_data)
            self._metrics.analysis_stages.append(AnalysisType.COMPETITIVE_POSITIONING.value)
            
            # Step 2: Market Opportunities
            opportunities = await self._analyze_market_opportunities(
                extraction_data, positioning
            )
            self._metrics.analysis_stages.append(AnalysisType.MARKET_OPPORTUNITIES.value)
            
            # Step 3: Risk Assessment
            risks = await self._analyze_risks(extraction_data)
            self._metrics.analysis_stages.append(AnalysisType.RISK_ASSESSMENT.value)
            
            # Step 4: Generate Recommendations
            recommendations = await self._generate_recommendations(
                extraction_data, positioning, opportunities
            )
            self._metrics.analysis_stages.append(AnalysisType.RECOMMENDATIONS.value)
            
            # Step 5: Synthesize Executive Summary
            executive_summary = await self._generate_executive_summary(
                extraction_data, positioning, opportunities, risks, recommendations
            )
            
            # Build StrategicInsight
            insight = self._build_strategic_insight(
                extraction_data=extraction_data,
                positioning=positioning,
                opportunities=opportunities,
                risks=risks,
                recommendations=recommendations,
                executive_summary=executive_summary,
            )
            
            self._metrics.total_duration_ms = int((time.time() - start_time) * 1000)
            
            logger.info(
                f"Strategic analysis completed for {brand_name}",
                duration_ms=self._metrics.total_duration_ms,
                llm_calls=self._metrics.llm_calls,
            )
            
            return insight
            
        except Exception as e:
            logger.error(f"Strategic analysis failed for {brand_name}: {e}")
            raise
    
    async def _analyze_competitive_positioning(
        self, extraction_data: ExtractionResult
    ) -> dict[str, Any]:
        """Analyze brand's competitive positioning."""
        logger.debug("Analyzing competitive positioning")
        
        extraction_json = self._prepare_extraction_data(extraction_data)
        
        prompt = COMPETITIVE_POSITIONING_PROMPT.format(
            brand_name=extraction_data.brand_name,
            extraction_data=extraction_json,
        )
        
        response = await self._call_llm(prompt, "competitive_positioning")
        return self._parse_json_response(response, "competitive_positioning")
    
    async def _analyze_market_opportunities(
        self,
        extraction_data: ExtractionResult,
        positioning: dict[str, Any],
    ) -> dict[str, Any]:
        """Identify market opportunities based on positioning."""
        logger.debug("Analyzing market opportunities")
        
        extraction_json = self._prepare_extraction_data(extraction_data)
        positioning_json = json.dumps(positioning, indent=2)
        
        prompt = MARKET_OPPORTUNITIES_PROMPT.format(
            brand_name=extraction_data.brand_name,
            extraction_data=extraction_json,
            positioning_data=positioning_json,
        )
        
        response = await self._call_llm(prompt, "market_opportunities")
        return self._parse_json_response(response, "market_opportunities")
    
    async def _analyze_risks(self, extraction_data: ExtractionResult) -> dict[str, Any]:
        """Assess risks and threats."""
        logger.debug("Analyzing risks")
        
        extraction_json = self._prepare_extraction_data(extraction_data)
        
        prompt = RISK_ASSESSMENT_PROMPT.format(
            brand_name=extraction_data.brand_name,
            extraction_data=extraction_json,
        )
        
        response = await self._call_llm(prompt, "risk_assessment")
        return self._parse_json_response(response, "risk_assessment")
    
    async def _generate_recommendations(
        self,
        extraction_data: ExtractionResult,
        positioning: dict[str, Any],
        opportunities: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate prioritized recommendations."""
        logger.debug("Generating recommendations")
        
        extraction_json = self._prepare_extraction_data(extraction_data)
        positioning_json = json.dumps(positioning, indent=2)
        opportunities_json = json.dumps(opportunities, indent=2)
        
        prompt = RECOMMENDATIONS_PROMPT.format(
            brand_name=extraction_data.brand_name,
            extraction_data=extraction_json,
            positioning_data=positioning_json,
            opportunities_data=opportunities_json,
        )
        
        response = await self._call_llm(prompt, "recommendations")
        result = self._parse_json_response(response, "recommendations")
        
        # Validate minimum 3 recommendations
        if len(result.get("recommendations", [])) < 3:
            logger.warning("LLM returned fewer than 3 recommendations, adding defaults")
            result = self._ensure_minimum_recommendations(result, extraction_data)
        
        return result
    
    async def _generate_executive_summary(
        self,
        extraction_data: ExtractionResult,
        positioning: dict[str, Any],
        opportunities: dict[str, Any],
        risks: dict[str, Any],
        recommendations: dict[str, Any],
    ) -> str:
        """Generate executive summary synthesizing all analysis."""
        logger.debug("Generating executive summary")
        
        # Extract key data points for summary
        price_range = extraction_data.price_range or (0, 0)
        price_range_str = f"${price_range[0]:.2f} - ${price_range[1]:.2f}" if price_range[0] else "N/A"
        
        # Extract advantages - handle dict or string format
        raw_advantages = positioning.get("competitive_advantages", [])[:3]
        advantages = []
        for adv in raw_advantages:
            if isinstance(adv, dict):
                advantages.append(adv.get("advantage", str(adv)))
            else:
                advantages.append(str(adv))
        
        # Extract opportunities - handle dict or string format
        raw_opportunities = (
            opportunities.get("high_priority_opportunities", []) +
            opportunities.get("underserved_categories", []) +
            opportunities.get("product_expansion", [])
        )[:2]
        top_opportunities = []
        for opp in raw_opportunities:
            if isinstance(opp, dict):
                top_opportunities.append(opp.get("opportunity", opp.get("category", str(opp))))
            else:
                top_opportunities.append(str(opp))
        
        # Extract key risks - handle dict or string format
        raw_risks = risks.get("detailed_risks", risks.get("risk_factors", []))
        key_risks = []
        for r in raw_risks:
            if isinstance(r, dict):
                if r.get("severity") in ["critical", "high"]:
                    key_risks.append(r.get("risk_name", r.get("risk", str(r))))
            else:
                key_risks.append(str(r))
        key_risks = key_risks[:2]
        
        # Extract top recommendation - handle dict or string format
        raw_recs = recommendations.get("recommendations", [])
        if raw_recs:
            first_rec = raw_recs[0]
            if isinstance(first_rec, dict):
                top_rec = first_rec.get("title", first_rec.get("action_item", "Optimize Amazon presence"))
            else:
                top_rec = str(first_rec)
        else:
            top_rec = "Optimize Amazon presence"
        
        # Get market position - handle dict or string format
        market_pos = positioning.get("market_position", "emerging")
        if isinstance(market_pos, dict):
            market_pos = market_pos.get("classification", "emerging")
        
        prompt = EXECUTIVE_SUMMARY_PROMPT.format(
            brand_name=extraction_data.brand_name,
            market_position=market_pos,
            product_count=len(extraction_data.top_products),
            average_rating=extraction_data.average_rating or "N/A",
            price_range=price_range_str,
            total_reviews=extraction_data.total_reviews,
            advantages=", ".join(advantages) if advantages else "Strong brand identity",
            opportunities=", ".join(top_opportunities) if top_opportunities else "Market expansion",
            risks=", ".join(key_risks) if key_risks else "Standard competitive pressures",
            top_recommendation=top_rec,
        )
        
        response = await self._call_llm(prompt, "executive_summary")
        
        # Clean up response (should be plain text, not JSON)
        summary = response.strip().strip('"').strip("'")
        
        # Ensure minimum length
        if len(summary) < 50:
            summary = self._generate_fallback_summary(extraction_data, positioning)
        
        return summary
    
    async def _generate_market_entry_strategy(
        self, extraction_data: ExtractionResult
    ) -> StrategicInsight:
        """Generate insights for brands with no Amazon presence."""
        logger.info(f"Generating market entry strategy for {extraction_data.brand_name}")
        
        return StrategicInsight(
            market_position=MarketPosition.EMERGING,
            market_position_rationale=(
                f"{extraction_data.brand_name} does not currently have a presence on Amazon. "
                "This represents a significant opportunity for market entry and growth."
            ),
            swot=SWOTAnalysis(
                strengths=["Established brand identity outside Amazon"],
                weaknesses=["No Amazon marketplace presence", "No existing reviews or ratings"],
                opportunities=[
                    "Untapped Amazon customer base",
                    "Fresh start with optimized listings",
                    "Competitor analysis before entry",
                ],
                threats=[
                    "Established competitors already on Amazon",
                    "Initial visibility challenges",
                    "Learning curve for Amazon operations",
                ],
            ),
            competitive_advantages=["Brand reputation from other channels"],
            risk_factors=[
                "High investment required for initial launch",
                "Time to build reviews and rankings",
            ],
            growth_recommendations=[
                "Develop Amazon entry strategy",
                "Research competitor pricing and positioning",
                "Create optimized product listings",
                "Plan launch marketing and early review strategy",
                "Consider Amazon Advertising budget",
            ],
            competitor_insights=[],
            brand_health_score=25.0,  # Low score for no presence
            confidence_score=0.5,  # Lower confidence due to limited data
            executive_summary=(
                f"{extraction_data.brand_name} is not currently selling on Amazon, "
                "representing a significant market opportunity. A strategic entry focusing on "
                "optimized listings and competitive pricing could capture market share. "
                "Immediate priority: Develop comprehensive Amazon launch strategy."
            ),
        )
    
    async def _generate_directional_insights(
        self, extraction_data: ExtractionResult
    ) -> StrategicInsight:
        """Generate insights when data is limited."""
        logger.info(f"Generating directional insights for {extraction_data.brand_name}")
        
        # Try to do some basic analysis with available data
        product_count = len(extraction_data.top_products)
        avg_rating = extraction_data.average_rating or 0
        
        position = MarketPosition.EMERGING
        if product_count >= 2 and avg_rating >= 4.0:
            position = MarketPosition.NICHE
        
        return StrategicInsight(
            market_position=position,
            market_position_rationale=(
                f"Based on limited data ({product_count} products), "
                f"{extraction_data.brand_name} appears to have a {position.value} market position. "
                "Additional data would improve analysis confidence."
            ),
            swot=SWOTAnalysis(
                strengths=[f"Presence on Amazon with {product_count} products"],
                weaknesses=["Limited product range visible"],
                opportunities=["Expand product catalog", "Increase visibility"],
                threats=["Competition from larger brands"],
            ),
            competitive_advantages=["Existing Amazon presence"],
            risk_factors=["Limited data for comprehensive analysis"],
            growth_recommendations=[
                "Expand product catalog on Amazon",
                "Improve product listing optimization",
                "Build review volume through customer engagement",
            ],
            competitor_insights=[],
            brand_health_score=50.0,
            confidence_score=0.4,  # Low confidence due to limited data
            executive_summary=(
                f"{extraction_data.brand_name} has a limited but present Amazon footprint with "
                f"{product_count} products. With limited data, directional insights suggest focusing on "
                "catalog expansion and listing optimization. More data needed for comprehensive strategy."
            ),
        )
    
    def _build_strategic_insight(
        self,
        extraction_data: ExtractionResult,
        positioning: dict[str, Any],
        opportunities: dict[str, Any],
        risks: dict[str, Any],
        recommendations: dict[str, Any],
        executive_summary: str,
    ) -> StrategicInsight:
        """Build StrategicInsight from analysis components."""
        
        # Map market position - handle both simple string and nested dict format
        market_pos = positioning.get("market_position", "emerging")
        if isinstance(market_pos, dict):
            position_str = market_pos.get("classification", "emerging").lower()
            position_rationale = market_pos.get("rationale", f"Based on product analysis for {extraction_data.brand_name}")
        else:
            position_str = str(market_pos).lower()
            position_rationale = positioning.get("position_rationale", f"Based on product analysis for {extraction_data.brand_name}")
        
        position_map = {
            "leader": MarketPosition.LEADER,
            "challenger": MarketPosition.CHALLENGER,
            "follower": MarketPosition.FOLLOWER,
            "niche": MarketPosition.NICHE,
            "emerging": MarketPosition.EMERGING,
        }
        market_position = position_map.get(position_str, MarketPosition.EMERGING)
        
        # Extract competitive advantages - handle list of dicts or list of strings
        raw_advantages = positioning.get("competitive_advantages", [])
        competitive_advantages = []
        for adv in raw_advantages[:5]:
            if isinstance(adv, dict):
                competitive_advantages.append(adv.get("advantage", str(adv)))
            else:
                competitive_advantages.append(str(adv))
        
        # Extract weaknesses from positioning or risks
        raw_weaknesses = positioning.get("weaknesses", risks.get("detailed_risks", []))
        weaknesses = []
        for w in raw_weaknesses[:5]:
            if isinstance(w, dict):
                weaknesses.append(w.get("weakness", w.get("risk_name", str(w))))
            else:
                weaknesses.append(str(w))
        
        # Extract opportunities from high_priority_opportunities or quick_wins
        raw_opportunities = opportunities.get("high_priority_opportunities", opportunities.get("quick_wins", []))
        opp_list = []
        for opp in raw_opportunities[:5]:
            if isinstance(opp, dict):
                opp_list.append(opp.get("opportunity", str(opp)))
            else:
                opp_list.append(str(opp))
        
        # Extract threats from risks
        raw_risks = risks.get("detailed_risks", risks.get("risk_factors", []))
        threats = []
        for r in raw_risks[:5]:
            if isinstance(r, dict):
                threats.append(r.get("risk_name", r.get("risk", str(r))))
            else:
                threats.append(str(r))
        
        # Build SWOT from parsed components
        swot = SWOTAnalysis(
            strengths=competitive_advantages,
            weaknesses=weaknesses,
            opportunities=opp_list,
            threats=threats,
        )
        
        # Build competitor insights
        competitor_insights = []
        if extraction_data.competitors_found:
            for competitor in extraction_data.competitors_found[:5]:
                competitor_insights.append(
                    CompetitorInsight(
                        name=competitor,
                        price_positioning="similar",  # Default, could be enhanced
                        key_differentiators=[],
                    )
                )
        
        # Extract recommendations - handle list of dicts with title/action_item
        raw_recommendations = recommendations.get("recommendations", [])
        growth_recommendations = []
        for rec in raw_recommendations[:5]:
            if isinstance(rec, dict):
                growth_recommendations.append(rec.get("title", rec.get("action_item", str(rec))))
            else:
                growth_recommendations.append(str(rec))

        # Extract structured recommendations
        detailed_recommendations = []
        for rec in raw_recommendations[:5]:
            if isinstance(rec, dict):
                # safely extract nested fields
                impact_data = rec.get("expected_impact", {})
                impact = impact_data.get("estimate", "Medium impact") if isinstance(impact_data, dict) else "Medium impact"
                
                effort_data = rec.get("effort", {})
                difficulty = effort_data.get("investment", "Medium") if isinstance(effort_data, dict) else "Medium"
                timeline = effort_data.get("time", "1-3 months") if isinstance(effort_data, dict) else "1-3 months"
                
                detailed_recommendations.append(
                    Recommendation(
                        priority=rec.get("priority", 3),
                        title=rec.get("title", "Strategic Recommendation"),
                        description=rec.get("description", "Implement structured improvement strategy."),
                        impact=impact,
                        difficulty=difficulty,
                        timeline=timeline
                    )
                )
        
        # Validate minimum recommendations
        if len(growth_recommendations) < 3:
            growth_recommendations.extend([
                "Optimize product listings for better visibility",
                "Develop customer review strategy",
                "Analyze competitor pricing and positioning",
            ])
            growth_recommendations = list(set(growth_recommendations))[:5]
        
        # Extract risk factors for separate field
        risk_factors = []
        for r in raw_risks[:5]:
            if isinstance(r, dict):
                risk_factors.append(r.get("risk_name", r.get("description", str(r))))
            else:
                risk_factors.append(str(r))
        
        # Calculate brand health score (0-100)
        brand_health = self._calculate_brand_health(extraction_data, positioning)
        
        # Calculate confidence score (0-1)
        confidence = self._calculate_confidence(extraction_data, positioning)
        
        return StrategicInsight(
            market_position=market_position,
            market_position_rationale=position_rationale,
            swot=swot,
            competitive_advantages=competitive_advantages,
            risk_factors=risk_factors,
            growth_recommendations=growth_recommendations,
            recommendations=detailed_recommendations,
            competitor_insights=competitor_insights,
            estimated_market_share=None,  # Optional - could be enhanced
            brand_health_score=brand_health,
            confidence_score=confidence,
            executive_summary=executive_summary,
        )
    
    def _calculate_brand_health(
        self, extraction_data: ExtractionResult, positioning: dict[str, Any]
    ) -> float:
        """Calculate brand health score (0-100)."""
        score = 0.0
        
        # Rating component (0-25)
        avg_rating = extraction_data.average_rating or 0
        score += min(25, (avg_rating / 5) * 25)
        
        # Review volume component (0-25)
        total_reviews = extraction_data.total_reviews
        if total_reviews >= 10000:
            score += 25
        elif total_reviews >= 1000:
            score += 20
        elif total_reviews >= 100:
            score += 15
        elif total_reviews >= 10:
            score += 10
        else:
            score += 5
        
        # Product count component (0-25)
        product_count = len(extraction_data.top_products)
        if product_count >= 50:
            score += 25
        elif product_count >= 20:
            score += 20
        elif product_count >= 10:
            score += 15
        elif product_count >= 5:
            score += 10
        else:
            score += 5
        
        # Market position component (0-25)
        market_pos = positioning.get("market_position", "emerging")
        if isinstance(market_pos, dict):
            position = market_pos.get("classification", "emerging").lower()
        else:
            position = str(market_pos).lower()
            
        position_scores = {
            "leader": 25,
            "challenger": 20,
            "niche": 18,
            "follower": 15,
            "emerging": 10,
        }
        score += position_scores.get(position, 10)
        
        return round(min(100, score), 1)
    
    def _calculate_confidence(
        self, extraction_data: ExtractionResult, positioning: dict[str, Any]
    ) -> float:
        """Calculate analysis confidence score (0-1)."""
        confidence = 0.5  # Base confidence
        
        # More products = higher confidence
        product_count = len(extraction_data.top_products)
        if product_count >= 20:
            confidence += 0.2
        elif product_count >= 10:
            confidence += 0.15
        elif product_count >= 5:
            confidence += 0.1
        
        # More reviews = higher confidence
        if extraction_data.total_reviews >= 1000:
            confidence += 0.15
        elif extraction_data.total_reviews >= 100:
            confidence += 0.1
        
        # High rating consistency = higher confidence
        if extraction_data.average_rating and extraction_data.average_rating >= 4.0:
            confidence += 0.1
        
        # LLM confidence
        if "confidence" in positioning:
             llm_confidence = str(positioning.get("confidence", "medium")).lower()
        else:
            # Check inside market_position dict
            market_pos = positioning.get("market_position", {})
            if isinstance(market_pos, dict):
                llm_confidence = str(market_pos.get("confidence", "medium")).lower()
            else:
                llm_confidence = "medium"

        if llm_confidence == "high":
            confidence += 0.05
        elif llm_confidence == "low":
            confidence -= 0.1
        
        return round(min(1.0, max(0.1, confidence)), 2)
    
    def _prepare_extraction_data(self, extraction_data: ExtractionResult) -> str:
        """Prepare extraction data for LLM prompt."""
        # Create a summarized version for the prompt
        summary = {
            "brand_name": extraction_data.brand_name,
            "domain": extraction_data.domain,
            "amazon_presence": {
                "found": extraction_data.amazon_presence.found,
                "confidence": extraction_data.amazon_presence.confidence.value if hasattr(extraction_data.amazon_presence.confidence, 'value') else extraction_data.amazon_presence.confidence,
                "evidence": extraction_data.amazon_presence.evidence[:5],
            },
            "primary_category": extraction_data.primary_category,
            "all_categories": extraction_data.all_categories[:10],
            "product_count": len(extraction_data.top_products),
            "estimated_total_products": extraction_data.estimated_product_count,
            "price_range": extraction_data.price_range,
            "average_rating": extraction_data.average_rating,
            "total_reviews": extraction_data.total_reviews,
            "competitors": extraction_data.competitors_found[:5],
            "top_products": [
                {
                    "asin": p.asin,
                    "title": p.title[:100],
                    "price": p.price,
                    "rating": p.rating,
                    "review_count": p.review_count,
                    "is_best_seller": p.is_best_seller,
                    "is_amazon_choice": p.is_amazon_choice,
                }
                for p in extraction_data.top_products[:10]
            ],
        }
        return json.dumps(summary, indent=2, default=str)
    
    async def _call_llm(self, prompt: str, analysis_type: str) -> str:
        """Call LLM with retry logic."""
        for attempt in range(self.max_retries + 1):
            try:
                # Use _call_api directly with system and user messages
                messages = [{"role": "user", "content": prompt}]
                
                response_text, usage = await self.llm_service._call_api(
                    messages=messages,
                    system=STRATEGIC_ANALYSIS_SYSTEM,
                    temperature=self.temperature,
                    max_tokens=4096,
                    use_cache=False,  # Strategic analysis should be fresh
                )
                self._metrics.llm_calls += 1
                self._metrics.tokens_used += usage.total_tokens
                return response_text
            except Exception as e:
                if attempt < self.max_retries:
                    logger.warning(
                        f"LLM call failed for {analysis_type}, retrying",
                        attempt=attempt + 1,
                        error=str(e),
                    )
                    await self._sleep(1)  # Brief pause before retry
                else:
                    raise
        return ""  # Should not reach here
    
    async def _sleep(self, seconds: float):
        """Async sleep helper."""
        import asyncio
        await asyncio.sleep(seconds)
    
    def _parse_json_response(self, response: str, analysis_type: str) -> dict[str, Any]:
        """Parse JSON response from LLM."""
        try:
            # Clean response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON for {analysis_type}: {e}")
            return self._get_fallback_response(analysis_type)
    
    def _get_fallback_response(self, analysis_type: str) -> dict[str, Any]:
        """Get fallback response for failed parsing."""
        fallbacks = {
            "competitive_positioning": {
                "market_position": "emerging",
                "position_rationale": "Unable to determine precise positioning due to parsing error",
                "competitive_advantages": ["Brand presence on Amazon"],
                "confidence": "low",
            },
            "market_opportunities": {
                "underserved_categories": [],
                "pricing_gaps": [],
                "product_expansion": [],
                "seasonal_opportunities": [],
                "geographic_expansion": [],
            },
            "risk_assessment": {
                "risk_factors": [],
                "overall_risk_level": "medium",
                "priority_actions": ["Conduct manual risk assessment"],
            },
            "recommendations": {
                "recommendations": [
                    {
                        "priority": 1,
                        "title": "Optimize Amazon listings",
                        "description": "Review and optimize product listings for better visibility",
                        "data_support": "Standard best practice",
                        "expected_impact": {"type": "revenue", "estimate": "5-15% improvement", "confidence": "medium"},
                        "effort": {"time": "2-4 weeks", "investment": "low", "resources": "Marketing team"},
                        "risk_factors": [],
                        "success_metrics": ["Improved search rankings", "Higher click-through rates"],
                    }
                ],
                "quick_wins": ["Review listing keywords"],
                "long_term_initiatives": ["Build comprehensive Amazon strategy"],
            },
        }
        return fallbacks.get(analysis_type, {})
    
    def _ensure_minimum_recommendations(
        self, recommendations: dict, extraction_data: ExtractionResult
    ) -> dict:
        """Ensure minimum 3 recommendations."""
        existing = recommendations.get("recommendations", [])
        
        default_recs = [
            {
                "priority": len(existing) + 1,
                "title": "Expand Amazon product catalog",
                "description": f"Consider adding more products to increase {extraction_data.brand_name}'s Amazon presence and capture additional market share.",
                "data_support": f"Currently {len(extraction_data.top_products)} products visible",
                "expected_impact": {"type": "revenue", "estimate": "10-25% growth potential", "confidence": "medium"},
                "effort": {"time": "4-8 weeks", "investment": "medium", "resources": "Product and operations teams"},
                "risk_factors": ["Inventory investment required"],
                "success_metrics": ["New product listings", "Incremental revenue"],
            },
            {
                "priority": len(existing) + 2,
                "title": "Optimize pricing strategy",
                "description": "Analyze competitor pricing and optimize price points for better conversion while maintaining margins.",
                "data_support": f"Price range: ${extraction_data.price_range[0] if extraction_data.price_range else 0:.2f} - ${extraction_data.price_range[1] if extraction_data.price_range else 0:.2f}",
                "expected_impact": {"type": "revenue", "estimate": "5-15% conversion improvement", "confidence": "medium"},
                "effort": {"time": "2-3 weeks", "investment": "low", "resources": "Pricing analyst"},
                "risk_factors": ["Margin compression risk"],
                "success_metrics": ["Conversion rate", "Revenue per unit"],
            },
            {
                "priority": len(existing) + 3,
                "title": "Build customer review volume",
                "description": "Implement review generation strategy to build social proof and improve product visibility.",
                "data_support": f"Current reviews: {extraction_data.total_reviews}",
                "expected_impact": {"type": "brand", "estimate": "Improved rankings and conversion", "confidence": "high"},
                "effort": {"time": "Ongoing", "investment": "low", "resources": "Customer success team"},
                "risk_factors": ["Requires consistent customer engagement"],
                "success_metrics": ["Review count growth", "Rating maintenance"],
            },
        ]
        
        # Add enough defaults to reach 3
        while len(existing) < 3 and default_recs:
            existing.append(default_recs.pop(0))
        
        recommendations["recommendations"] = existing
        return recommendations
    
    def _generate_fallback_summary(
        self, extraction_data: ExtractionResult, positioning: dict[str, Any]
    ) -> str:
        """Generate fallback executive summary."""
        position = positioning.get("market_position", "emerging")
        product_count = len(extraction_data.top_products)
        avg_rating = extraction_data.average_rating or "N/A"
        
        return (
            f"{extraction_data.brand_name} holds a {position} position on Amazon with "
            f"{product_count} products and an average rating of {avg_rating}. "
            f"Key opportunity: Optimize product listings and expand catalog. "
            f"Priority action: Develop comprehensive Amazon growth strategy."
        )
    
    def get_metrics(self) -> AnalysisMetrics:
        """Get metrics from the last analysis run."""
        return self._metrics
    
    async def close(self) -> None:
        """Close service connections."""
        if hasattr(self.llm_service, 'close'):
            await self.llm_service.close()


# =============================================================================
# Convenience Function
# =============================================================================

async def analyze_brand_strategically(
    extraction_data: ExtractionResult,
    llm_service: Optional[ClaudeService] = None,
    settings: Optional[Settings] = None,
) -> StrategicInsight:
    """
    Convenience function for strategic analysis.
    
    Args:
        extraction_data: Extraction result to analyze
        llm_service: Optional Claude service (created if not provided)
        settings: Optional settings
        
    Returns:
        StrategicInsight with analysis results
    """
    if llm_service is None:
        settings = settings or get_settings()
        llm_service = ClaudeService(settings)
    
    analyzer = StrategicAnalyzer(llm_service, settings)
    try:
        return await analyzer.analyze(extraction_data)
    finally:
        await analyzer.close()
