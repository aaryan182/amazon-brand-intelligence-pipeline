import pytest
from src.analyzers.prompts import (
    format_products_for_prompt,
    format_category_distribution,
    format_competitors_list,
    format_competitive_positioning_prompt,
    format_opportunity_prompt,
    format_recommendation_prompt,
    format_category_trend_prompt,
    format_risk_assessment_prompt,
    format_executive_summary_prompt,
    get_analysis_prompt,
    list_analysis_prompts,
    AnalysisPromptType
)

def test_format_products_for_prompt():
    products = [{"title": "P1", "asin": "A1", "price": 10.0, "rating": 4.0, "review_count": 50}]
    res = format_products_for_prompt(products)
    assert "P1" in res
    assert "A1" in res
    assert format_products_for_prompt([]) == "No products available"

def test_format_category_distribution():
    cats = {"Cat1": 5, "Cat2": 3}
    res = format_category_distribution(cats)
    assert "Cat1: 5 products" in res
    assert format_category_distribution({}) == "No category data available"

def test_format_competitors_list():
    comps = ["C1", "C2"]
    res = format_competitors_list(comps)
    assert "- C1" in res
    assert format_competitors_list([]) == "No competitors identified"

def test_format_all_analysis_prompts():
    # Test each formatter with dummy data
    brand = "TestBrand"
    
    # Competitive Positioning
    sys, user = format_competitive_positioning_prompt(
        brand, True, "high", ["Evidence"], 5, "Cat", ["AllCats"], 10, 5.0, 15.0, 10.0, 10.0, 4.5, 100, [], []
    )
    assert brand in user
    
    # Opportunity
    sys, user = format_opportunity_prompt(brand, "leader", 5, "Cat", 5.0, 15.0, 4.5, 100, [], {}, "PosSummary")
    assert brand in user
    assert "PosSummary" in user
    
    # Recommendation
    sys, user = format_recommendation_prompt(brand, "domain.com", "leader", 70.0, 5, "Cat", 5.0, 15.0, 4.5, 100, "CompSum", "OppSum", "RiskSum")
    assert "CompSum" in user
    
    # Category Trend
    sys, user = format_category_trend_prompt(brand, "Cat", ["C1"], {"C1": 1}, {"C1": 10.0}, {"C1": 4.0})
    assert "Cat" in user
    
    # Risk
    sys, user = format_risk_assessment_prompt(brand, "domain.com", "leader", 5, "Cat", 5.0, 15.0, 4.5, 100, [], [])
    assert "leader" in user
    
    # Executive Summary
    sys, user = format_executive_summary_prompt(brand, "leader", 70.0, 5, 4.5, 100, 5.0, 15.0, "POS", "OPP", "RISK", "REC")
    assert "POS" in user

def test_registry_access():
    prompt = get_analysis_prompt(AnalysisPromptType.COMPETITIVE_POSITIONING)
    assert "system" in prompt
    assert "formatter" in prompt
    
    with pytest.raises(KeyError):
        get_analysis_prompt("unknown")

def test_list_analysis_prompts():
    prompts = list_analysis_prompts()
    assert "competitive_positioning" in prompts
