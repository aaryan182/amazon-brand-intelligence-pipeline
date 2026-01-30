"""
Unit tests for the prompts module.

Tests cover:
- Prompt configuration
- Prompt formatting functions
- Prompt registry
- Edge cases and error handling
"""

import json
import pytest

from src.extractors.prompts import (
    # Configs
    PromptConfig,
    BRAND_EXTRACTION_CONFIG,
    PRODUCT_ENRICHMENT_CONFIG,
    CATEGORY_CLASSIFICATION_CONFIG,
    CONFIDENCE_ASSESSMENT_CONFIG,
    VALIDATION_RETRY_CONFIG,
    # Formatters
    format_brand_extraction_prompt,
    format_product_enrichment_prompt,
    format_category_classification_prompt,
    format_confidence_assessment_prompt,
    format_validation_retry_prompt,
    # Registry
    PROMPT_REGISTRY,
    get_prompt,
)


# =============================================================================
# Configuration Tests
# =============================================================================

class TestPromptConfigs:
    """Tests for prompt configuration objects."""
    
    def test_brand_extraction_config(self):
        """Test brand extraction configuration."""
        assert BRAND_EXTRACTION_CONFIG.name == "brand_data_extraction"
        assert BRAND_EXTRACTION_CONFIG.recommended_temperature == 0.3
        assert BRAND_EXTRACTION_CONFIG.recommended_max_tokens == 4096
        assert "claude" in BRAND_EXTRACTION_CONFIG.model.lower() or "sonnet" in BRAND_EXTRACTION_CONFIG.model.lower()
    
    def test_product_enrichment_config(self):
        """Test product enrichment configuration."""
        assert PRODUCT_ENRICHMENT_CONFIG.name == "product_detail_enrichment"
        assert PRODUCT_ENRICHMENT_CONFIG.recommended_temperature == 0.2
    
    def test_category_classification_config(self):
        """Test category classification configuration."""
        assert CATEGORY_CLASSIFICATION_CONFIG.name == "category_classification"
        assert CATEGORY_CLASSIFICATION_CONFIG.recommended_temperature == 0.3
    
    def test_confidence_assessment_config(self):
        """Test confidence assessment configuration."""
        assert CONFIDENCE_ASSESSMENT_CONFIG.name == "confidence_assessment"
        assert CONFIDENCE_ASSESSMENT_CONFIG.recommended_temperature == 0.2
    
    def test_validation_retry_config(self):
        """Test validation retry configuration."""
        assert VALIDATION_RETRY_CONFIG.name == "validation_retry"
        assert VALIDATION_RETRY_CONFIG.recommended_temperature == 0.1
    
    def test_all_configs_have_low_temperature(self):
        """Extraction tasks should use low temperature for consistency."""
        configs = [
            BRAND_EXTRACTION_CONFIG,
            PRODUCT_ENRICHMENT_CONFIG,
            CATEGORY_CLASSIFICATION_CONFIG,
            CONFIDENCE_ASSESSMENT_CONFIG,
            VALIDATION_RETRY_CONFIG,
        ]
        for config in configs:
            assert config.recommended_temperature <= 0.5, f"{config.name} should use low temperature"


# =============================================================================
# Formatter Tests
# =============================================================================

class TestFormatters:
    """Tests for prompt formatting functions."""
    
    def test_format_brand_extraction_string_results(self):
        """Test brand extraction prompt with string search results."""
        system, user = format_brand_extraction_prompt(
            brand_name="Nike",
            domain="nike.com",
            search_results="Product 1: Nike Air Max...",
        )
        
        assert "Amazon marketplace analyst" in system
        assert "Nike" in user
        assert "nike.com" in user
        assert "Product 1: Nike Air Max" in user
    
    def test_format_brand_extraction_list_results(self):
        """Test brand extraction prompt with list of dicts."""
        results = [
            {"title": "Nike Shoes", "price": 120.0},
            {"title": "Nike Shirt", "price": 40.0},
        ]
        
        system, user = format_brand_extraction_prompt(
            brand_name="Nike",
            domain="nike.com",
            search_results=results,
        )
        
        assert "Nike Shoes" in user
        assert "Nike Shirt" in user
        assert "120.0" in user
    
    def test_format_product_enrichment(self):
        """Test product enrichment prompt formatting."""
        product_data = {
            "asin": "B07XYZ1234",
            "title": "Nike Air Max",
            "price": None,
        }
        
        system, user = format_product_enrichment_prompt(
            brand_name="Nike",
            product_data=product_data,
        )
        
        assert "product data specialist" in system
        assert "B07XYZ1234" in user
        assert "Nike Air Max" in user
    
    def test_format_category_classification_with_list(self):
        """Test category classification with product list."""
        products = [
            {"title": "Running Shoes"},
            {"title": "Basketball Sneakers"},
        ]
        
        system, user = format_category_classification_prompt(
            brand_name="Nike",
            products_data=products,
        )
        
        assert "category taxonomy expert" in system
        assert "Running Shoes" in user
        assert "Basketball Sneakers" in user
    
    def test_format_confidence_assessment(self):
        """Test confidence assessment prompt formatting."""
        brand_data = {
            "products": [{"asin": "B123", "title": "Product"}],
            "has_official_store": True,
        }
        
        system, user = format_confidence_assessment_prompt(
            brand_name="Nike",
            brand_data=brand_data,
        )
        
        assert "brand verification specialist" in system
        assert "B123" in user
        assert "has_official_store" in user
    
    def test_format_validation_retry(self):
        """Test validation retry prompt formatting."""
        failed_output = '{"products": [incomplete...'
        errors = ["JSON parse error at line 1", "Missing closing bracket"]
        schema = {"products": "list", "confidence": "string"}
        
        system, user = format_validation_retry_prompt(
            brand_name="Nike",
            failed_output=failed_output,
            validation_errors=errors,
            expected_schema=schema,
        )
        
        assert "JSON validation" in system
        assert "incomplete" in user
        assert "JSON parse error" in user
        assert "Missing closing bracket" in user


# =============================================================================
# Registry Tests
# =============================================================================

class TestPromptRegistry:
    """Tests for prompt registry."""
    
    def test_registry_contains_all_prompts(self):
        """Test registry has all expected prompts."""
        expected_prompts = [
            "brand_extraction",
            "product_enrichment",
            "category_classification",
            "confidence_assessment",
            "validation_retry",
        ]
        
        for prompt_name in expected_prompts:
            assert prompt_name in PROMPT_REGISTRY
    
    def test_get_prompt_success(self):
        """Test successful prompt retrieval."""
        prompt = get_prompt("brand_extraction")
        
        assert "config" in prompt
        assert "system" in prompt
        assert "user_template" in prompt
        assert "formatter" in prompt
        assert callable(prompt["formatter"])
    
    def test_get_prompt_not_found(self):
        """Test error on missing prompt."""
        with pytest.raises(KeyError) as exc_info:
            get_prompt("nonexistent_prompt")
        
        assert "not found" in str(exc_info.value)
        assert "brand_extraction" in str(exc_info.value)  # Shows available
    
    def test_registry_entries_have_required_keys(self):
        """Test all registry entries have required keys."""
        required_keys = ["config", "system", "user_template", "formatter"]
        
        for name, entry in PROMPT_REGISTRY.items():
            for key in required_keys:
                assert key in entry, f"'{name}' missing key '{key}'"


# =============================================================================
# Content Quality Tests
# =============================================================================

class TestPromptQuality:
    """Tests for prompt content quality."""
    
    def test_brand_extraction_has_xml_tags(self):
        """Test brand extraction uses XML tags for structure."""
        prompt = get_prompt("brand_extraction")
        
        assert "<role>" in prompt["system"]
        assert "<task>" in prompt["user_template"]
        assert "<examples>" in prompt["user_template"]
        assert "<output_schema>" in prompt["user_template"]
    
    def test_prompts_have_examples(self):
        """Test prompts include few-shot examples."""
        # Validation retry is a correction prompt, not extraction - examples optional
        prompts_requiring_examples = [
            "brand_extraction",
            "product_enrichment", 
            "category_classification",
            "confidence_assessment",
        ]
        
        for name in prompts_requiring_examples:
            prompt = PROMPT_REGISTRY[name]
            user_template = prompt["user_template"].lower()
            # All extraction prompts should have examples
            assert "<examples>" in user_template or "example" in user_template, \
                f"'{name}' should include examples"
    
    def test_prompts_specify_json_output(self):
        """Test prompts specify JSON output format."""
        for name, prompt in PROMPT_REGISTRY.items():
            user_template = prompt["user_template"].lower()
            # Should mention JSON output
            assert "json" in user_template, \
                f"'{name}' should specify JSON output"
    
    def test_brand_extraction_has_edge_cases(self):
        """Test brand extraction handles edge cases."""
        system = PROMPT_REGISTRY["brand_extraction"]["system"]
        
        assert "edge_cases" in system.lower() or "missing" in system.lower()
        assert "third-party" in system.lower() or "third party" in system.lower()
