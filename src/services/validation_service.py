"""
Validation service for data validation and sanitization.

Provides comprehensive validation for all input and output data structures.
"""

import re
from typing import Any, Optional

from pydantic import ValidationError as PydanticValidationError

from src.models.schemas import BrandInput, AmazonProduct, ExtractionResult, AmazonCategory
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Custom validation error."""
    
    def __init__(self, code: str, message: str, details: Optional[dict] = None):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(message)


class ValidationService:
    """Service for validating all data structures in the pipeline."""
    
    BRAND_NAME_PATTERN = re.compile(r"^[\w\s\-\.&']+$", re.UNICODE)
    ASIN_PATTERN = re.compile(r"^[A-Z0-9]{10}$")
    
    def validate_brand_input(self, data: dict[str, Any]) -> BrandInput:
        """Validate and parse brand input data."""
        try:
            brand_input = BrandInput(**data)
        except Exception as e:
            logger.error("Brand input validation failed", error=str(e))
            raise ValidationError(
                code="INVALID_INPUT_FORMAT",
                message=f"Invalid input format: {str(e)}",
                details={"input": data},
            )
        
        # 'brand_name' might not be in BrandInput if it only has domain
        # The schema BrandInput currently only requires domain (see schemas.py I read earlier)
        # But if it has brand_name field:
        if hasattr(brand_input, 'brand_name'): 
             if not self.BRAND_NAME_PATTERN.match(brand_input.brand_name):
                raise ValidationError(
                    code="INVALID_BRAND_NAME",
                    message=f"Invalid brand name format: {brand_input.brand_name}",
                )
        
        # Category validation if present
        if hasattr(brand_input, 'category') and brand_input.category:
            if brand_input.category not in [c.value for c in AmazonCategory]:
                valid_categories = ", ".join([c.value for c in AmazonCategory])
                raise ValidationError(
                    code="INVALID_CATEGORY",
                    message=f"Invalid category '{brand_input.category}'. Valid: {valid_categories}",
                )
        
        return brand_input
    
    def validate_product_data(self, data: dict[str, Any]) -> Optional[AmazonProduct]:
        """Validate and parse product data."""
        try:
            return AmazonProduct(**data)
        except Exception as e:
            logger.warning("Product validation failed", error=str(e))
            return None
    
    def validate_asin(self, asin: str) -> bool:
        """Validate Amazon ASIN format."""
        return bool(self.ASIN_PATTERN.match(asin.upper()))
    
    def sanitize_text(self, text: str, max_length: int = 1000) -> str:
        """Sanitize and truncate text content."""
        sanitized = re.sub(r"<[^>]+>", "", text)
        sanitized = re.sub(r"\s+", " ", sanitized).strip()
        return sanitized[:max_length] if len(sanitized) > max_length else sanitized
    
    def validate_extraction_result(self, data: dict[str, Any]) -> ExtractionResult:
        """Validate complete extraction result structure."""
        try:
            return ExtractionResult(**data)
        except Exception as e:
            logger.error("Extraction result validation failed", error=str(e))
            raise ValidationError(
                code="INVALID_INPUT_FORMAT",
                message=f"Invalid extraction result: {e}",
                details={"error": str(e)},
            )
