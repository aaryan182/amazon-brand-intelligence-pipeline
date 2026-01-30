import pytest
from src.services.validation_service import ValidationService, ValidationError
from src.models.schemas import AmazonCategory

@pytest.fixture
def service():
    return ValidationService()

def test_validate_brand_input_valid(service):
    data = {"domain": "nike.com", "brand_name": "Nike"}
    result = service.validate_brand_input(data)
    assert result.domain == "nike.com"
    assert result.brand_name == "Nike"

def test_validate_brand_input_invalid_format(service):
    with pytest.raises(ValidationError) as exc:
        service.validate_brand_input(None) # type: ignore
    assert exc.value.code == "INVALID_INPUT_FORMAT"

def test_validate_brand_input_invalid_name(service):
    data = {"domain": "test.com", "brand_name": "Invalid @ Name"}
    with pytest.raises(ValidationError) as exc:
        service.validate_brand_input(data)
    assert exc.value.code == "INVALID_BRAND_NAME"

def test_validate_brand_input_invalid_category(service):
    data = {"domain": "test.com", "category": "invalid-category"}
    with pytest.raises(ValidationError) as exc:
        service.validate_brand_input(data)
    assert exc.value.code == "INVALID_INPUT_FORMAT"

def test_validate_product_data_valid(service):
    data = {
        "asin": "B001234567",
        "title": "Test Product",
        "price": 99.99,
        "rating": 4.5,
        "review_count": 100,
        "url": "https://amazon.com/dp/B001234567"
    }
    result = service.validate_product_data(data)
    assert result is not None
    assert result.asin == "B001234567"

def test_validate_product_data_invalid(service):
    data = {"asin": "short"}
    result = service.validate_product_data(data)
    assert result is None

def test_validate_asin(service):
    assert service.validate_asin("B001234567") is True
    assert service.validate_asin("b001234567") is True
    assert service.validate_asin("SHORT") is False
    assert service.validate_asin("TOO_LONG_123") is False

def test_sanitize_text(service):
    html = "<p>Hello <b>World</b></p>"
    assert service.sanitize_text(html) == "Hello World"
    
    long_text = "a" * 2000
    assert len(service.sanitize_text(long_text, max_length=100)) == 100

def test_validate_extraction_result_invalid(service):
    with pytest.raises(ValidationError):
        service.validate_extraction_result({"invalid": "data"})
