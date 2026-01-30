import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from src.services.llm_service import ClaudeService, TokenUsage, SchemaValidationError, ClaudeServiceError
from src.models.schemas import ExtractionResult, StrategicInsight, AmazonProduct, AmazonPresence, ConfidenceLevel

@pytest.mark.asyncio
async def test_llm_service_init(mock_settings):
    service = ClaudeService(settings=mock_settings)
    assert service.settings == mock_settings
    assert service.client is not None

@pytest.mark.asyncio
async def test_extract_structured_data(mock_settings):
    service = ClaudeService(settings=mock_settings)
    
    # Mocking _call_api to return a JSON string
    mock_json_response = json.dumps({
        "brand_name": "Test Brand",
        "asin": "B001",
        "price": 99.99
    })
    
    with patch.object(service, "_call_api", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = (mock_json_response, TokenUsage(input_tokens=10, output_tokens=10))
        
        # Test with a simple schema
        class SimpleBrand(pytest.importorskip("pydantic").BaseModel):
            brand_name: str
            asin: str
            price: float
            
        result = await service.extract_structured_data("some text", SimpleBrand)
        
        assert isinstance(result, SimpleBrand)
        assert result.brand_name == "Test Brand"
        assert result.price == 99.99
        mock_call.assert_called_once()

@pytest.mark.asyncio
async def test_generate_analysis(mock_settings, sample_extraction_result, sample_strategic_insight):
    service = ClaudeService(settings=mock_settings)
    
    # Mock return value for analysis
    mock_json_response = sample_strategic_insight.model_dump_json()
    
    with patch.object(service, "_call_api", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = (mock_json_response, TokenUsage(input_tokens=10, output_tokens=10))
        
        analysis = await service.generate_analysis(sample_extraction_result)
        
        assert isinstance(analysis, StrategicInsight)
        assert analysis.market_position == sample_strategic_insight.market_position
        mock_call.assert_called_once()

@pytest.mark.asyncio
async def test_schema_validation_error(mock_settings):
    service = ClaudeService(settings=mock_settings)
    
    # Return invalid JSON or JSON that doesn't match schema
    mock_invalid_json = json.dumps({"wrong_field": "data"})
    
    with patch.object(service, "_call_api", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = (mock_invalid_json, TokenUsage(input_tokens=10, output_tokens=10))
        
        class StrictSchema(pytest.importorskip("pydantic").BaseModel):
            required_field: str
            
        with pytest.raises(Exception): # validate_and_retry will eventually fail after retries
            await service.extract_structured_data("text", StrictSchema)

@pytest.mark.asyncio
async def test_call_api_success(mock_settings):
    service = ClaudeService(settings=mock_settings)
    
    # Mock anthropic client
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Response text")]
    mock_response.usage.input_tokens = 50
    mock_response.usage.output_tokens = 30
    
    with patch.object(service.client.messages, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_response
        
        text, usage = await service._call_api([{"role": "user", "content": "hi"}])
        
        assert text == "Response text"
        assert usage.input_tokens == 50
        assert usage.output_tokens == 30
        assert service.total_cost > 0

@pytest.mark.asyncio
async def test_call_api_retry_on_server_error(mock_settings):
    service = ClaudeService(settings=mock_settings)
    service.max_retries = 2
    
    # Mock anthropic client to fail once then succeed
    from anthropic import APIStatusError
    mock_error = APIStatusError("Server error", response=MagicMock(status_code=500), body={})
    
    mock_success_response = MagicMock()
    mock_success_response.content = [MagicMock(text="Success")]
    mock_success_response.usage.input_tokens = 10
    mock_success_response.usage.output_tokens = 10
    
    with patch.object(service.client.messages, "create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = [mock_error, mock_success_response]
        
        with patch("asyncio.sleep", new_callable=AsyncMock): # Speed up test
            text, usage = await service._call_api([{"role": "user", "content": "hi"}])
            
            assert text == "Success"
            assert mock_create.call_count == 2

@pytest.mark.asyncio
async def test_call_api_auth_error_no_retry(mock_settings):
    service = ClaudeService(settings=mock_settings)
    
    from anthropic import APIStatusError
    mock_error = APIStatusError("Unauthorized", response=MagicMock(status_code=401), body={})
    
    with patch.object(service.client.messages, "create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = mock_error
        
        with pytest.raises(ClaudeServiceError, match="Authentication failed"):
            await service._call_api([{"role": "user", "content": "hi"}])
        
        assert mock_create.call_count == 1

@pytest.mark.asyncio
async def test_call_api_cache_hit(mock_settings):
    service = ClaudeService(settings=mock_settings)
    service.enable_cache = True
    
    # First call to populate cache
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Cached text")]
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 10
    
    with patch.object(service.client.messages, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_response
        
        await service._call_api([{"role": "user", "content": "hi"}], use_cache=True)
        assert mock_create.call_count == 1
        
        # Second call should hit cache
        text, usage = await service._call_api([{"role": "user", "content": "hi"}], use_cache=True)
        assert text == "Cached text"
        assert mock_create.call_count == 1 # Still 1

def test_token_usage_calculation():
    usage = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
    # Model specific costs may vary, checking basic behavior
    usage.calculate_cost("claude-3-5-sonnet-20241022")
    assert usage.estimated_cost > 0
    assert usage.total_tokens == 150
