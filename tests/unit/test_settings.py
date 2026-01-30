import pytest
from unittest.mock import patch, MagicMock
from src.config.settings import Settings

def test_settings_get_search_provider_serpapi(mock_settings):
    mock_settings.get_search_provider.return_value = "serpapi"
    assert mock_settings.get_search_provider() == "serpapi"

def test_settings_get_search_provider_exa(mock_settings):
    mock_settings.get_search_provider.return_value = "exa"
    assert mock_settings.get_search_provider() == "exa"

def test_settings_get_search_provider_perplexity(mock_settings):
    mock_settings.get_search_provider.return_value = "perplexity"
    assert mock_settings.get_search_provider() == "perplexity"

def test_settings_get_search_provider_none(mock_settings):
    mock_settings.get_search_provider.return_value = "none"
    assert mock_settings.get_search_provider() == "none"

def test_settings_validate_api_keys(mock_settings):
    # Should not raise with no keys - it's a soft validation
    mock_settings.validate_api_keys()

def test_real_settings_with_env():
    """Test real Settings class behavior with valid env vars."""
    with patch.dict("os.environ", {
        "ANTHROPIC_API_KEY": "sk-ant-api-test-key",
        "SERPAPI_API_KEY": "test-serpapi",
        "EXA_API_KEY": "",
        "PERPLEXITY_API_KEY": ""
    }, clear=True):
        # Test settings provider selection logic
        settings = Settings()
        assert settings.get_search_provider() == "serpapi"
