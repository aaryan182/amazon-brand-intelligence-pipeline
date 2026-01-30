import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from click.testing import CliRunner
from src.main import cli

def test_cli_setup_logger():
    from src.main import setup_logger
    # Testing it doesn't crash
    setup_logger(verbose=True)
    setup_logger(verbose=False)

def test_cli_analyze_error_exit():
    runner = CliRunner()
    with patch("src.main.get_settings", side_effect=Exception("Config Error")):
        result = runner.invoke(cli, ["analyze", "test.com"])
        assert result.exit_code == 1
        assert "Error:" in result.output

def test_cli_batch_empty_file(tmp_path):
    runner = CliRunner()
    f = tmp_path / "empty.txt"
    f.write_text("")
    result = runner.invoke(cli, ["batch", str(f)])
    assert result.exit_code == 1
    assert "No domains found" in result.output

def test_cli_test_extraction_fail():
    runner = CliRunner()
    with patch("src.main.BrandIntelligencePipeline.run", side_effect=Exception("Fail")):
        # We need to mock extractor.extract_brand_data for test_extraction command
        with patch("src.main.BrandIntelligencePipeline.extractor") as mock_ext:
            mock_ext.extract_brand_data = AsyncMock(side_effect=Exception("Ext Fail"))
            result = runner.invoke(cli, ["test-extraction", "test.com"])
            # Exit code depends on how async_command handles exceptions and sys.exit(1)
            # The code has try-except sys.exit(1)
            assert result.exit_code == 1

def test_cli_validate_setup_fail():
    runner = CliRunner()
    with patch("src.main.get_settings", side_effect=Exception("Settings Fail")):
        result = runner.invoke(cli, ["validate-setup"])
        assert result.exit_code == 1
