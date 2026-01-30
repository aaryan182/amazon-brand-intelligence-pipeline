import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from src.utils.formatters import (
    format_extraction_table,
    format_top_products_table,
    format_recommendations_table,
    generate_final_report,
    save_report
)

def test_format_extraction_table(sample_extraction_result):
    table = format_extraction_table(sample_extraction_result)
    assert "| Metric | Value |" in table
    assert "| Amazon Presence | Yes (High Confidence) |" in table
    assert "| Product Count | ~3 |" in table

def test_format_top_products_table(sample_products):
    table = format_top_products_table(sample_products)
    assert "Test Product 1" in table
    assert "$29.99" in table
    assert "4.5⭐" in table

def test_format_recommendations_table(sample_strategic_insight):
    table = format_recommendations_table(sample_strategic_insight.recommendations)
    assert "Boost Ad Spend" in table
    assert "Immediate" in table

def test_generate_final_report_content(sample_extraction_result, sample_strategic_insight):
    report = generate_final_report(sample_extraction_result, sample_strategic_insight)
    assert "# Amazon Brand Intelligence Report: Test Brand" in report
    assert "## Executive Summary" in report
    assert "## Strategic Recommendations" in report
    assert "Test Brand shows promise" in report

@patch("src.utils.formatters.Path.write_text")
@patch("src.utils.formatters.Path.mkdir")
def test_save_report_markdown(mock_mkdir, mock_write, tmp_path):
    output = tmp_path / "test_report"
    content = "# Test Report"
    
    saved_path = save_report(content, output, "markdown")
    
    assert saved_path == output.with_suffix(".md")
    mock_write.assert_called_once()
    args = mock_write.call_args[0]
    assert args[0] == content

@patch("src.utils.formatters.markdown2")
@patch("src.utils.formatters.Path.write_text")
@patch("src.utils.formatters.Path.mkdir")
def test_save_report_html(mock_mkdir, mock_write, mock_markdown2, tmp_path):
    mock_markdown2.markdown.return_value = "<h1>Test</h1>"

    output = tmp_path / "test_report"
    content = "# Test Report"
    
    saved_path = save_report(content, output, "html")
    
    assert saved_path == output.with_suffix(".html")
    html_content = mock_write.call_args_list[0][0][0]
    assert "<!DOCTYPE html>" in html_content
    assert "<h1>Test</h1>" in html_content
@patch("src.utils.formatters.HTML")
@patch("src.utils.formatters.markdown2")
@patch("src.utils.formatters.Path.write_text")
@patch("src.utils.formatters.Path.mkdir")
def test_save_report_pdf(mock_mkdir, mock_write, mock_markdown2, mock_html, tmp_path):
    mock_markdown2.markdown.return_value = "<h1>Test</h1>"
    mock_pdf_instance = MagicMock()
    mock_html.return_value = mock_pdf_instance

    output = tmp_path / "test_report"
    content = "# Test Report"
    
    saved_path = save_report(content, output, "pdf")
    
    # PDF generation involves HTML intermediate
    assert saved_path == output.with_suffix(".pdf")
    assert mock_html.called

def test_save_report_invalid_format(tmp_path):
    with pytest.raises(ValueError, match="Unsupported format"):
        save_report("content", tmp_path / "test", "invalid")

@pytest.mark.asyncio
async def test_report_formatter_class(sample_extraction_result, sample_strategic_insight, tmp_path):
    from src.utils.formatters import ReportFormatter
    
    formatter = ReportFormatter(output_dir=tmp_path)
    
    with patch.object(formatter, "_generate_json_report") as mock_json:
        mock_json.return_value = tmp_path / "test.json"
        path = await formatter.generate_report(sample_extraction_result, sample_strategic_insight, "json")
        assert path.suffix == ".json"
        assert mock_json.called

    with patch("src.utils.formatters.save_report") as mock_save:
        mock_save.return_value = tmp_path / "test.md"
        path = await formatter.generate_report(sample_extraction_result, sample_strategic_insight, "markdown")
        assert mock_save.called
