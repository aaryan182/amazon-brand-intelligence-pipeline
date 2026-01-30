"""
Report formatting utilities.

Provides formatters for generating client-ready reports in multiple formats (Markdown, HTML, PDF).
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Union

# Optional dependencies for HTML/PDF generation
try:
    import markdown2
except ImportError:
    markdown2 = None

import sys
import os
from contextlib import contextmanager

@contextmanager
def suppress_stderr():
    """Suppress stderr/stdout during execution."""
    try:
        with open(os.devnull, "w") as devnull:
            old_stderr = sys.stderr
            sys.stderr = devnull
            try:
                yield
            finally:
                sys.stderr = old_stderr
    except Exception:
        # Fallback if devnull cannot be opened
        yield

try:
    with suppress_stderr():
        from weasyprint import HTML, CSS
except (ImportError, OSError, Exception):
    HTML = None
    CSS = None

from src.models.schemas import (
    AmazonProduct,
    ExtractionResult,
    StrategicInsight,
    FinalReport,
    Recommendation
)
from src.config.settings import get_settings

logger = logging.getLogger(__name__)


def format_extraction_table(extraction: ExtractionResult) -> str:
    """
    Create formatted markdown table for extraction metrics.
    
    | Metric | Value |
    |--------|-------|
    | Amazon Presence | Yes (High Confidence) |
    | Product Count | ~150 |
    | Primary Category | Sports & Outdoors |
    | Avg Rating | 4.5 ⭐ |
    """
    presence_icon = "✅" if extraction.amazon_presence.found else "❌"
    presence_str = f"{'Yes' if extraction.amazon_presence.found else 'No'} ({str(extraction.amazon_presence.confidence).title()} Confidence)"
    
    avg_rating = extraction.average_rating or 0.0
    rating_str = f"{avg_rating:.1f} ⭐" if avg_rating > 0 else "N/A"
    
    rows = [
        f"| Amazon Presence | {presence_str} |",
        f"| Product Count | ~{extraction.estimated_product_count} |",
        f"| Primary Category | {extraction.primary_category or 'N/A'} |",
        f"| Avg Rating | {rating_str} |",
        f"| Total Reviews | {extraction.total_reviews:,} |"
    ]
    
    header = "| Metric | Value |\n|--------|-------|"
    return header + "\n" + "\n".join(rows)


def format_top_products_table(products: List[AmazonProduct]) -> str:
    """
    Create formatted markdown table for top products.
    
    | Rank | Product | Price | Rating | Reviews |
    |------|---------|-------|--------|---------|
    | 1 | Product Name | $89.99 | 4.5⭐ | 1,234 |
    """
    if not products:
        return "*No products found.*"
        
    header = "| Rank | Product | Price | Rating | Reviews |\n|------|---------|-------|--------|---------|"
    rows = []
    
    for i, product in enumerate(products[:10], 1):
        price_str = f"${product.price:.2f}" if product.price else "-"
        rating_str = f"{product.rating}⭐" if product.rating else "-"
        reviews_str = f"{product.review_count:,}"
        
        # Truncate title if too long
        title = product.title[:60] + "..." if len(product.title) > 60 else product.title
        title = title.replace("|", "-") # Escape pipes
        
        rows.append(f"| {i} | {title} | {price_str} | {rating_str} | {reviews_str} |")
        
    return header + "\n" + "\n".join(rows)


def format_recommendations_table(recommendations: List[Recommendation]) -> str:
    """
    Create formatted markdown table for recommendations.
    
    | Priority | Recommendation | Impact | Difficulty | Timeline |
    |----------|----------------|--------|------------|----------|
    """
    if not recommendations:
        return "*No recommendations available.*"
        
    header = "| Priority | Recommendation | Impact | Difficulty | Timeline |\n|----------|----------------|--------|------------|----------|"
    rows = []
    
    # Sort by priority
    sorted_recs = sorted(recommendations, key=lambda x: x.priority)
    
    for rec in sorted_recs:
        title = rec.title.replace("|", "-")
        impact = rec.impact.replace("|", "-")
        difficulty = rec.difficulty.replace("|", "-")
        timeline = rec.timeline.replace("|", "-")
        
        rows.append(f"| {rec.priority} | {title} | {impact} | {difficulty} | {timeline} |")
        
    return header + "\n" + "\n".join(rows)


def generate_final_report(
    extraction: ExtractionResult,
    insights: StrategicInsight,
    format: str = "markdown"
) -> str:
    """
    Generate the complete final report content.
    
    Structure:
    # Amazon Brand Intelligence Report: {brand_name}
    ## Executive Summary
    ## Amazon Presence Overview
    ## Top Products
    ## Competitive Positioning
    ## Market Opportunities
    ## Strategic Recommendations
    ## Category Analysis
    """
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    
    # Helper for positioning text
    pos_text = insights.market_position_rationale
    
    # Helper for opportunities
    opps_text = ""
    if hasattr(insights.swot, 'opportunities'):
        for opp in insights.swot.opportunities:
            opps_text += f"- {opp}\n"
    
    # Determine recommendations source
    recs = insights.recommendations
    
    # Tables
    extraction_table = format_extraction_table(extraction)
    products_table = format_top_products_table(extraction.top_products)
    
    if recs:
        recs_table = format_recommendations_table(recs)
    else:
        # Fallback if no rich recommendations
        recs_table = "### Growth Recommendations\n"
        for rec in insights.growth_recommendations:
            recs_table += f"- {rec}\n"
            
    # Category Context (from extraction & insights)
    cat_analysis = f"Primary Category: **{extraction.primary_category}**\n\n"
    if extraction.all_categories:
        cat_analysis += "Related Categories:\n" + "\n".join([f"- {c}" for c in extraction.all_categories[:5]])
    
    report_content = f"""# Amazon Brand Intelligence Report: {extraction.brand_name}

## Executive Summary
{insights.executive_summary}

## Amazon Presence Overview
{extraction_table}

## Top Products
{products_table}

## Competitive Positioning
**Market Position:** {str(insights.market_position).title().replace('Marketposition.', '')}

{pos_text}

### SWOT Analysis
**Strengths:**
{chr(10).join(['- ' + s for s in insights.swot.strengths])}

**Weaknesses:**
{chr(10).join(['- ' + w for w in insights.swot.weaknesses])}

## Market Opportunities
{opps_text}

## Strategic Recommendations
{recs_table}

## Category Analysis
{cat_analysis}

---
Generated on: {timestamp}
Data Source: Amazon Marketplace
"""
    return report_content


def save_report(report: str, output_path: Path, format: str = "markdown") -> Path:
    """
    Save report to file, optionally converting format.
    
    Args:
        report: Markdown content
        output_path: Destination path (without extension, or with)
        format: 'markdown', 'html', or 'pdf'
    """
    # Ensure parent exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    base_path = output_path.with_suffix("")
    
    if format == "markdown":
        file_path = base_path.with_suffix(".md")
        file_path.write_text(report, encoding="utf-8")
        logger.info(f"Saved Markdown report to {file_path}")
        return file_path
        
    elif format == "html":
        if not markdown2:
            logger.warning("markdown2 not installed, saving as markdown instead.")
            return save_report(report, base_path, "markdown")
            
        html_content = markdown2.markdown(
            report, 
            extras=["tables", "fenced-code-blocks", "header-ids", "break-on-newline"]
        )
        
        # Wrap in basic styling
        styled_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 900px; margin: 0 auto; padding: 40px; line-height: 1.6; color: #333; }}
table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
th {{ background-color: #f5f5f5; }}
h1, h2, h3 {{ color: #2c3e50; margin-top: 30px; }}
h1 {{ border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }}
code {{ background: #f5f5f5; padding: 2px 5px; border-radius: 3px; }}
</style>
</head>
<body>
{html_content}
</body>
</html>
"""
        file_path = base_path.with_suffix(".html")
        file_path.write_text(styled_html, encoding="utf-8")
        logger.info(f"Saved HTML report to {file_path}")
        return file_path
        
    elif format == "pdf":
        if not (markdown2 and HTML):
            logger.warning("markdown2 or weasyprint not installed, saving as markdown instead.")
            return save_report(report, base_path, "markdown")
            
        # First generate HTML
        html_path = save_report(report, base_path, "html")
        
        # Then convert to PDF
        pdf_path = base_path.with_suffix(".pdf")
        HTML(filename=str(html_path)).write_pdf(str(pdf_path))
        logger.info(f"Saved PDF report to {pdf_path}")
        return pdf_path
        
    else:
        raise ValueError(f"Unsupported format: {format}")


class ReportFormatter:
    """
    Format and generate reports. 
    Kept for backward compatibility and integration with Orchestrator.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        settings = get_settings()
        self.output_dir = output_dir or settings.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    async def generate_report(
        self,
        extraction_result: ExtractionResult,
        strategic_insight: StrategicInsight,
        format_type: str = "markdown",
    ) -> Path:
        """
        Generate a report in the specified format using the new utility functions.
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        brand_name = extraction_result.brand_name.lower().replace(" ", "_")
        filename = f"{brand_name}_{timestamp}"
        output_path = self.output_dir / filename
        
        if format_type == "json":
            # Handled separately as it's just a dump
            return self._generate_json_report(extraction_result, strategic_insight, filename)
        
        # Generate content
        report_content = generate_final_report(extraction_result, strategic_insight, format_type)
        
        # Save file
        return save_report(report_content, output_path, format_type)

    async def save_report(
        self,
        final_report: FinalReport,
        output_dir: Path,
    ) -> Path:
        """
        Save a FinalReport object to file.
        Reconstructs content from sections if needed.
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        brand_name = final_report.brand_name.lower().replace(" ", "_")
        filename = f"{brand_name}_{timestamp}"
        output_path = output_dir / filename
        
        format_type = final_report.report_format
        
        if format_type == "json":
            file_path = output_path.with_suffix(".json")
            file_path.write_text(final_report.to_json(), encoding="utf-8")
            logger.info(f"Saved JSON report to {file_path}")
            return file_path
        
        # Reconstruct content from sections
        content = ""
        # Sort sections by order
        sorted_sections = sorted(final_report.sections, key=lambda x: x.order)
        
        # Add Title
        content += f"# {final_report.title}\n\n"
        
        for section in sorted_sections:
            content += f"## {section.title}\n\n"
            content += f"{section.content}\n\n"
            
        content += f"---\nGenerated by {final_report.generated_by}"
            
        return save_report(content, output_path, format_type)

    def _generate_json_report(
        self,
        extraction_result: ExtractionResult,
        strategic_insight: StrategicInsight,
        filename: str,
    ) -> Path:
        """Generate JSON format report."""
        report = FinalReport(
            title=f"Brand Intelligence Report: {extraction_result.brand_name}",
            brand_name=extraction_result.brand_name,
            domain=extraction_result.domain,
            extraction_result=extraction_result,
            strategic_insight=strategic_insight,
        )
        
        output_path = self.output_dir / f"{filename}.json"
        output_path.write_text(report.to_json(), encoding="utf-8")
        logger.info(f"Generated JSON report at {output_path}")
        return output_path
