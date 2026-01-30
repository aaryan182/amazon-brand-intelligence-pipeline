"""
Demo Runner Script for Amazon Brand Intelligence Pipeline.

Tests the pipeline on 3 brands with different expected product counts:
- Large brand: "patagonia.com" (100+ products)
- Medium brand: "allbirds.com" (20-50 products)
- Small/No presence: "localcoffeeroasters.com" (0-5 products)

Reports are saved to outputs/demo_reports/
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.orchestrator import BrandIntelligencePipeline, PipelineError
from src.utils.formatters import ReportFormatter
from src.utils.logger import get_logger, setup_logging

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)


# Demo brands to test
DEMO_BRANDS = [
    {
        "domain": "patagonia.com",
        "description": "Large outdoor apparel brand (expected: 100+ products)",
        "expected_range": "100+",
    },
    {
        "domain": "allbirds.com", 
        "description": "Medium sustainable footwear brand (expected: 20-50 products)",
        "expected_range": "20-50",
    },
    {
        "domain": "localcoffeeroasters.com",
        "description": "Small/local brand (expected: 0-5 products)",
        "expected_range": "0-5",
    },
]

# Output directory for demo reports
OUTPUT_DIR = Path("outputs/demo_reports")


def progress_callback(percent: int, message: str) -> None:
    """Callback to display progress updates."""
    bar_length = 30
    filled = int(bar_length * percent / 100)
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"\r  [{bar}] {percent:3d}% - {message}", end="", flush=True)


async def analyze_brand(pipeline: BrandIntelligencePipeline, brand_info: dict) -> dict:
    """
    Analyze a single brand and return results summary.
    
    Args:
        pipeline: Initialized pipeline instance
        brand_info: Dict with domain, description, expected_range
        
    Returns:
        Dict with analysis results or error info
    """
    domain = brand_info["domain"]
    start_time = datetime.now()
    
    try:
        print(f"\n  Starting analysis...")
        result = await pipeline.run(domain)
        
        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()
        
        # Use the formatter to save the report
        formatter = ReportFormatter()
        
        # Save JSON version
        result.report_format = "json"
        saved_path = await formatter.save_report(
            final_report=result,
            output_dir=OUTPUT_DIR
        )
        
        # Also save markdown version
        result.report_format = "markdown"
        md_path = await formatter.save_report(
            final_report=result,
            output_dir=OUTPUT_DIR
        )
        
        # Extract key metrics
        extraction = result.extraction_result
        analysis = result.strategic_insight
        
        product_count = len(extraction.top_products) if extraction and extraction.top_products else 0
        has_presence = extraction.amazon_presence.found if extraction and extraction.amazon_presence else False
        
        # Handle both enum and string cases for confidence
        confidence = "N/A"
        if extraction and extraction.amazon_presence and extraction.amazon_presence.confidence:
            conf_obj = extraction.amazon_presence.confidence
            confidence = conf_obj.value if hasattr(conf_obj, 'value') else str(conf_obj)
        
        return {
            "status": "success",
            "domain": domain,
            "duration_seconds": round(duration, 2),
            "product_count": product_count,
            "amazon_presence": has_presence,
            "confidence": confidence,
            "report_path": str(saved_path),
            "markdown_path": str(md_path),
            "expected_range": brand_info["expected_range"],
        }
        
    except PipelineError as e:
        duration = (datetime.now() - start_time).total_seconds()
        return {
            "status": "error",
            "domain": domain,
            "duration_seconds": round(duration, 2),
            "error_message": str(e),
            "error_type": "PipelineError",
            "expected_range": brand_info["expected_range"],
        }
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        return {
            "status": "error",
            "domain": domain,
            "duration_seconds": round(duration, 2),
            "error_message": str(e),
            "error_type": type(e).__name__,
            "expected_range": brand_info["expected_range"],
        }


async def run_demo():
    """
    Run the demo analysis on all test brands.
    """
    print("\n" + "=" * 70)
    print("  Amazon Brand Intelligence Pipeline - Demo Runner")
    print("=" * 70)
    print(f"\n  Testing {len(DEMO_BRANDS)} brands...")
    print(f"  Reports will be saved to: {OUTPUT_DIR.absolute()}")
    print()
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    # Create pipeline with context manager
    async with BrandIntelligencePipeline(
        progress_callback=progress_callback,
        enable_retries=True,
        max_retries=3,
    ) as pipeline:
        
        for i, brand_info in enumerate(DEMO_BRANDS, 1):
            domain = brand_info["domain"]
            description = brand_info["description"]
            
            print(f"\n{'=' * 70}")
            print(f"  [{i}/{len(DEMO_BRANDS)}] Analyzing: {domain}")
            print(f"  {description}")
            print("=" * 70)
            
            result = await analyze_brand(pipeline, brand_info)
            results.append(result)
            
            # Print result summary
            print()  # New line after progress bar
            if result["status"] == "success":
                print(f"\n  ✓ SUCCESS")
                print(f"    Duration: {result['duration_seconds']}s")
                print(f"    Products Found: {result['product_count']} (expected: {result['expected_range']})")
                print(f"    Amazon Presence: {result['amazon_presence']}")
                print(f"    Confidence: {result['confidence']}")
                print(f"    Report: {result['report_path']}")
            else:
                print(f"\n  ✗ ERROR: {result['error_type']}")
                print(f"    Duration: {result['duration_seconds']}s")
                print(f"    Message: {result['error_message'][:100]}...")
    
    # Print summary
    print("\n" + "=" * 70)
    print("  DEMO SUMMARY")
    print("=" * 70)
    
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]
    
    print(f"\n  Total brands analyzed: {len(results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    
    if successful:
        print("\n  Successful analyses:")
        for r in successful:
            print(f"    - {r['domain']}: {r['product_count']} products ({r['duration_seconds']}s)")
    
    if failed:
        print("\n  Failed analyses:")
        for r in failed:
            print(f"    - {r['domain']}: {r['error_type']} - {r['error_message'][:50]}...")
    
    print(f"\n  Reports saved to: {OUTPUT_DIR.absolute()}")
    print("=" * 70 + "\n")
    
    return results


if __name__ == "__main__":
    try:
        results = asyncio.run(run_demo())
        
        # Exit with error code if any failed
        failed_count = sum(1 for r in results if r["status"] != "success")
        sys.exit(failed_count)
        
    except KeyboardInterrupt:
        print("\n\n  Demo interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n  Fatal error: {e}")
        logger.exception("Demo runner failed")
        sys.exit(1)
