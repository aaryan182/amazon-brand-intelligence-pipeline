"""
Amazon Brand Intelligence Pipeline - CLI Entry Point.
Production-grade CLI using Click and Rich.
"""

import sys
import asyncio
import json
import logging
from pathlib import Path
from functools import wraps
from typing import Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table
from rich.logging import RichHandler

from src.config.settings import get_settings, Settings
from src.pipeline.orchestrator import BrandIntelligencePipeline
from src.models.schemas import ExtractionResult, StrategicInsight

# Initialize Rich Console
console = Console()

# =============================================================================
# Helper Functions
# =============================================================================

def async_command(f):
    """Decorator to run async click commands."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper

def setup_logger(verbose: bool):
    """Configure logging based on verbosity."""
    level = "DEBUG" if verbose else "WARNING"
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )
    # Silence third-party libs
    logging.getLogger("httpx").setLevel(logging.WARNING)

# =============================================================================
# CLI Group
# =============================================================================

@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Amazon Brand Intelligence Pipeline"""
    pass

# =============================================================================
# Commands
# =============================================================================

@cli.command()
@click.argument('domain')
@click.option('--output-dir', default='outputs/reports', help='Custom output directory')
@click.option('--format', type=click.Choice(['markdown', 'html', 'json']), default='markdown', help='Output format')
@click.option('--verbose', is_flag=True, help='Detailed logging')
@click.option('--save-json', is_flag=True, help='Save intermediate JSON')
@async_command
async def analyze(domain: str, output_dir: str, format: str, verbose: bool, save_json: bool):
    """
    Analyze a brand's Amazon presence.
    
    DOMAIN: The brand's domain (e.g., patagonia.com)
    """
    setup_logger(verbose)
    
    console.print(Panel.fit(f"[bold blue]Amazon Brand Intelligence Analysis[/bold blue]\nTarget: [cyan]{domain}[/cyan]"))
    
    start_time = asyncio.get_event_loop().time()
    
    try:
        settings = get_settings()
        # Override output dir if provided
        object.__setattr__(settings, 'output_dir', Path(output_dir))
        object.__setattr__(settings, 'report_format', format)
        
        async with BrandIntelligencePipeline(settings=settings) as pipeline:
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                
                # We can update progress via callback if orchestrator supported it properly
                # For now, we show indefinite spinner for phases
                
                task = progress.add_task("[cyan]Running pipeline...", total=None)
                
                # Hook into pipeline progress callback if we updated orchestrator to accept it
                # The orchestrator accepts progress_callback in init or run helper
                # But we are using context manager directly.
                
                # Let's attach a simplified callback
                def update_progress(pct, msg):
                    progress.update(task, description=f"[cyan]{msg}")
                
                pipeline.progress_callback = update_progress
                
                report = await pipeline.run(domain)
                
                progress.update(task, completed=True, description="[green]Analysis complete!")
        
        # Display Summary
        duration = asyncio.get_event_loop().time() - start_time
        
        table = Table(title="Analysis Summary", show_header=False)
        table.add_row("Brand Name", report.brand_name)
        table.add_row("Report ID", str(report.report_id))
        table.add_row("Status", "[green]Success[/green]")
        table.add_row("Duration", f"{duration:.2f}s")
        table.add_row("Output", str(report.sections[0].title if report.sections else "Generated"))
        
        console.print(table)
        console.print(f"[green]✓[/green] Report generated successfully.")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        if verbose:
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--output-dir', default='outputs/reports', help='Output directory')
@click.option('--concurrency', default=3, help='Max concurrent analyses')
@async_command
async def batch(file_path: str, output_dir: str, concurrency: int):
    """
    Process multiple brands from a file.
    
    FILE_PATH: Text file with one domain per line.
    """
    setup_logger(False)
    
    domains = [line.strip() for line in Path(file_path).read_text().splitlines() if line.strip()]
    
    if not domains:
        console.print("[red]No domains found in file.[/red]")
        sys.exit(1)
        
    console.print(f"[bold]Batch Processing [cyan]{len(domains)}[/cyan] brands with concurrency [cyan]{concurrency}[/cyan][/bold]")
    
    settings = get_settings()
    object.__setattr__(settings, 'output_dir', Path(output_dir))
    
    semaphore = asyncio.Semaphore(concurrency)
    
    async def process_one(domain):
        async with semaphore:
            try:
                # New pipeline instance per run to ensure isolation
                async with BrandIntelligencePipeline(settings=settings) as pipeline:
                    await pipeline.run(domain)
                return domain, True, None
            except Exception as e:
                return domain, False, str(e)

    with Progress(console=console) as progress:
        task = progress.add_task("[cyan]Processing...", total=len(domains))
        
        results = []
        # Create tasks
        tasks = [process_one(d) for d in domains]
        
        for coro in asyncio.as_completed(tasks):
            domain, success, error = await coro
            progress.advance(task)
            results.append((domain, success, error))
            
            if success:
                console.print(f"[green]✓ {domain}[/green]")
            else:
                console.print(f"[red]✗ {domain}: {error}[/red]")

    # Summary
    success_count = sum(1 for r in results if r[1])
    console.print(Panel(f"Batch Complete\nSuccess: [green]{success_count}[/green]\nFailed: [red]{len(domains) - success_count}[/red]"))


@cli.command()
@click.argument('domain')
@async_command
async def test_extraction(domain: str):
    """
    Run extraction step only (Step 1).
    Outputs JSON result to stdout.
    """
    setup_logger(False)
    
    try:
        settings = get_settings()
        async with BrandIntelligencePipeline(settings=settings) as pipeline:
            console.print(f"[dim]Extracting data for {domain}...[/dim]", style="italic")
            result = await pipeline.extractor.extract_brand_data(domain)
            
            # Print pretty JSON
            json_str = result.model_dump_json(indent=2)
            console.print_json(json_str)
            
    except Exception as e:
        console.print(f"[bold red]Extraction Failed:[/bold red] {e}")
        sys.exit(1)


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@async_command
async def test_analysis(input_file: str):
    """
    Run analysis step only (Step 2).
    Input: JSON file containing ExtractionResult.
    """
    setup_logger(False)
    
    try:
        content = Path(input_file).read_text(encoding='utf-8')
        data = json.loads(content)
        
        # Handle if it's the full report or just extraction
        if "extraction_result" in data:
            extraction_data = data["extraction_result"]
        else:
            extraction_data = data
            
        extraction = ExtractionResult(**extraction_data)
        
        settings = get_settings()
        async with BrandIntelligencePipeline(settings=settings) as pipeline:
            console.print(f"[dim]Analyzing data for {extraction.brand_name}...[/dim]", style="italic")
            insight = await pipeline.analyzer.analyze(extraction)
            
            # Print pretty JSON
            json_str = insight.model_dump_json(indent=2)
            console.print_json(json_str)
            
    except Exception as e:
        console.print(f"[bold red]Analysis Failed:[/bold red] {e}")
        sys.exit(1)


@cli.command()
def validate_setup():
    """Check API keys and environment configuration."""
    console.print("[bold]Validating Setup...[/bold]")
    
    try:
        settings = get_settings()
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Check")
        table.add_column("Status")
        table.add_column("Details")
        
        # Check Anthropic
        key = settings.anthropic_api_key.get_secret_value()
        status = "[green]Pass[/green]" if key.startswith("sk-") else "[red]Fail[/red]"
        table.add_row("Anthropic API Key", status, f"configured ({len(key)} chars)")
        
        # Check Search
        has_search = settings.serpapi_api_key or settings.exa_api_key
        status = "[green]Pass[/green]" if has_search else "[red]Fail[/red]"
        provider = settings.get_search_provider() if has_search else "None"
        table.add_row("Search Provider", status, provider)
        
        # Configuration
        table.add_row("Output Dir", "[green]Pass[/green]", str(settings.output_dir))
        table.add_row("Environment", "[blue]Info[/blue]", settings.app_env)
        
        console.print(table)
        
        if not has_search:
            console.print("\n[yellow]Warning: No search provider configured (SerpAPI or Exa). Extraction will be limited.[/yellow]")
            # For tests, we want to fail if important keys are missing
            if not key.startswith("sk-") or not has_search:
                 sys.exit(1)
            
    except Exception as e:
        console.print(f"[bold red]Configuration Error:[/bold red] {e}")
        sys.exit(1)

if __name__ == "__main__":
    cli()
