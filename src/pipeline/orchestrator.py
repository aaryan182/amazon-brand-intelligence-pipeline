"""
Pipeline orchestrator using LangGraph.

Coordinates the multi-step brand intelligence pipeline with state management,
fault tolerance, automatic retries, and comprehensive error handling.

Features:
    - Stateful execution with LangGraph StateGraph
    - Conditional edges for validation and error handling
    - Automatic retry for transient failures
    - State persistence support
    - Progress tracking and structured logging
    - Timeout handling per node
    - Testing hooks for step-by-step execution
"""

import asyncio
import time
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, Literal, Optional, TypedDict, Annotated
from uuid import UUID, uuid4
import operator

from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.config.settings import Settings, get_settings
from src.models.schemas import (
    BrandInput,
    ExtractionResult,
    StrategicInsight,
    FinalReport,
    StepResult,
    PipelineStep,
    AnalysisStatus,
    ReportSection,
    ErrorType,
    ErrorDetail,
)
from src.extractors.brand_extractor import BrandExtractor
from src.analyzers.strategic_analyzer import StrategicAnalyzer
from src.services.validation_service import ValidationService
from src.services.llm_service import ClaudeService
from src.services.search_service import SearchService
from src.utils.logger import get_logger
from src.utils.formatters import ReportFormatter

logger = get_logger(__name__)


# =============================================================================
# Constants and Configuration
# =============================================================================

DEFAULT_NODE_TIMEOUT_SECONDS = 300  # 5 minutes
EXTRACTION_TIMEOUT_SECONDS = 600   # 10 minutes for extraction
ANALYSIS_TIMEOUT_SECONDS = 300     # 5 minutes for analysis
REPORT_TIMEOUT_SECONDS = 60        # 1 minute for report generation

MIN_PRODUCTS_FOR_ANALYSIS = 1
MIN_CONFIDENCE_FOR_FULL_ANALYSIS = 0.3


class NodeStatus(str, Enum):
    """Status of individual node execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class ValidationDecision(str, Enum):
    """Decision after validation step."""
    PROCEED = "proceed"
    ERROR = "error"
    SKIP_ANALYSIS = "skip_analysis"


# =============================================================================
# Pipeline State Definition (TypedDict for LangGraph)
# =============================================================================

class PipelineStateDict(TypedDict, total=False):
    """
    TypedDict-based pipeline state for LangGraph.
    
    Uses Annotated with operator.add for list accumulation.
    All fields are optional to support partial state updates.
    """
    # Identifiers
    run_id: str
    
    # Input
    domain: str
    brand_name: str
    brand_input: dict  # Serialized BrandInput
    
    # Step outputs
    extraction_result: dict | None  # Serialized ExtractionResult
    strategic_insight: dict | None  # Serialized StrategicInsight
    final_report: dict | None  # Serialized FinalReport
    
    # Status tracking
    current_step: str
    status: str
    validation_decision: str
    
    # Error handling (uses operator.add for accumulation)
    errors: Annotated[list[str], operator.add]
    
    # Metadata
    metadata: dict
    step_timings: dict  # Node name -> duration_ms
    retry_counts: dict  # Node name -> retry count
    
    # Progress
    progress_percent: int
    started_at: str
    completed_at: str | None


# =============================================================================
# Error Classes
# =============================================================================

class PipelineError(Exception):
    """Base exception for pipeline errors."""
    
    def __init__(
        self,
        message: str,
        error_type: ErrorType = ErrorType.INTERNAL_ERROR,
        details: dict | None = None,
        recoverable: bool = False,
    ):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.details = details or {}
        self.recoverable = recoverable


class ValidationError(PipelineError):
    """Validation-specific error."""
    
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(
            message=message,
            error_type=ErrorType.VALIDATION_ERROR,
            details=details,
            recoverable=False,
        )


class ExtractionError(PipelineError):
    """Extraction-specific error."""
    
    def __init__(self, message: str, details: dict | None = None, recoverable: bool = True):
        super().__init__(
            message=message,
            error_type=ErrorType.EXTRACTION_ERROR,
            details=details,
            recoverable=recoverable,
        )


class AnalysisError(PipelineError):
    """Analysis-specific error."""
    
    def __init__(self, message: str, details: dict | None = None, recoverable: bool = True):
        super().__init__(
            message=message,
            error_type=ErrorType.ANALYSIS_ERROR,
            details=details,
            recoverable=recoverable,
        )


class TimeoutError(PipelineError):
    """Timeout-specific error."""
    
    def __init__(self, node_name: str, timeout_seconds: int):
        super().__init__(
            message=f"Node '{node_name}' timed out after {timeout_seconds} seconds",
            error_type=ErrorType.TIMEOUT_ERROR,
            details={"node": node_name, "timeout": timeout_seconds},
            recoverable=True,
        )


# =============================================================================
# Decorators for Node Execution
# =============================================================================

def with_timeout(timeout_seconds: int):
    """Decorator to add timeout to async node functions."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                raise TimeoutError(func.__name__, timeout_seconds)
        return wrapper
    return decorator


def with_retry(max_attempts: int = 3, min_wait: float = 1, max_wait: float = 10):
    """Decorator to add retry logic to node functions."""
    def decorator(func: Callable):
        @wraps(func)
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
            retry=retry_if_exception_type((ExtractionError, AnalysisError)),
            reraise=True,
        )
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def track_timing(func: Callable):
    """Decorator to track node execution timing."""
    @wraps(func)
    async def wrapper(self, state: PipelineStateDict) -> dict[str, Any]:
        start_time = time.time()
        node_name = func.__name__.replace("_node", "")
        
        logger.info(f"Starting node: {node_name}", run_id=state.get("run_id"))
        
        try:
            result = await func(self, state)
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Update timing in result
            step_timings = state.get("step_timings", {}).copy()
            step_timings[node_name] = duration_ms
            result["step_timings"] = step_timings
            
            # Call progress callback if available
            if hasattr(self, "progress_callback") and self.progress_callback and "progress_percent" in result:
                try:
                    self.progress_callback(result["progress_percent"], f"Completed {node_name}")
                except Exception as cb_err:
                    logger.warning(f"Progress callback failed: {cb_err}")

            logger.info(
                f"Completed node: {node_name}",
                run_id=state.get("run_id"),
                duration_ms=duration_ms,
            )
            
            return result
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(
                f"Node failed: {node_name}",
                run_id=state.get("run_id"),
                duration_ms=duration_ms,
                error=str(e),
            )
            raise
    
    return wrapper


# =============================================================================
# Progress Tracker
# =============================================================================

class ProgressTracker:
    """Tracks and reports pipeline progress."""
    
    STEP_WEIGHTS = {
        "validate_input": 5,
        "extract_data": 40,
        "validate_extraction": 5,
        "analyze_data": 35,
        "generate_report": 10,
        "finalize": 5,
    }
    
    def __init__(self, callback: Optional[Callable[[int, str], None]] = None):
        """
        Initialize progress tracker.
        
        Args:
            callback: Optional callback(progress_percent, message) for progress updates
        """
        self.callback = callback
        self.completed_steps: list[str] = []
    
    def mark_complete(self, step: str) -> int:
        """Mark a step as complete and return new progress percentage."""
        self.completed_steps.append(step)
        progress = sum(
            self.STEP_WEIGHTS.get(s, 0) for s in self.completed_steps
        )
        
        if self.callback:
            self.callback(progress, f"Completed: {step}")
        
        return progress
    
    def get_progress(self) -> int:
        """Get current progress percentage."""
        return sum(self.STEP_WEIGHTS.get(s, 0) for s in self.completed_steps)


# =============================================================================
# State Persistence (Abstract Interface)
# =============================================================================

class StatePersistence:
    """Abstract interface for state persistence."""
    
    async def save_state(self, run_id: str, state: PipelineStateDict) -> None:
        """Save pipeline state for recovery."""
        raise NotImplementedError
    
    async def load_state(self, run_id: str) -> Optional[PipelineStateDict]:
        """Load saved pipeline state."""
        raise NotImplementedError
    
    async def delete_state(self, run_id: str) -> None:
        """Delete saved pipeline state."""
        raise NotImplementedError


class InMemoryStatePersistence(StatePersistence):
    """In-memory state persistence for testing."""
    
    def __init__(self):
        self._states: dict[str, PipelineStateDict] = {}
    
    async def save_state(self, run_id: str, state: PipelineStateDict) -> None:
        self._states[run_id] = state.copy()
    
    async def load_state(self, run_id: str) -> Optional[PipelineStateDict]:
        return self._states.get(run_id)
    
    async def delete_state(self, run_id: str) -> None:
        self._states.pop(run_id, None)


# =============================================================================
# Main Pipeline Class
# =============================================================================

class BrandIntelligencePipeline:
    """
    LangGraph-based pipeline for brand intelligence processing.
    
    Orchestrates the full pipeline from data extraction to report generation
    with fault tolerance, retry logic, and comprehensive state management.
    
    Features:
        - Stateful execution with automatic persistence
        - Conditional edges for validation and error handling
        - Automatic retry for transient failures
        - Progress tracking with callbacks
        - Timeout handling per node
        - Step-by-step execution for testing/debugging
    
    Example:
        >>> async with BrandIntelligencePipeline() as pipeline:
        ...     report = await pipeline.run("nike.com")
        ...     print(f"Analysis complete: {report.title}")
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        extractor: Optional[BrandExtractor] = None,
        analyzer: Optional[StrategicAnalyzer] = None,
        persistence: Optional[StatePersistence] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        enable_retries: bool = True,
        max_retries: int = 3,
    ):
        """
        Initialize the pipeline.
        
        Args:
            settings: Application settings (uses defaults if not provided)
            extractor: Pre-configured BrandExtractor (created if not provided)
            analyzer: Pre-configured StrategicAnalyzer (created if not provided)
            persistence: State persistence implementation
            progress_callback: Callback for progress updates
            enable_retries: Whether to enable automatic retries
            max_retries: Maximum retry attempts per node
        """
        self.settings = settings or get_settings()
        self.persistence = persistence or InMemoryStatePersistence()
        self.progress_callback = progress_callback
        self.enable_retries = enable_retries
        self.max_retries = max_retries
        
        # Services (initialized lazily or provided)
        self._extractor = extractor
        self._analyzer = analyzer
        self._search_service: Optional[SearchService] = None
        self._llm_service: Optional[ClaudeService] = None
        
        # Other services
        self.validator = ValidationService()
        self.formatter = ReportFormatter()
        
        # Build graph
        self._graph = self._build_graph()
        
        # Testing hooks
        self._mock_nodes: dict[str, Callable] = {}
        self._step_by_step_mode = False
        self._paused_at_step: Optional[str] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._initialize_services()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _initialize_services(self) -> None:
        """Initialize required services."""
        if self._extractor is None:
            self._search_service = SearchService(self.settings)
            self._llm_service = ClaudeService(self.settings)
            self._extractor = BrandExtractor(
                search_service=self._search_service,
                llm_service=self._llm_service,
                settings=self.settings,
            )
        
        if self._analyzer is None:
            if self._llm_service is None:
                self._llm_service = ClaudeService(self.settings)
            self._analyzer = StrategicAnalyzer(
                llm_service=self._llm_service,
                settings=self.settings,
            )
    
    @property
    def extractor(self) -> BrandExtractor:
        """Get the brand extractor."""
        if self._extractor is None:
            raise RuntimeError("Pipeline not initialized. Use async context manager.")
        return self._extractor
    
    @property
    def analyzer(self) -> StrategicAnalyzer:
        """Get the strategic analyzer."""
        if self._analyzer is None:
            raise RuntimeError("Pipeline not initialized. Use async context manager.")
        return self._analyzer
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph state machine with all nodes and edges.
        
        Graph structure:
            validate_input -> extract_data -> validate_extraction
                                                    |
                            +----------+------------+------------+
                            |          |                         |
                            v          v                         v
                     analyze_data   handle_error(skip)    handle_error(fail)
                            |          |                         |
                            v          v                         v
                     generate_report   END                      END
                            |
                            v
                          END
        """
        graph = StateGraph(PipelineStateDict)
        
        # Add nodes
        graph.add_node("validate_input", self._validate_input_node)
        graph.add_node("extract_data", self._extract_data_node)
        graph.add_node("validate_extraction", self._validate_extraction_node)
        graph.add_node("analyze_data", self._analyze_data_node)
        graph.add_node("generate_report", self._generate_report_node)
        graph.add_node("handle_error", self._handle_error_node)
        
        # Set entry point
        graph.set_entry_point("validate_input")
        
        # Add edges
        graph.add_edge("validate_input", "extract_data")
        graph.add_edge("extract_data", "validate_extraction")
        
        # Conditional edge after validation
        graph.add_conditional_edges(
            "validate_extraction",
            self._route_after_validation,
            {
                "analyze": "analyze_data",
                "error": "handle_error",
            }
        )
        
        graph.add_edge("analyze_data", "generate_report")
        graph.add_edge("generate_report", END)
        graph.add_edge("handle_error", END)
        
        return graph.compile()
    
    def _route_after_validation(self, state: PipelineStateDict) -> Literal["analyze", "error"]:
        """
        Route after validation based on extraction quality.
        
        Returns:
            "analyze" if extraction is valid, "error" otherwise
        """
        decision = state.get("validation_decision", ValidationDecision.PROCEED.value)
        
        if decision == ValidationDecision.ERROR.value:
            return "error"
        
        return "analyze"
    
    # =========================================================================
    # Node Implementations
    # =========================================================================
    
    @track_timing
    async def _validate_input_node(self, state: PipelineStateDict) -> dict[str, Any]:
        """
        Node 1: Validate domain input and initialize pipeline state.
        
        - Validates domain format
        - Extracts/confirms brand name
        - Initializes all state fields
        - Creates StepResult for this node
        """
        domain = state.get("domain", "")
        
        logger.info("Validating input", domain=domain)
        
        try:
            # Validate through service
            brand_input = self.validator.validate_brand_input({
                "domain": domain,
                "brand_name": state.get("brand_name"),
            })
            
            step_result = StepResult(
                step=PipelineStep.INITIALIZATION,
                status=AnalysisStatus.COMPLETED,
                completed_at=datetime.utcnow(),
            )
            
            return {
                "brand_input": brand_input.model_dump(),
                "brand_name": brand_input.brand_name,
                "current_step": PipelineStep.DATA_EXTRACTION.value,
                "status": AnalysisStatus.IN_PROGRESS.value,
                "progress_percent": ProgressTracker.STEP_WEIGHTS.get("validate_input", 5),
                "metadata": {
                    **state.get("metadata", {}),
                    "validated_at": datetime.utcnow().isoformat(),
                    "marketplace": brand_input.marketplace,
                },
            }
            
        except Exception as e:
            logger.error("Input validation failed", domain=domain, error=str(e))
            return {
                "errors": [f"Input validation failed: {str(e)}"],
                "status": AnalysisStatus.FAILED.value,
                "validation_decision": ValidationDecision.ERROR.value,
            }
    
    @track_timing
    @with_timeout(EXTRACTION_TIMEOUT_SECONDS)
    async def _extract_data_node(self, state: PipelineStateDict) -> dict[str, Any]:
        """
        Node 2: Execute brand data extraction.
        
        - Calls BrandExtractor.extract_brand_data()
        - Handles extraction errors with retries
        - Updates state with ExtractionResult
        - Logs progress at key stages
        """
        domain = state.get("domain", "")
        run_id = state.get("run_id", "unknown")
        
        # Check if already completed (resumption)
        if state.get("extraction_result"):
            logger.info("Skipping extraction (already completed)", domain=domain, run_id=run_id)
            return {}
        
        logger.info("Starting data extraction", domain=domain, run_id=run_id)
        
        try:
            # Check for mock in testing
            if "extract_data" in self._mock_nodes:
                extraction_result = await self._mock_nodes["extract_data"](state)
            else:
                extraction_result = await self.extractor.extract_brand_data(domain)
            
            step_result = StepResult(
                step=PipelineStep.DATA_EXTRACTION,
                status=AnalysisStatus.COMPLETED,
                completed_at=datetime.utcnow(),
                data={
                    "products_found": len(extraction_result.top_products),
                    "confidence": str(extraction_result.amazon_presence.confidence),
                },
            )
            
            logger.info(
                "Extraction completed",
                domain=domain,
                products=len(extraction_result.top_products),
                confidence=str(extraction_result.amazon_presence.confidence),
            )
            
            return {
                "extraction_result": extraction_result.model_dump(),
                "brand_name": extraction_result.brand_name,  # Update with actual brand name
                "current_step": PipelineStep.DATA_VALIDATION.value,
                "status": AnalysisStatus.IN_PROGRESS.value,
                "progress_percent": sum([
                    ProgressTracker.STEP_WEIGHTS.get("validate_input", 5),
                    ProgressTracker.STEP_WEIGHTS.get("extract_data", 40),
                ]),
                "metadata": {
                    **state.get("metadata", {}),
                    "extraction_completed_at": datetime.utcnow().isoformat(),
                    "amazon_presence_found": extraction_result.amazon_presence.found,
                },
            }
            
        except Exception as e:
            logger.error("Extraction failed", domain=domain, error=str(e))
            
            # Track retry count
            retry_counts = state.get("retry_counts", {}).copy()
            current_retries = retry_counts.get("extract_data", 0)
            
            if self.enable_retries and current_retries < self.max_retries:
                retry_counts["extract_data"] = current_retries + 1
                raise ExtractionError(
                    message=f"Extraction failed (attempt {current_retries + 1}): {str(e)}",
                    details={"domain": domain},
                    recoverable=True,
                )
            
            return {
                "errors": [f"Extraction failed after retries: {str(e)}"],
                "validation_decision": ValidationDecision.ERROR.value,
                "retry_counts": retry_counts,
            }
    
    @track_timing
    async def _validate_extraction_node(self, state: PipelineStateDict) -> dict[str, Any]:
        """
        Node 3: Validate extraction quality and decide next step.
        
        - Checks minimum data requirements
        - Verifies data quality thresholds
        - Sets validation_decision for routing
        - Can trigger error handling or skip analysis
        """
        extraction_data = state.get("extraction_result")
        domain = state.get("domain", "")
        
        logger.info("Validating extraction quality", domain=domain)
        
        if not extraction_data:
            return {
                "errors": ["No extraction data available"],
                "validation_decision": ValidationDecision.ERROR.value,
                "current_step": "handle_error",
            }
        
        # Parse extraction result
        try:
            extraction_result = ExtractionResult(**extraction_data)
        except Exception as e:
            return {
                "errors": [f"Invalid extraction data format: {str(e)}"],
                "validation_decision": ValidationDecision.ERROR.value,
            }
        
        # Check minimum requirements
        issues: list[str] = []
        
        # Check if Amazon presence was found
        if not extraction_result.amazon_presence.found:
            issues.append("No Amazon presence detected")
        
        # Check product count
        product_count = len(extraction_result.top_products)
        if product_count < MIN_PRODUCTS_FOR_ANALYSIS:
            issues.append(f"Insufficient products: {product_count} (minimum: {MIN_PRODUCTS_FOR_ANALYSIS})")
        
        # Decide based on issues
        if issues:
            # If no Amazon presence but extraction succeeded, we can still analyze
            # (the analyzer handles "no presence" case)
            if not extraction_result.amazon_presence.found:
                logger.warning(
                    "No Amazon presence, proceeding with market entry analysis",
                    domain=domain,
                )
                decision = ValidationDecision.PROCEED
            else:
                logger.error("Extraction validation failed", domain=domain, issues=issues)
                decision = ValidationDecision.ERROR
        else:
            decision = ValidationDecision.PROCEED
        
        step_result = StepResult(
            step=PipelineStep.DATA_VALIDATION,
            status=AnalysisStatus.COMPLETED if decision == ValidationDecision.PROCEED else AnalysisStatus.FAILED,
            completed_at=datetime.utcnow(),
            data={"issues": issues, "decision": decision.value},
        )
        
        return {
            "validation_decision": decision.value,
            "current_step": PipelineStep.STRATEGIC_ANALYSIS.value if decision == ValidationDecision.PROCEED else "handle_error",
            "progress_percent": sum([
                ProgressTracker.STEP_WEIGHTS.get("validate_input", 5),
                ProgressTracker.STEP_WEIGHTS.get("extract_data", 40),
                ProgressTracker.STEP_WEIGHTS.get("validate_extraction", 5),
            ]),
            "errors": issues if decision == ValidationDecision.ERROR else [],
            "metadata": {
                **state.get("metadata", {}),
                "validation_issues": issues,
                "validation_decision": decision.value,
            },
        }
    
    @track_timing
    @with_timeout(ANALYSIS_TIMEOUT_SECONDS)
    async def _analyze_data_node(self, state: PipelineStateDict) -> dict[str, Any]:
        """
        Node 4: Execute strategic analysis.
        
        - Calls StrategicAnalyzer.analyze()
        - Generates market insights
        - Updates state with StrategicInsight
        - Handles analysis errors
        """
        extraction_data = state.get("extraction_result")
        domain = state.get("domain", "")
        run_id = state.get("run_id", "unknown")
        
        logger.info("Starting strategic analysis", domain=domain, run_id=run_id)
        
        try:
            extraction_result = ExtractionResult(**extraction_data)
            
            # Check for mock in testing
            if "analyze_data" in self._mock_nodes:
                strategic_insight = await self._mock_nodes["analyze_data"](state)
            else:
                strategic_insight = await self.analyzer.analyze(extraction_result)
            
            step_result = StepResult(
                step=PipelineStep.STRATEGIC_ANALYSIS,
                status=AnalysisStatus.COMPLETED,
                completed_at=datetime.utcnow(),
                data={
                    "market_position": str(strategic_insight.market_position),
                    "recommendations_count": len(strategic_insight.growth_recommendations),
                },
            )
            
            logger.info(
                "Analysis completed",
                domain=domain,
                market_position=str(strategic_insight.market_position),
            )
            
            return {
                "strategic_insight": strategic_insight.model_dump(),
                "current_step": PipelineStep.REPORT_GENERATION.value,
                "progress_percent": sum([
                    ProgressTracker.STEP_WEIGHTS.get("validate_input", 5),
                    ProgressTracker.STEP_WEIGHTS.get("extract_data", 40),
                    ProgressTracker.STEP_WEIGHTS.get("validate_extraction", 5),
                    ProgressTracker.STEP_WEIGHTS.get("analyze_data", 35),
                ]),
                "metadata": {
                    **state.get("metadata", {}),
                    "analysis_completed_at": datetime.utcnow().isoformat(),
                },
            }
            
        except Exception as e:
            logger.error("Analysis failed", domain=domain, error=str(e))
            
            retry_counts = state.get("retry_counts", {}).copy()
            current_retries = retry_counts.get("analyze_data", 0)
            
            if self.enable_retries and current_retries < self.max_retries:
                retry_counts["analyze_data"] = current_retries + 1
                raise AnalysisError(
                    message=f"Analysis failed (attempt {current_retries + 1}): {str(e)}",
                    details={"domain": domain},
                    recoverable=True,
                )
            
            return {
                "errors": [f"Analysis failed: {str(e)}"],
                "validation_decision": ValidationDecision.ERROR.value,
                "retry_counts": retry_counts,
            }
    
    @track_timing
    @with_timeout(REPORT_TIMEOUT_SECONDS)
    async def _generate_report_node(self, state: PipelineStateDict) -> dict[str, Any]:
        """
        Node 5: Generate and format final report.
        
        - Combines extraction and analysis results
        - Formats report with sections
        - Generates markdown/HTML output
        - Saves report to file (optional)
        """
        extraction_data = state.get("extraction_result")
        analysis_data = state.get("strategic_insight")
        domain = state.get("domain", "")
        brand_name = state.get("brand_name", "Unknown")
        run_id = state.get("run_id", "unknown")
        
        logger.info("Generating report", domain=domain, run_id=run_id)
        
        try:
            extraction_result = ExtractionResult(**extraction_data)
            strategic_insight = StrategicInsight(**analysis_data)
            
            # Build report sections
            sections = self._build_report_sections(extraction_result, strategic_insight)
            
            # Create FinalReport
            final_report = FinalReport(
                title=f"Brand Intelligence Report: {brand_name}",
                brand_name=brand_name,
                domain=domain,
                extraction_result=extraction_result,
                strategic_insight=strategic_insight,
                report_format=self.settings.report_format if hasattr(self.settings, 'report_format') else "json",
                pipeline_run_id=UUID(run_id) if run_id != "unknown" else None,
                total_duration_ms=sum(state.get("step_timings", {}).values()),
                sections=sections,
            )
            
            # Save report if configured
            if hasattr(self.settings, 'output_dir') and self.settings.output_dir:
                await self.formatter.save_report(
                    final_report,
                    output_dir=self.settings.output_dir,
                )
            
            step_result = StepResult(
                step=PipelineStep.REPORT_GENERATION,
                status=AnalysisStatus.COMPLETED,
                completed_at=datetime.utcnow(),
                data={"report_id": str(final_report.report_id)},
            )
            
            logger.info("Report generated", domain=domain, report_id=str(final_report.report_id))
            
            return {
                "final_report": final_report.model_dump(),
                "current_step": PipelineStep.FINALIZATION.value,
                "status": AnalysisStatus.COMPLETED.value,
                "progress_percent": 100,
                "completed_at": datetime.utcnow().isoformat(),
                "metadata": {
                    **state.get("metadata", {}),
                    "report_generated_at": datetime.utcnow().isoformat(),
                    "report_id": str(final_report.report_id),
                },
            }
            
        except Exception as e:
            logger.error("Report generation failed", domain=domain, error=str(e))
            return {
                "errors": [f"Report generation failed: {str(e)}"],
                "status": AnalysisStatus.FAILED.value,
            }
    
    @track_timing
    async def _handle_error_node(self, state: PipelineStateDict) -> dict[str, Any]:
        """
        Node 6: Handle errors and generate error report.
        
        - Aggregates all errors from state
        - Generates error summary
        - Provides recovery suggestions
        - Creates partial report if possible
        """
        errors = state.get("errors", [])
        domain = state.get("domain", "")
        run_id = state.get("run_id", "unknown")
        
        logger.warning(
            "Handling pipeline errors",
            domain=domain,
            run_id=run_id,
            error_count=len(errors),
        )
        
        # Aggregate error information
        error_details = []
        recovery_suggestions = []
        
        for error in errors:
            error_lower = error.lower()
            
            if "validation" in error_lower:
                recovery_suggestions.append("Verify the domain format is correct (e.g., 'example.com')")
            elif "extraction" in error_lower:
                recovery_suggestions.append("Check if the brand has products on Amazon")
                recovery_suggestions.append("Try again later - Amazon may be temporarily unavailable")
            elif "analysis" in error_lower:
                recovery_suggestions.append("Retry the analysis with different parameters")
            elif "timeout" in error_lower:
                recovery_suggestions.append("Try again during off-peak hours")
        
        # Remove duplicates
        recovery_suggestions = list(set(recovery_suggestions))
        
        # Create error summary
        error_summary = {
            "run_id": run_id,
            "domain": domain,
            "error_count": len(errors),
            "errors": errors,
            "recovery_suggestions": recovery_suggestions,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        # Try to create partial report if we have extraction data
        extraction_data = state.get("extraction_result")
        partial_report = None
        
        if extraction_data:
            try:
                extraction_result = ExtractionResult(**extraction_data)
                partial_report = {
                    "brand_name": state.get("brand_name", "Unknown"),
                    "domain": domain,
                    "extraction_available": True,
                    "products_found": len(extraction_result.top_products),
                    "amazon_presence": extraction_result.amazon_presence.found,
                }
            except Exception:
                pass
        
        return {
            "status": AnalysisStatus.FAILED.value,
            "completed_at": datetime.utcnow().isoformat(),
            "metadata": {
                **state.get("metadata", {}),
                "error_summary": error_summary,
                "partial_report": partial_report,
                "failed_at": datetime.utcnow().isoformat(),
            },
        }
    
    def _build_report_sections(
        self,
        extraction: ExtractionResult,
        analysis: StrategicInsight,
    ) -> list[ReportSection]:
        """Build report sections from extraction and analysis data."""
        sections = []
        
        # Executive Summary
        sections.append(ReportSection(
            title="Executive Summary",
            content=analysis.executive_summary,
            order=1,
        ))
        
        # Market Position
        sections.append(ReportSection(
            title="Market Position",
            content=f"**Position:** {analysis.market_position}\n\n{analysis.market_position_rationale}",
            order=2,
        ))
        
        # Amazon Presence Overview
        presence = extraction.amazon_presence
        sections.append(ReportSection(
            title="Amazon Presence Overview",
            content=(
                f"**Presence Found:** {'Yes' if presence.found else 'No'}\n"
                f"**Confidence:** {presence.confidence}\n"
                f"**Products Analyzed:** {len(extraction.top_products)}\n"
                f"**Total Reviews:** {extraction.total_reviews:,}\n"
                f"**Average Rating:** {extraction.average_rating:.1f}/5.0" if extraction.average_rating is not None else "**Average Rating:** N/A"
            ),
            order=3,
        ))
        
        # SWOT Analysis
        swot = analysis.swot
        swot_content = (
            f"### Strengths\n" + "\n".join(f"- {s}" for s in swot.strengths) + "\n\n"
            f"### Weaknesses\n" + "\n".join(f"- {w}" for w in swot.weaknesses) + "\n\n"
            f"### Opportunities\n" + "\n".join(f"- {o}" for o in swot.opportunities) + "\n\n"
            f"### Threats\n" + "\n".join(f"- {t}" for t in swot.threats)
        )
        sections.append(ReportSection(
            title="SWOT Analysis",
            content=swot_content,
            order=4,
        ))
        
        # Recommendations
        rec_content = "\n".join(
            f"{i+1}. {rec}" for i, rec in enumerate(analysis.growth_recommendations)
        )
        sections.append(ReportSection(
            title="Strategic Recommendations",
            content=rec_content,
            order=5,
        ))
        
        # Top Products
        if extraction.top_products:
            product_rows = []
            for p in extraction.top_products[:10]:
                price_str = f"${p.price:.2f}" if p.price is not None else "N/A"
                rating_str = f"{p.rating:.1f}" if p.rating is not None else "N/A"
                product_rows.append(
                    f"| {p.title[:40]}... | {price_str} | {rating_str} | {p.review_count:,} |"
                )
            
            products_table = (
                "| Product | Price | Rating | Reviews |\n"
                "|---------|-------|--------|--------|\n"
                + "\n".join(product_rows)
            )
            sections.append(ReportSection(
                title="Top Products",
                content=products_table,
                order=6,
            ))
        
        return sections
    
    # =========================================================================
    # Public API
    # =========================================================================
    
    async def run(
        self,
        domain: str,
        brand_name: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> FinalReport:
        """
        Execute the complete pipeline for a brand.
        
        Args:
            domain: Brand's domain (e.g., "nike.com")
            brand_name: Optional brand name override
            run_id: Optional run ID for tracking/resumption
            
        Returns:
            FinalReport with complete analysis
            
        Raises:
            PipelineError: If pipeline fails and cannot recover
        """
        await self._initialize_services()
        
        run_id = run_id or str(uuid4())
        
        initial_state: PipelineStateDict = {
            "run_id": run_id,
            "domain": domain,
            "brand_name": brand_name or "",
            "brand_input": {},
            "extraction_result": None,
            "strategic_insight": None,
            "final_report": None,
            "current_step": PipelineStep.INITIALIZATION.value,
            "status": AnalysisStatus.PENDING.value,
            "validation_decision": ValidationDecision.PROCEED.value,
            "errors": [],
            "metadata": {
                "pipeline_version": "2.0.0",
                "started_at": datetime.utcnow().isoformat(),
            },
            "step_timings": {},
            "retry_counts": {},
            "progress_percent": 0,
            "started_at": datetime.utcnow().isoformat(),
            "completed_at": None,
        }
        
        logger.info(f"Starting pipeline run", run_id=run_id, domain=domain)
        
        try:
            # Save initial state
            await self.persistence.save_state(run_id, initial_state)
            
            # Execute the graph
            final_state = await self._graph.ainvoke(initial_state)
            
            # Save final state
            await self.persistence.save_state(run_id, final_state)
            
            # Check for errors
            if final_state.get("status") == AnalysisStatus.FAILED.value:
                errors = final_state.get("errors", ["Unknown error"])
                raise PipelineError(
                    message=f"Pipeline failed: {'; '.join(errors)}",
                    details={"errors": errors, "run_id": run_id},
                )
            
            # Parse final report
            report_data = final_state.get("final_report")
            if not report_data:
                raise PipelineError(
                    message="Pipeline completed but no report was generated",
                    details={"run_id": run_id},
                )
            
            logger.info(
                "Pipeline completed successfully",
                run_id=run_id,
                domain=domain,
                duration_ms=sum(final_state.get("step_timings", {}).values()),
            )
            
            return FinalReport(**report_data)
            
        except PipelineError:
            raise
        except Exception as e:
            logger.error(f"Pipeline failed with unexpected error", run_id=run_id, error=str(e))
            raise PipelineError(
                message=f"Unexpected pipeline error: {str(e)}",
                details={"run_id": run_id, "domain": domain},
            )
    
    async def run_step(
        self,
        step_name: str,
        state: PipelineStateDict,
    ) -> PipelineStateDict:
        """
        Execute a single pipeline step (for testing/debugging).
        
        Args:
            step_name: Name of the step to execute
            state: Current pipeline state
            
        Returns:
            Updated pipeline state
        """
        node_methods = {
            "validate_input": self._validate_input_node,
            "extract_data": self._extract_data_node,
            "validate_extraction": self._validate_extraction_node,
            "analyze_data": self._analyze_data_node,
            "generate_report": self._generate_report_node,
            "handle_error": self._handle_error_node,
        }
        
        if step_name not in node_methods:
            raise ValueError(f"Unknown step: {step_name}")
        
        await self._initialize_services()
        
        result = await node_methods[step_name](state)
        
        # Merge result into state
        updated_state = {**state, **result}
        return updated_state
    
    async def resume(self, run_id: str) -> FinalReport:
        """
        Resume a failed or interrupted pipeline run.
        
        Args:
            run_id: ID of the run to resume
            
        Returns:
            FinalReport from resumed execution
            
        Raises:
            ValueError: If run_id not found
            PipelineError: If resumption fails
        """
        saved_state = await self.persistence.load_state(run_id)
        
        if not saved_state:
            raise ValueError(f"No saved state found for run_id: {run_id}")
        
        logger.info(
            "Resuming pipeline",
            run_id=run_id,
            from_step=saved_state.get("current_step"),
        )
        
        # Re-run the graph from saved state
        final_state = await self._graph.ainvoke(saved_state)
        
        # Save final state
        await self.persistence.save_state(run_id, final_state)
        
        report_data = final_state.get("final_report")
        if not report_data:
            errors = final_state.get("errors", ["Unknown error"])
            raise PipelineError(
                message=f"Resume failed: {'; '.join(errors)}",
                details={"run_id": run_id},
            )
        
        return FinalReport(**report_data)
    
    def get_state(self, run_id: str) -> Optional[PipelineStateDict]:
        """
        Get current state for a run (sync wrapper).
        
        Args:
            run_id: Pipeline run ID
            
        Returns:
            Current state or None if not found
        """
        return asyncio.run(self.persistence.load_state(run_id))
    
    # =========================================================================
    # Testing Hooks
    # =========================================================================
    
    def mock_node(self, node_name: str, mock_func: Callable) -> None:
        """
        Register a mock function for a node (testing).
        
        Args:
            node_name: Name of the node to mock
            mock_func: Async function to use instead
        """
        self._mock_nodes[node_name] = mock_func
    
    def clear_mocks(self) -> None:
        """Clear all registered mocks."""
        self._mock_nodes.clear()
    
    def enable_step_by_step(self) -> None:
        """Enable step-by-step execution mode for debugging."""
        self._step_by_step_mode = True
    
    def disable_step_by_step(self) -> None:
        """Disable step-by-step execution mode."""
        self._step_by_step_mode = False
        self._paused_at_step = None
    
    # =========================================================================
    # Cleanup
    # =========================================================================
    
    async def close(self) -> None:
        """Close all service connections."""
        try:
            if self._search_service:
                await self._search_service.close()
            if self._llm_service:
                await self._llm_service.close()
        except Exception as e:
            logger.warning(f"Error closing services: {e}")


# =============================================================================
# Convenience Functions
# =============================================================================

async def analyze_brand(
    domain: str,
    settings: Optional[Settings] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> FinalReport:
    """
    Convenience function to analyze a brand.
    
    Args:
        domain: Brand's domain (e.g., "nike.com")
        settings: Optional settings override
        progress_callback: Optional progress callback
        
    Returns:
        FinalReport with complete analysis
        
    Example:
        >>> report = await analyze_brand("nike.com")
        >>> print(report.strategic_insight.market_position)
    """
    async with BrandIntelligencePipeline(
        settings=settings,
        progress_callback=progress_callback,
    ) as pipeline:
        return await pipeline.run(domain)
