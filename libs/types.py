from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field, asdict, fields
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Dict, List

Message = dict[str, Any]  # keys role, content
MessageList = list[Message]


# ============================================================================
# Debug Logger
# ============================================================================

class DebugLogger:
    """Async-safe debug logger for evaluators.
    
    Creates debug files in a specified directory.
    Clears the file on first write, then appends subsequent entries.
    Thread-safe for concurrent async operations.
    
    Usage:
        logger = DebugLogger(Path("debug"), "search_results.txt")
        await logger.log({"query": "...", "results": "..."})
    """
    
    def __init__(self, debug_dir: Path, filename: str, enabled: bool = True):
        """Initialize debug logger.
        
        Args:
            debug_dir: Directory to store debug files
            filename: Name of the debug file
            enabled: Whether logging is enabled (if False, log() is a no-op)
        """
        self.enabled = enabled
        self.debug_dir = debug_dir
        self.filename = filename
        self._file_path = debug_dir / filename if enabled else None
        self._lock = asyncio.Lock()
        self._initialized = False
    
    async def _ensure_initialized(self) -> None:
        """Ensure debug directory exists and file is cleared on first use."""
        if self._initialized or not self.enabled:
            return
        
        async with self._lock:
            if self._initialized:  # Double-check after acquiring lock
                return
            
            # Create debug directory if it doesn't exist
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            
            # Clear file content (create empty file)
            with open(self._file_path, "w", encoding="utf-8") as f:
                f.write(f"# Debug log started at {datetime.now().isoformat()}\n")
                f.write(f"# File: {self.filename}\n")
                f.write("=" * 80 + "\n\n")
            
            self._initialized = True
    
    async def log(self, data: Dict[str, Any], separator: str = "=" * 80) -> None:
        """Log debug data to file.
        
        Args:
            data: Dictionary of key-value pairs to log
            separator: Separator between log entries
        """
        if not self.enabled:
            return
        
        await self._ensure_initialized()
        
        async with self._lock:
            with open(self._file_path, "a", encoding="utf-8") as f:
                f.write(f"\n{separator}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
                
                for key, value in data.items():
                    if isinstance(value, (list, dict)):
                        value = json.dumps(value, indent=2, ensure_ascii=False)
                    f.write(f"### {key}:\n{value}\n\n")
    
    async def log_claim_evaluation(
        self,
        claim_content: str,
        search_results: str = "",
        queries: List[str] | None = None,
        top_urls: List[str] | None = None,
        extra: Dict[str, Any] | None = None,
    ) -> None:
        """Log a claim evaluation with standardized format.
        
        Args:
            claim_content: The textual claim being evaluated
            search_results: Search results text
            queries: List of search queries used
            top_urls: List of top URLs found
            extra: Additional data to log
        """
        data = {"Claim": claim_content}
        
        if queries:
            data["Queries"] = "\n".join(f"  - {q}" for q in queries)
        
        if search_results:
            data["Search Results"] = search_results
        
        if top_urls:
            data["Top URLs"] = "\n".join(f"  - {url}" for url in top_urls)
        
        if extra:
            data.update(extra)
        
        await self.log(data)


@dataclass
class UsageStats:
    """Track token usage and other evaluation metrics.
    
    Provides clean initialization, accumulation, and serialization of usage stats.
    Can be subclassed to add additional metrics for specific evaluators.
    
    Example:
        >>> usage = UsageStats()
        >>> usage.accumulate(response.token_usage)
        >>> usage.print_summary("Evaluation")
        
        # Subclass for additional metrics:
        >>> @dataclass
        >>> class WebScraperUsageStats(UsageStats):
        >>>     nb_pdf_downloads: int = 0
        >>>     nb_embedding_calls: int = 0
    """
    # Token usage (common to all evaluators)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    
    # Optional: customize field display names for print_summary
    # Override in subclass: _display_names: ClassVar[Dict[str, str]] = {"field": "Display Name"}
    _display_names: ClassVar[Dict[str, str]] = {}
    
    def accumulate(self, other: Dict[str, Any] | "UsageStats") -> None:
        """Add stats from another source (dict or UsageStats).
        
        Args:
            other: Dict or UsageStats to accumulate from. Unknown keys are ignored.
        """
        if isinstance(other, UsageStats):
            other = asdict(other)
        
        for f in fields(self):
            if f.name.startswith('_'):
                continue  # Skip private/class vars
            current = getattr(self, f.name)
            setattr(self, f.name, current + other.get(f.name, 0))
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary for serialization."""
        return {k: v for k, v in asdict(self).items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UsageStats":
        """Create UsageStats from a dictionary, ignoring unknown keys."""
        known_fields = {f.name for f in fields(cls) if not f.name.startswith('_')}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)
    
    def _format_field_name(self, field_name: str) -> str:
        """Convert field name to display format.
        
        Uses _display_names if defined, otherwise converts snake_case to Title Case.
        """
        if field_name in self._display_names:
            return self._display_names[field_name]
        # Convert snake_case to Title Case
        return field_name.replace('_', ' ').title()
    
    def print_summary(self, label: str = "Total") -> None:
        """Print a formatted summary of all usage stats.
        
        Args:
            label: Label for the summary (e.g., "Evaluation", "Extraction")
        """
        print(f"\n✓ {label} Usage Stats:")
        for f in fields(self):
            if f.name.startswith('_'):
                continue
            value = getattr(self, f.name)
            display_name = self._format_field_name(f.name)
            print(f"  - {display_name}: {value:,}")


@dataclass
class SamplerResponse:
    """
    Response from a sampler.
    """

    response_text: str
    actual_queried_message_list: MessageList
    response_metadata: dict[str, Any]
    token_usage: dict[str, int] = field(default_factory=lambda: {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
    })


class SamplerBase:
    """
    Base class for defining a sampling model, which can be evaluated,
    or used as part of the grading process.
    """

    async def __call__(
        self,
        message_list: MessageList,
    ) -> SamplerResponse:
        raise NotImplementedError


@dataclass
class EvalResult:
    """
    Result of running an evaluation (usually consisting of many samples)
    """

    score: float | None  # top-line metric
    metrics: dict[str, float] | None  # other metrics
    htmls: list[str]  # strings of valid HTML
    convos: list[MessageList]  # sampled conversations
    metadata: dict[str, Any] | None  # Extra data such as rubric scores or sollen


@dataclass
class SingleEvalResult:
    """
    Result of evaluating a single sample
    """

    score: float | None
    metrics: dict[str, float] = field(default_factory=dict)
    html: str | None = None
    convo: MessageList | None = None  # sampled conversation
    example_level_metadata: dict[str, Any] | None = (
        None  # Extra data such as rubric scores or sollen
    )


class Eval:
    """
    Base class for defining an evaluation.
    """

    async def __call__(self, sampler: SamplerBase) -> EvalResult:
        raise NotImplementedError
