"""Dataclasses for items flowing through the pipeline queues."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ConversationItem:
    """A conversation to be processed."""
    
    conversation_id: int
    conversation: list[dict]  # List of message dicts
    metadata: Dict[str, Any] = field(default_factory=dict)
    max_claims_per_turn: int | None = None  # Limit claims extracted per turn


@dataclass
class ClaimItem:
    """An extracted claim to be verified.
    
    This is a thin wrapper around extracted data. The `data` dict contains
    all domain-specific fields (e.g., 'authority' for medical, 'reference_name' 
    for legal). Strategies interpret this data via build_textual_claim_for_websearch()
    and build_textual_claim_for_judging().
    """
    
    claim_id: str  # Unique ID for tracking through pipeline
    conversation_id: int
    turn_number: int
    
    # Raw extracted data - strategy interprets this
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Common metadata (separate from extracted data)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, extracted_data: dict, conversation_id: int, turn_number: int) -> "ClaimItem":
        """Create ClaimItem from extracted claim dict."""
        import uuid
        # Separate metadata from extracted data if present
        metadata = extracted_data.pop("metadata", {}) if "metadata" in extracted_data else {}
        return cls(
            claim_id=str(uuid.uuid4())[:8],
            conversation_id=conversation_id,
            turn_number=turn_number,
            data=extracted_data,
            metadata=metadata,
        )
    
    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return {
            "claim_id": self.claim_id,
            "conversation_id": self.conversation_id,
            "turn_number": self.turn_number,
            "data": self.data,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_cache_dict(cls, cache_data: dict) -> "ClaimItem":
        """Create ClaimItem from cached dict (preserves claim_id)."""
        return cls(
            claim_id=cache_data.get("claim_id", ""),
            conversation_id=cache_data.get("conversation_id", 0),
            turn_number=cache_data.get("turn_number", 0),
            data=cache_data.get("data", {}),
            metadata=cache_data.get("metadata", {}),
        )


@dataclass
class SearchTask:
    """A web search task."""
    
    claim_id: str
    conversation_id: int
    claim: ClaimItem
    
    # Search planning state
    queries_executed: List[str] = field(default_factory=list)
    search_results_raw: List[Dict[str, Any]] = field(default_factory=list)
    search_results_text: str = ""
    
    # URLs to fetch (determined by search planner)
    urls_to_fetch: List[str] = field(default_factory=list)
    pdf_urls: List[str] = field(default_factory=list)
    
    # Optimization flags
    whitelist_skip: bool = False  # True if web search was skipped due to known package
    dynamic_cache_hit: bool = False  # True if verdict came from dynamic cache
    cached_verdict_exists: bool = True  # Whether cached verdict says package exists


@dataclass
class FetchTask:
    """A URL fetch task."""
    
    claim_id: str
    conversation_id: int
    url: str
    
    # Context from search
    title: str = ""
    snippet: str = ""
    is_pdf: bool = False
    
    # Parent search task reference
    search_results_text: str = ""


@dataclass
class PDFTask:
    """A PDF download/conversion task."""
    
    claim_id: str
    conversation_id: int
    url: str
    
    # Context
    title: str = ""
    search_results_text: str = ""


@dataclass 
class ContentItem:
    """Fetched content ready for filtering."""
    
    claim_id: str
    conversation_id: int
    claim: ClaimItem
    
    # Fetched content
    contents: List[Dict[str, Any]] = field(default_factory=list)  # [{title, url, snippet, content}]
    pdf_contents: List[Dict[str, Any]] = field(default_factory=list)
    
    # Search context
    search_results_text: str = ""
    queries: List[str] = field(default_factory=list)
    
    # PDF tracking for aggregation
    expected_pdf_count: int = 0  # Number of PDFs queued for this claim
    
    # Optimization flags
    whitelist_skip: bool = False  # True if web search was skipped due to known package
    dynamic_cache_hit: bool = False  # True if verdict came from dynamic cache
    cached_verdict_exists: bool = True  # Whether cached verdict says package exists


@dataclass
class PDFResult:
    """Result from PDF conversion, ready for aggregation."""
    
    claim_id: str
    conversation_id: int
    url: str
    title: str
    content: str  # Markdown content from PDF
    success: bool = True
    error: str | None = None


@dataclass
class FilteredContent:
    """Content filtered by embedding similarity, ready for judgment."""
    
    claim_id: str
    conversation_id: int
    claim: ClaimItem
    
    # Filtered content
    filtered_content: str = ""
    search_results_text: str = ""
    
    # Search queries executed (for debugging/analysis)
    queries: List[str] = field(default_factory=list)
    
    # Tracking
    nb_embedding_calls: int = 0
    nb_blocks_encoded: int = 0
    
    # Fallback flag
    use_fallback: bool = False  # If True, use websearch-enabled LLM
    
    # Optimization flags
    whitelist_skip: bool = False  # True if web search was skipped due to known package
    dynamic_cache_hit: bool = False  # True if verdict came from dynamic cache
    cached_verdict_exists: bool = True  # Whether cached verdict says package exists


@dataclass
class JudgmentResult:
    """Final judgment result for a claim."""
    
    claim_id: str
    conversation_id: int
    turn_number: int
    claim: ClaimItem
    
    # Judgment
    reference_name: str = ""
    reference_grounding: str = ""
    content_grounding: str = ""
    hallucination: str = ""  # "Yes" / "No"
    abstention: str = ""  # "Yes" / "No"
    verification_error: str = ""  # "Yes" / "No"
    
    # Pipeline flags (for analysis / reporting)
    input_use_fallback: bool = False  # From FilteredContent.use_fallback
    judge_used_websearch_fallback: bool = False  # Judge used websearch-enabled sampler
    snippets_only: bool = False  # No filtered_content, but have snippet search_results_text

    # Coding-specific hallucination flags
    hallucinated_import_detected: bool = False
    hallucinated_install_detected: bool = False
    hallucinated_function_usage_detected: bool = False

    # Reasoning/evidence
    reason: str = ""  # Explanation from the judge
    
    # Search queries executed (for debugging/analysis)
    search_queries: List[str] = field(default_factory=list)

    # Early stopping flag (coding task)
    skipped_early_stopping: bool = False  # True if skipped due to early stopping
    
    # Whitelist skip flag (coding task - known packages)
    skipped_whitelist: bool = False  # True if skipped due to whitelisted package

    # Metadata
    error: str | None = None
    token_usage: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return {
            "claim_id": self.claim_id,
            "conversation_id": self.conversation_id,
            "turn_idx": self.turn_number,
            "reference_name": self.reference_name,
            "reference_grounding": self.reference_grounding,
            "content_grounding": self.content_grounding,
            "hallucination": self.hallucination,
            "abstention": self.abstention,
            "verification_error": self.verification_error,
            "input_use_fallback": self.input_use_fallback,
            "judge_used_websearch_fallback": self.judge_used_websearch_fallback,
            "snippets_only": self.snippets_only,
            "hallucinated_import_detected": self.hallucinated_import_detected,
            "hallucinated_install_detected": self.hallucinated_install_detected,
            "hallucinated_function_usage_detected": self.hallucinated_function_usage_detected,
            "skipped_early_stopping": self.skipped_early_stopping,
            "reason": self.reason,
            "search_queries": self.search_queries,
            "error": self.error,
            "claim": self.claim.data,
            "token_usage": self.token_usage,
        }

