from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

from ..models.work_items import ClaimItem


class DomainStrategy(ABC):
    """Abstract base class for domain-specific logic."""

    def __init__(self, base_path: Path):
        self.base_path = base_path
        # Prompts are always in judging_pipeline/prompts/<task_name>/
        self._prompts_base = Path(__file__).parent.parent / "prompts"

    @property
    @abstractmethod
    def task_name(self) -> str:
        """Name of the task/domain."""
        pass

    @property
    def extractor_prompt_path(self) -> Path:
        """Path to the extractor system prompt file."""
        return self._prompts_base / self.task_name / "sys-extractor.txt"

    @property
    def evaluator_prompt_path(self) -> Path:
        """Path to the evaluator system prompt file."""
        return self._prompts_base / self.task_name / "sys-evaluator.txt"

    @abstractmethod
    def get_extraction_user_prompt(self, content: str) -> str:
        """Generate the user prompt for claim extraction."""
        pass

    @abstractmethod
    def is_valid_claim(self, claim: dict) -> bool:
        """Check if an extracted claim is valid/sufficient for verification."""
        pass

    @abstractmethod
    def map_to_claim_item(self, data: dict, conversation_id: int, turn_number: int) -> ClaimItem:
        """Map extracted JSON data to a ClaimItem."""
        pass

    @abstractmethod
    def build_textual_claim_for_websearch(self, claim: ClaimItem) -> str:
        """Build a textual representation of the claim for web search queries."""
        pass

    @abstractmethod
    def build_textual_claim_for_judging(self, claim: ClaimItem) -> str:
        """Build a textual representation of the claim for judgment."""
        pass

    @abstractmethod
    def build_judgment_prompt(
        self,
        search_results: str,
        filtered_content: str,
        claim_text: str,
    ) -> str:
        """Build the user prompt for judgment."""
        pass

    @abstractmethod
    def build_snippets_only_judgment_prompt(
        self,
        search_results: str,
        claim_text: str,
    ) -> str:
        """Build the user prompt for judgment using only snippets."""
        pass

    @abstractmethod
    def build_fallback_judgment_prompt(self, claim_text: str) -> str:
        """Build the user prompt for fallback judgment (LLM web search)."""
        pass

    @property
    def search_planner_prompt(self) -> str | None:
        """Custom planner prompt for LLM-guided web search.
        
        Override in subclasses to provide domain-specific search planning.
        Returns None to use the default planner prompt in SerperSearchClient.
        
        The prompt should include these placeholders:
        - [STATEMENT]: The claim text to verify
        - [KNOWLEDGE]: Accumulated search results
        - [PREVIOUS_QUERIES]: List of queries already executed
        """
        return None

