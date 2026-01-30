"""Generic evaluation framework for LLM-as-a-judge evaluation.

This module provides base classes for:
1. Extractor - Extract atomic claims/facts from model responses (optional)
2. Evaluator - LLM-as-a-judge that scores responses against ground truth
3. SerperSearchClient - Web search utility using Serper API (optional)
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

import dotenv

from libs.schemas import Conversation
from libs.types import SamplerBase
from libs.serper import SerperSearchClient

from tqdm.asyncio import tqdm

dotenv.load_dotenv()


@dataclass
class ExtractionResult:
    """Result of extraction step."""

    conversation_id: int
    turn_number: int
    extracted_claims: List[str] | Dict[str, Any]  # Flexible format
    original_statement: str = ""  # The original assistant response text
    metadata: Dict[str, Any] | None = None


@dataclass
class EvaluationResult:
    """Result of evaluation step."""

    conversation_id: int
    score: float  # 0.0 to 1.0
    reasoning: str  # LLM judge's reasoning
    details: Dict[str, Any] | None = None  # Additional metrics
    metadata: Dict[str, Any] | None = None


class Extractor(ABC):
    """Base class for extracting claims/facts from model responses.

    Some tasks need extraction (research_questions, legal_cases),
    others don't (paper_authors).
    """

    def __init__(
        self,
        sampler: SamplerBase,
        extraction_semaphore: asyncio.Semaphore | None = None,
    ):
        """Initialize extractor with a sampler for LLM calls.

        Args:
            sampler: SamplerBase instance for making LLM calls
            extraction_semaphore: Optional semaphore to limit concurrent extraction calls
        """
        self.sampler = sampler
        self.extraction_semaphore = extraction_semaphore

    @abstractmethod
    async def _extract_impl(
        self,
        assistant_content: str,
        conversation_id: int,
        turn_number: int,
        metadata: Dict[str, Any],
    ) -> ExtractionResult:
        """Extract claims/facts from a single assistant turn (implementation).

        Subclasses implement this method. The public extract() method wraps
        this with semaphore control.

        Args:
            assistant_content: The assistant's response text
            conversation_id: ID of the conversation
            turn_number: Turn number in the conversation
            metadata: Ground truth metadata for context

        Returns:
            ExtractionResult with extracted claims
        """
        raise NotImplementedError("Subclasses must implement _extract_impl()")

    async def _extract_turn(
        self,
        assistant_content: str,
        conversation_id: int,
        turn_number: int,
        metadata: Dict[str, Any],
    ) -> ExtractionResult:
        """Extract from a single turn (with semaphore control).

        Args:
            assistant_content: The assistant's response text
            conversation_id: ID of the conversation
            turn_number: Turn number in the conversation
            metadata: Ground truth metadata

        Returns:
            ExtractionResult with extracted claims
        """
        if self.extraction_semaphore:
            async with self.extraction_semaphore:
                return await self._extract_impl(
                    assistant_content, conversation_id, turn_number, metadata
                )
        else:
            return await self._extract_impl(
                assistant_content, conversation_id, turn_number, metadata
            )

    async def extract_batch(
        self,
        conversations: List[Conversation],
        metadata_list: List[Dict[str, Any]],
    ) -> List[ExtractionResult]:
        """Extract from all assistant turns across conversations in parallel.

        Args:
            conversations: List of conversations
            metadata_list: List of metadata dicts

        Returns:
            List of ExtractionResults (one per assistant turn)
        """
        # Collect all assistant turns from all conversations
        tasks = []
        for conv, meta in zip(conversations, metadata_list):
            conversation_id = meta.get("conversation_id", 0)
            assistant_turn_number = 1  # Start from 1 for assistant turns
            for turn in conv.turns:
                if turn.role == "assistant":
                    tasks.append(
                        self._extract_turn(
                            turn.content, conversation_id, assistant_turn_number, meta
                        )
                    )
                    assistant_turn_number += 1

        # Extract all turns in parallel
        results = await tqdm.gather(*tasks)
        return list(results)


class Evaluator(ABC):
    """Base class for LLM-as-a-judge evaluation.

    Evaluates model responses against ground truth.
    """

    def __init__(
        self,
        sampler: SamplerBase,
        evaluation_semaphore: asyncio.Semaphore | None = None,
    ):
        """Initialize evaluator with a sampler.

        Args:
            sampler: SamplerBase instance for LLM judge
            evaluation_semaphore: Optional semaphore to limit concurrent evaluation calls
        """
        self.sampler = sampler
        self.evaluation_semaphore = evaluation_semaphore

    @abstractmethod
    async def _evaluate_impl(
        self,
        conversation: Conversation,
        metadata: Dict[str, Any],
        extraction_result: ExtractionResult | None = None,
    ) -> EvaluationResult:
        """Evaluate a conversation against ground truth (implementation).

        Subclasses implement this method. The public evaluate() method wraps
        this with semaphore control.

        Args:
            conversation: The conversation to evaluate
            metadata: Ground truth metadata
            extraction_result: Optional extraction result (if extractor was used)

        Returns:
            EvaluationResult with score and reasoning
        """
        raise NotImplementedError("Subclasses must implement _evaluate_impl()")

    async def evaluate(
        self,
        conversation: Conversation,
        metadata: Dict[str, Any],
        extraction_result: ExtractionResult | None = None,
    ) -> EvaluationResult:
        """Evaluate a conversation against ground truth (with semaphore control).

        Args:
            conversation: The conversation to evaluate
            metadata: Ground truth metadata
            extraction_result: Optional extraction result (if extractor was used)

        Returns:
            EvaluationResult with score and reasoning
        """
        if self.evaluation_semaphore:
            async with self.evaluation_semaphore:
                return await self._evaluate_impl(
                    conversation, metadata, extraction_result
                )
        else:
            return await self._evaluate_impl(conversation, metadata, extraction_result)

    async def evaluate_batch(
        self,
        conversations: List[Conversation],
        metadata_list: List[Dict[str, Any]],
        extraction_results: List[ExtractionResult] | None = None,
    ) -> List[EvaluationResult]:
        """Evaluate multiple conversations in parallel.

        Args:
            conversations: List of conversations
            metadata_list: List of metadata dicts
            extraction_results: Optional list of extraction results

        Returns:
            List of EvaluationResults
        """
        # Prepare extraction results iterator
        extractions = (
            extraction_results if extraction_results else [None] * len(conversations)
        )

        results = await tqdm.gather(
            *[
                self.evaluate(conv, meta, extraction)
                for conv, meta, extraction in zip(
                    conversations, metadata_list, extractions
                )
            ]
        )
        return list(results)


class NoOpExtractor(Extractor):
    """No-op extractor for tasks that don't need extraction.

    Used by tasks like paper_authors that evaluate directly.
    """

    async def _extract_impl(
        self,
        assistant_content: str,
        conversation_id: int,
        turn_number: int,
        metadata: Dict[str, Any],
    ) -> ExtractionResult:
        """Return the raw response without extraction.

        Args:
            assistant_content: The assistant's response text
            conversation_id: ID of the conversation
            turn_number: Turn number
            metadata: Metadata (unused)

        Returns:
            ExtractionResult with raw response
        """
        return ExtractionResult(
            conversation_id=conversation_id,
            turn_number=turn_number,
            extracted_claims=[assistant_content],  # Just the raw response
            original_statement=assistant_content,
            metadata=metadata,
        )
