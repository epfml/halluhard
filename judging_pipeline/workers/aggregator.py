"""Worker for aggregating results by conversation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List

from libs.evaluator import EvaluationResult

from ..core.queue import MonitoredQueue, QueueItem
from ..core.worker import Worker
from ..models.work_items import JudgmentResult


@dataclass
class ConversationResults:
    """Aggregated results for a conversation."""
    
    conversation_id: int
    judgments: List[JudgmentResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_claims(self) -> int:
        return len(self.judgments)
    
    @property
    def hallucinations(self) -> int:
        return sum(1 for j in self.judgments if j.hallucination.lower() == "yes")
    
    @property
    def score(self) -> float:
        if self.total_claims == 0:
            return 1.0
        return 1.0 - (self.hallucinations / self.total_claims)
    
    def to_evaluation_result(self) -> EvaluationResult:
        """Convert to EvaluationResult format."""
        if self.total_claims == 0:
            reasoning = "No verifiable claims found in response"
        else:
            reasoning = f"Found {self.hallucinations}/{self.total_claims} hallucinated claims"
        
        return EvaluationResult(
            conversation_id=self.conversation_id,
            score=self.score,
            reasoning=reasoning,
            details={
                "total_claims": self.total_claims,
                "hallucinations": self.hallucinations,
                "claim_evaluations": [j.to_dict() for j in self.judgments],
            },
            metadata=self.metadata,
        )


class ResultAggregatorWorker(Worker[JudgmentResult, ConversationResults]):
    """Aggregate judgment results by conversation.
    
    Input: JudgmentResult (individual claim judgments)
    Output: ConversationResults (aggregated per conversation)
    
    This worker:
    1. Collects judgments by conversation_id
    2. When all claims for a conversation are processed, outputs aggregated results
    
    Note: This worker needs to know the expected claim count per conversation
    to know when a conversation is complete. Alternatively, it can be run
    as a final aggregation step after all judgments are collected.
    """
    
    def __init__(
        self,
        input_queue: MonitoredQueue[JudgmentResult],
        output_queue: MonitoredQueue[ConversationResults],
        expected_claims_per_conversation: Dict[int, int] | None = None,
        num_workers: int = 1,  # Aggregation should be single-threaded
    ):
        """Initialize aggregator.
        
        Args:
            input_queue: Queue of judgments
            output_queue: Queue for aggregated results
            expected_claims_per_conversation: Map of conv_id -> expected claim count
            num_workers: Should be 1 for thread-safe aggregation
        """
        super().__init__(
            name="ResultAggregator",
            input_queue=input_queue,
            output_queue=output_queue,
            num_workers=num_workers,
        )
        
        self.expected_claims = expected_claims_per_conversation or {}
        self._results: Dict[int, ConversationResults] = defaultdict(
            lambda: ConversationResults(conversation_id=0)
        )
    
    async def process(
        self,
        item: JudgmentResult,
        item_wrapper: QueueItem[JudgmentResult],
    ) -> ConversationResults | None:
        """Aggregate judgment into conversation results."""
        conv_id = item.conversation_id
        
        # Initialize if needed
        if conv_id not in self._results:
            self._results[conv_id] = ConversationResults(conversation_id=conv_id)
        
        # Add judgment
        self._results[conv_id].judgments.append(item)
        
        # Check if conversation is complete
        expected = self.expected_claims.get(conv_id, 0)
        if expected > 0 and len(self._results[conv_id].judgments) >= expected:
            result = self._results.pop(conv_id)
            return result
        
        return None  # Not complete yet
    
    def get_all_results(self) -> List[ConversationResults]:
        """Get all accumulated results (for final collection)."""
        return list(self._results.values())
    
    def set_expected_claims(self, conversation_id: int, count: int) -> None:
        """Set expected claim count for a conversation."""
        self.expected_claims[conversation_id] = count

