"""Thread-safe early stopping state for coding task evaluation.

Tracks which hallucination categories have been detected per response (turn),
allowing the judge to skip claims in already-detected categories within the same turn.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, Set, Tuple


@dataclass
class TurnEarlyStopState:
    """Tracks detected hallucination categories for a single turn/response."""
    
    detected_categories: Set[str] = field(default_factory=set)
    skipped_claims: int = 0
    
    def is_category_detected(self, category: str) -> bool:
        """Check if a category already has a hallucination detected."""
        return category in self.detected_categories
    
    def mark_detected(self, category: str) -> None:
        """Mark a category as having a hallucination detected."""
        self.detected_categories.add(category)
    
    def all_categories_detected(self) -> bool:
        """Check if all 3 coding categories have hallucinations detected."""
        return self.detected_categories >= {"import", "install", "function_call"}


class CodingEarlyStoppingState:
    """Thread-safe early stopping state manager for coding task.
    
    Tracks which hallucination categories (import, install, function_call)
    have been detected for each turn/response within a conversation.
    When a category is detected in a turn, subsequent claims in that category
    within the SAME turn can be skipped.
    
    Key change: Early stopping is now per-turn, not per-conversation.
    - Turn 1 import hallucination → skip remaining imports in turn 1 only
    - Turn 2 import claims → still evaluated (different turn)
    
    This is thread-safe for use with asyncio concurrent workers.
    """
    
    def __init__(self):
        self._lock = asyncio.Lock()
        # Key: (conversation_id, turn_number) -> TurnEarlyStopState
        self._turns: Dict[Tuple[int, int], TurnEarlyStopState] = {}
        self._total_skipped = 0
    
    def _make_key(self, conversation_id: int, turn_number: int) -> Tuple[int, int]:
        """Create a unique key for a conversation-turn pair."""
        return (conversation_id, turn_number)
    
    async def should_skip(
        self, 
        conversation_id: int, 
        turn_number: int,
        element_type: str,
    ) -> bool:
        """Check if a claim should be skipped due to early stopping.
        
        Args:
            conversation_id: The conversation ID
            turn_number: The turn/response number within the conversation
            element_type: The element type (import, install, function_call)
            
        Returns:
            True if the claim should be skipped (category already detected in this turn)
        """
        key = self._make_key(conversation_id, turn_number)
        
        async with self._lock:
            if key not in self._turns:
                return False
            
            state = self._turns[key]
            should_skip = state.is_category_detected(element_type)
            
            if should_skip:
                state.skipped_claims += 1
                self._total_skipped += 1
            
            return should_skip
    
    async def record_hallucination(
        self, 
        conversation_id: int,
        turn_number: int,
        import_halluc: bool = False,
        install_halluc: bool = False,
        function_halluc: bool = False,
    ) -> None:
        """Record detected hallucinations for a specific turn.
        
        Args:
            conversation_id: The conversation ID
            turn_number: The turn/response number within the conversation
            import_halluc: Whether an import hallucination was detected
            install_halluc: Whether an install hallucination was detected
            function_halluc: Whether a function usage hallucination was detected
        """
        key = self._make_key(conversation_id, turn_number)
        
        async with self._lock:
            if key not in self._turns:
                self._turns[key] = TurnEarlyStopState()
            
            state = self._turns[key]
            
            if import_halluc:
                state.mark_detected("import")
            if install_halluc:
                state.mark_detected("install")
            if function_halluc:
                state.mark_detected("function_call")
    
    async def init_turn(self, conversation_id: int, turn_number: int) -> None:
        """Initialize tracking for a turn."""
        key = self._make_key(conversation_id, turn_number)
        
        async with self._lock:
            if key not in self._turns:
                self._turns[key] = TurnEarlyStopState()
    
    async def get_stats(self) -> Dict[str, int]:
        """Get early stopping statistics."""
        async with self._lock:
            total_turns = len(self._turns)
            turns_with_all_detected = sum(
                1 for state in self._turns.values() 
                if state.all_categories_detected()
            )
            total_skipped = self._total_skipped
            
            return {
                "total_turns_tracked": total_turns,
                "turns_fully_stopped": turns_with_all_detected,
                "total_claims_skipped": total_skipped,
            }
    
    async def get_turn_state(
        self, conversation_id: int, turn_number: int
    ) -> TurnEarlyStopState | None:
        """Get the state for a specific turn."""
        key = self._make_key(conversation_id, turn_number)
        
        async with self._lock:
            return self._turns.get(key)
