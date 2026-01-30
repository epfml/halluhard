"""Worker for extracting claims from conversations."""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Any

from libs.json_utils import extract_json_from_response, sanitize_json_string
from libs.types import SamplerBase

from ..core.queue import MonitoredQueue, QueueItem
from ..core.worker import Worker
from ..core.domain_strategy import DomainStrategy
from ..models.work_items import ConversationItem, ClaimItem
from ..logging_config import get_logger


MIN_CONTENT_LENGTH = 20
# Maximum time to spend extracting claims from a single turn (in seconds)
# This caps the total time including all retries, preventing indefinite hangs
MAX_EXTRACTION_TIMEOUT_PER_TURN = 300  # 5 minutes max per turn

logger = get_logger()


class ClaimExtractorWorker(Worker[ConversationItem, ClaimItem]):
    """Extract claims about sources from assistant responses.
    
    Input: ConversationItem (full conversation)
    Output: Multiple ClaimItem (one per extracted claim)
    
    This worker:
    1. Iterates through assistant turns in the conversation
    2. Sends each turn to LLM for claim extraction
    3. Outputs one ClaimItem per extracted claim
    """
    
    def __init__(
        self,
        input_queue: MonitoredQueue[ConversationItem],
        output_queue: MonitoredQueue[ClaimItem],
        sampler: SamplerBase,
        strategy: DomainStrategy,
        system_prompt: str | None = None,
        num_workers: int = 5,
        rate_limit_delay: float = 0.0,
        max_claims_per_category: int | None = None,
    ):
        """Initialize claim extractor.
        
        Args:
            input_queue: Queue of conversations
            output_queue: Queue for extracted claims
            sampler: LLM sampler for extraction
            strategy: Domain strategy for prompt generation and validation
            system_prompt: Optional custom extraction prompt (overrides strategy default)
            num_workers: Number of concurrent workers
            rate_limit_delay: Delay between extractions
            max_claims_per_category: For coding tasks, limit claims per category (import/install/function_call)
        """
        super().__init__(
            name="ClaimExtractor",
            input_queue=input_queue,
            output_queue=output_queue,
            num_workers=num_workers,
            rate_limit_delay=rate_limit_delay,
        )
        
        self.sampler = sampler
        self.strategy = strategy
        self.system_prompt = system_prompt or self._load_default_prompt()
        self.max_claims_per_category = max_claims_per_category
    
    def _load_default_prompt(self) -> str:
        """Load default extraction system prompt."""
        prompt_path = self.strategy.extractor_prompt_path
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8").strip()
        
        raise FileNotFoundError(f"System prompt file not found: {prompt_path}")
    
    async def process(
        self,
        item: ConversationItem,
        item_wrapper: QueueItem[ConversationItem],
    ) -> list[ClaimItem] | None:
        """Extract claims from all assistant turns in a conversation."""
        logger.debug(f"Processing conversation {item.conversation_id} with {len(item.conversation)} turns")
        
        claims = []
        assistant_turns_found = 0
        
        for turn_number, message in enumerate(item.conversation):
            if message.get("role") != "assistant":
                continue
            
            assistant_turns_found += 1
            content = message.get("content", "")
            if len(content) < MIN_CONTENT_LENGTH:
                logger.debug(f"Conv {item.conversation_id} turn {turn_number}: skipping (content too short: {len(content)} chars)")
                continue
            
            logger.debug(f"Conv {item.conversation_id} turn {turn_number}: extracting claims from {len(content)} chars")
            
            # Extract claims from this turn
            turn_claims = await self._extract_from_turn(
                content=content,
                conversation_id=item.conversation_id,
                turn_number=turn_number,
            )
            
            logger.debug(f"Conv {item.conversation_id} turn {turn_number}: extracted {len(turn_claims)} claims")
            
            # Limit claims per turn if specified
            # Skip limiting for coding tasks - early stopping handles efficiency there
            # (we need all claims to properly detect hallucinations in each category)
            is_coding_task = self.strategy.task_name == "coding"
            if not is_coding_task and item.max_claims_per_turn is not None and len(turn_claims) > item.max_claims_per_turn:
                original_count = len(turn_claims)
                turn_claims = random.sample(turn_claims, item.max_claims_per_turn)
                logger.debug(f"Conv {item.conversation_id} turn {turn_number}: sampled {len(turn_claims)}/{original_count} claims")
            
            claims.extend(turn_claims)
        
        logger.info(f"Conv {item.conversation_id}: {assistant_turns_found} assistant turns, {len(claims)} total claims extracted")
        
        if not claims:
            return None
        
        # For coding tasks, limit claims per category to avoid processing too many of the same type
        # This ensures we check all categories (import, install, function_call) but don't
        # process 50 imports when 3-5 would be enough to detect hallucinations
        is_coding_task = self.strategy.task_name == "coding"
        if is_coding_task and self.max_claims_per_category is not None:
            claims = self._limit_claims_per_category(claims, item.conversation_id)
        
        return claims
    
    def _limit_claims_per_category(self, claims: list[ClaimItem], conversation_id: int) -> list[ClaimItem]:
        """Limit claims per category for coding tasks.
        
        Groups claims by element_type (import, install, function_call) and
        limits each group to max_claims_per_category.
        """
        from collections import defaultdict
        
        # Group claims by category
        by_category: dict[str, list[ClaimItem]] = defaultdict(list)
        for claim in claims:
            element_type = claim.data.get("element_type", "unknown")
            by_category[element_type].append(claim)
        
        # Limit each category
        limited_claims = []
        for category, category_claims in by_category.items():
            if len(category_claims) > self.max_claims_per_category:
                # Sample claims to keep (prioritize earlier turns)
                category_claims.sort(key=lambda c: c.turn_number)
                sampled = category_claims[:self.max_claims_per_category]
                logger.debug(
                    f"Conv {conversation_id}: limited {category} claims from "
                    f"{len(category_claims)} to {len(sampled)}"
                )
                limited_claims.extend(sampled)
            else:
                limited_claims.extend(category_claims)
        
        if len(limited_claims) < len(claims):
            logger.info(
                f"Conv {conversation_id}: limited claims from {len(claims)} to "
                f"{len(limited_claims)} (max {self.max_claims_per_category}/category)"
            )
        
        return limited_claims
    
    async def _extract_from_turn(
        self,
        content: str,
        conversation_id: int,
        turn_number: int,
    ) -> list[ClaimItem]:
        """Extract claims from a single assistant turn.
        
        Has a total timeout to prevent indefinite hangs from repeated API timeouts.
        """
        extraction_prompt = self.strategy.get_extraction_user_prompt(content)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": extraction_prompt},
        ]
        
        try:
            # Wrap the sampler call with a total timeout to prevent indefinite hangs
            # This caps total time including all retries (the sampler has its own retry logic)
            response = await asyncio.wait_for(
                self.sampler(messages),
                timeout=MAX_EXTRACTION_TIMEOUT_PER_TURN,
            )
            response_text = response.response_text.strip()
            
            # Parse JSON
            response_text = extract_json_from_response(response_text)
            response_text = sanitize_json_string(response_text)
            parsed = json.loads(response_text)
            
            # Handle both single and array responses
            if isinstance(parsed, list):
                raw_claims = parsed
            elif isinstance(parsed, dict):
                raw_claims = [parsed]
            else:
                return []
            
            # Convert to ClaimItem objects
            claims = []
            for raw in raw_claims:
                if not self.strategy.is_valid_claim(raw):
                    continue
                
                raw["original_statement"] = content
                
                # Use strategy to map dict to ClaimItem
                claim = self.strategy.map_to_claim_item(
                    data=raw,
                    conversation_id=conversation_id,
                    turn_number=turn_number,
                )
                claims.append(claim)
            
            return claims
        
        except asyncio.TimeoutError:
            logger.warning(
                f"Conv {conversation_id} turn {turn_number}: extraction timed out after "
                f"{MAX_EXTRACTION_TIMEOUT_PER_TURN}s (content: {len(content)} chars). Skipping turn."
            )
            return []
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to extract claims from turn {turn_number} of conversation {conversation_id}: {e}")
            return []
