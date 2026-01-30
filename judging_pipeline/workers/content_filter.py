"""Worker for filtering content using embeddings."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, TYPE_CHECKING

from libs.information_extraction import extract_relevant_sentences

from ..core.queue import MonitoredQueue, QueueItem
from ..core.worker import Worker
from ..models.work_items import ClaimItem, ContentItem, FilteredContent
from ..logging_config import get_logger

if TYPE_CHECKING:
    from .early_stopping import CodingEarlyStoppingState

logger = get_logger()

# Type alias for claim text builder function
ClaimTextBuilder = Callable[[ClaimItem], str]


class ContentFilterWorker(Worker[ContentItem, FilteredContent]):
    """Filter content using embedding similarity.
    
    Input: ContentItem (with all fetched content)
    Output: FilteredContent (with relevant sentences)
    
    This worker:
    1. Combines HTML content and PDF content
    2. Uses embeddings to find relevant sentences
    3. Outputs filtered content ready for judgment
    """
    
    def __init__(
        self,
        input_queue: MonitoredQueue[ContentItem],
        output_queue: MonitoredQueue[FilteredContent],
        claim_text_builder: ClaimTextBuilder,
        max_output_words: int = 1500,
        block_size: int = 3000,
        overlap: int = 200,
        num_workers: int = 10,
        rate_limit_delay: float = 0.0,
        max_concurrent_embeddings: int = 15,
        early_stopping_state: "CodingEarlyStoppingState | None" = None,
    ):
        """Initialize content filter.
        
        Args:
            input_queue: Queue of content items
            output_queue: Queue for filtered content
            claim_text_builder: Function to convert ClaimItem to text for similarity matching.
                               Should come from strategy.build_textual_claim_for_websearch.
            max_output_words: Maximum words in filtered output
            block_size: Character size for embedding blocks
            overlap: Overlap between blocks
            num_workers: Number of concurrent workers
            rate_limit_delay: Delay between filtering
            max_concurrent_embeddings: Max concurrent embedding API calls (default: 15)
            early_stopping_state: Optional early stopping state for coding tasks
        """
        super().__init__(
            name="ContentFilter",
            input_queue=input_queue,
            output_queue=output_queue,
            num_workers=num_workers,
            rate_limit_delay=rate_limit_delay,
        )
        
        self.claim_text_builder = claim_text_builder
        self.max_output_words = max_output_words
        self.block_size = block_size
        self.overlap = overlap
        # Semaphore to limit concurrent embedding calls and prevent connection timeouts
        self._embedding_semaphore = asyncio.Semaphore(max_concurrent_embeddings)
        self.early_stopping_state = early_stopping_state
    
    async def process(
        self,
        item: ContentItem,
        item_wrapper: QueueItem[ContentItem],
    ) -> FilteredContent:
        """Filter content to find relevant passages."""
        # Early stopping check - skip embedding if category already has hallucination in this turn
        # The claim will still flow to JudgeWorker where it will be properly marked as skipped
        if self.early_stopping_state:
            element_type = item.claim.data.get("element_type")
            if element_type in ["import", "install", "function_call"]:
                if await self.early_stopping_state.should_skip(
                    item.conversation_id, item.claim.turn_number, element_type
                ):
                    logger.info(
                        f"⏭️  EARLY STOP [Filter]: {element_type} skipped (conv {item.conversation_id}, turn {item.claim.turn_number}) - no embedding"
                    )
                    # Return empty content - JudgeWorker will handle the skip
                    return FilteredContent(
                        claim_id=item.claim_id,
                        conversation_id=item.conversation_id,
                        claim=item.claim,
                        filtered_content="",
                        search_results_text=item.search_results_text,
                        queries=item.queries,
                        use_fallback=True,  # Will trigger early stopping check in judge
                        whitelist_skip=item.whitelist_skip,
                        dynamic_cache_hit=item.dynamic_cache_hit,
                        cached_verdict_exists=item.cached_verdict_exists,
                    )
        
        # Combine HTML and PDF content
        all_contents = item.contents + item.pdf_contents
        
        if not all_contents:
            # No content fetched - use fallback
            return FilteredContent(
                claim_id=item.claim_id,
                conversation_id=item.conversation_id,
                claim=item.claim,
                filtered_content="",
                search_results_text=item.search_results_text,
                queries=item.queries,
                use_fallback=True,
                whitelist_skip=item.whitelist_skip,
                dynamic_cache_hit=item.dynamic_cache_hit,
                cached_verdict_exists=item.cached_verdict_exists,
            )
        
        # Build claim text for similarity matching
        claim_text = self.claim_text_builder(item.claim)
        
        # Extract relevant sentences
        filtered_content, nb_embedding_calls, nb_blocks_encoded = await extract_relevant_sentences(
            websearch_results=all_contents,
            claim=claim_text,
            embedding_semaphore=self._embedding_semaphore,
            max_output_words=self.max_output_words,
            block_size=self.block_size,
            overlap=self.overlap,
        )
        
        return FilteredContent(
            claim_id=item.claim_id,
            conversation_id=item.conversation_id,
            claim=item.claim,
            filtered_content=filtered_content,
            search_results_text=item.search_results_text,
            queries=item.queries,
            nb_embedding_calls=nb_embedding_calls,
            nb_blocks_encoded=nb_blocks_encoded,
            use_fallback=not filtered_content,  # Fallback if no content after filtering
            whitelist_skip=item.whitelist_skip,
            dynamic_cache_hit=item.dynamic_cache_hit,
            cached_verdict_exists=item.cached_verdict_exists,
        )

