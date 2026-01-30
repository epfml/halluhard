"""Worker for aggregating HTML content with PDF content."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List

from ..logging_config import get_logger
from ..core.queue import MonitoredQueue
from ..core.worker import WorkerStats
from ..models.work_items import ContentItem, PDFResult

logger = get_logger()


@dataclass
class PendingContent:
    """Tracks pending content for a claim awaiting PDF results."""
    content_item: ContentItem
    pdf_results: List[PDFResult] = field(default_factory=list)
    
    @property
    def is_complete(self) -> bool:
        """Check if all expected PDFs have been received."""
        return len(self.pdf_results) >= self.content_item.expected_pdf_count
    
    def merge(self) -> ContentItem:
        """Merge PDF results into the content item."""
        # Convert successful PDF results to content dicts
        pdf_contents = []
        for pdf in self.pdf_results:
            if pdf.success and pdf.content:
                pdf_contents.append({
                    "title": pdf.title,
                    "url": pdf.url,
                    "snippet": "",
                    "content": pdf.content,
                })
        
        # Return new ContentItem with merged PDF contents
        return ContentItem(
            claim_id=self.content_item.claim_id,
            conversation_id=self.content_item.conversation_id,
            claim=self.content_item.claim,
            contents=self.content_item.contents,
            pdf_contents=pdf_contents,
            search_results_text=self.content_item.search_results_text,
            queries=self.content_item.queries,
            expected_pdf_count=self.content_item.expected_pdf_count,
            whitelist_skip=self.content_item.whitelist_skip,
            dynamic_cache_hit=self.content_item.dynamic_cache_hit,
            cached_verdict_exists=self.content_item.cached_verdict_exists,
        )


class ContentAggregatorWorker:
    """Aggregates HTML content with PDF conversion results.
    
    This worker reads from two queues:
    - content_queue: ContentItem from WebFetcher (with expected_pdf_count)
    - pdf_queue: PDFResult from PDFConverter
    
    It buffers items by claim_id and outputs merged ContentItem when all
    expected PDFs have been received.
    
    Flow:
        WebFetcher ──→ content_queue ──┐
                                       ├──→ ContentAggregator ──→ output_queue
        PDFConverter ──→ pdf_queue ────┘
    """
    
    def __init__(
        self,
        content_queue: MonitoredQueue[ContentItem],
        pdf_queue: MonitoredQueue[PDFResult],
        output_queue: MonitoredQueue[ContentItem],
        timeout_seconds: float = 300.0,  # Max time to wait for PDFs
    ):
        """Initialize aggregator.
        
        Args:
            content_queue: Queue of ContentItems from WebFetcher
            pdf_queue: Queue of PDFResults from PDFConverter
            output_queue: Queue for merged ContentItems
            timeout_seconds: Max time to wait for all PDFs for a claim
        """
        self.content_queue = content_queue
        self.pdf_queue = pdf_queue
        self.output_queue = output_queue
        self.timeout_seconds = timeout_seconds
        
        # Pending items buffer keyed by claim_id
        self._pending: Dict[str, PendingContent] = {}
        self._pending_timestamps: Dict[str, float] = {}
        
        # Worker state
        self._tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        self._internal_stats = {
            "content_received": 0,
            "pdf_received": 0,
            "merged_output": 0,
            "timeout_output": 0,
        }
        
        # Monitor-compatible stats (single worker with id 0)
        self._worker_stats: List[WorkerStats] = [
            WorkerStats(name="ContentAggregator", worker_id=0)
        ]
    
    @property
    def name(self) -> str:
        return "ContentAggregator"
    
    @property
    def stats(self) -> List[WorkerStats]:
        """Get statistics for monitor compatibility."""
        return self._worker_stats
    
    @property
    def total_processed(self) -> int:
        """Total items processed (merged + timeout outputs)."""
        return self._internal_stats["merged_output"] + self._internal_stats["timeout_output"]
    
    @property
    def total_failed(self) -> int:
        """Total items failed (none for aggregator)."""
        return 0
    
    async def start(self) -> None:
        """Start the aggregator workers."""
        self._shutdown_event.clear()
        self._worker_stats[0].is_running = True
        
        # Start content queue reader
        self._tasks.append(
            asyncio.create_task(
                self._read_content_queue(),
                name="aggregator-content-reader",
            )
        )
        
        # Start PDF queue reader
        self._tasks.append(
            asyncio.create_task(
                self._read_pdf_queue(),
                name="aggregator-pdf-reader",
            )
        )
        
        # Start timeout checker
        self._tasks.append(
            asyncio.create_task(
                self._check_timeouts(),
                name="aggregator-timeout-checker",
            )
        )
        
        logger.info(f"[{self.name}] Started")
    
    async def stop(self, timeout: float = 10.0) -> None:
        """Stop the aggregator."""
        self._shutdown_event.set()
        
        # Wait for tasks to complete
        if self._tasks:
            done, pending = await asyncio.wait(
                self._tasks,
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED,
            )
            
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            self._tasks.clear()
        
        # Flush any remaining pending items
        await self._flush_all_pending()
        
        self._worker_stats[0].is_running = False
        
        logger.info(
            f"[{self.name}] Stopped (content={self._internal_stats['content_received']}, "
            f"pdf={self._internal_stats['pdf_received']}, merged={self._internal_stats['merged_output']}, "
            f"timeout={self._internal_stats['timeout_output']})"
        )
    
    async def _read_content_queue(self) -> None:
        """Read ContentItems from content queue."""
        while not self._shutdown_event.is_set():
            try:
                item_wrapper = await asyncio.wait_for(
                    self.content_queue.get(),
                    timeout=0.5,
                )
                
                if item_wrapper is None:
                    continue
                
                item = item_wrapper.data
                self._internal_stats["content_received"] += 1
                self._worker_stats[0].current_item_id = item.claim_id
                
                claim_id = item.claim_id
                
                if item.expected_pdf_count == 0:
                    # No PDFs expected - output immediately
                    await self._output_item(item)
                else:
                    # Buffer and wait for PDFs
                    if claim_id in self._pending:
                        # Already have some PDFs, add content
                        self._pending[claim_id].content_item = item
                    else:
                        self._pending[claim_id] = PendingContent(content_item=item)
                        self._pending_timestamps[claim_id] = asyncio.get_event_loop().time()
                    
                    # Check if now complete
                    await self._check_and_output(claim_id)
                
                self.content_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.warning(f"[{self.name}] Error reading content queue: {e}")
    
    async def _read_pdf_queue(self) -> None:
        """Read PDFResults from PDF queue."""
        while not self._shutdown_event.is_set():
            try:
                item_wrapper = await asyncio.wait_for(
                    self.pdf_queue.get(),
                    timeout=0.5,
                )
                
                if item_wrapper is None:
                    continue
                
                pdf_result = item_wrapper.data
                self._internal_stats["pdf_received"] += 1
                
                claim_id = pdf_result.claim_id
                
                if claim_id in self._pending:
                    self._pending[claim_id].pdf_results.append(pdf_result)
                else:
                    # PDF arrived before content - create pending entry
                    # We'll add the content item when it arrives
                    self._pending[claim_id] = PendingContent(
                        content_item=None,  # Will be set when content arrives
                        pdf_results=[pdf_result],
                    )
                    self._pending_timestamps[claim_id] = asyncio.get_event_loop().time()
                
                # Check if now complete
                await self._check_and_output(claim_id)
                
                self.pdf_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.warning(f"[{self.name}] Error reading PDF queue: {e}")
    
    async def _check_and_output(self, claim_id: str) -> None:
        """Check if claim is complete and output if so."""
        if claim_id not in self._pending:
            return
        
        pending = self._pending[claim_id]
        
        # Need both content and all PDFs
        if pending.content_item is None:
            return
        
        if pending.is_complete:
            merged = pending.merge()
            await self._output_item(merged)
            self._internal_stats["merged_output"] += 1
            self._worker_stats[0].items_processed += 1
            self._worker_stats[0].current_item_id = None
            
            # Clean up
            del self._pending[claim_id]
            del self._pending_timestamps[claim_id]
    
    async def _check_timeouts(self) -> None:
        """Periodically check for timed out items."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
                current_time = asyncio.get_event_loop().time()
                timed_out = []
                
                for claim_id, start_time in self._pending_timestamps.items():
                    if current_time - start_time > self.timeout_seconds:
                        timed_out.append(claim_id)
                
                for claim_id in timed_out:
                    pending = self._pending.get(claim_id)
                    if pending and pending.content_item:
                        logger.warning(
                            f"[{self.name}] Timeout waiting for PDFs for claim {claim_id} "
                            f"(received {len(pending.pdf_results)}/{pending.content_item.expected_pdf_count})"
                        )
                        # Output what we have
                        merged = pending.merge()
                        await self._output_item(merged)
                        self._internal_stats["timeout_output"] += 1
                        self._worker_stats[0].items_processed += 1
                    
                    # Clean up
                    if claim_id in self._pending:
                        del self._pending[claim_id]
                    if claim_id in self._pending_timestamps:
                        del self._pending_timestamps[claim_id]
                        
            except Exception as e:
                logger.warning(f"[{self.name}] Error in timeout checker: {e}")
    
    async def _flush_all_pending(self) -> None:
        """Flush all pending items on shutdown."""
        for claim_id, pending in list(self._pending.items()):
            if pending.content_item:
                merged = pending.merge()
                await self._output_item(merged)
                self._internal_stats["timeout_output"] += 1
                self._worker_stats[0].items_processed += 1
        
        self._pending.clear()
        self._pending_timestamps.clear()
    
    async def _output_item(self, item: ContentItem) -> None:
        """Output a merged ContentItem."""
        await self.output_queue.put(
            item,
            claim_id=item.claim_id,
            conversation_id=item.conversation_id,
        )

