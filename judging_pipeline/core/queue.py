"""Monitored async queue with statistics tracking."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass
class QueueStats:
    """Statistics for a monitored queue."""
    
    name: str
    current_depth: int = 0
    total_enqueued: int = 0
    total_dequeued: int = 0
    total_failed: int = 0
    avg_wait_time_ms: float = 0.0
    max_wait_time_ms: float = 0.0
    
    # Internal tracking
    _wait_times: list = field(default_factory=list, repr=False)
    
    def record_wait_time(self, wait_time_ms: float) -> None:
        """Record a wait time for averaging."""
        self._wait_times.append(wait_time_ms)
        if len(self._wait_times) > 1000:
            self._wait_times = self._wait_times[-500:]  # Keep recent 500
        
        self.avg_wait_time_ms = sum(self._wait_times) / len(self._wait_times)
        self.max_wait_time_ms = max(self.max_wait_time_ms, wait_time_ms)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "current_depth": self.current_depth,
            "total_enqueued": self.total_enqueued,
            "total_dequeued": self.total_dequeued,
            "total_failed": self.total_failed,
            "avg_wait_time_ms": round(self.avg_wait_time_ms, 2),
            "max_wait_time_ms": round(self.max_wait_time_ms, 2),
        }


@dataclass
class QueueItem(Generic[T]):
    """Wrapper for queue items with timing metadata."""
    
    data: T
    enqueued_at: float = field(default_factory=time.monotonic)
    claim_id: str | None = None  # For tracking items through pipeline
    conversation_id: int | None = None
    attempt: int = 1
    max_attempts: int = 3


class MonitoredQueue(Generic[T]):
    """Async queue with monitoring and statistics.
    
    Features:
    - Track queue depth, throughput, wait times
    - Support for retry with max attempts
    - Dead letter queue for failed items
    - Non-blocking stats access
    
    Example:
        queue = MonitoredQueue[ClaimItem]("claims", maxsize=100)
        await queue.put(item)
        item = await queue.get()
        queue.task_done()
    """
    
    def __init__(
        self,
        name: str,
        maxsize: int = 0,
        enable_dead_letter: bool = True,
    ):
        """Initialize monitored queue.
        
        Args:
            name: Human-readable name for monitoring
            maxsize: Maximum queue size (0 = unlimited)
            enable_dead_letter: Whether to track failed items
        """
        self.name = name
        self._queue: asyncio.Queue[QueueItem[T]] = asyncio.Queue(maxsize=maxsize)
        self._stats = QueueStats(name=name)
        self._dead_letter: list[QueueItem[T]] = [] if enable_dead_letter else None
        self._closed = False
        self._lock = asyncio.Lock()
    
    @property
    def stats(self) -> QueueStats:
        """Get current queue statistics (non-blocking)."""
        self._stats.current_depth = self._queue.qsize()
        return self._stats
    
    @property
    def depth(self) -> int:
        """Current number of items in queue."""
        return self._queue.qsize()
    
    @property
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()
    
    async def put(
        self,
        item: T,
        claim_id: str | None = None,
        conversation_id: int | None = None,
    ) -> None:
        """Add item to queue.
        
        Args:
            item: The item to enqueue
            claim_id: Optional ID for tracking through pipeline
            conversation_id: Optional conversation ID for grouping results
        """
        if self._closed:
            raise RuntimeError(f"Queue '{self.name}' is closed")
        
        wrapped = QueueItem(
            data=item,
            claim_id=claim_id,
            conversation_id=conversation_id,
        )
        await self._queue.put(wrapped)
        
        async with self._lock:
            self._stats.total_enqueued += 1
    
    async def put_many(self, items: list[tuple[T, str | None, int | None]]) -> None:
        """Add multiple items to queue efficiently.
        
        Args:
            items: List of (item, claim_id, conversation_id) tuples
        """
        for item, claim_id, conv_id in items:
            await self.put(item, claim_id, conv_id)
    
    async def get(self) -> QueueItem[T]:
        """Get next item from queue (blocks until available).
        
        Returns:
            QueueItem wrapper with timing metadata
        """
        item = await self._queue.get()
        
        wait_time_ms = (time.monotonic() - item.enqueued_at) * 1000
        
        async with self._lock:
            self._stats.total_dequeued += 1
            self._stats.record_wait_time(wait_time_ms)
        
        return item
    
    async def get_nowait(self) -> QueueItem[T] | None:
        """Get item without blocking, returns None if empty."""
        try:
            item = self._queue.get_nowait()
            wait_time_ms = (time.monotonic() - item.enqueued_at) * 1000
            
            async with self._lock:
                self._stats.total_dequeued += 1
                self._stats.record_wait_time(wait_time_ms)
            
            return item
        except asyncio.QueueEmpty:
            return None
    
    def task_done(self) -> None:
        """Mark current task as complete."""
        self._queue.task_done()
    
    async def requeue(self, item: QueueItem[T]) -> bool:
        """Requeue a failed item for retry.
        
        Args:
            item: The item to retry
            
        Returns:
            True if requeued, False if max attempts reached (sent to dead letter)
        """
        item.attempt += 1
        item.enqueued_at = time.monotonic()  # Reset timing
        
        if item.attempt > item.max_attempts:
            async with self._lock:
                self._stats.total_failed += 1
            
            if self._dead_letter is not None:
                self._dead_letter.append(item)
            
            return False
        
        await self._queue.put(item)
        return True
    
    async def mark_failed(self, item: QueueItem[T]) -> None:
        """Mark item as permanently failed (goes to dead letter)."""
        async with self._lock:
            self._stats.total_failed += 1
        
        if self._dead_letter is not None:
            self._dead_letter.append(item)
    
    def get_dead_letters(self) -> list[QueueItem[T]]:
        """Get all items that failed permanently."""
        return self._dead_letter.copy() if self._dead_letter else []
    
    async def join(self) -> None:
        """Wait until all items are processed."""
        await self._queue.join()
    
    def close(self) -> None:
        """Close the queue (no more items can be added)."""
        self._closed = True
    
    def __repr__(self) -> str:
        return f"MonitoredQueue(name='{self.name}', depth={self.depth})"

