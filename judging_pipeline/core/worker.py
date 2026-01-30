"""Base worker class for pipeline stages."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from .queue import MonitoredQueue, QueueItem

TIn = TypeVar("TIn")
TOut = TypeVar("TOut")


@dataclass
class WorkerStats:
    """Statistics for a worker."""
    
    name: str
    worker_id: int
    items_processed: int = 0
    items_failed: int = 0
    total_processing_time_ms: float = 0.0
    is_running: bool = False
    current_item_id: str | None = None
    
    @property
    def avg_processing_time_ms(self) -> float:
        """Average processing time per item."""
        if self.items_processed == 0:
            return 0.0
        return self.total_processing_time_ms / self.items_processed
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "worker_id": self.worker_id,
            "items_processed": self.items_processed,
            "items_failed": self.items_failed,
            "avg_processing_time_ms": round(self.avg_processing_time_ms, 2),
            "is_running": self.is_running,
            "current_item_id": self.current_item_id,
        }


class Worker(ABC, Generic[TIn, TOut]):
    """Base class for pipeline workers.
    
    A worker reads items from an input queue, processes them,
    and writes results to output queue(s).
    
    Subclasses must implement:
    - process(): The actual work to do on each item
    
    Features:
    - Automatic retry on failure
    - Statistics tracking
    - Graceful shutdown
    - Rate limiting support
    
    Example:
        class MyWorker(Worker[InputType, OutputType]):
            async def process(self, item: InputType) -> OutputType | list[OutputType]:
                # Do work...
                return result
    """
    
    def __init__(
        self,
        name: str,
        input_queue: MonitoredQueue[TIn],
        output_queue: MonitoredQueue[TOut] | None = None,
        num_workers: int = 1,
        rate_limit_delay: float = 0.0,
    ):
        """Initialize worker.
        
        Args:
            name: Human-readable worker name
            input_queue: Queue to read items from
            output_queue: Queue to write results to (None for final stage)
            num_workers: Number of concurrent worker instances
            rate_limit_delay: Seconds to wait between processing items
        """
        self.name = name
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.num_workers = num_workers
        self.rate_limit_delay = rate_limit_delay
        
        self._tasks: list[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        self._worker_stats: list[WorkerStats] = [
            WorkerStats(name=name, worker_id=i) for i in range(num_workers)
        ]
        
        # Track all outputs for caching purposes
        self._all_outputs: list[TOut] = []
        self._track_outputs: bool = False
    
    @property
    def stats(self) -> list[WorkerStats]:
        """Get statistics for all worker instances."""
        return self._worker_stats
    
    @property
    def total_processed(self) -> int:
        """Total items processed across all workers."""
        return sum(w.items_processed for w in self._worker_stats)
    
    @property
    def total_failed(self) -> int:
        """Total items failed across all workers."""
        return sum(w.items_failed for w in self._worker_stats)
    
    def enable_output_tracking(self) -> None:
        """Enable tracking of all outputs for caching purposes."""
        self._track_outputs = True
        self._all_outputs = []
    
    def get_all_outputs(self) -> list[TOut]:
        """Get all tracked outputs. Only populated if enable_output_tracking() was called."""
        return self._all_outputs
    
    @abstractmethod
    async def process(self, item: TIn, item_wrapper: QueueItem[TIn]) -> TOut | list[TOut] | None:
        """Process a single item.
        
        Args:
            item: The input item to process
            item_wrapper: The queue wrapper (for accessing claim_id, conversation_id, etc.)
            
        Returns:
            - Single output item
            - List of output items (for 1-to-many transformations)
            - None to skip outputting (item still marked as processed)
            
        Raises:
            Exception: On failure (will be retried or sent to dead letter)
        """
        raise NotImplementedError
    
    async def setup(self) -> None:
        """Optional setup before processing starts. Override in subclass."""
        pass
    
    async def teardown(self) -> None:
        """Optional cleanup after processing ends. Override in subclass."""
        pass
    
    async def _worker_loop(self, worker_id: int) -> None:
        """Main loop for a single worker instance."""
        stats = self._worker_stats[worker_id]
        stats.is_running = True
        
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Try to get item with timeout to allow shutdown checks
                    try:
                        item_wrapper = await asyncio.wait_for(
                            self.input_queue.get(),
                            timeout=0.5,
                        )
                    except asyncio.TimeoutError:
                        continue
                    
                    stats.current_item_id = item_wrapper.claim_id
                    
                    # Process the item
                    start_time = time.monotonic()
                    
                    try:
                        result = await self.process(item_wrapper.data, item_wrapper)
                        
                        # Track timing
                        elapsed_ms = (time.monotonic() - start_time) * 1000
                        stats.total_processing_time_ms += elapsed_ms
                        stats.items_processed += 1
                        
                        # Send to output queue if we have results
                        if result is not None and self.output_queue is not None:
                            if isinstance(result, list):
                                for r in result:
                                    await self.output_queue.put(
                                        r,
                                        claim_id=item_wrapper.claim_id,
                                        conversation_id=item_wrapper.conversation_id,
                                    )
                                    # Track outputs if enabled
                                    if self._track_outputs:
                                        self._all_outputs.append(r)
                            else:
                                await self.output_queue.put(
                                    result,
                                    claim_id=item_wrapper.claim_id,
                                    conversation_id=item_wrapper.conversation_id,
                                )
                                # Track outputs if enabled
                                if self._track_outputs:
                                    self._all_outputs.append(result)
                        
                        self.input_queue.task_done()
                        
                        # Rate limiting
                        if self.rate_limit_delay > 0:
                            await asyncio.sleep(self.rate_limit_delay)
                    
                    except Exception as e:
                        # Processing failed
                        stats.items_failed += 1
                        
                        # Try to requeue
                        requeued = await self.input_queue.requeue(item_wrapper)
                        
                        if not requeued:
                            print(f"[{self.name}:{worker_id}] Item failed permanently after {item_wrapper.max_attempts} attempts: {e}")
                        else:
                            print(f"[{self.name}:{worker_id}] Item failed (attempt {item_wrapper.attempt - 1}), requeuing: {e}")
                        
                        self.input_queue.task_done()
                    
                    finally:
                        stats.current_item_id = None
                
                except asyncio.CancelledError:
                    break
        
        finally:
            stats.is_running = False
    
    async def start(self) -> None:
        """Start all worker instances."""
        await self.setup()
        
        for worker_id in range(self.num_workers):
            task = asyncio.create_task(
                self._worker_loop(worker_id),
                name=f"{self.name}-{worker_id}",
            )
            self._tasks.append(task)
        
        print(f"[{self.name}] Started {self.num_workers} worker(s)")
    
    async def stop(self, timeout: float = 5.0) -> None:
        """Stop all worker instances gracefully.
        
        Args:
            timeout: Maximum seconds to wait for workers to finish
        """
        self._shutdown_event.set()
        
        if self._tasks:
            # Wait for tasks to complete with timeout
            done, pending = await asyncio.wait(
                self._tasks,
                timeout=timeout,
            )
            
            # Cancel any remaining tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            self._tasks.clear()
        
        await self.teardown()
        
        print(f"[{self.name}] Stopped (processed: {self.total_processed}, failed: {self.total_failed})")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', workers={self.num_workers})"

