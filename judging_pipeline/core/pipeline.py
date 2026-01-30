"""Pipeline orchestrator for managing workers and queues."""

from __future__ import annotations

import asyncio
from typing import Any, TypeVar

from ..logging_config import get_logger
from .queue import MonitoredQueue
from .worker import Worker
from .monitor import QueueMonitor, PipelineSnapshot

logger = get_logger()

T = TypeVar("T")


class Pipeline:
    """Orchestrator for queue-based processing pipeline.
    
    Manages the lifecycle of:
    - Input/output queues for each stage
    - Worker pools for each stage
    - Monitoring and progress tracking
    
    Example:
        pipeline = Pipeline()
        
        # Add stages
        pipeline.add_stage(ClaimExtractor(...))
        pipeline.add_stage(WebSearcher(...))
        pipeline.add_stage(Judge(...))
        
        # Run
        results = await pipeline.run(input_items, progress_callback=print)
    """
    
    def __init__(self, name: str = "EvaluationPipeline"):
        """Initialize pipeline.
        
        Args:
            name: Human-readable pipeline name
        """
        self.name = name
        self.workers: list[Worker] = []
        self.queues: list[MonitoredQueue] = []
        self.monitor: QueueMonitor | None = None
        self._results_queue: MonitoredQueue | None = None
    
    def add_queue(self, queue: MonitoredQueue) -> MonitoredQueue:
        """Register a queue for monitoring.
        
        Args:
            queue: The queue to register
            
        Returns:
            The same queue (for chaining)
        """
        self.queues.append(queue)
        return queue
    
    def add_worker(self, worker: Worker) -> Worker:
        """Register a worker.
        
        Args:
            worker: The worker to register
            
        Returns:
            The same worker (for chaining)
        """
        self.workers.append(worker)
        return worker
    
    def set_results_queue(self, queue: MonitoredQueue) -> None:
        """Set the final results queue for completion tracking.
        
        Args:
            queue: The queue that collects final results
        """
        self._results_queue = queue
    
    async def run(
        self,
        input_queue: MonitoredQueue,
        total_items: int,
        monitor_interval: float = 2.0,
        progress_callback: callable | None = None,
    ) -> list[Any]:
        """Run the pipeline to completion.
        
        Args:
            input_queue: Queue with initial items
            total_items: Total number of items to process
            monitor_interval: Seconds between progress updates
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of final results
        """
        logger.info("=" * 60)
        logger.info(f"Starting {self.name}")
        logger.info(f"Total items: {total_items}")
        logger.info(f"Workers: {len(self.workers)}")
        logger.info("=" * 60)
        
        # Setup monitor
        self.monitor = QueueMonitor(
            queues=self.queues,
            workers=self.workers,
            total_items=total_items,
        )
        
        if self._results_queue:
            self.monitor.set_final_queue(self._results_queue)
        
        if progress_callback:
            self.monitor.on_update(lambda s: progress_callback(str(s)))
        else:
            self.monitor.on_update(lambda s: print(str(s)))
        
        # Start monitor
        await self.monitor.start(interval=monitor_interval)
        
        # Start all workers
        for worker in self.workers:
            await worker.start()
        
        # Wait for input queue to be processed
        await input_queue.join()
        
        # Wait for all intermediate queues to empty
        for queue in self.queues:
            if queue != input_queue:
                await queue.join()
        
        # Stop all workers
        for worker in self.workers:
            await worker.stop()
        
        # Stop monitor and get final stats
        final_snapshot = await self.monitor.stop()
        
        # Collect results from results queue
        results = []
        if self._results_queue:
            while not self._results_queue.empty:
                item = await self._results_queue.get_nowait()
                if item:
                    results.append(item.data)
                    self._results_queue.task_done()
        
        logger.info("=" * 60)
        logger.info(f"{self.name} Complete")
        logger.info(f"Total processed: {final_snapshot.total_completed}")
        logger.info(f"Total failed: {final_snapshot.total_failed}")
        logger.info(f"Elapsed: {final_snapshot.elapsed_seconds:.1f}s")
        logger.info("=" * 60)
        
        return results
    
    async def run_streaming(
        self,
        input_queue: MonitoredQueue,
        results_queue: MonitoredQueue,
        total_items: int,
        monitor_interval: float = 2.0,
        progress_callback: callable | None = None,
    ):
        """Run pipeline and yield results as they complete.
        
        This is useful for processing results as they arrive rather than
        waiting for the entire pipeline to complete.
        
        Args:
            input_queue: Queue with initial items
            results_queue: Queue where final results are placed
            total_items: Total number of items to process
            monitor_interval: Seconds between progress updates
            progress_callback: Optional callback for progress updates
            
        Yields:
            Results as they complete
        """
        # Setup monitor
        self.monitor = QueueMonitor(
            queues=self.queues,
            workers=self.workers,
            total_items=total_items,
        )
        self.monitor.set_final_queue(results_queue)
        
        if progress_callback:
            self.monitor.on_update(lambda s: progress_callback(str(s)))
        
        # Start monitor and workers
        await self.monitor.start(interval=monitor_interval)
        
        for worker in self.workers:
            await worker.start()
        
        # Yield results as they complete
        completed = 0
        while completed < total_items:
            item = await results_queue.get()
            yield item.data
            results_queue.task_done()
            completed += 1
        
        # Cleanup
        for worker in self.workers:
            await worker.stop()
        
        await self.monitor.stop()

