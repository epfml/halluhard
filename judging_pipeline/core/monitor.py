"""Queue monitor for real-time pipeline visibility."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..logging_config import get_logger
from .queue import MonitoredQueue
from .worker import Worker

logger = get_logger()


@dataclass
class PipelineSnapshot:
    """Snapshot of pipeline state at a point in time."""
    
    timestamp: float
    queues: dict[str, dict]  # queue_name -> stats dict
    workers: dict[str, list[dict]]  # worker_name -> list of worker stats
    worker_queues: dict[str, tuple[str | None, str | None]]  # worker_name -> (input_queue, output_queue)
    total_completed: int
    total_failed: int
    elapsed_seconds: float
    
    def __str__(self) -> str:
        """Human-readable summary."""
        lines = [
            f"━━━ Pipeline Status ({self.elapsed_seconds:.1f}s elapsed) ━━━",
            f"Completed: {self.total_completed} | Failed: {self.total_failed}",
            "",
            "Queues:",
        ]
        
        max_depth = max((q.get("current_depth", 0) for q in self.queues.values()), default=1)
        
        for name, stats in self.queues.items():
            depth = stats.get("current_depth", 0)
            bar_width = 20
            filled = int((depth / max(max_depth, 1)) * bar_width) if max_depth > 0 else 0
            bar = "█" * filled + "░" * (bar_width - filled)
            
            # Don't warn on results queue - items accumulate there by design
            is_results_queue = name.lower() == "results"
            warning = " ⚠️ BOTTLENECK" if depth > 50 and not is_results_queue else ""
            lines.append(f"  {name:20s} [{bar}] {depth:4d} items{warning}")
        
        lines.append("")
        lines.append("Workers:")
        
        for name, worker_stats in self.workers.items():
            # Count workers that are both running AND actively processing an item
            active = sum(1 for w in worker_stats if w.get("is_running") and w.get("current_item_id"))
            total_processed = sum(w.get("items_processed", 0) for w in worker_stats)
            total_failed = sum(w.get("items_failed", 0) for w in worker_stats)
            avg_time = sum(w.get("avg_processing_time_ms", 0) for w in worker_stats) / len(worker_stats) if worker_stats else 0
            
            # Show queue flow
            in_q, out_q = self.worker_queues.get(name, (None, None))
            queue_info = ""
            if in_q and out_q:
                queue_info = f" ({in_q} → {out_q})"
            elif in_q:
                queue_info = f" ({in_q} →)"
            elif out_q:
                queue_info = f" (→ {out_q})"
            
            # Show failures if any
            fail_info = f" | {total_failed} failed" if total_failed > 0 else ""
            
            lines.append(f"  {name:20s} {active:3d}/{len(worker_stats):<3d} active | {total_processed:4d} done{fail_info} | {avg_time:5.0f}ms avg{queue_info}")
        
        lines.append("━" * 50)
        return "\n".join(lines)


class QueueMonitor:
    """Monitor pipeline queues and workers in real-time.
    
    Features:
    - Periodic snapshots of queue depths
    - Worker throughput tracking
    - Bottleneck detection
    - Progress callbacks for UI updates
    
    Example:
        monitor = QueueMonitor(
            queues=[claims_queue, search_queue, ...],
            workers=[extractor, searcher, ...],
        )
        monitor.on_update(lambda snapshot: print(snapshot))
        await monitor.start(interval=2.0)
    """
    
    def __init__(
        self,
        queues: list[MonitoredQueue] | None = None,
        workers: list[Worker] | None = None,
        total_items: int | None = None,
    ):
        """Initialize monitor.
        
        Args:
            queues: List of queues to monitor
            workers: List of workers to monitor
            total_items: Total expected items for progress calculation
        """
        self.queues: list[MonitoredQueue] = queues or []
        self.workers: list[Worker] = workers or []
        self.total_items = total_items
        
        self._callbacks: list[Callable[[PipelineSnapshot], None]] = []
        self._task: asyncio.Task | None = None
        self._start_time: float | None = None
        self._shutdown_event = asyncio.Event()
        self._final_queue: MonitoredQueue | None = None  # Track final output queue
    
    def add_queue(self, queue: MonitoredQueue) -> None:
        """Add a queue to monitor."""
        self.queues.append(queue)
    
    def add_worker(self, worker: Worker) -> None:
        """Add a worker to monitor."""
        self.workers.append(worker)
    
    def set_final_queue(self, queue: MonitoredQueue) -> None:
        """Set the final output queue for completion tracking."""
        self._final_queue = queue
    
    def on_update(self, callback: Callable[[PipelineSnapshot], None]) -> None:
        """Register callback for snapshot updates.
        
        Args:
            callback: Function called with each new snapshot
        """
        self._callbacks.append(callback)
    
    def take_snapshot(self) -> PipelineSnapshot:
        """Take current pipeline snapshot."""
        elapsed = time.monotonic() - self._start_time if self._start_time else 0.0
        
        # Collect queue stats
        queue_stats = {}
        for queue in self.queues:
            queue_stats[queue.name] = queue.stats.to_dict()
        
        # Collect worker stats and queue associations
        worker_stats = {}
        worker_queues = {}
        total_processed = 0
        total_failed = 0
        
        for worker in self.workers:
            worker_stats[worker.name] = [w.to_dict() for w in worker.stats]
            total_processed += worker.total_processed
            total_failed += worker.total_failed
            
            # Get queue names from worker
            in_queue = getattr(worker, 'input_queue', None)
            out_queue = getattr(worker, 'output_queue', None)
            # ContentAggregator uses content_queue instead of input_queue
            if in_queue is None:
                in_queue = getattr(worker, 'content_queue', None)
            
            in_name = in_queue.name if in_queue and hasattr(in_queue, 'name') else None
            out_name = out_queue.name if out_queue and hasattr(out_queue, 'name') else None
            worker_queues[worker.name] = (in_name, out_name)
        
        # Use final queue completed count if available
        # Note: Use total_enqueued since results queue items are enqueued but not dequeued during run
        if self._final_queue:
            total_processed = self._final_queue.stats.total_enqueued
        
        return PipelineSnapshot(
            timestamp=time.time(),
            queues=queue_stats,
            workers=worker_stats,
            worker_queues=worker_queues,
            total_completed=total_processed,
            total_failed=total_failed,
            elapsed_seconds=elapsed,
        )
    
    async def _monitor_loop(self, interval: float) -> None:
        """Internal monitoring loop."""
        while not self._shutdown_event.is_set():
            snapshot = self.take_snapshot()
            
            for callback in self._callbacks:
                try:
                    callback(snapshot)
                except Exception as e:
                    logger.warning(f"[Monitor] Callback error: {e}")
            
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=interval,
                )
                break
            except asyncio.TimeoutError:
                pass
    
    async def start(self, interval: float = 2.0) -> None:
        """Start monitoring in the background.
        
        Args:
            interval: Seconds between snapshots
        """
        self._start_time = time.monotonic()
        self._shutdown_event.clear()
        
        self._task = asyncio.create_task(
            self._monitor_loop(interval),
            name="queue-monitor",
        )
        
        logger.info(f"[Monitor] Started (interval: {interval}s)")
    
    async def stop(self) -> PipelineSnapshot:
        """Stop monitoring and return final snapshot.
        
        Returns:
            Final pipeline snapshot
        """
        self._shutdown_event.set()
        
        if self._task:
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        
        final_snapshot = self.take_snapshot()
        logger.info("[Monitor] Stopped")
        logger.info(f"\n{final_snapshot}")
        
        return final_snapshot
    
    def print_snapshot(self) -> None:
        """Print current snapshot to console."""
        logger.info(f"\n{self.take_snapshot()}")

