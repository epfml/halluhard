"""Workers for PDF downloading and conversion."""

from __future__ import annotations

from typing import Any

from libs.information_extraction import extract_pdf_as_markdown

from ..logging_config import get_logger
from ..core.queue import MonitoredQueue, QueueItem
from ..core.worker import Worker
from ..models.work_items import PDFTask, PDFResult

logger = get_logger()


class PDFDownloaderWorker(Worker[PDFTask, PDFTask]):
    """Download PDFs and pass to converter.
    
    Input: PDFTask (with URL)
    Output: PDFTask (with downloaded content marked)
    
    Note: This is a simplified version. The actual PDF download happens
    in PDFConverterWorker using extract_pdf_as_markdown which handles
    both download and conversion.
    """
    
    def __init__(
        self,
        input_queue: MonitoredQueue[PDFTask],
        output_queue: MonitoredQueue[PDFTask],
        num_workers: int = 5,
        rate_limit_delay: float = 0.5,
    ):
        super().__init__(
            name="PDFDownloader",
            input_queue=input_queue,
            output_queue=output_queue,
            num_workers=num_workers,
            rate_limit_delay=rate_limit_delay,
        )
    
    async def process(
        self,
        item: PDFTask,
        item_wrapper: QueueItem[PDFTask],
    ) -> PDFTask:
        """Pass through to converter (download happens there)."""
        return item


class PDFConverterWorker(Worker[PDFTask, PDFResult]):
    """Convert PDFs to markdown.
    
    Input: PDFTask (with URL)
    Output: PDFResult with markdown content (or error)
    
    This worker:
    1. Downloads the PDF
    2. Converts to markdown using markitdown
    3. Outputs PDFResult for aggregation (always outputs, even on failure)
    """
    
    def __init__(
        self,
        input_queue: MonitoredQueue[PDFTask],
        output_queue: MonitoredQueue[PDFResult],
        timeout_seconds: int = 30,
        conversion_timeout_seconds: int = 60,
        max_pdf_size_mb: int = 2,
        num_workers: int = 10,
        rate_limit_delay: float = 0.0,
    ):
        """Initialize PDF converter.
        
        Args:
            input_queue: Queue of PDF tasks
            output_queue: Queue for converted content
            timeout_seconds: Download timeout
            conversion_timeout_seconds: Conversion timeout
            max_pdf_size_mb: Maximum PDF size
            num_workers: Number of concurrent workers
            rate_limit_delay: Delay between conversions
        """
        super().__init__(
            name="PDFConverter",
            input_queue=input_queue,
            output_queue=output_queue,
            num_workers=num_workers,
            rate_limit_delay=rate_limit_delay,
        )
        
        self.timeout_seconds = timeout_seconds
        self.conversion_timeout_seconds = conversion_timeout_seconds
        self.max_pdf_size_mb = max_pdf_size_mb
    
    async def process(
        self,
        item: PDFTask,
        item_wrapper: QueueItem[PDFTask],
    ) -> PDFResult:
        """Download and convert PDF to markdown.
        
        Always returns a PDFResult so the aggregator can track completion.
        """
        try:
            markdown = await extract_pdf_as_markdown(
                item.url,
                pdf_semaphore=None,  # Worker handles concurrency
                max_retries=1,
                timeout_seconds=self.timeout_seconds,
                conversion_timeout_seconds=self.conversion_timeout_seconds,
                max_pdf_size_mb=self.max_pdf_size_mb,
            )
            
            if not markdown:
                return PDFResult(
                    claim_id=item.claim_id,
                    conversation_id=item.conversation_id,
                    url=item.url,
                    title=item.title or "PDF",
                    content="",
                    success=False,
                    error="Conversion returned empty content",
                )
            
            return PDFResult(
                claim_id=item.claim_id,
                conversation_id=item.conversation_id,
                url=item.url,
                title=item.title or "PDF Content",
                content=markdown,
                success=True,
            )
        
        except Exception as e:
            logger.warning(f"Failed to convert PDF {item.url}: {e}")
            return PDFResult(
                claim_id=item.claim_id,
                conversation_id=item.conversation_id,
                url=item.url,
                title=item.title or "PDF",
                content="",
                success=False,
                error=str(e),
            )

