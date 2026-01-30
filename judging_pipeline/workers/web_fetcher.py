"""Worker for fetching web pages."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING
from urllib.parse import urlparse

from libs.browser_fetcher import BrowserFetcher, extract_pdf_links
from libs.html_cleaner import HtmlCleaner
from libs.information_extraction import check_if_url_is_pdf

from ..core.queue import MonitoredQueue, QueueItem
from ..core.worker import Worker
from ..models.work_items import SearchTask, ContentItem, PDFTask
from ..logging_config import get_logger

if TYPE_CHECKING:
    from .early_stopping import CodingEarlyStoppingState

logger = get_logger()

MAX_PDFS_TO_FETCH = 1


class WebFetcherWorker(Worker[SearchTask, ContentItem]):
    """Fetch web page content for search results.
    
    Input: SearchTask (with URLs to fetch)
    Output: ContentItem (with fetched content) + PDFTask (to pdf_queue)
    
    This worker:
    1. Fetches each URL in the search task
    2. Cleans HTML to extract text
    3. Identifies PDF links for separate processing
    4. Outputs ContentItem with all fetched content
    """
    
    def __init__(
        self,
        input_queue: MonitoredQueue[SearchTask],
        output_queue: MonitoredQueue[ContentItem],
        pdf_queue: MonitoredQueue[PDFTask] | None = None,
        max_words: int | None = None,
        num_workers: int = 20,
        rate_limit_delay: float = 0.0,
        early_stopping_state: "CodingEarlyStoppingState | None" = None,
    ):
        """Initialize web fetcher.
        
        Args:
            input_queue: Queue of search tasks
            output_queue: Queue for fetched content
            pdf_queue: Optional queue for PDF download tasks
            max_words: Maximum words per page
            num_workers: Number of concurrent workers
            rate_limit_delay: Delay between fetches
            early_stopping_state: Optional early stopping state for coding tasks
        """
        super().__init__(
            name="WebFetcher",
            input_queue=input_queue,
            output_queue=output_queue,
            num_workers=num_workers,
            rate_limit_delay=rate_limit_delay,
        )
        
        self.pdf_queue = pdf_queue
        self._browser_fetcher: BrowserFetcher | None = None
        self._html_cleaner: HtmlCleaner | None = None
        self._max_words = max_words
        self.early_stopping_state = early_stopping_state
    
    async def setup(self) -> None:
        """Initialize browser and cleaner."""
        self._browser_fetcher = BrowserFetcher()
        self._html_cleaner = HtmlCleaner(max_words=self._max_words)
    
    async def process(
        self,
        item: SearchTask,
        item_wrapper: QueueItem[SearchTask],
    ) -> ContentItem | None:
        """Fetch all URLs in search task."""
        # Early stopping check - skip fetching if category already has hallucination in this turn
        if self.early_stopping_state:
            element_type = item.claim.data.get("element_type")
            if element_type in ["import", "install", "function_call"]:
                if await self.early_stopping_state.should_skip(
                    item.conversation_id, item.claim.turn_number, element_type
                ):
                    logger.info(
                        f"⏭️  EARLY STOP [Fetcher]: {element_type} skipped (conv {item.conversation_id}, turn {item.claim.turn_number}) - no HTTP fetch"
                    )
                    # Return empty content - will flow through pipeline and be skipped at judge
                    return ContentItem(
                        claim_id=item.claim_id,
                        conversation_id=item.conversation_id,
                        claim=item.claim,
                        contents=[],
                        pdf_contents=[],
                        search_results_text=item.search_results_text,
                        queries=item.queries_executed,
                        expected_pdf_count=0,
                        whitelist_skip=item.whitelist_skip,
                        dynamic_cache_hit=item.dynamic_cache_hit,
                        cached_verdict_exists=item.cached_verdict_exists,
                    )
        
        contents = []
        pdf_links_found = []
        
        # Build URL -> snippet mapping from search results
        url_to_info = {}
        for result_set in item.search_results_raw:
            for organic in result_set.get("organic", []):
                link = organic.get("link", "")
                if link:
                    url_to_info[link] = {
                        "title": organic.get("title", ""),
                        "snippet": organic.get("snippet", ""),
                    }
        
        # Fetch each URL
        for url in item.urls_to_fetch:
            info = url_to_info.get(url, {"title": "", "snippet": ""})
            
            # Check if PDF
            if await check_if_url_is_pdf(url):
                # Queue for PDF processing
                if self.pdf_queue:
                    await self.pdf_queue.put(
                        PDFTask(
                            claim_id=item.claim_id,
                            conversation_id=item.conversation_id,
                            url=url,
                            title=info["title"],
                            search_results_text=item.search_results_text,
                        ),
                        claim_id=item.claim_id,
                        conversation_id=item.conversation_id,
                    )
                continue
            
            # Fetch HTML
            result = await self._fetch_url(url)
            
            if result["success"]:
                contents.append({
                    "title": info["title"],
                    "url": url,
                    "snippet": info["snippet"],
                    "content": result["content"],
                })
                
                # Collect PDF links from page
                pdf_links_found.extend(result.get("pdf_links", []))
        
        # Queue PDF links for processing
        if self.pdf_queue:
            # Combine and deduplicate PDF URLs, prioritizing search results
            all_pdf_urls: list[str] = []
            seen_urls: set[str] = set()
            
            # First add PDFs from search (higher priority)
            for pdf_url in item.pdf_urls:
                if pdf_url not in seen_urls:
                    all_pdf_urls.append(pdf_url)
                    seen_urls.add(pdf_url)
            
            # Then add PDFs found on pages
            for pdf_url in pdf_links_found:
                if pdf_url not in seen_urls:
                    all_pdf_urls.append(pdf_url)
                    seen_urls.add(pdf_url)
            
            # Queue up to MAX_PDFS_TO_FETCH
            pdfs_to_queue = all_pdf_urls[:MAX_PDFS_TO_FETCH]
            for pdf_url in pdfs_to_queue:
                await self.pdf_queue.put(PDFTask(
                    claim_id=item.claim_id,
                    conversation_id=item.conversation_id,
                    url=pdf_url,
                    title="PDF",
                    search_results_text=item.search_results_text,
                ))
            expected_pdf_count = len(pdfs_to_queue)
        else:
            expected_pdf_count = 0
        
        return ContentItem(
            claim_id=item.claim_id,
            conversation_id=item.conversation_id,
            claim=item.claim,
            contents=contents,
            pdf_contents=[],  # Will be filled by aggregator
            search_results_text=item.search_results_text,
            queries=item.queries_executed,
            expected_pdf_count=expected_pdf_count,
            whitelist_skip=item.whitelist_skip,
            dynamic_cache_hit=item.dynamic_cache_hit,
            cached_verdict_exists=item.cached_verdict_exists,
        )
    
    async def _fetch_url(self, url: str) -> dict:
        """Fetch and clean a single URL."""
        try:
            html, error = await self._browser_fetcher.fetch_html(url, force_selenium=False)
            
            if error:
                return {"success": False, "content": "", "error": error, "pdf_links": []}
            
            # Extract PDF links
            pdf_links = extract_pdf_links(html, base_url=url)
            
            # Clean HTML
            cleaned = self._html_cleaner.clean(html, source_url=url)
            
            if not cleaned or len(cleaned.strip()) < 100:
                return {"success": False, "content": "", "error": "No meaningful content", "pdf_links": pdf_links}
            
            return {"success": True, "content": cleaned, "error": None, "pdf_links": pdf_links}
        
        except Exception as e:
            return {"success": False, "content": "", "error": str(e), "pdf_links": []}

