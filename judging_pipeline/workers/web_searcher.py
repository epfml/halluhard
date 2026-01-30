"""Worker for web search using Serper API."""

from __future__ import annotations

import asyncio
from typing import Any, Callable
from urllib.parse import urlparse

from libs.serper import SerperSearchClient
from libs.types import SamplerBase

from ..logging_config import get_logger
from ..core.queue import MonitoredQueue, QueueItem
from ..core.worker import Worker
from ..core.domain_strategy import DomainStrategy
from ..models.work_items import ClaimItem, SearchTask, FetchTask, PDFTask
from .early_stopping import CodingEarlyStoppingState
from .package_cache import PackageVerdictCache

logger = get_logger()

MAX_URLS_TO_FETCH = 2
MAX_PDFS_TO_FETCH = 1

# Type alias for claim text builder function
ClaimTextBuilder = Callable[[ClaimItem], str]


class WebSearcherWorker(Worker[ClaimItem, SearchTask]):
    """Perform web searches for claims using Serper API.
    
    Input: ClaimItem (extracted claim)
    Output: SearchTask (with search results and URLs to fetch)
    
    This worker:
    1. Generates search queries using LLM
    2. Executes searches via Serper API
    3. Identifies top URLs and PDF links
    4. Outputs SearchTask with URLs ready for fetching
    
    The claim_text_builder parameter allows customization per task type
    (research_questions, legal_cases, medical_guidelines, etc.)
    """
    
    def __init__(
        self,
        input_queue: MonitoredQueue[ClaimItem],
        output_queue: MonitoredQueue[SearchTask],
        search_sampler: SamplerBase,
        claim_text_builder: ClaimTextBuilder,
        strategy: DomainStrategy | None = None,
        serper_api_key: str | None = None,
        serper_log_file: str | None = None,
        max_searches: int = 3,
        num_results: int = 5,
        num_workers: int = 10,
        rate_limit_delay: float = 0.1,  # Serper rate limiting
        max_concurrent_searches: int = 15,  # Concurrent Serper API calls
        early_stopping_state: CodingEarlyStoppingState | None = None,
        package_cache: PackageVerdictCache | None = None,
    ):
        """Initialize web searcher.
        
        Args:
            input_queue: Queue of claims to search
            output_queue: Queue for search results
            search_sampler: LLM sampler for query generation
            claim_text_builder: Function to convert ClaimItem to search text.
                               Should come from strategy.build_textual_claim_for_websearch.
            strategy: Domain strategy for custom planner prompts (optional).
                     If provided and has search_planner_prompt, it will be used.
            serper_api_key: Serper API key (or from env)
            serper_log_file: Optional file path to log planner interactions
            max_searches: Maximum iterative searches per claim
            num_results: Results per search
            num_workers: Number of concurrent workers
            rate_limit_delay: Delay between searches
            max_concurrent_searches: Max concurrent Serper API calls (default: 15)
            early_stopping_state: Optional early stopping state for coding tasks
        """
        super().__init__(
            name="WebSearcher",
            input_queue=input_queue,
            output_queue=output_queue,
            num_workers=num_workers,
            rate_limit_delay=rate_limit_delay,
        )
        
        self.search_sampler = search_sampler
        self.claim_text_builder = claim_text_builder
        self.strategy = strategy
        self.max_searches = max_searches
        self.num_results = num_results
        self._serper_client: SerperSearchClient | None = None
        self._serper_api_key = serper_api_key
        self._serper_log_file = serper_log_file
        # Semaphore to limit concurrent Serper API calls and prevent connection timeouts
        self._search_semaphore = asyncio.Semaphore(max_concurrent_searches)
        self.early_stopping_state = early_stopping_state
        self.package_cache = package_cache
    
    async def setup(self) -> None:
        """Initialize and start Serper client session."""
        self._serper_client = SerperSearchClient(
            api_key=self._serper_api_key,
            logger=get_logger(),
            log_file=self._serper_log_file,
        )
        await self._serper_client.start()
    
    async def teardown(self) -> None:
        """Close Serper client session."""
        if self._serper_client:
            await self._serper_client.close()
            self._serper_client = None
    
    async def process(
        self,
        item: ClaimItem,
        item_wrapper: QueueItem[ClaimItem],
    ) -> SearchTask | None:
        """Perform web search for a claim."""
        # Check early stopping before expensive web search (coding tasks only)
        # Early stopping is now per-turn: only skip if hallucination found in SAME turn
        if self.early_stopping_state:
            element_type = item.data.get("element_type")
            if element_type in ["import", "install", "function_call"]:
                if await self.early_stopping_state.should_skip(
                    item.conversation_id, item.turn_number, element_type
                ):
                    logger.info(f"⏭️  EARLY STOP [Searcher]: {element_type} skipped (conv {item.conversation_id}, turn {item.turn_number}) - no web search")
                    return None  # Skip this claim entirely
        
        # Check package whitelist and dynamic cache for import/install claims (coding optimization)
        # Skip web search for well-known packages OR packages already verified
        if self.package_cache:
            element_type = item.data.get("element_type")
            package_name = item.data.get("package_name") or item.data.get("module_name", "")
            code_snippet = item.data.get("code_snippet", "")
            
            # For imports: always use cache if package is known
            # For installs: only use cache if NO version specifier (e.g., "pip install numpy" OK,
            #               but "pip install numpy==3.2" needs search to verify version exists)
            use_cache = False
            if element_type == "import" and package_name:
                use_cache = True
            elif element_type == "install" and package_name:
                # Check for version specifiers in code snippet
                version_specifiers = ["==", ">=", "<=", ">", "<", "~=", "!=", "@", "["]
                has_version = any(spec in code_snippet for spec in version_specifiers)
                if not has_version:
                    use_cache = True
            
            if use_cache:
                # First check static whitelist (synchronous, fast)
                whitelist_verdict = self.package_cache.check_whitelist(package_name, element_type)
                if whitelist_verdict:
                    logger.info(f"⚡ WHITELIST [Searcher]: {element_type} '{package_name}' (conv {item.conversation_id}) - no web search")
                    # Return a SearchTask with pre-filled results indicating package exists
                    return SearchTask(
                        claim_id=item.claim_id,
                        conversation_id=item.conversation_id,
                        claim=item,
                        queries_executed=[f"[SKIPPED] Well-known package: {package_name}"],
                        search_results_raw=[],
                        search_results_text=f"Package '{package_name}' is a well-known, verified package. No web search needed.",
                        urls_to_fetch=[],
                        pdf_urls=[],
                        whitelist_skip=True,
                    )
                
                # Then check dynamic cache (packages verified during this run)
                cached_verdict = await self.package_cache.get(package_name)
                if cached_verdict:
                    logger.info(f"🔄 DYNAMIC CACHE [Searcher]: {element_type} '{package_name}' (conv {item.conversation_id}) - already verified, skipping web search")
                    # Return a SearchTask that skips web search based on cached verdict
                    return SearchTask(
                        claim_id=item.claim_id,
                        conversation_id=item.conversation_id,
                        claim=item,
                        queries_executed=[f"[CACHED] Previously verified: {package_name}"],
                        search_results_raw=[],
                        search_results_text=f"Package '{package_name}' was verified earlier: {cached_verdict.reason}",
                        urls_to_fetch=[],
                        pdf_urls=[],
                        whitelist_skip=True,  # Reuse whitelist_skip flag to skip judging LLM call
                        dynamic_cache_hit=True,  # New flag to indicate dynamic cache hit
                        cached_verdict_exists=cached_verdict.exists,  # Store whether package exists
                    )
        
        # Check if claim has a direct URL
        claimed_url = item.data.get("claimed_url", "").strip()
        direct_url = None
        if claimed_url:
            if self._is_valid_url(claimed_url):
                direct_url = claimed_url
                logger.info(f"🔗 DIRECT URL [Searcher]: Found claimed_url for claim {item.claim_id}: {claimed_url}")
            else:
                logger.debug(f"⚠️ Invalid URL in claim {item.claim_id}: {claimed_url}")
        
        # Build claim text using the injected builder
        claim_text = self.claim_text_builder(item)
        
        if not claim_text.strip():
            return None
        
        # Conditional logic: if URL exists, enhance claim text for URL-based search
        if direct_url:
            # URL exists: enhance claim text to prominently include URL for Serper search
            if direct_url not in claim_text:
                # URL not in claim text - add it prominently
                claim_text = f"{claim_text}\n\n[IMPORTANT: Verify against this URL: {direct_url}]"
                logger.debug(f"Added URL to claim_text for Serper search: {direct_url}")
            else:
                # URL already in claim text - ensure it's visible to LLM planner
                logger.debug(f"URL already present in claim_text, will be used by Serper search planner")
        # If no URL, keep current logic (claim_text as-is from strategy builder)
        
        # Get custom planner prompt from strategy if available
        custom_planner_prompt = None
        if self.strategy and self.strategy.search_planner_prompt:
            custom_planner_prompt = self.strategy.search_planner_prompt
        
        # Perform verification search
        (
            raw_results,
            text_results,
            queries,
            urls_with_positions,
            token_usage,
            top_urls,
        ) = await self._serper_client.perform_verification_search(
            claim_text=claim_text,
            sampler=self.search_sampler,
            max_searches=self.max_searches,
            num_results=self.num_results,
            search_semaphore=self._search_semaphore,
            context=f"claim={item.claim_id}",
            custom_planner_prompt=custom_planner_prompt,
        )

        def is_pdf_url(url: str) -> bool:
            """Check if URL points to a PDF."""
            return url.endswith(".pdf") or "/pdf/" in url
        
        logger.debug(f"WebSearcher claim {item.claim_id}: search returned {len(raw_results) if raw_results else 0} result sets")
        
        if not raw_results:
            # Search failed - still create task for fallback handling
            # But include direct URL if available (handle both PDF and HTML)
            urls_to_fetch_fallback = []
            pdf_urls_fallback = []
            if direct_url:
                if is_pdf_url(direct_url):
                    pdf_urls_fallback.append(direct_url)
                    logger.info(f"📄 Using direct PDF URL despite search failure: {direct_url}")
                else:
                    urls_to_fetch_fallback.append(direct_url)
                    logger.info(f"🔗 Using direct URL despite search failure: {direct_url}")
            
            logger.debug(f"WebSearcher claim {item.claim_id}: no results, creating fallback task")
            return SearchTask(
                claim_id=item.claim_id,
                conversation_id=item.conversation_id,
                claim=item,
                queries_executed=queries,
                search_results_raw=[],
                search_results_text="No search results found.",
                urls_to_fetch=urls_to_fetch_fallback,
                pdf_urls=pdf_urls_fallback[:MAX_PDFS_TO_FETCH],
            )
        
        # Collect all PDF URLs first (from organic results)
        pdf_urls = []
        
        # Priority 0: If direct URL is a PDF, add it to PDF queue
        if direct_url and is_pdf_url(direct_url):
            pdf_urls.append(direct_url)
            logger.info(f"📄 DIRECT PDF URL [Searcher]: Found PDF URL for claim {item.claim_id}: {direct_url}")
        
        # Collect PDFs from search results
        for result_set in raw_results:
            for organic in result_set.get("organic", []):
                link = organic.get("link", "")
                if is_pdf_url(link) and link not in pdf_urls:
                    pdf_urls.append(link)
        
        pdf_set = set(pdf_urls)
        
        # Identify HTML URLs to fetch (excluding PDFs)
        # Priority 1: Direct claimed URL if provided (and not a PDF)
        urls_to_fetch = []
        if direct_url and not is_pdf_url(direct_url):
            urls_to_fetch.append(direct_url)
            logger.debug(f"Added direct URL to fetch list: {direct_url}")
        
        # Priority 2: LLM-selected top URLs from search
        for url in (top_urls or []):
            if url not in urls_to_fetch and not is_pdf_url(url) and url not in pdf_set:
                urls_to_fetch.append(url)
            if len(urls_to_fetch) >= MAX_URLS_TO_FETCH:
                break
        
        # Priority 3: Fill with top organic results if needed (excluding PDFs)
        if len(urls_to_fetch) < MAX_URLS_TO_FETCH:
            seen = set(urls_to_fetch)
            sorted_urls = [url for pos, url in sorted(urls_with_positions, key=lambda x: x[0])]
            for url in sorted_urls:
                if url not in seen and not is_pdf_url(url):
                    urls_to_fetch.append(url)
                    seen.add(url)
                if len(urls_to_fetch) >= MAX_URLS_TO_FETCH:
                    break
        
        search_task = SearchTask(
            claim_id=item.claim_id,
            conversation_id=item.conversation_id,
            claim=item,
            queries_executed=queries,
            search_results_raw=raw_results,
            search_results_text=text_results,
            urls_to_fetch=urls_to_fetch,
            pdf_urls=pdf_urls[:MAX_PDFS_TO_FETCH],  # Limit PDFs
        )
        
        logger.debug(f"WebSearcher completed claim {item.claim_id}: {len(urls_to_fetch)} HTML URLs, {len(pdf_urls[:MAX_PDFS_TO_FETCH])} PDF URLs")
        return search_task
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if string is a valid URL."""
        try:
            parsed = urlparse(url)
            return parsed.scheme in ['http', 'https'] and parsed.netloc != ''
        except:
            return False

