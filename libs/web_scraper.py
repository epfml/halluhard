"""Web scraping orchestration: Serper API + Browser Fetching + HTML Cleaning."""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Dict, Any, List
from urllib.parse import urlparse

# Fix Windows asyncio SSL issues
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import dotenv
from libs.browser_fetcher import BrowserFetcher, extract_pdf_links
from libs.html_cleaner import HtmlCleaner
from libs.information_extraction import check_if_url_is_pdf, extract_pdf_as_markdown
from libs.serper import SerperSearchClient
from libs.types import SamplerBase

dotenv.load_dotenv()


class WebScraper:
    """Orchestrate web scraping: Serper search + browser fetching + cleaning."""
    
    def __init__(
        self,
        search_query_sampler: SamplerBase | None = None,
        serper_api_key: str | None = None,
        max_words: int | None = None,
        search_semaphore: asyncio.Semaphore | None = None,
        static_fetch_semaphore: asyncio.Semaphore | None = None,
        dynamic_fetch_semaphore: asyncio.Semaphore | None = None,
        pdf_semaphore: asyncio.Semaphore | None = None,
        max_iterations_serper_searches: int = 5,
        num_results_serper_search: int = 5,
    ):
        """Initialize web scraper.
        
        Args:
            search_query_sampler: Optional SamplerBase instance for agentic search
            serper_api_key: Serper API key for Google Search
            max_words: Max words to keep per page
            search_semaphore: Optional semaphore to limit concurrent Serper API calls (default: 20)
            static_fetch_semaphore: Optional semaphore to limit concurrent static fetches
            dynamic_fetch_semaphore: Optional semaphore to limit concurrent dynamic fetches
            pdf_semaphore: Optional semaphore to limit concurrent PDF downloads
            max_iterations_serper_searches: Maximum number of Serper searches
            num_results_serper_search: Number of results per Serper search
        """
        self.search_query_sampler = search_query_sampler
        self.serper_api_key = serper_api_key or os.environ.get("SERPER_API_KEY")
        self.serper_client = SerperSearchClient(self.serper_api_key)
        self.browser_fetcher = BrowserFetcher()
        self.html_cleaner = HtmlCleaner(max_words=max_words)
        self.search_semaphore = search_semaphore  # Limit concurrent Serper API calls
        self.static_fetch_semaphore = static_fetch_semaphore  # Limit concurrent static fetches
        self.dynamic_fetch_semaphore = dynamic_fetch_semaphore  # Limit concurrent dynamic fetches
        self.pdf_semaphore = pdf_semaphore  # Limit concurrent PDF downloads
        self.max_iterations_serper_searches = max_iterations_serper_searches
        self.num_results_serper_search = num_results_serper_search


    async def search_and_fetch(
        self,
        claim: Dict[str, Any],
        num_serper_results: int = 5,
        num_urls_to_fetch: int = 2,
        no_final_fallback: bool = False,
    ) -> Dict[str, Any]:
        """Search and fetch content for a research claim.
        
        Pipeline:
        1. Check if URL exists in claim → fetch directly
        2. If no URL or fetch fails → Serper search
        3. Fetch top N URLs from Serper results
        4. Clean all fetched content
        5. Return cleaned content + fallback snippets
        
        Args:
            claim: Research claim dict (must have: title, authors, year, source_type, url)
            num_serper_results: Number of Serper results to request
            num_urls_to_fetch: Number of top URLs to fetch from Serper
            no_final_fallback: If True, do not return fallback content if all fetches fail
                If False, return serper snippets content if all fetches fail and mark as a success
                If True, return success False and empty content if all fetches fail

        Returns:
            Dict with: {
                'success': bool,
                'content': str (cleaned content),
                'method': str ('direct_url', 'serper_fetch', or 'serper_snippets_only'),
                'pdf_links': List[str] (all PDF links found across all fetched pages)
            }
        """
        # Step 1: Try direct URL if available
        claimed_url = claim.get('url', '').strip()
        if claimed_url and self._is_valid_url(claimed_url):
            direct_result = await self._fetch_and_clean_url(claimed_url)
            
            if direct_result['success']:
                return {
                    'success': True,
                    'serper_search_output': '',
                    'content': [{"title": claim.get('title', ''), "url": claimed_url, "snippet": "", "content": direct_result['content']}],
                    'method': 'direct_url',
                    'pdf_links': direct_result.get('pdf_links', []),
                    'queries': []
                }
            else:
                print(f"  ⚠️ Direct claimed URL fetch failed: {direct_result['error']}, falling back to Serper search.")
        
        # Step 2: Call Serper API
        if not claim.get('claimed_content'):
            print(f"  ⚠️ No claimed content found, using Serper search with title, year and authors.")
            claim_text = f"{claim.get('title', '')} {claim.get('year', '')} {claim.get('authors', '')}"
        
        search_results, serper_text_results, queries, urls_with_positions, token_usage, top_urls = await self._call_serper(claim.get('claimed_content', ''))
        
        if not search_results or len(search_results) == 0:
            print(f"  ⚠️ No Serper results found")
            return {
                'success': True,
                'serper_search_output': '',
                'content': [{"title": "Google Search Results", "url": "", "snippet": "", "content": "No Google Search Results match the title, year and authors of the claim. It is a reference failure."}],
                'method': 'failed',
                'error': 'No Serper results found',
                'pdf_links': [],
                'token_usage': token_usage,
                'queries': queries
            }
        
        # Step 3: Fetch top N URLs (prioritizing planner selection)
        urls_to_fetch = []
        if top_urls:
            # Use URLs selected by the planner
            urls_to_fetch = top_urls[:num_urls_to_fetch]
            # print(f"  → Planner selected {len(urls_to_fetch)} URLs: {urls_to_fetch}")
        
        # If planner didn't return enough URLs, fill with top organic results
        if len(urls_to_fetch) < num_urls_to_fetch:
            # Remove duplicates from urls_with_positions on url basis
            urls_with_positions = list({url: (pos, url) for pos, url in urls_with_positions}.values())
            
            sorted_organic = [url for position, url in sorted(urls_with_positions, key=lambda x: x[0])]
            for url in sorted_organic:
                if len(urls_to_fetch) >= num_urls_to_fetch:
                    break
                if url not in urls_to_fetch:
                    urls_to_fetch.append(url)

        fetched_results = await self._fetch_multiple_urls(urls_to_fetch)
        
        # Step 4: Combine cleaned content and collect PDF links
        combined_content = []
        all_pdf_links = []
        
        # Map URL to its Serper snippet info for retrieval
        url_to_snippet = {}
        for res_list in search_results:
             for organic in res_list.get('organic', []):
                  if organic.get('link'):
                       url_to_snippet[organic.get('link')] = organic

        for i, (url, result) in enumerate(zip(urls_to_fetch, fetched_results)):
            # Retrieve the correct snippet info using the URL
            serper_info = url_to_snippet.get(url, {})
            title = serper_info.get('title', 'Unknown Title')
            snippet = serper_info.get('snippet', '')
            
            # Collect PDF links from this result
            if 'pdf_links' in result:
                all_pdf_links.extend(result['pdf_links'])
            
            if result['success']:
                combined_content.append({"title": title, "url": url, "snippet": snippet, "content": result['content']})
        
        # Remove duplicate PDF links
        all_pdf_links = list(dict.fromkeys(all_pdf_links))  # Preserves order, removes duplicates
        
        # Step 5: Return results
        if len(combined_content) > 0:
            return {
                'success': True,
                'serper_search_output': serper_text_results,
                'content': combined_content,
                'method': 'serper_fetch',
                'pdf_links': all_pdf_links,
                'token_usage': token_usage,
                'queries': queries,
                'top_urls': top_urls
            }
        else:
            # Fallback: Return Serper snippets only
            print("  ⚠️ All URL fetches failed, using Serper snippets as fallback")
            
            return {
                'success': not no_final_fallback,
                'serper_search_output': serper_text_results,
                'content': [{"title": "Google Search Results", "url": "", "snippet": "", "content": serper_text_results}],
                'method': 'serper_snippets_only',
                'pdf_links': [],
                'token_usage': token_usage,
                'queries': queries,
                'top_urls': top_urls
            }
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if string is a valid URL."""
        try:
            parsed = urlparse(url)
            return parsed.scheme in ['http', 'https'] and parsed.netloc != ''
        except:
            return False
    
    async def _call_serper(self, claim_text: str) -> Tuple[Dict[str, Any], str, List[str], List[Tuple[int, str]], Dict[str, int], List[str]]:
        """Call Serper API for Google search using agentic verification search if sampler provided.
        
        Args:
            claim_text: Text of the claim to verify (or query string if no sampler)
            
        Returns:
            Tuple of (Search results, List of queries, List of URLs with positions, Token usage, Top URLs)
        """
        raw_results, text_results, queries, urls_with_positions, token_usage, top_urls = await self.serper_client.perform_verification_search(
            claim_text=claim_text,
            sampler=self.search_query_sampler,
            max_searches=self.max_iterations_serper_searches,
            num_results=self.num_results_serper_search,
            search_semaphore=self.search_semaphore
        )
        
        return raw_results, text_results, queries, urls_with_positions, token_usage, top_urls
    
    async def _fetch_and_clean_url(self, url: str) -> Dict[str, Any]:
        """Fetch URL with browser and clean HTML, or download PDF if URL points to a PDF.
        
        Args:
            url: URL to fetch
            
        Returns:
            Dict with: success, content (cleaned or PDF markdown), error, pdf_links
        """        
        # Check if URL points to a PDF - if so, handle it directly
        if await check_if_url_is_pdf(url):
            # Download and convert PDF to markdown
            pdf_markdown = await extract_pdf_as_markdown(
                url,
                pdf_semaphore=self.pdf_semaphore,
            )
            
            if pdf_markdown:
                # Return PDF content as markdown
                return {
                    'success': True,
                    'content': pdf_markdown,
                    'error': None,
                    'pdf_links': []  # We don't want to re-explore later the same PDF link
                }
            else:
                # PDF download failed, treat as error
                return {
                    'success': False,
                    'content': '',
                    'error': 'PDF download or conversion failed',
                    'pdf_links': [] # We don't want to re-explore later the same PDF link
                }
        
        # Not a PDF - proceed with normal HTML fetching
        # First attempt: try trafilatura (fast)        
        if self.static_fetch_semaphore is not None:
            async with self.static_fetch_semaphore:
                html, error = await self.browser_fetcher.fetch_html(url, force_selenium=False)
        else:
            html, error = await self.browser_fetcher.fetch_html(url, force_selenium=False)
        
        if error:
            return {'success': False, 'content': '', 'error': error, 'pdf_links': []}

        # Extract PDF links
        pdf_links = extract_pdf_links(html, base_url=url)
        if pdf_links and len(pdf_links) > 0:
            print(f"  → Found {len(pdf_links)} PDF links")
        
        # Clean HTML
        cleaned = self.html_cleaner.clean(html, source_url=url)
        
        # If no meaningful content extracted, retry with Selenium
        if not cleaned or len(cleaned.strip()) < 100:            
            # Second attempt: force Selenium (more robust for JS-heavy sites)
            # if self.dynamic_fetch_semaphore is not None:
            #     async with self.dynamic_fetch_semaphore:
            #         html_selenium, error_selenium = await self.browser_fetcher.fetch_html(url, force_selenium=True)
            # else:
            #     html_selenium, error_selenium = await self.browser_fetcher.fetch_html(url, force_selenium=True)
            
            # if error_selenium:
            #     return {'success': False, 'content': '', 'error': f'Trafilatura and Selenium both failed. Selenium error: {error_selenium}', 'pdf_links': pdf_links}
            
            # # Extract PDF links again (Selenium might find different links)
            # pdf_links_selenium = extract_pdf_links(html_selenium, base_url=url)
            # if pdf_links_selenium:
            #     pdf_links.extend(pdf_links_selenium)
            #     pdf_links = list(dict.fromkeys(pdf_links))  # Remove duplicates
            
            # # Clean the Selenium HTML
            # cleaned = self.html_cleaner.clean(html_selenium, source_url=url)
            
            if not cleaned or len(cleaned.strip()) < 100:
                print(f"  ⚠️ Static fetching and Selenium both failed to extract meaningful content")
                return {'success': False, 'content': '', 'error': 'No meaningful content extracted even with Selenium', 'pdf_links': pdf_links}
        
        return {'success': True, 'content': cleaned, 'error': None, 'pdf_links': pdf_links}
    
    async def _fetch_multiple_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Fetch multiple URLs concurrently.
        
        Args:
            urls: List of URLs to fetch
            
        Returns:
            List of result dicts (same order as input)
        """
        tasks = [self._fetch_and_clean_url(url) for url in urls]
        results = await asyncio.gather(*tasks)
        return results
    