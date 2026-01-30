"""Browser-based webpage fetcher using Selenium with Chromium."""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from typing import List
from urllib.parse import urljoin, urlparse

# Fix Windows asyncio SSL issues
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import trafilatura
import httpx
import random

_logger = logging.getLogger(__name__)


# =============================================================================
# Shared HTTP Client (connection pooling for efficiency)
# =============================================================================

_http_client: httpx.AsyncClient | None = None

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


async def get_http_client() -> httpx.AsyncClient:
    """Get or create the shared httpx client with connection pooling."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        # Force HTTP/1.1 to avoid TLS/HTTP2 issues on Windows
        _http_client = httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(20.0, connect=10.0),
            limits=httpx.Limits(max_connections=30, max_keepalive_connections=15),
            headers=_DEFAULT_HEADERS,
            verify=True,
            http1=True,
            http2=False,
        )
        _logger.debug("Created shared httpx client for web fetching (max_connections=30, http1=True)")
    return _http_client


async def fetch_url_async(url: str, timeout: int = 20, max_retries: int = 2) -> str | None:
    """Fetch URL content asynchronously with timeout and retry.
    
    Uses a shared client with connection pooling for efficiency.
    
    Args:
        url: URL to fetch
        timeout: Timeout in seconds (default 20s)
        max_retries: Maximum retry attempts (default 2)
        
    Returns:
        HTML content or None if failed
    """
    for attempt in range(max_retries + 1):
        try:
            # Random jitter before request to spread out bursts
            await asyncio.sleep(random.uniform(0, 0.2))
            
            client = await get_http_client()
            response = await client.get(url, timeout=timeout)
            response.raise_for_status()
            return response.text
                
        except httpx.HTTPStatusError as e:
            # HTTP errors (403, 404, 500, etc.) - don't retry
            if e.response.status_code not in [403, 401]:
                _logger.debug(f"httpx HTTP {e.response.status_code} for {url[:80]}")
            return None
        except (httpx.TimeoutException, httpx.ConnectError, OSError) as e:
            if attempt < max_retries:
                # Exponential backoff with jitter
                base_wait = 2 ** attempt
                jitter = random.uniform(0, base_wait * 0.5)
                wait_time = base_wait + jitter
                _logger.debug(f"httpx connection error, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries}): {type(e).__name__}")
                await asyncio.sleep(wait_time)
                continue
            _logger.debug(f"httpx error for {url[:80]} after {max_retries} retries: {type(e).__name__}")
            return None
        except Exception as e:
            # Only log unexpected errors (skip common SSL issues)
            error_str = str(e)
            if error_str and "SSL" not in error_str and "certificate" not in error_str:
                _logger.debug(f"httpx error for {url[:80]}: {type(e).__name__}")
            return None
    return None


async def close_shared_client() -> None:
    """Close the shared HTTP client. Call at program shutdown."""
    global _http_client
    if _http_client is not None and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None


async def extract_with_trafilatura(html: str) -> str:
    """Extract text from HTML using trafilatura.
    
    Args:
        html: HTML content
        
    Returns:
        Extracted text or empty string if failed
    """
    try:
        # Run trafilatura.extract in thread pool since it's CPU-bound
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, trafilatura.extract, html)
        return result or ""
    except Exception as e:
        _logger.debug(f"trafilatura.extract failed: {e}")
        return ""


class BrowserFetcher:
    """Fetch webpages using Selenium Chromium browser and extract clean content."""
    
    def __init__(
        self,
        static_fetch_semaphore: asyncio.Semaphore | None = None,
        dynamic_fetch_semaphore: asyncio.Semaphore | None = None,
        headless: bool = True,
        page_load_timeout: int = 60,
        wait_after_load: float = 3.0,
    ):
        """Initialize browser fetcher.
        
        Args:
            static_fetch_semaphore: Optional semaphore to limit concurrent static fetches
            dynamic_fetch_semaphore: Optional semaphore to limit concurrent dynamic fetches
            headless: Run browser in headless mode
            page_load_timeout: Maximum time to wait for page load (seconds)
            wait_after_load: Time to wait after page loads for dynamic content (seconds)
        """
        self.headless = headless
        self.page_load_timeout = page_load_timeout
        self.wait_after_load = wait_after_load
    
    def _create_driver(self) -> webdriver.Chrome:
        """Create a new Chrome WebDriver instance.
        
        Returns:
            Configured Chrome WebDriver
        """
        chrome_options = Options()
        
        if self.headless:
            chrome_options.add_argument('--headless=new')
        
        chrome_options.add_argument('--window-size=1920,1080')
        
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(self.page_load_timeout)
        
        return driver
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL by removing tracking parameters.
        
        Args:
            url: Original URL
            
        Returns:
            Normalized URL with tracking params removed
        """
        from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
        
        try:
            parsed = urlparse(url)
            
            # List of tracking parameters to remove
            tracking_params = {
                'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
                'fbclid', 'gclid', 'msclkid', 'mc_cid', 'mc_eid',
                '_ga', '_gid', '_hsenc', '_hsmi',
                'ref', 'referrer', 'source'
            }
            
            # Parse query parameters
            if parsed.query:
                query_params = parse_qs(parsed.query, keep_blank_values=True)
                
                # Remove tracking parameters (case-insensitive)
                filtered_params = {
                    k: v for k, v in query_params.items()
                    if k.lower() not in tracking_params
                }
                
                # Rebuild query string
                new_query = urlencode(filtered_params, doseq=True) if filtered_params else ''
                clean_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, ''))
            else:
                # No query params, just remove fragment
                clean_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, '', ''))
                
        except:
            clean_url = url
        
        return clean_url
    
    async def _fetch_html_async(self, url: str, max_retries: int = 2, force_selenium: bool = False) -> tuple[str, str | None]:
        """Fetch HTML from URL, trying httpx+trafilatura first then Selenium as fallback.
        
        Args:
            url: URL to fetch
            max_retries: Maximum number of retry attempts for Selenium fallback
            force_selenium: If True, skip httpx/trafilatura and use Selenium directly
            
        Returns:
            Tuple of (html_content, error_message)
        """
        # Normalize URL (strip tracking params)
        clean_url = self._normalize_url(url)
        
        # Try httpx first (fast and lightweight) unless forced to use Selenium
        trafilatura_error = None
        if not force_selenium:
            try:
                # Fetch HTML - timeout is handled inside fetch_url_async
                html = await fetch_url_async(clean_url, timeout=20)
                
                if not html:
                    trafilatura_error = "httpx fetch returned no content"
                    raise Exception(trafilatura_error)
                                
                # Check for meaningful content
                if len(html) < 500:
                    trafilatura_error = f"httpx returned little content ({len(html)} bytes)"
                    raise Exception(trafilatura_error)
                
                return html, None
                
            except Exception as e:
                trafilatura_error = str(e)
        else:
            trafilatura_error = "Forced Selenium mode"
        
        # httpx/trafilatura failed, try Selenium as fallback
        # Only log if it's not a common paywall/auth error
        # if "403" not in str(trafilatura_error) and "401" not in str(trafilatura_error):
        #     print(f"  [DEBUG] httpx failed ({trafilatura_error}), trying Selenium for {clean_url[:80]}")
            
        # last_error = None
        # for attempt in range(max_retries + 1):
        #     driver = None
        #     try:
        #         # Run Selenium in thread pool since it's blocking
        #         loop = asyncio.get_event_loop()
                
        #         def _selenium_fetch():
        #             driver = self._create_driver()
        #             try:
        #                 # Navigate to URL
        #                 driver.get(clean_url)
                        
        #                 # Wait for dynamic content to load
        #                 if self.wait_after_load > 0:
        #                     time.sleep(self.wait_after_load + (random.random() - 0.5))
                        
        #                 # Get the fully rendered HTML
        #                 html = driver.page_source
                        
        #                 # Check if we got meaningful content
        #                 if len(html) < 500:
        #                     preview = html[:500] if html else "(empty)"
        #                     raise Exception(f"Page returned very little content ({len(html)} bytes). Content preview: {preview}")
                        
        #                 return html
        #             finally:
        #                 try:
        #                     driver.quit()
        #                 except:
        #                     pass
                                
        #         # Run with timeout (page_load_timeout + processing time)
        #         html = await asyncio.wait_for(
        #             loop.run_in_executor(None, _selenium_fetch),
        #             timeout=self.page_load_timeout + 10
        #         )
                
        #         return html, None
                
        #     except asyncio.TimeoutError:
        #         last_error = Exception(f"Selenium timeout (>{self.page_load_timeout + 10}s)")
        #         print(f"  [DEBUG] Selenium timeout for {clean_url[:80]}")
        #     except Exception as e:
        #         last_error = e
        #         if attempt < max_retries:
        #             # Exponential backoff: 2, 4 seconds
        #             wait_time = 2 ** (attempt + 1)
        #             print(f"  [DEBUG] Selenium attempt {attempt+1} failed, retrying in {wait_time}s...")
        #             await asyncio.sleep(wait_time)
        #             continue
        
        # # Both methods failed
        # print(f"  ✗ Both httpx and Selenium failed for {clean_url[:80]}")
        # error_msg = f"httpx: {trafilatura_error}; Selenium: {str(last_error) if last_error else 'Unknown error'}"
        # return "", error_msg

        return "", trafilatura_error
    
    async def fetch_html(self, url: str, force_selenium: bool = False) -> tuple[str, str | None]:
        """Fetch raw HTML from a URL.
        
        Args:
            url: URL to fetch
            force_selenium: If True, skip httpx/trafilatura and use Selenium directly
            
        Returns:
            Tuple of (html_content, error_message)
        """
        # Call the async fetch method directly (no executor needed anymore)
        return await self._fetch_html_async(url, max_retries=2, force_selenium=force_selenium)


def extract_pdf_links(html_content: str, base_url: str | None = None) -> List[str]:
    """Extract all PDF links from HTML content.
    
    Finds all <a> tags with href attributes pointing to PDF files.
    Converts relative URLs to absolute URLs if base_url is provided.
    
    Args:
        html_content: HTML content to parse
        base_url: Base URL for resolving relative links (optional)
        
    Returns:
        List of absolute URLs pointing to PDF files
    """
    if not html_content:
        return []
    
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        pdf_links = []
        
        # Find all <a> tags with href attribute
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # Check if link points to a PDF
            href_lower = href.lower()
            is_pdf = (
                href_lower.endswith('.pdf') or 
                '/pdf/' in href_lower or 
                '.pdf?' in href_lower or
                href_lower.endswith('.pdf/')
            )
            
            if is_pdf:
                # Convert relative URLs to absolute if base_url provided
                if base_url:
                    absolute_url = urljoin(base_url, href)
                else:
                    absolute_url = href
                
                # Only add if it's a valid URL (starts with http/https or is relative)
                parsed = urlparse(absolute_url)
                if parsed.scheme in ['http', 'https', ''] or absolute_url.startswith('/'):
                    if absolute_url not in pdf_links:  # Avoid duplicates
                        pdf_links.append(absolute_url)
        
        return pdf_links
    
    except Exception as e:
        _logger.warning(f"Error extracting PDF links: {e}")
        return []

