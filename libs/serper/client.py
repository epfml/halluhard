"""Serper API client for web search with LLM-guided query planning."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
from urllib.parse import urlparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import httpx

from libs.types import SamplerBase, UsageStats
from libs.json_utils import extract_json_from_response, sanitize_json_string

MAX_URLS_TO_FETCH = 2
MAX_PDFS_TO_FETCH = 1


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SerperUsageStats(UsageStats):
    """Usage stats with Serper-specific metrics."""
    total_serper_requests: int = 0


@dataclass
class SearchResult:
    """Structured search result."""
    raw_results: List[Dict[str, Any]] = field(default_factory=list)
    formatted_text: str = ""
    queries: List[str] = field(default_factory=list)
    urls_with_positions: List[Tuple[int, str]] = field(default_factory=list)
    top_urls: List[str] = field(default_factory=list)
    usage: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Prompt Templates
# =============================================================================

_STATEMENT_PLACEHOLDER = '[STATEMENT]'
_KNOWLEDGE_PLACEHOLDER = '[KNOWLEDGE]'
_PREVIOUS_QUERIES_PLACEHOLDER = '[PREVIOUS_QUERIES]'

# Default planner prompt - used for all search steps unless strategy provides custom prompt
_NEXT_SEARCH_PROMPT = f"""Instructions:

1. You are given a **STATEMENT** and some **KNOWLEDGE** gathered from **PREVIOUS QUERIES**.
2. The STATEMENT references a **source** (paper, book, report, dataset, author, etc.). Your goal is to **identify that source** and verify whether the STATEMENT accurately reflects it.
3. Analyze the KNOWLEDGE collected so far (indexed as [Step.Item]).
4. Decide if you have enough information to verify the statement AND identify the source.
   - If YES:
     - Set "continue_searching" to false.
     - Select the **two most relevant URLs** from the KNOWLEDGE that best support your verification (prioritize sources that are freely accessible and not protected by aggressive anti-bot measures, such as arXiv rather than researchgate or harvard).
     - **IMPORTANT**: You MUST refer to them by their index (e.g. "[1.1]", "[2.3]"). Do NOT output the full URL string, to avoid transcription errors.
   - If NO:
     - Set "continue_searching" to true.
     - Generate a new Google Search query to fill the gaps.
     - The query must be clearly **different** from PREVIOUS QUERIES.

STRICT RULES FOR THE QUERY:
- Do **NOT** copy any numbers, equations, variables, mathematical symbols, quoted passages, or units from the STATEMENT or KNOWLEDGE.
- Do **NOT** include any content in quotation marks.
- Do **NOT** use `site:` filters.
- Use only **broad textual identifiers**: reference title, author names, year, topic keywords.

**SPECIAL INSTRUCTIONS IF URL IS PROVIDED IN STATEMENT:**
- If the STATEMENT contains a URL (http:// or https://), you MUST prioritize searching for content from that URL.
- Extract the domain name from the URL and include it in your search query (e.g., if URL is "https://arxiv.org/abs/2023.12345", include "arxiv" in your query).
- Include the URL identifier (e.g., paper ID, document ID) if present in the URL.
- Your first search query should specifically target the URL/domain mentioned in the STATEMENT.
- Example: If STATEMENT contains "https://arxiv.org/abs/2023.12345", generate a query like "arxiv 2023.12345" or "arxiv abs 2023.12345".

OUTPUT FORMAT:
You must output a valid JSON object in the following format:

```json
{{
  "reasoning": "Brief explanation of your decision...",
  "continue_searching": boolean,
  "search_query": "Your query string here (required if continue_searching is true, else null)",
  "relevant_urls": ["[1.1]", "[2.3]"] (list of indices as strings, required if continue_searching is false)
}}
```

PREVIOUS QUERIES:
{_PREVIOUS_QUERIES_PLACEHOLDER}

KNOWLEDGE:
{_KNOWLEDGE_PLACEHOLDER}

STATEMENT:
{_STATEMENT_PLACEHOLDER}
"""

_FINAL_SELECTION_PROMPT = """You have completed the search process or reached the limit.
Based on the collected KNOWLEDGE below, identify the **two most relevant URLs** that verify the STATEMENT.
Prioritize sources that are freely accessible and not protected by aggressive anti-bot measures, such as arXiv rather than researchgate or harvard.

STATEMENT:
{statement}

KNOWLEDGE:
{knowledge}

Output ONLY a JSON object:
{{
  "relevant_urls": ["[1.1]", "[2.3]"]
}}
"""


# =============================================================================
# Serper Client
# =============================================================================

class SerperSearchClient:
    """Async client for Serper API web search.
    
    Features:
    - Reusable aiohttp session for connection pooling
    - Automatic retry with exponential backoff
    - LLM-guided multi-step search planning
    - Structured logging
    
    Usage:
        async with SerperSearchClient() as client:
            result = await client.search("python asyncio tutorial")
            
        # Or manual lifecycle:
        client = SerperSearchClient()
        await client.start()
        try:
            result = await client.search("query")
        finally:
            await client.close()
    """
    
    BASE_URL = "https://google.serper.dev/search"
    # Timeout configuration for httpx
    DEFAULT_TIMEOUT = httpx.Timeout(90.0, connect=30.0)
    
    # Default connection limits - conservative to prevent connection exhaustion
    DEFAULT_CONN_LIMIT = 50  # Total connections
    
    def __init__(
        self,
        api_key: str | None = None,
        logger: logging.Logger | None = None,
        log_file: str | None = None,
        conn_limit: int | None = None,
    ):
        """Initialize Serper client.
        
        Args:
            api_key: Serper API key. Falls back to SERPER_API_KEY env var.
            logger: Logger instance. Falls back to module logger if None.
            log_file: Optional file path to log planner interactions.
            conn_limit: Total connection limit (default: 50)
        """
        self.api_key = api_key or os.environ.get("SERPER_API_KEY")
        if not self.api_key:
            raise ValueError("SERPER_API_KEY environment variable must be set")
        
        self.logger = logger or logging.getLogger(__name__)
        self.log_file = log_file
        self.conn_limit = conn_limit or self.DEFAULT_CONN_LIMIT
        
        # Session management (using httpx)
        self._client: httpx.AsyncClient | None = None
        self._owns_client: bool = True
    
    # -------------------------------------------------------------------------
    # Client Lifecycle
    # -------------------------------------------------------------------------
    
    async def start(self) -> None:
        """Start the httpx client."""
        if self._client is None or self._client.is_closed:
            # Create httpx client with bounded connection limits
            # Force HTTP/1.1 to avoid HTTP/2 multiplexing issues on Windows
            self._client = httpx.AsyncClient(
                timeout=self.DEFAULT_TIMEOUT,
                limits=httpx.Limits(
                    max_connections=self.conn_limit,
                    max_keepalive_connections=self.conn_limit // 2,
                ),
                headers={
                    "X-API-KEY": self.api_key,
                    "Content-Type": "application/json",
                },
                http1=True,  # Force HTTP/1.1 - avoids TLS/HTTP2 issues on Windows
                http2=False,
            )
            self._owns_client = True
            self.logger.debug(f"Serper httpx client started (limit={self.conn_limit}, http1=True)")
    
    async def close(self) -> None:
        """Close the httpx client."""
        if self._client and not self._client.is_closed and self._owns_client:
            client_id = id(self._client)
            await self._client.aclose()
            self._client = None
            self.logger.debug(f"Serper httpx client closed (id={client_id})")
    
    async def __aenter__(self) -> "SerperSearchClient":
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
    
    def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure client is available, raise if not."""
        if self._client is None or self._client.is_closed:
            raise RuntimeError(
                "Client not started. Use 'async with client:' or call 'await client.start()'"
            )
        return self._client
    
    # -------------------------------------------------------------------------
    # Core Search API
    # -------------------------------------------------------------------------
    
    async def search(
        self,
        query: str,
        num_results: int = 5,
        max_retries: int = 3,
        context: str | None = None,
    ) -> Tuple[Dict[str, Any], int]:
        """Perform a web search using Serper API.
        
        Args:
            query: Search query string
            num_results: Number of results to return
            max_retries: Maximum retries for rate limit/connection errors
            
        Returns:
            Tuple of (search results dict, total requests made)
            
        Raises:
            RuntimeError: If client not started
            Exception: If all retries fail
        """
        client = self._ensure_client()
        payload = {"q": query, "num": num_results}
        total_requests = 0
        
        prefix = f"[{context}] " if context else ""
        self.logger.debug(
            f"{prefix}Serper search request: query={query!r}, num_results={num_results}"
        )
        
        for attempt in range(max_retries + 1):
            try:
                total_requests += 1
                self.logger.debug(f"{prefix}Serper API call attempt {attempt + 1}/{max_retries + 1}")
                
                response = await client.post(self.BASE_URL, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    num_organic = len(result.get("organic", []))
                    self.logger.debug(
                        f"{prefix}Serper search success: {num_organic} organic results, "
                        f"total_requests={total_requests}"
                    )
                    return result, total_requests
                
                if response.status_code == 429:  # Rate limit
                    if attempt < max_retries:
                        wait_time = 2 ** (attempt + 1)
                        self.logger.debug(
                            f"{prefix}Serper rate limit (429), retrying in {wait_time}s "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    
                    raise Exception(f"Serper rate limit exceeded: {response.text}")
                
                raise Exception(f"Serper API error {response.status_code}: {response.text}")
                    
            except (httpx.ConnectError, httpx.TimeoutException, httpx.ConnectTimeout, 
                    ConnectionResetError, OSError) as e:
                if attempt < max_retries:
                    # Exponential backoff with jitter to prevent thundering herd
                    base_wait = 2 ** attempt  # 1, 2, 4 seconds
                    jitter = random.uniform(0, base_wait * 0.5)  # Up to 50% jitter
                    wait_time = base_wait + jitter
                    self.logger.debug(
                        f"{prefix}Serper connection error, retrying in {wait_time:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}"
                    )
                    # self.logger.debug("Serper connection error traceback:", exc_info=True)
                    await asyncio.sleep(wait_time)
                    continue
                
                # self.logger.debug("Serper final failure traceback:", exc_info=True)
                raise Exception(f"Serper connection failed after {max_retries} retries: {e}")
        
        raise Exception("Serper search failed after all retries")
    
    # -------------------------------------------------------------------------
    # Multi-Step Verification Search
    # -------------------------------------------------------------------------
    
    async def perform_verification_search(
        self,
        claim_text: str,
        sampler: SamplerBase,
        max_searches: int = 5,
        num_results: int = 5,
        search_semaphore: asyncio.Semaphore | None = None,
        context: str | None = None,
        custom_planner_prompt: str | None = None,
    ) -> Tuple[List[Dict], str, List[str], List[Tuple], Dict, List[str]]:
        """Perform LLM-guided multi-step search to verify a claim.
        
        The LLM iteratively decides what to search for and when to stop,
        selecting the most relevant URLs for verification.
        
        Args:
            claim_text: The claim to verify
            sampler: LLM sampler for query planning
            max_searches: Maximum search iterations
            num_results: Results per search
            search_semaphore: Optional semaphore for concurrency control
            context: Optional context string for logging
            custom_planner_prompt: Optional custom prompt for the LLM planner.
                If provided, overrides the default _NEXT_SEARCH_PROMPT.
                Should contain placeholders: [STATEMENT], [KNOWLEDGE], [PREVIOUS_QUERIES]
            
        Returns:
            Tuple of:
                - raw_results: List of raw search results per step
                - formatted_text: Human-readable combined results
                - queries: List of queries executed
                - urls_with_positions: List of (position, url) tuples
                - usage: Token usage statistics
                - top_urls: Top URLs selected by LLM (max 2)
        """
        raw_results: List[Dict[str, Any]] = []
        queries: List[str] = []
        final_urls: List[str] = []
        usage = SerperUsageStats()
        
        prefix = f"[{context}] " if context else ""
        self.logger.debug(
            f"{prefix}Starting verification search: claim_text={claim_text[:100]!r}..., "
            f"max_searches={max_searches}, num_results={num_results}"
        )
        
        for step in range(max_searches):
            self.logger.debug(f"{prefix}Verification search step {step + 1}/{max_searches}")
            
            # Ask LLM to plan this step
            # Uses strategy's custom prompt or default _NEXT_SEARCH_PROMPT
            # LLM can decide to stop at any step (including step 0)
            decision = await self._plan_next_step(
                claim_text=claim_text,
                raw_results=raw_results,
                queries=queries,
                sampler=sampler,
                step=step,
                usage=usage,
                custom_planner_prompt=custom_planner_prompt,
            )
            
            if decision is None:
                break
            
            # LLM can decide to stop at any step (even step 0 if it determines no search needed)
            if not decision.get("continue_searching", True):
                # LLM decided to stop - extract selected URLs if any results exist
                url_refs = decision.get("relevant_urls", [])
                final_urls = self._resolve_url_references(url_refs, raw_results)
                self.logger.debug(
                    f"{prefix}LLM decided to stop searching at step {step + 1}. Selected URLs: {url_refs} -> {final_urls}"
                )
                break
            
            query = decision.get("search_query")
            if not query:
                self.logger.debug("No search query provided, stopping")
                break
            
            self.logger.debug(f"{prefix}LLM generated query: {query!r}")
            queries.append(query)
            
            # Execute search
            try:
                await asyncio.sleep(random.uniform(0, 0.3))  # 0-300ms jitter
                
                if search_semaphore:
                    async with search_semaphore:
                        results, requests_made = await self.search(query, num_results, context=context)
                else:
                    results, requests_made = await self.search(query, num_results, context=context)
                
                usage.total_serper_requests += requests_made
                raw_results.append(results)
                
            except Exception as e:
                self.logger.debug(
                    f"{prefix}Search step {step + 1} failed for query={query!r}: "
                    f"{type(e).__name__}: {e}"
                )
                break
        
        # Final URL selection if not already done
        if not final_urls and raw_results:
            final_urls = await self._select_final_urls(
                claim_text=claim_text,
                raw_results=raw_results,
                sampler=sampler,
                usage=usage,
            )
        
        # Build return values
        urls_with_positions = self._extract_urls_with_positions(raw_results)
        formatted_text = self._format_all_results(raw_results)
        
        self.logger.debug(
            f"{prefix}Verification search complete: {len(queries)} queries, "
            f"{len(raw_results)} search steps, {len(final_urls)} final URLs"
        )

        
        return (
            raw_results,
            formatted_text,
            queries,
            urls_with_positions,
            usage.to_dict(),
            final_urls[:MAX_URLS_TO_FETCH],
        )
    
    # -------------------------------------------------------------------------
    # Planning Helpers
    # -------------------------------------------------------------------------
    
    def _extract_urls_from_text(self, text: str) -> List[str]:
        """Extract URLs from text."""
        # Pattern to match http:// or https:// URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text)
        # Filter to only valid URLs
        valid_urls = []
        for url in urls:
            try:
                parsed = urlparse(url)
                if parsed.scheme in ['http', 'https'] and parsed.netloc:
                    valid_urls.append(url)
            except:
                pass
        return valid_urls
    
    def _enhance_prompt_for_url(self, prompt: str, claim_text: str) -> str:
        """Enhance prompt with URL-specific instructions if URL is detected."""
        urls = self._extract_urls_from_text(claim_text)
        if not urls:
            return prompt
        
        # Extract domain and identifier from first URL
        url = urls[0]
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.replace('www.', '')
            path_parts = [p for p in parsed.path.split('/') if p]
            identifier = path_parts[-1] if path_parts else None
        except:
            domain = None
            identifier = None
        
        # Add URL-specific instructions
        domain_part = f'"{domain}"' if domain else "the domain"
        identifier_part = f'"{identifier}"' if identifier else "the identifier"
        
        url_instructions = f"""

**⚠️ URL DETECTED - PRIORITY SEARCH REQUIRED:**
A URL was found in the STATEMENT above: {url}
- Domain: {domain if domain else 'unknown'}
- Identifier: {identifier if identifier else 'none'}

**CRITICAL SEARCH REQUIREMENTS:**
1. Your search query MUST prioritize finding content from this specific URL.
2. Include {domain_part} in your search query.
3. If an identifier is present, include {identifier_part} in your query.
4. Your first query should specifically target this URL to verify the claim.
5. Example queries: "{domain} {identifier}" or "{domain} {identifier} [relevant keywords from STATEMENT]"

Do NOT skip searching for this URL - it is the primary source to verify.
"""
        
        # Insert URL instructions before OUTPUT FORMAT section (after STRICT RULES)
        if "OUTPUT FORMAT:" in prompt:
            prompt = prompt.replace("OUTPUT FORMAT:", url_instructions + "\nOUTPUT FORMAT:")
        else:
            # If OUTPUT FORMAT not found, append before STATEMENT section
            prompt = prompt.replace("STATEMENT:", url_instructions + "\nSTATEMENT:")
        
        return prompt
    
    async def _plan_next_step(
        self,
        claim_text: str,
        raw_results: List[Dict[str, Any]],
        queries: List[str],
        sampler: SamplerBase,
        step: int,
        usage: SerperUsageStats,
        custom_planner_prompt: str | None = None,
    ) -> Dict[str, Any] | None:
        """Ask LLM to plan the next search step.
        
        Uses custom_planner_prompt if provided by strategy, otherwise _NEXT_SEARCH_PROMPT.
        On step 0, KNOWLEDGE will be "N/A" since no results yet - LLM should generate initial query.
        LLM can decide to stop at any step if it determines no search is needed.
        """
        knowledge = self._format_knowledge_with_indices(raw_results)
        past_queries = self._format_past_queries(queries)
        
        # Use custom prompt if provided, otherwise default
        base_prompt = custom_planner_prompt or _NEXT_SEARCH_PROMPT
        
        prompt = (
            base_prompt
            .replace(_STATEMENT_PLACEHOLDER, claim_text)
            .replace(_KNOWLEDGE_PLACEHOLDER, knowledge)
            .replace(_PREVIOUS_QUERIES_PLACEHOLDER, past_queries)
        )
        
        # Enhance prompt if URL is detected in claim_text
        prompt = self._enhance_prompt_for_url(prompt, claim_text)
        
        try:
            response = await sampler([{"role": "user", "content": prompt}])
            response_text = response.response_text.strip()
            
            # Log interaction
            self._log_planner_interaction(step, claim_text, prompt, response_text)
            
            # Track usage
            if response.token_usage:
                usage.accumulate(response.token_usage)
            
            # Parse response
            decision = self._parse_planner_response(response_text, step)
            
            return decision
            
        except Exception as e:
            self.logger.debug(f"Planning step {step + 1} failed: {e}")
            return None
    
    async def _select_final_urls(
        self,
        claim_text: str,
        raw_results: List[Dict[str, Any]],
        sampler: SamplerBase,
        usage: SerperUsageStats,
    ) -> List[str]:
        """Final URL selection when search loop completed without selection."""
        prompt = _FINAL_SELECTION_PROMPT.format(
            statement=claim_text,
            knowledge=self._format_knowledge_with_indices(raw_results),
        )
        
        try:
            response = await sampler([{"role": "user", "content": prompt}])
            
            if response.token_usage:
                usage.accumulate(response.token_usage)
            
            json_text = extract_json_from_response(response.response_text.strip())
            decision = json.loads(sanitize_json_string(json_text))
            url_refs = decision.get("relevant_urls", [])
            
            return self._resolve_url_references(url_refs, raw_results)
            
        except Exception as e:
            self.logger.debug(f"Final URL selection failed: {e}")
            return []
    
    def _parse_planner_response(self, response_text: str, step: int) -> Dict[str, Any]:
        """Parse JSON response from planner LLM."""
        try:
            json_text = extract_json_from_response(response_text)
            json_text = sanitize_json_string(json_text)
            return json.loads(json_text)
            
        except Exception as e:
            self.logger.debug(f"Failed to parse planner JSON (step {step + 1}): {e}")
            
            # Fallback: try to extract query from markdown code block
            query = self._extract_query_fallback(response_text)
            if query:
                return {"continue_searching": True, "search_query": query}
            
            return {"continue_searching": False}
    
    def _extract_query_fallback(self, response: str) -> str | None:
        """Extract query from markdown code block as fallback."""
        match = re.search(r"```(?:\w+)?\s*(.*?)```", response, re.DOTALL)
        return match.group(1).strip() if match else None
    
    # -------------------------------------------------------------------------
    # URL Resolution
    # -------------------------------------------------------------------------
    
    def _resolve_url_references(
        self,
        url_refs: List[str],
        raw_results: List[Dict[str, Any]],
    ) -> List[str]:
        """Resolve URL references (e.g., [1.1]) to actual URLs."""
        # Build set of all valid URLs for validation
        valid_urls = {
            res.get("link")
            for step in raw_results
            for res in step.get("organic", [])
            if res.get("link")
        }
        
        resolved = []
        for ref in url_refs:
            ref = str(ref).strip()
            
            # Try to parse as index reference [step.item]
            match = re.match(r"\[?(\d+)\.(\d+)\]?", ref)
            if match:
                step_idx = int(match.group(1)) - 1
                item_idx = int(match.group(2)) - 1
                
                if 0 <= step_idx < len(raw_results):
                    organic = raw_results[step_idx].get("organic", [])
                    if 0 <= item_idx < len(organic):
                        link = organic[item_idx].get("link")
                        if link:
                            resolved.append(link)
                            continue
            
            # Check if it's a direct URL we know about
            if ref in valid_urls:
                resolved.append(ref)
            elif ref.startswith("http"):
                # Unknown URL but looks valid - keep it
                resolved.append(ref)
        
        return resolved
    
    # -------------------------------------------------------------------------
    # Formatting Helpers
    # -------------------------------------------------------------------------
    
    def _format_past_queries(self, queries: List[str]) -> str:
        """Format past queries for prompt."""
        if not queries:
            return "N/A"
        return "- " + "\n- ".join(queries)
    
    def _format_knowledge_with_indices(self, raw_results: List[Dict[str, Any]]) -> str:
        """Format search results with indices for LLM selection."""
        if not raw_results:
            return "N/A"
        
        sections = []
        for i, step_results in enumerate(raw_results, 1):
            section_parts = [f"## Search Step {i}"]
            
            # Answer Box
            if step_results.get("answerBox"):
                ab = step_results["answerBox"]
                content = ab.get("snippet") or ab.get("answer") or str(ab)
                section_parts.append(f"[Step {i} AnswerBox] {content}")
            
            # Organic results
            for j, result in enumerate(step_results.get("organic", []), 1):
                title = result.get("title", "No Title")
                link = result.get("link", "No Link")
                snippet = result.get("snippet", "No Snippet")
                section_parts.append(
                    f"[{i}.{j}] Title: {title}\nLink: {link}\nSnippet: {snippet}"
                )
            
            sections.append("\n\n".join(section_parts))
        
        return "\n\n".join(sections)
    
    def _format_all_results(self, raw_results: List[Dict[str, Any]]) -> str:
        """Format all results into human-readable text."""
        if not raw_results:
            return "No search results found."
        
        formatted_steps = [
            self._format_single_result(result) for result in raw_results
        ]
        return "\n\n".join(formatted_steps)
    
    def _format_single_result(self, results: Dict[str, Any]) -> str:
        """Format a single search result into readable text."""
        snippets = []
        
        # Answer Box
        if results.get("answerBox"):
            ab = results["answerBox"]
            for key in ("answer", "snippet", "snippetHighlighted"):
                value = ab.get(key)
                if value:
                    if isinstance(value, list):
                        snippets.append(" ".join(str(x) for x in value))
                    else:
                        snippets.append(str(value).replace("\n", " "))
        
        # Knowledge Graph
        if results.get("knowledgeGraph"):
            kg = results["knowledgeGraph"]
            title = kg.get("title", "")
            
            if kg.get("type"):
                snippets.append(f"{title}: {kg['type']}.")
            if kg.get("description"):
                snippets.append(str(kg["description"]))
            
            for attr, value in kg.get("attributes", {}).items():
                snippets.append(f"{title} {attr}: {value}.")
        
        # Organic Results
        for result in results.get("organic", []):
            for key in ("title", "date", "link", "snippet"):
                if result.get(key):
                    snippets.append(str(result[key]))
            
            for attr, value in result.get("attributes", {}).items():
                snippets.append(f"{attr}: {value}.")
        
        return " -- ".join(snippets) if snippets else "No results"
    
    def _extract_urls_with_positions(
        self,
        raw_results: List[Dict[str, Any]],
    ) -> List[Tuple[int, str]]:
        """Extract all URLs with their positions from results."""
        urls = []
        for step_results in raw_results:
            for result in step_results.get("organic", []):
                position = result.get("position", 999)
                link = result.get("link")
                if link:
                    urls.append((position, link))
        return urls
    
    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    
    def _log_planner_interaction(
        self,
        step: int,
        claim_text: str,
        prompt: str,
        response: str,
    ) -> None:
        """Log planner interaction to file if configured."""
        if not self.log_file:
            return
        
        try:
            log_entry = (
                f"{'=' * 80}\n"
                f"STEP: {step + 1}\n"
                f"CLAIM: {claim_text}\n\n"
                f"--- INPUT PROMPT ---\n{prompt}\n\n"
                f"--- PLANNER RESPONSE ---\n{response}\n"
                f"{'=' * 80}\n\n"
            )
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_entry)
                
        except Exception as e:
            self.logger.warning(f"Failed to write to log file: {e}")
