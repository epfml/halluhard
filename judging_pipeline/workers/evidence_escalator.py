"""Targeted evidence retrieval for claims the judge could not verify.

When a judge reports that a specific field (a pinpoint page, a reporter volume)
is neither confirmed nor contradicted by the evidence it was given, this runs the
query the judge asked for and returns what it finds, so the claim can be judged
again on better evidence instead of being scored on retrieval luck.
"""

from __future__ import annotations

import re

from libs.browser_fetcher import BrowserFetcher
from libs.html_cleaner import HtmlCleaner
from libs.information_extraction import check_if_url_is_pdf
from libs.serper.client import SerperSearchClient

from ..models.work_items import ClaimItem
from ..logging_config import get_logger

logger = get_logger()

MAX_PAGES_TO_FETCH = 1
MAX_CONTENT_WORDS = 1200
MAX_QUERY_WORDS = 14

# Judges tend to answer with a sentence ("Search for the official opinion text
# showing ...") rather than a query; search engines return nothing for that.
_INSTRUCTION_PREFIX = re.compile(
    r"^\s*(please\s+)?(do\s+a\s+|run\s+a\s+|perform\s+a\s+)?(web\s+)?(search|look ?up|find|verify|confirm|check)"
    r"[\s:]*(for|that|whether|the)?[\s:]*",
    re.I,
)


def _to_query(text: str) -> str:
    """Reduce a judge's request to something a search engine can use."""
    text = (text or "").strip()
    if not text:
        return ""
    # A quoted example is usually the actual query the judge had in mind.
    quoted = re.findall(r'"([^"]{8,})"', text)
    if quoted:
        text = max(quoted, key=len)
    text = _INSTRUCTION_PREFIX.sub("", text)
    text = re.split(r"\b(e\.g\.|such as|for example|i\.e\.)\b", text, maxsplit=1)[0]
    text = text.replace('"', " ").replace(",", " ")
    return " ".join(text.split()[:MAX_QUERY_WORDS]).strip(" .;:-")


class EvidenceEscalator:
    """Callable that answers a judge's request for more evidence."""

    def __init__(
        self,
        serper_client: SerperSearchClient,
        num_results: int = 10,
        fetch_pages: bool = True,
    ):
        """Initialize the escalator.

        Args:
            serper_client: Shared Serper client (reuses the pipeline's connections)
            num_results: Results to request for the escalation query. Wider than the
                pipeline default of 5, since the first pass already missed this field.
            fetch_pages: Whether to fetch the top hit's full text, not just snippets.
                Pinpoints usually need page text, so this defaults on.
        """
        self.serper_client = serper_client
        self._client_started = False
        self.num_results = num_results
        self.fetch_pages = fetch_pages
        self._browser_fetcher = BrowserFetcher() if fetch_pages else None
        self._html_cleaner = HtmlCleaner() if fetch_pages else None

    async def close(self) -> None:
        """Close the Serper client if this escalator started it."""
        if self._client_started:
            await self.serper_client.close()
            self._client_started = False
        self._browser_fetcher = None

    async def __call__(self, claim: ClaimItem, requested: str, context: str = "") -> str:
        """Return extra evidence for `requested`, or "" if nothing was retrieved."""
        citation = " ".join(
            (claim.data.get("reference_name") or claim.data.get("claimed_url") or "").split()[:MAX_QUERY_WORDS]
        ).strip()
        # Try the judge's (sanitized) request first, then the citation itself.
        candidates = [q for q in (_to_query(requested), citation) if q]
        if not candidates:
            return ""

        organic = []
        for query in candidates:
            try:
                # search() requires a started client; the worker lifecycle that normally
                # starts one does not own this client, so start it on first use.
                if not self._client_started:
                    await self.serper_client.start()
                    self._client_started = True
                results, _ = await self.serper_client.search(
                    query, self.num_results, context=f"escalation:{context}"
                )
            except Exception as e:
                logger.debug(f"Escalation search failed for {query!r}: {type(e).__name__}: {e}")
                continue
            organic = results.get("organic", []) if isinstance(results, dict) else []
            if organic:
                logger.debug(f"Escalation query {query!r} returned {len(organic)} results")
                break
        if not organic:
            return ""

        parts = [
            f"{i}. {r.get('title', '')}\n   {r.get('link', '')}\n   {r.get('snippet', '')}"
            for i, r in enumerate(organic, 1)
        ]

        if self.fetch_pages:
            for result in organic[:MAX_PAGES_TO_FETCH]:
                url = result.get("link", "")
                if not url or await check_if_url_is_pdf(url):
                    continue
                try:
                    html, error = await self._browser_fetcher.fetch_html(url, force_selenium=False)
                    if error or not html:
                        continue
                    cleaned = self._html_cleaner.clean(html, source_url=url)
                    if cleaned:
                        text = " ".join(cleaned.split()[:MAX_CONTENT_WORDS])
                        parts.append(f"\nFull text of {url}:\n{text}")
                except Exception as e:
                    logger.debug(f"Escalation fetch failed for {url}: {type(e).__name__}: {e}")

        return "\n".join(parts)
