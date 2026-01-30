"""Worker for extracting content from search snippets (no URL fetching)."""

from __future__ import annotations

from ..core.queue import MonitoredQueue, QueueItem
from ..core.worker import Worker
from ..models.work_items import SearchTask, ContentItem


class SnippetExtractorWorker(Worker[SearchTask, ContentItem]):
    """Extract content from search snippets without fetching URLs.
    
    Input: SearchTask (with search results from Serper)
    Output: ContentItem (with snippets as content)
    
    This worker is used for the Serper-only pipeline where we don't
    fetch full web pages - we just use the snippets returned by the
    search API.
    
    Contrast with WebFetcherWorker which fetches full HTML pages.
    """
    
    def __init__(
        self,
        input_queue: MonitoredQueue[SearchTask],
        output_queue: MonitoredQueue[ContentItem],
        num_workers: int = 10,
    ):
        """Initialize snippet extractor.
        
        Args:
            input_queue: Queue of search tasks
            output_queue: Queue for content items
            num_workers: Number of concurrent workers
        """
        super().__init__(
            name="SnippetExtractor",
            input_queue=input_queue,
            output_queue=output_queue,
            num_workers=num_workers,
        )
    
    async def process(
        self,
        item: SearchTask,
        item_wrapper: QueueItem[SearchTask],
    ) -> ContentItem:
        """Extract snippets from search results as content."""
        contents = []
        
        # Extract snippets from all search result sets
        for result_set in item.search_results_raw:
            # Answer box (if present)
            if result_set.get("answerBox"):
                ab = result_set["answerBox"]
                answer = ab.get("answer") or ab.get("snippet") or ""
                if answer:
                    contents.append({
                        "title": "Answer Box",
                        "url": "",
                        "snippet": str(answer),
                        "content": str(answer),
                    })
            
            # Knowledge graph (if present)
            if result_set.get("knowledgeGraph"):
                kg = result_set["knowledgeGraph"]
                kg_content = []
                if kg.get("description"):
                    kg_content.append(kg["description"])
                for attr, value in kg.get("attributes", {}).items():
                    kg_content.append(f"{attr}: {value}")
                
                if kg_content:
                    contents.append({
                        "title": kg.get("title", "Knowledge Graph"),
                        "url": "",
                        "snippet": "",
                        "content": "\n".join(kg_content),
                    })
            
            # Organic results
            for organic in result_set.get("organic", []):
                title = organic.get("title", "")
                snippet = organic.get("snippet", "")
                link = organic.get("link", "")
                
                if snippet:
                    contents.append({
                        "title": title,
                        "url": link,
                        "snippet": snippet,
                        "content": snippet,  # For Serper, snippet IS the content
                    })
        
        return ContentItem(
            claim_id=item.claim_id,
            conversation_id=item.conversation_id,
            claim=item.claim,
            contents=contents,
            pdf_contents=[],
            search_results_text=item.search_results_text,
            queries=item.queries_executed,
            whitelist_skip=item.whitelist_skip,
            dynamic_cache_hit=item.dynamic_cache_hit,
            cached_verdict_exists=item.cached_verdict_exists,
        )

