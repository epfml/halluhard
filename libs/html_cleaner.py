"""HTML content cleaning module."""

from __future__ import annotations

import trafilatura


class HtmlCleaner:
    """Clean and extract main content from HTML."""
    
    def __init__(self, max_words: int | None = None):
        """Initialize HTML cleaner.
        
        Args:
            max_words: Maximum words to keep in output
        """
        self.max_words = max_words
    
    def clean(self, html: str, source_url: str = "") -> str:
        """Clean HTML and extract main content.
        
        Args:
            html: Raw HTML content
            source_url: Source URL (for context)
            
        Returns:
            Cleaned text content (markdown format)
        """
        # Use trafilatura for initial extraction
        cleaned = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            include_images=False,
            include_links=True,
            output_format='markdown',
            url=source_url,
            favor_recall=True,
        )
        
        if not cleaned:
            return ""
        
        # Truncate to max_words
        words = cleaned.split()
        if self.max_words and len(words) > self.max_words:
            cleaned = ' '.join(words[:self.max_words])
            cleaned += f"\n\n[Content truncated at {self.max_words} words]"
        
        return cleaned

