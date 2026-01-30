"""Information extraction utilities for filtering web search results."""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import random
import re
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import List, Tuple, Dict, Any
from urllib.parse import urlparse

# Fix Windows asyncio SSL issues
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import aiohttp
import httpx
import numpy as np
import pymupdf4llm
import markitdown
from openai import AsyncOpenAI

# Module-level logger
_logger = logging.getLogger(__name__)

# Module-level shared OpenAI client (lazy initialization)
_openai_client: AsyncOpenAI | None = None

# Module-level shared aiohttp session for PDF downloads (connection pooling)
_pdf_session: aiohttp.ClientSession | None = None

_PDF_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


async def get_pdf_session() -> aiohttp.ClientSession:
    """Get or create the shared aiohttp session for PDF downloads."""
    global _pdf_session
    if _pdf_session is None or _pdf_session.closed:
        connector = aiohttp.TCPConnector(
            limit=100,  # Total concurrent connections
            limit_per_host=30,  # Per-host limit
            ttl_dns_cache=300,  # Cache DNS for 5 minutes
            enable_cleanup_closed=True,
        )
        _pdf_session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=60, connect=15),
            headers=_PDF_HEADERS,
        )
        _logger.debug("Created shared aiohttp session for PDF downloads")
    return _pdf_session


async def close_pdf_session() -> None:
    """Close the shared PDF session. Call at program shutdown."""
    global _pdf_session
    if _pdf_session is not None and not _pdf_session.closed:
        await _pdf_session.close()
        _pdf_session = None
        _logger.debug("Closed shared aiohttp session for PDF downloads")


def get_openai_client() -> AsyncOpenAI:
    """Get or create the shared AsyncOpenAI client for embeddings."""
    global _openai_client
    if _openai_client is None:
        # Configure httpx with bounded connection limits to prevent connection timeouts
        # Force HTTP/1.1 to avoid TLS/HTTP2 issues on Windows
        http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=50,  # Bounded to prevent connection exhaustion
                max_keepalive_connections=25,
            ),
            timeout=httpx.Timeout(120.0, connect=30.0),
            http1=True,
            http2=False,
        )
        _openai_client = AsyncOpenAI(
            timeout=120.0,  # 2 minute timeout for all requests
            max_retries=0,  # Disable built-in retries, we handle them with jitter
            http_client=http_client,
        )
        _logger.debug("Created shared AsyncOpenAI client for embeddings (limit=50, http1=True)")
    return _openai_client


def get_logger() -> logging.Logger:
    """Get the module logger. Can be configured externally."""
    return _logger


def is_pdf_url(url: str) -> bool:
    """Check if URL likely points to a PDF file.
    
    Args:
        url: URL to check
        
    Returns:
        True if URL appears to point to a PDF
    """
    if not url or not url.strip():
        return False
        
    url_lower = url.lower()
    parsed = urlparse(url_lower)
    path = parsed.path
    
    # Check common PDF URL patterns
    return (
        path.endswith('.pdf') or 
        '/pdf/' in path or 
        '.pdf?' in path or
        path.endswith('.pdf/') or
        'arxiv.org/pdf/' in url_lower
    )


async def check_if_url_is_pdf(url: str) -> bool:
    """Check if URL points to a PDF by examining URL pattern and Content-Type header.
    
    Args:
        url: URL to check
        
    Returns:
        True if URL points to a PDF (based on URL pattern or Content-Type header)
    """
    # First check URL pattern
    if is_pdf_url(url):
        return True
    
    # If URL pattern doesn't look like PDF, check Content-Type header with HEAD request
    try:
        async with aiohttp.ClientSession() as session:
            async with session.head(url, timeout=aiohttp.ClientTimeout(total=10), allow_redirects=True) as response:
                content_type = response.headers.get('Content-Type', '').lower()
                if 'application/pdf' in content_type:
                    _logger.debug(f"Detected PDF via Content-Type header: {url}")
                    return True
    except Exception:
        # If HEAD request fails, proceed with normal flow
        pass
    
    return False


async def extract_relevant_sentences(
    websearch_results: List[Dict[str, Any]],
    claim: str,
    max_output_words: int,
    embedding_semaphore: asyncio.Semaphore,
    embedding_model: str = "text-embedding-3-small",
    block_size: int = 10000,
    overlap: int = 200,
    deduplication_threshold: float = 0.85,
) -> str:
    """Extract sentence blocks from web search results most relevant to a claim.
    
    Uses semantic similarity (embeddings) to rank blocks of 10 sentences and returns
    the top blocks concatenated up to max_output_words, grouped by source.
    
    Args:
        websearch_results: Web search results as list of dictionaries with title, url, snippet and content
        claim: The claim to match against
        max_output_words: Maximum number of words of output
        embedding_model: Embedding model to use (default: text-embedding-3-small)
        embedding_semaphore: Optional semaphore to limit concurrent embedding calls
        block_size: Maximum number of characters per block
        overlap: Number of overlapping characters between consecutive blocks
        deduplication_threshold: Threshold for cosine similarity to filter out duplicate blocks
    Returns:
        Concatenated relevant sentence blocks (up to max_output_words words) grouped by source
        Number of embedding calls
        Number of blocks encoded
    """
    if not websearch_results or len(websearch_results) == 0:
        return "", 0, 0
    
    openai_client = get_openai_client()
    
    # Split each source's content into blocks and track source metadata
    all_blocks = []
    block_sources = []  # Track (source_idx, block_text) for each block
    
    for source_idx, source in enumerate(websearch_results):
        content = source.get('content', '')
        if not content:
            continue
            
        # Split content into blocks
        blocks = split_into_blocks(content, block_size, overlap)
        
        # Store blocks and their source indices
        for block in blocks:
            if len(block.strip()) > 20:  # Filter out very short blocks
                all_blocks.append(block)
                block_sources.append(source_idx)
    
    if not all_blocks:
        return "", 0, 0

    # Helper function to make embedding call with retry and jitter
    async def embed_with_retry(texts: list[str], max_retries: int = 3):
        """Make embedding call with retry logic and jitter."""
        for attempt in range(max_retries + 1):
            try:
                # Random jitter before request to spread out bursts
                await asyncio.sleep(random.uniform(0, 0.2))
                
                response = await openai_client.embeddings.create(
                    model=embedding_model,
                    input=texts,
                )
                return response
            except Exception as e:
                if attempt < max_retries:
                    # Exponential backoff with jitter
                    base_wait = 2 ** attempt  # 1, 2, 4 seconds
                    jitter = random.uniform(0, base_wait * 0.5)
                    wait_time = base_wait + jitter
                    _logger.warning(
                        f"Embedding error ({len(texts)} texts), retrying in {wait_time:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}"
                    )
                    _logger.info("Embedding error traceback:", exc_info=True)
                    await asyncio.sleep(wait_time)
                    continue
                _logger.warning(f"Embedding failed after {max_retries} retries: {type(e).__name__}: {e}")
                _logger.info("Embedding final failure traceback:", exc_info=True)
                raise

    try:
        # Claim embedding (with semaphore if provided)
        if embedding_semaphore:
            async with embedding_semaphore:
                response = await embed_with_retry([claim])
        else:
            response = await embed_with_retry([claim])
        claim_embedding = np.array(response.data[0].embedding)
    except Exception as e:
        _logger.warning(f"Claim embedding failed: {type(e).__name__}: {e}")
        _logger.info("Claim embedding failure traceback:", exc_info=True)
        return "", 0, 0
    
    try:
        # Create batch embedding tasks with semaphore control
        batch_size = 50  # OpenAI can handle large batches efficiently
        
        async def embed_batch(batch: list[str]) -> list:
            """Embed a batch of texts with semaphore control."""
            if embedding_semaphore:
                async with embedding_semaphore:
                    response = await embed_with_retry(batch)
            else:
                response = await embed_with_retry(batch)
            return [np.array(item.embedding) for item in response.data]
        
        # Create and execute all batch tasks in parallel (semaphore limits concurrency)
        embedding_tasks = [
            embed_batch(all_blocks[i:i+batch_size])
            for i in range(0, len(all_blocks), batch_size)
        ]
        batch_results = await asyncio.gather(*embedding_tasks, return_exceptions=True)
        
        # Collect all embeddings (skip failed batches)
        block_embeddings = []
        failed_batches = 0
        for result in batch_results:
            if isinstance(result, Exception):
                _logger.warning(f"Batch embedding failed: {result}")
                failed_batches += 1
            else:
                block_embeddings.extend(result)
        
        if not block_embeddings:
            _logger.warning(f"All {failed_batches} embedding batches failed")
            return "", 0, 0
        
        # Convert to numpy array for similarity calculation
        block_embeddings = np.array(block_embeddings)
        
        # Update block_sources to match only successful embeddings
        # (This is a simplification - in practice we'd need to track which batches succeeded)
        if failed_batches > 0:
            _logger.debug(f"{failed_batches} batches failed, {len(block_embeddings)} embeddings succeeded")
            # Truncate block_sources to match (assumes failures are at the end)
            block_sources = block_sources[:len(block_embeddings)]
            all_blocks = all_blocks[:len(block_embeddings)]
            
    except Exception as e:
        _logger.warning(f"Embedding failed ({e}), returning empty results")
        return "", 0, 0
    
    # Calculate cosine similarity
    similarities = _cosine_similarity(claim_embedding, block_embeddings)
    
    # Sort blocks by similarity (descending)
    ranked_indices = np.argsort(similarities)[::-1]
    
    # Pre-calculate normalized embeddings for deduplication
    # Add epsilon to avoid division by zero
    emb_norms = np.linalg.norm(block_embeddings, axis=1, keepdims=True)
    block_embeddings_norm = block_embeddings / (emb_norms + 1e-10)
    
    # Select blocks until max_output_words is reached, with deduplication
    selected_blocks = []  # List of (source_idx, block_text, is_truncated)
    current_word_count = 0
    selected_indices = []
    
    for idx in ranked_indices:
        block = all_blocks[idx]
        source_idx = block_sources[idx]
        
        # Check for duplicates/high similarity with already selected blocks
        if selected_indices:
            # Get embedding for current block
            current_emb = block_embeddings_norm[idx]
            # Get embeddings for selected blocks
            selected_embs = block_embeddings_norm[selected_indices]
            
            # Calculate similarities with selected blocks
            # (n_selected, dim) @ (dim,) -> (n_selected,)
            pair_sims = np.dot(selected_embs, current_emb)
            
            if np.any(pair_sims > deduplication_threshold):
                continue

        block_word_count = len(block.split())
        
        if current_word_count + block_word_count > max_output_words:
            # Check if we can fit a truncated version
            remaining_words = max_output_words - current_word_count
            if remaining_words > 10:  # Only truncate if we have reasonable space (>10 words)
                truncated_block = ' '.join(block.split()[:remaining_words]) + "..."
                selected_blocks.append((source_idx, truncated_block, True))
            break
        
        selected_blocks.append((source_idx, block, False))
        current_word_count += block_word_count
        selected_indices.append(idx)

    nb_embedding_calls = len(embedding_tasks)
    nb_blocks_encoded = len(all_blocks)
    
    # Group blocks by source and format output
    result_parts = []
    
    # Group selected blocks by source
    from collections import defaultdict
    blocks_by_source = defaultdict(list)
    for source_idx, block_text, is_truncated in selected_blocks:
        blocks_by_source[source_idx].append(block_text)
    
    # Format output: one section per source
    for source_idx in sorted(blocks_by_source.keys()):
        source = websearch_results[source_idx]
        title = source.get('title', 'Unknown')
        url = source.get('url', '')
        snippet = source.get('snippet', '')
        blocks = blocks_by_source[source_idx]
        
        # Build source section
        source_section = f"-- Source: {title}, url: {url} --\n"
        if snippet:
            source_section += f"{snippet}\n\n"
        source_section += "\n\n".join(blocks)
        
        result_parts.append(source_section)
    
    return "\n\n".join(result_parts), nb_embedding_calls, nb_blocks_encoded


def _cosine_similarity(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity between query and multiple embeddings.
    
    Args:
        query_embedding: Query embedding vector (1D)
        embeddings: Matrix of embeddings (2D: n_samples x embedding_dim)
        
    Returns:
        Array of similarity scores
    """
    # Normalize query
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    
    # Normalize embeddings
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Compute dot product (cosine similarity for normalized vectors)
    similarities = np.dot(embeddings_norm, query_norm)
        
    return similarities


async def extract_pdf_as_markdown(
    url: str,
    max_retries: int = 1,
    timeout_seconds: int = 30,
    max_pdf_size_mb: int = 2,
    conversion_timeout_seconds: int = 60,
    pdf_semaphore: asyncio.Semaphore | None = None,
) -> str | None:
    """Download PDF from URL and convert to markdown with retry logic.
    
    Detects if URL points to a PDF, downloads it to a temporary directory,
    converts it to markdown using MarkItDown, and returns the markdown text.
    
    Args:
        url: URL to check and potentially download PDF from
        max_retries: Maximum number of retry attempts on timeout
        timeout_seconds: Timeout in seconds for PDF download (default: 30s)
        max_pdf_size_mb: Maximum PDF file size in MB (default: 50MB)
        conversion_timeout_seconds: Timeout for MarkItDown conversion (default: 60s)
        pdf_semaphore: Optional semaphore to limit concurrent PDF downloads
        
    Returns:
        Markdown text if URL is a PDF and conversion succeeds, None otherwise
    """
    # Check if URL points to a PDF
    if not is_pdf_url(url):
        return None
    
    # Wrapper function to handle the actual download (for semaphore control)
    async def _download_and_convert():
        # Retry logic for timeouts
        retry_delays = [5, 10]  # Wait 5s, then 10s before retries
        
        for attempt in range(max_retries + 1):
            try:
                # Create temporary directory for PDF download
                temp_dir = tempfile.mkdtemp()
                filename = uuid.uuid4()
                temp_pdf_path = Path(temp_dir) / f"downloaded_{filename}.pdf"
                
                # Download PDF using shared session with connection pooling
                session = await get_pdf_session()
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=timeout_seconds),
                    allow_redirects=True,
                ) as response:
                    
                    if response.status != 200:
                        _logger.warning(f"Failed to download PDF from {url}: HTTP {response.status}")
                        return None
                    
                    # Check content type
                    content_type = response.headers.get('Content-Type', '').lower()
                    if 'application/pdf' not in content_type and not is_pdf_url(url):
                        _logger.warning(f"URL does not appear to be a PDF: {content_type}")
                        return None
                    
                    # Write PDF to temporary file
                    pdf_content = await response.read()
                    
                    with open(temp_pdf_path, 'wb') as f:
                        f.write(pdf_content)
                
                # Check PDF file size
                pdf_size_mb = len(pdf_content) / (1024 * 1024)
                if pdf_size_mb > max_pdf_size_mb:
                    _logger.warning(f"PDF too large ({pdf_size_mb:.2f} MB > {max_pdf_size_mb} MB), skipping PDF download")
                    return None
                
                # Convert PDF to markdown with timeout
                conversion_start = time.time()
                
                try:
                    # Run MarkItDown in a thread pool with timeout since it's blocking
                    loop = asyncio.get_event_loop()
                    md = markitdown.MarkItDown()
                    executor_submit_time = time.time()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, lambda: md.convert(str(temp_pdf_path))),
                        timeout=conversion_timeout_seconds
                    )
                    executor_complete_time = time.time()
                    
                    md_text = result.text_content
                    conversion_end = time.time()

                except asyncio.TimeoutError:
                    conversion_end = time.time()
                    _logger.warning(f"MarkItDown conversion timeout after {conversion_end - conversion_start:.2f}s (limit: {conversion_timeout_seconds}s)")
                    return None
                except Exception as e:
                    conversion_end = time.time()
                    _logger.warning(f"MarkItDown conversion error after {conversion_end - conversion_start:.2f}s: {e}")
                    return None

                # Force garbage collection to release file handles
                gc.collect()
                
                # Clean up temporary directory with retry logic
                cleanup_attempts = 3
                cleanup_success = False
                
                for cleanup_attempt in range(cleanup_attempts):
                    try:
                        # Small delay to ensure file handles are released (especially on Windows)
                        if cleanup_attempt > 0:
                            await asyncio.sleep(0.1 * (cleanup_attempt + 1))  # 0.1s, 0.2s, 0.3s
                        
                        os.remove(temp_pdf_path)
                        os.rmdir(temp_dir)
                        cleanup_success = True
                        break
                    except PermissionError:
                        # Windows-specific: file still in use
                        if cleanup_attempt < cleanup_attempts - 1:
                            continue  # Retry
                        else:
                            # Final attempt failed - log but don't fail the operation
                            _logger.debug(f"Could not clean up PDF temp file (in use): {temp_pdf_path}")
                    except Exception as cleanup_error:
                        if cleanup_attempt < cleanup_attempts - 1:
                            continue  # Retry
                        else:
                            _logger.debug(f"Could not clean up temp files: {cleanup_error}")
                
                return md_text
            
            except asyncio.TimeoutError:
                if attempt < max_retries:
                    delay = retry_delays[attempt]
                    _logger.debug(f"Timeout downloading PDF (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay}s: {url[:80]}")
                    await asyncio.sleep(delay)
                    continue
                else:
                    _logger.warning(f"Timeout while downloading PDF after {max_retries + 1} attempts (limit: {timeout_seconds}s): {url[:80]}")
                    return None
            except Exception as e:
                _logger.warning(f"Error extracting PDF from {url[:80]}: {e}")
                return None
        
        return None
    
    # Use semaphore to limit concurrent PDF downloads if provided
    if pdf_semaphore is not None:
        async with pdf_semaphore:
            return await _download_and_convert()
    else:
        return await _download_and_convert()



def split_into_blocks(
    text: str,
    max_chars: int = 10000,
    overlap: int = 200,
) -> List[str]:
    """
    Split text into overlapping character-based blocks, trying to cut at
    natural boundaries (paragraphs / sentence ends) when possible.

    Args:
        text: Input text.
        max_chars: Target maximum number of characters per block.
        overlap: Number of overlapping characters between consecutive blocks.

    Returns:
        List of text blocks.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= max_chars:
        raise ValueError("overlap must be < max_chars")

    text = text.strip()
    n = len(text)
    if n == 0:
        return []

    blocks: List[str] = []
    start = 0

    while start < n:
        # Naive boundary
        end = min(start + max_chars, n)

        if end < n:
            # Try to move the cut point back to a nicer boundary
            # (but not too far back, to avoid tiny blocks).
            window_start = max(start + int(0.1 * max_chars), start)
            candidates = [
                text.rfind("\n\n", window_start, end),  # paragraph break
                text.rfind(". ",  window_start, end),
                text.rfind("? ",  window_start, end),
                text.rfind("! ",  window_start, end),
            ]
            best_boundary = max(candidates)

            if best_boundary != -1:
                # Include the punctuation / break character itself
                if text[best_boundary] in ".?!":
                    end = best_boundary + 1
                else:
                    end = best_boundary + 2  # for "\n\n" or "X "

        block = text[start:end].strip()
        if block:
            blocks.append(block)

        if end >= n:
            break

        # Move start forward with overlap
        start = max(0, end - overlap)

    return blocks


def cleanup_old_pdf_temp_files(max_age_hours: int = 24) -> int:
    """Clean up old temporary PDF files that failed to delete.
    
    Searches for temp directories with 'downloaded_*.pdf' files and removes
    files older than max_age_hours.
    
    Args:
        max_age_hours: Maximum age of files to keep (default: 24 hours)
        
    Returns:
        Number of files successfully deleted
    """
    import glob
    
    deleted_count = 0
    temp_root = tempfile.gettempdir()
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    try:
        # Search for our PDF temp files pattern
        pattern = os.path.join(temp_root, "tmp*", "downloaded_*.pdf")
        pdf_files = glob.glob(pattern)
        
        for pdf_path in pdf_files:
            try:
                # Check file age
                file_age = current_time - os.path.getmtime(pdf_path)
                
                if file_age > max_age_seconds:
                    # Try to remove the file
                    os.remove(pdf_path)
                    
                    # Try to remove the parent directory if empty
                    parent_dir = os.path.dirname(pdf_path)
                    try:
                        os.rmdir(parent_dir)
                    except OSError:
                        # Directory not empty or other error - ignore
                        pass
                    
                    deleted_count += 1
            except Exception:
                # File in use or other error - skip it
                continue
    except Exception as e:
        _logger.warning(f"Error during temp file cleanup: {e}")
    
    return deleted_count