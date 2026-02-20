"""Anthropic Claude sampler - async wrapper for Claude API"""

import logging
import os
import asyncio
import random
from typing import Any, Optional

import anthropic
from anthropic import AsyncAnthropic
import dotenv

from libs.types import MessageList, SamplerBase, SamplerResponse

dotenv.load_dotenv()

_logger = logging.getLogger(__name__)

# Shared Anthropic client for all samplers (connection pooling)
_shared_anthropic_client: AsyncAnthropic | None = None


def get_shared_anthropic_client() -> AsyncAnthropic:
    """Get or create the shared AsyncAnthropic client for all samplers."""
    global _shared_anthropic_client
    if _shared_anthropic_client is None:
        _shared_anthropic_client = AsyncAnthropic(
            max_retries=0,  # Sampler handles retries with jitter
        )
        _logger.debug("Created shared Anthropic client")
    return _shared_anthropic_client


class AnthropicSampler(SamplerBase):
    """
    Sample from Anthropic's Claude chat completion API
    """

    # Beta header for effort parameter (Claude Opus 4.5 only - GA on Opus 4.6+)
    EFFORT_BETA = "effort-2025-11-24"

    def _is_opus_46_or_newer(self) -> bool:
        """Check if the model is Claude Opus 4.6 or newer.
        
        Opus 4.6 has effort parameter GA (no beta header needed) and supports 'max' effort level.
        """
        model_lower = self.model.lower()
        # Match claude-opus-4-6, claude-opus-4.6, or any version after 4.6
        return "opus-4-6" in model_lower or "opus-4.6" in model_lower

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        system_message: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4048,
        max_retries: int = 5,
        effort: Optional[str] = None,
        websearch: bool = False,
        max_web_searches: int = 5,
    ):
        """
        Initialize the Anthropic sampler.
        
        Args:
            model: Model name (e.g., "claude-sonnet-4-5", "claude-opus-4-5-20251101")
            system_message: Optional system message to prepend
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            max_retries: Number of retries on transient errors
            effort: Optional effort level ("low", "medium", "high", "max"). 
                    Supported by Claude Opus 4.5+ models. Controls token usage vs thoroughness.
                    Note: "max" effort is only available on Opus 4.6+.
                    See: https://platform.claude.com/docs/en/build-with-claude/effort
            websearch: Enable web search tool for real-time information.
                      See: https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
            max_web_searches: Maximum number of web searches per request (default 5)
        """
        self.api_key_name = "ANTHROPIC_API_KEY"
        assert os.environ.get("ANTHROPIC_API_KEY"), "Please set ANTHROPIC_API_KEY"
        self.client = get_shared_anthropic_client()
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.effort = effort
        self.websearch = websearch
        self.max_web_searches = max_web_searches
        
        # Validate effort parameter
        valid_effort_levels = ["low", "medium", "high"]
        if self._is_opus_46_or_newer():
            valid_effort_levels.append("max")
        if effort is not None and effort not in valid_effort_levels:
            raise ValueError(f"effort must be one of {valid_effort_levels}, got: {effort}")
        
        # Build a descriptive tag for logging
        tag_parts = [model]
        if effort:
            tag_parts.append(f"effort={effort}")
        if websearch:
            tag_parts.append("websearch")
        self._log_tag = f"{tag_parts[0]}[{','.join(tag_parts[1:])}]" if len(tag_parts) > 1 else model

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        return {"role": str(role), "content": content}

    def _extract_token_usage(self, response: Any) -> dict[str, int]:
        """Extract token usage from Anthropic API response.
        
        Args:
            response: Anthropic API response object
            
        Returns:
            Dictionary with token counts
        """
        token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
        }
        
        usage = getattr(response, "usage", None)
        
        if usage:
            token_usage["input_tokens"] = getattr(usage, "input_tokens", 0)
            token_usage["output_tokens"] = getattr(usage, "output_tokens", 0)
            token_usage["total_tokens"] = token_usage["input_tokens"] + token_usage["output_tokens"]
            
            # Anthropic reports cache read/creation tokens
            token_usage["cached_tokens"] = getattr(usage, "cache_read_input_tokens", 0)
        
        return token_usage

    async def __call__(self, message_list: MessageList) -> SamplerResponse:
        # Anthropic doesn't accept "system" role in messages - extract and use system param
        # Also handle "developer" role (OpenAI's equivalent) by treating it as system
        system_messages = []
        msgs = []
        
        for msg in message_list:
            role = msg.get("role", "")
            if role in ("system", "developer"):
                # Collect system/developer messages
                content = msg.get("content", "")
                if isinstance(content, str):
                    system_messages.append(content)
                elif isinstance(content, list):
                    # Handle structured content
                    text_parts = [
                        item.get("text", "") for item in content 
                        if isinstance(item, dict) and item.get("type") == "text"
                    ]
                    system_messages.append(" ".join(text_parts))
            else:
                msgs.append(msg)
        
        # Combine all system messages (from input + constructor)
        all_system_parts = []
        if self.system_message:
            all_system_parts.append(self.system_message)
        all_system_parts.extend(system_messages)
        combined_system = "\n\n".join(all_system_parts) if all_system_parts else None
        
        trial = 0

        while True:
            try:
                # Random jitter before request to spread out bursts
                await asyncio.sleep(random.uniform(0, 0.2))
                
                # Prepare common arguments
                kwargs = {
                    "model": self.model,
                    "messages": msgs,
                    "temperature": self.temperature,
                }
                
                # Only include max_tokens if explicitly set
                if self.max_tokens is not None:
                    kwargs["max_tokens"] = self.max_tokens
                
                # Add combined system message if any
                if combined_system:
                    kwargs["system"] = combined_system
                
                # Add web search tool if enabled
                # See: https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
                if self.websearch:
                    kwargs["tools"] = [{
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": self.max_web_searches,
                    }]
                
                # Handle effort parameter
                if self.effort is not None:
                    kwargs["output_config"] = {"effort": self.effort}
                    if self._is_opus_46_or_newer():
                        # Opus 4.6+: effort is GA, no beta header needed
                        response = await self.client.messages.create(**kwargs)
                    else:
                        # Opus 4.5: use beta API with effort header
                        kwargs["betas"] = [self.EFFORT_BETA]
                        response = await self.client.beta.messages.create(**kwargs)
                else:
                    response = await self.client.messages.create(**kwargs)
                
                # Extract text from response content
                # For web search responses, we need to handle multiple block types
                content = ""
                citations = []
                web_search_count = 0
                web_search_results = []
                
                if response.content:
                    text_parts = []
                    for block in response.content:
                        if hasattr(block, "text"):
                            text_parts.append(block.text)
                            # Extract citations if present (these may contain web search results)
                            if hasattr(block, "citations") and block.citations:
                                for citation in block.citations:
                                    if hasattr(citation, "url"):
                                        citation_data = {
                                            "url": citation.url,
                                            "title": getattr(citation, "title", ""),
                                            "cited_text": getattr(citation, "cited_text", ""),
                                        }
                                        citations.append(citation_data)
                                        
                                        # Add as inline citation after the text: [Title (URL)]
                                        if citation_data["title"] or citation_data["url"]:
                                            if citation_data["title"] and citation_data["url"]:
                                                inline_citation = f" [{citation_data['title']} ({citation_data['url']})]"
                                            elif citation_data["title"]:
                                                inline_citation = f" [{citation_data['title']}]"
                                            elif citation_data["url"]:
                                                inline_citation = f" [({citation_data['url']})]"
                                            else:
                                                inline_citation = ""
                                            
                                            if inline_citation:
                                                text_parts.append(inline_citation)
                                                web_search_count += 1
                        elif hasattr(block, "type"):
                            # Check for web search tool result blocks
                            block_type = getattr(block, "type", None)
                            if block_type in ("web_search_tool_result", "tool_result"):
                                # Extract web search results from the content array
                                if hasattr(block, "content") and isinstance(block.content, list):
                                    for item in block.content:
                                        # Check if this is a web_search_result item
                                        if isinstance(item, dict) and item.get("type") == "web_search_result":
                                            web_search_count += 1
                                            title = item.get("title", "")
                                            url = item.get("url", "")
                                            author = item.get("author", "")
                                            
                                            # Store web search result
                                            web_search_results.append({
                                                "url": url,
                                                "title": title,
                                                "author": author,
                                                "content": "",  # encrypted_content is not useful
                                            })
                                            
                                            # Add as inline citation: [Title (URL)]
                                            if title or url:
                                                if title and url:
                                                    citation_text = f" [{title} ({url})]"
                                                elif title:
                                                    citation_text = f" [{title}]"
                                                elif url:
                                                    citation_text = f" [({url})]"
                                                else:
                                                    citation_text = ""
                                                
                                                if citation_text:
                                                    text_parts.append(citation_text)
                    
                    content = "".join(text_parts)
                
                # Extract token usage from response
                token_usage = self._extract_token_usage(response)
                
                # Build response metadata
                response_metadata = {
                    "usage": response.usage,
                    "stop_reason": response.stop_reason,
                }
                if self.websearch:
                    response_metadata["web_search_count"] = web_search_count
                    response_metadata["citations"] = citations
                    response_metadata["web_search_results"] = web_search_results
                
                return SamplerResponse(
                    response_text=content,
                    response_metadata=response_metadata,
                    actual_queried_message_list=msgs,
                    token_usage=token_usage,
                )
            except anthropic.BadRequestError as e:
                _logger.warning(f"[{self._log_tag}] Bad Request Error: {e}")
                raise RuntimeError(f"Anthropic API BadRequestError: {e}") from e
            except anthropic.RateLimitError as e:
                if trial >= self.max_retries:
                    _logger.warning(f"[{self._log_tag}] Max retries ({self.max_retries}) exceeded due to rate limit: {e}")
                    raise RuntimeError(
                        f"Anthropic API rate limit error after {self.max_retries} retries: {e}"
                    ) from e
                # Exponential backoff with jitter to prevent thundering herd
                base_backoff = 2**trial
                jitter = random.uniform(0, base_backoff * 0.5)
                exception_backoff = base_backoff + jitter
                _logger.debug(f"[{self._log_tag}] Rate limit error, retrying {trial} after {exception_backoff:.1f}s: {e}")
                await asyncio.sleep(exception_backoff)
                trial += 1
            except (anthropic.APITimeoutError, asyncio.TimeoutError, anthropic.APIConnectionError) as e:
                if trial >= self.max_retries:
                    _logger.warning(f"[{self._log_tag}] Max retries ({self.max_retries}) exceeded due to connection/timeout: {e}")
                    raise RuntimeError(
                        f"Anthropic API connection/timeout after {self.max_retries} retries: {e}"
                    ) from e
                # Exponential backoff with jitter
                base_backoff = 2**trial
                jitter = random.uniform(0, base_backoff * 0.5)
                exception_backoff = base_backoff + jitter
                _logger.debug(f"[{self._log_tag}] Connection/timeout error, retrying {trial} after {exception_backoff:.1f}s: {e}")
                await asyncio.sleep(exception_backoff)
                trial += 1
            except Exception as e:
                if trial >= self.max_retries:
                    _logger.warning(f"[{self._log_tag}] Max retries ({self.max_retries}) exceeded: {type(e).__name__}: {e}")
                    raise RuntimeError(
                        f"Anthropic API error after {self.max_retries} retries: {e}"
                    ) from e
                # Exponential backoff with jitter
                base_backoff = 2**trial
                jitter = random.uniform(0, base_backoff * 0.5)
                exception_backoff = base_backoff + jitter
                _logger.debug(f"[{self._log_tag}] API error, retrying {trial} after {exception_backoff:.1f}s: {type(e).__name__}: {e}")
                await asyncio.sleep(exception_backoff)
                trial += 1
