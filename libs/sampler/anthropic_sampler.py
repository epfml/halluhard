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

    # Anthropic only caches a prefix that reaches a minimum length: ~1024
    # tokens for Opus/Sonnet, ~2048 for Haiku. Below it the cache_control
    # marker is silently ignored, so we skip marking short prompts rather than
    # emit markers that do nothing. Thresholds are in characters at a
    # deliberately conservative ~4 chars/token.
    MIN_CACHEABLE_CHARS = 4400
    MIN_CACHEABLE_CHARS_HAIKU = 8800

    def _is_opus_46_or_newer(self) -> bool:
        """Check if the model is Claude Opus 4.6 or newer.
        
        Opus 4.6 has effort parameter GA (no beta header needed) and supports 'max' effort level.
        """
        model_lower = self.model.lower()
        # Match claude-opus-4-6, claude-opus-4.6, or any version after 4.6
        return "opus-4-6" in model_lower or "opus-4.6" in model_lower

    def _is_opus_47(self) -> bool:
        """Check if the model is Claude Opus 4.7.
        
        Opus 4.7 supports adaptive thinking mode.
        """
        model_lower = self.model.lower()
        return "opus-4-7" in model_lower or "opus-4.7" in model_lower

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        system_message: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 10000,
        max_retries: int = 5,
        effort: Optional[str] = None,
        websearch: bool = False,
        max_web_searches: int = 5,
        prompt_caching: bool = True,
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
            prompt_caching: Mark the reusable prefix of each request with
                      cache_control so repeated system prompts (and, on
                      multi-turn requests, the conversation so far) are billed
                      at the cache-read rate instead of in full.
                      See: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
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
        self.prompt_caching = prompt_caching
        
        # Validate effort parameter
        valid_effort_levels = ["low", "medium", "high"]
        if self._is_opus_46():
            valid_effort_levels.append("max")
        if effort is not None and effort not in valid_effort_levels:
            raise ValueError(f"effort must be one of {valid_effort_levels}, got: {effort}")
        
        # Validate thinking parameter (only for Opus 4.7)
        if thinking and not self._is_opus_47():
            raise ValueError("thinking mode is only supported for Claude Opus 4.7")
        
        # Validate temperature for thinking mode (must be 1.0)
        if thinking and temperature != 1.0:
            raise ValueError(f"temperature must be 1.0 when thinking is enabled, got: {temperature}")
        
        # Build a descriptive tag for logging
        tag_parts = [model]
        if effort:
            tag_parts.append(f"effort={effort}")
        if websearch:
            tag_parts.append("websearch")
        if thinking:
            tag_parts.append("thinking")
        self._log_tag = f"{tag_parts[0]}[{','.join(tag_parts[1:])}]" if len(tag_parts) > 1 else model

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        return {"role": str(role), "content": content}

    def _min_cacheable_chars(self) -> int:
        """Smallest prompt worth marking for cache, in characters."""
        if "haiku" in self.model.lower():
            return self.MIN_CACHEABLE_CHARS_HAIKU
        return self.MIN_CACHEABLE_CHARS

    @staticmethod
    def _conversation_chars(msgs: list) -> int:
        """Rough character size of a message list, for cache-threshold checks."""
        total = 0
        for msg in msgs:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        total += len(block.get("text") or "")
                    elif isinstance(block, str):
                        total += len(block)
        return total

    def _build_system_param(self, combined_system: str):
        """System prompt, carrying a cache breakpoint when it is long enough.

        Anthropic caches everything before the breakpoint, and tools are
        ordered ahead of the system prompt, so this single marker covers the
        web search tool definition as well. Every caller in this repo sends one
        fixed system prompt plus per-item content, so the whole prefix is
        reusable across the run.
        """
        if not self.prompt_caching or len(combined_system) < self._min_cacheable_chars():
            return combined_system
        return [
            {
                "type": "text",
                "text": combined_system,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def _apply_conversation_cache_breakpoint(self, msgs: list) -> list:
        """Cache the conversation so far on multi-turn requests.

        Marking the tail of the newest turn means the *next* turn reads this
        entire conversation from cache. A single-turn request has no next turn
        to repay the 1.25x cache-write surcharge, so this only applies once
        there is a real conversation prefix (user, assistant, user).
        """
        if not self.prompt_caching or len(msgs) < 3:
            return msgs

        last = msgs[-1]
        if not isinstance(last, dict):
            return msgs

        # What has to clear the minimum is the whole prefix being cached, not
        # the newest message: a one-line follow-up on a long conversation is
        # exactly the case caching pays off best.
        if self._conversation_chars(msgs) < self._min_cacheable_chars():
            return msgs

        content = last.get("content")
        if isinstance(content, str):
            blocks = [{"type": "text", "text": content}]
        elif isinstance(content, list) and content:
            blocks = list(content)
        else:
            return msgs

        if not isinstance(blocks[-1], dict):
            return msgs

        blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
        return msgs[:-1] + [{**last, "content": blocks}]

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
            "cache_creation_tokens": 0,
            "reasoning_tokens": 0,
        }
        
        usage = getattr(response, "usage", None)
        
        if usage:
            token_usage["input_tokens"] = getattr(usage, "input_tokens", 0) or 0
            token_usage["output_tokens"] = getattr(usage, "output_tokens", 0) or 0
            
            # Anthropic reports cache reads (billed at 0.1x) and cache writes
            # (billed at 1.25x) separately.
            token_usage["cached_tokens"] = getattr(usage, "cache_read_input_tokens", 0) or 0
            token_usage["cache_creation_tokens"] = getattr(usage, "cache_creation_input_tokens", 0) or 0
            
            # input_tokens counts only the uncached remainder, so cached reads
            # and writes have to be added back for a comparable total.
            token_usage["total_tokens"] = (
                token_usage["input_tokens"]
                + token_usage["cached_tokens"]
                + token_usage["cache_creation_tokens"]
                + token_usage["output_tokens"]
            )
        
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
                    "messages": self._apply_conversation_cache_breakpoint(msgs),
                    "temperature": self.temperature,
                }

                # Only include temperature if set. Newer models (e.g. Sonnet 5,
                # Fable 5) deprecate the temperature parameter and 400 if it's sent.
                if self.temperature is not None:
                    kwargs["temperature"] = self.temperature

                # Only include max_tokens if explicitly set
                if self.max_tokens is not None:
                    kwargs["max_tokens"] = self.max_tokens
                
                # Add combined system message if any, with a cache breakpoint
                if combined_system:
                    kwargs["system"] = self._build_system_param(combined_system)
                
                # Add web search tool if enabled
                # See: https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
                if self.websearch:
                    kwargs["tools"] = [{
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": self.max_web_searches,
                    }]
                
                # Handle effort and thinking parameters
                if self.effort is not None:
                    kwargs["output_config"] = {"effort": self.effort}
                
                if self.thinking:
                    # Opus 4.7: adaptive thinking mode
                    kwargs["thinking"] = {"type": "adaptive"}
                
                # Determine which API to use and whether to stream
                # Use streaming for thinking mode (Opus 4.7) since it may exceed 10 minute timeout
                use_streaming = self.thinking
                
                if self.effort is not None and not self._is_opus_46():
                    # Opus 4.5: use beta API with effort header for effort parameter
                    kwargs["betas"] = [self.EFFORT_BETA]
                    if use_streaming:
                        stream = self.client.beta.messages.stream(**kwargs)
                        content, usage_data = await self._collect_streamed_response(stream)
                        # Create a mock response object with collected data
                        response = type('obj', (object,), {
                            'content': [type('obj', (object,), {'text': content})()],
                            'usage': usage_data,
                            'stop_reason': 'end_turn',
                        })()
                    else:
                        response = await self.client.beta.messages.create(**kwargs)
                else:
                    if use_streaming:
                        stream = self.client.messages.stream(**kwargs)
                        content, usage_data = await self._collect_streamed_response(stream)
                        # Create a mock response object with collected data
                        response = type('obj', (object,), {
                            'content': [type('obj', (object,), {'text': content})()],
                            'usage': usage_data,
                            'stop_reason': 'end_turn',
                        })()
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
