"""Adapted from response_sampler.py in OpenAI's Simple-Evals to be async and support websearch"""

import hashlib
import logging
import os
import asyncio
import random
from typing import Any

import httpx
import openai
from openai import AsyncOpenAI

from libs.types import MessageList, SamplerBase, SamplerResponse

import dotenv

dotenv.load_dotenv()

_logger = logging.getLogger(__name__)


def content_to_text(content: Any) -> str:
    """Flatten a message's content (str or list of content blocks) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text") or block.get("input_text") or "")
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""

# Shared OpenAI client for all samplers (connection pooling)
_shared_openai_client: AsyncOpenAI | None = None


def get_shared_openai_client(max_connections: int = 50) -> AsyncOpenAI:
    """Get or create the shared AsyncOpenAI client for all samplers.
    
    Args:
        max_connections: Max concurrent connections. Bounded to prevent
                         connection exhaustion and ClientOSErrors.
                         Default 50 is sufficient with connection reuse.
    """
    global _shared_openai_client
    if _shared_openai_client is None:
        # Configure httpx with bounded connection limits to prevent connection timeouts
        # Force HTTP/1.1 to avoid TLS/HTTP2 issues on Windows
        http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections // 2,
            ),
            timeout=httpx.Timeout(300.0, connect=60.0),  # 5 min total, 60s connect
            http1=True,
            http2=False,
        )
        _shared_openai_client = AsyncOpenAI(
            timeout=300.0,  # 5 minute timeout per request
            max_retries=0,  # Samplers handle retries themselves with jitter
            http_client=http_client,
        )
        _logger.debug(f"Created shared OpenAI client (max_connections={max_connections}, http1=True)")
    return _shared_openai_client


class ResponsesSampler(SamplerBase):
    """
    Sample from OpenAI's responses API
    """

    # Class-level defaults so that subclasses which deliberately skip
    # ResponsesSampler.__init__ but reuse __call__ (see
    # NvidiaInferenceResponsesSampler) still resolve these attributes.
    prompt_cache_key: str | None = None
    _send_prompt_cache_key: bool = True

    def __init__(
        self,
        model: str,
        system_message: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        max_retries: int = 10,
        websearch: bool = False,
        prompt_cache_key: str | None = None,
    ):
        self.api_key_name = "OPENAI_API_KEY"
        assert os.environ.get("OPENAI_API_KEY"), "Please set OPENAI_API_KEY"
        self.client = get_shared_openai_client()  # Use shared client for connection pooling
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        self.websearch = websearch
        self.prompt_cache_key = prompt_cache_key
        # Turned off permanently if the endpoint rejects the parameter, so an
        # older or non-OpenAI-hosted deployment degrades instead of failing.
        self._send_prompt_cache_key = True
        
        # Build a descriptive tag for logging
        tag_parts = [model]
        if reasoning_effort:
            tag_parts.append(reasoning_effort)
        if websearch:
            tag_parts.append("websearch")
        self._log_tag = "-".join(tag_parts)

    def _handle_image(
        self,
        image: str,
        encoding: str = "base64",
        format: str = "png",
        fovea: int = 768,
    ) -> dict[str, Any]:
        new_image = {
            "type": "input_image",
            "image_url": f"data:image/{format};{encoding},{image}",
        }
        return new_image

    def _handle_text(self, text: str) -> dict[str, Any]:
        return {"type": "input_text", "text": text}

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        return {"role": role, "content": content}

    def _derive_prompt_cache_key(self, message_list: MessageList) -> str | None:
        """Stable cache key for the invariant head of this prompt.

        Prompt caching on the Responses API is automatic: OpenAI reuses any
        prefix it has seen recently (>=1024 tokens) and bills those tokens at
        roughly a tenth of the uncached rate. The one lever a client has is
        `prompt_cache_key`, which keeps requests that share a prefix pinned to
        the same cache instead of scattering them across machines. Every task
        here sends one fixed system prompt followed by per-item content, so we
        key on the system prompt: one key per distinct prompt, which is exactly
        the low-cardinality grouping the parameter wants.

        See https://developers.openai.com/api/docs/guides/prompt-caching
        """
        if self.prompt_cache_key:
            return self.prompt_cache_key

        prefix_parts = []
        for msg in message_list:
            if not isinstance(msg, dict) or msg.get("role") not in ("system", "developer"):
                break
            prefix_parts.append(content_to_text(msg.get("content", "")))

        if not prefix_parts:
            return None

        digest = hashlib.sha256("\n".join(prefix_parts).encode("utf-8")).hexdigest()[:16]
        return f"{self.model}:{digest}"

    def _extract_token_usage(self, response: Any) -> dict[str, int]:
        """Extract token usage from OpenAI API response.
        
        Args:
            response: OpenAI API response object
            
        Returns:
            Dictionary with token counts including cached and reasoning tokens
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
            token_usage["total_tokens"] = getattr(usage, "total_tokens", 0)
            
            # GPT-5 introduces detailed usage breakdowns (these are objects, not dicts)
            input_details = getattr(usage, "input_tokens_details", None)
            output_details = getattr(usage, "output_tokens_details", None)
            
            # Cached and reasoning tokens are reported here
            if input_details:
                token_usage["cached_tokens"] = getattr(input_details, "cached_tokens", 0)
            if output_details:
                token_usage["reasoning_tokens"] = getattr(output_details, "reasoning_tokens", 0)
        
        return token_usage

    async def __call__(self, message_list: MessageList) -> SamplerResponse:
        if self.system_message:
            message_list = [
                self._pack_message("developer", self.system_message)
            ] + message_list
        trial = 0
        while True:
            try:
                # Random jitter before request to spread out bursts
                await asyncio.sleep(random.uniform(0, 0.2))
                
                # Prepare common arguments
                kwargs = {
                    "model": self.model,
                    "input": message_list,
                }
                
                # Pin requests sharing a system prompt to the same prompt cache.
                if self._send_prompt_cache_key:
                    cache_key = self._derive_prompt_cache_key(message_list)
                    if cache_key:
                        kwargs["prompt_cache_key"] = cache_key

                # Add websearch tools if enabled
                if self.websearch:
                    kwargs["tools"] = [{"type": "web_search"}]

                # Add reasoning or temperature/max_tokens
                if self.reasoning_effort:
                    kwargs["reasoning"] = {"effort": self.reasoning_effort}
                    response = await self.client.responses.create(**kwargs)
                else:
                    # Only add temperature if not None (some models don't support it)
                    if self.temperature is not None:
                        kwargs["temperature"] = self.temperature

                    if self.max_tokens is not None:
                        kwargs["max_output_tokens"] = self.max_tokens

                    response = await self.client.responses.create(**kwargs)

                # Check if response is incomplete
                if response.status == 'incomplete':
                    if response.incomplete_details and response.incomplete_details.reason == 'max_output_tokens':
                        _logger.warning(f"[{self._log_tag}] Response truncated due to max_output_tokens limit ({self.max_tokens})")
                        # For incomplete responses, we might need to handle this differently
                        # For now, we'll try to get any available text
                        response_text = response.output_text or ""
                    else:
                        _logger.warning(f"[{self._log_tag}] Response incomplete for unknown reason: {response.incomplete_details}")
                        response_text = response.output_text or ""
                else:
                    response_text = response.output_text

                # Extract token usage from response
                token_usage = self._extract_token_usage(response)

                return SamplerResponse(
                    response_text=response_text,
                    response_metadata={"usage": response.usage, "status": response.status},
                    actual_queried_message_list=message_list,
                    token_usage=token_usage,
                )
            except openai.BadRequestError as e:
                # An endpoint that does not know `prompt_cache_key` should cost
                # us caching, not the request. Drop it and retry once.
                if self._send_prompt_cache_key and "prompt_cache_key" in str(e):
                    _logger.warning(
                        f"[{self._log_tag}] Endpoint rejected prompt_cache_key; "
                        f"disabling it for this sampler: {e}"
                    )
                    self._send_prompt_cache_key = False
                    continue
                _logger.warning(f"[{self._log_tag}] Bad Request Error: {e}")
                raise RuntimeError(f"OpenAI API BadRequestError: {e}") from e
            except openai.RateLimitError as e:
                if trial >= self.max_retries:
                    _logger.warning(f"[{self._log_tag}] Max retries ({self.max_retries}) exceeded due to rate limit: {e}")
                    raise RuntimeError(
                        f"OpenAI API rate limit error after {self.max_retries} retries: {e}"
                    ) from e
                # Exponential backoff with jitter to prevent thundering herd
                base_backoff = 2**trial
                jitter = random.uniform(0, base_backoff * 0.5)
                exception_backoff = base_backoff + jitter
                _logger.debug(f"[{self._log_tag}] Rate limit error, retrying {trial} after {exception_backoff:.1f}s: {e}")
                await asyncio.sleep(exception_backoff)
                trial += 1
            except (openai.APITimeoutError, asyncio.TimeoutError, openai.APIConnectionError) as e:
                if trial >= self.max_retries:
                    _logger.warning(f"[{self._log_tag}] Max retries ({self.max_retries}) exceeded due to connection/timeout: {e}")
                    raise RuntimeError(
                        f"OpenAI API connection/timeout after {self.max_retries} retries: {e}"
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
                        f"OpenAI API error after {self.max_retries} retries: {e}"
                    ) from e
                # Exponential backoff with jitter
                base_backoff = 2**trial
                jitter = random.uniform(0, base_backoff * 0.5)
                exception_backoff = base_backoff + jitter
                _logger.debug(f"[{self._log_tag}] API error, retrying {trial} after {exception_backoff:.1f}s: {type(e).__name__}: {e}")
                await asyncio.sleep(exception_backoff)
                trial += 1
