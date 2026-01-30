"""Kimi K2 sampler - async wrapper for Moonshot AI's Kimi API.

Kimi K2 uses an OpenAI-compatible API.
See: https://platform.moonshot.ai/docs/guide/kimi-k2-quickstart
"""

import logging
import os
import asyncio
import random
from typing import Any, Optional

import openai
from openai import AsyncOpenAI
import httpx
import dotenv

from libs.types import MessageList, SamplerBase, SamplerResponse

dotenv.load_dotenv()

_logger = logging.getLogger(__name__)

# Shared Kimi client for all samplers (connection pooling)
_shared_kimi_client: AsyncOpenAI | None = None


def get_shared_kimi_client(max_connections: int = 50) -> AsyncOpenAI:
    """Get or create the shared AsyncOpenAI client for Kimi API.

    Uses a bounded httpx client to avoid connection exhaustion/timeouts under high concurrency
    (common in our response generation scripts).
    """
    global _shared_kimi_client
    if _shared_kimi_client is None:
        api_key = os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            raise ValueError("Please set MOONSHOT_API_KEY environment variable")

        http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections // 2,
            ),
            timeout=httpx.Timeout(300.0, connect=60.0),
            http1=True,
            http2=False,
        )
        _shared_kimi_client = AsyncOpenAI(
            base_url="https://api.moonshot.ai/v1",
            api_key=api_key,
            timeout=300.0,
            max_retries=0,  # samplers handle retries with jitter
            http_client=http_client,
        )
        _logger.debug("Created shared Kimi client (http1=True)")
    return _shared_kimi_client


class KimiSampler(SamplerBase):
    """
    Sample from Moonshot AI's Kimi chat completion API.
    
    Kimi K2 is a powerful MoE model with 1T total parameters (32B activated).
    Uses OpenAI-compatible API format.
    
    See: https://platform.moonshot.ai/docs/guide/kimi-k2-quickstart
    """

    def __init__(
        self,
        model: str = "kimi-k2-0711-preview",
        system_message: Optional[str] = None,
        temperature: float = 0.6,
        max_tokens: int = 4096,
        max_retries: int = 5,
    ):
        """
        Initialize the Kimi sampler.
        
        Args:
            model: Model name (e.g., "kimi-k2-0711-preview", "moonshot-v1-8k")
            system_message: Optional system message to prepend
            temperature: Sampling temperature (Kimi recommends 0.6 for K2)
            max_tokens: Maximum tokens in response
            max_retries: Number of retries on transient errors
        """
        self.api_key_name = "MOONSHOT_API_KEY"
        assert os.environ.get("MOONSHOT_API_KEY"), "Please set MOONSHOT_API_KEY"
        self.client = get_shared_kimi_client()
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        
        # Build a descriptive tag for logging
        self._log_tag = model

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        return {"role": str(role), "content": content}

    def _extract_token_usage(self, response: Any) -> dict[str, int]:
        """Extract token usage from OpenAI-compatible API response.
        
        Args:
            response: API response object
            
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
            token_usage["input_tokens"] = getattr(usage, "prompt_tokens", 0)
            token_usage["output_tokens"] = getattr(usage, "completion_tokens", 0)
            token_usage["total_tokens"] = getattr(usage, "total_tokens", 0)
        
        return token_usage

    def _sanitize_messages(self, messages: list) -> list:
        """Sanitize messages to ensure compatibility with Kimi API.
        
        Kimi API strictly requires non-empty content for assistant messages.
        This filters out empty assistant messages that may have been added
        from previous failed/filtered responses.
        """
        sanitized = []
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "")
            
            # Skip empty assistant messages (Kimi strictly rejects these)
            if role == "assistant" and (not content or not str(content).strip()):
                _logger.warning(f"[{self._log_tag}] Filtering empty assistant message from history")
                continue
            
            sanitized.append(msg)
        
        return sanitized

    async def __call__(self, message_list: MessageList) -> SamplerResponse:
        # Add system message if provided
        msgs = list(message_list)
        if self.system_message:
            msgs.insert(0, self._pack_message("system", self.system_message))
        
        # Sanitize messages - Kimi API strictly rejects empty assistant messages
        msgs = self._sanitize_messages(msgs)
        
        trial = 0

        while True:
            try:
                # Random jitter before request to spread out bursts
                await asyncio.sleep(random.uniform(0, 0.2))
                
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=msgs,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                
                content = response.choices[0].message.content or ""
                
                # Kimi may return empty responses due to content filtering
                # Raise error early so conversation generator knows the response failed
                if not content.strip():
                    raise RuntimeError(
                        f"Kimi returned empty response (likely content filtered). "
                        f"Model: {self.model}"
                    )
                
                # Extract token usage from response
                token_usage = self._extract_token_usage(response)
                
                return SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": response.usage},
                    actual_queried_message_list=msgs,
                    token_usage=token_usage,
                )
            except openai.BadRequestError as e:
                _logger.warning(f"[{self._log_tag}] Bad Request Error: {e}")
                raise RuntimeError(f"Kimi API BadRequestError: {e}") from e
            except openai.RateLimitError as e:
                if trial >= self.max_retries:
                    _logger.warning(f"[{self._log_tag}] Max retries ({self.max_retries}) exceeded due to rate limit: {e}")
                    raise RuntimeError(
                        f"Kimi API rate limit error after {self.max_retries} retries: {e}"
                    ) from e
                # Exponential backoff with jitter
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
                        f"Kimi API connection/timeout after {self.max_retries} retries: {e}"
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
                        f"Kimi API error after {self.max_retries} retries: {e}"
                    ) from e
                # Exponential backoff with jitter
                base_backoff = 2**trial
                jitter = random.uniform(0, base_backoff * 0.5)
                exception_backoff = base_backoff + jitter
                _logger.debug(f"[{self._log_tag}] API error, retrying {trial} after {exception_backoff:.1f}s: {type(e).__name__}: {e}")
                await asyncio.sleep(exception_backoff)
                trial += 1

