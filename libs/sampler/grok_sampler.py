"""xAI Grok sampler - async wrapper for xAI's Grok API.

xAI provides an OpenAI-compatible API (base_url=https://api.x.ai/v1).
See: https://docs.x.ai/developers/quickstart
"""

import asyncio
import logging
import os
import random
from typing import Any, Optional

import httpx
import openai
from openai import AsyncOpenAI

import dotenv

from libs.types import MessageList, SamplerBase, SamplerResponse

dotenv.load_dotenv()

_logger = logging.getLogger(__name__)

# Shared xAI client for all samplers (connection pooling)
_shared_grok_client: AsyncOpenAI | None = None

# xAI recommends longer timeout for reasoning models (e.g. 3600s)
_DEFAULT_TIMEOUT = 3600.0


def get_shared_grok_client(max_connections: int = 50) -> AsyncOpenAI:
    """Get or create the shared AsyncOpenAI client for xAI Grok API."""
    global _shared_grok_client
    if _shared_grok_client is None:
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("Please set XAI_API_KEY environment variable")

        http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections // 2,
            ),
            timeout=httpx.Timeout(_DEFAULT_TIMEOUT, connect=60.0),
            http1=True,
            http2=False,
        )
        _shared_grok_client = AsyncOpenAI(
            base_url="https://api.x.ai/v1",
            api_key=api_key,
            timeout=_DEFAULT_TIMEOUT,
            max_retries=0,
            http_client=http_client,
        )
        _logger.debug("Created shared xAI Grok client")
    return _shared_grok_client


class GrokSampler(SamplerBase):
    """
    Sample from xAI's Grok API (OpenAI-compatible Responses API).

    See: https://docs.x.ai/developers/quickstart
    """

    def __init__(
        self,
        model: str = "grok-4-1-fast-reasoning",
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 8192,
        max_retries: int = 5,
        timeout: Optional[float] = None,
    ):
        """
        Initialize the Grok sampler.

        Args:
            model: Model name (e.g. "grok-4-1-fast-reasoning", "grok-4").
            system_message: Optional system message to prepend.
            temperature: Sampling temperature.
            max_tokens: Max output tokens.
            max_retries: Number of retries on transient errors.
            timeout: Request timeout in seconds (default 3600 for reasoning models).
        """
        self.api_key_name = "XAI_API_KEY"
        assert os.environ.get("XAI_API_KEY"), "Please set XAI_API_KEY"
        self.client = get_shared_grok_client()
        self.model = model
        self.system_message = system_message
        self.temperature = temperature if temperature is not None else 0.0
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.timeout = timeout if timeout is not None else _DEFAULT_TIMEOUT
        self._log_tag = model

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        return {"role": str(role), "content": content}

    def _extract_token_usage(self, response: Any) -> dict[str, int]:
        """Extract token usage from xAI/OpenAI-compatible response."""
        token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
        }
        usage = getattr(response, "usage", None)
        if usage:
            token_usage["input_tokens"] = getattr(usage, "input_tokens", 0) or getattr(usage, "prompt_tokens", 0)
            token_usage["output_tokens"] = getattr(usage, "output_tokens", 0) or getattr(usage, "completion_tokens", 0)
            token_usage["total_tokens"] = getattr(usage, "total_tokens", 0)
            input_details = getattr(usage, "input_tokens_details", None)
            output_details = getattr(usage, "output_tokens_details", None)
            if input_details:
                token_usage["cached_tokens"] = getattr(input_details, "cached_tokens", 0)
            if output_details:
                token_usage["reasoning_tokens"] = getattr(output_details, "reasoning_tokens", 0)
        return token_usage

    def _get_response_text(self, response: Any) -> str:
        """Get response text from Responses API output (output_text or output[].content)."""
        text = getattr(response, "output_text", None)
        if text is not None and isinstance(text, str):
            return text or ""
        output = getattr(response, "output", None)
        if output and len(output) > 0:
            first = output[0]
            content = getattr(first, "content", None)
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    getattr(part, "text", part) if hasattr(part, "text") else str(part)
                    for part in content
                )
        return ""

    async def __call__(self, message_list: MessageList) -> SamplerResponse:
        msgs = list(message_list)
        if self.system_message:
            msgs.insert(0, self._pack_message("system", self.system_message))

        trial = 0
        while True:
            try:
                await asyncio.sleep(random.uniform(0, 0.2))

                kwargs = {
                    "model": self.model,
                    "input": msgs,
                }
                if self.temperature is not None:
                    kwargs["temperature"] = self.temperature
                if self.max_tokens is not None:
                    kwargs["max_output_tokens"] = self.max_tokens

                response = await self.client.responses.create(**kwargs)

                response_text = self._get_response_text(response)
                status = getattr(response, "status", None)
                if status == "incomplete":
                    if getattr(response, "incomplete_details", None):
                        _logger.warning(
                            f"[{self._log_tag}] Response incomplete: {response.incomplete_details}"
                        )
                    response_text = response_text or ""

                token_usage = self._extract_token_usage(response)

                return SamplerResponse(
                    response_text=response_text,
                    response_metadata={"usage": getattr(response, "usage", None), "status": status},
                    actual_queried_message_list=msgs,
                    token_usage=token_usage,
                )
            except openai.BadRequestError as e:
                _logger.warning(f"[{self._log_tag}] Bad Request Error: {e}")
                raise RuntimeError(f"xAI API BadRequestError: {e}") from e
            except openai.RateLimitError as e:
                if trial >= self.max_retries:
                    raise RuntimeError(
                        f"xAI API rate limit after {self.max_retries} retries: {e}"
                    ) from e
                backoff = 2**trial + random.uniform(0, 2**trial * 0.5)
                _logger.debug(f"[{self._log_tag}] Rate limit, retry {trial} in {backoff:.1f}s: {e}")
                await asyncio.sleep(backoff)
                trial += 1
            except (openai.APITimeoutError, asyncio.TimeoutError, openai.APIConnectionError) as e:
                if trial >= self.max_retries:
                    raise RuntimeError(
                        f"xAI API connection/timeout after {self.max_retries} retries: {e}"
                    ) from e
                backoff = 2**trial + random.uniform(0, 2**trial * 0.5)
                _logger.debug(f"[{self._log_tag}] Connection/timeout, retry {trial} in {backoff:.1f}s: {e}")
                await asyncio.sleep(backoff)
                trial += 1
            except Exception as e:
                if trial >= self.max_retries:
                    raise RuntimeError(
                        f"xAI API error after {self.max_retries} retries: {e}"
                    ) from e
                backoff = 2**trial + random.uniform(0, 2**trial * 0.5)
                _logger.debug(f"[{self._log_tag}] Error, retry {trial} in {backoff:.1f}s: {type(e).__name__}: {e}")
                await asyncio.sleep(backoff)
                trial += 1
