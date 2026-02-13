"""GLM sampler - Zhipu GLM models via Modal Direct (OpenAI-compatible).

Uses https://api.us-west-2.modal.direct/v1/chat/completions.
Set MODAL_GLM_API_KEY to your Modal bearer token.
Set thinking=True for reasoning mode (extra_body; may be ignored by Modal).
"""

import logging
import os
import asyncio
import random
from typing import Any, AsyncIterator, Optional

import openai
from openai import AsyncOpenAI
import httpx
import dotenv

from libs.types import MessageList, SamplerBase, SamplerResponse

dotenv.load_dotenv()

_logger = logging.getLogger(__name__)

# Modal Direct API (OpenAI-compatible chat completions)
MODAL_GLM_BASE = "https://api.us-west-2.modal.direct/v1"

# Shared client for GLM (connection pooling)
_shared_glm_client: AsyncOpenAI | None = None


def get_shared_glm_client(max_connections: int = 50) -> AsyncOpenAI:
    """Get or create the shared AsyncOpenAI client for Modal GLM API."""
    global _shared_glm_client
    if _shared_glm_client is None:
        api_key = os.getenv("MODAL_GLM_API_KEY")
        if not api_key:
            raise ValueError("Please set MODAL_GLM_API_KEY environment variable")

        http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections // 2,
            ),
            timeout=httpx.Timeout(300.0, connect=60.0),
            http1=True,
            http2=False,
        )
        _shared_glm_client = AsyncOpenAI(
            base_url=MODAL_GLM_BASE,
            api_key=api_key,
            timeout=300.0,
            max_retries=0,
            http_client=http_client,
        )
        _logger.debug("Created shared GLM (Modal) client")
    return _shared_glm_client


class GlmSampler(SamplerBase):
    """
    Sample from Zhipu GLM models via Modal Direct API.

    OpenAI-compatible endpoint. Thinking mode via extra_body (if supported).
    """

    def __init__(
        self,
        model: str = "zai-org/GLM-5-FP8",
        system_message: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_retries: int = 5,
        thinking: bool = False,
    ):
        """
        Initialize the GLM sampler.

        Args:
            model: Model id (e.g. "zai-org/GLM-5-FP8")
            system_message: Optional system message to prepend
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            max_retries: Number of retries on transient errors
            thinking: If True, send extra_body.thinking (may be ignored by Modal)
        """
        self.api_key_name = "MODAL_GLM_API_KEY"
        assert os.environ.get("MODAL_GLM_API_KEY"), "Please set MODAL_GLM_API_KEY"
        self.client = get_shared_glm_client()
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.thinking = thinking
        tag_parts = [model]
        if thinking:
            tag_parts.append("thinking")
        self._log_tag = "-".join(tag_parts)

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        return {"role": str(role), "content": content}

    def _extract_token_usage(self, response: Any) -> dict[str, int]:
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
        """Drop empty assistant messages so the API does not reject the request."""
        sanitized = []
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "")
            if role == "assistant" and (not content or not str(content).strip()):
                _logger.warning(
                    f"[{self._log_tag}] Filtering empty assistant message from history"
                )
                continue
            sanitized.append(msg)
        return sanitized

    async def stream(
        self,
        message_list: MessageList,
        *,
        tool_stream: bool = False,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Stream chat completion from Z.AI.

        Yields chunks with optional keys:
          - "content_delta": str – incremental response text
          - "reasoning_delta": str – incremental reasoning (thinking mode)
          - "tool_calls_delta": list – streaming tool call info (when tool_stream=True)
          - "usage": dict – token usage (on final chunk when available)

        See: https://docs.z.ai/guides/tools/stream-tool#stream-tool-call
        """
        msgs = list(message_list)
        if self.system_message:
            msgs.insert(0, self._pack_message("system", self.system_message))
        msgs = self._sanitize_messages(msgs)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
            "extra_body": {
                "thinking": {"type": "enabled" if self.thinking else "disabled"}
            },
        }
        if tool_stream and tools:
            kwargs["tool_stream"] = True
            kwargs["tools"] = tools

        response = await self.client.chat.completions.create(**kwargs)
        last_usage = None

        async for chunk in response:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = getattr(choice, "delta", None)
            out: dict[str, Any] = {}
            if delta is not None:
                if hasattr(delta, "reasoning_content") and getattr(delta, "reasoning_content"):
                    out["reasoning_delta"] = delta.reasoning_content or ""
                if getattr(delta, "content", None):
                    out["content_delta"] = delta.content or ""
                if hasattr(delta, "tool_calls") and getattr(delta, "tool_calls"):
                    out["tool_calls_delta"] = list(delta.tool_calls) if delta.tool_calls else []
            if hasattr(chunk, "usage") and getattr(chunk, "usage", None):
                last_usage = chunk.usage
                out["usage"] = self._extract_token_usage(chunk.usage)
            if out:
                yield out

    async def __call__(self, message_list: MessageList) -> SamplerResponse:
        msgs = list(message_list)
        if self.system_message:
            msgs.insert(0, self._pack_message("system", self.system_message))
        msgs = self._sanitize_messages(msgs)

        trial = 0
        while True:
            try:
                await asyncio.sleep(random.uniform(0, 0.2))

                kwargs = {
                    "model": self.model,
                    "messages": msgs,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                kwargs["extra_body"] = {
                    "thinking": {"type": "enabled" if self.thinking else "disabled"}
                }

                response = await self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""

                if not content.strip():
                    raise RuntimeError(
                        f"GLM (Modal) returned empty response. Model: {self.model}"
                    )

                token_usage = self._extract_token_usage(response)
                return SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": response.usage},
                    actual_queried_message_list=msgs,
                    token_usage=token_usage,
                )
            except openai.BadRequestError as e:
                _logger.warning(f"[{self._log_tag}] Bad Request Error: {e}")
                raise RuntimeError(f"GLM API BadRequestError: {e}") from e
            except openai.RateLimitError as e:
                if trial >= self.max_retries:
                    raise RuntimeError(
                        f"GLM API rate limit after {self.max_retries} retries: {e}"
                    ) from e
                base_backoff = 2**trial
                jitter = random.uniform(0, base_backoff * 0.5)
                _logger.debug(
                    f"[{self._log_tag}] Rate limit, retry {trial} after {base_backoff + jitter:.1f}s: {e}"
                )
                await asyncio.sleep(base_backoff + jitter)
                trial += 1
            except (
                openai.APITimeoutError,
                asyncio.TimeoutError,
                openai.APIConnectionError,
            ) as e:
                if trial >= self.max_retries:
                    raise RuntimeError(
                        f"GLM API connection/timeout after {self.max_retries} retries: {e}"
                    ) from e
                base_backoff = 2**trial
                jitter = random.uniform(0, base_backoff * 0.5)
                _logger.debug(
                    f"[{self._log_tag}] Connection/timeout, retry {trial} after {base_backoff + jitter:.1f}s: {e}"
                )
                await asyncio.sleep(base_backoff + jitter)
                trial += 1
            except Exception as e:
                if trial >= self.max_retries:
                    raise RuntimeError(
                        f"GLM API error after {self.max_retries} retries: {e}"
                    ) from e
                base_backoff = 2**trial
                jitter = random.uniform(0, base_backoff * 0.5)
                _logger.debug(
                    f"[{self._log_tag}] API error, retry {trial} after {base_backoff + jitter:.1f}s: {type(e).__name__}: {e}"
                )
                await asyncio.sleep(base_backoff + jitter)
                trial += 1
