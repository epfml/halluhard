"""Sampler for NVIDIA's internal inference gateway (inference.nvidia.com).

The catalog is browsable at https://inference.nvidia.com; the programmatic API
is an OpenAI-compatible LiteLLM proxy at https://inference-api.nvidia.com/v1.
Authenticates with an NVIDIA_INFERENCE_API_KEY (e.g. "sk-...").

Model ids are the catalog "label" strings, e.g. "us/azure/openai/eccn-gpt-5-mini".
Note that several catalog models (e.g. the GPT-5 family) are reasoning models:
a small token budget is consumed entirely by reasoning and yields empty content,
so max_tokens defaults high.
"""

import logging
import os
import asyncio
import random
from typing import Any, Optional

import httpx
import openai
from openai import AsyncOpenAI

from libs.types import MessageList, SamplerBase, SamplerResponse
from libs.sampler.openai_sampler import ResponsesSampler

import dotenv

dotenv.load_dotenv()

_logger = logging.getLogger(__name__)

NVIDIA_INFERENCE_BASE_URL = "https://inference-api.nvidia.com/v1"

# Shared client for all NVIDIA inference samplers (connection pooling)
_shared_nvidia_inference_client: AsyncOpenAI | None = None


def get_shared_nvidia_inference_client(max_connections: int = 50) -> AsyncOpenAI:
    """Get or create the shared AsyncOpenAI client pointed at the NVIDIA gateway."""
    global _shared_nvidia_inference_client
    if _shared_nvidia_inference_client is None:
        http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections // 2,
            ),
            timeout=httpx.Timeout(300.0, connect=60.0),
            http1=True,
            http2=False,
        )
        _shared_nvidia_inference_client = AsyncOpenAI(
            base_url=NVIDIA_INFERENCE_BASE_URL,
            api_key=os.getenv("NVIDIA_INFERENCE_API_KEY"),
            timeout=300.0,
            max_retries=0,  # Sampler handles retries itself with jitter
            http_client=http_client,
        )
        _logger.debug(
            f"Created shared NVIDIA inference client (base_url={NVIDIA_INFERENCE_BASE_URL}, "
            f"max_connections={max_connections})"
        )
    return _shared_nvidia_inference_client


class NvidiaInferenceSampler(SamplerBase):
    """
    Sample from NVIDIA's internal inference gateway via its OpenAI-compatible
    (LiteLLM) chat completions API.
    """

    def __init__(
        self,
        model: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
        max_retries: int = 10,
    ):
        self.api_key_name = "NVIDIA_INFERENCE_API_KEY"
        assert os.environ.get(
            "NVIDIA_INFERENCE_API_KEY"
        ), "Please set NVIDIA_INFERENCE_API_KEY"
        self.client = get_shared_nvidia_inference_client()
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._log_tag = model

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        return {"role": str(role), "content": content}

    def _extract_token_usage(self, response: Any) -> dict[str, int]:
        """Extract token usage from a chat completions response."""
        token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
        }
        usage = getattr(response, "usage", None)
        if usage:
            token_usage["input_tokens"] = getattr(usage, "prompt_tokens", 0) or 0
            token_usage["output_tokens"] = getattr(usage, "completion_tokens", 0) or 0
            token_usage["total_tokens"] = getattr(usage, "total_tokens", 0) or 0

            output_details = getattr(usage, "completion_tokens_details", None)
            input_details = getattr(usage, "prompt_tokens_details", None)
            if output_details:
                token_usage["reasoning_tokens"] = (
                    getattr(output_details, "reasoning_tokens", 0) or 0
                )
            if input_details:
                token_usage["cached_tokens"] = (
                    getattr(input_details, "cached_tokens", 0) or 0
                )
        return token_usage

    async def __call__(self, message_list: MessageList) -> SamplerResponse:
        msgs = list(message_list)
        if self.system_message:
            msgs.insert(0, self._pack_message("system", self.system_message))

        trial = 0
        while True:
            try:
                # Random jitter before request to spread out bursts
                await asyncio.sleep(random.uniform(0, 0.2))

                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": msgs,
                    # Reasoning models on this gateway require max_completion_tokens;
                    # max_tokens is deprecated/ignored for them.
                    "max_completion_tokens": self.max_tokens,
                }
                if self.temperature is not None:
                    kwargs["temperature"] = self.temperature

                response = await self.client.chat.completions.create(**kwargs)

                content = response.choices[0].message.content or ""
                finish_reason = response.choices[0].finish_reason
                if not content and finish_reason == "length":
                    _logger.warning(
                        f"[{self._log_tag}] Empty content with finish_reason=length; "
                        f"max_tokens ({self.max_tokens}) was likely consumed by reasoning."
                    )

                token_usage = self._extract_token_usage(response)
                return SamplerResponse(
                    response_text=content,
                    response_metadata={
                        "usage": response.usage,
                        "finish_reason": finish_reason,
                    },
                    actual_queried_message_list=msgs,
                    token_usage=token_usage,
                )
            except openai.BadRequestError as e:
                _logger.warning(f"[{self._log_tag}] Bad Request Error: {e}")
                raise RuntimeError(
                    f"NVIDIA inference API BadRequestError: {e}"
                ) from e
            except openai.RateLimitError as e:
                if trial >= self.max_retries:
                    raise RuntimeError(
                        f"NVIDIA inference API rate limit error after {self.max_retries} retries: {e}"
                    ) from e
                base_backoff = 2**trial
                exception_backoff = base_backoff + random.uniform(0, base_backoff * 0.5)
                _logger.debug(
                    f"[{self._log_tag}] Rate limit, retry {trial} after {exception_backoff:.1f}s: {e}"
                )
                await asyncio.sleep(exception_backoff)
                trial += 1
            except (
                openai.APITimeoutError,
                asyncio.TimeoutError,
                openai.APIConnectionError,
            ) as e:
                if trial >= self.max_retries:
                    raise RuntimeError(
                        f"NVIDIA inference API connection/timeout after {self.max_retries} retries: {e}"
                    ) from e
                base_backoff = 2**trial
                exception_backoff = base_backoff + random.uniform(0, base_backoff * 0.5)
                _logger.debug(
                    f"[{self._log_tag}] Connection/timeout, retry {trial} after {exception_backoff:.1f}s: {e}"
                )
                await asyncio.sleep(exception_backoff)
                trial += 1
            except Exception as e:
                if trial >= self.max_retries:
                    raise RuntimeError(
                        f"NVIDIA inference API error after {self.max_retries} retries: {e}"
                    ) from e
                base_backoff = 2**trial
                exception_backoff = base_backoff + random.uniform(0, base_backoff * 0.5)
                _logger.debug(
                    f"[{self._log_tag}] API error, retry {trial} after {exception_backoff:.1f}s: {type(e).__name__}: {e}"
                )
                await asyncio.sleep(exception_backoff)
                trial += 1


class NvidiaInferenceResponsesSampler(ResponsesSampler):
    """Sample from the NVIDIA inference gateway via its OpenAI-compatible
    Responses API (/v1/responses).

    Identical behavior to ResponsesSampler (retries, parsing, web_search tool),
    but targets the gateway with NVIDIA_INFERENCE_API_KEY. The gateway only
    exposes web search through the Responses API — chat completions reject the
    web_search tool — so this is the sampler to use for web-grounded nvinfer
    models (e.g. the judge's web-grounding fallback).
    """

    def __init__(
        self,
        model: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        max_retries: int = 10,
        websearch: bool = False,
    ):
        # Note: we intentionally do NOT call super().__init__ — it asserts
        # OPENAI_API_KEY and binds the OpenAI client. We mirror its attributes
        # but swap in the shared NVIDIA gateway client/key.
        self.api_key_name = "NVIDIA_INFERENCE_API_KEY"
        assert os.environ.get(
            "NVIDIA_INFERENCE_API_KEY"
        ), "Please set NVIDIA_INFERENCE_API_KEY"
        self.client = get_shared_nvidia_inference_client()
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        self.websearch = websearch

        tag_parts = [model]
        if reasoning_effort:
            tag_parts.append(reasoning_effort)
        if websearch:
            tag_parts.append("websearch")
        self._log_tag = "-".join(tag_parts)
