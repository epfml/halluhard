import os
import asyncio
from typing import Any, Optional

import openai
from openai import AsyncOpenAI
import dotenv

from libs.types import MessageList, SamplerBase, SamplerResponse

dotenv.load_dotenv()


class NemotronSampler(SamplerBase):
    """
    Sample from NVIDIA's Nemotron models via the OpenAI-compatible
    NVIDIA NIM endpoint (https://integrate.api.nvidia.com/v1).

    Authenticates with an NGC API key (NGC_API_KEY, e.g. "nvapi-...").
    Reasoning Nemotron models support thinking via the `enable_thinking`
    and `reasoning_budget` extra-body parameters.
    """

    def __init__(
        self,
        model: str = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        system_message: Optional[str] = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 65536,
        enable_thinking: bool = True,
        reasoning_budget: Optional[int] = 16384,
        max_retries: int = 5,
    ):
        self.api_key_name = "NGC_API_KEY"
        assert os.environ.get("NGC_API_KEY"), "Please set NGC_API_KEY"
        self.client = AsyncOpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NGC_API_KEY"),
        )
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.reasoning_budget = reasoning_budget
        self.max_retries = max_retries

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    async def __call__(self, message_list: MessageList) -> SamplerResponse:
        # Add system message if provided
        msgs = list(message_list)
        if self.system_message:
            msgs.insert(0, self._pack_message("system", self.system_message))

        extra_body: dict[str, Any] = {}
        if self.enable_thinking:
            extra_body["enable_thinking"] = True
            if self.reasoning_budget is not None:
                extra_body["reasoning_budget"] = self.reasoning_budget

        trial = 0

        while True:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=msgs,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    extra_body=extra_body or None,
                )

                message = response.choices[0].message
                content = message.content or ""
                reasoning_content = getattr(message, "reasoning_content", None)

                usage = response.usage
                token_usage = {
                    "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                    "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(usage, "total_tokens", 0) or 0,
                    "cached_tokens": 0,
                    "reasoning_tokens": 0,
                }

                return SamplerResponse(
                    response_text=content,
                    response_metadata={
                        "usage": usage,
                        "reasoning_content": reasoning_content,
                    },
                    actual_queried_message_list=msgs,
                    token_usage=token_usage,
                )
            except openai.BadRequestError as e:
                print(f"Bad Request Error: {e}")
                raise RuntimeError(f"Nemotron API BadRequestError: {e}") from e
            except Exception as e:
                if trial >= self.max_retries:
                    print(f"Max retries ({self.max_retries}) exceeded: {e}")
                    raise RuntimeError(
                        f"Nemotron API error after {self.max_retries} retries: {e}"
                    ) from e
                exception_backoff = 2**trial  # exponential back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                await asyncio.sleep(exception_backoff)
                trial += 1
