import os
import asyncio
from typing import Any, Optional

import openai
from openai import AsyncOpenAI
import dotenv

from libs.types import MessageList, SamplerBase, SamplerResponse


class DeepSeekSampler(SamplerBase):
    """
    Sample from DeepSeek's chat completion API
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        system_message: Optional[str] = None,
        temperature: float = 1.0,
        thinking: bool = False,
        reasoning_effort: Optional[str] = None,
        max_retries: int = 5,
    ):
        self.api_key_name = "DEEPSEEK_API_KEY"
        assert os.environ.get("DEEPSEEK_API_KEY"), "Please set DEEPSEEK_API_KEY"
        self.client = AsyncOpenAI(
            base_url="https://api.deepseek.com",
            api_key=os.getenv("DEEPSEEK_API_KEY")
        )
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    async def __call__(self, message_list: MessageList) -> SamplerResponse:
        # Add system message if provided
        msgs = list(message_list)
        if self.system_message:
            msgs.insert(0, self._pack_message("system", self.system_message))
        
        trial = 0

        while True:
            try:
                kwargs = {
                    "model": self.model,
                    "messages": msgs,
                }
                if self.thinking:
                    kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
                    if self.reasoning_effort:
                        kwargs["reasoning_effort"] = self.reasoning_effort
                else:
                    kwargs["temperature"] = self.temperature
                response = await self.client.chat.completions.create(**kwargs)
                
                content = response.choices[0].message.content or ""
                return SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": response.usage},
                    actual_queried_message_list=msgs,
                )
            except openai.BadRequestError as e:
                print(f"Bad Request Error: {e}")
                raise RuntimeError(f"DeepSeek API BadRequestError: {e}") from e
            except Exception as e:
                if trial >= self.max_retries:
                    print(f"Max retries ({self.max_retries}) exceeded: {e}")
                    raise RuntimeError(
                        f"DeepSeek API error after {self.max_retries} retries: {e}"
                    ) from e
                exception_backoff = 2**trial  # exponential back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                await asyncio.sleep(exception_backoff)
                trial += 1 