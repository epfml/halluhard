"""Kimi K2 / K2.5 sampler - async wrapper for Moonshot AI's Kimi API.

Kimi uses an OpenAI-compatible API (base_url=https://api.moonshot.ai/v1).
For kimi-k2.5 with thinking enabled (default), the API returns reasoning_content
(thinking) then content; use temperature=1.0 and max_tokens >= 16000 per docs.
See: https://platform.moonshot.ai/docs/guide/use-kimi-k2-thinking-model
     https://platform.moonshot.ai/docs/guide/kimi-k2-quickstart
"""

import json
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


# Per Moonshot docs: thinking models need max_tokens >= 16000 and temperature=1.0 for k2.5
_DEFAULT_MAX_TOKENS_K25 = 16384
_DEFAULT_MAX_TOKENS_OTHER = 8192
_K25_TEMPERATURE = 1.0

# Built-in web search: declare in tools; when model returns tool_calls, we resubmit arguments as-is.
# See: https://platform.moonshot.ai/docs/guide/use-web-search
_WEB_SEARCH_TOOLS = [
    {
        "type": "builtin_function",
        "function": {"name": "$web_search"},
    }
]

# When web search is enabled, ask the model to present URLs as inline clickable markdown links.
_WEB_SEARCH_CITATION_INSTRUCTION = (
    "When citing web sources, present each URL as an inline clickable markdown link at the point of use, "
    "e.g. [Title](URL) or [Author. Title](URL). Do not list URLs only in a separate references block."
)

# Maximum number of tool call iterations to prevent infinite loops
_MAX_TOOL_ITERATIONS = 10


class KimiSampler(SamplerBase):
    """
    Sample from Moonshot AI's Kimi chat completion API.

    Supports kimi-k2.5 (thinking enabled by default) and kimi-k2-thinking / older models.
    For k2.5: use temperature=1.0 and max_tokens >= 16000 per official guide.
    See: https://platform.moonshot.ai/docs/guide/use-kimi-k2-thinking-model
    """

    def __init__(
        self,
        model: str = "kimi-k2.5",
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: int = 5,
        record_thinking: bool = False,
        web_search: bool = False,
    ):
        """
        Initialize the Kimi sampler.

        Args:
            model: Model name (e.g., "kimi-k2.5", "kimi-k2-thinking", "kimi-k2-0711-preview")
            system_message: Optional system message to prepend
            temperature: Sampling temperature. Thinking models should use 1.0.
            max_tokens: Max tokens (reasoning + content). For thinking models use >= 16000.
            max_retries: Number of retries on transient errors
            record_thinking: If True, include reasoning_content in response_metadata["reasoning"]. Default False.
            web_search: If True, enable built-in $web_search tool.
        """
        self.api_key_name = "MOONSHOT_API_KEY"
        assert os.environ.get("MOONSHOT_API_KEY"), "Please set MOONSHOT_API_KEY"
        self.client = get_shared_kimi_client()
        self.model = model
        self.system_message = system_message
        is_k25 = "k2.5" in model.lower()
        if temperature is not None:
            self.temperature = temperature
        else:
            self.temperature = _K25_TEMPERATURE if is_k25 else 0.0
        self.max_tokens = max_tokens if max_tokens is not None else (
            _DEFAULT_MAX_TOKENS_K25 if is_k25 else _DEFAULT_MAX_TOKENS_OTHER
        )
        self.max_retries = max_retries
        self.record_thinking = record_thinking
        self.web_search = web_search

        # Build a descriptive tag for logging
        self._log_tag = model

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        return {"role": str(role), "content": content}

    def _extract_token_usage(self, response_or_usage: Any) -> dict[str, int]:
        """Extract token usage from OpenAI-compatible API response or usage object.

        Args:
            response_or_usage: Either a completion response (with .usage) or a usage object
                              (e.g. from streaming chunk.usage) with prompt_tokens, etc.

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
        usage = getattr(response_or_usage, "usage", None)
        if usage is None and hasattr(response_or_usage, "prompt_tokens"):
            usage = response_or_usage
        if usage:
            token_usage["input_tokens"] = getattr(usage, "prompt_tokens", 0)
            token_usage["output_tokens"] = getattr(usage, "completion_tokens", 0)
            token_usage["total_tokens"] = getattr(usage, "total_tokens", 0)
            token_usage["reasoning_tokens"] = getattr(usage, "reasoning_tokens", 0)
        return token_usage

    def _sanitize_messages(self, messages: list, preserve_reasoning: bool = False) -> list:
        """Sanitize messages to ensure compatibility with Kimi API.

        Kimi API strictly requires non-empty content for assistant messages.
        
        Args:
            messages: List of messages to sanitize
            preserve_reasoning: If False (default), strip reasoning_content from historical messages
                              to save tokens. If True, preserve existing reasoning_content.
        """
        sanitized = []
        for idx, msg in enumerate(messages):
            msg = dict(msg)
            content = msg.get("content", "")
            role = msg.get("role", "")

            # Skip empty assistant messages (Kimi strictly rejects these)
            if role == "assistant" and (not content or not str(content).strip()):
                if not msg.get("tool_calls"):
                    _logger.warning(
                        f"[{self._log_tag}] Filtering empty assistant message at index {idx} from history"
                    )
                    continue

            # fixing error: Error generating conversation for 'Should the Federal Communications Com
            # mission’s det...': Kimi API BadRequestError: Error code: 400 - {'error': {'m
            # essage': 'thinking is enabled but reasoning_content is missing in assistant 
            # tool call message at index
            # API 400: every assistant message with tool_calls must have non-empty reasoning_content.
            if role == "assistant" and msg.get("tool_calls"):
                rc = msg.get("reasoning_content") or ""
                msg["reasoning_content"] = rc.strip() if rc.strip() else " "
            elif role == "assistant" and not preserve_reasoning:
                msg.pop("reasoning_content", None)

            sanitized.append(msg)
        return sanitized

    def _accumulate_tool_calls(
        self, tool_calls_acc: list[dict], delta_tool_calls: Any
    ) -> None:
        """Merge streamed delta.tool_calls into tool_calls_acc by index."""
        if not delta_tool_calls:
            return
        for d in delta_tool_calls:
            idx = getattr(d, "index", len(tool_calls_acc))
            while len(tool_calls_acc) <= idx:
                tool_calls_acc.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
            acc = tool_calls_acc[idx]
            if getattr(d, "id", None):
                acc["id"] = d.id or acc["id"]
            if getattr(d, "type", None):
                acc["type"] = d.type or acc["type"]
            fn = getattr(d, "function", None)
            if fn:
                if getattr(fn, "name", None):
                    acc["function"]["name"] = (acc["function"]["name"] or "") + (fn.name or "")
                if getattr(fn, "arguments", None):
                    acc["function"]["arguments"] = (acc["function"]["arguments"] or "") + (fn.arguments or "")

    # Markers that indicate reasoning/thinking ends and the actual answer follows (Kimi sometimes sends both in content).
    _ANSWER_MARKERS = (
        "The answer is:",
        "Good.  Perfect.  Output.  I have completed the task.  Submitting.  Final.  The answer is:",
        "Good. Perfect. Output. I have completed the task. Submitting. Final. The answer is:",
        "Final answer:",
        "**Answer:**",
        "**The answer is:**",
    )

    @staticmethod
    def _strip_reasoning_from_content(content: str) -> str:
        """If the model put reasoning in content before a marker like 'The answer is:', return only the part after it."""
        if not content or not content.strip():
            return content
        # Try longest markers first so we strip the full reasoning prefix (e.g. "Good. Perfect. ... The answer is:").
        markers = sorted(KimiSampler._ANSWER_MARKERS, key=len, reverse=True)
        for marker in markers:
            idx = content.find(marker)
            if idx != -1:
                return content[idx + len(marker) :].lstrip()
        return content

    async def _call_with_web_search(self, msgs: list) -> tuple[str, Any, list[str], dict[str, int]]:
        """Run chat with $web_search tool; loop until finish_reason is stop. Uses streaming.
        
        Per official Moonshot docs, the tool response should simply echo back the arguments;
        the API handles the actual web search internally.
        See: https://docs.aimlapi.com/api-references/text-models-llm/moonshot/kimi-k2-preview
        """
        total_usage = {
            "input_tokens": 0, 
            "output_tokens": 0, 
            "total_tokens": 0, 
            "cached_tokens": 0, 
            "reasoning_tokens": 0
        }
        all_reasoning_parts: list[str] = []
        messages = list(msgs)
        last_usage = None
        iteration = 0

        while iteration < _MAX_TOOL_ITERATIONS:
            iteration += 1
            reasoning_parts: list[str] = []
            
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                stream=True,
                temperature=self.temperature,
                tools=_WEB_SEARCH_TOOLS,
                stream_options={"include_usage": True},  # Ensure usage data is included in stream
            )
            
            content_parts: list[str] = []
            tool_calls_acc: list[dict] = []
            finish_reason = "stop"
            last_choice = None

            async for chunk in stream:
                if chunk.choices:
                    choice = chunk.choices[0]
                    last_choice = choice
                    delta = getattr(choice, "delta", None)
                    if delta is not None:
                        if hasattr(delta, "reasoning_content") and getattr(delta, "reasoning_content"):
                            reasoning_parts.append(getattr(delta, "reasoning_content"))
                        if getattr(delta, "content", None):
                            content_parts.append(delta.content or "")
                        if hasattr(delta, "tool_calls") and getattr(delta, "tool_calls"):
                            self._accumulate_tool_calls(tool_calls_acc, delta.tool_calls)
                    if getattr(choice, "finish_reason", None):
                        finish_reason = choice.finish_reason
                if hasattr(chunk, "usage") and getattr(chunk, "usage", None):
                    last_usage = chunk.usage

            # Some APIs send tool_calls only on the last chunk (choice.message.tool_calls)
            if finish_reason == "tool_calls" and not tool_calls_acc and last_choice is not None:
                msg = getattr(last_choice, "message", None)
                if msg is not None:
                    tcs = getattr(msg, "tool_calls", None)
                    if tcs:
                        for tc in tcs:
                            fn = getattr(tc, "function", None)
                            tool_calls_acc.append({
                                "id": getattr(tc, "id", "") or "",
                                "type": getattr(tc, "type", "function") or "function",
                                "function": {
                                    "name": getattr(fn, "name", "") if fn else "",
                                    "arguments": getattr(fn, "arguments", "{}") if fn else "{}",
                                },
                            })

            if last_usage:
                total_usage["input_tokens"] += getattr(last_usage, "prompt_tokens", 0)
                total_usage["output_tokens"] += getattr(last_usage, "completion_tokens", 0)
                total_usage["total_tokens"] += getattr(last_usage, "total_tokens", 0)
                total_usage["reasoning_tokens"] += getattr(last_usage, "reasoning_tokens", 0)

            content = "".join(content_parts)
            reasoning_content = "".join(reasoning_parts)
            if reasoning_parts:
                all_reasoning_parts.extend(reasoning_parts)
            
            if finish_reason != "tool_calls" or not tool_calls_acc:
                if iteration >= _MAX_TOOL_ITERATIONS:
                    _logger.warning(
                        f"[{self._log_tag}] Web search reached max iterations ({_MAX_TOOL_ITERATIONS})"
                    )
                return content, last_usage, all_reasoning_parts, total_usage

            # Build assistant message for next round. API requires non-empty reasoning_content when tool_calls present.
            if not reasoning_content.strip():
                reasoning_content = " "
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": content or None,
                "reasoning_content": reasoning_content,
            }
            if tool_calls_acc:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.get("id") or "",
                        "type": tc.get("type") or "function",
                        "function": {
                            "name": (tc.get("function") or {}).get("name") or "",
                            "arguments": (tc.get("function") or {}).get("arguments") or "{}",
                        },
                    }
                    for tc in tool_calls_acc
                ]
            messages.append(assistant_msg)

            # Per official Moonshot docs: simply echo back the arguments
            # The API handles the actual web search internally
            for tc in tool_calls_acc:
                name = (tc.get("function") or {}).get("name") or ""
                args_str = (tc.get("function") or {}).get("arguments") or "{}"
                try:
                    arguments = json.loads(args_str)
                except json.JSONDecodeError:
                    arguments = {}
                
                if name == "$web_search":
                    tool_result = arguments  # API handles the actual search
                else:
                    tool_result = {"error": f"Unknown tool: {name}"}
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id") or "",
                    "name": name,
                    "content": json.dumps(tool_result, ensure_ascii=False),  # Handle Unicode properly
                })
        
        # If we exit the loop due to max iterations, log warning and return what we have
        _logger.warning(
            f"[{self._log_tag}] Web search exceeded max iterations ({_MAX_TOOL_ITERATIONS}), returning partial result"
        )
        return content, last_usage, all_reasoning_parts, total_usage

    async def __call__(self, message_list: MessageList) -> SamplerResponse:
        # First sanitize incoming messages with preserve_reasoning=False to strip historical reasoning
        # This saves tokens and avoids context pollution from past turns
        msgs = self._sanitize_messages(list(message_list), preserve_reasoning=False)
        
        # Add system message if provided
        system_parts = []
        if self.system_message:
            system_parts.append(self.system_message)
        if self.web_search:
            system_parts.append(_WEB_SEARCH_CITATION_INSTRUCTION)
        if system_parts:
            msgs.insert(0, self._pack_message("system", "\n\n".join(system_parts)))

        trial = 0

        while True:
            try:
                await asyncio.sleep(random.uniform(0, 0.2))

                if self.web_search:
                    content, usage, reasoning_parts, token_usage = await self._call_with_web_search(msgs)
                else:
                    stream = await self.client.chat.completions.create(
                        model=self.model,
                        messages=msgs,
                        max_tokens=self.max_tokens,
                        stream=True,
                        temperature=self.temperature,
                    )
                    content_parts: list[str] = []
                    reasoning_parts = []
                    usage = None
                    async for chunk in stream:
                        if chunk.choices:
                            choice = chunk.choices[0]
                            delta = getattr(choice, "delta", None)
                            if delta is not None:
                                if hasattr(delta, "reasoning_content") and getattr(delta, "reasoning_content"):
                                    reasoning_parts.append(getattr(delta, "reasoning_content"))
                                if getattr(delta, "content", None):
                                    content_parts.append(delta.content or "")
                        if hasattr(chunk, "usage") and getattr(chunk, "usage", None):
                            usage = chunk.usage
                    content = "".join(content_parts)
                    token_usage = self._extract_token_usage(usage) if usage else {
                        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                        "cached_tokens": 0, "reasoning_tokens": 0,
                    }

                # Kimi sometimes sends reasoning in content; strip everything before "The answer is:" (or similar).
                content = self._strip_reasoning_from_content(content)

                if not content.strip():
                    raise RuntimeError(
                        f"Kimi returned empty response (likely content filtered). Model: {self.model}"
                    )

                # Response structure: API returns streaming chunks with choice.delta.content (and
                # optionally delta.reasoning_content). We accumulate into a single string. We return
                # SamplerResponse(response_text=content, response_metadata={usage, [reasoning]}, token_usage).
                # Optional: print raw model response to stdout (e.g. KIMI_RAW_RESPONSE=1 python ...)
                if os.environ.get("KIMI_RAW_RESPONSE", "").strip() == "1":
                    print("=== Kimi raw model response (response_text) ===")
                    print(content)
                    print("=== end raw response ===")

                _logger.debug(
                    "[%s] raw model response (length=%d): %.200s%s",
                    self._log_tag,
                    len(content),
                    content,
                    "..." if len(content) > 200 else "",
                )

                response_metadata: dict[str, Any] = {"usage": usage}
                if self.record_thinking and reasoning_parts:
                    response_metadata["reasoning"] = "".join(reasoning_parts)

                return SamplerResponse(
                    response_text=content,
                    response_metadata=response_metadata,
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