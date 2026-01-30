"""Gemini 3 sampler using Google's GenAI SDK.

Based on: https://ai.google.dev/gemini-api/docs/gemini-3

Supports:
- Gemini 3 Pro and Flash models
- Thinking levels (low, medium, high, minimal)
- Google Search grounding
"""

import logging
import os
import asyncio
import random
from typing import Any, Optional

import dotenv

dotenv.load_dotenv()

_logger = logging.getLogger(__name__)

# Lazy import to avoid requiring google-genai if not used
_genai_client = None


def get_genai_client():
    """Get or create the shared Google GenAI client."""
    global _genai_client
    if _genai_client is None:
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "google-genai package not installed. "
                "Install with: pip install google-genai"
            )
        
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Please set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
        
        _genai_client = genai.Client(api_key=api_key)
        _logger.debug("Created Gemini GenAI client")
    
    return _genai_client


class GeminiSampler:
    """
    Sample from Google's Gemini 3 API.
    
    Supports thinking levels and Google Search grounding.
    """

    def __init__(
        self,
        model: str = "gemini-3-flash-preview",
        system_message: Optional[str] = None,
        temperature: float = 1.0,  # Gemini 3 default
        max_tokens: int = 8192,
        thinking_level: Optional[str] = None,  # low, medium, high, minimal (Flash only)
        max_retries: int = 5,
        websearch: bool = False,
    ):
        """
        Initialize the Gemini sampler.
        
        Args:
            model: Model name (e.g., "gemini-3-pro-preview", "gemini-3-flash-preview")
            system_message: Optional system instruction
            temperature: Sampling temperature (default 1.0 for Gemini 3)
            max_tokens: Maximum output tokens
            thinking_level: Thinking depth - "low", "medium", "high", or "minimal" (Flash only)
            max_retries: Number of retries on transient errors
            websearch: Enable Google Search grounding for real-time information
        """
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking_level = thinking_level
        self.max_retries = max_retries
        self.websearch = websearch
        
        # Validate thinking_level
        valid_levels = ["low", "medium", "high", "minimal", None]
        if thinking_level not in valid_levels:
            raise ValueError(f"thinking_level must be one of {valid_levels}")
        
        # minimal is only supported by Flash
        if thinking_level == "minimal" and "flash" not in model.lower():
            _logger.warning(f"thinking_level='minimal' is only supported by Flash models, not {model}")
        
        # Build a descriptive tag for logging
        tag_parts = [model]
        if thinking_level:
            tag_parts.append(thinking_level)
        if websearch:
            tag_parts.append("websearch")
        self._log_tag = "-".join(tag_parts)

    def _convert_messages(self, message_list: list) -> list:
        """Convert OpenAI-style messages to Gemini format."""
        contents = []
        
        for msg in message_list:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Map roles
            if role in ("system", "developer"):
                # System messages are handled separately in Gemini
                continue
            elif role == "assistant":
                role = "model"
            else:
                role = "user"
            
            # Handle content
            if isinstance(content, str):
                contents.append({
                    "role": role,
                    "parts": [{"text": content}]
                })
            elif isinstance(content, list):
                # Handle multimodal content
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            parts.append({"text": item.get("text", "")})
                        elif item.get("type") == "image_url":
                            # Handle image URLs if needed
                            parts.append({"text": f"[Image: {item.get('image_url', '')}]"})
                    elif isinstance(item, str):
                        parts.append({"text": item})
                if parts:
                    contents.append({"role": role, "parts": parts})
        
        return contents

    def _extract_system_message(self, message_list: list) -> Optional[str]:
        """Extract system message from message list."""
        system_parts = []
        
        for msg in message_list:
            role = msg.get("role", "")
            if role in ("system", "developer"):
                content = msg.get("content", "")
                if isinstance(content, str):
                    system_parts.append(content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            system_parts.append(item.get("text", ""))
                        elif isinstance(item, str):
                            system_parts.append(item)
        
        if system_parts:
            return "\n\n".join(system_parts)
        return self.system_message

    def _extract_token_usage(self, response: Any) -> dict:
        """Extract token usage from Gemini response."""
        token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        
        # Try to get usage metadata
        usage = getattr(response, "usage_metadata", None)
        if usage:
            token_usage["input_tokens"] = getattr(usage, "prompt_token_count", 0)
            token_usage["output_tokens"] = getattr(usage, "candidates_token_count", 0)
            token_usage["total_tokens"] = getattr(usage, "total_token_count", 0)
        
        return token_usage

    async def __call__(self, message_list: list):
        """Generate a response from Gemini.
        
        Args:
            message_list: List of messages in OpenAI format
            
        Returns:
            SamplerResponse-like object with response_text and token_usage
        """
        from google import genai
        from google.genai import types
        
        client = get_genai_client()
        
        # Extract system message and convert messages
        system_instruction = self._extract_system_message(message_list)
        contents = self._convert_messages(message_list)
        
        # Build config
        config_kwargs = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }
        
        # Add thinking config if specified
        if self.thinking_level:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=self.thinking_level
            )
        
        # Add tools if websearch enabled
        tools = None
        if self.websearch:
            tools = [{"google_search": {}}]
        
        trial = 0
        while True:
            try:
                # Random jitter before request
                await asyncio.sleep(random.uniform(0, 0.2))
                
                # Build request kwargs
                request_kwargs = {
                    "model": self.model,
                    "contents": contents,
                    "config": types.GenerateContentConfig(**config_kwargs),
                }
                
                if system_instruction:
                    request_kwargs["config"].system_instruction = system_instruction
                
                if tools:
                    request_kwargs["config"].tools = tools
                
                # Use async generate_content
                # Note: google-genai uses sync API, wrap in executor
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: client.models.generate_content(**request_kwargs)
                )
                
                # Extract response text
                response_text = ""
                if response.candidates:
                    candidate = response.candidates[0]
                    if candidate.content and candidate.content.parts:
                        text_parts = [
                            part.text for part in candidate.content.parts 
                            if hasattr(part, 'text') and part.text
                        ]
                        response_text = "".join(text_parts)
                
                # Extract token usage
                token_usage = self._extract_token_usage(response)
                
                # Return response in compatible format
                return _GeminiResponse(
                    response_text=response_text,
                    token_usage=token_usage,
                    raw_response=response,
                )
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check for rate limit errors
                if "rate" in error_str or "quota" in error_str or "429" in error_str:
                    if trial >= self.max_retries:
                        _logger.warning(f"[{self._log_tag}] Max retries ({self.max_retries}) exceeded due to rate limit: {e}")
                        raise RuntimeError(f"Gemini API rate limit after {self.max_retries} retries: {e}") from e
                    
                    base_backoff = 2 ** trial
                    jitter = random.uniform(0, base_backoff * 0.5)
                    exception_backoff = base_backoff + jitter
                    _logger.debug(f"[{self._log_tag}] Rate limit error, retrying {trial} after {exception_backoff:.1f}s: {e}")
                    await asyncio.sleep(exception_backoff)
                    trial += 1
                    continue
                
                # Check for transient errors
                if any(x in error_str for x in ["timeout", "connection", "unavailable", "500", "503"]):
                    if trial >= self.max_retries:
                        _logger.warning(f"[{self._log_tag}] Max retries ({self.max_retries}) exceeded: {e}")
                        raise RuntimeError(f"Gemini API error after {self.max_retries} retries: {e}") from e
                    
                    base_backoff = 2 ** trial
                    jitter = random.uniform(0, base_backoff * 0.5)
                    exception_backoff = base_backoff + jitter
                    _logger.debug(f"[{self._log_tag}] Transient error, retrying {trial} after {exception_backoff:.1f}s: {e}")
                    await asyncio.sleep(exception_backoff)
                    trial += 1
                    continue
                
                # Non-retryable error
                _logger.error(f"[{self._log_tag}] API error: {type(e).__name__}: {e}")
                raise RuntimeError(f"Gemini API error: {e}") from e


class _GeminiResponse:
    """Response wrapper for compatibility with existing code."""
    
    def __init__(self, response_text: str, token_usage: dict, raw_response: Any = None):
        self.response_text = response_text
        self.token_usage = token_usage
        self.raw_response = raw_response

