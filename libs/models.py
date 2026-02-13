"""Centralized model registry for sampler configurations.

Maps model string identifiers to their sampler configurations.
This makes it easy to use predefined model configurations across all tasks.
"""

from libs.sampler.openai_sampler import ResponsesSampler
from libs.sampler.deepseek_sampler import DeepSeekSampler
from libs.sampler.anthropic_sampler import AnthropicSampler
from libs.sampler.kimi_sampler import KimiSampler
from libs.sampler.gemini_sampler import GeminiSampler
from libs.sampler.grok_sampler import GrokSampler
from libs.sampler.openrouter_sampler import OpenRouterSampler


def get_sampler(model_name: str):
    """Get a configured sampler for the given model name.

    Args:
        model_name: Model identifier string (e.g., "gpt-5-mini-high", "gemini-3-flash-websearch")

    Returns:
        Configured sampler instance

    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model: '{model_name}'. Available models: {available}"
        )

    config = MODEL_REGISTRY[model_name]
    if model_name.startswith("deepseek-"):
        return DeepSeekSampler(**config)
    elif model_name.startswith("claude-"):
        return AnthropicSampler(**config)
    elif model_name.startswith("kimi-"):
        return KimiSampler(**config)
    elif model_name.startswith("gemini-"):
        return GeminiSampler(**config)
    elif model_name.startswith("glm-"):
        # Can be either Z.AI or OpenRouter sampler. We use OpenRouter sampler for more concurrency.
        return OpenRouterSampler(**config)
    else:
        return ResponsesSampler(**config)


# Model configurations registry
# Each entry maps a model name to sampler configuration parameters
MODEL_REGISTRY = {
    # GPT-5 variants without reasoning
    "gpt-5": {
        "model": "gpt-5-chat-latest",
        "reasoning_effort": None,
        "websearch": False,
    },
    "gpt-5-websearch": {
        "model": "gpt-5-chat-latest",
        "reasoning_effort": None,
        "websearch": True,
    },
    # GPT-5 chat variants with reasoning levels
    "gpt-5-minimal": {
        "model": "gpt-5",
        "reasoning_effort": "minimal",
        "websearch": False,
    },
    "gpt-5-low": {
        "model": "gpt-5",
        "reasoning_effort": "low",
        "websearch": False,
    },
    "gpt-5-medium": {
        "model": "gpt-5",
        "reasoning_effort": "medium",
        "websearch": False,
    },
    "gpt-5-high": {
        "model": "gpt-5",
        "reasoning_effort": "high",
        "websearch": False,
    },
    "gpt-5-minimal-websearch": {
        "model": "gpt-5",
        "reasoning_effort": "minimal",
        "websearch": True,
    },
    "gpt-5-low-websearch": {
        "model": "gpt-5",
        "reasoning_effort": "low",
        "websearch": True,
    },
    "gpt-5-medium-websearch": {
        "model": "gpt-5",
        "reasoning_effort": "medium",
        "websearch": True,
    },
    "gpt-5-high-websearch": {
        "model": "gpt-5",
        "reasoning_effort": "high",
        "websearch": True,
    },
    # GPT-5 nano with reasoning levels
    "gpt-5-nano": {
        "model": "gpt-5-nano",
        "reasoning_effort": "minimal",
        "websearch": False,
    },
    "gpt-5-nano-minimal": {
        "model": "gpt-5-nano",
        "reasoning_effort": "minimal",
        "websearch": False,
    },
    "gpt-5-nano-low": {
        "model": "gpt-5-nano",
        "reasoning_effort": "low",
        "websearch": False,
    },
    "gpt-5-nano-medium": {
        "model": "gpt-5-nano",
        "reasoning_effort": "medium",
        "websearch": False,
    },
    "gpt-5-nano-high": {
        "model": "gpt-5-nano",
        "reasoning_effort": "high",
        "websearch": False,
    },
    "gpt-5-nano-low-websearch": {
        "model": "gpt-5-nano",
        "reasoning_effort": "low",
        "websearch": True,
    },
    # GPT-5 mini with reasoning levels
    "gpt-5-mini": {
        "model": "gpt-5-mini",
        "reasoning_effort": "minimal",
        "websearch": False,
    },
    "gpt-5-mini-minimal": {
        "model": "gpt-5-mini",
        "reasoning_effort": "minimal",
        "websearch": False,
    },
    "gpt-5-mini-low": {
        "model": "gpt-5-mini",
        "reasoning_effort": "low",
        "websearch": False,
    },
    "gpt-5-mini-medium": {
        "model": "gpt-5-mini",
        "reasoning_effort": "medium",
        "websearch": False,
    },
    "gpt-5-mini-high": {
        "model": "gpt-5-mini",
        "reasoning_effort": "high",
    },
    "gpt-5-nano-minimal-websearch": {
        "model": "gpt-5-nano",
        "reasoning_effort": "minimal",
        "websearch": True,
    },
    "gpt-4": {
        "model": "gpt-4-0613",
        "reasoning_effort": None,
        "websearch": False,
    },
    # GPT-5 mini with reasoning + websearch
    "gpt-5-mini-websearch": {
        "model": "gpt-5-mini",
        "reasoning_effort": "low",
        "websearch": True,
    },
    "gpt-5-mini-low-websearch": {
        "model": "gpt-5-mini",
        "reasoning_effort": "low",
        "websearch": True,
    },
    "gpt-5-mini-medium-websearch": {
        "model": "gpt-5-mini",
        "reasoning_effort": "medium",
        "websearch": True,
    },
    "gpt-5-mini-high-websearch": {
        "model": "gpt-5-mini",
        "reasoning_effort": "high",
        "websearch": True,
    },
    "gpt-5.2": {
        "model": "gpt-5.2-2025-12-11",
        "reasoning_effort": None,
        "websearch": False,
    },
    "gpt-5.2-medium-websearch": {
        "model": "gpt-5.2",
        "reasoning_effort": "medium",
        "websearch": True,
    },
    "gpt-5.2-medium": {
        "model": "gpt-5.2",
        "reasoning_effort": "medium",
        "websearch": False,
    },
    "gpt-5.1": {
        "model": "gpt-5.1-chat-latest",
        "reasoning_effort": None,
        "websearch": False,
    },
    "gpt-5.1-websearch": {
        "model": "gpt-5.1-chat-latest",
        "reasoning_effort": None,
        "websearch": True,
    },
    "gpt-5.1-minimal": {
        "model": "gpt-5.1",
        "reasoning_effort": "minimal",
        "websearch": False,
    },
    "gpt-5.1-low": {
        "model": "gpt-5.1",
        "reasoning_effort": "low",
        "websearch": False,
    },
    "gpt-5.1-medium": {
        "model": "gpt-5.1",
        "reasoning_effort": "medium",
        "websearch": False,
    },
    "gpt-5.1-high": {
        "model": "gpt-5.1",
        "reasoning_effort": "high",
        "websearch": False,
    },
    "gpt-5.1-minimal-websearch": {
        "model": "gpt-5.1",
        "reasoning_effort": "minimal",
        "websearch": True,
    },
    "gpt-5.1-low-websearch": {
        "model": "gpt-5.1",
        "reasoning_effort": "low",
        "websearch": True,
    },
    "gpt-5.1-medium-websearch": {
        "model": "gpt-5.1",
        "reasoning_effort": "medium",
        "websearch": True,
    },
    "gpt-5.1-high-websearch": {
        "model": "gpt-5.1",
        "reasoning_effort": "high",
        "websearch": True,
    },
    "gpt-5.1-mini": {
        "model": "gpt-5.1-mini",
        "reasoning_effort": "minimal",
        "websearch": False,
    },
    "gpt-5.1-mini-minimal": {
        "model": "gpt-5.1-mini",
        "reasoning_effort": "minimal",
        "websearch": False,
    },
    "gpt-5.1-mini-low": {
        "model": "gpt-5.1-mini",
        "reasoning_effort": "low",
        "websearch": False,
    },
    "gpt-5.1-mini-medium": {
        "model": "gpt-5.1-mini",
        "reasoning_effort": "medium",
        "websearch": False,
    },
    "gpt-5.1-mini-high": {
        "model": "gpt-5.1-mini",
        "reasoning_effort": "high",
        "websearch": False,
    },
    "gpt-5.1-mini-minimal-websearch": {
        "model": "gpt-5.1-mini",
        "reasoning_effort": "minimal",
        "websearch": True,
    },
    "gpt-5.1-mini-low-websearch": {
        "model": "gpt-5.1-mini",
        "reasoning_effort": "low",
        "websearch": True,
    },
    "gpt-5.1-mini-medium-websearch": {
        "model": "gpt-5.1-mini",
        "reasoning_effort": "medium",
        "websearch": True,
    },
    "gpt-5.1-mini-high-websearch": {
        "model": "gpt-5.1-mini",
        "reasoning_effort": "high",
        "websearch": True,
    },
    # DeepSeek variants
    "deepseek-chat": {
        "model": "deepseek-chat",
        "temperature": 0.0,
    },
    "deepseek-reasoner": {
        "model": "deepseek-reasoner",
        "temperature": 0.0,
    },
    "deepseek-chat-temp-1": {
        "model": "deepseek-chat",
        "temperature": 1.0,
    },
    "deepseek-reasoner-temp-1": {
        "model": "deepseek-reasoner",
        "temperature": 1.0,
    },
    # Anthropic variants
    "claude-sonnet-4-5": {
        "model": "claude-sonnet-4-5",
        "temperature": 0.0,
    },
    "claude-sonnet-4-5-websearch": {
        "model": "claude-sonnet-4-5",
        "temperature": 0.0,
        "websearch": True,
    },
    "claude-haiku-4-5": {
        "model": "claude-haiku-4-5",
        "temperature": 0.0,
    },
    "claude-haiku-4-5-websearch": {
        "model": "claude-haiku-4-5",
        "temperature": 0.0,
        "websearch": True,
    },
    "claude-opus-4-5": {
        "model": "claude-opus-4-5",
        "temperature": 0.0,
    },
    "claude-opus-4-5-temp-1": {
        "model": "claude-opus-4-5",
        "temperature": 1.0,
    },
    "claude-opus-4-5-websearch": {
        "model": "claude-opus-4-5",
        "temperature": 0.0,
        "websearch": True,
    },
    # Claude Opus 4.5 with effort levels (only model that supports effort parameter)
    "claude-opus-4-5-low": {
        "model": "claude-opus-4-5",
        "temperature": 0.0,
        "effort": "low",
    },
    "claude-opus-4-5-low-websearch": {
        "model": "claude-opus-4-5",
        "temperature": 0.0,
        "effort": "low",
        "websearch": True,
    },
    "claude-opus-4-5-medium": {
        "model": "claude-opus-4-5",
        "temperature": 0.0,
        "effort": "medium",
    },
    "claude-opus-4-5-medium-websearch": {
        "model": "claude-opus-4-5",
        "temperature": 0.0,
        "effort": "medium",
        "websearch": True,
    },
    "claude-opus-4-5-high": {
        "model": "claude-opus-4-5",
        "temperature": 0.0,
        "effort": "high",
    },
    "claude-opus-4-5-high-websearch": {
        "model": "claude-opus-4-5",
        "temperature": 0.0,
        "effort": "high",
        "websearch": True,
    },
    "claude-sonnet-4-0": {
        "model": "claude-sonnet-4-0",
        "temperature": 0.0,
    },
    "claude-sonnet-4-0-websearch": {
        "model": "claude-sonnet-4-0",
        "temperature": 0.0,
        "websearch": True,
    },
    "claude-3-7-sonnet": {
        "model": "claude-3-7-sonnet-latest",
        "temperature": 0.0,
    },
    "claude-3-7-sonnet-websearch": {
        "model": "claude-3-7-sonnet-latest",
        "temperature": 0.0,
        "websearch": True,
    },
    # Kimi (Moonshot AI) variants
    # See: https://platform.moonshot.ai/docs/guide/kimi-k2-quickstart
    "kimi-k2.5": {
        "model": "kimi-k2.5",
        "temperature": 1.0,
    },
    "kimi-k2.5-websearch": {
        "model": "kimi-k2.5",
        "temperature": 1.0,
        "max_tokens": 32768,
        "web_search": True,
    },
    "kimi-k2-thinking": {
        "model": "kimi-k2-thinking-turbo",
        "temperature": 1.0,
    },
    "kimi-k2-preview": {
        "model": "kimi-k2-turbo-preview",
        "temperature": 0.6,
    },
    "kimi-k2-temp-0": {
        "model": "kimi-k2-0711-preview",
        "temperature": 0.0,
    },
    # =========================================================================
    # Gemini 3 models (https://ai.google.dev/gemini-api/docs/gemini-3)
    # =========================================================================
    # Gemini 3 Pro - most capable, complex reasoning
    "gemini-3-pro": {
        "model": "gemini-3-pro-preview",
        "thinking_level": "high",
        "websearch": False,
    },
    "gemini-3-pro-low": {
        "model": "gemini-3-pro-preview",
        "thinking_level": "low",
        "websearch": False,
    },
    "gemini-3-pro-high": {
        "model": "gemini-3-pro-preview",
        "thinking_level": "high",
        "websearch": False,
    },
    "gemini-3-pro-websearch": {
        "model": "gemini-3-pro-preview",
        "thinking_level": "high",
        "websearch": True,
    },
    "gemini-3-pro-low-websearch": {
        "model": "gemini-3-pro-preview",
        "thinking_level": "low",
        "websearch": True,
    },
    "gemini-3-pro-high-websearch": {
        "model": "gemini-3-pro-preview",
        "thinking_level": "high",
        "websearch": True,
    },
    # Gemini 3 Flash - fast and efficient, Pro-level intelligence
    "gemini-3-flash": {
        "model": "gemini-3-flash-preview",
        "thinking_level": "high",
        "websearch": False,
    },
    "gemini-3-flash-minimal": {
        "model": "gemini-3-flash-preview",
        "thinking_level": "minimal",
        "websearch": False,
    },
    "gemini-3-flash-low": {
        "model": "gemini-3-flash-preview",
        "thinking_level": "low",
        "websearch": False,
    },
    "gemini-3-flash-medium": {
        "model": "gemini-3-flash-preview",
        "thinking_level": "medium",
        "websearch": False,
    },
    "gemini-3-flash-high": {
        "model": "gemini-3-flash-preview",
        "thinking_level": "high",
        "websearch": False,
    },
    "gemini-3-flash-websearch": {
        "model": "gemini-3-flash-preview",
        "thinking_level": "high",
        "websearch": True,
    },
    "gemini-3-flash-minimal-websearch": {
        "model": "gemini-3-flash-preview",
        "thinking_level": "minimal",
        "websearch": True,
    },
    "gemini-3-flash-low-websearch": {
        "model": "gemini-3-flash-preview",
        "thinking_level": "low",
        "websearch": True,
    },
    "gemini-3-flash-medium-websearch": {
        "model": "gemini-3-flash-preview",
        "thinking_level": "medium",
        "websearch": True,
    },
    "gemini-3-flash-high-websearch": {
        "model": "gemini-3-flash-preview",
        "thinking_level": "high",
        "websearch": True,
    },
    # OpenRouter – GLM-5 /4.7 (Zhipu via z-ai, https://openrouter.ai/models)
    "glm-5": {
        "model": "z-ai/glm-5",
        "temperature": 0.0,
        "thinking": False,
    },
    "glm-5-thinking": {
        "model": "z-ai/glm-5",
        "temperature": 0.0,
        "thinking": True,
    },
    "glm-4.7": {
        "model": "z-ai/glm-4.7",
        "temperature": 1.0,
        "thinking": False,
    },
    "glm-4.7-thinking": {
        "model": "z-ai/glm-4.7",
        "temperature": 1.0,
        "thinking": True,
    },
    "glm-4.7-temp-0": {
        "model": "z-ai/glm-4.7",
        "temperature": 0.0,
        "thinking": False,
    },
    "glm-4.7-temp-0-thinking": {
        "model": "z-ai/glm-4.7",
        "temperature": 0.0,
        "thinking": True,
    },
    # xAI Grok (https://docs.x.ai/developers/quickstart)
    "grok-4": {
        "model": "grok-4-0709",
        "temperature": 0.0,
    },
    "grok-4-1-fast-reasoning": {
        "model": "grok-4-1-fast-reasoning",
        "temperature": 0.0,
    },
    # Aliases for convenience
    "default": {
        "model": "gpt-5-mini",
        "reasoning_effort": "minimal",
        "websearch": False,
    },
    "default-websearch": {
        "model": "gpt-5-mini",
        "reasoning_effort": "minimal",
        "websearch": True,
    },
}


def list_available_models() -> list[str]:
    """Get list of all available model names."""
    return sorted(MODEL_REGISTRY.keys())
