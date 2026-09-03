"""LLM provider abstraction. Lets the agent talk to Anthropic or OpenAI
through a single interface."""

from .base import LLMProvider, ProviderResponse, make_provider
from .anthropic import AnthropicProvider

__all__ = ["LLMProvider", "ProviderResponse", "AnthropicProvider", "make_provider"]

# OpenAIProvider is gated on the openai SDK being installed.
try:
    from .openai import OpenAIProvider  # noqa: F401
    __all__.append("OpenAIProvider")
except ImportError:
    pass
