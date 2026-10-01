"""LLM provider abstraction. Lets the agent talk to Anthropic or OpenAI
through a single interface."""

from .registry import BUILTIN_PROVIDERS, register_provider, registered_providers, unregister_provider  # noqa: F401
from .base import LLMProvider, ProviderResponse, make_provider
from .anthropic import AnthropicProvider
from ..types import Fallback
from .fallback import FallbackProvider

__all__ = [
    "BUILTIN_PROVIDERS", "register_provider", "registered_providers", "unregister_provider", "LLMProvider", "ProviderResponse", "AnthropicProvider", "make_provider",
    "Fallback", "FallbackProvider"]

# OpenAIProvider is gated on the openai SDK being installed.
try:
    from .openai import OpenAIProvider  # noqa: F401
    __all__.append("OpenAIProvider")
except ImportError:
    pass

# GeminiProvider is gated on the google-genai SDK being installed.
try:
    from .gemini import GeminiProvider  # noqa: F401
    __all__.append("GeminiProvider")
except ImportError:
    pass

# OpenRouterProvider is gated on the openai SDK being installed.
try:
    from .openrouter import OpenRouterProvider  # noqa: F401
    __all__.append("OpenRouterProvider")
except ImportError:
    pass

# AzureOpenAIProvider is gated on the openai SDK being installed.
try:
    from .azure_openai import AzureOpenAIProvider  # noqa: F401
    __all__.append("AzureOpenAIProvider")
except ImportError:
    pass
