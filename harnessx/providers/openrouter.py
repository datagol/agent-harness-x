"""OpenRouter provider — OpenAI-compatible provider with OpenRouter-specific
defaults, attribution headers, and reasoning/thinking support.
"""

from __future__ import annotations

import json
import os
from typing import Any

try:
    from openai import AsyncOpenAI
except ImportError as e:
    raise ImportError(
        "OpenRouterProvider requires the openai package. "
        "Install with: pip install 'harnessx[openrouter]'"
    ) from e

from .openai import OpenAIProvider

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(OpenAIProvider):
    """Talks to OpenRouter's OpenAI-compatible Chat Completions API.

    Features:
    - Default base URL set to OpenRouter (https://openrouter.ai/api/v1).
    - Automatic credential resolution via OPENROUTER_API_KEY (falling back to OPENAI_API_KEY).
    - Attribution header configuration for OpenRouter rankings (HTTP-Referer, X-Title).
    - Reasoning and thinking token extraction across reasoning models (DeepSeek-R1, Claude 3.7 Sonnet).
    - Resilient token estimation for multi-vendor model namespaces.
    """

    name = "openrouter"
    # OpenRouter forwards requests to many upstreams; prompt_cache_key is not
    # sent because not every upstream accepts it. Upstream automatic caching
    # still applies to a stable prefix.
    supports_prompt_cache_key = False

    def __init__(
        self,
        client: AsyncOpenAI | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        site_url: str | None = None,
        app_name: str | None = None,
        default_headers: dict[str, str] | None = None,
        **client_kwargs: Any,
    ) -> None:
        if client is not None:
            self.client = client
        else:
            resolved_api_key = (
                api_key
                or os.environ.get("OPENROUTER_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or ""
            )
            resolved_base_url = (
                base_url
                or os.environ.get("OPENROUTER_BASE_URL")
                or DEFAULT_OPENROUTER_BASE_URL
            )
            resolved_site_url = (
                site_url
                or os.environ.get("OPENROUTER_SITE_URL")
                or os.environ.get("OR_SITE_URL")
            )
            resolved_app_name = (
                app_name
                or os.environ.get("OPENROUTER_APP_NAME")
                or os.environ.get("OR_APP_NAME")
            )

            headers = dict(default_headers or {})
            if resolved_site_url:
                headers["HTTP-Referer"] = resolved_site_url
            if resolved_app_name:
                headers["X-Title"] = resolved_app_name

            self.client = AsyncOpenAI(
                api_key=resolved_api_key or "missing-openrouter-key",
                base_url=resolved_base_url,
                default_headers=headers if headers else None,
                **client_kwargs,
            )

    async def count_tokens(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        system: str | None,
        tools: list[dict[str, Any]],
    ) -> int:
        """Estimate token count across diverse OpenRouter model families."""
        try:
            import tiktoken
            # Strip provider prefix if present (e.g., 'openai/gpt-4o' -> 'gpt-4o')
            sub_model = model.split("/")[-1] if "/" in model else model
            try:
                enc = tiktoken.encoding_for_model(model)
            except Exception:
                enc = tiktoken.encoding_for_model(sub_model)

            text = system or ""
            for m in messages:
                text += "\n" + json.dumps(m, default=str)
            for t in tools or []:
                text += "\n" + json.dumps(t, default=str)
            return len(enc.encode(text))
        except Exception:
            # Multi-vendor heuristic: ~3 chars per token across diverse models
            return (len(str(messages)) + len(system or "") + len(str(tools or []))) // 3
