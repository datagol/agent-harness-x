"""Azure OpenAI provider — same wire format as OpenAI, different client.

Azure OpenAI speaks the OpenAI Chat Completions protocol, so this provider
subclasses :class:`OpenAIProvider` and reuses its entire translation /
``create`` / ``stream`` layer wholesale. Only three things differ:

1. Client construction — Azure needs :class:`openai.AsyncAzureOpenAI`, which
   uses ``api-key`` auth, routes through ``/deployments/{deployment}``, and
   requires an ``api-version`` query param. A bare ``base_url`` swap on the
   standard client does not reach Azure.
2. Deployment routing — on Azure the ``model`` sent to the API is the
   *deployment name*. Set ``AgentConfig.model`` to your deployment, or pin it
   here via ``azure_deployment=``.
3. Token counting — deployment names rarely match tiktoken's model registry.

Two consumer routes (this is a library; it never loads ``.env`` itself):

  # 1. Config-driven, credentials from the environment
  #    (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, OPENAI_API_VERSION)
  agent = Agent(config=AgentConfig(provider="azure", model="<deployment>"))

  # 2. Explicit construction + injection
  provider = AzureOpenAIProvider(
      azure_endpoint="https://<resource>.openai.azure.com",
      api_key="<key>",
      api_version="2024-10-21",
      azure_deployment="<deployment>",
  )
  agent = Agent(config=AgentConfig(provider="azure", model="<deployment>"),
                provider=provider)

Token-based auth (``azure_ad_token`` / ``azure_ad_token_provider``) is
supported: pass it through as a keyword argument and omit ``api_key``.
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator

try:
    from openai import AsyncAzureOpenAI
except ImportError as e:
    raise ImportError(
        "AzureOpenAIProvider requires the openai package. "
        "Install with: pip install 'harnessx[azure]' "
        "(the openai extra covers Azure too)."
    ) from e

from ..types import ProviderResponse, StreamChunk
from ..types import PromptCacheHint
from .openai import OpenAIProvider


class AzureOpenAIProvider(OpenAIProvider):
    """Talks to Azure OpenAI's Chat Completions API via ``AsyncAzureOpenAI``.

    Inherits OpenAIProvider's request translation, response normalization, and
    streaming. Only client wiring, the deployment-name override, and token
    estimation are specialized here.
    """

    # Azure OpenAI caches prefixes automatically; prompt_cache_key
    # support depends on the API version, so it is not sent.
    supports_prompt_cache_key = False

    name = "azure"

    def __init__(
        self,
        client: AsyncAzureOpenAI | None = None,
        *,
        azure_endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
        azure_deployment: str | None = None,
        **client_kwargs: Any,
    ) -> None:
        if client is not None:
            # A preconfigured client manages its own auth/endpoint/version;
            # use it as-is and skip all env validation and SDK construction.
            super().__init__(client=client)
        else:
            resolved_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
            resolved_api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
            resolved_api_version = api_version or os.environ.get("OPENAI_API_VERSION")

            if not resolved_endpoint:
                raise ValueError(
                    "AzureOpenAIProvider requires an endpoint: pass azure_endpoint=... "
                    "or set the AZURE_OPENAI_ENDPOINT environment variable."
                )
            if not resolved_api_version:
                raise ValueError(
                    "AzureOpenAIProvider requires an API version: pass api_version=... "
                    "or set the OPENAI_API_VERSION environment variable."
                )

            # Pass only the args that resolved to a value, so token-based auth
            # (azure_ad_token / azure_ad_token_provider via client_kwargs) is
            # not overridden by an empty/dummy api_key.
            kwargs: dict[str, Any] = {
                "azure_endpoint": resolved_endpoint,
                "api_version": resolved_api_version,
            }
            if resolved_api_key is not None:
                kwargs["api_key"] = resolved_api_key
            kwargs.update(client_kwargs)
            # The engine owns retries (RetryPolicy); the SDK's own would compound them.
            kwargs.setdefault("max_retries", 0)

            super().__init__(client=AsyncAzureOpenAI(**kwargs))

        # Harness-level request-model override (does not pin the SDK URL).
        self.azure_deployment = azure_deployment

    async def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        system: str | None,
        tools: list[dict[str, Any]],
        max_tokens: int,
        temperature: float | None = None,
        cache: PromptCacheHint | None = None,
    ) -> ProviderResponse:
        return await super().create(
            model=self.azure_deployment or model,
            messages=messages,
            system=system,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            cache=cache,
        )

    async def stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        system: str | None,
        tools: list[dict[str, Any]],
        max_tokens: int,
        temperature: float | None = None,
        cache: PromptCacheHint | None = None,
    ) -> AsyncIterator[StreamChunk]:
        async for chunk in super().stream(
            model=self.azure_deployment or model,
            messages=messages,
            system=system,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            cache=cache,
        ):
            yield chunk

    async def count_tokens(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        system: str | None,
        tools: list[dict[str, Any]],
    ) -> int:
        """Approximate prompt token count.

        The estimate is approximate: an Azure deployment name does not identify
        the underlying tokenizer, so we try the model registry, then fall back
        to ``o200k_base`` (gpt-4o/4.1-era), then a char-count heuristic.
        """
        try:
            import tiktoken
            try:
                enc = tiktoken.encoding_for_model(model)
            except KeyError:
                enc = tiktoken.get_encoding("o200k_base")
        except Exception:
            return (len(str(messages)) + len(system or "") + len(str(tools))) // 3

        text = system or ""
        for m in messages:
            text += "\n" + json.dumps(m, default=str)
        for t in tools or []:
            text += "\n" + json.dumps(t, default=str)
        return len(enc.encode(text))
