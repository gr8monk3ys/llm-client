"""``LLMClient`` - the one interface this package exists to provide."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from .cache import DEFAULT_MAXSIZE, DEFAULT_TTL, ResponseCache
from .errors import LLMConfigurationError
from .providers import Provider, get_provider_class
from .retry import DEFAULT_INITIAL_DELAY, DEFAULT_MAX_DELAY, build_retry
from .types import Completion, Message, Usage, normalize_messages

__all__ = ["LLMClient"]

DEFAULT_PROVIDER = "openai"


class LLMClient:
    """Talk to any supported provider through one method.

    ``provider`` and ``model`` fall back to the environment
    (``LLM_PROVIDER``; then ``<PROVIDER>_MODEL``, then ``LLM_MODEL``), so a
    deployment can switch models without touching code::

        client = LLMClient()
        print(client.complete("Say hi").text)

    Retry, caching and error normalisation happen here so that every provider
    behaves the same; see :mod:`llm_client.providers` for what is vendor-specific.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        cache: bool = False,
        cache_ttl: float = DEFAULT_TTL,
        cache_maxsize: int = DEFAULT_MAXSIZE,
        retry_initial_delay: float = DEFAULT_INITIAL_DELAY,
        retry_max_delay: float = DEFAULT_MAX_DELAY,
    ):
        provider_name = (provider or os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
        provider_cls = get_provider_class(provider_name)

        self._provider: Provider = provider_cls(
            self._resolve_model(provider_cls, model),
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        self._retry = build_retry(max_retries, retry_initial_delay, retry_max_delay)
        self._cache = ResponseCache(maxsize=cache_maxsize, ttl=cache_ttl) if cache else None
        self.max_retries = max_retries

    # -- identity ----------------------------------------------------------

    @staticmethod
    def _resolve_model(provider_cls: type[Provider], model: str | None) -> str:
        """Explicit argument, then ``<PROVIDER>_MODEL``, then ``LLM_MODEL``, then the default.

        The provider-specific variable beats the generic one: it is the more
        specific statement of intent, and it is what both source projects set.
        """
        resolved = (
            model
            or os.getenv(f"{provider_cls.name.upper()}_MODEL")
            or os.getenv("LLM_MODEL")
            or provider_cls.default_model
        )
        if not resolved:
            raise LLMConfigurationError(f"No model for provider {provider_cls.name!r}")
        return resolved

    @property
    def provider(self) -> str:
        """The provider name, e.g. ``"anthropic"``."""
        return self._provider.name

    @property
    def model(self) -> str:
        """The resolved model id."""
        return self._provider.model

    @property
    def backend(self) -> Provider:
        """The underlying adapter. Reach for this only to get at vendor specifics."""
        return self._provider

    def __repr__(self) -> str:
        return f"LLMClient(provider={self.provider!r}, model={self.model!r})"

    # -- completion --------------------------------------------------------

    def complete(
        self,
        messages: str | Sequence[Message] | Sequence[dict[str, Any]],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> Completion:
        """Send a prompt or a message list and get one :class:`Completion` back.

        ``json_mode`` asks the provider for JSON natively where it can
        (OpenAI-compatible, Gemini, Ollama) and instructs the model where it
        cannot (Anthropic). It does not parse or validate the result.
        """
        turns = normalize_messages(messages)
        params = {"temperature": temperature, "max_tokens": max_tokens, "json_mode": json_mode}
        key = self._cache_key(turns, system, params)

        if key is not None and self._cache is not None:
            hit = self._cache.get(key)
            if hit is not None:
                return self._cached_completion(hit)

        call = self._retry(self._provider.complete)
        result = call(
            turns,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
        self._store(key, result)
        return result

    async def acomplete(
        self,
        messages: str | Sequence[Message] | Sequence[dict[str, Any]],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> Completion:
        """The ``async`` twin of :meth:`complete`, sharing its cache and retry policy."""
        turns = normalize_messages(messages)
        params = {"temperature": temperature, "max_tokens": max_tokens, "json_mode": json_mode}
        key = self._cache_key(turns, system, params)

        if key is not None and self._cache is not None:
            hit = self._cache.get(key)
            if hit is not None:
                return self._cached_completion(hit)

        call = self._retry(self._provider.acomplete)
        result = await call(
            turns,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
        self._store(key, result)
        return result

    # -- cache -------------------------------------------------------------

    def _cache_key(
        self, turns: list[Message], system: str | None, params: dict[str, Any]
    ) -> str | None:
        if self._cache is None:
            return None
        return ResponseCache.key(self.provider, self.model, turns, system, params)

    def _cached_completion(self, text: str) -> Completion:
        # No usage and no raw response: nothing was billed for this one.
        return Completion(text=text, model=self.model, provider=self.provider, usage=Usage())

    def _store(self, key: str | None, result: Completion) -> None:
        if key is not None and self._cache is not None:
            self._cache.set(key, result.text)

    @property
    def cache_enabled(self) -> bool:
        return self._cache is not None

    def clear_cache(self) -> None:
        """Drop every cached response."""
        if self._cache is not None:
            self._cache.clear()

    def cache_stats(self) -> dict[str, Any]:
        """Size, capacity and TTL of the cache; ``enabled: False`` when off."""
        if self._cache is None:
            return {"enabled": False, "current_size": 0, "max_size": 0, "ttl_seconds": 0.0}
        return {"enabled": True, **self._cache.stats()}
