"""Provider adapters and the registry that maps a name to one.

Everything vendor-specific lives here: SDK imports, request shapes, response
shapes and error classification. ``LLMClient`` knows only :class:`Provider`.
"""

from __future__ import annotations

import importlib
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar

import httpx

from .errors import (
    CLIENT_ERROR_STATUS_CODES,
    RETRYABLE_STATUS_CODES,
    LLMClientError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMProviderError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
)
from .types import Completion, Message, Usage

__all__ = [
    "Provider",
    "PROVIDERS",
    "get_provider_class",
    "available_providers",
    "provider_available",
    "ANTHROPIC_MOST_CAPABLE_MODEL",
]

logger = logging.getLogger(__name__)

#: The Anthropic model to reach for when capability matters more than cost.
ANTHROPIC_MOST_CAPABLE_MODEL = "claude-fable-5"

_JSON_INSTRUCTION = (
    "Respond with a single valid JSON value and nothing else. "
    "Do not wrap it in markdown fences or add commentary."
)


def _import_sdk(module: str, extra: str) -> Any:
    """Import a provider SDK, or explain exactly how to install it."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise LLMConfigurationError(
            f"The {module!r} package is required for this provider. "
            f'Install it with: pip install "llm-client[{extra}]"'
        ) from exc


def _classify_by_status(exc: Exception, provider: str, status: int | None) -> LLMProviderError:
    """Map an HTTP status onto the exception hierarchy."""
    if status == 429:
        return LLMRateLimitError(f"{provider} rate limit exceeded: {exc}")
    if status is not None and status in RETRYABLE_STATUS_CODES:
        return LLMServerError(f"{provider} server error: {exc}", status_code=status)
    if status is not None and status in CLIENT_ERROR_STATUS_CODES:
        return LLMClientError(f"{provider} client error: {exc}", status_code=status)
    return LLMProviderError(f"{provider} error: {exc}", status_code=status)


def _classify_httpx(exc: Exception, provider: str) -> LLMProviderError | None:
    """Classify transport-level failures shared by every httpx-based provider."""
    if isinstance(exc, httpx.TimeoutException):
        return LLMTimeoutError(f"{provider} timeout: {exc}")
    if isinstance(exc, httpx.ConnectError):
        return LLMConnectionError(f"{provider} connection error: {exc}")
    if isinstance(exc, httpx.HTTPStatusError):
        return _classify_by_status(exc, provider, exc.response.status_code)
    if isinstance(exc, httpx.RequestError):
        return LLMConnectionError(f"{provider} request error: {exc}")
    return None


class Provider(ABC):
    """One vendor, normalised.

    Subclasses translate in both directions - request shape and error shape -
    and nothing else. Retry, caching and env resolution belong to the client.
    """

    name: ClassVar[str]
    default_model: ClassVar[str]
    #: Environment variables searched, in order, for an API key.
    api_key_envs: ClassVar[tuple[str, ...]] = ()
    #: Optional-dependency extra that supplies this provider's SDK.
    sdk_extra: ClassVar[str] = ""
    #: Endpoint used when the caller passes no ``base_url``.
    default_base_url: ClassVar[str | None] = None

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ):
        self.model = model
        self.timeout = timeout
        self.base_url = base_url or self.default_base_url
        self.api_key = api_key or self._api_key_from_env()

    @classmethod
    def _api_key_from_env(cls) -> str | None:
        for env in cls.api_key_envs:
            value = os.getenv(env)
            if value:
                return value
        return None

    def require_api_key(self) -> str:
        if not self.api_key:
            envs = " or ".join(self.api_key_envs)
            raise LLMConfigurationError(
                f"No API key for provider {self.name!r}. Set {envs} or pass api_key= to LLMClient."
            )
        return self.api_key

    @classmethod
    def available(cls) -> bool:
        """Whether this provider looks usable right now (credentials present)."""
        return bool(cls._api_key_from_env())

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None,
        temperature: float | None,
        max_tokens: int | None,
        json_mode: bool,
    ) -> Completion: ...

    @abstractmethod
    async def acomplete(
        self,
        messages: list[Message],
        *,
        system: str | None,
        temperature: float | None,
        max_tokens: int | None,
        json_mode: bool,
    ) -> Completion: ...

    def classify(self, exc: Exception) -> LLMProviderError:
        """Normalise a vendor exception. Subclasses extend, never replace."""
        if isinstance(exc, LLMProviderError):
            return exc
        classified = _classify_httpx(exc, self.name)
        if classified is not None:
            return classified
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        return _classify_by_status(exc, self.name, status if isinstance(status, int) else None)


# ---------------------------------------------------------------------------
# OpenAI and the OpenAI-compatible endpoints
# ---------------------------------------------------------------------------


class _OpenAICompatible(Provider):
    """Anything speaking ``/chat/completions``.

    Groq, Mistral and Together each shipped their own SDK in the source repos;
    all three expose an OpenAI-compatible endpoint, so one adapter plus a
    ``base_url`` replaces three dependencies.
    """

    sdk_extra = "openai"

    def _sdk(self) -> Any:
        return _import_sdk("openai", self.sdk_extra)

    def _client(self, is_async: bool) -> Any:
        openai = self._sdk()
        factory = openai.AsyncOpenAI if is_async else openai.OpenAI
        # max_retries=0: this package owns the retry policy, and letting the
        # SDK retry too would multiply the attempts (3 x 3 = 9).
        return factory(
            api_key=self.require_api_key(),
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=0,
        )

    def _request(
        self,
        messages: list[Message],
        system: str | None,
        temperature: float | None,
        max_tokens: int | None,
        json_mode: bool,
    ) -> dict[str, Any]:
        payload: list[dict[str, str]] = []
        if system:
            payload.append({"role": "system", "content": system})
        payload.extend(dict(m) for m in messages)

        kwargs: dict[str, Any] = {"model": self.model, "messages": payload}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    def _completion(self, response: Any) -> Completion:
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        return Completion(
            text=choice.message.content or "",
            model=getattr(response, "model", None) or self.model,
            provider=self.name,
            usage=Usage(
                input_tokens=getattr(usage, "prompt_tokens", None),
                output_tokens=getattr(usage, "completion_tokens", None),
            ),
            raw=response,
        )

    def complete(self, messages, *, system, temperature, max_tokens, json_mode) -> Completion:
        client = self._client(is_async=False)
        try:
            response = client.chat.completions.create(
                **self._request(messages, system, temperature, max_tokens, json_mode)
            )
        except Exception as exc:
            raise self.classify(exc) from exc
        return self._completion(response)

    async def acomplete(
        self, messages, *, system, temperature, max_tokens, json_mode
    ) -> Completion:
        client = self._client(is_async=True)
        try:
            response = await client.chat.completions.create(
                **self._request(messages, system, temperature, max_tokens, json_mode)
            )
        except Exception as exc:
            raise self.classify(exc) from exc
        return self._completion(response)

    def classify(self, exc: Exception) -> LLMProviderError:
        if isinstance(exc, LLMProviderError):
            return exc
        try:
            import openai
        except ImportError:
            return super().classify(exc)

        if isinstance(exc, openai.APITimeoutError):
            return LLMTimeoutError(f"{self.name} timeout: {exc}")
        if isinstance(exc, openai.APIConnectionError):
            return LLMConnectionError(f"{self.name} connection error: {exc}")
        if isinstance(exc, openai.RateLimitError):
            return LLMRateLimitError(f"{self.name} rate limit exceeded: {exc}")
        if isinstance(exc, openai.InternalServerError):
            status = getattr(exc, "status_code", 500) or 500
            return LLMServerError(f"{self.name} server error: {exc}", status_code=status)
        if isinstance(
            exc,
            openai.BadRequestError
            | openai.AuthenticationError
            | openai.PermissionDeniedError
            | openai.NotFoundError,
        ):
            return LLMClientError(
                f"{self.name} client error: {exc}",
                status_code=getattr(exc, "status_code", 400) or 400,
            )
        return super().classify(exc)


class OpenAIProvider(_OpenAICompatible):
    name = "openai"
    default_model = "gpt-4o"
    api_key_envs = ("OPENAI_API_KEY",)


class GroqProvider(_OpenAICompatible):
    name = "groq"
    default_model = "llama-3.3-70b-versatile"
    api_key_envs = ("GROQ_API_KEY",)
    default_base_url = "https://api.groq.com/openai/v1"


class MistralProvider(_OpenAICompatible):
    name = "mistral"
    default_model = "mistral-large-latest"
    api_key_envs = ("MISTRAL_API_KEY",)
    default_base_url = "https://api.mistral.ai/v1"


class TogetherProvider(_OpenAICompatible):
    name = "together"
    default_model = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    api_key_envs = ("TOGETHER_API_KEY",)
    default_base_url = "https://api.together.xyz/v1"


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


class AnthropicProvider(Provider):
    """Claude via the Messages API."""

    name = "anthropic"
    default_model = "claude-sonnet-5"
    api_key_envs = ("ANTHROPIC_API_KEY",)
    sdk_extra = "anthropic"
    #: Anthropic requires max_tokens; this is the fallback when none is given.
    default_max_tokens: ClassVar[int] = 4096

    def _client(self, is_async: bool) -> Any:
        anthropic = _import_sdk("anthropic", self.sdk_extra)
        factory = anthropic.AsyncAnthropic if is_async else anthropic.Anthropic
        return factory(
            api_key=self.require_api_key(),
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=0,
        )

    def _request(
        self,
        messages: list[Message],
        system: str | None,
        temperature: float | None,
        max_tokens: int | None,
        json_mode: bool,
    ) -> dict[str, Any]:
        # Anthropic takes the system prompt as its own parameter, not as a
        # message. Hoist any system turns so a caller can pass one either way.
        system_parts = [str(m["content"]) for m in messages if m["role"] == "system"]
        if system:
            system_parts.insert(0, system)
        if json_mode:
            system_parts.append(_JSON_INSTRUCTION)

        turns = [dict(m) for m in messages if m["role"] != "system"]
        if not turns:
            raise LLMProviderError(
                "anthropic requires at least one user or assistant message", retryable=False
            )

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.default_max_tokens,
            "messages": turns,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        if temperature is not None:
            # Not a bug: the Messages API removed sampling parameters, and the
            # anthropic SDK dropped `temperature` from create() with them, so
            # forwarding one is a TypeError rather than a weaker answer.
            logger.debug(
                "ignoring temperature=%s: the Anthropic Messages API takes no sampling parameters",
                temperature,
            )
        return kwargs

    def _completion(self, response: Any) -> Completion:
        # Check why generation stopped before trusting the content: a refused
        # request is a 200 with no usable text.
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None)
            raise LLMClientError(
                f"anthropic refused the request (category={category})", status_code=400
            )

        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        usage = getattr(response, "usage", None)
        return Completion(
            text=text,
            model=getattr(response, "model", None) or self.model,
            provider=self.name,
            usage=Usage(
                input_tokens=getattr(usage, "input_tokens", None),
                output_tokens=getattr(usage, "output_tokens", None),
            ),
            raw=response,
        )

    def complete(self, messages, *, system, temperature, max_tokens, json_mode) -> Completion:
        kwargs = self._request(messages, system, temperature, max_tokens, json_mode)
        client = self._client(is_async=False)
        try:
            response = client.messages.create(**kwargs)
        except Exception as exc:
            raise self.classify(exc) from exc
        return self._completion(response)

    async def acomplete(
        self, messages, *, system, temperature, max_tokens, json_mode
    ) -> Completion:
        kwargs = self._request(messages, system, temperature, max_tokens, json_mode)
        client = self._client(is_async=True)
        try:
            response = await client.messages.create(**kwargs)
        except Exception as exc:
            raise self.classify(exc) from exc
        return self._completion(response)

    def classify(self, exc: Exception) -> LLMProviderError:
        if isinstance(exc, LLMProviderError):
            return exc
        try:
            import anthropic
        except ImportError:
            return super().classify(exc)

        if isinstance(exc, anthropic.APITimeoutError):
            return LLMTimeoutError(f"anthropic timeout: {exc}")
        if isinstance(exc, anthropic.APIConnectionError):
            return LLMConnectionError(f"anthropic connection error: {exc}")
        if isinstance(exc, anthropic.RateLimitError):
            return LLMRateLimitError(f"anthropic rate limit exceeded: {exc}")
        if isinstance(exc, anthropic.InternalServerError):
            status = getattr(exc, "status_code", 500) or 500
            return LLMServerError(f"anthropic server error: {exc}", status_code=status)
        if isinstance(
            exc,
            anthropic.BadRequestError
            | anthropic.AuthenticationError
            | anthropic.PermissionDeniedError
            | anthropic.NotFoundError,
        ):
            return LLMClientError(
                f"anthropic client error: {exc}",
                status_code=getattr(exc, "status_code", 400) or 400,
            )
        return super().classify(exc)


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------


class GeminiProvider(Provider):
    """Gemini via the ``google-genai`` SDK."""

    name = "gemini"
    default_model = "gemini-1.5-pro"
    api_key_envs = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    sdk_extra = "gemini"

    def _client(self) -> Any:
        genai = _import_sdk("google.genai", self.sdk_extra)
        return genai.Client(api_key=self.require_api_key())

    def _request(
        self,
        messages: list[Message],
        system: str | None,
        temperature: float | None,
        max_tokens: int | None,
        json_mode: bool,
    ) -> dict[str, Any]:
        # Gemini calls the assistant "model" and takes the system prompt out of
        # band, same as Anthropic. Dicts avoid importing the SDK's type module.
        system_parts = [str(m["content"]) for m in messages if m["role"] == "system"]
        if system:
            system_parts.insert(0, system)

        contents = [
            {
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": [{"text": str(m["content"])}],
            }
            for m in messages
            if m["role"] != "system"
        ]
        if not contents:
            raise LLMProviderError(
                "gemini requires at least one user or assistant message", retryable=False
            )

        config: dict[str, Any] = {}
        if system_parts:
            config["system_instruction"] = "\n\n".join(system_parts)
        if temperature is not None:
            config["temperature"] = temperature
        if max_tokens is not None:
            config["max_output_tokens"] = max_tokens
        if json_mode:
            config["response_mime_type"] = "application/json"

        kwargs: dict[str, Any] = {"model": self.model, "contents": contents}
        if config:
            kwargs["config"] = config
        return kwargs

    def _completion(self, response: Any) -> Completion:
        meta = getattr(response, "usage_metadata", None)
        return Completion(
            text=getattr(response, "text", None) or "",
            model=self.model,
            provider=self.name,
            usage=Usage(
                input_tokens=getattr(meta, "prompt_token_count", None),
                output_tokens=getattr(meta, "candidates_token_count", None),
            ),
            raw=response,
        )

    def complete(self, messages, *, system, temperature, max_tokens, json_mode) -> Completion:
        kwargs = self._request(messages, system, temperature, max_tokens, json_mode)
        client = self._client()
        try:
            response = client.models.generate_content(**kwargs)
        except Exception as exc:
            raise self.classify(exc) from exc
        return self._completion(response)

    async def acomplete(
        self, messages, *, system, temperature, max_tokens, json_mode
    ) -> Completion:
        kwargs = self._request(messages, system, temperature, max_tokens, json_mode)
        client = self._client()
        try:
            response = await client.aio.models.generate_content(**kwargs)
        except Exception as exc:
            raise self.classify(exc) from exc
        return self._completion(response)


# ---------------------------------------------------------------------------
# Ollama (local, no API key)
# ---------------------------------------------------------------------------


class OllamaProvider(Provider):
    """A local Ollama server, over its REST API. No SDK, no key."""

    name = "ollama"
    default_model = "llama3.2"
    default_base_url = "http://localhost:11434"

    def __init__(self, model: str, *, base_url: str | None = None, **kwargs: Any):
        # Resolved before super().__init__, which would otherwise fill base_url
        # from the class default and hide OLLAMA_BASE_URL entirely.
        resolved = base_url or os.getenv("OLLAMA_BASE_URL") or self.default_base_url or ""
        super().__init__(model, base_url=resolved.rstrip("/"), **kwargs)

    @classmethod
    def _api_key_from_env(cls) -> str | None:
        return None

    def require_api_key(self) -> str:  # pragma: no cover - Ollama needs none
        return ""

    @classmethod
    def available(cls) -> bool:
        """Ollama has no key to check, so ask the server whether it is up."""
        base = (os.getenv("OLLAMA_BASE_URL") or cls.default_base_url or "").rstrip("/")
        try:
            return httpx.get(f"{base}/api/tags", timeout=2.0).status_code == 200
        except httpx.HTTPError:
            return False

    def _request(
        self,
        messages: list[Message],
        system: str | None,
        temperature: float | None,
        max_tokens: int | None,
        json_mode: bool,
    ) -> dict[str, Any]:
        payload: list[dict[str, str]] = []
        if system:
            payload.append({"role": "system", "content": system})
        payload.extend(dict(m) for m in messages)

        options: dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        body: dict[str, Any] = {"model": self.model, "messages": payload, "stream": False}
        if options:
            body["options"] = options
        if json_mode:
            body["format"] = "json"
        return body

    def _completion(self, data: dict[str, Any]) -> Completion:
        return Completion(
            text=(data.get("message") or {}).get("content", ""),
            model=data.get("model") or self.model,
            provider=self.name,
            usage=Usage(
                input_tokens=data.get("prompt_eval_count"),
                output_tokens=data.get("eval_count"),
            ),
            raw=data,
        )

    def complete(self, messages, *, system, temperature, max_tokens, json_mode) -> Completion:
        body = self._request(messages, system, temperature, max_tokens, json_mode)
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(f"{self.base_url}/api/chat", json=body)
                response.raise_for_status()
                return self._completion(response.json())
        except Exception as exc:
            raise self.classify(exc) from exc

    async def acomplete(
        self, messages, *, system, temperature, max_tokens, json_mode
    ) -> Completion:
        body = self._request(messages, system, temperature, max_tokens, json_mode)
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=body)
                response.raise_for_status()
                return self._completion(response.json())
        except Exception as exc:
            raise self.classify(exc) from exc

    def classify(self, exc: Exception) -> LLMProviderError:
        classified = super().classify(exc)
        if isinstance(classified, LLMConnectionError):
            return LLMConnectionError(f"{classified}. Is Ollama running at {self.base_url}?")
        return classified


# ---------------------------------------------------------------------------
# Mock (tests, offline development)
# ---------------------------------------------------------------------------


class MockProvider(Provider):
    """Deterministic, offline, free. Useful as a default in test suites."""

    name = "mock"
    default_model = "mock-model"

    def __init__(self, model: str = default_model, **kwargs: Any):
        super().__init__(model, **kwargs)
        self.call_count = 0
        self.last_messages: list[Message] = []
        #: Set to a callable to control the response text.
        self.responder: Callable[[list[Message]], str] | None = None

    @classmethod
    def available(cls) -> bool:
        return True

    def require_api_key(self) -> str:  # pragma: no cover - the mock needs none
        return ""

    def _text(self, messages: list[Message]) -> str:
        self.call_count += 1
        self.last_messages = list(messages)
        if self.responder is not None:
            return self.responder(messages)
        last = messages[-1]["content"] if messages else ""
        return f"Mock response for prompt ({len(last)} characters)"

    def complete(self, messages, *, system, temperature, max_tokens, json_mode) -> Completion:
        return Completion(
            text=self._text(messages), model=self.model, provider=self.name, usage=Usage()
        )

    async def acomplete(
        self, messages, *, system, temperature, max_tokens, json_mode
    ) -> Completion:
        return self.complete(
            messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )


#: Name -> adapter. Ordering is the fallback preference used by
#: :func:`available_providers`.
PROVIDERS: dict[str, type[Provider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "groq": GroqProvider,
    "mistral": MistralProvider,
    "together": TogetherProvider,
    "ollama": OllamaProvider,
    "mock": MockProvider,
}

#: ``google`` was the name both source repos used for Gemini.
_ALIASES = {"google": "gemini", "claude": "anthropic", "gpt": "openai"}


def get_provider_class(name: str) -> type[Provider]:
    """Look up a provider by name, accepting the legacy aliases."""
    key = name.strip().lower()
    key = _ALIASES.get(key, key)
    try:
        return PROVIDERS[key]
    except KeyError:
        raise LLMConfigurationError(
            f"Unknown provider: {name!r}. Available: {', '.join(PROVIDERS)}"
        ) from None


def provider_available(name: str) -> bool:
    """Whether ``name`` has the credentials (or the running server) it needs."""
    return get_provider_class(name).available()


def available_providers() -> list[str]:
    """Provider names that look usable right now, in preference order."""
    return [name for name, cls in PROVIDERS.items() if name != "mock" and cls.available()]
