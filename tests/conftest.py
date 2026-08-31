"""Fake provider SDKs, installed into ``sys.modules``.

The package imports each SDK lazily by name, so a module object in
``sys.modules`` is enough to exercise every code path with no network and no
real dependency installed.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest


class Recorder:
    """Captures what a fake SDK was called with, and replays a response."""

    def __init__(self, response: Any = None):
        self.response = response
        self.calls: list[dict[str, Any]] = []
        self.init_calls: list[dict[str, Any]] = []
        self._failures: list[Exception] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._failures:
            raise self._failures.pop(0)
        return self.response

    async def acall(self, **kwargs: Any) -> Any:
        return self(**kwargs)

    def fail_times(self, n: int, error: Exception) -> None:
        """Fail the next ``n`` calls with ``error``, then succeed."""
        self._failures = [error] * n

    @property
    def last(self) -> dict[str, Any]:
        return self.calls[-1]

    @property
    def call_count(self) -> int:
        return len(self.calls)


def _obj(**kwargs: Any) -> Any:
    return types.SimpleNamespace(**kwargs)


def _error_classes(server_status: int) -> dict[str, type[Exception]]:
    """The exception surface both the OpenAI and Anthropic SDKs expose."""

    class APIError(Exception):
        status_code: int | None = None

    class APIConnectionError(APIError):
        pass

    class APITimeoutError(APIConnectionError):
        pass

    class RateLimitError(APIError):
        status_code = 429

    class InternalServerError(APIError):
        status_code = server_status

    class BadRequestError(APIError):
        status_code = 400

    class AuthenticationError(APIError):
        status_code = 401

    class PermissionDeniedError(APIError):
        status_code = 403

    class NotFoundError(APIError):
        status_code = 404

    return {
        "APIError": APIError,
        "APIConnectionError": APIConnectionError,
        "APITimeoutError": APITimeoutError,
        "RateLimitError": RateLimitError,
        "InternalServerError": InternalServerError,
        "BadRequestError": BadRequestError,
        "AuthenticationError": AuthenticationError,
        "PermissionDeniedError": PermissionDeniedError,
        "NotFoundError": NotFoundError,
    }


def _sdk_module(
    module_name: str, sync_name: str, async_name: str, attach: str, recorder: Recorder
) -> types.ModuleType:
    """Build a fake SDK module whose client exposes ``attach`` with ``.create``."""
    module = types.ModuleType(module_name)
    for name, cls in _error_classes(500 if module_name == "openai" else 503).items():
        setattr(module, name, cls)

    class _Endpoint:
        def __init__(self, is_async: bool):
            self._is_async = is_async

        def create(self, **kwargs: Any) -> Any:
            return recorder.acall(**kwargs) if self._is_async else recorder(**kwargs)

    def make_client(is_async: bool) -> type:
        class _Client:
            def __init__(self, **kwargs: Any):
                recorder.init_calls.append(kwargs)
                endpoint = _Endpoint(is_async)
                if attach == "chat.completions":
                    self.chat = _obj(completions=endpoint)
                else:
                    setattr(self, attach, endpoint)

        return _Client

    setattr(module, sync_name, make_client(False))
    setattr(module, async_name, make_client(True))
    return module


def openai_response(text: str = "hi", model: str = "gpt-4o") -> Any:
    return _obj(
        choices=[_obj(message=_obj(content=text))],
        model=model,
        usage=_obj(prompt_tokens=11, completion_tokens=7),
    )


def anthropic_response(
    text: str = "hi", model: str = "claude-sonnet-5", stop_reason: str = "end_turn"
) -> Any:
    return _obj(
        content=[_obj(type="text", text=text)],
        model=model,
        stop_reason=stop_reason,
        stop_details=None,
        usage=_obj(input_tokens=5, output_tokens=3),
    )


def gemini_response(text: str = "hi") -> Any:
    return _obj(text=text, usage_metadata=_obj(prompt_token_count=9, candidates_token_count=4))


def _genai_modules(recorder: Recorder) -> tuple[types.ModuleType, types.ModuleType]:
    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")

    class _Models:
        def __init__(self, is_async: bool):
            self._is_async = is_async

        def generate_content(self, **kwargs: Any) -> Any:
            return recorder.acall(**kwargs) if self._is_async else recorder(**kwargs)

    class Client:
        def __init__(self, **kwargs: Any):
            recorder.init_calls.append(kwargs)
            self.models = _Models(is_async=False)
            self.aio = _obj(models=_Models(is_async=True))

    genai.Client = Client
    google.genai = genai
    return google, genai


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

_ENV_VARS = (
    "LLM_PROVIDER",
    "LLM_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "GROQ_MODEL",
    "MISTRAL_API_KEY",
    "MISTRAL_MODEL",
    "TOGETHER_API_KEY",
    "TOGETHER_MODEL",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No ambient provider configuration leaks into a test."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def fake_openai(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    recorder = Recorder(response=openai_response())
    monkeypatch.setitem(
        sys.modules,
        "openai",
        _sdk_module("openai", "OpenAI", "AsyncOpenAI", "chat.completions", recorder),
    )
    for key in ("OPENAI_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY", "TOGETHER_API_KEY"):
        monkeypatch.setenv(key, "test-key")
    return recorder


@pytest.fixture
def fake_anthropic(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    recorder = Recorder(response=anthropic_response())
    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        _sdk_module("anthropic", "Anthropic", "AsyncAnthropic", "messages", recorder),
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return recorder


@pytest.fixture
def fake_gemini(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    recorder = Recorder(response=gemini_response())
    google, genai = _genai_modules(recorder)
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    return recorder


@pytest.fixture
def no_sdks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every provider SDK unimportable.

    A meta-path finder, not a ``builtins.__import__`` patch: the package
    reaches for its SDKs through ``importlib.import_module``, which does not
    go through ``__import__``.
    """
    blocked = {"openai", "anthropic", "google.genai"}

    class Blocker:
        def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
            if fullname in blocked:
                raise ImportError(f"No module named {fullname!r}")
            return None

    for name in blocked:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [Blocker(), *sys.meta_path])
