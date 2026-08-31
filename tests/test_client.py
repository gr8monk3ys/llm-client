"""Dispatch, configuration resolution and the shape of the public interface."""

from __future__ import annotations

import pytest

from llm_client import (
    Completion,
    LLMClient,
    LLMConfigurationError,
    Usage,
    available_providers,
    provider_available,
)
from llm_client.providers import AnthropicProvider, GeminiProvider
from llm_client.types import normalize_messages


class TestDispatch:
    def test_provider_name_selects_the_adapter(self, fake_anthropic):
        client = LLMClient("anthropic", "claude-sonnet-5")
        assert isinstance(client.backend, AnthropicProvider)
        assert client.provider == "anthropic"

    def test_google_is_an_alias_for_gemini(self, fake_gemini):
        client = LLMClient("google")
        assert isinstance(client.backend, GeminiProvider)
        assert client.provider == "gemini"

    def test_provider_comes_from_the_environment(self, monkeypatch, fake_anthropic):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        assert LLMClient().provider == "anthropic"

    def test_default_provider_is_openai(self, fake_openai):
        assert LLMClient().provider == "openai"

    def test_unknown_provider_names_the_alternatives(self):
        with pytest.raises(LLMConfigurationError) as excinfo:
            LLMClient("wishful-thinking")
        assert "wishful-thinking" in str(excinfo.value)
        assert "anthropic" in str(excinfo.value)

    def test_each_registered_provider_is_constructible(
        self, fake_openai, fake_anthropic, fake_gemini
    ):
        from llm_client import PROVIDERS

        for name in PROVIDERS:
            assert LLMClient(name).provider == name


class TestModelResolution:
    def test_explicit_argument_wins(self, monkeypatch, fake_anthropic):
        monkeypatch.setenv("ANTHROPIC_MODEL", "from-provider-env")
        monkeypatch.setenv("LLM_MODEL", "from-generic-env")
        assert LLMClient("anthropic", "explicit").model == "explicit"

    def test_provider_env_beats_generic_env(self, monkeypatch, fake_anthropic):
        monkeypatch.setenv("ANTHROPIC_MODEL", "from-provider-env")
        monkeypatch.setenv("LLM_MODEL", "from-generic-env")
        assert LLMClient("anthropic").model == "from-provider-env"

    def test_generic_env_beats_the_default(self, monkeypatch, fake_anthropic):
        monkeypatch.setenv("LLM_MODEL", "from-generic-env")
        assert LLMClient("anthropic").model == "from-generic-env"

    def test_anthropic_default_is_current(self, fake_anthropic):
        assert LLMClient("anthropic").model == "claude-sonnet-5"

    def test_most_capable_anthropic_model_is_published(self):
        from llm_client import ANTHROPIC_MOST_CAPABLE_MODEL

        assert ANTHROPIC_MOST_CAPABLE_MODEL == "claude-fable-5"


class TestCredentials:
    def test_missing_key_is_a_configuration_error(self, monkeypatch, fake_openai):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        client = LLMClient("openai")
        with pytest.raises(LLMConfigurationError) as excinfo:
            client.complete("hello")
        assert "OPENAI_API_KEY" in str(excinfo.value)

    def test_explicit_key_beats_the_environment(self, fake_openai):
        client = LLMClient("openai", api_key="explicit-key")
        client.complete("hello")
        assert fake_openai.init_calls[0]["api_key"] == "explicit-key"

    def test_gemini_accepts_either_key_variable(self, monkeypatch, fake_gemini):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
        LLMClient("gemini").complete("hi")
        assert fake_gemini.init_calls[0]["api_key"] == "google-key"

    def test_provider_available_reflects_the_environment(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert provider_available("anthropic") is False
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        assert provider_available("anthropic") is True

    def test_available_providers_lists_only_configured_ones(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        monkeypatch.setattr(
            "llm_client.providers.OllamaProvider.available", classmethod(lambda cls: False)
        )
        assert available_providers() == ["openai"]


class TestMessageNormalisation:
    def test_a_bare_string_becomes_one_user_turn(self):
        assert normalize_messages("hi") == [{"role": "user", "content": "hi"}]

    def test_a_message_list_is_preserved(self):
        turns = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        assert normalize_messages(turns) == turns

    def test_empty_list_is_rejected(self):
        with pytest.raises(ValueError):
            normalize_messages([])

    def test_unknown_role_is_rejected(self):
        with pytest.raises(ValueError):
            normalize_messages([{"role": "wizard", "content": "a"}])

    def test_malformed_message_is_rejected(self):
        with pytest.raises(TypeError):
            normalize_messages([{"content": "no role"}])


class TestCompletion:
    def test_usage_totals(self):
        assert Usage(3, 4).total_tokens == 7
        assert Usage().total_tokens is None

    def test_str_is_the_text(self):
        assert str(Completion(text="hello", model="m", provider="p")) == "hello"

    def test_completion_carries_provider_metadata(self, fake_openai):
        result = LLMClient("openai").complete("hi")
        assert (result.text, result.provider, result.model) == ("hi", "openai", "gpt-4o")
        assert result.usage == Usage(input_tokens=11, output_tokens=7)
        assert result.raw is not None


class TestAsync:
    async def test_acomplete_uses_the_async_sdk_client(self, fake_openai):
        result = await LLMClient("openai").acomplete("hi")
        assert result.text == "hi"
        assert fake_openai.last["messages"] == [{"role": "user", "content": "hi"}]

    async def test_acomplete_on_anthropic(self, fake_anthropic):
        result = await LLMClient("anthropic").acomplete("hi", system="be brief")
        assert result.text == "hi"
        assert fake_anthropic.last["system"] == "be brief"

    async def test_acomplete_on_gemini(self, fake_gemini):
        assert (await LLMClient("gemini").acomplete("hi")).text == "hi"


class TestMock:
    def test_mock_needs_no_key_and_counts_calls(self):
        client = LLMClient("mock")
        client.complete("hello")
        client.complete("hello there")
        assert client.backend.call_count == 2
        assert client.complete("x").text.startswith("Mock response")

    def test_responder_overrides_the_text(self):
        client = LLMClient("mock")
        client.backend.responder = lambda turns: turns[-1]["content"].upper()
        assert client.complete("shout").text == "SHOUT"
