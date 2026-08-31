"""What each adapter actually sends, and what it makes of what comes back."""

from __future__ import annotations

import warnings

import httpx
import pytest
from conftest import anthropic_response

from llm_client import LLMClient, LLMClientError, LLMConfigurationError, LLMConnectionError
from llm_client.providers import GroqProvider, MistralProvider, TogetherProvider


class TestOpenAICompatible:
    def test_system_is_prepended_as_a_message(self, fake_openai):
        LLMClient("openai").complete("hi", system="be terse")
        assert fake_openai.last["messages"] == [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ]

    def test_optional_parameters_are_omitted_when_unset(self, fake_openai):
        LLMClient("openai").complete("hi")
        assert "temperature" not in fake_openai.last
        assert "max_tokens" not in fake_openai.last
        assert "response_format" not in fake_openai.last

    def test_parameters_are_forwarded_when_set(self, fake_openai):
        LLMClient("openai").complete("hi", temperature=0.2, max_tokens=64)
        assert fake_openai.last["temperature"] == 0.2
        assert fake_openai.last["max_tokens"] == 64

    def test_json_mode_uses_the_native_response_format(self, fake_openai):
        LLMClient("openai").complete("hi", json_mode=True)
        assert fake_openai.last["response_format"] == {"type": "json_object"}

    def test_sdk_retries_are_disabled_so_this_package_owns_the_policy(self, fake_openai):
        LLMClient("openai").complete("hi")
        assert fake_openai.init_calls[0]["max_retries"] == 0

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("groq", GroqProvider.default_base_url),
            ("mistral", MistralProvider.default_base_url),
            ("together", TogetherProvider.default_base_url),
        ],
    )
    def test_compatible_providers_point_at_their_own_endpoint(self, fake_openai, name, expected):
        LLMClient(name).complete("hi")
        assert fake_openai.init_calls[0]["base_url"] == expected
        assert expected.startswith("https://")

    def test_base_url_can_be_overridden(self, fake_openai):
        LLMClient("openai", base_url="http://localhost:8000/v1").complete("hi")
        assert fake_openai.init_calls[0]["base_url"] == "http://localhost:8000/v1"


class TestAnthropic:
    def test_system_goes_to_the_system_parameter_not_the_messages(self, fake_anthropic):
        LLMClient("anthropic").complete(
            [{"role": "system", "content": "from message"}, {"role": "user", "content": "hi"}],
            system="from argument",
        )
        sent = fake_anthropic.last
        assert sent["system"] == "from argument\n\nfrom message"
        assert sent["messages"] == [{"role": "user", "content": "hi"}]

    def test_max_tokens_is_always_sent_because_the_api_requires_it(self, fake_anthropic):
        LLMClient("anthropic").complete("hi")
        assert fake_anthropic.last["max_tokens"] == 4096

    @pytest.mark.parametrize("model", ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-6"])
    def test_temperature_reaches_models_that_accept_it(self, fake_anthropic, model):
        """Sampling was removed from particular models, not from the API.

        The SDK dropped ``temperature`` from ``create()``'s signature, so it
        travels in ``extra_body`` rather than as a named argument.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # a warning here would be wrong
            LLMClient("anthropic", model).complete("hi", temperature=0.7)
        assert fake_anthropic.last["extra_body"] == {"temperature": 0.7}
        assert "temperature" not in fake_anthropic.last

    @pytest.mark.parametrize(
        "model",
        [
            "claude-sonnet-5",
            "claude-opus-5",
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-fable-5",
            "claude-mythos-5",
            "claude-opus-5-20260101",  # dated snapshots match by prefix
        ],
    )
    def test_temperature_is_dropped_where_it_would_be_a_400(self, fake_anthropic, model):
        with pytest.warns(UserWarning, match="rejects sampling parameters"):
            LLMClient("anthropic", model).complete("hi", temperature=0.7)
        assert "extra_body" not in fake_anthropic.last
        assert "temperature" not in fake_anthropic.last

    def test_dropping_temperature_warns_once_per_client(self, fake_anthropic):
        """Loud enough to notice, quiet enough not to be filtered out."""
        client = LLMClient("anthropic", "claude-sonnet-5")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(5):
                client.complete("hi", temperature=0.7)
        assert len(caught) == 1

        # ...but a fresh client says it again.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            LLMClient("anthropic", "claude-sonnet-5").complete("hi", temperature=0.7)
        assert len(caught) == 1

    def test_no_temperature_means_no_extra_body_and_no_warning(self, fake_anthropic):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            LLMClient("anthropic", "claude-sonnet-5").complete("hi")
        assert "extra_body" not in fake_anthropic.last

    def test_the_sampling_predicate_is_public(self):
        from llm_client import anthropic_accepts_sampling

        assert anthropic_accepts_sampling("claude-haiku-4-5") is True
        assert anthropic_accepts_sampling("claude-sonnet-4-6") is True
        assert anthropic_accepts_sampling("claude-sonnet-5") is False
        assert anthropic_accepts_sampling("claude-opus-4-8") is False

    def test_json_mode_is_an_instruction_because_there_is_no_native_flag(self, fake_anthropic):
        LLMClient("anthropic").complete("hi", json_mode=True)
        assert "valid JSON" in fake_anthropic.last["system"]

    def test_every_text_block_is_concatenated(self, fake_anthropic):
        import types

        fake_anthropic.response = types.SimpleNamespace(
            content=[
                types.SimpleNamespace(type="thinking", thinking="..."),
                types.SimpleNamespace(type="text", text="one "),
                types.SimpleNamespace(type="text", text="two"),
            ],
            model="claude-sonnet-5",
            stop_reason="end_turn",
            stop_details=None,
            usage=types.SimpleNamespace(input_tokens=1, output_tokens=2),
        )
        assert LLMClient("anthropic").complete("hi").text == "one two"

    def test_a_refusal_raises_rather_than_returning_empty_text(self, fake_anthropic):
        import types

        fake_anthropic.response = anthropic_response(text="", stop_reason="refusal")
        fake_anthropic.response.stop_details = types.SimpleNamespace(category="cyber")
        with pytest.raises(LLMClientError) as excinfo:
            LLMClient("anthropic").complete("hi")
        assert "cyber" in str(excinfo.value)

    def test_a_prompt_with_only_a_system_turn_is_rejected(self, fake_anthropic):
        with pytest.raises(Exception):
            LLMClient("anthropic").complete([{"role": "system", "content": "only this"}])


class TestGemini:
    def test_assistant_turns_are_renamed_to_model(self, fake_gemini):
        LLMClient("gemini").complete(
            [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        )
        assert fake_gemini.last["contents"] == [
            {"role": "user", "parts": [{"text": "a"}]},
            {"role": "model", "parts": [{"text": "b"}]},
        ]

    def test_system_becomes_a_system_instruction(self, fake_gemini):
        LLMClient("gemini").complete("hi", system="be terse")
        assert fake_gemini.last["config"]["system_instruction"] == "be terse"

    def test_json_mode_sets_the_response_mime_type(self, fake_gemini):
        LLMClient("gemini").complete("hi", json_mode=True)
        assert fake_gemini.last["config"]["response_mime_type"] == "application/json"

    def test_usage_is_read_from_the_metadata(self, fake_gemini):
        usage = LLMClient("gemini").complete("hi").usage
        assert (usage.input_tokens, usage.output_tokens) == (9, 4)


class TestOllama:
    def _transport(self, monkeypatch, handler):
        """Replace the transport, not the client: the request/response pairing
        that ``raise_for_status`` depends on has to stay intact."""
        captured: dict = {}

        def fake_post(self, url, json=None, **kwargs):
            captured["url"] = url
            captured["json"] = json
            response = handler()
            response.request = httpx.Request("POST", url, json=json)
            return response

        monkeypatch.setattr(httpx.Client, "post", fake_post)
        return captured

    def test_it_posts_a_chat_request(self, monkeypatch):
        captured = self._transport(
            monkeypatch,
            lambda: httpx.Response(
                200,
                json={
                    "model": "llama3.2",
                    "message": {"content": "hi"},
                    "prompt_eval_count": 4,
                    "eval_count": 2,
                },
            ),
        )
        result = LLMClient("ollama").complete("hello", temperature=0.1, json_mode=True)
        assert captured["url"] == "http://localhost:11434/api/chat"
        assert captured["json"]["stream"] is False
        assert captured["json"]["format"] == "json"
        assert captured["json"]["options"] == {"temperature": 0.1}
        assert (result.text, result.usage.input_tokens) == ("hi", 4)

    def test_base_url_comes_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://box:99")
        captured = self._transport(
            monkeypatch, lambda: httpx.Response(200, json={"message": {"content": "x"}})
        )
        LLMClient("ollama").complete("hello")
        assert captured["url"] == "http://box:99/api/chat"

    def test_a_dead_server_says_so(self, monkeypatch):
        def boom():
            raise httpx.ConnectError("refused")

        self._transport(monkeypatch, boom)
        with pytest.raises(LLMConnectionError) as excinfo:
            LLMClient("ollama").complete("hello")
        assert "Is Ollama running" in str(excinfo.value)


class TestMissingSDK:
    @pytest.mark.parametrize(
        ("provider", "module"),
        [("openai", "openai"), ("anthropic", "anthropic"), ("gemini", "google.genai")],
    )
    def test_missing_sdk_explains_the_install(self, monkeypatch, no_sdks, provider, module):
        monkeypatch.setenv(f"{provider.upper()}_API_KEY", "k")
        with pytest.raises(LLMConfigurationError) as excinfo:
            LLMClient(provider).complete("hi")
        message = str(excinfo.value)
        assert module in message
        assert 'pip install "llm-client[' in message

    def test_providers_without_an_sdk_still_work(self, no_sdks):
        assert LLMClient("mock").complete("hi").text.startswith("Mock response")


class TestAvailabilityProbe:
    """A probe answers; it never raises. Callers use it to pick a provider."""

    def test_ollama_probe_survives_a_non_httpx_failure(self, monkeypatch):
        from llm_client import provider_available

        def boom(*args, **kwargs):
            raise Exception("Connection refused")

        monkeypatch.setattr(httpx, "get", boom)
        assert provider_available("ollama") is False

    def test_ollama_probe_survives_an_httpx_failure(self, monkeypatch):
        from llm_client import provider_available

        def boom(*args, **kwargs):
            raise httpx.ConnectError("refused")

        monkeypatch.setattr(httpx, "get", boom)
        assert provider_available("ollama") is False

    def test_ollama_probe_reports_a_live_server(self, monkeypatch):
        from llm_client import provider_available

        monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(200))
        assert provider_available("ollama") is True
