"""Which failures are worth another attempt, and which are not."""

from __future__ import annotations

import httpx
import pytest

from llm_client import (
    LLMClient,
    LLMClientError,
    LLMConnectionError,
    LLMProviderError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
    is_retryable,
)


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    """Backoff is correct but slow; the tests care about counts, not clocks."""
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda _seconds: None)


def _errors(fake):
    import sys

    return sys.modules["openai"]


class TestClassification:
    def test_rate_limit_is_retryable(self, fake_openai):
        sdk = _errors(fake_openai)
        fake_openai.fail_times(1, sdk.RateLimitError("slow down"))
        assert LLMClient("openai").complete("hi").text == "hi"
        assert fake_openai.call_count == 2

    def test_server_error_is_retryable(self, fake_openai):
        sdk = _errors(fake_openai)
        fake_openai.fail_times(2, sdk.InternalServerError("boom"))
        assert LLMClient("openai").complete("hi").text == "hi"
        assert fake_openai.call_count == 3

    def test_timeout_is_retryable(self, fake_openai):
        sdk = _errors(fake_openai)
        fake_openai.fail_times(1, sdk.APITimeoutError("too slow"))
        LLMClient("openai").complete("hi")
        assert fake_openai.call_count == 2

    def test_auth_failure_is_not_retried(self, fake_openai):
        sdk = _errors(fake_openai)
        fake_openai.fail_times(5, sdk.AuthenticationError("bad key"))
        with pytest.raises(LLMClientError):
            LLMClient("openai").complete("hi")
        assert fake_openai.call_count == 1

    def test_bad_request_is_not_retried(self, fake_openai):
        sdk = _errors(fake_openai)
        fake_openai.fail_times(5, sdk.BadRequestError("nope"))
        with pytest.raises(LLMClientError):
            LLMClient("openai").complete("hi")
        assert fake_openai.call_count == 1

    def test_anthropic_errors_classify_the_same_way(self, fake_anthropic):
        import sys

        sdk = sys.modules["anthropic"]
        fake_anthropic.fail_times(1, sdk.RateLimitError("slow down"))
        assert LLMClient("anthropic").complete("hi").text == "hi"
        assert fake_anthropic.call_count == 2


class TestExhaustion:
    def test_the_provider_error_survives_the_retry_wrapper(self, fake_openai):
        sdk = _errors(fake_openai)
        fake_openai.fail_times(10, sdk.RateLimitError("still limited"))
        with pytest.raises(LLMRateLimitError):
            LLMClient("openai", max_retries=2).complete("hi")
        assert fake_openai.call_count == 3

    def test_max_retries_zero_means_one_attempt(self, fake_openai):
        sdk = _errors(fake_openai)
        fake_openai.fail_times(10, sdk.InternalServerError("boom"))
        with pytest.raises(LLMServerError):
            LLMClient("openai", max_retries=0).complete("hi")
        assert fake_openai.call_count == 1


class TestPredicate:
    @pytest.mark.parametrize(
        "error",
        [
            LLMRateLimitError("x"),
            LLMServerError("x", status_code=503),
            LLMTimeoutError("x"),
            LLMConnectionError("x"),
            httpx.ConnectError("x"),
            httpx.ReadTimeout("x"),
        ],
    )
    def test_retryable(self, error):
        assert is_retryable(error) is True

    @pytest.mark.parametrize(
        "error",
        [LLMClientError("x", status_code=401), ValueError("x"), KeyError("x")],
    )
    def test_not_retryable(self, error):
        assert is_retryable(error) is False

    def test_httpx_status_errors_split_on_the_code(self):
        request = httpx.Request("POST", "http://example.invalid")
        retryable = httpx.HTTPStatusError(
            "x", request=request, response=httpx.Response(503, request=request)
        )
        fatal = httpx.HTTPStatusError(
            "x", request=request, response=httpx.Response(401, request=request)
        )
        assert is_retryable(retryable) is True
        assert is_retryable(fatal) is False


class TestAsyncRetry:
    async def test_async_calls_retry_too(self, fake_openai):
        sdk = _errors(fake_openai)
        fake_openai.fail_times(1, sdk.InternalServerError("boom"))
        assert (await LLMClient("openai").acomplete("hi")).text == "hi"
        assert fake_openai.call_count == 2


class TestUnclassifiedFailures:
    """Retrying is the dangerous default: only known-transient failures repeat."""

    def test_a_bug_inside_a_provider_is_not_retried(self, fake_openai, monkeypatch):
        def broken(*args, **kwargs):
            raise KeyError("unexpected response shape")

        monkeypatch.setattr("llm_client.providers._OpenAICompatible._completion", broken)
        with pytest.raises(Exception) as excinfo:
            LLMClient("openai").complete("hi")
        assert isinstance(excinfo.value, KeyError)
        assert fake_openai.call_count == 1

    def test_an_error_with_no_status_is_not_retried(self, fake_openai):
        sdk = _errors(fake_openai)
        fake_openai.fail_times(5, sdk.APIError("something odd"))
        with pytest.raises(LLMProviderError):
            LLMClient("openai").complete("hi")
        assert fake_openai.call_count == 1

    def test_an_unrecognised_5xx_is_retried(self, fake_openai):
        sdk = _errors(fake_openai)
        error = sdk.APIError("teapot on fire")
        error.status_code = 507
        fake_openai.fail_times(1, error)
        LLMClient("openai").complete("hi")
        assert fake_openai.call_count == 2

    def test_an_unrecognised_4xx_is_not_retried(self, fake_openai):
        sdk = _errors(fake_openai)
        error = sdk.APIError("gone")
        error.status_code = 410
        fake_openai.fail_times(5, error)
        with pytest.raises(LLMProviderError):
            LLMClient("openai").complete("hi")
        assert fake_openai.call_count == 1

    def test_529_overloaded_is_retried(self, fake_anthropic):
        import sys

        sdk = sys.modules["anthropic"]
        error = sdk.APIError("overloaded")
        error.status_code = 529
        fake_anthropic.fail_times(1, error)
        assert LLMClient("anthropic").complete("hi").text == "hi"
        assert fake_anthropic.call_count == 2

    def test_a_bare_provider_error_is_not_retryable_by_default(self):
        assert is_retryable(LLMProviderError("who knows")) is False
        assert is_retryable(LLMProviderError("known bad", retryable=True)) is True


class TestBackoffShape:
    def test_the_exponential_base_is_configurable(self, monkeypatch):
        """resume-AI exposes llm_retry_exponential_base; it has to reach tenacity."""
        captured = {}
        import llm_client.retry as retry_module

        real = retry_module.wait_exponential_jitter

        def spy(**kwargs):
            captured.update(kwargs)
            return real(**kwargs)

        monkeypatch.setattr(retry_module, "wait_exponential_jitter", spy)
        LLMClient(
            "mock",
            max_retries=4,
            retry_initial_delay=0.5,
            retry_max_delay=12.0,
            retry_exp_base=3.0,
        )
        assert captured == {"initial": 0.5, "max": 12.0, "exp_base": 3.0}
