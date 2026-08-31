"""Which failures are worth another attempt, and which are not."""

from __future__ import annotations

import httpx
import pytest

from llm_client import (
    LLMClient,
    LLMClientError,
    LLMConnectionError,
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
