---
status: accepted
date: 2026-11-22
---

# Provider errors are not retried unless known to be transient

`LLMProviderError.retryable` defaults to `False`. Only rate limits, timeouts, connection failures and the 5xx statuses in `RETRYABLE_STATUS_CODES` set it to true, and `is_retryable()` in `src/llm_client/errors.py` is the single place the retry policy reads it.

A bug inside a provider adapter (for example a `KeyError` on an unexpected response shape) is not flaky, and retrying it only turns one clear failure into a slow one. Retrying an auth failure also spends the rate limit. Retrying is therefore opt-in per error type rather than the default.

Consequence: a new transient failure mode must be added explicitly as a retryable error, otherwise it surfaces on the first attempt.
