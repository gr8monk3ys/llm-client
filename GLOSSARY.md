# Glossary

The domain language of `llm-client`. Terms are taken from the names in `src/llm_client/`.

## Language

**LLMClient**:
The single public entry point. Resolves a **Provider**, wraps it with retry and an optional **ResponseCache**, and exposes `complete()` and `acomplete()`.
_Avoid_: wrapper, SDK

**Provider**:
One vendor adapter, subclass of `Provider`. Translates the request shape and the error shape for one vendor and does nothing else; retry, caching and env resolution belong to **LLMClient**.
_Avoid_: backend (except `LLMClient.backend`, the accessor that returns the adapter), vendor class

**Provider name**:
The registry key in `PROVIDERS` (`openai`, `anthropic`, `gemini`, `groq`, `mistral`, `together`, `ollama`, `mock`). The aliases `google`, `claude` and `gpt` resolve to `gemini`, `anthropic` and `openai`.

**OpenAI-compatible provider**:
A **Provider** that speaks the OpenAI chat API against a different endpoint: `groq`, `mistral`, `together`, and `openai` itself.

**MockProvider**:
The deterministic, offline **Provider** named `mock`. Never returned by `available_providers()`.

**Message**:
One chat turn: a `role` and a `content` string.

**Role**:
`system`, `user` or `assistant`. A `system` role inside a message list is accepted, but the `system` argument is preferred.

**Completion**:
What every **Provider** returns: `text`, `model`, `provider`, a **Usage**, and `raw`, the untouched SDK response.

**Usage**:
Input and output token counts. Fields are `None` when the provider does not report them.

**json_mode**:
A flag asking for JSON output. Native where the provider supports it, an added instruction where it does not (Anthropic). The result is not parsed or validated.

**ResponseCache**:
The optional per-client TTL cache of completion text. A cache hit yields a **Completion** with an empty **Usage** and `raw=None`.

**Cache key**:
The SHA-256 of provider, model, messages, system prompt and sampling params, serialised with sorted keys.

**Retryable error**:
An `LLMProviderError` whose `retryable` flag is true: rate limit (429), timeout, connection failure, and the statuses in `RETRYABLE_STATUS_CODES` (429, 500, 502, 503, 504, 529). Everything else, including an unexpected provider bug, is not retried.

**Client error**:
`LLMClientError`: a 4xx other than 429, such as a bad key or bad request. Never retried.

**Sampling-removed model**:
An Anthropic model whose id starts with one of `SAMPLING_REMOVED_PREFIXES`; it rejects `temperature` with an HTTP 400. `anthropic_accepts_sampling()` answers this.

## Relationships

- An **LLMClient** owns exactly one **Provider** and at most one **ResponseCache**.
- A **Provider** turns a list of **Messages** into one **Completion**.
- A failure leaves a **Provider** as an `LLMError` subclass, so callers never import a vendor SDK to catch it.
