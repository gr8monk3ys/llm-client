# llm-client

One small interface over OpenAI, Anthropic, Gemini, Ollama and the
OpenAI-compatible endpoints (Groq, Mistral, Together), with retry, an optional
TTL cache, and one exception hierarchy for all of them.

You get `LLMClient.complete()`. Everything vendor-shaped — SDK imports, request
shapes, token-usage field names, which HTTP codes are worth retrying — stays
behind it.

> **Not the `llm-client` on PyPI.** The name is taken by an unrelated package
> (`llm-client` 0.8.0, "SDK for using LLM"). `pip install llm-client` installs
> that one, not this one. Install from git, as below.

## Install

```bash
pip install "llm-client[all] @ git+https://github.com/gr8monk3ys/llm-client@v0.1.2"
```

Provider SDKs are optional extras — take only what you use:
`[openai]` (also covers groq, mistral and together), `[anthropic]`, `[gemini]`,
or `[all]`. Ollama and the mock provider need no SDK at all. Importing a
provider without its SDK raises `LLMConfigurationError` naming the extra.

## Usage

```python
from llm_client import LLMClient

client = LLMClient("anthropic", "claude-sonnet-5", cache=True)

result = client.complete("Name three prime numbers.", system="Answer tersely.")
print(result.text)  # "2, 3, 5"
print(result.usage.total_tokens)  # 34
print(result.provider, result.model)  # anthropic claude-sonnet-5

# A message list works wherever a bare prompt does.
result = client.complete(
    [{"role": "user", "content": "Give me a JSON object with one key, 'ok'."}],
    json_mode=True,
    max_tokens=64,
)

# Same call, async.
result = await client.acomplete("Name three primes.")
```

With no arguments, `LLMClient()` reads `LLM_PROVIDER` and `LLM_MODEL` from the
environment, so a deployment can change model without a code change.

## Providers

| `provider` | Extra | API key | Default model | Endpoint |
|---|---|---|---|---|
| `openai` | `[openai]` | `OPENAI_API_KEY` | `gpt-4o` | OpenAI |
| `anthropic` | `[anthropic]` | `ANTHROPIC_API_KEY` | `claude-sonnet-5` | Anthropic Messages |
| `gemini` (alias `google`) | `[gemini]` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | `gemini-1.5-pro` | Google GenAI |
| `groq` | `[openai]` | `GROQ_API_KEY` | `llama-3.3-70b-versatile` | `api.groq.com/openai/v1` |
| `mistral` | `[openai]` | `MISTRAL_API_KEY` | `mistral-large-latest` | `api.mistral.ai/v1` |
| `together` | `[openai]` | `TOGETHER_API_KEY` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | `api.together.xyz/v1` |
| `ollama` | — | none | `llama3.2` | `OLLAMA_BASE_URL`, default `http://localhost:11434` |
| `mock` | — | none | `mock-model` | none — deterministic, offline |

Groq, Mistral and Together all speak the OpenAI chat-completions protocol, so
they share one adapter and one dependency; `base_url` is the only difference.

For Anthropic, `claude-sonnet-5` is the default and `ANTHROPIC_MOST_CAPABLE_MODEL`
(`claude-fable-5`) names the model to reach for when capability matters more
than cost.

**`temperature` on Anthropic depends on the model.** Sampling was removed from
particular models, not from the Messages API. Opus 4.6, Sonnet 4.6, Sonnet 4.5,
Haiku 4.5 and older still honour it, and this package forwards it (through
`extra_body`, because the SDK dropped `temperature` from `messages.create()`'s
signature when the newer models lost it). Fable 5, Mythos 5, Opus 5, Opus 4.8,
Opus 4.7 and Sonnet 5 reject it with an HTTP 400, so it is dropped and a
`UserWarning` is raised once per client. `anthropic_accepts_sampling(model)`
answers the question directly, and `SAMPLING_REMOVED_PREFIXES` is the list.
Every other provider takes `temperature` as normal. Every other provider's default is inherited from the projects this
package was extracted from and may be behind that vendor's current line — set
`LLM_MODEL` or pass `model=` rather than relying on it.

### Configuration

Resolution order for the model: the `model=` argument, then `<PROVIDER>_MODEL`
(e.g. `ANTHROPIC_MODEL`), then `LLM_MODEL`, then the provider's default. The
provider-specific variable wins because it is the more specific statement of
intent. The provider itself comes from `provider=` or `LLM_PROVIDER`, defaulting
to `openai`.

## Retry

`max_retries` (default 3) retries with exponential backoff and jitter on rate
limits, 5xx (529 included — Anthropic's "overloaded" is a back-off signal),
timeouts and connection failures. Tune the curve with `retry_initial_delay`,
`retry_max_delay` and `retry_exp_base`. Provider SDKs are constructed with
their own retries disabled so the two policies cannot multiply.

**Only known-transient failures are retried.** A 4xx is not: a bad key or a
malformed request fails identically every time, and retrying one only spends
the rate limit. Neither is anything unclassified — an exception carrying no
HTTP status is a bug, here or in an SDK, and retrying a `KeyError` four times
turns one clear failure into a slow one. `LLMProviderError` therefore defaults
to `retryable=False`; something must be *known* to be transient to repeat.

Every failure arrives as one of these, whatever SDK raised it:

| Exception | Raised for | Retried |
|---|---|---|
| `LLMConfigurationError` | unknown provider, missing key, missing SDK | — |
| `LLMRateLimitError` | 429 | yes |
| `LLMServerError` | 5xx | yes |
| `LLMTimeoutError` | request timeout | yes |
| `LLMConnectionError` | provider unreachable | yes |
| `LLMClientError` | 4xx other than 429, and an Anthropic refusal | no |

All of them subclass `LLMError`; the retryable ones subclass `LLMProviderError`.

## Cache

`cache=True` turns on a per-client in-memory TTL cache (`cache_ttl`, default one
hour; `cache_maxsize`, default 100), keyed on provider, model, messages, system
prompt and every request parameter. It is off by default, and it is per client
instance — two clients never share entries. A cache hit returns a `Completion`
with empty `usage` and `raw=None`, because nothing was billed for it.

## JSON

`json_mode=True` asks for JSON natively where the provider supports it
(`response_format` on the OpenAI-compatible providers, `response_mime_type` on
Gemini, `format` on Ollama) and appends an instruction to the system prompt on
Anthropic, which has no equivalent flag. It does not parse or validate the
result — that is the caller's job.

## Development

```bash
uv sync --group dev
uv run pytest
```

The tests install fake SDK modules into `sys.modules`, so the whole suite runs
with no provider installed and no network.

## License

MIT
