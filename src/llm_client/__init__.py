"""One small interface over several LLM providers.

    from llm_client import LLMClient

    client = LLMClient("anthropic", "claude-sonnet-5")
    print(client.complete("Name three primes.").text)

The provider list, the retry policy and the optional cache are documented in
the README. Everything below is the supported public surface.
"""

from .cache import ResponseCache
from .client import LLMClient
from .errors import (
    LLMClientError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
    is_retryable,
)
from .providers import (
    ANTHROPIC_MOST_CAPABLE_MODEL,
    PROVIDERS,
    Provider,
    available_providers,
    get_provider_class,
    provider_available,
)
from .types import Completion, Message, Usage

__version__ = "0.1.1"

__all__ = [
    "LLMClient",
    "Completion",
    "Message",
    "Usage",
    "Provider",
    "PROVIDERS",
    "get_provider_class",
    "available_providers",
    "provider_available",
    "ANTHROPIC_MOST_CAPABLE_MODEL",
    "ResponseCache",
    "LLMError",
    "LLMConfigurationError",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMConnectionError",
    "LLMServerError",
    "LLMClientError",
    "is_retryable",
    "__version__",
]
