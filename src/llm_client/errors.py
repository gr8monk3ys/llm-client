"""Exception hierarchy and the retry classification that hangs off it.

Every provider failure is normalised into one of these before it leaves the
package, so callers never have to import a vendor SDK to catch an error, and
:func:`is_retryable` has a single thing to look at.
"""

from __future__ import annotations

import httpx

__all__ = [
    "LLMError",
    "LLMConfigurationError",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMConnectionError",
    "LLMServerError",
    "LLMClientError",
    "RETRYABLE_STATUS_CODES",
    "CLIENT_ERROR_STATUS_CODES",
    "is_retryable",
]


class LLMError(Exception):
    """Base exception for everything this package raises."""


class LLMConfigurationError(LLMError):
    """Misconfiguration: unknown provider, missing API key, missing SDK."""


class LLMProviderError(LLMError):
    """A provider returned an error.

    ``retryable`` is the single source of truth for the retry policy; the
    subclasses below only exist to set it (and ``status_code``) correctly.

    It defaults to ``False``. Retrying is the dangerous default: a bug inside
    a provider - a ``KeyError`` on an unexpected response shape - is not
    flaky, and retrying it four times only turns one clear failure into a
    slow one. Something must be *known* to be transient to be retried.
    """

    def __init__(self, message: str, status_code: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class LLMRateLimitError(LLMProviderError):
    """Rate limited by the provider (HTTP 429)."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message, status_code=429, retryable=True)
        self.retry_after = retry_after


class LLMTimeoutError(LLMProviderError):
    """The request timed out."""

    def __init__(self, message: str):
        super().__init__(message, status_code=None, retryable=True)


class LLMConnectionError(LLMProviderError):
    """The provider could not be reached."""

    def __init__(self, message: str):
        super().__init__(message, status_code=None, retryable=True)


class LLMServerError(LLMProviderError):
    """The provider returned a 5xx."""

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message, status_code=status_code, retryable=True)


class LLMClientError(LLMProviderError):
    """The provider returned a 4xx other than 429 - a bad key, a bad request.

    Never retried: retrying an auth failure just spends the rate limit.
    """

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message, status_code=status_code, retryable=False)


#: Statuses worth trying again. 529 is Anthropic's "overloaded", which is
#: explicitly a back-off-and-retry signal.
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504, 529})

#: Statuses that will fail identically on every attempt.
CLIENT_ERROR_STATUS_CODES = frozenset({400, 401, 403, 404, 405, 422})


def is_retryable(exc: BaseException) -> bool:
    """Whether ``exc`` should trigger another attempt.

    Providers normalise their own SDK exceptions before raising, so in practice
    this only ever sees an :class:`LLMProviderError`. The ``httpx`` arm is the
    safety net for anything raised outside a provider's ``try`` block.
    """
    if isinstance(exc, LLMProviderError):
        return exc.retryable
    if isinstance(exc, httpx.TimeoutException | httpx.ConnectError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS_CODES
    return False
