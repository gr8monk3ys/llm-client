"""Exponential backoff with jitter, applied to sync and async calls alike.

tenacity, configured once and reused, with only the knobs ``LLMClient``
actually exposes.
"""

from __future__ import annotations

import logging
from typing import Any

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .errors import is_retryable

__all__ = [
    "build_retry",
    "DEFAULT_INITIAL_DELAY",
    "DEFAULT_MAX_DELAY",
    "DEFAULT_EXP_BASE",
]

logger = logging.getLogger(__name__)

DEFAULT_INITIAL_DELAY = 1.0
DEFAULT_MAX_DELAY = 30.0
DEFAULT_EXP_BASE = 2.0


def build_retry(
    max_retries: int,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    exp_base: float = DEFAULT_EXP_BASE,
) -> Any:
    """Build a tenacity decorator that retries only retryable failures.

    ``max_retries`` counts *retries*, so ``stop_after_attempt`` gets one more
    than that - the first call is not a retry. ``reraise=True`` means callers
    see the provider's own error, never a ``RetryError`` wrapper.

    tenacity dispatches on whether the wrapped callable is a coroutine
    function, so one decorator covers ``complete`` and ``acomplete``.
    """
    return retry(
        retry=retry_if_exception(is_retryable),
        stop=stop_after_attempt(max_retries + 1),
        wait=wait_exponential_jitter(initial=initial_delay, max=max_delay, exp_base=exp_base),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
