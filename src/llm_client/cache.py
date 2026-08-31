"""Optional in-memory TTL cache, keyed on everything that changes the answer.

Per-client, not module-global: two clients with different providers, models or
temperatures never see each other's entries even if the key derivation is wrong.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from cachetools import TTLCache

from .types import Message

__all__ = ["ResponseCache", "DEFAULT_MAXSIZE", "DEFAULT_TTL"]

DEFAULT_MAXSIZE = 100
DEFAULT_TTL = 3600.0


class ResponseCache:
    """A TTL cache of completion text.

    Only the text is cached. Usage counts describe the call that was actually
    billed, so replaying them for a cache hit would be a lie; a cached
    ``Completion`` carries an empty ``Usage`` and ``raw=None``.
    """

    def __init__(self, maxsize: int = DEFAULT_MAXSIZE, ttl: float = DEFAULT_TTL):
        self._cache: TTLCache[str, str] = TTLCache(maxsize=maxsize, ttl=ttl)

    @staticmethod
    def key(
        provider: str,
        model: str,
        messages: list[Message],
        system: str | None,
        params: dict[str, Any],
    ) -> str:
        """Hash every input that can change the response.

        ``sort_keys`` matters: an unsorted ``json.dumps`` makes the key depend
        on dict insertion order and silently halves the hit rate.
        """
        payload = json.dumps(
            {
                "provider": provider,
                "model": model,
                "messages": messages,
                "system": system,
                "params": params,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> str | None:
        return self._cache.get(key)

    def set(self, key: str, text: str) -> None:
        self._cache[key] = text

    def clear(self) -> None:
        self._cache.clear()

    def stats(self) -> dict[str, Any]:
        return {
            "current_size": len(self._cache),
            "max_size": self._cache.maxsize,
            "ttl_seconds": self._cache.ttl,
        }

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: object) -> bool:
        return key in self._cache
