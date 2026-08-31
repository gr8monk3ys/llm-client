"""The data that crosses the package boundary: a message in, a completion out."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

__all__ = ["Message", "Role", "Usage", "Completion", "normalize_messages"]

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    """One chat turn. ``system`` is accepted here but prefer the ``system`` argument."""

    role: Role
    content: str


@dataclass(frozen=True)
class Usage:
    """Token accounting. Fields are ``None`` when the provider does not report them."""

    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)


@dataclass(frozen=True)
class Completion:
    """What every provider returns, whatever shape its SDK used.

    ``raw`` is the untouched SDK response object, for the rare caller that needs
    something this package does not model. Everything else is normalised.
    """

    text: str
    model: str
    provider: str
    usage: Usage = field(default_factory=Usage)
    raw: Any = None

    def __str__(self) -> str:
        return self.text


def normalize_messages(
    messages: str | Sequence[Message] | Sequence[dict[str, Any]],
) -> list[Message]:
    """Accept a bare prompt or a message list; always return a message list.

    This is the whole reason ``complete()`` takes one positional argument
    instead of two mutually exclusive keyword arguments.
    """
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]

    if not isinstance(messages, Sequence):
        raise TypeError(
            f"messages must be a str or a sequence of messages, got {type(messages).__name__}"
        )

    normalized: list[Message] = []
    for i, message in enumerate(messages):
        try:
            role = message["role"]
            content = message["content"]
        except (TypeError, KeyError) as exc:
            raise TypeError(f"messages[{i}] must be a mapping with 'role' and 'content'") from exc
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"messages[{i}] has unsupported role {role!r}")
        normalized.append({"role": role, "content": str(content)})

    if not normalized:
        raise ValueError("messages must not be empty")
    return normalized
