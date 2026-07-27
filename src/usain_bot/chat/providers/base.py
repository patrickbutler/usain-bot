"""Provider-agnostic chat types + interface.

Mirrors the StorageBackend/GarminAdapter pattern used elsewhere: the
orchestration loop (chat/session.py) and tool dispatch (chat/tools.py)
only ever depend on `LLMProvider` and these normalized dataclasses,
never on a concrete SDK. OpenAI (openai_provider.py) is the default;
Gemini (gemini_provider.py) and Anthropic (anthropic_provider.py) are
swappable alternates — adding another vendor means adding one new file
under chat/providers/ and a factory entry — nothing in the
orchestration loop, tool definitions, or web/chat layer changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class TextBlock:
    text: str
    type: Literal["text"] = "text"


@dataclass(frozen=True)
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: Literal["tool_use"] = "tool_use"


@dataclass(frozen=True)
class ToolResultBlock:
    tool_use_id: str
    content: str
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"


ContentBlock = Union[TextBlock, ToolUseBlock, ToolResultBlock]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: list[ContentBlock]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


StopReason = Literal["end_turn", "tool_use", "max_tokens", "error"]


@dataclass(frozen=True)
class ProviderResponse:
    content: list[ContentBlock]
    stop_reason: StopReason
    raw_error: Optional[str] = None


class LLMProviderError(Exception):
    """Raised for provider-level failures (auth, network, bad config) so
    the web layer can surface a clean message instead of a stack trace."""


class LLMProvider(ABC):
    """One request/response turn against a chat-capable LLM with tool use.
    Implementations are stateless — all conversation state lives in the
    `messages` list the caller passes in, so the orchestration loop can
    swap providers mid-conversation if it ever needed to."""

    @abstractmethod
    def run_turn(
        self,
        system: str,
        messages: list[ChatMessage],
        tools: list[ToolSpec],
    ) -> ProviderResponse:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...
