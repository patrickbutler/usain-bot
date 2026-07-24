"""Anthropic (Claude) implementation of LLMProvider. This is the only
file in the codebase that imports the `anthropic` SDK — everything else
talks to `LLMProvider`."""

from __future__ import annotations

import os

from .base import (
    ChatMessage,
    ContentBlock,
    LLMProvider,
    LLMProviderError,
    ProviderResponse,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_MAX_TOKENS = 1536


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - dependency wiring
            raise LLMProviderError("`anthropic` is not installed. `pip install anthropic`.") from exc

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    @classmethod
    def from_env(cls, model: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS) -> "AnthropicProvider":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMProviderError(
                "ANTHROPIC_API_KEY is not set. Chat is optional — the rest of usain-bot works "
                "without it — but the chat tab needs it (see .env.example)."
            )
        return cls(api_key=api_key, model=model, max_tokens=max_tokens)

    @property
    def model_name(self) -> str:
        return self._model

    def run_turn(self, system: str, messages: list[ChatMessage], tools: list[ToolSpec]) -> ProviderResponse:
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[_to_anthropic_message(m) for m in messages],
                tools=[_to_anthropic_tool(t) for t in tools],
            )
        except Exception as exc:  # noqa: BLE001 - surface any SDK/network failure uniformly
            raise LLMProviderError(f"Anthropic API call failed: {exc}") from exc

        content = [_from_anthropic_block(b) for b in response.content]
        stop_reason = response.stop_reason if response.stop_reason in ("end_turn", "tool_use", "max_tokens") else "error"
        return ProviderResponse(content=content, stop_reason=stop_reason)


def _to_anthropic_message(message: ChatMessage) -> dict:
    return {"role": message.role, "content": [_to_anthropic_block(b) for b in message.content]}


def _to_anthropic_block(block: ContentBlock) -> dict:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ToolUseBlock):
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    if isinstance(block, ToolResultBlock):
        return {
            "type": "tool_result", "tool_use_id": block.tool_use_id,
            "content": block.content, "is_error": block.is_error,
        }
    raise TypeError(f"Unknown content block type: {type(block)}")


def _from_anthropic_block(block) -> ContentBlock:
    if block.type == "text":
        return TextBlock(text=block.text)
    if block.type == "tool_use":
        return ToolUseBlock(id=block.id, name=block.name, input=block.input)
    raise TypeError(f"Unexpected Anthropic response block type: {block.type}")


def _to_anthropic_tool(tool: ToolSpec) -> dict:
    return {"name": tool.name, "description": tool.description, "input_schema": tool.input_schema}
