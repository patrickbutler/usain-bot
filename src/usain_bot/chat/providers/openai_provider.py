"""OpenAI (ChatGPT) implementation of LLMProvider — the default chat
provider. This is the only file in the codebase that imports the
`openai` SDK — everything else talks to `LLMProvider`.

Translation note: OpenAI's Chat Completions API represents a tool
result as its own message with `role: "tool"`, not — as Anthropic
does — a content block inside a `role: "user"` message. The
provider-agnostic orchestration loop (chat/session.py) only knows the
normalized convention "a `user`-role ChatMessage whose content is
ToolResultBlocks carries tool results back to the model"; it's this
file's job to expand that into however many separate `role: "tool"`
messages OpenAI expects. Nothing upstream of this file needs to know
that distinction exists.
"""

from __future__ import annotations

import json
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

DEFAULT_MODEL = "gpt-4o"
DEFAULT_MAX_TOKENS = 1536


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS):
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - dependency wiring
            raise LLMProviderError("`openai` is not installed. `pip install openai`.") from exc

        self._client = openai.OpenAI(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    @classmethod
    def from_env(cls, model: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS) -> "OpenAIProvider":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise LLMProviderError(
                "OPENAI_API_KEY is not set. Chat is optional — the rest of usain-bot works "
                "without it — but the chat tab needs it (see .env.example)."
            )
        return cls(api_key=api_key, model=model, max_tokens=max_tokens)

    @property
    def model_name(self) -> str:
        return self._model

    def run_turn(self, system: str, messages: list[ChatMessage], tools: list[ToolSpec]) -> ProviderResponse:
        wire_messages = [{"role": "system", "content": system}] + _to_openai_messages(messages)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_completion_tokens=self._max_tokens,
                messages=wire_messages,
                tools=[_to_openai_tool(t) for t in tools] if tools else None,
            )
        except Exception as exc:  # noqa: BLE001 - surface any SDK/network failure uniformly
            raise LLMProviderError(f"OpenAI API call failed: {exc}") from exc

        choice = response.choices[0]
        content = _from_openai_message(choice.message)
        stop_reason = _map_finish_reason(choice.finish_reason)
        return ProviderResponse(content=content, stop_reason=stop_reason)


def _map_finish_reason(finish_reason: str) -> str:
    if finish_reason == "tool_calls":
        return "tool_use"
    if finish_reason == "stop":
        return "end_turn"
    if finish_reason == "length":
        return "max_tokens"
    return "error"


def _to_openai_messages(messages: list[ChatMessage]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        texts = [b.text for b in m.content if isinstance(b, TextBlock)]
        tool_uses = [b for b in m.content if isinstance(b, ToolUseBlock)]
        tool_results = [b for b in m.content if isinstance(b, ToolResultBlock)]

        if tool_results:
            # Our normalized convention bundles tool results into one
            # user-role message; OpenAI wants one role="tool" message per
            # result, keyed by tool_call_id.
            for tr in tool_results:
                out.append({"role": "tool", "tool_call_id": tr.tool_use_id, "content": tr.content})
            continue

        msg: dict = {"role": m.role, "content": "\n".join(texts) if texts else (None if tool_uses else "")}
        if tool_uses:
            msg["tool_calls"] = [
                {
                    "id": tu.id, "type": "function",
                    "function": {"name": tu.name, "arguments": json.dumps(tu.input)},
                }
                for tu in tool_uses
            ]
        out.append(msg)
    return out


def _from_openai_message(message) -> list[ContentBlock]:
    blocks: list[ContentBlock] = []
    if message.content:
        blocks.append(TextBlock(text=message.content))
    if message.tool_calls:
        for call in message.tool_calls:
            try:
                args = json.loads(call.function.arguments) if call.function.arguments else {}
            except json.JSONDecodeError:
                args = {}
            blocks.append(ToolUseBlock(id=call.id, name=call.function.name, input=args))
    return blocks


def _to_openai_tool(tool: ToolSpec) -> dict:
    return {
        "type": "function",
        "function": {"name": tool.name, "description": tool.description, "parameters": tool.input_schema},
    }
