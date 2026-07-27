"""Google Gemini implementation of LLMProvider. This is the only file
in the codebase that imports the `google-genai` SDK — everything else
talks to `LLMProvider`.

Translation notes (Gemini's function-calling shape differs from both
OpenAI's and Anthropic's):

- Roles are "user" / "model", not "user" / "assistant" — the assistant
  role in our normalized ChatMessage maps to Gemini's "model".
- A tool call is a `Part` with a `function_call` (name + args dict,
  args already a dict, not a JSON string like OpenAI's). A tool result
  is a `Part` with a `function_response` keyed by the function *name*,
  not an opaque call id the way OpenAI/Anthropic key on `tool_call_id`.
  Our normalized `ToolResultBlock` only carries `tool_use_id`, so this
  file reconstructs the id->name mapping by scanning the ToolUseBlocks
  already present earlier in the same `messages` list before it can
  build a FunctionResponse part.
- Finish reason doesn't have a dedicated "the model wants to call a
  tool" value the way OpenAI's `finish_reason="tool_calls"` does —
  Gemini reports "STOP" either way, so whether a turn is `tool_use` is
  inferred from *content* (did any part carry a function_call), not the
  finish reason.
"""

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

DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_MAX_TOKENS = 1536


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS):
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - dependency wiring
            raise LLMProviderError("`google-genai` is not installed. `pip install google-genai`.") from exc

        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    @classmethod
    def from_env(cls, model: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS) -> "GeminiProvider":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise LLMProviderError(
                "GEMINI_API_KEY is not set. Chat is optional — the rest of usain-bot works "
                "without it — but the chat tab needs it (see .env.example)."
            )
        return cls(api_key=api_key, model=model, max_tokens=max_tokens)

    @property
    def model_name(self) -> str:
        return self._model

    def run_turn(self, system: str, messages: list[ChatMessage], tools: list[ToolSpec]) -> ProviderResponse:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=self._max_tokens,
            tools=[types.Tool(function_declarations=[_to_gemini_function_declaration(t) for t in tools])] if tools else None,
            # We drive the tool-call loop ourselves (chat/session.py) so the
            # SDK must not try to execute functions on our behalf.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=_to_gemini_contents(messages),
                config=config,
            )
        except Exception as exc:  # noqa: BLE001 - surface any SDK/network failure uniformly
            raise LLMProviderError(f"Gemini API call failed: {exc}") from exc

        if not response.candidates:
            raise LLMProviderError("Gemini returned no candidates (likely blocked by safety filters).")

        content, stop_reason = _from_gemini_candidate(response.candidates[0])
        return ProviderResponse(content=content, stop_reason=stop_reason)


def _to_gemini_contents(messages: list[ChatMessage]) -> list:
    from google.genai import types

    # Tool results carry only the opaque call id; Gemini's function
    # response needs the function *name*, so recover it from whichever
    # ToolUseBlock earlier in the conversation had a matching id.
    id_to_name: dict[str, str] = {}
    for m in messages:
        for b in m.content:
            if isinstance(b, ToolUseBlock):
                id_to_name[b.id] = b.name

    contents = []
    for m in messages:
        role = "model" if m.role == "assistant" else "user"
        parts = []
        for b in m.content:
            if isinstance(b, TextBlock):
                parts.append(types.Part.from_text(text=b.text))
            elif isinstance(b, ToolUseBlock):
                parts.append(types.Part.from_function_call(name=b.name, args=b.input))
            elif isinstance(b, ToolResultBlock):
                name = id_to_name.get(b.tool_use_id, "unknown_tool")
                parts.append(types.Part.from_function_response(name=name, response={"output": b.content}))
        if parts:
            contents.append(types.Content(role=role, parts=parts))
    return contents


def _from_gemini_candidate(candidate) -> tuple[list[ContentBlock], str]:
    from google.genai import types

    blocks: list[ContentBlock] = []
    has_tool_call = False
    parts = candidate.content.parts if candidate.content and candidate.content.parts else []
    for i, part in enumerate(parts):
        if part.text:
            blocks.append(TextBlock(text=part.text))
        if part.function_call:
            has_tool_call = True
            call_id = part.function_call.id or f"call_{i}"
            blocks.append(ToolUseBlock(id=call_id, name=part.function_call.name, input=dict(part.function_call.args or {})))

    if has_tool_call:
        stop_reason = "tool_use"
    elif candidate.finish_reason in (types.FinishReason.STOP, None):
        stop_reason = "end_turn"
    elif candidate.finish_reason == types.FinishReason.MAX_TOKENS:
        stop_reason = "max_tokens"
    else:
        stop_reason = "error"

    return blocks, stop_reason


def _to_gemini_function_declaration(tool: ToolSpec):
    from google.genai import types

    return types.FunctionDeclaration(
        name=tool.name, description=tool.description, parameters_json_schema=tool.input_schema,
    )
