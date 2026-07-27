"""Tests for GeminiProvider. No real network calls: constructs real
google-genai `types` objects (the package is a hard dependency of the
`chat` extra, so this isn't optional/skippable coverage) to test the
translation logic and mocks the SDK call site for a full run_turn()
pass, mirroring tests/test_chat_providers.py's OpenAI coverage.
"""

from google.genai import types

import pytest

from usain_bot.chat.providers.base import (
    ChatMessage,
    LLMProviderError,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)
from usain_bot.chat.providers.factory import get_llm_provider
from usain_bot.chat.providers.gemini_provider import (
    DEFAULT_MODEL,
    GeminiProvider,
    _from_gemini_candidate,
    _to_gemini_contents,
    _to_gemini_function_declaration,
)
from usain_bot.config import load_config


class TestGeminiRequestTranslation:
    def test_text_message_maps_user_role(self):
        contents = _to_gemini_contents([ChatMessage(role="user", content=[TextBlock(text="hi there")])])
        assert len(contents) == 1
        assert contents[0].role == "user"
        assert contents[0].parts[0].text == "hi there"

    def test_assistant_role_maps_to_model(self):
        contents = _to_gemini_contents([ChatMessage(role="assistant", content=[TextBlock(text="ok")])])
        assert contents[0].role == "model"

    def test_tool_use_becomes_function_call_part_with_dict_args(self):
        contents = _to_gemini_contents([
            ChatMessage(role="assistant", content=[ToolUseBlock(id="call_1", name="get_today_recommendation", input={"days": 7})]),
        ])
        part = contents[0].parts[0]
        assert part.function_call.name == "get_today_recommendation"
        assert part.function_call.args == {"days": 7}

    def test_tool_result_recovers_function_name_from_earlier_tool_use(self):
        messages = [
            ChatMessage(role="assistant", content=[ToolUseBlock(id="call_1", name="get_plan_overview", input={})]),
            ChatMessage(role="user", content=[ToolResultBlock(tool_use_id="call_1", content='{"plan_version": 3}')]),
        ]
        contents = _to_gemini_contents(messages)
        result_part = contents[1].parts[0]
        assert result_part.function_response.name == "get_plan_overview"
        assert result_part.function_response.response == {"output": '{"plan_version": 3}'}

    def test_unmatched_tool_result_id_falls_back_gracefully(self):
        messages = [ChatMessage(role="user", content=[ToolResultBlock(tool_use_id="orphan", content="x")])]
        contents = _to_gemini_contents(messages)
        assert contents[0].parts[0].function_response.name == "unknown_tool"

    def test_tool_spec_translation_passes_json_schema_through(self):
        spec = ToolSpec(name="get_run_history", description="desc", input_schema={"type": "object", "properties": {"days": {"type": "integer"}}})
        decl = _to_gemini_function_declaration(spec)
        assert decl.name == "get_run_history"
        assert decl.description == "desc"
        assert decl.parameters_json_schema == spec.input_schema


class TestGeminiResponseParsing:
    def test_text_only_response_is_end_turn(self):
        candidate = types.Candidate(
            content=types.Content(role="model", parts=[types.Part.from_text(text="Run easy today.")]),
            finish_reason=types.FinishReason.STOP,
        )
        blocks, stop_reason = _from_gemini_candidate(candidate)
        assert blocks == [TextBlock(text="Run easy today.")]
        assert stop_reason == "end_turn"

    def test_function_call_response_is_tool_use_regardless_of_finish_reason(self):
        # Gemini reports STOP even when it wants to call a tool -- stop_reason
        # must come from content, not finish_reason alone.
        candidate = types.Candidate(
            content=types.Content(role="model", parts=[types.Part.from_function_call(name="get_today_recommendation", args={})]),
            finish_reason=types.FinishReason.STOP,
        )
        blocks, stop_reason = _from_gemini_candidate(candidate)
        assert stop_reason == "tool_use"
        assert len(blocks) == 1
        assert blocks[0].name == "get_today_recommendation"
        assert blocks[0].input == {}

    def test_missing_call_id_gets_a_synthetic_one(self):
        candidate = types.Candidate(
            content=types.Content(role="model", parts=[types.Part.from_function_call(name="x", args={})]),
            finish_reason=types.FinishReason.STOP,
        )
        blocks, _ = _from_gemini_candidate(candidate)
        assert blocks[0].id  # non-empty, deterministic per response

    def test_max_tokens_finish_reason(self):
        candidate = types.Candidate(
            content=types.Content(role="model", parts=[types.Part.from_text(text="cut off")]),
            finish_reason=types.FinishReason.MAX_TOKENS,
        )
        _, stop_reason = _from_gemini_candidate(candidate)
        assert stop_reason == "max_tokens"

    def test_safety_block_maps_to_error(self):
        candidate = types.Candidate(
            content=types.Content(role="model", parts=[]),
            finish_reason=types.FinishReason.SAFETY,
        )
        _, stop_reason = _from_gemini_candidate(candidate)
        assert stop_reason == "error"


class TestGeminiProviderFromEnv:
    def test_raises_clean_error_without_api_key(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(LLMProviderError, match="GEMINI_API_KEY"):
            GeminiProvider.from_env()

    def test_constructs_with_api_key(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")
        provider = GeminiProvider.from_env()
        assert provider.model_name == DEFAULT_MODEL


class TestGeminiProviderRunTurn:
    def test_run_turn_end_to_end_against_mocked_client(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")
        provider = GeminiProvider.from_env()

        fake_candidate = types.Candidate(
            content=types.Content(role="model", parts=[types.Part.from_text(text="Run 4 miles easy.")]),
            finish_reason=types.FinishReason.STOP,
        )
        fake_response = types.GenerateContentResponse(candidates=[fake_candidate])
        captured = {}

        def fake_generate_content(**kwargs):
            captured.update(kwargs)
            return fake_response

        provider._client.models.generate_content = fake_generate_content

        tools = [ToolSpec(name="get_today_recommendation", description="d", input_schema={"type": "object", "properties": {}})]
        result = provider.run_turn("system prompt", [ChatMessage(role="user", content=[TextBlock(text="what today?")])], tools)

        assert result.stop_reason == "end_turn"
        assert result.content == [TextBlock(text="Run 4 miles easy.")]
        assert captured["config"].system_instruction == "system prompt"
        assert captured["config"].tools[0].function_declarations[0].name == "get_today_recommendation"

    def test_run_turn_wraps_sdk_exceptions(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")
        provider = GeminiProvider.from_env()

        def boom(**kwargs):
            raise RuntimeError("network exploded")

        provider._client.models.generate_content = boom
        with pytest.raises(LLMProviderError, match="Gemini API call failed"):
            provider.run_turn("sys", [ChatMessage(role="user", content=[TextBlock(text="hi")])], [])

    def test_run_turn_raises_on_empty_candidates(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")
        provider = GeminiProvider.from_env()
        provider._client.models.generate_content = lambda **kwargs: types.GenerateContentResponse(candidates=[])
        with pytest.raises(LLMProviderError, match="no candidates"):
            provider.run_turn("sys", [ChatMessage(role="user", content=[TextBlock(text="hi")])], [])


class TestFactoryResolvesGemini:
    def test_factory_resolves_gemini_provider(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")
        cfg = load_config("config.yaml")
        cfg.chat.provider = "gemini"
        cfg.chat.model = "gemini-2.0-flash"
        provider = get_llm_provider(cfg)
        assert isinstance(provider, GeminiProvider)
