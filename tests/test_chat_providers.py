"""Tests for the concrete LLMProvider implementations. No real network
calls: OpenAI/Anthropic client construction doesn't touch the network,
so these test the translation logic and the from_env() error paths
directly, and monkeypatch the SDK call site for a full run_turn() pass.
"""

import json
from types import SimpleNamespace

import pytest

from usain_bot.chat.providers.base import (
    ChatMessage,
    LLMProviderError,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)
from usain_bot.chat.providers.factory import get_llm_provider, get_required_env_var
from usain_bot.chat.providers.openai_provider import (
    DEFAULT_MODEL,
    OpenAIProvider,
    _from_openai_message,
    _map_finish_reason,
    _to_openai_messages,
    _to_openai_tool,
)
from usain_bot.config import load_config


class TestOpenAIMessageTranslation:
    def test_text_message_round_trips(self):
        wire = _to_openai_messages([ChatMessage(role="user", content=[TextBlock(text="hi there")])])
        assert wire == [{"role": "user", "content": "hi there"}]

    def test_tool_result_becomes_role_tool_message(self):
        wire = _to_openai_messages([
            ChatMessage(role="user", content=[ToolResultBlock(tool_use_id="call_1", content='{"ok": true}')]),
        ])
        assert wire == [{"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}'}]

    def test_multiple_tool_results_become_multiple_tool_messages(self):
        wire = _to_openai_messages([
            ChatMessage(role="user", content=[
                ToolResultBlock(tool_use_id="call_1", content="a"),
                ToolResultBlock(tool_use_id="call_2", content="b"),
            ]),
        ])
        assert wire == [
            {"role": "tool", "tool_call_id": "call_1", "content": "a"},
            {"role": "tool", "tool_call_id": "call_2", "content": "b"},
        ]

    def test_assistant_tool_use_becomes_tool_calls_with_json_string_args(self):
        wire = _to_openai_messages([
            ChatMessage(role="assistant", content=[ToolUseBlock(id="call_1", name="get_today_recommendation", input={"days": 7})]),
        ])
        assert wire[0]["role"] == "assistant"
        assert wire[0]["content"] is None
        assert wire[0]["tool_calls"][0]["id"] == "call_1"
        assert wire[0]["tool_calls"][0]["function"]["name"] == "get_today_recommendation"
        assert json.loads(wire[0]["tool_calls"][0]["function"]["arguments"]) == {"days": 7}

    def test_tool_spec_translation(self):
        spec = ToolSpec(name="get_run_history", description="desc", input_schema={"type": "object", "properties": {}})
        wire = _to_openai_tool(spec)
        assert wire == {"type": "function", "function": {"name": "get_run_history", "description": "desc", "parameters": spec.input_schema}}


class TestOpenAIResponseParsing:
    def test_text_only_response(self):
        message = SimpleNamespace(content="Run easy today.", tool_calls=None)
        blocks = _from_openai_message(message)
        assert blocks == [TextBlock(text="Run easy today.")]

    def test_tool_call_response_parses_json_arguments(self):
        message = SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(id="call_1", function=SimpleNamespace(name="get_plan_overview", arguments="{}"))],
        )
        blocks = _from_openai_message(message)
        assert blocks == [ToolUseBlock(id="call_1", name="get_plan_overview", input={})]

    def test_malformed_json_arguments_does_not_crash(self):
        message = SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(id="call_1", function=SimpleNamespace(name="x", arguments="not json"))],
        )
        blocks = _from_openai_message(message)
        assert blocks[0].input == {}

    def test_text_and_tool_calls_together(self):
        message = SimpleNamespace(
            content="checking...",
            tool_calls=[SimpleNamespace(id="call_1", function=SimpleNamespace(name="get_today_recommendation", arguments="{}"))],
        )
        blocks = _from_openai_message(message)
        assert len(blocks) == 2

    @pytest.mark.parametrize("finish_reason,expected", [
        ("tool_calls", "tool_use"), ("stop", "end_turn"), ("length", "max_tokens"), ("content_filter", "error"),
    ])
    def test_finish_reason_mapping(self, finish_reason, expected):
        assert _map_finish_reason(finish_reason) == expected


class TestOpenAIProviderFromEnv:
    def test_raises_clean_error_without_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(LLMProviderError, match="OPENAI_API_KEY"):
            OpenAIProvider.from_env()

    def test_constructs_with_api_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
        provider = OpenAIProvider.from_env()
        assert provider.model_name == DEFAULT_MODEL


class TestOpenAIProviderRunTurn:
    def test_run_turn_end_to_end_against_mocked_client(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
        provider = OpenAIProvider.from_env()

        fake_response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="Run 4 miles easy.", tool_calls=None),
            finish_reason="stop",
        )])
        captured = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return fake_response

        provider._client.chat.completions.create = fake_create

        tools = [ToolSpec(name="get_today_recommendation", description="d", input_schema={"type": "object", "properties": {}})]
        result = provider.run_turn("system prompt", [ChatMessage(role="user", content=[TextBlock(text="what today?")])], tools)

        assert result.stop_reason == "end_turn"
        assert result.content == [TextBlock(text="Run 4 miles easy.")]
        assert captured["messages"][0] == {"role": "system", "content": "system prompt"}
        assert captured["tools"][0]["function"]["name"] == "get_today_recommendation"

    def test_run_turn_wraps_sdk_exceptions(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
        provider = OpenAIProvider.from_env()

        def boom(**kwargs):
            raise RuntimeError("network exploded")

        provider._client.chat.completions.create = boom
        with pytest.raises(LLMProviderError, match="OpenAI API call failed"):
            provider.run_turn("sys", [ChatMessage(role="user", content=[TextBlock(text="hi")])], [])


class TestFactory:
    def test_required_env_var_mapping(self):
        assert get_required_env_var("openai") == "OPENAI_API_KEY"
        assert get_required_env_var("anthropic") == "ANTHROPIC_API_KEY"
        assert get_required_env_var("bogus") == ""

    def test_defaults_to_openai(self):
        cfg = load_config("config.yaml")
        assert cfg.chat.provider == "openai"
        assert cfg.chat.model == "gpt-4o"

    def test_factory_resolves_openai_provider(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
        cfg = load_config("config.yaml")
        provider = get_llm_provider(cfg)
        assert isinstance(provider, OpenAIProvider)

    def test_factory_resolves_anthropic_provider(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-fake")
        cfg = load_config("config.yaml")
        cfg.chat.provider = "anthropic"
        cfg.chat.model = "claude-sonnet-5"
        provider = get_llm_provider(cfg)
        from usain_bot.chat.providers.anthropic_provider import AnthropicProvider
        assert isinstance(provider, AnthropicProvider)

    def test_factory_rejects_unknown_provider(self):
        cfg = load_config("config.yaml")
        cfg.chat.provider = "bogus-vendor"
        with pytest.raises(ValueError, match="bogus-vendor"):
            get_llm_provider(cfg)
