"""Tests the provider-agnostic tool-calling loop with a scripted fake
provider — no real Anthropic API calls. This is what proves the loop
itself (and the "swap the LLM" design) works independent of any vendor.
"""

from datetime import date

import pytest

from usain_bot.chat.providers.base import (
    ChatMessage,
    LLMProvider,
    LLMProviderError,
    ProviderResponse,
    TextBlock,
    ToolUseBlock,
)
from usain_bot.chat.session import run_chat_turn
from usain_bot.config import load_config
from usain_bot.garmin_adapter.mock import MockGarminAdapter
from usain_bot.service import CoachService
from usain_bot.storage.local import LocalBackend


class ScriptedProvider(LLMProvider):
    """Replays a fixed sequence of responses, one per call to run_turn,
    regardless of what's actually in `messages` -- enough to prove the
    session loop drives tool execution and stops correctly."""

    def __init__(self, responses: list[ProviderResponse]):
        self._responses = list(responses)
        self.calls: list[list[ChatMessage]] = []

    def run_turn(self, system, messages, tools):
        self.calls.append(messages)
        if not self._responses:
            raise AssertionError("ScriptedProvider ran out of scripted responses")
        return self._responses.pop(0)

    @property
    def model_name(self) -> str:
        return "scripted-fake"


class AlwaysErrorsProvider(LLMProvider):
    def run_turn(self, system, messages, tools):
        raise LLMProviderError("simulated outage")

    @property
    def model_name(self) -> str:
        return "broken-fake"


@pytest.fixture
def config(tmp_path):
    cfg = load_config("config.yaml")
    cfg.storage.data_dir = str(tmp_path)
    return cfg


@pytest.fixture
def storage(config) -> LocalBackend:
    backend = LocalBackend(config.storage.data_dir, config.storage.db_filename, config.storage.references_dir)
    yield backend
    backend.close()


@pytest.fixture
def service(config, storage, fixture_path) -> CoachService:
    svc = CoachService(config, storage, MockGarminAdapter(fixture_path))
    svc.get_today(date(2026, 7, 24))  # seed a plan so tools have something to read
    return svc


class TestChatSessionLoop:
    def test_text_only_reply_no_tools(self, service):
        provider = ScriptedProvider([
            ProviderResponse(content=[TextBlock(text="You're all set for today.")], stop_reason="end_turn"),
        ])
        result = run_chat_turn(provider, service, "hi")
        assert result.reply == "You're all set for today."
        assert result.tool_calls == []

    def test_single_tool_call_then_final_answer(self, service):
        provider = ScriptedProvider([
            ProviderResponse(
                content=[ToolUseBlock(id="t1", name="get_today_recommendation", input={})],
                stop_reason="tool_use",
            ),
            ProviderResponse(content=[TextBlock(text="Run easy today.")], stop_reason="end_turn"),
        ])
        result = run_chat_turn(provider, service, "what should I run today?")
        assert result.tool_calls == ["get_today_recommendation"]
        assert result.reply == "Run easy today."
        # tool result should have been fed back as a user-role message with real data in it
        second_call_messages = provider.calls[1]
        assert any("recommendation" in str(m.content) for m in second_call_messages)

    def test_multiple_tool_calls_in_one_turn(self, service):
        provider = ScriptedProvider([
            ProviderResponse(
                content=[
                    ToolUseBlock(id="t1", name="get_today_recommendation", input={}),
                    ToolUseBlock(id="t2", name="get_plan_overview", input={}),
                ],
                stop_reason="tool_use",
            ),
            ProviderResponse(content=[TextBlock(text="Here's both.")], stop_reason="end_turn"),
        ])
        result = run_chat_turn(provider, service, "give me everything")
        assert set(result.tool_calls) == {"get_today_recommendation", "get_plan_overview"}

    def test_mutation_tool_actually_changes_the_plan(self, service):
        before_version = service.get_plan().version
        provider = ScriptedProvider([
            ProviderResponse(
                content=[ToolUseBlock(id="t1", name="ease_upcoming_week", input={})],
                stop_reason="tool_use",
            ),
            ProviderResponse(content=[TextBlock(text="Eased next week for you.")], stop_reason="end_turn"),
        ])
        run_chat_turn(provider, service, "make next week easier")
        assert service.get_plan().version == before_version + 1

    def test_provider_error_surfaces_as_reply_not_exception(self, service):
        result = run_chat_turn(AlwaysErrorsProvider(), service, "hi")
        assert "unavailable" in result.reply.lower()

    def test_conversation_persisted(self, service):
        provider = ScriptedProvider([
            ProviderResponse(content=[TextBlock(text="Sure thing.")], stop_reason="end_turn"),
        ])
        run_chat_turn(provider, service, "hello there")
        history = service.storage.get_conversation_history()
        texts = [h.text for h in history]
        assert "hello there" in texts
        assert "Sure thing." in texts

    def test_runaway_tool_loop_terminates(self, service):
        # every response calls a tool again, forever -- loop must give up cleanly.
        responses = [
            ProviderResponse(content=[ToolUseBlock(id=f"t{i}", name="get_today_recommendation", input={})], stop_reason="tool_use")
            for i in range(10)
        ]
        provider = ScriptedProvider(responses)
        result = run_chat_turn(provider, service, "loop forever")
        assert result.reply  # some fallback message, not a crash
