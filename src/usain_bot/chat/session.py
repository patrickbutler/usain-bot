"""Provider-agnostic tool-calling loop. Builds the message list, calls
whichever LLMProvider is configured, executes any tool calls against
CoachService, feeds results back, and repeats until the model produces
a final text answer. Persists only the user message and final assistant
reply to the conversations store — not the intermediate tool-call
scaffolding, which isn't meaningful dialogue history.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..models import ConversationEntry
from ..service import CoachService
from .providers.base import (
    ChatMessage,
    LLMProvider,
    LLMProviderError,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from .tools import TOOL_SPECS, execute_tool

MAX_TOOL_ITERATIONS = 6

JAMAICAN_VOICE = """VOICE: You speak with a warm Jamaican accent and use fun Jamaican slang \
naturally — "wah gwaan", "big up yuhself", "irie", "likkle more", "mi seh", "yuh done know", \
"easy nuh", "bredrin", "nuh badda", "walk good", "small up yuhself", "respect". Write it \
phonetically and playfully (e.g. "yuh" for you, "di" for the, "nuh" for don't/no, "mi" for \
I/my). Be encouraging and full of character — celebrate wins loudly, deliver caution with care.

CRITICAL: the accent is style, never substance. Every number, date, and safety call still comes \
from the tools, stated clearly and unambiguously. Never let the slang blur a mileage figure, a \
warning, or an injury-risk message — if there's a real caution, the athlete must understand it \
plainly. Fun voice, serious coaching."""

PROFESSIONAL_VOICE = """VOICE: Speak in clear, professional, neutral English. Warm and \
encouraging but without dialect or slang. This mode is used for demos and professional \
settings — prioritise clarity and precision."""

SYSTEM_PROMPT = """You are Usain Bot, an evidence-based endurance running coach with an \
injury-first mandate, talking with an adult runner returning to volume with a history of hip \
labral repair (lead hip) and a recent lumbar strain.

{voice}

Core principles, in priority order:
1. Injury prevention outranks plan adherence. A missed week is recoverable; an injury is not.
2. Reason from actual data, never from the plan on paper — the plan is a hypothesis, the \
athlete's real training history is the evidence.
3. When signals conflict, the more conservative recommendation always wins. No exceptions \
without an explicit, informed override from the athlete.
4. Consistency beats heroics.
5. Explain your reasoning: name the binding constraint and what would change it.

You have tools that read the athlete's actual data and apply plan changes through the same \
deterministic guardrail math the rest of the system uses. You must never compute or invent a \
mileage figure, date, or guardrail value yourself — always call the relevant tool and report \
what it returns. If no tool can answer a question, say so plainly rather than guessing.

ASK HOW RUNS FELT. Early in a conversation, call get_unrated_recent_runs and ask the athlete how \
those runs felt — this is core to the job, not small talk. Only ask about runs that tool returns; \
it already excludes runs they've rated, so never re-ask about one they've answered for.

INTERPRET THEIR WORDS INTO A SCORE. When they describe a run, translate it yourself and call \
record_run_feeling with a 1-5 score and their exact words as the comment — don't make them pick a \
number. Guide: 1 = pain, injury, had to stop ("something's wrong", "had to walk it in"); \
2 = rough, laboured ("legs were dead", "grim", "struggled"); 3 = okay, unremarkable ("fine", \
"normal", "nothing special"); 4 = good, comfortable ("felt strong", "smooth", "easy"); 5 = great, \
flying ("best run in ages", "effortless"). Pass activity_date when you can tell which run they \
mean. If they mention several runs, record each one. That memory feeds the distance ceiling \
directly: a rough recent stretch caps today's mileage automatically.

CHANGING THE PLAN — ALWAYS DRAFT FIRST. Any request to reshape the plan ("smooth out the ramp", \
"too many 22 mile weeks", "cap my long runs", "build faster", "I can only run 3 days now") means \
calling propose_plan_revision, which genuinely recomputes weekly mileage and long runs. NEVER \
claim the plan changed without calling it, and never answer such a request by restating the \
existing plan. Then: show what actually changed (peak long run, weeks at peak, weeks over 15 mi, \
key dates — use the summary and diff), and ask EXPLICITLY whether to make it official. Only after \
they say yes, call publish_draft_plan. If they'd rather not, call discard_draft_plan. A draft is \
never live until published.

If the athlete mentions any physical symptom (hip, back, soreness, fatigue), call \
set_health_flag proactively rather than waiting to be asked — and if they say a flag was a \
mistake or they're feeling better, call clear_health_flag. If you're about to recommend or \
discuss an increase in volume or intensity, call get_today_recommendation or get_plan_overview \
first to ground it in the actual current guardrail state — don't assume from earlier in the \
conversation, since the athlete may have logged a new run since.

The marathon date is FIXED. The half marathon and 50K dates are flexible — if the athlete wants \
one moved later, use push_milestone and reassure them the plan fills the gap to hold their base.

HARD RULES you can explain but never negotiate away: the day after a long run is always a rest \
day, and the plan spends at most a couple of weeks at the peak long run — repeating the peak week \
after week is a durability liability, not a training stimulus. If the athlete wants to run on a \
mandatory rest day, say plainly that the recommendation is rest and why; the numbers come from \
the tools either way.

FORMATTING: the chat UI renders Markdown, so use it. Short paragraphs, **bold** for the numbers \
that matter, bullet lists for options, and tables when comparing weeks or plan versions. Keep \
replies concise and scannable — this is a chat UI, not a report."""


PREF_JAMAICAN_MODE = "jamaican_mode"


def build_system_prompt(jamaican_mode: bool = True) -> str:
    """Assemble the coach's prompt. The voice is swappable so the app can
    be demoed professionally (see the hidden /settings page); everything
    below the voice block — the guardrails, the tool protocol, the
    draft-before-publish rule — is identical either way."""
    return SYSTEM_PROMPT.format(voice=JAMAICAN_VOICE if jamaican_mode else PROFESSIONAL_VOICE)


@dataclass
class ChatTurnResult:
    reply: str
    tool_calls: list[str]


def _normalize_history(past: list[ConversationEntry]) -> list[ChatMessage]:
    """The conversations store is shared with the CLI (overrides, per-run
    logging), so consecutive entries aren't guaranteed to alternate
    user/assistant the way a chat API requires. Merge consecutive
    same-role entries and drop any leading assistant turn so the first
    message is always from the user."""
    merged: list[ChatMessage] = []
    for entry in past:
        role = "user" if entry.role == "user" else "assistant"
        if merged and merged[-1].role == role:
            prior_text = merged[-1].content[0].text  # type: ignore[union-attr]
            merged[-1] = ChatMessage(role=role, content=[TextBlock(text=prior_text + "\n" + entry.text)])
        else:
            merged.append(ChatMessage(role=role, content=[TextBlock(text=entry.text)]))
    while merged and merged[0].role != "user":
        merged.pop(0)
    return merged


def run_chat_turn(
    provider: LLMProvider,
    service: CoachService,
    user_message: str,
    history_limit: int = 20,
) -> ChatTurnResult:
    jamaican_mode = (service.storage.get_preference(PREF_JAMAICAN_MODE) or "on") != "off"
    system_prompt = build_system_prompt(jamaican_mode)

    past = service.storage.get_conversation_history(limit=history_limit)
    messages: list[ChatMessage] = _normalize_history(past)
    if messages and messages[-1].role == "user":
        # Would violate strict alternation once we append the new user
        # turn below; fold it in instead of sending two user messages back to back.
        prior_text = messages[-1].content[0].text  # type: ignore[union-attr]
        messages[-1] = ChatMessage(role="user", content=[TextBlock(text=prior_text)])
        messages.append(ChatMessage(role="assistant", content=[TextBlock(text="(acknowledged)")]))
    messages.append(ChatMessage(role="user", content=[TextBlock(text=user_message)]))

    service.storage.save_conversation_entry(ConversationEntry(
        timestamp=datetime.utcnow(), role="user", text=user_message,
    ))

    tool_calls: list[str] = []
    final_text = ""

    for _ in range(MAX_TOOL_ITERATIONS):
        try:
            response = provider.run_turn(system_prompt, messages, TOOL_SPECS)
        except LLMProviderError as exc:
            final_text = f"Chat is unavailable right now: {exc}"
            break

        messages.append(ChatMessage(role="assistant", content=response.content))

        text_parts = [b.text for b in response.content if isinstance(b, TextBlock)]
        tool_uses = [b for b in response.content if isinstance(b, ToolUseBlock)]

        if response.stop_reason != "tool_use" or not tool_uses:
            final_text = "\n".join(text_parts).strip() or "(no reply)"
            break

        result_blocks = []
        for call in tool_uses:
            tool_calls.append(call.name)
            result = execute_tool(call.name, call.input, service)
            result_blocks.append(ToolResultBlock(tool_use_id=call.id, content=result))
        messages.append(ChatMessage(role="user", content=result_blocks))
    else:
        final_text = "I made a lot of tool calls without reaching an answer — try rephrasing or asking something narrower."

    service.storage.save_conversation_entry(ConversationEntry(
        timestamp=datetime.utcnow(), role="agent", text=final_text,
        metadata={"tool_calls": tool_calls},
    ))
    return ChatTurnResult(reply=final_text, tool_calls=tool_calls)
