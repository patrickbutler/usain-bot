"""Config-driven provider selection — the only place besides individual
provider modules that knows a concrete provider exists. To add a new
vendor: write chat/providers/<vendor>.py implementing LLMProvider, then
add one branch here. Nothing in chat/session.py, chat/tools.py, or
web/app.py needs to change.
"""

from __future__ import annotations

from ...config import Config
from .base import LLMProvider


def get_llm_provider(config: Config) -> LLMProvider:
    provider = config.chat.provider.lower()

    if provider == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider.from_env(model=config.chat.model, max_tokens=config.chat.max_tokens)

    raise ValueError(f"Unknown chat provider: {provider!r} (expected 'anthropic' for now)")
