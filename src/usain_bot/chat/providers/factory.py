"""Config-driven provider selection — the only place besides individual
provider modules that knows a concrete provider exists. To add a new
vendor: write chat/providers/<vendor>.py implementing LLMProvider, then
add one branch here (and one entry in REQUIRED_ENV_VAR if it needs an
API key). Nothing in chat/session.py, chat/tools.py, or web/app.py
needs to change.
"""

from __future__ import annotations

from ...config import Config
from .base import LLMProvider

# Which environment variable each provider needs, so callers (cli.py's
# `serve` command) can give a provider-agnostic "chat won't work without
# X" warning instead of hardcoding one vendor's variable name.
REQUIRED_ENV_VAR: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def get_llm_provider(config: Config) -> LLMProvider:
    provider = config.chat.provider.lower()

    if provider == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider.from_env(model=config.chat.model, max_tokens=config.chat.max_tokens)

    if provider == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider.from_env(model=config.chat.model, max_tokens=config.chat.max_tokens)

    if provider == "gemini":
        from .gemini_provider import GeminiProvider

        return GeminiProvider.from_env(model=config.chat.model, max_tokens=config.chat.max_tokens)

    raise ValueError(f"Unknown chat provider: {provider!r} (expected 'openai', 'anthropic', or 'gemini')")


def get_required_env_var(provider: str) -> str:
    return REQUIRED_ENV_VAR.get(provider.lower(), "")
