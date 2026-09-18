"""Provider registry for the OpenAI-compatible examples in this repository.

The chapter demos intentionally keep their provider-specific prompts and tool
contracts local.  This module owns only the repeatable plumbing: provider
aliases, environment-variable lookup, model ids for OpenRouter, and the
fallback decision when a direct provider credential is unavailable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class ProviderSpec:
    """A direct OpenAI-compatible provider and its local configuration."""

    name: str
    key_variables: tuple[str, ...]
    default_base_url: str
    default_model: str
    base_url_variable: str | None = None

    @property
    def key_vars(self) -> tuple[str, ...]:
        """Backward-compatible name used by existing chapter runners."""

        return self.key_variables

    @property
    def requires_key(self) -> bool:
        """Whether this provider needs a credential for a direct request."""

        return bool(self.key_variables)

    def api_key(self) -> str:
        """Return the first configured credential without ever logging it."""

        return next((os.getenv(name, "") for name in self.key_variables if os.getenv(name)), "")

    @property
    def base_url(self) -> str:
        if self.base_url_variable:
            return os.getenv(self.base_url_variable, self.default_base_url)
        return self.default_base_url


@dataclass(frozen=True)
class ResolvedBackend:
    """The endpoint, credential and model selected for one model call."""

    provider: str
    api_key: str
    base_url: str
    model: str
    using_openrouter: bool = False


PROVIDERS: Mapping[str, ProviderSpec] = {
    "openai": ProviderSpec("openai", ("OPENAI_API_KEY",), "https://api.openai.com/v1", "gpt-5.6-luna", "OPENAI_BASE_URL"),
    "openrouter": ProviderSpec("openrouter", ("OPENROUTER_API_KEY",), OPENROUTER_BASE_URL, "openai/gpt-5.6-luna", "OPENROUTER_BASE_URL"),
    "moonshot": ProviderSpec("moonshot", ("MOONSHOT_API_KEY", "KIMI_API_KEY"), "https://api.moonshot.cn/v1", "kimi-k3", "MOONSHOT_BASE_URL"),
    "dashscope": ProviderSpec("dashscope", ("DASHSCOPE_API_KEY",), "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus", "DASHSCOPE_BASE_URL"),
    "doubao": ProviderSpec("doubao", ("ARK_API_KEY", "DOUBAO_API_KEY"), "https://ark.cn-beijing.volces.com/api/v3", "doubao-seed-1-6-thinking-250715", "ARK_BASE_URL"),
    "siliconflow": ProviderSpec("siliconflow", ("SILICONFLOW_API_KEY",), "https://api.siliconflow.cn/v1", "Qwen/Qwen3.5-397B-A17B", "SILICONFLOW_BASE_URL"),
    "deepseek": ProviderSpec("deepseek", ("DEEPSEEK_API_KEY",), "https://api.deepseek.com", "deepseek-v4-flash", "DEEPSEEK_BASE_URL"),
    "zhipu": ProviderSpec("zhipu", ("ZHIPU_API_KEY",), "https://open.bigmodel.cn/api/paas/v4", "glm-4.7", "ZHIPU_BASE_URL"),
    "gemini": ProviderSpec("gemini", ("GEMINI_API_KEY", "GOOGLE_API_KEY"), "https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-3.5-flash", "GEMINI_BASE_URL"),
    "groq": ProviderSpec("groq", ("GROQ_API_KEY",), "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", "GROQ_BASE_URL"),
    "together": ProviderSpec("together", ("TOGETHER_API_KEY",), "https://api.together.xyz/v1", "meta-llama/Llama-3.3-70B-Instruct-Turbo", "TOGETHER_BASE_URL"),
    # Ollama is deliberately keyless: it is the local/offline option.
    "ollama": ProviderSpec("ollama", (), "http://localhost:11434/v1", "qwen2.5:7b", "OLLAMA_BASE_URL"),
}

ALIASES = {
    "ark": "doubao",
    "bailian": "dashscope",
    "kimi": "moonshot",
    "qwen": "dashscope",
}
SUPPORTED_PROVIDERS = tuple(PROVIDERS)


def canonical_provider(provider: str | None) -> str:
    """Normalize a documented provider name while preserving unknown names."""

    value = (provider or "").strip().lower()
    return ALIASES.get(value, value)


def is_openrouter_key(api_key: str | None) -> bool:
    """Recognize an OpenRouter key without inspecting or serializing its value."""

    return bool(api_key) and (
        api_key == os.getenv("OPENROUTER_API_KEY") or api_key.startswith("sk-or-")
    )


def map_model_to_openrouter(model: str | None, *, substitute_unknown: bool = False) -> str:
    """Map common provider-native ids to the ids expected by OpenRouter."""

    requested = (model or "").strip()
    if not requested or "/" in requested:
        return requested

    normalized = requested.lower()
    if normalized.startswith(("gpt-", "o1", "o3", "o4")):
        return f"openai/{requested}"
    if normalized.startswith("claude-"):
        return f"anthropic/{requested}"
    if normalized.startswith("kimi-") or normalized.startswith("moonshot-"):
        # Kimi K3 is a direct-provider model id; the maintained OpenRouter
        # equivalent used by the chapter examples is K2.6.
        return "moonshotai/kimi-k2.6"
    if normalized.startswith(("deepseek-", "deepseek/")):
        return f"deepseek/{requested}"
    if normalized.startswith(("gemini-", "google/")):
        return f"google/{requested}"

    return os.getenv("OPENROUTER_MODEL", "openai/gpt-5.6-luna") if substitute_unknown else requested


def _requires_openrouter(model: str, chosen_by_reader: bool) -> bool:
    """Keep the repository's documented default gpt-5 routing behaviour."""

    return not chosen_by_reader and model.lower().startswith(("gpt-5", "o1", "o3", "o4"))


def resolve_backend(
    provider: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    chosen_by_reader: bool = True,
) -> ResolvedBackend:
    """Resolve one provider call, using OpenRouter only when policy permits it."""

    canonical = canonical_provider(provider)
    if canonical not in PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider!r}")

    spec = PROVIDERS[canonical]
    requested_model = model or spec.default_model
    direct_key = api_key if api_key is not None else spec.api_key()
    direct_base_url = base_url or spec.base_url

    if canonical == "ollama":
        return ResolvedBackend(canonical, direct_key or "", direct_base_url, requested_model)

    openrouter_key = PROVIDERS["openrouter"].api_key()
    use_openrouter = canonical != "openrouter" and bool(openrouter_key) and (
        not direct_key or _requires_openrouter(requested_model, chosen_by_reader)
    )
    if canonical == "openrouter":
        use_openrouter = True
        openrouter_key = direct_key or openrouter_key

    if use_openrouter and openrouter_key:
        return ResolvedBackend(
            "openrouter",
            openrouter_key,
            PROVIDERS["openrouter"].base_url,
            map_model_to_openrouter(requested_model, substitute_unknown=canonical == "openrouter"),
            True,
        )
    if direct_key:
        return ResolvedBackend(canonical, direct_key, direct_base_url, requested_model)

    expected = " or ".join(spec.key_variables) or "no API key"
    raise ValueError(
        f"No API key found for provider '{canonical}'. Set {expected} or OPENROUTER_API_KEY."
    )


def resolve_llm_backend(
    primary_key: str | None,
    primary_base_url: str,
    model: str,
) -> tuple[str, str, str, bool]:
    """Compatibility adapter for older Moonshot-first chapter examples."""

    backend = resolve_backend(
        "moonshot",
        model=model,
        api_key=primary_key,
        base_url=primary_base_url,
        chosen_by_reader=False,
    )
    return backend.api_key, backend.base_url, backend.model, backend.using_openrouter

