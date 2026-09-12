"""Shared test fixtures and environment isolation.

``src.config`` reads a developer's ``.env`` at import time, so without this the
suite's behaviour would depend on which API keys happen to be configured on the
machine running it -- tests that assert on the default three-provider ensemble
would fail for anyone who has narrowed MEMEDD_ENSEMBLE_PROVIDERS, and pass again
after they unset it. That is a miserable failure mode to debug.

The autouse fixture below pins every ambient-configurable value back to its
documented default for the duration of each test. Tests that care about a
specific value still set it themselves with monkeypatch, which wins.
"""

from __future__ import annotations

import pytest

from src import balances, config, data_fetchers


@pytest.fixture(autouse=True)
def isolate_ambient_config(monkeypatch):
    """Neutralize any local .env so tests run identically everywhere."""
    # No provider keys by default: tests that want a ready provider either
    # inject a fake client or set the key explicitly.
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY",
                 "GOPLUS_APP_KEY", "GOPLUS_APP_SECRET"):
        monkeypatch.setattr(config, name, "")

    monkeypatch.setattr(config, "LLM_PROVIDER", "none")
    monkeypatch.setattr(config, "LLM_MODEL", "")
    monkeypatch.setattr(config, "ENSEMBLE_PROVIDERS", ("xai", "anthropic", "openai"))
    monkeypatch.setattr(config, "ENSEMBLE_BLEND_WEIGHT", 0.35)
    monkeypatch.setattr(config, "LLM_TEMPERATURE", 0.2)

    # Per-provider model ids, so assertions on the wire format are stable even
    # if the developer has overridden them.
    monkeypatch.setattr(config, "ANTHROPIC_MODEL", "claude-opus-5")
    monkeypatch.setattr(config, "OPENAI_MODEL", "gpt-4.1")
    monkeypatch.setattr(config, "XAI_MODEL", "grok-4")
    monkeypatch.setattr(config, "XAI_BASE_URL", "https://api.x.ai/v1")
    monkeypatch.setattr(config, "OPENAI_BASE_URL", "")

    # API base URLs, so a proxy override cannot point tests at a real host.
    monkeypatch.setattr(config, "DEXSCREENER_BASE", "https://api.dexscreener.com")
    monkeypatch.setattr(config, "GOPLUS_BASE", "https://api.gopluslabs.io")


@pytest.fixture(autouse=True)
def clear_fetch_caches():
    """Stop cached responses leaking between tests."""
    data_fetchers.clear_caches()
    balances.clear_ledger_cache()
    yield
    data_fetchers.clear_caches()
    balances.clear_ledger_cache()
