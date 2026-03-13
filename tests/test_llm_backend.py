"""Tests for LLM_BACKEND routing in dependencies.get_llm_client()."""

import pytest

from app.dependencies import get_llm_client, get_settings
from app.settings import Settings


def test_unsupported_backend_raises_value_error(monkeypatch):
    """An unknown LLM_BACKEND must raise ValueError with a helpful message."""
    monkeypatch.setenv("LLM_BACKEND", "ollama")

    # Clear lru_cache so get_settings() picks up the monkeypatched env var
    get_settings.cache_clear()
    get_llm_client.cache_clear()

    with pytest.raises(ValueError, match="Unsupported LLM_BACKEND 'ollama'"):
        get_llm_client()

    # Restore cache state for other tests
    get_settings.cache_clear()
    get_llm_client.cache_clear()


def test_anthropic_backend_is_default():
    """Default LLM_BACKEND must be 'anthropic'."""
    s = Settings()
    assert s.llm_backend == "anthropic"
