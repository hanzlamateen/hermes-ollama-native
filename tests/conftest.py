"""Shared fixtures for the hermes-ollama-native test suite."""

from __future__ import annotations

import pytest

from hermes_ollama_native import adapter


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    """The Ollama detection probe memoizes positives for the process lifetime;
    clear it around every test so cases don't leak cached results into each other."""
    adapter._OLLAMA_PROBE_CACHE.clear()
    yield
    adapter._OLLAMA_PROBE_CACHE.clear()


@pytest.fixture
def native_flag(monkeypatch):
    """Enable the HERMES_OLLAMA_NATIVE feature flag for the duration of a test."""
    monkeypatch.setenv("HERMES_OLLAMA_NATIVE", "1")
    return None
