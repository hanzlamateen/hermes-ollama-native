"""Wiring tests for the runtime monkeypatch (shim.py). These prove the hooks
actually route native-Ollama endpoints to OllamaNativeClient and leave everything
else untouched — without needing a real Hermes install (we patch the probe)."""

from __future__ import annotations

import types

from hermes_ollama_native import adapter, shim


def test_hook1_routes_native_to_adapter_and_passes_through_otherwise(monkeypatch):
    monkeypatch.setenv("HERMES_OLLAMA_NATIVE", "1")
    monkeypatch.setattr(adapter, "_probe_is_ollama", lambda root, timeout=2.0: True)

    seen = {}

    def orig_create(agent, client_kwargs, *, reason, shared):
        seen["orig"] = (reason, shared)
        return ("ORIG_CLIENT", client_kwargs)

    fake_module = types.SimpleNamespace(create_openai_client=orig_create)
    shim._patch_runtime(fake_module)
    assert getattr(fake_module.create_openai_client, "_hon_wrapped", False) is True

    # Native Ollama base_url (even with /v1) → OllamaNativeClient, original not called.
    client = fake_module.create_openai_client(
        object(),
        {"base_url": "http://localhost:11434/v1", "api_key": "ollama"},
        reason="primary",
        shared=False,
    )
    assert isinstance(client, adapter.OllamaNativeClient)
    assert "orig" not in seen

    # A non-Ollama endpoint (probe says no) → falls through to the original verbatim.
    monkeypatch.setattr(adapter, "_probe_is_ollama", lambda root, timeout=2.0: False)
    result = fake_module.create_openai_client(
        object(),
        {"base_url": "https://api.openai.com/v1"},
        reason="primary",
        shared=False,
    )
    assert result == ("ORIG_CLIENT", {"base_url": "https://api.openai.com/v1"})
    assert seen["orig"] == ("primary", False)


def test_hook1_routes_regardless_of_provider_name(monkeypatch):
    """The shim routes on the base_url probe alone and never inspects agent.provider,
    so it is immune to the upstream 'custom:<instance>' selection bug: a named
    instance like 'custom:ollama-2' still gets the native client."""
    monkeypatch.setenv("HERMES_OLLAMA_NATIVE", "1")
    monkeypatch.setattr(adapter, "_probe_is_ollama", lambda root, timeout=2.0: True)

    def orig_create(agent, client_kwargs, *, reason, shared):
        return "ORIG"

    fake_module = types.SimpleNamespace(create_openai_client=orig_create)
    shim._patch_runtime(fake_module)

    for provider in ("custom", "custom:ollama-2", "custom:Ollama"):
        agent = types.SimpleNamespace(provider=provider)
        client = fake_module.create_openai_client(
            agent,
            {"base_url": "http://localhost:11434/v1", "api_key": "ollama"},
            reason="primary",
            shared=False,
        )
        assert isinstance(client, adapter.OllamaNativeClient), provider


def test_hook1_is_idempotent(monkeypatch):
    monkeypatch.setenv("HERMES_OLLAMA_NATIVE", "1")

    def orig_create(agent, client_kwargs, *, reason, shared):
        return "ORIG"

    fake_module = types.SimpleNamespace(create_openai_client=orig_create)
    shim._patch_runtime(fake_module)
    first = fake_module.create_openai_client
    shim._patch_runtime(fake_module)  # second call must not double-wrap
    assert fake_module.create_openai_client is first


def test_hook2_native_swap(monkeypatch):
    monkeypatch.setenv("HERMES_OLLAMA_NATIVE", "1")
    monkeypatch.setattr(adapter, "_probe_is_ollama", lambda root, timeout=2.0: True)

    native = shim._native_swap(
        types.SimpleNamespace(
            base_url="http://localhost:11434/v1", api_key="ollama", timeout=None, default_headers={}
        )
    )
    assert isinstance(native, adapter.OllamaNativeClient)
    assert native.base_url == "http://localhost:11434"

    # Non-Ollama client (probe negative) is left alone.
    monkeypatch.setattr(adapter, "_probe_is_ollama", lambda root, timeout=2.0: False)
    assert shim._native_swap(types.SimpleNamespace(base_url="https://api.openai.com/v1")) is None
    assert shim._native_swap(None) is None
