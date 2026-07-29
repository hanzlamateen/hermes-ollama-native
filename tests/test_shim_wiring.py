"""Wiring tests for the runtime monkeypatch (shim.py). These prove the hooks
actually route native-Ollama endpoints to OllamaNativeClient and leave everything
else untouched — without needing a real Hermes install (we patch the probe)."""

from __future__ import annotations

import types

from hermes_ollama_native import adapter, shim


def test_hook1_routes_native_to_adapter_and_passes_through_otherwise(monkeypatch):
    monkeypatch.setenv("HERMES_OLLAMA_NATIVE", "1")
    monkeypatch.setattr(adapter, "_probe_is_ollama", lambda root, **_kw: True)

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
    monkeypatch.setattr(adapter, "_probe_is_ollama", lambda root, **_kw: False)
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
    monkeypatch.setattr(adapter, "_probe_is_ollama", lambda root, **_kw: True)

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
    monkeypatch.setattr(adapter, "_probe_is_ollama", lambda root, **_kw: True)

    native = shim._native_swap(
        types.SimpleNamespace(
            base_url="http://localhost:11434/v1", api_key="ollama", timeout=None, default_headers={}
        )
    )
    assert isinstance(native, adapter.OllamaNativeClient)
    assert native.base_url == "http://localhost:11434"

    # Non-Ollama client (probe negative) is left alone.
    monkeypatch.setattr(adapter, "_probe_is_ollama", lambda root, **_kw: False)
    assert shim._native_swap(types.SimpleNamespace(base_url="https://api.openai.com/v1")) is None
    assert shim._native_swap(None) is None


def test_hook1_injects_keepalive_http_client(monkeypatch):
    import httpx

    monkeypatch.setenv("HERMES_OLLAMA_NATIVE", "1")
    monkeypatch.setattr(adapter, "_probe_is_ollama", lambda root, **_kw: True)

    keepalive = httpx.Client()
    fake_agent = types.SimpleNamespace(
        provider="custom", _build_keepalive_http_client=lambda base_url: keepalive
    )

    def orig_create(agent, client_kwargs, *, reason, shared):
        return "ORIG"

    fake_module = types.SimpleNamespace(create_openai_client=orig_create)
    shim._patch_runtime(fake_module)
    client = fake_module.create_openai_client(
        fake_agent,
        {"base_url": "http://localhost:11434/v1", "api_key": "ollama"},
        reason="primary",
        shared=False,
    )
    assert isinstance(client, adapter.OllamaNativeClient)
    assert client._http is keepalive  # agent's keepalive client was injected
    keepalive.close()


def test_hook3_resolves_custom_instance_to_custom_profile(monkeypatch):
    monkeypatch.setenv("HERMES_OLLAMA_NATIVE", "1")
    registry = {"custom": "CUSTOM_PROFILE"}

    def get_provider_profile(name):
        return registry.get(name)

    fake = types.SimpleNamespace(get_provider_profile=get_provider_profile)
    shim._patch_profiles(fake)

    assert fake.get_provider_profile("custom") == "CUSTOM_PROFILE"  # bare unchanged
    assert fake.get_provider_profile("custom:ollama-2") == "CUSTOM_PROFILE"  # falls back
    assert fake.get_provider_profile("gemini") is None  # non-custom unaffected


def test_hook1_resolves_verify_without_provider_ssl_keys(monkeypatch):
    """verify is resolved unconditionally (the resolver reads env CA bundles that
    never appear in client_kwargs) and forwarded to the probe."""
    import sys

    monkeypatch.setenv("HERMES_OLLAMA_NATIVE", "1")
    captured = {}
    monkeypatch.setattr(adapter, "_probe_is_ollama", lambda root, **kw: captured.update(kw) or True)
    fake_ssl_verify = types.SimpleNamespace(
        resolve_httpx_verify=lambda *, ca_bundle=None, ssl_verify=None, base_url="": "ENV_CA_CTX"
    )
    monkeypatch.setitem(sys.modules, "agent", types.SimpleNamespace(ssl_verify=fake_ssl_verify))
    monkeypatch.setitem(sys.modules, "agent.ssl_verify", fake_ssl_verify)

    fake_module = types.SimpleNamespace(
        create_openai_client=lambda agent, client_kwargs, *, reason, shared: "ORIG"
    )
    shim._patch_runtime(fake_module)
    client = fake_module.create_openai_client(
        object(),
        {"base_url": "http://localhost:11434/v1", "api_key": "ollama"},  # no ssl keys
        reason="primary",
        shared=False,
    )
    assert isinstance(client, adapter.OllamaNativeClient)
    assert captured.get("verify") == "ENV_CA_CTX"


def test_hook1_keepalive_verify_by_signature_inspection(monkeypatch):
    """A verify-accepting builder gets verify; a legacy builder is called plain;
    a TypeError raised INSIDE a verify-accepting builder is not silently retried
    without verify (which would hand out a client missing the custom CA)."""
    import sys

    import httpx

    monkeypatch.setenv("HERMES_OLLAMA_NATIVE", "1")
    monkeypatch.setattr(adapter, "_probe_is_ollama", lambda root, **_kw: True)
    fake_ssl_verify = types.SimpleNamespace(
        resolve_httpx_verify=lambda *, ca_bundle=None, ssl_verify=None, base_url="": "CA_CTX"
    )
    monkeypatch.setitem(sys.modules, "agent", types.SimpleNamespace(ssl_verify=fake_ssl_verify))
    monkeypatch.setitem(sys.modules, "agent.ssl_verify", fake_ssl_verify)

    fake_module = types.SimpleNamespace(
        create_openai_client=lambda agent, client_kwargs, *, reason, shared: "ORIG"
    )
    shim._patch_runtime(fake_module)
    kwargs = {"base_url": "http://localhost:11434/v1", "api_key": "ollama"}

    # 1. builder accepts verify → receives the resolved value.
    keepalive = httpx.Client()
    calls = []

    def modern_builder(base_url, *, verify=True):
        calls.append(verify)
        return keepalive

    client = fake_module.create_openai_client(
        types.SimpleNamespace(_build_keepalive_http_client=modern_builder),
        dict(kwargs),
        reason="primary",
        shared=False,
    )
    assert client._http is keepalive and calls == ["CA_CTX"]

    # 2. legacy builder (no verify param) → called plain, no TypeError.
    client = fake_module.create_openai_client(
        types.SimpleNamespace(_build_keepalive_http_client=lambda base_url: keepalive),
        dict(kwargs),
        reason="primary",
        shared=False,
    )
    assert client._http is keepalive

    # 3. TypeError inside a verify-accepting builder → no blind retry without
    # verify; the keepalive is skipped and the client builds its own transport.
    calls.clear()

    def broken_builder(base_url, *, verify=True):
        calls.append(verify)
        raise TypeError("boom inside builder")

    client = fake_module.create_openai_client(
        types.SimpleNamespace(_build_keepalive_http_client=broken_builder),
        dict(kwargs),
        reason="primary",
        shared=False,
    )
    assert isinstance(client, adapter.OllamaNativeClient)
    assert calls == ["CA_CTX"]  # called once — never re-called without verify
    assert client._http is not keepalive
    keepalive.close()


def test_hook3_gate_splits_query_bearing_base_url(monkeypatch):
    """A config base_url carrying URL-token auth must probe a well-formed
    /api/version URL with the token as query params — not a mangled URL that
    404s into a definitive 'not Ollama' and drops num_ctx injection."""
    import sys

    monkeypatch.setenv("HERMES_OLLAMA_NATIVE", "1")
    captured = {}
    monkeypatch.setattr(
        adapter, "_probe_is_ollama", lambda root, **kw: captured.update(kw, root=root) or True
    )
    fake_config = types.SimpleNamespace(
        load_config_readonly=lambda: {
            "custom_providers": [{"name": "myoll", "base_url": "http://h:11434/v1?token=abc"}]
        }
    )
    fake_providers = types.SimpleNamespace(
        custom_provider_slug=lambda n: "custom:" + n.strip().lower().replace(" ", "-")
    )
    fake_pkg = types.SimpleNamespace(config=fake_config, providers=fake_providers)
    monkeypatch.setitem(sys.modules, "hermes_cli", fake_pkg)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", fake_config)
    monkeypatch.setitem(sys.modules, "hermes_cli.providers", fake_providers)

    assert shim._named_custom_is_ollama("custom:myoll") is True
    assert captured["root"] == "http://h:11434"  # query split off, /v1 stripped
    assert captured.get("params") == {"token": "abc"}  # token still reaches the probe


def test_hook3_noop_when_flag_off(monkeypatch):
    monkeypatch.delenv("HERMES_OLLAMA_NATIVE", raising=False)

    def get_provider_profile(name):
        return {"custom": "P"}.get(name)

    fake = types.SimpleNamespace(get_provider_profile=get_provider_profile)
    shim._patch_profiles(fake)
    assert fake.get_provider_profile("custom:ollama-2") is None  # no fallback when off


def test_hook2_degrades_on_unexpected_return_arity(monkeypatch):
    monkeypatch.setenv("HERMES_OLLAMA_NATIVE", "1")

    def orig(*a, **k):
        return ("only-one-value",)  # not the expected (client, model) shape

    fake = types.SimpleNamespace(resolve_provider_client=orig, resolve_vision_provider_client=orig)
    shim._patch_aux(fake)
    # The bad arity is swallowed; the original result is returned unchanged.
    assert fake.resolve_provider_client() == ("only-one-value",)
