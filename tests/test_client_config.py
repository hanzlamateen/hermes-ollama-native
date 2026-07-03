"""Client construction invariants: the long default timeout must survive even when
an external http_client is injected, and the httpx client must be reachable as
`_client` for the agent's socket-abort path."""

from __future__ import annotations

import httpx

from hermes_ollama_native.adapter import OllamaNativeClient


def test_default_timeout_applies_even_with_injected_client():
    # The agent injects a keepalive httpx.Client that carries httpx's short 5s
    # default; a request with no explicit timeout must still use our long default,
    # not fall back to 5s.
    injected = httpx.Client(timeout=httpx.Timeout(5.0))
    client = OllamaNativeClient(base_url="http://h:11434", http_client=injected)
    kwargs = client._timeout_kwargs(None)
    assert kwargs["timeout"] is client._default_timeout  # concrete, never {}
    assert isinstance(client._default_timeout, httpx.Timeout)
    assert client._default_timeout.read == 600.0


def test_explicit_timeout_is_passed_through():
    client = OllamaNativeClient(base_url="http://h:11434")
    t = httpx.Timeout(30.0)
    assert client._timeout_kwargs(t)["timeout"] is t


def test_constructor_timeout_becomes_default():
    t = httpx.Timeout(123.0)
    client = OllamaNativeClient(base_url="http://h:11434", timeout=t)
    assert client._default_timeout is t


def test_client_alias_exposed_for_socket_abort():
    # The agent resolves the pool via getattr(client, "_client"); alias it to _http.
    client = OllamaNativeClient(base_url="http://h:11434")
    assert client._client is client._http
