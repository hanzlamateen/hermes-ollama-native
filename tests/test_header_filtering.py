"""Regression: when the aux path mirrors an OpenAI SDK client's default_headers,
those can contain the SDK's Omit()/NotGiven() sentinels for unset headers (e.g.
OpenAI-Organization / OpenAI-Project). httpx rejects non-str/bytes header values,
which would break every aux side-task — so the adapter must drop them."""

from __future__ import annotations

from hermes_ollama_native.adapter import OllamaNativeClient


class _Sentinel:
    """Stand-in for the OpenAI SDK's Omit()/NotGiven() unset-header markers."""


def test_non_string_header_values_are_dropped():
    client = OllamaNativeClient(
        base_url="http://h:11434",
        default_headers={
            "X-Real": "keep-me",
            "X-Bytes": b"bytes-ok",
            "OpenAI-Organization": _Sentinel(),  # must be dropped, not sent to httpx
            "OpenAI-Project": None,
        },
    )
    headers = client._headers()
    assert headers["X-Real"] == "keep-me"
    assert headers["X-Bytes"] == b"bytes-ok"
    assert "OpenAI-Organization" not in headers
    assert "OpenAI-Project" not in headers
    # Every surviving value must be str/bytes (what httpx accepts).
    assert all(isinstance(v, (str, bytes)) for v in headers.values())


def test_authorization_only_for_non_sentinel_api_key():
    # The local "ollama" sentinel api_key → no Authorization header.
    assert "Authorization" not in OllamaNativeClient(base_url="http://h:11434")._headers()
    assert (
        "Authorization"
        not in OllamaNativeClient(base_url="http://h:11434", api_key="ollama")._headers()
    )
    # A real key (e.g. a reverse proxy in front of Ollama) → Bearer header.
    headers = OllamaNativeClient(base_url="http://h:11434", api_key="secret-token")._headers()
    assert headers["Authorization"] == "Bearer secret-token"
