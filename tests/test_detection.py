"""Detection + client wiring tests: native_root URL handling, the flag-gated
/api/version probe that positively identifies Ollama, and an end-to-end chat call
that proves num_ctx reaches the wire."""

from __future__ import annotations

import json

import httpx
import respx

from hermes_ollama_native.adapter import (
    OllamaNativeClient,
    is_native_ollama_base_url,
    native_root,
)


def test_native_root_strips_v1_suffix():
    assert native_root("http://h:11434/v1") == "http://h:11434"
    assert native_root("http://h:11434/v1/") == "http://h:11434"
    assert native_root("http://h:11434/") == "http://h:11434"
    assert native_root("http://h:11434") == "http://h:11434"


def test_native_root_strips_surrounding_whitespace():
    # base_url sourced from a config file / env var often carries a trailing newline;
    # without stripping, the /v1 suffix check fails and the probe URL is malformed.
    assert native_root("http://h:11434/v1\n") == "http://h:11434"
    assert native_root("  http://h:11434/v1  ") == "http://h:11434"


def test_detection_off_when_flag_unset(monkeypatch):
    monkeypatch.delenv("HERMES_OLLAMA_NATIVE", raising=False)
    # No network should be touched when the flag is off.
    assert is_native_ollama_base_url("http://h:11434/v1") is False


@respx.mock
def test_detection_positive_for_ollama_endpoint(native_flag):
    respx.get("http://h:11434/api/version").mock(
        return_value=httpx.Response(200, json={"version": "0.30.0"})
    )
    # Works whether or not the configured URL carries the /v1 suffix.
    assert is_native_ollama_base_url("http://h:11434/v1") is True
    assert is_native_ollama_base_url("http://h:11434") is True


@respx.mock
def test_detection_negative_for_non_ollama(native_flag):
    # A non-Ollama "custom" endpoint (vLLM/llama.cpp/LM Studio) has no /api/version.
    respx.get("http://other:8000/api/version").mock(return_value=httpx.Response(404))
    assert is_native_ollama_base_url("http://other:8000/v1") is False


@respx.mock
def test_detection_rejects_200_without_version_field(native_flag):
    # A 200 whose JSON lacks a string "version" must not false-positive.
    respx.get("http://h:11434/api/version").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    assert is_native_ollama_base_url("http://h:11434") is False


@respx.mock
def test_chat_completion_posts_num_ctx_to_api_chat():
    """End-to-end through the client facade: the request that hits /api/chat must
    carry options.num_ctx, and the response is translated to OpenAI shape."""
    route = respx.post("http://h:11434/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "gemma4:e2b",
                "message": {"role": "assistant", "content": "hello"},
                "done_reason": "stop",
                "prompt_eval_count": 11,
                "eval_count": 3,
            },
        )
    )
    client = OllamaNativeClient(base_url="http://h:11434/v1")
    result = client.chat.completions.create(
        model="gemma4:e2b",
        messages=[{"role": "user", "content": "hi"}],
        extra_body={"options": {"num_ctx": 64000}},
    )

    sent = json.loads(route.calls.last.request.content)
    assert route.calls.last.request.url.path == "/api/chat"
    assert sent["options"]["num_ctx"] == 64000
    assert result.choices[0].message.content == "hello"
    assert result.usage.prompt_tokens == 11


@respx.mock
def test_streaming_chat_yields_openai_chunks():
    ndjson = (
        '{"message":{"content":"he"}}\n'
        '{"message":{"content":"llo"}}\n'
        '{"done":true,"done_reason":"stop","prompt_eval_count":5,"eval_count":2}\n'
    )
    respx.post("http://h:11434/api/chat").mock(return_value=httpx.Response(200, text=ndjson))
    client = OllamaNativeClient(base_url="http://h:11434")
    chunks = list(
        client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "hi"}], stream=True
        )
    )
    text = "".join(c.choices[0].delta.content or "" for c in chunks)
    assert text == "hello"
    assert chunks[-1].choices[0].finish_reason == "stop"


@respx.mock
def test_version_probe_sends_auth_material(native_flag):
    # Behind an authenticated reverse proxy / private CA, an unconfigured probe
    # would 401 and native routing would silently never engage — the probe must
    # carry the caller's auth headers and query params.
    from hermes_ollama_native.adapter import _OLLAMA_PROBE_CACHE

    _OLLAMA_PROBE_CACHE.clear()
    probe = respx.get("http://h:11434/api/version").mock(
        return_value=httpx.Response(200, json={"version": "0.32.5"})
    )
    assert (
        is_native_ollama_base_url(
            "http://h:11434/v1",
            api_key="proxy-secret",
            default_headers={"CF-Access-Client-Id": "cid"},
            default_query={"access_token": "tok"},
        )
        is True
    )
    req = probe.calls.last.request
    assert req.headers["Authorization"] == "Bearer proxy-secret"
    assert req.headers["CF-Access-Client-Id"] == "cid"
    assert dict(req.url.params) == {"access_token": "tok"}
    _OLLAMA_PROBE_CACHE.clear()


@respx.mock
def test_version_probe_positive_expires_and_redetects(native_flag, monkeypatch):
    """Positives are bounded by _VERSION_PROBE_TTL: a server swap on the same port
    re-detects within the TTL instead of hard-404ing every turn until restart."""
    import time as real_time
    from types import SimpleNamespace

    from hermes_ollama_native import adapter

    adapter._OLLAMA_PROBE_CACHE.clear()
    now = {"t": 5000.0}
    monkeypatch.setattr(
        adapter, "time", SimpleNamespace(monotonic=lambda: now["t"], time=real_time.time)
    )
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json={"version": "0.32.5"})  # Ollama
        return httpx.Response(404)  # server swapped (vLLM on the same port)

    respx.get("http://h:11434/api/version").mock(side_effect=handler)
    assert is_native_ollama_base_url("http://h:11434/v1") is True
    assert is_native_ollama_base_url("http://h:11434/v1") is True  # cached
    assert calls["n"] == 1
    now["t"] += adapter._VERSION_PROBE_TTL + 1
    assert is_native_ollama_base_url("http://h:11434/v1") is False  # re-detected
    assert calls["n"] == 2
    adapter._OLLAMA_PROBE_CACHE.clear()


@respx.mock
def test_stream_error_frame_raises_with_message():
    # Ollama reports mid-stream failures (runner OOM) as an in-band {"error": ...}
    # frame on the already-200 stream — surface the message, don't end cleanly.
    import pytest

    from hermes_ollama_native.adapter import OllamaAPIError

    ndjson = '{"message":{"content":"par"}}\n{"error":"CUDA out of memory"}\n'
    respx.post("http://h:11434/api/chat").mock(return_value=httpx.Response(200, text=ndjson))
    client = OllamaNativeClient(base_url="http://h:11434")
    with pytest.raises(OllamaAPIError, match="CUDA out of memory"):
        list(
            client.chat.completions.create(
                model="m", messages=[{"role": "user", "content": "x"}], stream=True
            )
        )


@respx.mock
def test_stream_transport_errors_propagate_unwrapped():
    # Transport failures keep their concrete httpx types so Hermes's mid-stream
    # reconnect (isinstance checks on httpx.ConnectError/RemoteProtocolError/...)
    # keeps working — a wrapped OllamaAPIError would disable reconnects.
    import pytest

    respx.post("http://h:11434/api/chat").mock(side_effect=httpx.ConnectError("peer gone"))
    client = OllamaNativeClient(base_url="http://h:11434")
    with pytest.raises(httpx.ConnectError):
        list(
            client.chat.completions.create(
                model="m", messages=[{"role": "user", "content": "x"}], stream=True
            )
        )


@respx.mock
def test_default_query_attached_to_requests():
    # URL-token auth / api-version params extracted from a query-bearing base_url
    # must reach every native request, like the stock OpenAI client's default_query.
    chat = respx.post("http://h:11434/api/chat").mock(
        return_value=httpx.Response(
            200, json={"message": {"role": "assistant", "content": "ok"}, "done_reason": "stop"}
        )
    )
    client = OllamaNativeClient(base_url="http://h:11434", default_query={"access_token": "sekrit"})
    client.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])
    assert dict(chat.calls.last.request.url.params) == {"access_token": "sekrit"}


def test_remote_image_url_degrades_to_text_marker():
    # Native /api/chat only accepts base64 images; a remote http(s) URL degrades
    # to a visible text marker instead of vanishing silently.
    from hermes_ollama_native.adapter import build_ollama_request

    body = build_ollama_request(
        model="m",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this? "},
                    {"type": "image_url", "image_url": {"url": "https://x.test/cat.png"}},
                ],
            }
        ],
    )
    msg = body["messages"][0]
    assert "[Image: https://x.test/cat.png]" in msg["content"]
    assert not msg.get("images")


def test_version_probe_passes_verify(native_flag, monkeypatch):
    # respx can't observe the verify kwarg — assert it at the httpx.get call
    # boundary. A dropped verify would silently fail private-CA endpoints.
    from unittest.mock import patch

    from hermes_ollama_native import adapter

    adapter._OLLAMA_PROBE_CACHE.clear()
    resp = httpx.Response(200, json={"version": "0.32.5"})
    with patch("hermes_ollama_native.adapter.httpx.get", return_value=resp) as get:
        assert is_native_ollama_base_url("http://h:11434/v1", verify="/etc/ssl/corp-ca.pem") is True
    assert get.call_args.kwargs["verify"] == "/etc/ssl/corp-ca.pem"
    adapter._OLLAMA_PROBE_CACHE.clear()
