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
