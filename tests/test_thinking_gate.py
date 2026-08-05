"""The /api/show thinking-capability gate: `think` is only ever suppressed on a
positive "model lacks the thinking capability" signal — newer Ollama 400s when
`think` is sent to such a model, while older (pre-capability) servers accept it
for everything."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import respx

from hermes_ollama_native import adapter
from hermes_ollama_native.adapter import OllamaAPIError, OllamaNativeClient

_CHAT_OK = {"message": {"role": "assistant", "content": "ok"}, "done_reason": "stop"}


def _mock_show_and_chat(show_json):
    respx.post("http://h:11434/api/show").mock(return_value=httpx.Response(200, json=show_json))
    return respx.post("http://h:11434/api/chat").mock(
        return_value=httpx.Response(200, json=_CHAT_OK)
    )


@respx.mock
def test_think_dropped_for_non_thinking_model():
    chat = _mock_show_and_chat({"capabilities": ["completion", "tools"]})
    client = OllamaNativeClient(base_url="http://h:11434")
    client.chat.completions.create(
        model="llama3.2", messages=[{"role": "user", "content": "x"}], extra_body={"think": False}
    )
    sent = json.loads(chat.calls.last.request.content)
    assert "think" not in sent


@respx.mock
def test_think_kept_for_thinking_model():
    chat = _mock_show_and_chat({"capabilities": ["completion", "tools", "thinking"]})
    client = OllamaNativeClient(base_url="http://h:11434")
    client.chat.completions.create(
        model="qwen3.5", messages=[{"role": "user", "content": "x"}], extra_body={"think": False}
    )
    sent = json.loads(chat.calls.last.request.content)
    assert sent["think"] is False


@respx.mock
def test_think_string_level_passes_through():
    # Newer Ollama accepts think as a string level ("low"/"medium"/"high"/"max");
    # the adapter forwards the value verbatim (no bool coercion).
    chat = _mock_show_and_chat({"capabilities": ["completion", "thinking"]})
    client = OllamaNativeClient(base_url="http://h:11434")
    client.chat.completions.create(
        model="gpt-oss", messages=[{"role": "user", "content": "x"}], extra_body={"think": "high"}
    )
    sent = json.loads(chat.calls.last.request.content)
    assert sent["think"] == "high"


@respx.mock
def test_reasoning_effort_translated_to_native_think():
    # Hermes's custom profile emits a configured reasoning level as TOP-LEVEL
    # reasoning_effort (the field /v1 honors), not extra_body.think — the native
    # client must translate it or configured levels silently fall back to the
    # server default.
    chat = _mock_show_and_chat({"capabilities": ["completion", "thinking"]})
    client = OllamaNativeClient(base_url="http://h:11434")
    client.chat.completions.create(
        model="gpt-oss", messages=[{"role": "user", "content": "x"}], reasoning_effort="high"
    )
    sent = json.loads(chat.calls.last.request.content)
    assert sent["think"] == "high"


@respx.mock
def test_reasoning_effort_none_maps_to_think_false():
    chat = _mock_show_and_chat({"capabilities": ["completion", "thinking"]})
    client = OllamaNativeClient(base_url="http://h:11434")
    client.chat.completions.create(
        model="gpt-oss", messages=[{"role": "user", "content": "x"}], reasoning_effort="none"
    )
    sent = json.loads(chat.calls.last.request.content)
    assert sent["think"] is False


@respx.mock
def test_explicit_think_wins_over_reasoning_effort():
    chat = _mock_show_and_chat({"capabilities": ["completion", "thinking"]})
    client = OllamaNativeClient(base_url="http://h:11434")
    client.chat.completions.create(
        model="gpt-oss",
        messages=[{"role": "user", "content": "x"}],
        reasoning_effort="high",
        extra_body={"think": "low"},
    )
    sent = json.loads(chat.calls.last.request.content)
    assert sent["think"] == "low"


@respx.mock
def test_think_kept_when_capabilities_absent_and_memoized():
    # Pre-capability Ollama: /api/show 200 without a capabilities key. A 200 is
    # definitive, so it must be memoized — not re-probed on every request.
    show = respx.post("http://h:11434/api/show").mock(
        return_value=httpx.Response(200, json={"model_info": {}})
    )
    chat = respx.post("http://h:11434/api/chat").mock(
        return_value=httpx.Response(200, json=_CHAT_OK)
    )
    client = OllamaNativeClient(base_url="http://h:11434")
    for _ in range(3):
        client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "x"}], extra_body={"think": True}
        )
    sent = json.loads(chat.calls.last.request.content)
    assert sent["think"] is True
    assert show.call_count == 1
    # Definitive (200) answers are memoized permanently: (supports, None).
    assert adapter._THINKING_CAP_CACHE[("http://h:11434", "m")][1] is None


@respx.mock
def test_think_kept_when_probe_fails_and_failure_is_ttl_cached():
    show = respx.post("http://h:11434/api/show").mock(side_effect=httpx.ConnectError("down"))
    chat = respx.post("http://h:11434/api/chat").mock(
        return_value=httpx.Response(200, json=_CHAT_OK)
    )
    client = OllamaNativeClient(base_url="http://h:11434")
    for _ in range(3):
        client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "x"}], extra_body={"think": False}
        )
    sent = json.loads(chat.calls.last.request.content)
    assert sent["think"] is False  # uncertainty never suppresses think (fail-open)
    # The failure is TTL-cached: a persistently failing /api/show must not add a
    # round trip to every chat call.
    assert show.call_count == 1
    # ...but TTL-cached, not permanent: (supports, expires_at) with a real expiry.
    assert adapter._THINKING_CAP_CACHE[("http://h:11434", "m")][1] is not None


@respx.mock
def test_think_kept_on_non_200_probe_and_bounded():
    # e.g. a proxy that 404s /api/show while /api/chat works, or an unknown model.
    show = respx.post("http://h:11434/api/show").mock(return_value=httpx.Response(404))
    chat = respx.post("http://h:11434/api/chat").mock(
        return_value=httpx.Response(200, json=_CHAT_OK)
    )
    client = OllamaNativeClient(base_url="http://h:11434")
    for _ in range(3):
        client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "x"}], extra_body={"think": True}
        )
    assert json.loads(chat.calls.last.request.content)["think"] is True
    assert show.call_count == 1  # non-200 TTL-cached, not re-probed per request
    assert adapter._THINKING_CAP_CACHE[("http://h:11434", "m")][1] is not None


@respx.mock
def test_probe_sends_auth_headers():
    # Behind an authenticated reverse proxy an unauthenticated probe would 401
    # and silently disable the gate — the probe must send the client's headers.
    show = respx.post("http://h:11434/api/show").mock(
        return_value=httpx.Response(200, json={"capabilities": ["completion"]})
    )
    respx.post("http://h:11434/api/chat").mock(return_value=httpx.Response(200, json=_CHAT_OK))
    client = OllamaNativeClient(base_url="http://h:11434", api_key="proxy-secret")
    client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "x"}], extra_body={"think": False}
    )
    assert show.calls.last.request.headers["Authorization"] == "Bearer proxy-secret"


@respx.mock
def test_no_probe_when_think_absent():
    show = respx.post("http://h:11434/api/show").mock(
        return_value=httpx.Response(200, json={"capabilities": []})
    )
    respx.post("http://h:11434/api/chat").mock(return_value=httpx.Response(200, json=_CHAT_OK))
    client = OllamaNativeClient(base_url="http://h:11434")
    client.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])
    assert show.call_count == 0  # zero overhead unless `think` is present


@respx.mock
def test_probe_result_is_cached_per_model():
    show = respx.post("http://h:11434/api/show").mock(
        return_value=httpx.Response(200, json={"capabilities": ["completion"]})
    )
    respx.post("http://h:11434/api/chat").mock(return_value=httpx.Response(200, json=_CHAT_OK))
    client = OllamaNativeClient(base_url="http://h:11434")
    for _ in range(3):
        client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "x"}], extra_body={"think": False}
        )
    assert show.call_count == 1  # definitive answer memoized
    assert adapter._THINKING_CAP_CACHE[("http://h:11434", "m")][1] is None


@respx.mock
def test_failure_ttl_expires_then_recovers_to_definitive(monkeypatch):
    """The load-bearing TTL contract: a cached failure is re-probed after the TTL
    lapses, and a now-healthy server yields a definitive answer that is memoized
    (and, for a non-thinking model, finally suppresses `think`).

    Time is virtualized via a module-local stub so only the adapter's clock moves.
    """
    import time as real_time

    now = {"t": 1000.0}
    monkeypatch.setattr(
        adapter, "time", SimpleNamespace(monotonic=lambda: now["t"], time=real_time.time)
    )

    calls = {"n": 0}

    def show_handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("down")
        return httpx.Response(200, json={"capabilities": ["completion"]})

    respx.post("http://h:11434/api/show").mock(side_effect=show_handler)
    chat = respx.post("http://h:11434/api/chat").mock(
        return_value=httpx.Response(200, json=_CHAT_OK)
    )
    client = OllamaNativeClient(base_url="http://h:11434")
    kw = dict(model="m", messages=[{"role": "user", "content": "x"}], extra_body={"think": False})

    client.chat.completions.create(**kw)  # probe fails -> fail-open, TTL-cached
    assert json.loads(chat.calls.last.request.content)["think"] is False
    client.chat.completions.create(**kw)  # within TTL -> cached, no re-probe
    assert calls["n"] == 1

    now["t"] += adapter._NEG_PROBE_TTL + 1  # TTL lapses
    client.chat.completions.create(**kw)  # re-probe -> definitive: no thinking cap
    assert calls["n"] == 2
    assert "think" not in json.loads(chat.calls.last.request.content)
    assert adapter._THINKING_CAP_CACHE[("http://h:11434", "m")] == (False, None)

    client.chat.completions.create(**kw)  # memoized -> no further probes
    assert calls["n"] == 2


@respx.mock
def test_error_body_message_is_parsed():
    """Ollama reports failures as {"error": "<str>"} — surface the message itself."""
    respx.post("http://h:11434/api/chat").mock(
        return_value=httpx.Response(400, json={"error": "model does not support thinking"})
    )
    client = OllamaNativeClient(base_url="http://h:11434")
    try:
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])
        raise AssertionError("expected OllamaAPIError")
    except OllamaAPIError as e:
        assert e.status_code == 400
        assert "model does not support thinking" in str(e)
        assert '{"error"' not in str(e)  # the message, not the raw JSON blob


@respx.mock
def test_error_body_non_json_falls_back_to_raw_text():
    # A plain-text error (overloaded proxy, HTML gateway page) must surface its
    # raw text, not "<unreadable>".
    respx.post("http://h:11434/api/chat").mock(
        return_value=httpx.Response(502, text="upstream proxy exploded")
    )
    client = OllamaNativeClient(base_url="http://h:11434")
    try:
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])
        raise AssertionError("expected OllamaAPIError")
    except OllamaAPIError as e:
        assert e.status_code == 502
        assert "upstream proxy exploded" in str(e)
        assert "<unreadable>" not in str(e)
