"""Translation unit tests — the request/response/stream/embeddings mapping between
the OpenAI shape the agent speaks and Ollama's native /api/chat + /api/embed wire
format. These are pure functions (no network)."""

from __future__ import annotations

import json

from hermes_ollama_native.adapter import _translate_embeddings as translate_embeddings
from hermes_ollama_native.adapter import (
    build_ollama_request,
    translate_ollama_response,
    translate_stream_line,
)

# ── request construction ──────────────────────────────────────────────────────


def test_num_ctx_is_preserved_verbatim():
    """The whole point: per-request num_ctx survives into the native payload."""
    req = build_ollama_request(
        model="gemma4:e2b",
        messages=[{"role": "user", "content": "hi"}],
        options={"num_ctx": 64000},
    )
    assert req["model"] == "gemma4:e2b"
    assert req["options"]["num_ctx"] == 64000
    assert req["stream"] is False


def test_sampling_params_merge_without_clobbering_options():
    req = build_ollama_request(
        model="m",
        messages=[{"role": "user", "content": "x"}],
        temperature=0.5,
        top_p=0.9,
        max_tokens=128,
        seed=7,
        stop="STOP",
        options={"num_ctx": 8192, "temperature": 0.1},  # explicit options win
    )
    opts = req["options"]
    assert opts["num_ctx"] == 8192
    assert opts["temperature"] == 0.1  # not overwritten by the 0.5 arg
    assert opts["top_p"] == 0.9
    assert opts["num_predict"] == 128  # max_tokens -> num_predict
    assert opts["seed"] == 7
    assert opts["stop"] == ["STOP"]  # string stop normalized to a list


def test_tools_pass_through_and_think_flag():
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    req = build_ollama_request(
        model="m",
        messages=[{"role": "user", "content": "x"}],
        tools=tools,
        think=True,
    )
    assert req["tools"] == tools
    assert req["think"] is True


def test_assistant_tool_call_arguments_string_to_object():
    """OpenAI sends tool-call arguments as a JSON *string*; Ollama wants an object."""
    req = build_ollama_request(
        model="m",
        messages=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "lookup", "arguments": '{"q": "cats"}'}}
                ],
            },
            {"role": "tool", "name": "lookup", "content": "result"},
        ],
    )
    assistant = req["messages"][0]
    assert assistant["tool_calls"][0]["function"]["arguments"] == {"q": "cats"}
    tool_msg = req["messages"][1]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_name"] == "lookup"


def test_tool_call_arguments_non_object_json_coerced_to_empty():
    # A JSON string that decodes to a non-object (null / array / scalar) must become
    # {} — Ollama expects tool-call arguments as an object, not None/list/int.
    for bad in ("null", "[1, 2]", "5", '"a string"'):
        req = build_ollama_request(
            model="m",
            messages=[
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "f", "arguments": bad}}],
                }
            ],
        )
        assert req["messages"][0]["tool_calls"][0]["function"]["arguments"] == {}, bad


def test_multimodal_image_data_url_extracted():
    req = build_ollama_request(
        model="m",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
                ],
            }
        ],
    )
    msg = req["messages"][0]
    assert msg["content"] == "what is this"
    assert msg["images"] == ["QUJD"]


# ── non-streaming response translation ────────────────────────────────────────


def test_response_content_and_usage_mapping():
    resp = translate_ollama_response(
        {
            "model": "m",
            "message": {"role": "assistant", "content": "hello"},
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "eval_count": 5,
        },
        model="m",
    )
    assert resp.choices[0].message.content == "hello"
    assert resp.choices[0].finish_reason == "stop"
    assert resp.usage.prompt_tokens == 10
    assert resp.usage.completion_tokens == 5
    assert resp.usage.total_tokens == 15


def test_response_tool_calls_object_to_string():
    resp = translate_ollama_response(
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}],
            },
        },
        model="m",
    )
    msg = resp.choices[0].message
    assert resp.choices[0].finish_reason == "tool_calls"
    assert msg.content is None  # content suppressed when only tool calls
    tc = msg.tool_calls[0]
    assert tc.type == "function"
    assert tc.function.name == "f"
    assert json.loads(tc.function.arguments) == {"a": 1}


def test_length_finish_reason():
    resp = translate_ollama_response(
        {"message": {"role": "assistant", "content": "x"}, "done_reason": "length"},
        model="m",
    )
    assert resp.choices[0].finish_reason == "length"


def test_length_finish_reason_wins_over_tool_calls():
    """A response Ollama cut off is incomplete even after a tool call, as on /v1."""
    tc = {"function": {"name": "look_up", "arguments": {"query": "x"}}}
    tool_idx = {"n": 0}
    streamed = translate_stream_line({"message": {"tool_calls": [tc]}}, "m", tool_idx)
    assert streamed[0].choices[0].delta.tool_calls[0].function.name == "look_up"

    terminal = translate_stream_line(
        {"done": True, "done_reason": "length", "prompt_eval_count": 4096}, "m", tool_idx
    )
    assert terminal[-1].choices[0].finish_reason == "length"

    resp = translate_ollama_response(
        {"message": {"role": "assistant", "tool_calls": [tc]}, "done_reason": "length"},
        model="m",
    )
    assert resp.choices[0].finish_reason == "length"


# ── streaming (NDJSON) translation ────────────────────────────────────────────


def test_stream_content_delta():
    chunks = translate_stream_line({"message": {"content": "hi"}}, "m", {"n": 0})
    assert len(chunks) == 1
    assert chunks[0].choices[0].delta.content == "hi"
    assert chunks[0].object == "chat.completion.chunk"


def test_stream_tool_call_delta_assigns_fresh_index():
    tool_idx = {"n": 0}
    line = {"message": {"tool_calls": [{"function": {"name": "f", "arguments": {"x": 1}}}]}}
    chunks = translate_stream_line(line, "m", tool_idx)
    delta = chunks[0].choices[0].delta
    assert delta.tool_calls[0].function.name == "f"
    assert json.loads(delta.tool_calls[0].function.arguments) == {"x": 1}
    assert delta.tool_calls[0].index == 0
    assert tool_idx["n"] == 1  # index counter advanced


def test_stream_done_emits_finish_and_usage():
    chunks = translate_stream_line(
        {"done": True, "done_reason": "stop", "prompt_eval_count": 3, "eval_count": 2},
        "m",
        {"n": 0},
    )
    final = chunks[-1]
    assert final.choices[0].finish_reason == "stop"
    assert final.usage.prompt_tokens == 3
    assert final.usage.completion_tokens == 2


def test_stream_thinking_becomes_reasoning_delta():
    chunks = translate_stream_line({"message": {"thinking": "hmm"}}, "m", {"n": 0})
    assert chunks[0].choices[0].delta.reasoning == "hmm"


# ── embeddings translation ────────────────────────────────────────────────────


def test_embeddings_translation():
    out = translate_embeddings(
        {
            "model": "nomic-embed-text",
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
            "prompt_eval_count": 4,
        },
        model="nomic-embed-text",
    )
    assert out.object == "list"
    assert [d.embedding for d in out.data] == [[0.1, 0.2], [0.3, 0.4]]
    assert out.data[0].index == 0
    assert out.usage.prompt_tokens == 4
