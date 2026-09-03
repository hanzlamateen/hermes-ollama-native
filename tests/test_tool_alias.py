"""Regression for Ollama's XML marker colliding with Hermes's bridge name."""

import copy
import json

import httpx
import pytest

from hermes_ollama_native.adapter import AsyncOllamaNativeClient, OllamaNativeClient
from hermes_ollama_native.tool_alias import alias_request, restore_response


def request(model="granite4:3b-h"):
    return {
        "model": model,
        "options": {"num_ctx": 65536},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "tool_call",
                    "description": "Use tool_call for deferred tools.",
                    "parameters": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}},
                        "required": ["name", "arguments"],
                    },
                },
            }
        ],
        "messages": [
            {"role": "system", "content": "Use `tool_call`; preserve <tool_call></tool_call>."},
            {"role": "user", "content": "Show literal tool_call from my file."},
            {
                "role": "assistant",
                "content": "tool_call",
                "tool_calls": [
                    {
                        "id": "abc",
                        "function": {
                            "name": "tool_call",
                            "arguments": {
                                "name": "process",
                                "arguments": {"action": "list", "text": "tool_call"},
                            },
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "name": "tool_call",
                "tool_call_id": "abc",
                "content": "File contains tool_call",
            },
        ],
    }


def test_request_preserves_schemas_arguments_ids_and_user_data():
    original = request()
    snapshot = copy.deepcopy(original)
    converted, names = alias_request(original)
    assert original == snapshot
    assert names == {"hermes_dispatch": "tool_call"}
    assert converted["options"] == original["options"]
    assert (
        converted["tools"][0]["function"]["parameters"]
        == original["tools"][0]["function"]["parameters"]
    )
    assert (
        converted["messages"][0]["content"]
        == "Use `hermes_dispatch`; preserve <tool_call></tool_call>."
    )
    assert converted["messages"][1] == original["messages"][1]
    call = converted["messages"][2]["tool_calls"][0]
    assert call["function"]["name"] == "hermes_dispatch"
    assert (
        call["function"]["arguments"]
        == original["messages"][2]["tool_calls"][0]["function"]["arguments"]
    )
    assert call["id"] == "abc"
    assert converted["messages"][2]["content"] == "tool_call"
    assert converted["messages"][3]["name"] == "hermes_dispatch"
    assert converted["messages"][3]["content"] == original["messages"][3]["content"]


@pytest.mark.parametrize("model", ["qwen3.5:4b", "granite3.3:8b", "gemma4:26b"])
def test_other_models_unchanged(model):
    original = request(model)
    converted, names = alias_request(original)
    assert converted is original and names == {}


def test_opt_out(monkeypatch):
    monkeypatch.setenv("HERMES_OLLAMA_GRANITE_TOOL_ALIAS", "0")
    original = request()
    assert alias_request(original) == (original, {})


def test_no_bridge_is_noop():
    original = request()
    original["tools"] = []
    original["messages"] = []
    assert alias_request(original)[0] is original


def test_alias_collision_and_history_without_current_bridge():
    original = request()
    original["tools"][0]["function"]["name"] = "hermes_dispatch"
    converted, names = alias_request(original)
    assert names == {"hermes_dispatch_1": "tool_call"}
    assert converted["tools"][0]["function"]["name"] == "hermes_dispatch"
    assert converted["messages"][2]["tool_calls"][0]["function"]["name"] == "hermes_dispatch_1"


def test_restore_changes_names_only():
    payload = {
        "message": {
            "content": "hermes_dispatch",
            "tool_calls": [
                {
                    "id": "abc",
                    "function": {
                        "name": "hermes_dispatch",
                        "arguments": {"name": "process", "arguments": {"text": "hermes_dispatch"}},
                    },
                },
                {"id": "def", "function": {"name": "read_file", "arguments": {"path": "x"}}},
            ],
        }
    }
    snapshot = copy.deepcopy(payload)
    restored = restore_response(payload, {"hermes_dispatch": "tool_call"})
    assert payload == snapshot
    assert restored["message"]["tool_calls"][0]["function"]["name"] == "tool_call"
    restored["message"]["tool_calls"][0]["function"]["name"] = "hermes_dispatch"
    assert restored == payload


def mock_client(stream):
    def handler(req):
        body = json.loads(req.content)
        assert body["options"]["num_ctx"] == 65536
        assert body["tools"][0]["function"]["name"] == "hermes_dispatch"
        calls = [
            {
                "id": "abc",
                "function": {
                    "name": "hermes_dispatch",
                    "arguments": {"name": "process", "arguments": {"action": "list"}},
                },
            }
        ]
        if stream:
            frames = [
                {"message": {"tool_calls": calls}},
                {"done": True, "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 2},
            ]
            return httpx.Response(200, text="\n".join(map(json.dumps, frames)))
        return httpx.Response(200, json={"message": {"tool_calls": calls}, "done": True})

    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("stream", [False, True])
def test_adapter_restores_legitimate_bridge_before_dispatch(stream):
    original = request()
    client = OllamaNativeClient(base_url="http://localhost:11434", http_client=mock_client(stream))
    result = client.chat.completions.create(
        model=original["model"],
        messages=original["messages"],
        tools=original["tools"],
        extra_body={"options": {"num_ctx": 65536}},
        stream=stream,
    )
    calls = (
        [
            tc
            for c in result
            for choice in c.choices
            for tc in (getattr(choice.delta, "tool_calls", None) or [])
        ]
        if stream
        else result.choices[0].message.tool_calls
    )
    assert calls[0].function.name == "tool_call"
    assert json.loads(calls[0].function.arguments) == {
        "name": "process",
        "arguments": {"action": "list"},
    }
    client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_async_adapter_uses_same_translation(stream):
    original = request()
    client = AsyncOllamaNativeClient(
        OllamaNativeClient(base_url="http://localhost:11434", http_client=mock_client(stream))
    )
    result = await client.chat.completions.create(
        model=original["model"],
        messages=original["messages"],
        tools=original["tools"],
        extra_body={"options": {"num_ctx": 65536}},
        stream=stream,
    )
    if stream:
        calls = []
        async for chunk in result:
            for choice in chunk.choices:
                calls.extend(getattr(choice.delta, "tool_calls", None) or [])
    else:
        calls = result.choices[0].message.tool_calls
    assert calls[0].function.name == "tool_call"
