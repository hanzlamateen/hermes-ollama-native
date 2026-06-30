"""The async facade (used by Hermes' auxiliary-client path) wraps the sync client.
Verify a non-streaming await round-trips and that base_url/api_key are exposed (the
aux Hook needs them to detect and mirror the client)."""

from __future__ import annotations

import httpx
import respx

from hermes_ollama_native.adapter import AsyncOllamaNativeClient, OllamaNativeClient


def test_async_client_exposes_base_url_and_api_key():
    sync = OllamaNativeClient(base_url="http://h:11434/v1", api_key="ollama")
    aclient = AsyncOllamaNativeClient(sync)
    assert aclient.base_url == "http://h:11434"
    assert aclient.api_key == "ollama"


@respx.mock
async def test_async_chat_completion_roundtrips():
    respx.post("http://h:11434/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "async hi"},
                "done_reason": "stop",
                "prompt_eval_count": 2,
                "eval_count": 1,
            },
        )
    )
    aclient = AsyncOllamaNativeClient(OllamaNativeClient(base_url="http://h:11434"))
    result = await aclient.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "hi"}]
    )
    assert result.choices[0].message.content == "async hi"
