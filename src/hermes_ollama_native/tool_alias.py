"""Request-local workaround for Granite's <tool_call> / tool_call name collision.

Ollama 0.33.2's generic parser selects a tool name from the opening XML tag.
Keep the bridge available under a wire-only alias, restoring it before Hermes
dispatch. Never infer tool names from malformed arguments.
"""

from __future__ import annotations

import copy
import os
import re
from typing import Any

_REFERENCE = re.compile(r"(?<![\w</])tool_call(?![\w>])")


def alias_request(request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Copy only affected requests; return wire-name -> original-name mapping."""
    model = str(request.get("model", "")).lower().rsplit("/", 1)[-1]
    if not model.startswith(("granite4", "granite-4")):
        return request, {}
    if os.getenv("HERMES_OLLAMA_GRANITE_TOOL_ALIAS", "1").strip().lower() in {
        "0",
        "false",
        "off",
        "no",
    }:
        return request, {}

    names = {t.get("function", {}).get("name") for t in request.get("tools", [])}
    # Also translate old bridge history if the current turn no longer exposes it.
    history_names = {
        c.get("function", {}).get("name")
        for m in request.get("messages", [])
        for c in m.get("tool_calls", [])
    }
    if "tool_call" not in names | history_names:
        return request, {}
    alias = "hermes_dispatch"
    suffix = 0
    while alias in names | history_names:
        suffix += 1
        alias = f"hermes_dispatch_{suffix}"

    result = copy.deepcopy(request)
    for tool in result.get("tools", []):
        fn = tool.get("function", {})
        if fn.get("name") == "tool_call":
            fn["name"] = alias
        if isinstance(fn.get("description"), str):
            fn["description"] = _REFERENCE.sub(alias, fn["description"])
    for message in result.get("messages", []):
        # Only instruction references; never rewrite user data, tool results,
        # arguments, or assistant prose that may contain literal file contents.
        if message.get("role") in {"system", "developer"}:
            if isinstance(message.get("content"), str):
                message["content"] = _REFERENCE.sub(alias, message["content"])
        for call in message.get("tool_calls", []):
            fn = call.get("function", {})
            if fn.get("name") == "tool_call":
                fn["name"] = alias
        if message.get("role") == "tool":
            for key in ("name", "tool_name"):
                if message.get(key) == "tool_call":
                    message[key] = alias
    return result, {alias: "tool_call"}


def restore_response(payload: dict[str, Any], names: dict[str, str]) -> dict[str, Any]:
    """Restore only declared function names in native responses/stream frames."""
    if not names:
        return payload
    result = copy.deepcopy(payload)
    for call in (result.get("message") or {}).get("tool_calls") or []:
        fn = call.get("function") or {}
        if fn.get("name") in names:
            fn["name"] = names[fn["name"]]
    return result
