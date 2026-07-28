# hermes-ollama-native

[![CI](https://github.com/hanzlamateen/hermes-ollama-native/actions/workflows/ci.yml/badge.svg)](https://github.com/hanzlamateen/hermes-ollama-native/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/hermes-ollama-native.svg)](https://pypi.org/project/hermes-ollama-native/)
[![Python](https://img.shields.io/pypi/pyversions/hermes-ollama-native.svg)](https://pypi.org/project/hermes-ollama-native/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Make a local [Hermes Agent](https://github.com/NousResearch/hermes-agent) talk to **Ollama's native `/api/chat` API** instead of its OpenAI-compatible `/v1` endpoint — so per-request `num_ctx` is honored and tool calls stream correctly.

> **Status:** an explicit, temporary stopgap until the native adapter lands upstream in
> [NousResearch/hermes-agent#4505](https://github.com/NousResearch/hermes-agent/issues/4505).
> When that merges, you won't need this package.

## The problem

When Hermes runs against a local Ollama server, it speaks Ollama's OpenAI-compatible `/v1/chat/completions` endpoint. That endpoint has two costly limitations:

1. **It silently ignores per-request `num_ctx`.** Ollama maps only a fixed allowlist of OpenAI sampling params to model options; `num_ctx` isn't one of them. So every model loads at Ollama's **4096** context default no matter what context window Hermes computed — regardless of whether you send `num_ctx` nested in `options` or top-level. (Verified through Ollama `v0.30.11` / `main`; tracked in Ollama issue [#11409](https://github.com/ollama/ollama/issues/11409).)
2. **It can corrupt streamed `tool_calls` deltas.** Multiple users report that the same local model returns broken tool calls via `/v1` but correct tool calls via native `/api/chat` — which makes an *agentic* assistant unable to act.

Ollama's **native `/api/chat`** endpoint honors `options.num_ctx` and returns correct tool-call deltas. This package routes Hermes' local-Ollama traffic there.

## How it works

`hermes-ollama-native` ships a drop-in, OpenAI-SDK-shaped client (`OllamaNativeClient`) that POSTs to `/api/chat` + `/api/embed` and translates requests, responses, NDJSON streams, tool calls, multimodal content, reasoning, and token usage back to the OpenAI shape Hermes already consumes. It is modeled directly on Hermes' own `gemini_native_adapter.py`, so `api_mode` stays `chat_completions` and the rest of the agent loop is unchanged.

A small runtime shim wraps two of Hermes' client-construction chokepoints and returns the native client **only** when the target is positively identified as Ollama:

- **Hook 1 — main chat:** `agent.agent_runtime_helpers.create_openai_client` (covers the primary client and the per-request sync + stream clients).
- **Hook 2 — side tasks:** `agent.auxiliary_client.resolve_provider_client` / `resolve_vision_provider_client` (compression, title, vision, web-extract). On by default; set `HERMES_OLLAMA_NATIVE_AUX=0` to disable just this.

Detection is defensive by construction: it confirms the endpoint is Ollama with a cached `GET /api/version` probe before routing, so it never mis-routes a non-Ollama "custom" endpoint such as vLLM, llama.cpp, or LM Studio. Everything is wrapped in `try/except` and **degrades to stock `/v1` on any error**, so it can never crash agent startup.

> ⚠️ **It monkeypatches private Hermes internals.** A future Hermes release could rename those functions; the shim is symbol-anchored and fails safely (reverts to `/v1`) rather than crashing, but native mode would silently stop engaging until this package is updated. This is why the durable home is upstream #4505.

## Install

```bash
pip install hermes-ollama-native
```

Requires Python 3.10+ and a Hermes Agent installed in the same environment. The only runtime dependency is `httpx`.

## Quickstart

Native mode only engages when the `HERMES_OLLAMA_NATIVE` environment variable is truthy (`1`/`true`/`yes`/`on`). Pick one of two activation styles:

**Option 1 — persistent (recommended).** Install a one-line startup hook into the current environment so every Hermes process picks it up automatically:

```bash
export HERMES_OLLAMA_NATIVE=1
hermes-ollama-native install-pth
hermes            # run Hermes as usual
```

**Option 2 — explicit.** Activate from your own launcher/entrypoint before the agent builds its client (no startup hook, nothing global):

```python
import hermes_ollama_native

hermes_ollama_native.activate()
```

Either way, point Hermes at your local Ollama as a custom endpoint (the standard local-Ollama config — you do **not** need to strip `/v1` yourself; the adapter handles it):

```yaml
# ~/.hermes/config.yaml
model:
  default: "gemma4:e2b"
  provider: "custom"
  base_url: "http://localhost:11434/v1"
```

**Verify it worked.** With a model loaded through Hermes, check the context Ollama actually allocated:

```bash
ollama ps      # CONTEXT should be your configured value, not 4096
```

## CLI

```text
hermes-ollama-native status         # show flag + startup-hook state
hermes-ollama-native install-pth     # persist auto-activation in this environment
hermes-ollama-native uninstall-pth   # remove the startup hook
```

## Configuration

| Environment variable | Default | Effect |
|---|---|---|
| `HERMES_OLLAMA_NATIVE` | _(unset)_ | Master switch. Unset/false → the patch is a complete no-op and Hermes stays on `/v1`. Set truthy → native `/api/chat` routing for confirmed-Ollama endpoints. |
| `HERMES_OLLAMA_NATIVE_AUX` | `1` | When native mode is on, also route auxiliary side-tasks (compression/title/vision) through native. Set `0` to keep only the main chat path on native. |

**Turning it off** needs no "deactivate" step: every hook re-checks `HERMES_OLLAMA_NATIVE` per call, so unsetting it transparently reverts the agent to `/v1` (run `uninstall-pth` too if you installed the startup hook).

## Compatibility & limitations

- Targets a Hermes Agent that exposes `agent.agent_runtime_helpers.create_openai_client` and `agent.auxiliary_client.resolve_provider_client` / `resolve_vision_provider_client` (the structure as of Hermes `v0.17.x`).
- Patches **private** internals — see the warning above. Pinned/tested against the Hermes releases noted in the changelog.
- Only routes endpoints positively identified as Ollama via `GET /api/version`. vLLM / llama.cpp / LM Studio (also `provider: custom`) are left on `/v1`.

## Development

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest -q
```

## Status & license

This is a standalone, general-purpose adapter; its durable home is the upstream contribution tracked in [NousResearch/hermes-agent#4505](https://github.com/NousResearch/hermes-agent/issues/4505). Once that lands, this package becomes unnecessary.

Licensed under the [MIT License](LICENSE).
