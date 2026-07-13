# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Thinking-capability gate: `think` is only sent when the model advertises the
  `thinking` capability (probed via `/api/show` with the client's auth headers;
  definitive answers memoized per endpoint + model, failures TTL-cached). Newer
  Ollama 400s on a truthy `think` for models without the capability
  (`think: false` is tolerated); the gate drops `think` entirely for non-thinking
  models as the safe superset, mirroring Ollama's own relax path. Older
  (pre-capability) servers accept a forwarded `think` for any model; on a failed
  probe the gate fails open and forwards `think` unchanged. Zero overhead when
  `think` isn't requested. Suggested by @laurinaitis on
  NousResearch/hermes-agent#55606.

### Fixed

- Error messages now surface Ollama's `{"error": "<str>"}` body text directly
  instead of the raw JSON blob (raw body retained as fallback for non-JSON errors).

## [0.2.0] - 2026-07-03

### Added

- Hook 3: resolve a named custom instance (`custom:<name>`) to the `custom` provider
  profile so per-request `num_ctx` is injected for it. Without this, a `custom:<name>`
  Ollama endpoint routed to the native client would still load at Ollama's 4096 default
  (the registry lookup is an exact match, so the legacy request-build path dropped
  `num_ctx`). Gated on `HERMES_OLLAMA_NATIVE`.

### Fixed

- `native_root` now strips surrounding whitespace, so a `base_url` with a trailing
  newline (common from config files / env vars) no longer silently disables native
  routing.
- The native client keeps its long read timeout (600s) even when an external
  `http_client` is injected — a request with no explicit timeout no longer falls back
  to httpx's 5s default.
- Expose the httpx client as `_client` (alias of `_http`) so a host's socket-abort /
  interrupt path (which follows the OpenAI-SDK `_client` convention) can reach the
  connection pool.
- Tool-call `arguments` that JSON-decode to a non-object (`null` / array / scalar) are
  coerced to `{}` — Ollama expects an object.
- The optional startup `.pth` is now self-healing (a leftover `.pth` after
  `pip uninstall` no-ops instead of printing a `site` traceback on every start),
  installs into the site-packages dir that actually holds the package (correct for
  `--user`/system installs), and `_autoload` checks the flag before importing httpx.
  `uninstall-pth` is race-safe and `status` now reports `HERMES_OLLAMA_NATIVE_AUX`.
- Shim hardening: Hook 1 now injects the agent's keepalive `http_client` (faster
  dead-peer detection); the meta-path finder serializes its self-removal (thread-safe
  concurrent target imports) and degrades cleanly if a loader forbids wrapping; Hook 2
  tolerates an unexpected resolver return arity without raising.

### Tests

- Confirm multi-instance support: the shim selects the native client from the
  endpoint probe alone (it never inspects `agent.provider`), so several Ollama
  providers named `custom:<instance>` all use the native path. This makes the
  package immune to the upstream `custom:<instance>` selection bug fixed in
  NousResearch/hermes-agent#55606.

## [0.1.0] - 2026-06-30

### Added

- `OllamaNativeClient` / `AsyncOllamaNativeClient` — drop-in, OpenAI-SDK-shaped
  clients over Ollama's native `/api/chat` and `/api/embed` endpoints, translating
  requests, non-streaming responses, NDJSON streams, tool calls, multimodal content,
  reasoning, and token usage to the OpenAI shape the Hermes Agent consumes.
- Runtime shim activating two hooks when `HERMES_OLLAMA_NATIVE` is set:
  - Hook 1: `agent.agent_runtime_helpers.create_openai_client` (main chat).
  - Hook 2: `agent.auxiliary_client.resolve_provider_client` /
    `resolve_vision_provider_client` (side tasks; `HERMES_OLLAMA_NATIVE_AUX=0` to disable).
- Defensive `GET /api/version` detection (positive results memoized, negatives cached
  with a short TTL) so non-Ollama `custom` endpoints (vLLM/llama.cpp/LM Studio) are
  never mis-routed.
- Explicit `activate()` API and a `hermes-ollama-native` CLI
  (`install-pth` / `uninstall-pth` / `status`) for persistent, opt-in startup activation.

[Unreleased]: https://github.com/hanzlamateen/hermes-ollama-native/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/hanzlamateen/hermes-ollama-native/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/hanzlamateen/hermes-ollama-native/releases/tag/v0.1.0
