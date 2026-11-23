# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.4] - 2026-11-23

### Fixed

- When Ollama stops a response with `done_reason="length"` after it has sent a
  tool call, the adapter now reports `length` instead of `tool_calls`. Hermes
  then treats the response as cut off and retries it instead of running the
  call. Ollama's own `/v1` maps it the same way: it only turns `stop` into
  `tool_calls`. Found by @kvnloo on NousResearch/hermes-agent#73285.

## [0.3.3] - 2026-08-05

### Fixed

- Top-level `reasoning_effort` — the format Hermes's custom profile emits for
  configured reasoning levels — is translated to the native `think` value
  instead of being silently discarded (which fell back to the server default
  for `low`/`high`/`max`). An explicit `extra_body.think` wins, `"none"` maps
  to `think=False` (preserving the profile's disabled pairing), and level
  strings pass through unchanged, exactly as Ollama's own `/v1`
  `reasoning_effort` mapping does — so `/v1` and native stay
  behavior-identical per level. The thinking-capability gate still applies
  after translation.

## [0.3.2] - 2026-07-29

### Fixed

- Hook 3 no longer mangles a query-bearing custom-provider base_url: the query
  string (URL-token auth) is split off before probing, so the probe hits a
  well-formed `/api/version` with the token as query params — instead of 404ing
  into a definitive "not Ollama" that silently dropped num_ctx injection for a
  genuine Ollama instance for up to an hour. The gate's probe also now carries
  the entry's TLS trust (its ssl fields, or the env CA bundles).
- Hook 1 resolves TLS `verify` unconditionally, matching stock
  `create_openai_client`: the resolver also honors the `HERMES_CA_BUNDLE` /
  `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE` / `CURL_CA_BUNDLE` env vars, which
  never appear in client_kwargs — previously a private-CA endpoint trusted via
  env var probe-failed as indeterminate and native routing never engaged.
- The probe's `default_headers` now override the api_key-derived Authorization,
  matching real-request `_headers()` precedence — a placeholder api_key with
  real gateway auth in `extra_headers` now probes with the credentials requests
  will actually send.
- The keepalive builder's older-signature fallback uses signature inspection
  instead of `except TypeError`, which could also swallow a TypeError raised
  inside the builder and silently rebuild the client without the custom CA.
- `data:` image URLs are matched case-insensitively (RFC 2397): an uppercase
  `DATA:` base64 URL is decoded as an image instead of leaking a truncated
  base64 marker into the prompt text.
- An `{"error": ""}` error body no longer produces a bare-trailing-colon
  message; the raw body is kept as the diagnostic.

## [0.3.1] - 2026-07-28

### Fixed

- The `/api/version` probe now sends the endpoint's auth material (api_key /
  default headers / query params) and honors the resolved TLS `verify` — behind an
  authenticated reverse proxy or a private CA the unauthenticated probe would 401 /
  SSLError and native routing silently never engaged for exactly those deployments.
  A 401/403 probe response now logs at warning instead of staying silent.
- Streaming transport errors (ReadTimeout, ConnectError, RemoteProtocolError, …)
  propagate with their concrete httpx types instead of being wrapped in
  `OllamaAPIError` — Hermes's mid-stream reconnect classifies on isinstance checks
  against those exact types, so the wrap silently disabled reconnects and could
  misroute a dropped peer into history compression.
- In-band `{"error": "<str>"}` NDJSON frames on an already-200 stream (runner OOM,
  model crash) now raise with the server's actual message instead of being silently
  dropped as a clean-looking end of stream.
- `/api/version` probe caching reworked: definitive outcomes (positive, or
  provably-not-Ollama — 404 / wrong-shape 200) are cached with a bounded 1h TTL,
  so a server swap on the same port re-detects within the hour (instead of
  hard-404ing every turn until restart) and a non-Ollama endpoint costs at most
  one probe per hour on the client-build path (was one per minute). Indeterminate
  failures (transport blip, 401/403, 5xx) keep the previous 60s retry TTL, so a
  restarting Ollama or a fixed credential still recovers within a minute.
- Non-base64 `image_url` parts (remote http(s) URLs) degrade to a visible
  `[Image: <url>]` text marker plus a warning log instead of vanishing silently —
  native `/api/chat` only accepts base64 images, and a silent drop had the model
  confidently describing an image it never received.
- `default_query` (URL-token auth, api-version params extracted from a
  query-bearing base_url) is accepted by the client and forwarded on every native
  request, matching the stock OpenAI client's behavior.
- Hook 3 (`custom:<name>` → `custom` profile resolution) is now gated on the named
  instance's endpoint actually identifying as Ollama (resolved best-effort from
  config — both the legacy `custom_providers` list and the keyed `providers`
  schema, using the entry's api_key/key_env/extra_headers for the probe): the
  profile injects a default `max_tokens` and reasoning fields that gateways with
  lower completion caps or strict request validation can reject. The gate fails
  closed only on a definitive "not Ollama" probe answer; unresolvable instances
  and indeterminate probe failures keep the previous behavior (fail-open toward
  Ollama, matching this package's opt-in intent).

## [0.3.0] - 2026-07-13

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

[0.3.4]: https://github.com/hanzlamateen/hermes-ollama-native/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/hanzlamateen/hermes-ollama-native/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/hanzlamateen/hermes-ollama-native/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/hanzlamateen/hermes-ollama-native/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/hanzlamateen/hermes-ollama-native/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/hanzlamateen/hermes-ollama-native/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/hanzlamateen/hermes-ollama-native/releases/tag/v0.1.0
