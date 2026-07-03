# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/hanzlamateen/hermes-ollama-native/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/hanzlamateen/hermes-ollama-native/releases/tag/v0.1.0
