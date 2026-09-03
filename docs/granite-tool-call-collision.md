# Granite 4 tool-call collision in Ollama

## Scope

Validated on 2026-09-03 with Hermes Agent 0.21.0, Ollama 0.33.2,
`granite4:3b-h`, a 65536-token context, and this adapter based on upstream
`hermes-ollama-native` v0.3.3 (`f4a4971`).

Hermes exposes a deferred bridge whose real function name is `tool_call`.
Granite 4 formats native calls inside an XML marker named `<tool_call>`. Ollama's
generic parser scans its buffer for any exposed tool name before it extracts the
JSON call. When the Hermes bridge is present, the marker itself matches the real
tool name first. Ollama can therefore return a call named `tool_call` while
retaining arguments intended for `search_files`, `terminal`, or another tool.
Hermes then rejects those arguments because the bridge requires `name` and
`arguments`.

The relevant upstream inputs are Ollama's
[`findTool` parser](https://github.com/ollama/ollama/blob/v0.33.2/tools/tools.go)
and the [`granite4:3b-h` template](https://ollama.com/library/granite4:3b-h/blobs/0f6ec9740c76).

## Fix

For Granite 4 requests that expose the bridge, the adapter deep-copies the
request and gives `tool_call` a collision-free wire name, normally
`hermes_dispatch`. It changes the function schema, its system/developer
instruction references, and matching tool-call history. It does not rewrite XML
markers, user text, tool results, arguments, IDs, or parameter schemas. Responses
are mapped back to `tool_call` before Hermes sees them. Other model families are
unchanged.

The workaround is enabled by default for Granite 4. Set
`HERMES_OLLAMA_GRANITE_TOOL_ALIAS=0` to disable it for diagnosis.

## Validation

The automated suite has 86 passing tests, including sync, streaming, async,
history, alias collision, input immutability, opt-out, and non-Granite cases.

The live suite ran nine fresh Hermes conversations against Ollama at 65536
context while retaining every Hermes tool available in the environment. Runtime
guards constrained effects to the read-only fixture; they did not remove tools
from the model-facing schema. Automatic title generation was disabled only in
the test process because its concurrent auxiliary request repeatedly reloaded the
same local model at a different context.

| Task | Routing | Exact requested result | Observed model behavior |
|---|---:|---:|---|
| List files, 3 runs | 3/3 | 3/3 | Correct filenames in every run. |
| First 20 lines, 3 runs | 3/3 | 1/3 | One response replaced ASCII hyphens with Unicode hyphens. One read the correct text but also attempted an unnecessary, blocked `write_file`; its final text was exact. |
| Chapter 2 next 20 lines, 3 runs | 3/3 | 2/3 | One response stopped after 14 of 20 lines. |

All nine runs avoided the original malformed bridge call. The remaining
failures concern Granite's instruction following and output fidelity, rather than
adapter routing. Warm list runs took about 11.6 seconds; the first list run took
18.3 seconds. Read and chapter runs took 22.7 to 29.9 seconds.

After installing the wheel into Hermes, a separate three-run list check imported
the adapter from `site-packages` and passed routing and exact-output validation
3/3. The cold run took 53.0 seconds and the warm runs took 11.89 and 11.94
seconds. `ollama ps` reported 2.8 GB, 100% GPU, and a 65536-token context.

Run the automated checks from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
```

## Install and rollback

Build the patched wheel:

```powershell
.\.venv\Scripts\python.exe -m build
```

Install it into the Hermes environment only after stopping active Hermes
sessions:

```powershell
$hermesPython = 'C:\path\to\hermes-agent\venv\Scripts\python.exe'
$wheel = Get-ChildItem .\dist\*.whl | Sort-Object LastWriteTime -Descending | Select-Object -First 1
& $hermesPython -m pip install --force-reinstall --no-deps $wheel.FullName
& $hermesPython -m hermes_ollama_native status
```

Rollback to the published package:

```powershell
& $hermesPython -m pip install --force-reinstall 'hermes-ollama-native==0.3.3'
& $hermesPython -m hermes_ollama_native status
```

The `.pth` startup hook is independent of the wheel replacement and remains in
place. The master `HERMES_OLLAMA_NATIVE` switch still controls native routing.
