"""hermes-ollama-native — route a Hermes Agent's local Ollama traffic through the
native ``/api/chat`` API instead of the OpenAI-compatible ``/v1`` endpoint.

Why: Ollama's ``/v1`` endpoint silently ignores per-request ``num_ctx`` (so local
models load at the 4096 default) and can corrupt streamed ``tool_calls`` deltas.
The native ``/api/chat`` endpoint honors ``options.num_ctx`` and returns correct
tool calls. This package wraps Hermes' client-construction chokepoints with a
drop-in, OpenAI-SDK-shaped client over ``/api/chat`` + ``/api/embed``.

Activation is explicit — merely importing this package never monkeypatches anything:

  * Call :func:`activate` once before the agent builds its client, **or**
  * Install a persistent startup hook with ``hermes-ollama-native install-pth``.

Either way the patch is a no-op unless the ``HERMES_OLLAMA_NATIVE`` environment
variable is truthy. Because every installed hook re-checks the flag per call,
unsetting that variable transparently reverts the agent to ``/v1`` without an
explicit "deactivate" step.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .adapter import (
    AsyncOllamaNativeClient,
    OllamaNativeClient,
    is_native_ollama_base_url,
)
from .shim import install as _install

__all__ = [
    "OllamaNativeClient",
    "AsyncOllamaNativeClient",
    "is_native_ollama_base_url",
    "activate",
    "__version__",
]

try:
    __version__ = version("hermes-ollama-native")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"


def activate() -> None:
    """Install the runtime monkeypatch that routes native-Ollama endpoints through
    :class:`OllamaNativeClient`.

    Idempotent and safe to call repeatedly. No-ops unless ``HERMES_OLLAMA_NATIVE``
    is truthy; even once installed, every hook re-checks the flag per call, so the
    patch is transparent when the flag is off.
    """
    _install()
