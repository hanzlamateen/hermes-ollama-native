"""Startup hook imported by the optional ``.pth`` (installed via
``hermes-ollama-native install-pth``).

Python's ``site`` module executes ``import`` lines in ``.pth`` files at interpreter
startup, so importing this module activates the native-Ollama patch for every
process in the environment. Wrapped so it can never break a host process, and a
no-op unless ``HERMES_OLLAMA_NATIVE`` is truthy.
"""

from __future__ import annotations

try:
    from hermes_ollama_native import activate

    activate()
except Exception:  # never break interpreter startup
    import logging

    logging.getLogger("hermes_ollama_native.autoload").warning(
        "hermes-ollama-native autoload failed; agent stays on /v1", exc_info=True
    )
