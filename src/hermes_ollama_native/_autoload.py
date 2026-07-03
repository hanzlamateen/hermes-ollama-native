"""Startup hook imported by the optional ``.pth`` (installed via
``hermes-ollama-native install-pth``).

Python's ``site`` module runs the ``.pth``'s ``import`` line at interpreter startup,
which imports this module. To keep that startup path cheap for *every* process in the
environment, we check the ``HERMES_OLLAMA_NATIVE`` flag with a plain ``os.environ``
read **before** importing the package (which would pull in httpx). When the flag is
off this module does essentially nothing. Wrapped so it can never break startup.
"""

from __future__ import annotations

import os

if os.environ.get("HERMES_OLLAMA_NATIVE", "").strip().lower() in ("1", "true", "yes", "on"):
    try:
        from hermes_ollama_native import activate

        activate()
    except Exception:  # never break interpreter startup
        import logging

        logging.getLogger("hermes_ollama_native.autoload").warning(
            "hermes-ollama-native autoload failed; agent stays on /v1", exc_info=True
        )
