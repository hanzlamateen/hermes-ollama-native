"""Install/remove the optional startup ``.pth`` that auto-activates the native-Ollama
patch for every interpreter in the current environment.

The ``.pth`` is a single line — ``import hermes_ollama_native._autoload`` — written
into the active interpreter's site-packages (``purelib``). Python's ``site`` module
runs ``import`` lines in ``.pth`` files at startup, so the patch self-installs there
(still a no-op unless ``HERMES_OLLAMA_NATIVE`` is set).

This is strictly opt-in: nothing writes the ``.pth`` unless the user runs
``hermes-ollama-native install-pth``. We deliberately do **not** ship the ``.pth``
inside the wheel, because auto-installing a startup hook on ``pip install`` is both
fragile across install modes and surprising to users.
"""

from __future__ import annotations

import sysconfig
from pathlib import Path

PTH_NAME = "hermes_ollama_native.pth"
PTH_LINE = "import hermes_ollama_native._autoload\n"


def pth_path() -> Path:
    """Path the ``.pth`` would occupy in the current environment's site-packages."""
    return Path(sysconfig.get_paths()["purelib"]) / PTH_NAME


def install_pth() -> Path:
    """Write the startup ``.pth`` into site-packages and return its path."""
    path = pth_path()
    path.write_text(PTH_LINE, encoding="utf-8")
    return path


def uninstall_pth() -> bool:
    """Remove the startup ``.pth`` if present. Returns True if a file was removed."""
    path = pth_path()
    if path.exists():
        path.unlink()
        return True
    return False


def is_installed() -> bool:
    """True when the startup ``.pth`` exists in the current environment."""
    return pth_path().exists()
