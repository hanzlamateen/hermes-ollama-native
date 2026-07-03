"""Install/remove the optional startup ``.pth`` that auto-activates the native-Ollama
patch for every interpreter in the current environment.

The ``.pth`` holds a single ``import``-prefixed line that Python's ``site`` module
executes at interpreter startup. It is written into the site-packages directory that
actually contains this package, so it works for venv, ``--user``, and system installs
alike (still a no-op unless ``HERMES_OLLAMA_NATIVE`` is set).

The line is defensive: it checks the package still exists before importing the
autoload module, so a leftover ``.pth`` after ``pip uninstall`` no-ops silently
instead of printing a ``site`` traceback on every startup.

This is strictly opt-in: nothing writes the ``.pth`` unless the user runs
``hermes-ollama-native install-pth``. We deliberately do **not** ship the ``.pth``
inside the wheel, because auto-installing a startup hook on ``pip install`` is both
fragile across install modes and surprising to users.
"""

from __future__ import annotations

from pathlib import Path

PTH_NAME = "hermes_ollama_native.pth"
# Self-healing single line: only import the autoload module if the package is still
# importable, so an orphaned .pth (after `pip uninstall`) can't spam a ModuleNotFound
# traceback from `site` on every interpreter start.
PTH_LINE = (
    "import importlib.util as _u; "
    "_u.find_spec('hermes_ollama_native') and "
    "__import__('hermes_ollama_native._autoload')\n"
)


def pth_path() -> Path:
    """Path the ``.pth`` occupies — the site-packages dir that contains this package.

    Derived from this module's own location (``<site>/hermes_ollama_native/_pth_install.py``)
    rather than ``sysconfig``'s default-scheme ``purelib``, which is wrong for
    ``--user`` installs and root-only for a system Python.
    """
    return Path(__file__).resolve().parent.parent / PTH_NAME


def install_pth() -> Path:
    """Write the startup ``.pth`` into site-packages and return its path."""
    path = pth_path()
    path.write_text(PTH_LINE, encoding="utf-8")
    return path


def uninstall_pth() -> bool:
    """Remove the startup ``.pth`` if present. Returns True if a file was removed.

    Race-safe: uses ``unlink()`` and treats a concurrently-removed file as absent
    rather than raising.
    """
    try:
        pth_path().unlink()
        return True
    except FileNotFoundError:
        return False


def is_installed() -> bool:
    """True when the startup ``.pth`` exists in the current environment."""
    return pth_path().exists()
