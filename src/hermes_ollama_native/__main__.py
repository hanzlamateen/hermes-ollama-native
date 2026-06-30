"""Command-line interface for hermes-ollama-native.

Usage::

    hermes-ollama-native status         # show flag + startup-hook state
    hermes-ollama-native install-pth     # persist auto-activation in this environment
    hermes-ollama-native uninstall-pth   # remove the startup hook

Equivalent to ``python -m hermes_ollama_native``.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from . import __version__
from ._pth_install import PTH_NAME, install_pth, is_installed, pth_path, uninstall_pth

_TRUTHY = ("1", "true", "yes", "on")


def _flag_state() -> str:
    raw = os.environ.get("HERMES_OLLAMA_NATIVE", "")
    on = raw.strip().lower() in _TRUTHY
    return f"{'on' if on else 'off'} (HERMES_OLLAMA_NATIVE={raw!r})"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hermes-ollama-native",
        description=(
            "Route a Hermes Agent's local Ollama traffic through the native "
            "/api/chat API (honors num_ctx + correct tool-call streaming)."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"hermes-ollama-native {__version__}"
    )
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser(
        "install-pth",
        help="persist auto-activation by writing a startup .pth into site-packages",
    )
    sub.add_parser("uninstall-pth", help="remove the startup .pth")
    sub.add_parser("status", help="show flag + startup-hook state")

    args = parser.parse_args(argv)

    if args.cmd == "install-pth":
        try:
            path = install_pth()
        except OSError as exc:
            print(
                f"error: could not write the startup hook to {pth_path()}: {exc}\n"
                "Use a virtualenv you own, or activate explicitly with "
                "`hermes_ollama_native.activate()` instead.",
                file=sys.stderr,
            )
            return 1
        print(f"Installed startup hook: {path}")
        print(
            "It activates on every interpreter start in this environment "
            "(a no-op unless HERMES_OLLAMA_NATIVE is set)."
        )
        return 0
    if args.cmd == "uninstall-pth":
        try:
            removed = uninstall_pth()
        except OSError as exc:
            print(f"error: could not remove {pth_path()}: {exc}", file=sys.stderr)
            return 1
        print("Removed startup hook." if removed else f"No startup hook found ({PTH_NAME}).")
        return 0
    if args.cmd == "status":
        print(f"version:      {__version__}")
        print(f"flag:         {_flag_state()}")
        print(f"startup .pth: {'installed' if is_installed() else 'not installed'} ({pth_path()})")
        print(
            "note: native routing also requires the target base_url to be a live "
            "Ollama endpoint (probed via GET /api/version per request)."
        )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
