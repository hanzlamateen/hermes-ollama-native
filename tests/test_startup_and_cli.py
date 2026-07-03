"""The startup autoload's flag-gate and the CLI surface."""

from __future__ import annotations

from pathlib import Path

import hermes_ollama_native
from hermes_ollama_native.__main__ import main


def _run_autoload(monkeypatch):
    """Exec _autoload.py's source in a fresh namespace (as the .pth import would),
    with hermes_ollama_native.activate patched to record calls."""
    calls = []
    monkeypatch.setattr(hermes_ollama_native, "activate", lambda: calls.append(1))
    src = Path(hermes_ollama_native.__file__).resolve().parent / "_autoload.py"
    code = compile(src.read_text(encoding="utf-8"), str(src), "exec")
    exec(code, {"__name__": "hermes_ollama_native._autoload"})
    return calls


def test_autoload_is_noop_when_flag_off(monkeypatch):
    monkeypatch.delenv("HERMES_OLLAMA_NATIVE", raising=False)
    assert _run_autoload(monkeypatch) == []


def test_autoload_activates_when_flag_on(monkeypatch):
    monkeypatch.setenv("HERMES_OLLAMA_NATIVE", "1")
    assert _run_autoload(monkeypatch) == [1]


def test_cli_status_reports_flag_aux_and_pth(monkeypatch, capsys):
    monkeypatch.delenv("HERMES_OLLAMA_NATIVE", raising=False)
    monkeypatch.delenv("HERMES_OLLAMA_NATIVE_AUX", raising=False)
    rc = main(["status"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "version:" in out
    assert "HERMES_OLLAMA_NATIVE=" in out  # main flag line
    assert "aux flag:" in out  # the newly-surfaced AUX flag
    assert "startup .pth:" in out


def test_cli_no_subcommand_returns_nonzero(capsys):
    rc = main([])
    assert rc == 1  # prints help, non-zero exit
