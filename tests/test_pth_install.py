"""Startup-hook (.pth) install/uninstall behavior."""

from __future__ import annotations

from pathlib import Path

import hermes_ollama_native
from hermes_ollama_native import _pth_install


def test_pth_path_is_beside_the_installed_package():
    # Must land in the site-packages dir that actually holds the package (correct for
    # venv/--user/system), not sysconfig's default-scheme purelib.
    pkg_dir = Path(hermes_ollama_native.__file__).resolve().parent
    assert _pth_install.pth_path() == pkg_dir.parent / _pth_install.PTH_NAME


def test_pth_line_is_import_prefixed_and_self_healing():
    # site only executes .pth lines that start with "import"; and the line must
    # no-op (not raise) if the package was uninstalled → guarded by find_spec.
    assert _pth_install.PTH_LINE.startswith("import ")
    assert "find_spec('hermes_ollama_native')" in _pth_install.PTH_LINE
    assert _pth_install.PTH_LINE.endswith("\n")


def test_install_uninstall_roundtrip(tmp_path, monkeypatch):
    target = tmp_path / _pth_install.PTH_NAME
    monkeypatch.setattr(_pth_install, "pth_path", lambda: target)

    assert _pth_install.is_installed() is False
    written = _pth_install.install_pth()
    assert written == target
    assert target.read_text(encoding="utf-8") == _pth_install.PTH_LINE
    assert _pth_install.is_installed() is True

    assert _pth_install.uninstall_pth() is True
    assert target.exists() is False
    # Race-safe: removing an already-absent hook returns False, no exception.
    assert _pth_install.uninstall_pth() is False


def test_pth_line_execs_without_error(monkeypatch):
    # Simulate what site.addpackage does with the .pth line: exec it. With the
    # package present and the flag off it imports _autoload (a no-op) and must not
    # raise. This is the line that runs on every interpreter start, so a syntax or
    # import error here would spam every process.
    monkeypatch.delenv("HERMES_OLLAMA_NATIVE", raising=False)
    exec(compile(_pth_install.PTH_LINE, "<pth>", "exec"), {})
