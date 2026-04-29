"""
mufasa.cli.workbench_launcher
=============================

Entry point for the ``mufasa`` console script. Delegates to the
workbench UI (``mufasa.ui_qt.workbench_app:main``) while handling
Linux runtime quirks that the bare console script doesn't.

Linux-only by design — the project targets Linux exclusively.

What this layer does
--------------------

A bare ``console_scripts`` entry point works fine when the user
is already inside the active conda env. It can misbehave when:

* The user types ``mufasa`` from a fresh terminal where the
  conda env isn't active. The shim might resolve to a different
  Python or fail with ``command not found``.
* Display server detection picks the wrong Qt platform plugin
  (Wayland sessions where the wayland plugin isn't bundled,
  ThinLinc/Xvnc sessions that need software rendering, etc.)

This launcher:

1. Verifies the import chain works under the current Python.
   If anything is broken, prints a clear diagnostic about the
   conda env and exits non-zero.
2. Sets ``QT_QPA_PLATFORM`` from
   :func:`mufasa.ui_qt.linux_env.recommended_qpa_platform` if
   the user hasn't already set it.
3. Forwards all args to the workbench's ``main()``.

The launcher itself is a Python entry point. ``pip install``
rewrites the shebang to the correct interpreter at install time,
so moving the codebase to a different workstation and reinstalling
adapts the path automatically — no hardcoded paths in shell
scripts.

Usage
-----

After ``pip install -e .``::

    mufasa                    # launches the workbench
    mufasa --project /p.ini   # opens a project on launch
    mufasa --help             # forwards to workbench's argparse

Existing entry points remain available:

* ``mufasa-workbench`` — leaner direct launch (no env diagnostics)
* ``mufasa-tk`` — legacy tkinter UI

Why a separate ``mufasa-workbench`` entry point coexists with
``mufasa``: scripts and CI that want a deterministic minimal
launch path skip the env-diagnostic step. Day-to-day humans use
``mufasa``.
"""
from __future__ import annotations

import os
import sys
from typing import Optional


def _diagnose_env(exc: BaseException) -> str:
    """Build a human-readable diagnostic message for an import
    failure. Tells the user which Python is in use and what conda
    env (if any) appears active, so they can spot mismatches.

    Common scenario this diagnoses: user types ``mufasa`` from a
    fresh terminal where conda hasn't been activated. The shebang
    on the entry-point shim points to the env's Python, which
    works — but if the env was rebuilt or the codebase was
    re-installed, the shebang might be stale. The diagnostic
    surfaces the exact paths so the user can compare.
    """
    py = sys.executable
    prefix = sys.prefix
    conda_prefix = os.environ.get("CONDA_PREFIX", "<not set>")
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "<not set>")
    venv = os.environ.get("VIRTUAL_ENV", "<not set>")

    lines = [
        f"mufasa launcher: failed to import workbench backend.",
        f"  Reason: {type(exc).__name__}: {exc}",
        f"",
        f"Python in use:",
        f"  sys.executable: {py}",
        f"  sys.prefix:     {prefix}",
        f"",
        f"Active environment markers:",
        f"  CONDA_PREFIX:        {conda_prefix}",
        f"  CONDA_DEFAULT_ENV:   {conda_env}",
        f"  VIRTUAL_ENV:         {venv}",
        f"",
        f"If sys.prefix differs from CONDA_PREFIX, the conda env",
        f"isn't actually active for this Python. Try:",
        f"  conda activate <your-mufasa-env>",
        f"  mufasa",
        f"",
        f"If you recently rebuilt the env or moved the repo, reinstall:",
        f"  pip install -e .",
    ]
    return "\n".join(lines)


def _set_qt_platform_default() -> None:
    """Set ``QT_QPA_PLATFORM`` from
    :func:`mufasa.ui_qt.linux_env.recommended_qpa_platform`
    if the user hasn't already chosen one.

    Only sets, never overrides. If the user exports
    ``QT_QPA_PLATFORM=xcb`` to force X11 (e.g. to work around a
    Wayland-specific Qt bug), we respect that.
    """
    if os.environ.get("QT_QPA_PLATFORM"):
        return  # respect user's explicit choice
    try:
        from mufasa.ui_qt import linux_env
        recommended = linux_env.recommended_qpa_platform()
        if recommended:
            os.environ["QT_QPA_PLATFORM"] = recommended
    except Exception:
        # If linux_env can't be imported, the workbench import
        # will fail next and produce a more useful error anyway.
        pass


def main(argv: Optional[list[str]] = None) -> int:
    """Console-script entry. Forwards to ``workbench_app.main``."""
    # Try the import upfront so we can produce a helpful error
    # message instead of a bare ImportError traceback if something
    # is misconfigured.
    try:
        from mufasa.ui_qt.workbench_app import main as workbench_main
    except Exception as exc:
        print(_diagnose_env(exc), file=sys.stderr)
        return 2

    _set_qt_platform_default()

    return workbench_main(argv)


if __name__ == "__main__":
    sys.exit(main())
