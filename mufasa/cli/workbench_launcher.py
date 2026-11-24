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
    mufasa help               # the commands (mufasa.cli.commands); also --help
    mufasa list models        # and videos, pose, 3d, runs, calibrations, steps, tools, projects
    mufasa info FILE          # what a file holds
    mufasa project            # the current project's summary
    mufasa doctor             # the environment
    mufasa run <step> ...     # the 3-D pipeline from the command line (mufasa.cli.run)

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
        "mufasa launcher: failed to import workbench backend.",
        f"  Reason: {type(exc).__name__}: {exc}",
        "",
        "Python in use:",
        f"  sys.executable: {py}",
        f"  sys.prefix:     {prefix}",
        "",
        "Active environment markers:",
        f"  CONDA_PREFIX:        {conda_prefix}",
        f"  CONDA_DEFAULT_ENV:   {conda_env}",
        f"  VIRTUAL_ENV:         {venv}",
        "",
        "If sys.prefix differs from CONDA_PREFIX, the conda env",
        "isn't actually active for this Python. Try:",
        "  conda activate <your-mufasa-env>",
        "  mufasa",
        "",
        "If you recently rebuilt the env or moved the repo, reinstall:",
        "  pip install -e .",
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


# 122lv: the commands that open no window (mufasa.cli.commands.COMMANDS; kept here so the launcher
# imports nothing more until one is asked for)
COMMANDS = ("help", "list", "info", "project", "run", "doctor", "version")


def main(argv: list[str] | None = None) -> int:
    """Console-script entry. Forwards to ``workbench_app.main``; ``mufasa run ...`` runs pipeline
    steps from the command line instead (:mod:`mufasa.cli.run`, no Qt needed)."""
    args = list(sys.argv[1:] if argv is None else argv)
    # 122lv: `mufasa help`, `mufasa list …`, `info`, `project`, `doctor`, `version`, `run …`
    # (mufasa.cli.commands); `-h` / `--help` give the overview, the workbench's options in it
    if args and (args[0] in COMMANDS or args[0] in ("-h", "--help", "-V", "--version")):
        from mufasa.cli.commands import main as commands_main
        return commands_main(args)
    if args and not args[0].startswith("-"):
        print(f"mufasa: no command '{args[0]}' (`mufasa help` lists them)", file=sys.stderr)
        return 2
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
