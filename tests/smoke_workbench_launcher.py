"""Tests for mufasa.cli.workbench_launcher.

The launcher is a Linux-only thin wrapper that:
1. Forwards args to mufasa.ui_qt.workbench_app:main
2. Sets QT_QPA_PLATFORM if not already set
3. Emits a diagnostic when the import chain fails

We can test (1) and (3) by mocking. (2) is verified by AST
inspection — actually setting it would pollute the test process
env.

    PYTHONPATH=. python tests/smoke_workbench_launcher.py
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from unittest import mock


def main() -> int:
    src_path = "mufasa/cli/workbench_launcher.py"
    src = Path(src_path).read_text()
    tree = ast.parse(src)

    # ------------------------------------------------------------------ #
    # Case 1: file structure — main(), _diagnose_env, _set_qt_platform_default
    # ------------------------------------------------------------------ #
    fn_names = {
        n.name for n in tree.body if isinstance(n, ast.FunctionDef)
    }
    expected = {"main", "_diagnose_env", "_set_qt_platform_default"}
    missing = expected - fn_names
    assert not missing, f"Launcher missing functions: {sorted(missing)}"

    # ------------------------------------------------------------------ #
    # Case 2: main() takes Optional[list[str]] argv (matches workbench_app)
    # ------------------------------------------------------------------ #
    main_fn = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    arg_names = [a.arg for a in main_fn.args.args]
    assert arg_names == ["argv"], (
        f"main() should take a single 'argv' parameter, got {arg_names}"
    )
    # Default of None
    if main_fn.args.defaults:
        d = main_fn.args.defaults[0]
        assert isinstance(d, ast.Constant) and d.value is None, (
            "argv should default to None"
        )

    # ------------------------------------------------------------------ #
    # Case 3: main() forwards return value of workbench_main
    # ------------------------------------------------------------------ #
    main_src = ast.unparse(main_fn)
    assert "workbench_main(argv)" in main_src, (
        "main() should call workbench_main(argv)"
    )
    assert "return workbench_main" in main_src, (
        "main() should return what workbench_main returns "
        "(propagates exit code)"
    )

    # ------------------------------------------------------------------ #
    # Case 4: import is wrapped in try/except so a broken env produces
    # a diagnostic instead of an opaque traceback
    # ------------------------------------------------------------------ #
    has_try = False
    for sub in ast.walk(main_fn):
        if isinstance(sub, ast.Try):
            # Body should contain the workbench_app import
            body_src = "\n".join(ast.unparse(s) for s in sub.body)
            if "workbench_app" in body_src:
                has_try = True
    assert has_try, (
        "workbench_app import should be wrapped in try/except for "
        "diagnostic"
    )

    # ------------------------------------------------------------------ #
    # Case 5: _set_qt_platform_default respects user's existing
    # QT_QPA_PLATFORM
    # ------------------------------------------------------------------ #
    qt_fn = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_set_qt_platform_default"
    )
    qt_src = ast.unparse(qt_fn)
    # Should check for existing QT_QPA_PLATFORM and return early
    assert "QT_QPA_PLATFORM" in qt_src
    assert "os.environ" in qt_src
    # Should NOT use os.environ[...] = ... assignment when var is set
    # (i.e. there should be a guard before any assignment).
    # Just check the flow has an early-return idiom.
    assert "return" in qt_src

    # ------------------------------------------------------------------ #
    # Case 6: diagnostic mentions the practical fix steps
    # ------------------------------------------------------------------ #
    diag_fn = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_diagnose_env"
    )
    diag_src = ast.unparse(diag_fn)
    for required_token in [
        "sys.executable", "sys.prefix",
        "CONDA_PREFIX", "CONDA_DEFAULT_ENV",
        "conda activate", "pip install -e",
    ]:
        assert required_token in diag_src, (
            f"_diagnose_env should mention {required_token!r}"
        )

    # ------------------------------------------------------------------ #
    # Case 7: pyproject.toml's [project.scripts] points `mufasa` at
    # the new launcher, not the old chooser
    # ------------------------------------------------------------------ #
    pyproject = Path("pyproject.toml").read_text()
    # Find the line for `mufasa = ...`
    mufasa_line = None
    for line in pyproject.splitlines():
        stripped = line.strip()
        if stripped.startswith("mufasa") and "=" in stripped:
            # Could be mufasa, mufasa-tk, mufasa-workbench, or mufasa-chooser
            name = stripped.split("=")[0].strip()
            if name == "mufasa":
                mufasa_line = stripped
                break
    assert mufasa_line is not None, (
        "Couldn't find `mufasa = ...` line in [project.scripts]"
    )
    assert "workbench_launcher" in mufasa_line, (
        f"`mufasa` script should point at workbench_launcher, "
        f"got: {mufasa_line!r}"
    )

    # ------------------------------------------------------------------ #
    # Case 8: existing entry points are preserved
    # ------------------------------------------------------------------ #
    for required in ["mufasa-workbench", "mufasa-tk"]:
        assert required in pyproject, (
            f"Existing entry point {required!r} should be preserved"
        )
    # And mufasa-chooser was added as alias for the old `mufasa` behavior
    assert "mufasa-chooser" in pyproject, (
        "mufasa-chooser alias should preserve old `mufasa` (chooser) "
        "behavior for any scripts that depended on it"
    )

    # ------------------------------------------------------------------ #
    # Case 9: actually invoke main() with a stubbed workbench_main and
    # verify the return value is forwarded
    # ------------------------------------------------------------------ #
    # Stub: when main() tries to import workbench_app, give it a fake
    # one whose .main returns a known value.
    fake_module = type(sys)("mufasa.ui_qt.workbench_app")
    fake_module.main = lambda argv=None: 42
    sys.modules["mufasa.ui_qt.workbench_app"] = fake_module
    # And stub linux_env to avoid Qt detection side effects
    fake_linux_env = type(sys)("mufasa.ui_qt.linux_env")
    fake_linux_env.recommended_qpa_platform = lambda: ""
    sys.modules["mufasa.ui_qt.linux_env"] = fake_linux_env
    # We need mufasa.ui_qt to exist as a package shim
    if "mufasa" not in sys.modules:
        sys.modules["mufasa"] = type(sys)("mufasa")
        sys.modules["mufasa"].__path__ = []
    if "mufasa.cli" not in sys.modules:
        sys.modules["mufasa.cli"] = type(sys)("mufasa.cli")
    if "mufasa.ui_qt" not in sys.modules:
        sys.modules["mufasa.ui_qt"] = type(sys)("mufasa.ui_qt")

    # Load the launcher module and run main()
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mufasa.cli.workbench_launcher",
        src_path,
    )
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    rv = launcher.main([])
    assert rv == 42, (
        f"main() should return whatever workbench_main returns, "
        f"got {rv}"
    )

    # ------------------------------------------------------------------ #
    # Case 10: when the import fails, main() returns 2 and prints
    # the diagnostic to stderr
    # ------------------------------------------------------------------ #
    # Reset by removing workbench_app
    sys.modules["mufasa.ui_qt.workbench_app"] = None
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        rv = launcher.main([])
    assert rv == 2, (
        f"main() should return 2 on import failure, got {rv}"
    )
    err = buf.getvalue()
    assert "mufasa launcher" in err
    assert "conda activate" in err

    print("smoke_workbench_launcher: 10/10 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
