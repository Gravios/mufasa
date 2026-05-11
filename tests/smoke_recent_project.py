"""Smoke test for patch 121i: persistent recent-project path.

Exercises the helpers in mufasa.ui_qt.recent_project (no
PySide6 dependency, so the functional tests run anywhere).
Uses a temporary file via monkeypatching _RECENT_PROJECT_PATH
so the test never touches the real ~/.config/mufasa/recent.

    PYTHONPATH=. python tests/smoke_recent_project.py
"""
from __future__ import annotations

import ast
import os
import sys
import tempfile
from pathlib import Path


def _check_helpers_exist() -> None:
    """AST-level: helpers and module constant are present in
    the dedicated recent_project module.
    """
    src_path = Path("mufasa/ui_qt/recent_project.py")
    tree = ast.parse(src_path.read_text())
    fn_names = {
        n.name for n in tree.body
        if isinstance(n, ast.FunctionDef)
    }
    for needed in ("save_recent_project", "load_recent_project"):
        assert needed in fn_names, (
            f"recent_project should define {needed!r}; "
            f"got {sorted(fn_names)}"
        )
    # Module constant for path
    module_names = {
        target.id
        for node in tree.body if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert "_RECENT_PROJECT_PATH" in module_names, (
        "recent_project should define _RECENT_PROJECT_PATH "
        "module-level constant"
    )


def _check_main_wiring() -> None:
    """workbench_app.main() must consult load_recent_project as
    a fallback and call save_recent_project after resolving a
    project. Aliased imports (with as_… local names) are fine.
    """
    src = Path("mufasa/ui_qt/workbench_app.py").read_text()
    tree = ast.parse(src)

    # The aliased imports — workbench_app uses _load_/_save_
    # internally but imports them from recent_project.
    imported_alias_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (
            node.module == "mufasa.ui_qt.recent_project"
        ):
            for alias in node.names:
                imported_alias_names.add(alias.asname or alias.name)
    assert "_save_recent_project" in imported_alias_names, (
        "workbench_app should import save_recent_project "
        "(as _save_recent_project) from recent_project"
    )
    assert "_load_recent_project" in imported_alias_names, (
        "workbench_app should import load_recent_project "
        "(as _load_recent_project) from recent_project"
    )

    main_fn = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    main_src = ast.unparse(main_fn)
    assert "_load_recent_project" in main_src, (
        "main() should fall back to load_recent_project when "
        "neither --project nor auto-discover produces a path"
    )
    assert "_save_recent_project" in main_src, (
        "main() should call save_recent_project to persist "
        "the resolved project for next launch"
    )
    assert "no_recent" in main_src, (
        "main() should respect a --no-recent flag"
    )

    # _parse_args must register --no-recent
    parse = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_parse_args"
    )
    parse_src = ast.unparse(parse)
    assert "--no-recent" in parse_src, (
        "_parse_args should register a --no-recent flag"
    )


def _check_workbench_save_on_switch() -> None:
    """MufasaWorkbench._switch_to_project must call
    save_recent_project so File→Open updates the persisted
    pointer.
    """
    src = Path("mufasa/ui_qt/workbench.py").read_text()
    tree = ast.parse(src)
    wb_class = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef)
        and n.name == "MufasaWorkbench"
    )
    sw = next(
        n for n in wb_class.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_switch_to_project"
    )
    sw_src = ast.unparse(sw)
    assert "save_recent_project" in sw_src, (
        "_switch_to_project should call save_recent_project "
        "to update the persisted recent-project pointer"
    )


def _check_save_load_roundtrip() -> None:
    """Functional: save then load returns the same path. Stale
    entries (file moved/deleted) return None.
    """
    import mufasa.ui_qt.recent_project as rp

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        recent_file = td / "recent"
        fake_project = td / "project_folder" / "project_config.ini"
        fake_project.parent.mkdir()
        fake_project.write_text("[General settings]\n")

        orig = rp._RECENT_PROJECT_PATH
        rp._RECENT_PROJECT_PATH = recent_file
        try:
            assert rp.load_recent_project() is None

            rp.save_recent_project(str(fake_project))
            assert recent_file.is_file()
            loaded = rp.load_recent_project()
            assert loaded is not None
            assert loaded.resolve() == fake_project.resolve()

            fake_project.unlink()
            assert rp.load_recent_project() is None

            recent_file.write_text("   \n")
            assert rp.load_recent_project() is None

            fake_project.write_text("[General settings]\n")
            rp.save_recent_project(str(fake_project))
            loaded2 = rp.load_recent_project()
            assert loaded2 is not None
            assert loaded2 == fake_project.resolve()
        finally:
            rp._RECENT_PROJECT_PATH = orig


def _check_save_creates_parent() -> None:
    """save() should create the .config/mufasa/ parent dir on
    first use (mkdir parents=True).
    """
    import mufasa.ui_qt.recent_project as rp
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        deep = td / "a" / "b" / "c" / "recent"
        fake_project = td / "project_config.ini"
        fake_project.write_text("[General settings]\n")
        orig = rp._RECENT_PROJECT_PATH
        rp._RECENT_PROJECT_PATH = deep
        try:
            rp.save_recent_project(str(fake_project))
            assert deep.is_file()
        finally:
            rp._RECENT_PROJECT_PATH = orig


def _check_save_swallows_oserror() -> None:
    """save() with an unwriteable path silently no-ops."""
    import mufasa.ui_qt.recent_project as rp
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        orig = rp._RECENT_PROJECT_PATH
        rp._RECENT_PROJECT_PATH = td  # directory, not file
        try:
            rp.save_recent_project(str(td / "x.ini"))
            assert rp.load_recent_project() is None
        finally:
            rp._RECENT_PROJECT_PATH = orig


def main() -> int:
    _check_helpers_exist()
    _check_main_wiring()
    _check_workbench_save_on_switch()
    _check_save_load_roundtrip()
    _check_save_creates_parent()
    _check_save_swallows_oserror()
    print("smoke_recent_project: 6/6 checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
