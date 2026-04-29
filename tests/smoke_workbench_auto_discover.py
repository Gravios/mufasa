"""Tests for project_config.ini auto-discovery in workbench_app.

Mufasa projects are laid out as:
    <project_path>/
    └── <project_name>/
        └── project_folder/        ← DirNames.PROJECT.value
            ├── project_config.ini ← THE FILE
            ├── csv/
            ├── videos/
            └── ...

So discovery has to handle both "user is in project_folder/ or
deeper" and "user is in <project_name>/" (one level above).

Sandbox-runnable; the helper is pure-stdlib so we extract and
exec it without importing PySide6.

    PYTHONPATH=. python tests/smoke_workbench_auto_discover.py
"""
from __future__ import annotations

import ast
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional


def _load_helper():
    """Extract _auto_discover_project + supporting constants from
    workbench_app.py and exec into a namespace. We can't import the
    module directly because PySide6 isn't available in the sandbox.
    """
    src = Path("mufasa/ui_qt/workbench_app.py").read_text()
    tree = ast.parse(src)
    wanted = {
        "_AUTO_DISCOVER_MAX_DEPTH",
        "_PROJECT_CONFIG_FILENAME",
        "_PROJECT_FOLDER_NAME",
        "_auto_discover_project",
    }
    pieces = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    pieces.append(ast.unparse(node))
        elif isinstance(node, ast.FunctionDef) and node.name in wanted:
            pieces.append(ast.unparse(node))
    assert len(pieces) >= 4, (
        f"Expected to find 4 names ({wanted}); found {len(pieces)}"
    )
    ns = {"Path": Path, "Optional": Optional}
    exec("\n".join(pieces), ns)
    return ns


def main() -> int:
    ns = _load_helper()
    auto_discover = ns["_auto_discover_project"]
    max_depth: int = ns["_AUTO_DISCOVER_MAX_DEPTH"]
    config_name: str = ns["_PROJECT_CONFIG_FILENAME"]
    folder_name: str = ns["_PROJECT_FOLDER_NAME"]
    assert config_name == "project_config.ini"
    assert folder_name == "project_folder"

    # ------------------------------------------------------------------ #
    # Case 1: config in CWD (user cd'd into project_folder/)
    # ------------------------------------------------------------------ #
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td).resolve()
        config = td_path / config_name
        config.write_text("[General]\nproject_name = test\n")
        result = auto_discover(td_path)
        assert result is not None and result.resolve() == config.resolve(), (
            f"Should find config in CWD; got {result}"
        )

    # ------------------------------------------------------------------ #
    # Case 2: user one level above project_folder/, config inside
    # (this is the case where user cd'd into <project_name>/)
    # ------------------------------------------------------------------ #
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td).resolve()
        pf = td_path / folder_name
        pf.mkdir()
        config = pf / config_name
        config.write_text("[General]\n")
        result = auto_discover(td_path)
        assert result is not None and result.resolve() == config.resolve(), (
            f"Should find config in project_folder/ subdir; got {result}"
        )

    # ------------------------------------------------------------------ #
    # Case 3: user inside project_folder/csv/ — config one up
    # ------------------------------------------------------------------ #
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td).resolve()
        pf = td_path / folder_name
        pf.mkdir()
        config = pf / config_name
        config.write_text("[General]\n")
        csv_sub = pf / "csv"
        csv_sub.mkdir()
        result = auto_discover(csv_sub)
        assert result is not None and result.resolve() == config.resolve(), (
            f"Should find config one level up; got {result}"
        )

    # ------------------------------------------------------------------ #
    # Case 4: user 2 levels deep inside project_folder/
    # (e.g. project_folder/csv/features_extracted/)
    # ------------------------------------------------------------------ #
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td).resolve()
        pf = td_path / folder_name
        pf.mkdir()
        config = pf / config_name
        config.write_text("[General]\n")
        deep = pf / "csv" / "features_extracted"
        deep.mkdir(parents=True)
        result = auto_discover(deep)
        assert result is not None and result.resolve() == config.resolve()

    # ------------------------------------------------------------------ #
    # Case 5: discovery exactly at max_depth (boundary)
    # ------------------------------------------------------------------ #
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td).resolve()
        config = td_path / config_name
        config.write_text("[General]\n")
        deep = td_path
        for i in range(max_depth):
            deep = deep / f"level_{i}"
        deep.mkdir(parents=True)
        result = auto_discover(deep)
        assert result is not None and result.resolve() == config.resolve(), (
            f"Should find config at exactly max_depth; got {result}"
        )

    # ------------------------------------------------------------------ #
    # Case 6: does NOT discover beyond max_depth (avoids picking up
    # unrelated projects from sibling trees)
    # ------------------------------------------------------------------ #
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td).resolve()
        config = td_path / config_name
        config.write_text("[General]\n")
        # One level deeper than max_depth
        deep = td_path
        for i in range(max_depth + 1):
            deep = deep / f"level_{i}"
        deep.mkdir(parents=True)
        result = auto_discover(deep)
        assert result is None, (
            f"Should NOT find config beyond max_depth; got {result}"
        )

    # ------------------------------------------------------------------ #
    # Case 7: returns None when no config anywhere
    # ------------------------------------------------------------------ #
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td).resolve()
        sub = td_path / "some" / "deep" / "directory"
        sub.mkdir(parents=True)
        result = auto_discover(sub)
        assert result is None, (
            f"Should return None when no config anywhere; got {result}"
        )

    # ------------------------------------------------------------------ #
    # Case 8: closest match wins when multiple configs exist (an
    # outer ancestor has one AND a closer one has another). The
    # walk-up returns at the first hit, which is the closest.
    # ------------------------------------------------------------------ #
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td).resolve()
        outer_config = td_path / config_name
        outer_config.write_text("[outer]\n")
        inner_dir = td_path / "inner"
        inner_dir.mkdir()
        inner_config = inner_dir / config_name
        inner_config.write_text("[inner]\n")
        # User is in inner_dir → should pick inner config
        result = auto_discover(inner_dir)
        assert result is not None and result.resolve() == inner_config.resolve(), (
            f"Closest match should win; got {result}"
        )

    # ------------------------------------------------------------------ #
    # Case 9: filesystem-root behavior — doesn't crash when called
    # near root
    # ------------------------------------------------------------------ #
    # Walk up from /tmp itself; root has no config (presumably).
    # This just verifies the function terminates and returns
    # without raising.
    result = auto_discover(Path("/tmp"))
    assert result is None or isinstance(result, Path), (
        "Walk-up to root should terminate cleanly"
    )

    # ------------------------------------------------------------------ #
    # Case 10: structural — main() in workbench_app routes through
    # _auto_discover_project when --project is not given AND
    # --no-auto-discover is not set
    # ------------------------------------------------------------------ #
    src = Path("mufasa/ui_qt/workbench_app.py").read_text()
    assert "no_auto_discover" in src, (
        "main() should support a --no-auto-discover flag for users "
        "who want to disable the magic"
    )
    assert "_auto_discover_project" in src
    # The dispatch logic: if args.project else if not args.no_auto_discover
    assert "args.no_auto_discover" in src, (
        "main() should check args.no_auto_discover before invoking "
        "the helper"
    )
    # User-visible feedback: print to stderr so the load isn't silent
    assert "Auto-loading project config" in src, (
        "main() should announce the auto-load so users aren't "
        "surprised which project opened"
    )

    print("smoke_workbench_auto_discover: 10/10 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
