"""
tests/smoke_122cv_duplicate_rois_qt.py
========================================

Patch 122cv: DuplicateROIsBySourceTarget Qt port + Tk deletion.

Fourth and final of the 4 subprocess-launched popups identified
in 122cr's discovery. The substantial one — 197-line Tk popup
with multi-stage UI, hand-rolled shift+click range-select, and
status-bar feedback. Ported to a Qt dialog using
QListWidget(ExtendedSelection) for native range-select.

After this patch:
* Subprocess-popup count: 1 → 0.
* `_launch_tk_popup` helper in `roi_video_table.py` removed —
  no remaining callers.
* The subprocess-bridge era for ROI dialogs is closed.

Coverage
--------
1.  Tk popup `duplicate_rois_by_source_target_popup.py` is gone.
2.  Qt dialog `ui_qt/dialogs/duplicate_rois_source_target.py`
    exists.
3.  Qt dialog defines `DuplicateRoisDialog` class.
4.  Qt dialog subclasses `QDialog`.
5.  Qt dialog has `init_failed()` helper.
6.  Qt dialog uses `QListWidget` (not custom checkbox grid —
    the Qt-idiomatic upgrade over Tk's bespoke shift+click
    bookkeeping).
7.  Qt dialog sets selection mode to `ExtendedSelection`
    (native shift+click range-select + ctrl+click toggle).
8.  Qt dialog imports the 3 ROI backend helpers
    (`get_roi_data_for_video_name`, `change_roi_dict_video_name`,
    `get_roi_df_from_dict`).
9.  `roi_video_table.py:_action_duplicate` no longer contains
    the string-literal subprocess launch.
10. `roi_video_table.py:_action_duplicate` calls
    `DuplicateRoisDialog`.
11. `roi_video_table.py:_action_duplicate` emits `rois_modified`
    on accept.
12. `_launch_tk_popup` method is no longer defined as an active
    `FunctionDef` in `roi_video_table.py` (comment references
    are fine).
13. No file imports from the deleted Tk popup module.
14. `tk_surface_audit.md` §2g records DELETED 122cv +
    states the subprocess-bridge era is closed.
15. mufasa/ui/pop_ups/ count ≤ 75 (was 76 post-122cu).
16. All mufasa/**/*.py files parse cleanly.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


CHECKS_PASSED = 0
CHECKS_TOTAL = 0


def check(label: str, cond: bool, *, detail: str = "") -> None:
    global CHECKS_PASSED, CHECKS_TOTAL
    CHECKS_TOTAL += 1
    if cond:
        CHECKS_PASSED += 1
    else:
        print(f"FAIL: {label}" + (f" — {detail}" if detail else ""))


def main() -> int:
    pkg = REPO_ROOT / "mufasa"

    # 1. Tk popup gone
    check(
        "Tk popup duplicate_rois_by_source_target_popup.py is gone",
        not (pkg / "ui" / "pop_ups"
             / "duplicate_rois_by_source_target_popup.py").exists(),
    )

    # 2-8. Qt dialog shape
    qt = pkg / "ui_qt" / "dialogs" / "duplicate_rois_source_target.py"
    check(
        "Qt dialog ui_qt/dialogs/duplicate_rois_source_target.py exists",
        qt.exists(),
    )

    qt_src = qt.read_text() if qt.exists() else ""
    qt_tree = ast.parse(qt_src) if qt_src else None

    if qt_tree is not None:
        dialog_cls = None
        for n in qt_tree.body:
            if (isinstance(n, ast.ClassDef)
                    and n.name == "DuplicateRoisDialog"):
                dialog_cls = n
                break
        check("DuplicateRoisDialog class defined",
              dialog_cls is not None)

        if dialog_cls is not None:
            base_names = [ast.unparse(b) for b in dialog_cls.bases]
            check(
                "DuplicateRoisDialog subclasses QDialog",
                "QDialog" in base_names,
            )
            method_names = {
                m.name for m in dialog_cls.body
                if isinstance(m, ast.FunctionDef)
            }
            check(
                "Qt dialog has init_failed() helper",
                "init_failed" in method_names,
            )

        check(
            "Qt dialog uses QListWidget (replaces Tk's per-video "
            "checkbox grid + bespoke shift-click bookkeeping)",
            "QListWidget" in qt_src,
        )
        check(
            "Qt dialog sets ExtendedSelection mode (native "
            "shift+click range + ctrl+click toggle)",
            "ExtendedSelection" in qt_src,
        )

        # Check the 3 backend imports
        backend_imports_found = set()
        target_imports = {"get_roi_data_for_video_name",
                          "change_roi_dict_video_name",
                          "get_roi_df_from_dict"}
        for node in ast.walk(qt_tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == "mufasa.roi_tools.roi_utils":
                    for alias in node.names:
                        if alias.name in target_imports:
                            backend_imports_found.add(alias.name)
        check(
            f"Qt dialog imports the 3 ROI backend helpers "
            f"(got {len(backend_imports_found)}/3: "
            f"{sorted(backend_imports_found)})",
            backend_imports_found == target_imports,
        )

    # 9-12. roi_video_table.py wiring
    rvt = (pkg / "ui_qt" / "dialogs"
           / "roi_video_table.py").read_text()
    check(
        "_action_duplicate no longer contains the subprocess "
        "Tk-launch string",
        "from mufasa.ui.pop_ups.duplicate_rois_by_source_target_popup"
        not in rvt,
    )
    check(
        "_action_duplicate calls DuplicateRoisDialog",
        "DuplicateRoisDialog" in rvt,
    )
    rvt_tree = ast.parse(rvt)
    emit_in_action = False
    for n in ast.walk(rvt_tree):
        if (isinstance(n, ast.FunctionDef)
                and n.name == "_action_duplicate"):
            body_src = ast.unparse(n)
            if "rois_modified.emit" in body_src:
                emit_in_action = True
                break
    check(
        "_action_duplicate emits rois_modified on accept",
        emit_in_action,
    )

    # The _launch_tk_popup method should no longer be defined as
    # an active FunctionDef. (Comment references are fine.)
    launch_helper_defined = any(
        isinstance(n, ast.FunctionDef) and n.name == "_launch_tk_popup"
        for n in ast.walk(rvt_tree)
    )
    check(
        "_launch_tk_popup method is no longer defined "
        "(comment references in the file are fine; this checks "
        "active method existence only)",
        not launch_helper_defined,
    )

    # 13. No leftover importers
    leftover = []
    for f in pkg.rglob("*.py"):
        try:
            t = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(t):
            if (isinstance(node, ast.ImportFrom)
                and node.module ==
                "mufasa.ui.pop_ups.duplicate_rois_by_source_target_popup"):
                leftover.append(f"{f.relative_to(pkg)}:{node.lineno}")
    check(
        "no file imports from the deleted Tk popup module",
        leftover == [],
        detail=", ".join(leftover),
    )

    # 14. Doc updates + subprocess-bridge era closure
    audit = (REPO_ROOT / "docs" / "tk_surface_audit.md").read_text()
    check(
        "tk_surface_audit.md §2g records DELETED 122cv",
        "DELETED 122cv" in audit
        and "duplicate_rois_source_target" in audit,
    )
    check(
        "tk_surface_audit.md §2g notes subprocess-bridge era closed",
        "subprocess-bridge era" in audit
        and "closed" in audit,
    )

    # 15. Count
    popups_count = sum(
        1 for _ in (pkg / "ui" / "pop_ups").glob("*.py")
        if _.name != "__init__.py"
    )
    check(
        f"mufasa/ui/pop_ups/ count ≤ 75 (was 76 post-122cu; "
        f"got {popups_count})",
        popups_count <= 75,
    )

    # 16. Parse-clean
    parse_errors = []
    for f in sorted(pkg.rglob("*.py")):
        try:
            ast.parse(f.read_text())
        except SyntaxError as e:
            parse_errors.append(f"{f}: {e}")
    check(
        f"All mufasa/**/*.py files parse cleanly "
        f"({sum(1 for _ in pkg.rglob('*.py'))} files)",
        parse_errors == [],
        detail=(parse_errors[0] if parse_errors else ""),
    )

    print(
        f"smoke_122cv_duplicate_rois_qt: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
