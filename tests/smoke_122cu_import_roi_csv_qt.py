"""
tests/smoke_122cu_import_roi_csv_qt.py
========================================

Patch 122cu: ROIDefinitionsCSVImporterPopUp Qt port + Tk
deletion.

Third of the 4 subprocess-launched popups (smaller of the 3
remaining after 122cs/122ct) ported. Subprocess-popup count:
2 → 1.

Coverage
--------
1.  Tk popup `import_roi_csv_popup.py` is gone.
2.  Qt dialog `ui_qt/dialogs/import_roi_csv.py` exists.
3.  Qt dialog defines `ImportRoiCsvDialog` class.
4.  Qt dialog subclasses `QDialog`.
5.  Qt dialog has `init_failed()` helper.
6.  Qt dialog imports the backend `ROIDefinitionsCSVImporter`.
7.  Qt dialog uses `QFileDialog.getOpenFileName` (the Qt-native
    file-picker replacement for `FileSelect`).
8.  `roi_video_table.py:_action_import_csv` no longer contains
    the string-literal subprocess launch.
9.  `roi_video_table.py:_action_import_csv` calls
    `ImportRoiCsvDialog`.
10. `roi_video_table.py:_action_import_csv` emits
    `rois_modified` on accept (the import writes new ROIs;
    the parent table needs to refresh).
11. No file imports from the deleted Tk popup module.
12. `tk_surface_audit.md` §2g records DELETED 122cu (count: 2 → 1).
13. mufasa/ui/pop_ups/ count dropped by 1.
14. All mufasa/**/*.py files parse cleanly.
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
        "Tk popup import_roi_csv_popup.py is gone",
        not (pkg / "ui" / "pop_ups"
             / "import_roi_csv_popup.py").exists(),
    )

    # 2-7. Qt dialog shape
    qt = pkg / "ui_qt" / "dialogs" / "import_roi_csv.py"
    check("Qt dialog ui_qt/dialogs/import_roi_csv.py exists",
          qt.exists())

    qt_src = qt.read_text() if qt.exists() else ""
    qt_tree = ast.parse(qt_src) if qt_src else None

    if qt_tree is not None:
        dialog_cls = None
        for n in qt_tree.body:
            if (isinstance(n, ast.ClassDef)
                    and n.name == "ImportRoiCsvDialog"):
                dialog_cls = n
                break
        check("ImportRoiCsvDialog class defined",
              dialog_cls is not None)

        if dialog_cls is not None:
            base_names = [ast.unparse(b) for b in dialog_cls.bases]
            check(
                "ImportRoiCsvDialog subclasses QDialog",
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

        backend_imported = any(
            isinstance(n, ast.ImportFrom)
            and n.module == "mufasa.roi_tools.import_roi_csvs"
            and any(a.name == "ROIDefinitionsCSVImporter"
                    for a in n.names)
            for n in ast.walk(qt_tree)
        )
        check(
            "Qt dialog imports backend ROIDefinitionsCSVImporter",
            backend_imported,
        )

        check(
            "Qt dialog uses QFileDialog.getOpenFileName "
            "(Qt-native file picker; replaces Tk's FileSelect)",
            "getOpenFileName" in qt_src,
        )

    # 8-10. Wiring
    rvt = (pkg / "ui_qt" / "dialogs"
           / "roi_video_table.py").read_text()
    check(
        "_action_import_csv no longer contains the subprocess "
        "Tk-launch string",
        "from mufasa.ui.pop_ups.import_roi_csv_popup" not in rvt,
    )
    check(
        "_action_import_csv calls ImportRoiCsvDialog",
        "ImportRoiCsvDialog" in rvt,
    )
    # Confirm the emit happens specifically inside this action by
    # locating its body and checking for the signal.
    rvt_tree = ast.parse(rvt)
    emit_in_action = False
    for n in ast.walk(rvt_tree):
        if (isinstance(n, ast.FunctionDef)
                and n.name == "_action_import_csv"):
            body_src = ast.unparse(n)
            if "rois_modified.emit" in body_src:
                emit_in_action = True
                break
    check(
        "_action_import_csv emits rois_modified on accept "
        "(parent table needs to refresh)",
        emit_in_action,
    )

    # 11. No leftover importers
    leftover = []
    for f in pkg.rglob("*.py"):
        try:
            t = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(t):
            if (isinstance(node, ast.ImportFrom)
                    and node.module ==
                    "mufasa.ui.pop_ups.import_roi_csv_popup"):
                leftover.append(f"{f.relative_to(pkg)}:{node.lineno}")
    check(
        "no file imports from the deleted Tk popup module",
        leftover == [],
        detail=", ".join(leftover),
    )

    # 12. Doc updates
    audit = (REPO_ROOT / "docs" / "tk_surface_audit.md").read_text()
    check(
        "tk_surface_audit.md §2g records DELETED 122cu "
        "(count: 2 → 1)",
        "DELETED 122cu" in audit
        and "import_roi_csv" in audit,
    )

    # 13. Count
    popups_count = sum(
        1 for _ in (pkg / "ui" / "pop_ups").glob("*.py")
        if _.name != "__init__.py"
    )
    check(
        f"mufasa/ui/pop_ups/ count ≤ 76 (was 77 post-122ct; "
        f"got {popups_count})",
        popups_count <= 76,
    )

    # 14. Parse-clean
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
        f"smoke_122cu_import_roi_csv_qt: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
