"""
tests/smoke_122cs_roi_size_standardizer_qt.py
==============================================

Patch 122cs: ROISizeStandardizerPopUp Qt port + Tk deletion.

The smallest of the 4 subprocess-launched popups identified in
122cr's discovery (44 lines of Tk code) ported to a Qt-native
QDialog. The subprocess-launch bridge in
`ui_qt/dialogs/roi_video_table.py:_action_standardize` is
replaced with a direct dialog invocation; the Tk popup file
is deleted.

After this patch the subprocess-popup count drops from 4 to 3.
The remaining 3 (`duplicate_rois_by_source_target_popup`,
`import_roi_csv_popup`, `min_max_draw_size_popup`) follow the
same port pattern in future patches.

Coverage
--------
1.  Tk popup `roi_size_standardizer_popup.py` is gone.
2.  Qt dialog `ui_qt/dialogs/roi_size_standardizer.py` exists.
3.  Qt dialog defines `ROISizeStandardizerDialog` class.
4.  Qt dialog subclasses `QDialog`.
5.  Qt dialog imports the backend `ROISizeStandardizer`.
6.  Qt dialog has an `init_failed()` helper (caller-friendly
    no-show pattern for missing-data scenarios).
7.  `roi_video_table.py:_action_standardize` no longer contains
    the string-literal subprocess launch.
8.  `roi_video_table.py:_action_standardize` calls
    `ROISizeStandardizerDialog`.
9.  No file imports from the deleted Tk popup module.
10. `tk_surface_audit.md` §2g records the deletion (subprocess
    popup count: 4 → 3).
11. mufasa/ui/pop_ups/ file count dropped by 1.
12. All mufasa/**/*.py files parse cleanly.
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

    # ==================================================================
    # 1. Tk popup deleted
    # ==================================================================
    tk_popup = pkg / "ui" / "pop_ups" / "roi_size_standardizer_popup.py"
    check("Tk popup roi_size_standardizer_popup.py is gone",
          not tk_popup.exists())

    # ==================================================================
    # 2-6. Qt dialog exists and has the right shape
    # ==================================================================
    qt_dialog = pkg / "ui_qt" / "dialogs" / "roi_size_standardizer.py"
    check("Qt dialog ui_qt/dialogs/roi_size_standardizer.py exists",
          qt_dialog.exists())

    qt_dialog_src = qt_dialog.read_text() if qt_dialog.exists() else ""
    qt_dialog_tree = ast.parse(qt_dialog_src) if qt_dialog_src else None

    if qt_dialog_tree is not None:
        # Find ROISizeStandardizerDialog
        dialog_cls = None
        for n in qt_dialog_tree.body:
            if (isinstance(n, ast.ClassDef)
                    and n.name == "ROISizeStandardizerDialog"):
                dialog_cls = n
                break
        check("ROISizeStandardizerDialog class defined",
              dialog_cls is not None)

        if dialog_cls is not None:
            base_names = [ast.unparse(b) for b in dialog_cls.bases]
            check(
                "ROISizeStandardizerDialog subclasses QDialog",
                "QDialog" in base_names,
            )

            method_names = {
                m.name for m in dialog_cls.body
                if isinstance(m, ast.FunctionDef)
            }
            check(
                "Qt dialog has init_failed() helper (for no-show on "
                "missing-data init failure)",
                "init_failed" in method_names,
            )

        # Backend import — required
        backend_imported = False
        for node in ast.walk(qt_dialog_tree):
            if isinstance(node, ast.ImportFrom):
                if (node.module == "mufasa.roi_tools.ROI_size_standardizer"
                        and any(a.name == "ROISizeStandardizer"
                                for a in node.names)):
                    backend_imported = True
        check(
            "Qt dialog imports backend ROISizeStandardizer",
            backend_imported,
        )

    # ==================================================================
    # 7-8. roi_video_table.py wiring
    # ==================================================================
    rvt_src = (pkg / "ui_qt" / "dialogs" /
               "roi_video_table.py").read_text()
    # The old subprocess pattern strings should be gone
    check(
        "_action_standardize no longer contains the subprocess "
        "Tk-launch string",
        "from mufasa.ui.pop_ups.roi_size_standardizer_popup" not in rvt_src
        or rvt_src.count(
            "from mufasa.ui.pop_ups.roi_size_standardizer_popup") == 0,
    )
    check(
        "_action_standardize calls ROISizeStandardizerDialog",
        "ROISizeStandardizerDialog" in rvt_src,
    )

    # ==================================================================
    # 9. No file imports from the deleted module
    # ==================================================================
    leftover = []
    for f in pkg.rglob("*.py"):
        try:
            t = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(t):
            if (isinstance(node, ast.ImportFrom)
                    and node.module ==
                    "mufasa.ui.pop_ups.roi_size_standardizer_popup"):
                leftover.append(
                    f"{f.relative_to(pkg)}:{node.lineno}")
    check(
        "no file imports from the deleted Tk popup module",
        leftover == [],
        detail=", ".join(leftover),
    )

    # ==================================================================
    # 10. tk_surface_audit.md updated
    # ==================================================================
    audit = (REPO_ROOT / "docs" / "tk_surface_audit.md").read_text()
    check(
        "tk_surface_audit.md §2g records the deletion "
        "(subprocess popup count: 4 → 3)",
        "DELETED 122cs" in audit
        and "roi_size_standardizer" in audit,
    )

    # ==================================================================
    # 11. pop_ups file count dropped
    # ==================================================================
    popups_count = sum(
        1 for _ in (pkg / "ui" / "pop_ups").glob("*.py")
        if _.name != "__init__.py"
    )
    # Was 79 post-122cr; 122cs deleted 1 more → 78
    check(
        f"mufasa/ui/pop_ups/ count ≤ 78 (was 79 post-122cr; "
        f"got {popups_count})",
        popups_count <= 78,
    )

    # ==================================================================
    # 12. All files parse cleanly
    # ==================================================================
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
        f"smoke_122cs_roi_size_standardizer_qt: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
