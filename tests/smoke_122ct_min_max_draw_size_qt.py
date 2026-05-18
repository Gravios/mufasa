"""
tests/smoke_122ct_min_max_draw_size_qt.py
===========================================

Patch 122ct: SetMinMaxDrawWindowSize Qt port + Tk deletion.

Second of the 4 subprocess-launched popups identified in 122cr's
discovery. 69-line Tk popup → Qt dialog with QDoubleSpinBox-based
ratio inputs; subprocess-launch bridge in
`ui_qt/dialogs/roi_video_table.py:_action_min_max_draw_size` is
replaced with a direct dialog invocation; Tk popup deleted.

Subprocess-popup count: 3 → 2.

Coverage
--------
1.  Tk popup `min_max_draw_size_popup.py` is gone.
2.  Qt dialog `ui_qt/dialogs/min_max_draw_size.py` exists.
3.  Qt dialog defines `MinMaxDrawSizeDialog` class.
4.  Qt dialog subclasses `QDialog`.
5.  Qt dialog has `init_failed()` helper.
6.  Qt dialog uses QDoubleSpinBox (the Qt-idiomatic improvement
    over Tk dropdowns of discrete float values).
7.  `roi_video_table.py:_action_min_max_draw_size` no longer
    contains the string-literal subprocess launch.
8.  `roi_video_table.py:_action_min_max_draw_size` calls
    `MinMaxDrawSizeDialog`.
9.  No file imports from the deleted Tk popup module.
10. `tk_surface_audit.md` §2g records the deletion
    (subprocess popup count: 3 → 2).
11. mufasa/ui/pop_ups/ count dropped by 1.
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

    # 1. Tk popup gone
    tk = pkg / "ui" / "pop_ups" / "min_max_draw_size_popup.py"
    check("Tk popup min_max_draw_size_popup.py is gone",
          not tk.exists())

    # 2-6. Qt dialog shape
    qt = pkg / "ui_qt" / "dialogs" / "min_max_draw_size.py"
    check("Qt dialog ui_qt/dialogs/min_max_draw_size.py exists",
          qt.exists())

    qt_src = qt.read_text() if qt.exists() else ""
    qt_tree = ast.parse(qt_src) if qt_src else None

    if qt_tree is not None:
        dialog_cls = None
        for n in qt_tree.body:
            if (isinstance(n, ast.ClassDef)
                    and n.name == "MinMaxDrawSizeDialog"):
                dialog_cls = n
                break
        check("MinMaxDrawSizeDialog class defined",
              dialog_cls is not None)

        if dialog_cls is not None:
            base_names = [ast.unparse(b) for b in dialog_cls.bases]
            check(
                "MinMaxDrawSizeDialog subclasses QDialog",
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
            "Qt dialog uses QDoubleSpinBox (Qt-idiomatic numeric "
            "input; replaces Tk's MufasaDropDown of discrete floats)",
            "QDoubleSpinBox" in qt_src,
        )

    # 7-8. Wiring
    rvt = (pkg / "ui_qt" / "dialogs" /
           "roi_video_table.py").read_text()
    check(
        "_action_min_max_draw_size no longer contains the "
        "subprocess Tk-launch string",
        "from mufasa.ui.pop_ups.min_max_draw_size_popup" not in rvt,
    )
    check(
        "_action_min_max_draw_size calls MinMaxDrawSizeDialog",
        "MinMaxDrawSizeDialog" in rvt,
    )

    # 9. No file imports the deleted module
    leftover: list[str] = []
    for f in pkg.rglob("*.py"):
        try:
            t = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(t):
            if (isinstance(node, ast.ImportFrom)
                    and node.module ==
                    "mufasa.ui.pop_ups.min_max_draw_size_popup"):
                leftover.append(
                    f"{f.relative_to(pkg)}:{node.lineno}")
    check(
        "no file imports from the deleted Tk popup module",
        leftover == [],
        detail=", ".join(leftover),
    )

    # 10. Doc updates
    audit = (REPO_ROOT / "docs" / "tk_surface_audit.md").read_text()
    check(
        "tk_surface_audit.md §2g records the deletion "
        "(subprocess popup count: 3 → 2)",
        "DELETED 122ct" in audit
        and "min_max_draw_size" in audit,
    )

    # 11. Count
    popups_count = sum(
        1 for _ in (pkg / "ui" / "pop_ups").glob("*.py")
        if _.name != "__init__.py"
    )
    check(
        f"mufasa/ui/pop_ups/ count ≤ 77 (was 78 post-122cs; "
        f"got {popups_count})",
        popups_count <= 77,
    )

    # 12. All files parse cleanly
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
        f"smoke_122ct_min_max_draw_size_qt: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
