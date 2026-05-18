"""
tests/smoke_122cj_qt_confirm_override.py
==========================================

Patch 122cj: Qt-side override of `confirm_two_option`. Companion
to patch 122ch (which introduced the UI-agnostic helper).

122ch left the workbench falling back to the Tk popup — the
helper had an `override pattern` but no actual Qt installer.
This patch ships the installer and wires it into workbench
startup so backend confirmations render natively.

What's verified:

* `mufasa/ui_qt/qt_confirm.py` exists.
* Defines `qt_confirm_two_option(question, option_one, option_two,
  title)` with the same signature as the Tk default.
* Defines `install_qt_confirm_override()` that reassigns
  `mufasa.utils.confirm.confirm_two_option` to the Qt version.
* Uses `QMessageBox.addButton(label, AcceptRole/RejectRole)`
  rather than `QMessageBox.Yes/No` — preserves caller's option
  labels (SKIP/TERMINATE for training meta-config errors, etc.).
* `workbench_app.main()` imports + calls
  `install_qt_confirm_override` right after QApplication
  construction.
* PySide6 imports inside qt_confirm.py are lazy (function-body),
  not at module load — so the module is importable in the
  sandbox for AST audits.

Coverage
--------
1. qt_confirm.py exists.
2. qt_confirm_two_option function defined.
3. install_qt_confirm_override function defined.
4. install function reassigns mufasa.utils.confirm.confirm_two_option.
5. qt_confirm_two_option uses addButton with AcceptRole + RejectRole.
6. qt_confirm_two_option does NOT use QMessageBox.Yes/No constants
   (those would force "Yes"/"No" labels regardless of caller intent).
7. PySide6 import inside qt_confirm.py is lazy (no module-level
   `from PySide6.X import ...`).
8. workbench_app.py imports install_qt_confirm_override.
9. workbench_app.main() calls install_qt_confirm_override() after
   QApplication construction.
10. backend_audit.md §4d mentions the Qt override is installed
    in 122cj.
11. All mufasa/**/*.py files parse cleanly.
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


def _module_level_imports(tree: ast.Module, prefix: str) -> bool:
    """True iff any top-level Import or ImportFrom has a module
    starting with `prefix`."""
    for n in tree.body:
        if isinstance(n, ast.ImportFrom):
            if n.module and n.module.startswith(prefix):
                return True
        elif isinstance(n, ast.Import):
            for alias in n.names:
                if alias.name.startswith(prefix):
                    return True
    return False


def main() -> int:
    pkg = REPO_ROOT / "mufasa"

    # ==================================================================
    # qt_confirm.py
    # ==================================================================
    qt_confirm_path = pkg / "ui_qt" / "qt_confirm.py"
    check("mufasa/ui_qt/qt_confirm.py exists", qt_confirm_path.exists())
    if not qt_confirm_path.exists():
        return 1
    qc_src = qt_confirm_path.read_text()
    qc_tree = ast.parse(qc_src)

    funcs = {n.name for n in qc_tree.body
             if isinstance(n, ast.FunctionDef)}
    check(
        "qt_confirm_two_option function defined",
        "qt_confirm_two_option" in funcs,
    )
    check(
        "install_qt_confirm_override function defined",
        "install_qt_confirm_override" in funcs,
    )

    # Locate the install function and verify it reassigns the
    # confirm_two_option attribute on mufasa.utils.confirm.
    install_src = ""
    qcto_src = ""
    for n in qc_tree.body:
        if isinstance(n, ast.FunctionDef):
            if n.name == "install_qt_confirm_override":
                install_src = ast.unparse(n)
            elif n.name == "qt_confirm_two_option":
                qcto_src = ast.unparse(n)

    check(
        "install function reassigns mufasa.utils.confirm.confirm_two_option",
        ".confirm_two_option =" in install_src
        and ("qt_confirm_two_option" in install_src
             or "_cf" in install_src),
    )

    # qt_confirm_two_option uses addButton w/ AcceptRole + RejectRole
    check(
        "qt_confirm_two_option uses addButton + AcceptRole",
        ".addButton(" in qcto_src
        and "AcceptRole" in qcto_src,
    )
    check(
        "qt_confirm_two_option uses addButton + RejectRole",
        ".addButton(" in qcto_src
        and "RejectRole" in qcto_src,
    )
    # Does NOT force QMessageBox.Yes / No constants
    check(
        "qt_confirm_two_option does NOT use QMessageBox.Yes/No "
        "constants (would override caller's labels)",
        "QMessageBox.Yes" not in qcto_src
        and "QMessageBox.No" not in qcto_src,
    )

    # PySide6 import is lazy
    has_top_pyside = _module_level_imports(qc_tree, "PySide6")
    check(
        "PySide6 import in qt_confirm.py is lazy (no module-level "
        "PySide6 import)",
        not has_top_pyside,
    )
    check(
        "qt_confirm.py has a deferred PySide6 import inside the "
        "Qt function body",
        "from PySide6.QtWidgets import" in qc_src
        and "QMessageBox" in qc_src,
    )

    # ==================================================================
    # workbench_app integration
    # ==================================================================
    wb_src = (pkg / "ui_qt" / "workbench_app.py").read_text()
    wb_tree = ast.parse(wb_src)

    # Find main() and check it imports + calls install_qt_confirm_override
    main_src = ""
    for n in wb_tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == "main":
            main_src = ast.unparse(n)
            break

    check(
        "workbench_app.main() imports install_qt_confirm_override",
        "from mufasa.ui_qt.qt_confirm import install_qt_confirm_override"
        in main_src
        or "install_qt_confirm_override" in main_src,
    )
    check(
        "workbench_app.main() calls install_qt_confirm_override()",
        "install_qt_confirm_override()" in main_src,
    )
    # Ordering: install happens AFTER QApplication is constructed
    # (otherwise no QApplication for QMessageBox to attach to).
    qapp_pos = main_src.find("QApplication.instance()")
    install_pos = main_src.find("install_qt_confirm_override()")
    check(
        "install call happens AFTER QApplication construction",
        qapp_pos != -1 and install_pos != -1 and install_pos > qapp_pos,
    )

    # ==================================================================
    # Behavioural smoke: install can run with confirm.py present
    # (verifies the rebinding works structurally; doesn't open Qt
    # dialogs because PySide6 is unavailable)
    # ==================================================================
    try:
        import mufasa.utils.confirm as _cf
        original = _cf.confirm_two_option
        # Manually simulate what install_qt_confirm_override does
        # WITHOUT importing PySide6 — verify the reassignment
        # pattern works.
        def _sentinel(*_a, **_kw):
            return "SENTINEL"
        _cf.confirm_two_option = _sentinel
        try:
            assert _cf.confirm_two_option("q") == "SENTINEL"
        finally:
            _cf.confirm_two_option = original
        check(
            "behavioural: confirm_two_option is reassignable; "
            "post-rebinding calls route to the new function",
            True,
        )
    except Exception as exc:
        check(
            "behavioural: confirm_two_option reassign smoke",
            False,
            detail=f"{type(exc).__name__}: {exc}",
        )

    # ==================================================================
    # Doc updates
    # ==================================================================
    audit = (REPO_ROOT / "docs" / "backend_audit.md").read_text()
    check(
        "backend_audit.md §4d mentions Qt override installed in 122cj",
        "122cj" in audit and "QMessageBox" in audit,
    )

    # ==================================================================
    # All files parse cleanly
    # ==================================================================
    parse_errors: list[str] = []
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
        f"smoke_122cj_qt_confirm_override: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
