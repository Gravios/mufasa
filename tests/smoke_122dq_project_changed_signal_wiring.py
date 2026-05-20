"""
tests/smoke_122dq_project_changed_signal_wiring.py
====================================================

Patch 122dq: forward-compat ``projectChanged`` signal on
``MufasaWorkbench`` + wire it to ``ROIDefineWidget.set_config_path``
inside ``build_roi_page``.

Architecture context
--------------------
Today ``MufasaWorkbench._switch_to_project`` handles project
switching by tearing down the whole workbench and building a new
one. Embedded widgets (including ``ROIDefineWidget``) get the new
``config_path`` at construction time through the rebuild — there
is no need for an in-place mutation path in the current code.

However, the ``ROIDefineWidget`` class has had a public
``set_config_path()`` API since patch 122dn that nothing called.
Session handoff caveat #3 flagged this as fragile:

   "ROIDefineWidget has no project-switch handling.
   set_config_path() API exists; nothing calls it. The current
   workbench tears down + rebuilds pages on project change, so this
   MAY work — but if pages persist across project switches, the
   embedded widget will stay stuck on the initial state."

Patch 122dq closes that gap by adding the signal-to-method wiring
even though the signal is currently masked by the teardown
behavior. The API is no longer dead; a future page-persistence
architecture will work without further changes.

What this patch landed
----------------------
1. ``MufasaWorkbench`` — added ``projectChanged = Signal(str)`` as a
   class-level Qt signal.
2. ``MufasaWorkbench._switch_to_project`` — emits
   ``self.projectChanged.emit(config_path)`` as its first action
   (before save_recent_project / build_workbench / teardown).
3. ``mufasa/ui_qt/pages/roi_page.py`` — refactored
   ``_make_define_widget`` from a module-level function to a closure
   inside ``build_roi_page``. The closure captures the ``workbench``
   reference and adds:
       workbench.projectChanged.connect(w.set_config_path)
   inside the factory so the connection is made when the embedded
   widget is constructed (lazily, on first section expand).

Coverage
--------
1.  ``MufasaWorkbench`` declares ``projectChanged`` as a class-level
    attribute (AST detects ``projectChanged = Signal(...)``).
2.  ``projectChanged`` is a ``Signal(str)`` (the payload type is
    ``str``, matching the ``config_path`` argument type).
3.  ``_switch_to_project`` emits ``projectChanged`` (substring check
    on the method body).
4.  The emission happens before the teardown calls (the ``emit`` line
    appears before the ``build_workbench(...)`` call in the method).
5.  ``ROIDefineWidget.set_config_path`` still exists as a method
    (didn't get accidentally deleted in the refactor).
6.  ``_make_define_widget`` is no longer a module-level function in
    ``roi_page.py`` (proves the closure refactor landed).
7.  ``build_roi_page`` contains a nested ``def _make_define_widget``
    (the closure is in the expected place).
8.  The closure connects ``workbench.projectChanged`` to
    ``w.set_config_path`` (substring check on the function body).
9.  The closure still calls ``set_embedded_mode(True)`` (the patch
    didn't accidentally drop the embedded-mode behavior).
10. ``build_roi_page`` still registers ``Definitions`` via
    ``add_section_widget`` and the other four sections via
    ``add_section`` (full section-list integrity).
11. 122do baseline tripwire: still no ``Optional[`` in non-docstring
    positions across ``mufasa/ui_qt/``.
12. All ``mufasa/**/*.py`` parse cleanly.
"""
from __future__ import annotations

import ast
import re
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


def _find_class(tree: ast.AST, name: str) -> ast.ClassDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _find_method(cls: ast.ClassDef, name: str) -> ast.FunctionDef | None:
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def main() -> int:
    pkg = REPO_ROOT / "mufasa"
    uiqt = pkg / "ui_qt"

    wb_path = uiqt / "workbench.py"
    wb_src = wb_path.read_text()
    wb_tree = ast.parse(wb_src)
    wb_cls = _find_class(wb_tree, "MufasaWorkbench")
    assert wb_cls is not None, "MufasaWorkbench class missing"

    # 1. projectChanged declared as class-level attribute.
    has_signal_attr = False
    signal_call: ast.Call | None = None
    for node in wb_cls.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (isinstance(tgt, ast.Name)
                        and tgt.id == "projectChanged"):
                    has_signal_attr = True
                    if isinstance(node.value, ast.Call):
                        signal_call = node.value
                    break
    check(
        "MufasaWorkbench declares `projectChanged` as a class-level "
        "attribute",
        has_signal_attr,
    )

    # 2. projectChanged = Signal(str).
    is_signal_str = False
    if signal_call is not None:
        # Signal(str) — call to a name `Signal` with one positional
        # argument that is a Name `str`.
        if (isinstance(signal_call.func, ast.Name)
                and signal_call.func.id == "Signal"
                and len(signal_call.args) == 1
                and isinstance(signal_call.args[0], ast.Name)
                and signal_call.args[0].id == "str"):
            is_signal_str = True
    check(
        "projectChanged is a `Signal(str)` (the payload is the new "
        "config_path)",
        is_signal_str,
    )

    # 3. _switch_to_project emits projectChanged.
    switch_method = _find_method(wb_cls, "_switch_to_project")
    assert switch_method is not None, "_switch_to_project missing"
    switch_src = ast.unparse(switch_method)
    check(
        "_switch_to_project emits `self.projectChanged.emit(...)`",
        re.search(
            r"self\.projectChanged\.emit\s*\(",
            switch_src,
        ) is not None,
    )

    # 4. emit happens before build_workbench.
    emit_idx = switch_src.find("projectChanged.emit")
    build_idx = switch_src.find("build_workbench(")
    check(
        "projectChanged emission happens before build_workbench(...)"
        " call in _switch_to_project (signal fires before teardown)",
        emit_idx != -1 and build_idx != -1 and emit_idx < build_idx,
    )

    # 5. ROIDefineWidget.set_config_path still exists.
    rdw_path = uiqt / "dialogs" / "roi_define_panel.py"
    rdw_tree = ast.parse(rdw_path.read_text())
    rdw_cls = _find_class(rdw_tree, "ROIDefineWidget")
    assert rdw_cls is not None, "ROIDefineWidget class missing"
    set_cp = _find_method(rdw_cls, "set_config_path")
    check(
        "ROIDefineWidget.set_config_path still exists as a method "
        "(refactor didn't accidentally delete it)",
        set_cp is not None,
    )

    # 6-7. roi_page.py refactor: _make_define_widget is no longer
    # module-level; build_roi_page contains a nested def for it.
    rp_path = uiqt / "pages" / "roi_page.py"
    rp_src = rp_path.read_text()
    rp_tree = ast.parse(rp_src)
    module_level_make = any(
        isinstance(n, ast.FunctionDef) and n.name == "_make_define_widget"
        for n in rp_tree.body
    )
    check(
        "_make_define_widget is no longer a module-level function in "
        "roi_page.py (closure refactor landed)",
        not module_level_make,
    )

    build_fn: ast.FunctionDef | None = None
    for n in rp_tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == "build_roi_page":
            build_fn = n
            break
    assert build_fn is not None, "build_roi_page missing"
    nested_make = any(
        isinstance(n, ast.FunctionDef) and n.name == "_make_define_widget"
        for n in build_fn.body
    )
    check(
        "build_roi_page contains a nested `def _make_define_widget` "
        "(the closure is in the expected place)",
        nested_make,
    )

    # 8. closure connects workbench.projectChanged to set_config_path.
    build_src = ast.unparse(build_fn)
    has_connect = bool(re.search(
        r"workbench\.projectChanged\.connect\s*\(\s*w\.set_config_path\s*\)",
        build_src,
    ))
    check(
        "The closure connects `workbench.projectChanged` → "
        "`w.set_config_path` (this is the 122dq wiring)",
        has_connect,
    )

    # 9. closure still calls set_embedded_mode(True).
    check(
        "The closure still calls `w.set_embedded_mode(True)` "
        "(122dn behavior preserved across the refactor)",
        "set_embedded_mode(True)" in build_src,
    )

    # 10. Section-list integrity.
    has_widget_section = (
        'add_section_widget("Definitions"' in rp_src
        or "add_section_widget('Definitions'" in rp_src
    )
    expected_form_sections = [
        "Maintenance", "Analyze", "Visualize", "Features",
    ]
    sections_present = all(
        f'"{name}"' in rp_src or f"'{name}'" in rp_src
        for name in expected_form_sections
    )
    check(
        "build_roi_page still registers `Definitions` via "
        "add_section_widget and the other four sections "
        "(Maintenance / Analyze / Visualize / Features)",
        has_widget_section and sections_present,
    )

    # 11. 122do baseline tripwire.
    optional_hits = []
    for f in sorted(uiqt.rglob("*.py")):
        src = f.read_text()
        for m in re.finditer(r"\bOptional\[", src):
            preceding = src[:m.start()]
            tq3 = preceding.count('"""') + preceding.count("'''")
            if tq3 % 2 == 0:
                optional_hits.append(str(f.relative_to(uiqt)))
                break
    check(
        "122do baseline preserved: no `Optional[` in non-docstring "
        "positions across mufasa/ui_qt/",
        not optional_hits,
        detail=("; ".join(optional_hits[:3])),
    )

    # 12. Parse-clean.
    parse_errors = []
    file_count = 0
    for f in sorted(pkg.rglob("*.py")):
        file_count += 1
        try:
            ast.parse(f.read_text())
        except SyntaxError as e:
            parse_errors.append(
                f"{f.relative_to(REPO_ROOT)}: {e}")
    check(
        f"All mufasa/**/*.py parse cleanly ({file_count} files)",
        parse_errors == [],
        detail=(parse_errors[0] if parse_errors else ""),
    )

    print(
        f"smoke_122dq_project_changed_signal_wiring: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
