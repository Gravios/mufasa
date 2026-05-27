"""
tests/smoke_122fb_roi_maintenance_removed.py
==============================================

Patch 122fb — remove the standalone "Maintenance" ROI section.
Its CSV-import and size-standardize actions now live behind a
popup launched from the Definitions panel.

User decision (Tue May 26, 2026, confirming option (b) of the
three options discussed in 122ex's commit thread):

> b, I'm going to ignore the egocentric alignment badge for now.
> [...]
> ROI: DefinitionsMove the standardize sizes to Definitions, and
> remove Maintenance.

WHAT THIS PATCH LANDED
======================

mufasa/ui_qt/pages/roi_page.py:
* ``page.add_section("Maintenance", ...)`` removed.
* ROIManageForm import dropped (no longer used directly here).
* Inline comment documents the move + redirect.

mufasa/ui_qt/dialogs/roi_define_panel.py:
* New "Import / standardize…" button in the toolbar, after the
  Edit button.
* New ``_on_maintenance_clicked`` handler opens ROIManageForm in
  a modal QDialog. On dialog close, panel auto-syncs so newly
  imported ROIs appear immediately.
* Bails with a status flash if no config_path is available.

The form itself (ROIManageForm) is UNCHANGED. Only its placement
moved from "inline workbench section" to "popup dialog launched
from the Definitions panel toolbar." This preserves the existing
form's contract (collect_args / target / dispatch logic) while
satisfying the user's UX intent of "Definitions handles ROI work
directly, no separate Maintenance section."

The form's third action ("Draw ROIs (interactive)") is now
redundant with the Definitions panel's own Draw button. Left in
place; documented as deferred clean-up. Removing it would change
the form's ACTIONS list, build, collect_args, and target — bigger
than the placement change this patch is scoped to.

Coverage
--------
Section removal (2 checks):
1.  roi_page.py no longer registers a "Maintenance" section.
2.  ROIManageForm import dropped from roi_page.py (F401 hygiene).

Popup wiring (3 checks):
3.  roi_define_panel.py declares maintenance_btn ("Import /
    standardize…").
4.  roi_define_panel.py defines _on_maintenance_clicked.
5.  _on_maintenance_clicked instantiates ROIManageForm and
    shows it via QDialog.exec.

Cross-patch invariants (5 checks):
6.  122fa state preserved: classifier pages have short renames.
7.  122ez state preserved: SECTIONS['egocentric'].detect_path
    is callable.
8.  Other ROI sections (Definitions, Analyze, Visualize, Features)
    still present.
9.  Parse-clean.
10. 122do baseline (Optional[] hygiene).
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


def main() -> int:
    rp_path = REPO_ROOT / "mufasa" / "ui_qt" / "pages" / "roi_page.py"
    rp_src = rp_path.read_text()
    rp_tree = ast.parse(rp_src)

    # 1. No more "Maintenance" add_section call.
    maintenance_calls = []
    for node in ast.walk(rp_tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_section"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "Maintenance"):
            maintenance_calls.append(node.lineno)
    check(
        "roi_page.py no longer registers an active 'Maintenance' "
        "section (122fb removed it; CSV import + size standardize "
        "moved to a popup on the Definitions panel)",
        not maintenance_calls,
        detail=(
            f"found Maintenance add_section calls at lines: "
            f"{maintenance_calls}"
        ),
    )

    # 2. ROIManageForm import dropped from roi_page.py.
    # Use AST to check the actual import statements (the string
    # "ROIManageForm" still appears in a comment — that's intentional
    # documentation, not a real import).
    rp_imports = set()
    for node in ast.walk(rp_tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                rp_imports.add(alias.name)
    check(
        "roi_page.py no longer imports ROIManageForm (F401 "
        "hygiene — the form is now used only by the popup "
        "launched from roi_define_panel.py)",
        "ROIManageForm" not in rp_imports,
        detail=(f"imports: {sorted(rp_imports)}"),
    )

    # -----------------------------------------------------------------
    # Popup wiring
    # -----------------------------------------------------------------
    rdp_path = (REPO_ROOT / "mufasa" / "ui_qt" / "dialogs"
                / "roi_define_panel.py")
    rdp_src = rdp_path.read_text()
    rdp_tree = ast.parse(rdp_src)

    # 3. maintenance_btn declared in toolbar.
    check(
        "roi_define_panel.py declares self.maintenance_btn "
        "with label 'Import / standardize…' (122fb toolbar "
        "addition)",
        ("self.maintenance_btn" in rdp_src
         and "Import / standardize" in rdp_src),
    )

    # 4. _on_maintenance_clicked method exists.
    has_handler = False
    for node in ast.walk(rdp_tree):
        if (isinstance(node, ast.FunctionDef)
                and node.name == "_on_maintenance_clicked"):
            has_handler = True
            break
    check(
        "roi_define_panel.py defines _on_maintenance_clicked "
        "(handler for the new toolbar button)",
        has_handler,
    )

    # 5. _on_maintenance_clicked builds a QDialog with ROIManageForm.
    handler_uses_dialog = False
    handler_uses_form = False
    for node in ast.walk(rdp_tree):
        if (isinstance(node, ast.FunctionDef)
                and node.name == "_on_maintenance_clicked"):
            body_src = ast.unparse(node)
            handler_uses_dialog = (
                "QDialog(" in body_src
                and ".exec(" in body_src
            )
            handler_uses_form = "ROIManageForm(" in body_src
            break
    check(
        "_on_maintenance_clicked instantiates ROIManageForm "
        "inside a QDialog and calls .exec() (the popup pattern)",
        handler_uses_dialog and handler_uses_form,
        detail=(
            f"uses_dialog={handler_uses_dialog} "
            f"uses_form={handler_uses_form}"
        ),
    )

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
    cp_src = (REPO_ROOT / "mufasa" / "ui_qt" / "pages"
              / "classifier_page.py").read_text()
    check(
        "122fa state preserved: build_train_classifier_page "
        "calls add_page('Train') (short verb-form rename)",
        'add_page("Train"' in cp_src,
    )

    from mufasa.section_provenance import SECTIONS
    egospec = SECTIONS["egocentric"]
    check(
        "122ez state preserved: SECTIONS['egocentric']."
        "detect_path is callable",
        callable(egospec.detect_path),
    )

    # 8. Other ROI sections still present.
    other_sections = {"Definitions", "Analyze", "Visualize", "Features"}
    rp_sections = set()
    for node in ast.walk(rp_tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in (
                    "add_section", "add_section_widget")
                and node.args
                and isinstance(node.args[0], ast.Constant)):
            rp_sections.add(node.args[0].value)
    check(
        "roi_page.py still registers Definitions, Analyze, "
        "Visualize, Features sections (only Maintenance was "
        "removed)",
        other_sections.issubset(rp_sections),
        detail=(
            f"missing: {sorted(other_sections - rp_sections)}; "
            f"present: {sorted(rp_sections)}"
        ),
    )

    pkg = REPO_ROOT / "mufasa"
    parse_errors = []
    file_count = 0
    for f in sorted(pkg.rglob("*.py")):
        file_count += 1
        try:
            ast.parse(f.read_text())
        except SyntaxError as e:
            parse_errors.append(f"{f.relative_to(REPO_ROOT)}: {e}")
    check(
        f"All mufasa/**/*.py parse cleanly ({file_count} files)",
        not parse_errors,
        detail=(parse_errors[0] if parse_errors else ""),
    )

    uiqt = pkg / "ui_qt"
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
        "122do baseline preserved: no `Optional[` in non-"
        "docstring positions across mufasa/ui_qt/",
        not optional_hits,
        detail=("; ".join(optional_hits[:3])),
    )

    print(
        f"smoke_122fb_roi_maintenance_removed: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
