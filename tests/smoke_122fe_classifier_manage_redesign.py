"""
tests/smoke_122fe_classifier_manage_redesign.py
==================================================

Patch 122fe — Manage classifiers UI redesigned as a table.

User request (Tue May 26, 2026):

> Manage classifiers : Manage classifiers, should present the
> classifiers in a table format with a delete button, and show
> info button for each classifier. Guard against no classifiers.
> At the bottom of the table there should be a "+" button for
> adding a new classifier. Guard against empty or duplicate name.
> no need for a run button. Each classifier should have an
> associated key that is used by Frame labeling.

The legacy ACTIONS-dropdown form is replaced with a table view:
one row per classifier, columns Name | Key | Info | Delete, with
a "+ Add classifier" button at the bottom that opens a modal.
The Run button inherited from OperationForm is hidden — actions
are immediate (no submit step).

NEW STORAGE
===========

Each classifier can have a single-character keyboard hotkey,
stored in project.toml::

    [classifiers]
    targets = ["Attack", "Groom"]

    [classifiers.keys]
    Attack = "a"
    Groom = "g"

Frame labelling (separate patch) will read this map to bind
hotkeys during annotation. Legacy projects that have only
``targets`` and no ``keys`` show "(unset)" in the Key column.

NEW HELPERS
===========

* ``_read_classifier_keys(config_path) → dict[str, str]``
* ``_write_classifier_keys(config_path, keys) → None``
* ``_find_classifier_model(config_path, name) → Path | None``

EXISTING HELPERS UNCHANGED
===========================

* ``_read_classifiers(config_path) → list[str]``
* ``_write_classifiers(config_path, targets) → None``

NEW UI COMPONENTS
=================

* ``_AddClassifierDialog`` — modal for adding a new classifier
  (name + key, with inline validation: empty name blocks OK,
  duplicate name blocks OK, key collision warns but allows).
* ``_EditKeyDialog`` — modal for reassigning a classifier's key.
* ``ClassifierManageForm`` — rewritten as table view. Hides the
  inherited Run button. Per-row Info / Delete buttons. Table
  footer "+ Add classifier" button. Empty-state label when no
  classifiers defined.

REMOVED
=======

* ``_AddClfPanel``, ``_RemoveClfPanel``, ``_PrintClfPanel`` —
  the three QStackedWidget panels behind the legacy ACTIONS
  dropdown.

Coverage
--------
Storage helpers (3 checks):
1.  _read_classifier_keys returns empty dict for projects with
    no [classifiers.keys] block.
2.  _write_classifier_keys round-trips through project.toml.
3.  _find_classifier_model returns None when no model exists,
    and a Path when one does (best-effort path matching).

Add dialog (2 checks):
4.  _AddClassifierDialog._validate disables OK for empty name.
5.  _AddClassifierDialog._validate disables OK for duplicate name.

Form structure (4 checks):
6.  ClassifierManageForm has section_id=None (config-edit only,
    no provenance recording).
7.  ClassifierManageForm.build hides the inherited Run button.
8.  ClassifierManageForm declares the four columns (Name, Key,
    Info, Delete) via setHorizontalHeaderLabels.
9.  ClassifierManageForm declares an empty_label (placeholder
    when there are no classifiers — empty-state UX).

Removed legacy code (1 check):
10. _AddClfPanel, _RemoveClfPanel, _PrintClfPanel removed from
    the module (the QStackedWidget panels of the ACTIONS-
    dropdown form).

Cross-patch invariants (3 checks):
11. 122fd state preserved: roi_canvas.shape_selected signal.
12. 122fc state preserved: SECTIONS['import_video'] registered.
13. Parse-clean.
"""
from __future__ import annotations

import ast
import re
import sys
import tempfile
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


def _extract_helpers() -> dict:
    """AST-extract the three storage helpers from classifier.py so
    they can be tested without importing PySide6 (which isn't
    available in the sandbox)."""
    cls_path = (REPO_ROOT / "mufasa" / "ui_qt" / "forms"
                / "classifier.py")
    tree = ast.parse(cls_path.read_text())
    # Stub the project_layout imports.
    from mufasa.project_layout import (  # noqa: F401
        read_project_toml,
        write_project_toml,
        project_metadata_from_config,
    )
    ns: dict = {
        "Path": Path,
        "read_project_toml": read_project_toml,
        "write_project_toml": write_project_toml,
        "project_metadata_from_config": project_metadata_from_config,
    }
    # Exec just the three helper FunctionDefs.
    wanted = {
        "_read_classifiers", "_write_classifiers",
        "_read_classifier_keys", "_write_classifier_keys",
        "_find_classifier_model",
    }
    for node in tree.body:
        if (isinstance(node, ast.FunctionDef)
                and node.name in wanted):
            # Rewrite local imports inside the function body to use
            # the names already in `ns` (read_project_toml etc.).
            # The functions use `from mufasa.project_layout import
            # ...` locally; we exec the function source which
            # re-runs those imports under ns.
            exec(compile(ast.Module(body=[node], type_ignores=[]),
                          "<smoke>", "exec"), ns)
    return ns


def main() -> int:
    helpers = _extract_helpers()
    _read_classifier_keys = helpers["_read_classifier_keys"]
    _write_classifier_keys = helpers["_write_classifier_keys"]
    _find_classifier_model = helpers["_find_classifier_model"]

    # -----------------------------------------------------------------
    # Storage helpers
    # -----------------------------------------------------------------
    # 1. _read_classifier_keys on empty project.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = root / "project.toml"
        cfg.write_text(
            'project_layout_version = 1\n[project]\nname = "t"\n',
        )
        got = _read_classifier_keys(str(cfg))
        check(
            "_read_classifier_keys returns empty dict for projects "
            "with no [classifiers.keys] block",
            got == {},
            detail=(f"got {got!r}"),
        )

    # 2. _write_classifier_keys round-trip.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = root / "project.toml"
        cfg.write_text(
            'project_layout_version = 1\n[project]\nname = "t"\n'
            '[classifiers]\ntargets = ["A", "B"]\n',
        )
        _write_classifier_keys(str(cfg), {"A": "a", "B": "b"})
        roundtripped = _read_classifier_keys(str(cfg))
        check(
            "_write_classifier_keys round-trips through "
            "project.toml's [classifiers.keys] table",
            roundtripped == {"A": "a", "B": "b"},
            detail=(f"got {roundtripped!r}"),
        )

    # 3. _find_classifier_model — best-effort match.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = root / "project.toml"
        cfg.write_text('[project]\nname = "t"\n')
        # No model yet → None.
        none_case = _find_classifier_model(str(cfg), "Attack")
        # Create one.
        (root / "models").mkdir()
        (root / "models" / "Attack.sav").write_text("model")
        found_case = _find_classifier_model(str(cfg), "Attack")
        check(
            "_find_classifier_model returns None when no model "
            "exists; returns the path when one does (best-effort "
            "convention match)",
            none_case is None and found_case is not None,
            detail=(
                f"none_case={none_case!r}, found_case={found_case!r}"
            ),
        )

    # -----------------------------------------------------------------
    # AST checks of the redesigned form
    # -----------------------------------------------------------------
    cls_path = (REPO_ROOT / "mufasa" / "ui_qt" / "forms"
                / "classifier.py")
    cls_src = cls_path.read_text()
    cls_tree = ast.parse(cls_src)

    # Locate the relevant classes.
    classes = {
        node.name: node for node in ast.walk(cls_tree)
        if isinstance(node, ast.ClassDef)
    }

    # 4. _AddClassifierDialog._validate disables OK on empty name.
    add_dlg = classes.get("_AddClassifierDialog")
    has_validate = False
    if add_dlg is not None:
        for m in add_dlg.body:
            if isinstance(m, ast.FunctionDef) and m.name == "_validate":
                v_src = ast.unparse(m)
                # Look for: name check + ok_btn.setEnabled(False).
                if ("not name" in v_src
                        and "setEnabled(False)" in v_src):
                    has_validate = True
                break
    check(
        "_AddClassifierDialog._validate disables OK when name is "
        "empty (guard against empty-name per user request)",
        has_validate,
    )

    # 5. Duplicate name validation.
    has_dup_check = False
    if add_dlg is not None:
        for m in add_dlg.body:
            if isinstance(m, ast.FunctionDef) and m.name == "_validate":
                v_src = ast.unparse(m)
                if ("in existing" in v_src
                        and "already exists" in v_src):
                    has_dup_check = True
                break
    check(
        "_AddClassifierDialog._validate disables OK for duplicate "
        "names (guard against duplicates per user request)",
        has_dup_check,
    )

    # 6. ClassifierManageForm has section_id=None.
    mgr = classes.get("ClassifierManageForm")
    section_id_is_none = False
    if mgr is not None:
        for m in mgr.body:
            if isinstance(m, ast.Assign):
                for tgt in m.targets:
                    if (isinstance(tgt, ast.Name)
                            and tgt.id == "section_id"
                            and isinstance(m.value, ast.Constant)
                            and m.value.value is None):
                        section_id_is_none = True
    check(
        "ClassifierManageForm.section_id = None (config-edit only; "
        "no provenance recording — 'no need for a run button' per "
        "the user request)",
        section_id_is_none,
    )

    # 7. build() hides run_btn.
    hides_run_btn = False
    if mgr is not None:
        for m in mgr.body:
            if isinstance(m, ast.FunctionDef) and m.name == "build":
                b_src = ast.unparse(m)
                if "run_btn.setVisible(False)" in b_src:
                    hides_run_btn = True
                break
    check(
        "ClassifierManageForm.build() hides the inherited Run "
        "button via self.run_btn.setVisible(False)",
        hides_run_btn,
    )

    # 8. Four columns: Name, Key, Info, Delete.
    has_four_cols = False
    if mgr is not None:
        for m in mgr.body:
            if isinstance(m, ast.FunctionDef) and m.name == "build":
                b_src = ast.unparse(m)
                # ast.unparse uses single quotes for strings
                if ("'Name'" in b_src and "'Key'" in b_src
                        and "'Info'" in b_src and "'Delete'" in b_src
                        and "setColumnCount(4)" in b_src):
                    has_four_cols = True
                break
    check(
        "ClassifierManageForm.build() declares 4-column table "
        "with headers Name, Key, Info, Delete",
        has_four_cols,
    )

    # 9. Empty-state label.
    has_empty_label = "self.empty_label" in cls_src
    has_no_classifiers_text = "No classifiers defined" in cls_src
    check(
        "ClassifierManageForm declares an empty_label "
        "placeholder shown when there are no classifiers "
        "(guard against no-classifiers per user request)",
        has_empty_label and has_no_classifiers_text,
        detail=(
            f"empty_label={has_empty_label} "
            f"no_classifiers_text={has_no_classifiers_text}"
        ),
    )

    # 10. Legacy panel classes removed.
    legacy_classes = {
        "_AddClfPanel", "_RemoveClfPanel", "_PrintClfPanel",
    }
    remaining_legacy = legacy_classes & set(classes.keys())
    check(
        "Legacy ACTIONS-dropdown panels removed: "
        "_AddClfPanel, _RemoveClfPanel, _PrintClfPanel are no "
        "longer defined (clean replacement of the legacy form)",
        not remaining_legacy,
        detail=(f"remaining: {sorted(remaining_legacy)}"),
    )

    # -----------------------------------------------------------------
    # Cross-patch invariants
    # -----------------------------------------------------------------
    rc_src = (REPO_ROOT / "mufasa" / "ui_qt" / "dialogs"
              / "roi_canvas.py").read_text()
    check(
        "122fd state preserved: ROICanvas declares shape_selected "
        "signal",
        "shape_selected" in rc_src and "Signal(int)" in rc_src,
    )

    from mufasa.section_provenance import SECTIONS
    check(
        "122fc state preserved: SECTIONS['import_video'] "
        "registered",
        "import_video" in SECTIONS,
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

    print(
        f"smoke_122fe_classifier_manage_redesign: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
