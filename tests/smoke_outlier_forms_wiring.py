"""
tests/smoke_outlier_forms_wiring.py
===================================

Patch 122c: AST-level wiring checks for the two new
outlier-correction forms and the redesigned pose-cleanup page.

PySide6 isn't available in the sandbox so we verify wiring at
the source level (same pattern as smoke_pose_cleanup_v2_wiring).

Coverage:

* :class:`RunOutlierCorrectionForm` exists, subclasses
  OperationForm, has build/collect_args/target, and:
    - uses :class:`InputSourcePicker` in build()
    - calls both ``OutlierCorrecterMovement`` and
      ``OutlierCorrecterLocation`` from target()
    - branches on ``do_movement`` / ``do_location`` flags
    - writes ``run.toml`` provenance via ``write_run_toml``
* :class:`SkipOutlierCorrectionForm` exists with the same
  scaffolding, and:
    - calls ``OutlierCorrectionSkipper`` from target()
    - writes ``run.toml`` provenance via ``write_run_toml``
* Both classes appear in ``__all__``.
* The pose-cleanup page builder lists sections in the redesigned
  order (interpolate → kalman_v2 → run-outlier → skip-outlier →
  egocentric → advanced/legacy) and the legacy section stacks
  the three legacy forms.
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


def _load_module(path: Path) -> ast.AST:
    return ast.parse(path.read_text())


def _find_class(tree: ast.AST, name: str) -> ast.ClassDef | None:
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef) and n.name == name:
            return n
    return None


def _class_methods(cls: ast.ClassDef) -> set[str]:
    return {
        n.name for n in cls.body
        if isinstance(n, ast.FunctionDef)
    }


def _all_export_list(tree: ast.AST) -> list[str]:
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "__all__"
            and isinstance(node.value, ast.List)
        ):
            return [
                elt.value for elt in node.value.elts
                if isinstance(elt, ast.Constant)
            ]
    return []


def check_run_outlier_form(tree: ast.AST) -> None:
    cls = _find_class(tree, "RunOutlierCorrectionForm")
    check(
        "RunOutlierCorrectionForm class exists",
        cls is not None,
    )
    if cls is None:
        return

    base_names = [
        b.id for b in cls.bases if isinstance(b, ast.Name)
    ]
    check(
        "RunOutlierCorrectionForm subclasses OperationForm",
        "OperationForm" in base_names,
    )

    methods = _class_methods(cls)
    for required in ("build", "collect_args", "target"):
        check(
            f"RunOutlierCorrectionForm defines {required}()",
            required in methods,
        )

    cls_src = ast.unparse(cls)

    # build() must instantiate InputSourcePicker
    check(
        "RunOutlierCorrectionForm.build() uses InputSourcePicker",
        "InputSourcePicker" in cls_src,
    )
    check(
        "RunOutlierCorrectionForm exposes do_movement / do_location",
        "do_movement" in cls_src and "do_location" in cls_src,
    )
    # target() must invoke both backends
    check(
        "RunOutlierCorrectionForm calls OutlierCorrecterMovement",
        "OutlierCorrecterMovement" in cls_src,
    )
    check(
        "RunOutlierCorrectionForm calls OutlierCorrecterLocation",
        "OutlierCorrecterLocation" in cls_src,
    )
    # target() must branch on both flags
    check(
        "RunOutlierCorrectionForm branches on do_movement/do_location",
        "do_movement and do_location" in cls_src
        or ("if do_movement" in cls_src and "if do_location" in cls_src)
        or "elif do_movement" in cls_src,
    )
    # Provenance: write_run_toml for v1 projects
    check(
        "RunOutlierCorrectionForm writes run.toml provenance",
        "write_run_toml" in cls_src,
    )
    check(
        "RunOutlierCorrectionForm gates provenance on v1 root",
        "resolve_v1_project_root" in cls_src
        or "v1_root" in cls_src,
    )


def check_skip_outlier_form(tree: ast.AST) -> None:
    cls = _find_class(tree, "SkipOutlierCorrectionForm")
    check(
        "SkipOutlierCorrectionForm class exists",
        cls is not None,
    )
    if cls is None:
        return

    base_names = [
        b.id for b in cls.bases if isinstance(b, ast.Name)
    ]
    check(
        "SkipOutlierCorrectionForm subclasses OperationForm",
        "OperationForm" in base_names,
    )

    methods = _class_methods(cls)
    for required in ("build", "collect_args", "target"):
        check(
            f"SkipOutlierCorrectionForm defines {required}()",
            required in methods,
        )

    cls_src = ast.unparse(cls)
    check(
        "SkipOutlierCorrectionForm calls OutlierCorrectionSkipper",
        "OutlierCorrectionSkipper" in cls_src,
    )
    check(
        "SkipOutlierCorrectionForm writes run.toml provenance",
        "write_run_toml" in cls_src,
    )
    check(
        "SkipOutlierCorrectionForm gates provenance on v1 root",
        "resolve_v1_project_root" in cls_src
        or "v1_root" in cls_src,
    )


def check_exports(tree: ast.AST) -> None:
    exports = _all_export_list(tree)
    for name in (
        "RunOutlierCorrectionForm",
        "SkipOutlierCorrectionForm",
    ):
        check(
            f"forms module exports {name}",
            name in exports,
            detail=f"got {exports}",
        )


def check_page_order(page_tree: ast.AST) -> None:
    """The pose-cleanup page builder must lay out sections in the
    redesigned order. We scan the build function for add_section
    calls and verify section titles appear in sequence.
    """
    # Find the build_pose_cleanup_page function
    func = None
    for n in ast.walk(page_tree):
        if (
            isinstance(n, ast.FunctionDef)
            and n.name == "build_pose_cleanup_page"
        ):
            func = n
            break
    check("build_pose_cleanup_page function exists", func is not None)
    if func is None:
        return

    # Walk the body, collect add_section calls in order
    section_titles: list[str] = []
    section_forms: list[list[str]] = []
    for node in ast.walk(func):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_section"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            title = node.args[0].value
            section_titles.append(title)
            # Second positional arg is the [(form_cls, kwargs)] list
            forms_in_section: list[str] = []
            if len(node.args) >= 2 and isinstance(
                node.args[1], ast.List
            ):
                for tup in node.args[1].elts:
                    if isinstance(tup, ast.Tuple) and tup.elts:
                        first = tup.elts[0]
                        if isinstance(first, ast.Name):
                            forms_in_section.append(first.id)
            section_forms.append(forms_in_section)

    # Patch 122bf: 122x moved "Preprocess Videos" and "Video
    # Calibration" from data_import_page to this page. The 6
    # original sections are still present in the same relative
    # order, with the two new ones prepended.
    expected_order = [
        "Preprocess Videos",
        "Video Calibration",
        "Interpolate missing frames",
        "Kalman v2 smoothing",
        "Run outlier correction",
        "Skip outlier correction",
        "Egocentric alignment",
        "Advanced / legacy",
    ]
    check(
        f"page has {len(expected_order)} sections in redesigned order",
        section_titles == expected_order,
        detail=f"got {section_titles}",
    )

    # Legacy section must stack all three legacy forms
    legacy_idx = (
        section_titles.index("Advanced / legacy")
        if "Advanced / legacy" in section_titles else -1
    )
    if legacy_idx >= 0:
        legacy_forms = section_forms[legacy_idx]
        for expected_form in (
            "SmoothingForm",
            "OutlierSettingsForm",
            "DropBodypartsForm",
        ):
            check(
                f"legacy section includes {expected_form}",
                expected_form in legacy_forms,
                detail=f"got {legacy_forms}",
            )

    # Single-form sections each contain exactly the expected form
    expected_single = {
        "Interpolate missing frames":  "InterpolateForm",
        "Kalman v2 smoothing":         "KalmanV2SmoothingForm",
        "Run outlier correction":      "RunOutlierCorrectionForm",
        "Skip outlier correction":     "SkipOutlierCorrectionForm",
        "Egocentric alignment":        "EgocentricAlignmentForm",
    }
    for title, form_name in expected_single.items():
        if title in section_titles:
            i = section_titles.index(title)
            forms = section_forms[i]
            check(
                f"section '{title}' hosts {form_name}",
                forms == [form_name],
                detail=f"got {forms}",
            )

    # Make sure the obsolete top-level sections are gone
    for stale in (
        "Smoothing",
        "Outlier correction settings",
        "Drop body-parts",
    ):
        check(
            f"page no longer has stale top-level section '{stale}'",
            stale not in section_titles,
        )


def main() -> int:
    forms_path = REPO_ROOT / "mufasa" / "ui_qt" / "forms" / "pose_cleanup.py"
    page_path = REPO_ROOT / "mufasa" / "ui_qt" / "pages" / "pose_cleanup_page.py"
    forms_tree = _load_module(forms_path)
    page_tree = _load_module(page_path)

    check_run_outlier_form(forms_tree)
    check_skip_outlier_form(forms_tree)
    check_exports(forms_tree)
    check_page_order(page_tree)

    # Page imports both new form names
    page_src = page_path.read_text()
    for name in (
        "RunOutlierCorrectionForm",
        "SkipOutlierCorrectionForm",
    ):
        check(
            f"page builder imports {name}",
            name in page_src,
        )

    # Form module imports the two outlier backends + skipper inline
    # (lazy imports inside target() are fine — verify they show up
    # in the file at all so a refactor doesn't silently drop them).
    forms_src = forms_path.read_text()
    for name in (
        "OutlierCorrecterMovement",
        "OutlierCorrecterLocation",
        "OutlierCorrectionSkipper",
    ):
        check(
            f"forms module references {name}",
            name in forms_src,
        )

    print(
        f"smoke_outlier_forms_wiring: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
