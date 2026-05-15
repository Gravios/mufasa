"""
tests/smoke_122ar_validate_classifier_qt.py
=============================================

Patch 122ar: Qt port of :class:`ValidationVideoPopUp` — the
per-video out-of-sample validation video runner. New form
:class:`ValidateClassifierForm` is an inline
:class:`OperationForm` with the same in-frame +
pop-out-dockable pattern as 122aj's frame labeller, 122al's
batch pre-processor, 122ap's run-inference, and 122aq's
train-classifier forms. Wired into the Classifier page as the
fourth section.

AST-only — PySide6 isn't in the sandbox.

Coverage:

1. New form file exists, parses, defines ValidateClassifierForm
   subclassing OperationForm.
2. Critical methods present (build, collect_args, target,
   _toggle_pop_out, _find_main_window, plus the option/setting
   group builders).
3. Target dispatches single-core vs multi-core based on
   core_cnt — imports both ValidateModelOneVideo and
   ValidateModelOneVideoMultiprocess.
4. Pop-out machinery uses QDockWidget with the 122aj feature
   set (Movable | Floatable | Closable, AllDockWidgetAreas).
5. _find_main_window walks parent chain to a QMainWindow.
6. collect_args validates required inputs (model file exists
   on disk, feature file exists on disk, threshold in [0, 1],
   min_bout non-negative).
7. Major fields present (font_size, text_space, circle_size,
   text_opacity, text_thickness, bp_palette, show_pose,
   show_animal_names, show_clf_conf, show_bbox, core_cnt,
   gantt_type).
8. AUTO handling (font_size / text_space / circle_size return
   None to the validator when AUTO).
9. Classifier page imports ValidateClassifierForm and adds the
   'Validate classifier' section in position 4.
10. 122ar recorded in all touched files.
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


def _attr_defined(src: str, attr: str) -> bool:
    """Space-tolerant check for `attr` being the LHS of an
    assignment somewhere in `src` (skips comments)."""
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        if attr in line and "=" in line:
            lhs = line.split("=", 1)[0].strip()
            if lhs == attr:
                return True
    return False


def main() -> int:
    # ==================================================================
    # 1. New form file
    # ==================================================================
    form_path = (REPO_ROOT / "mufasa" / "ui_qt" / "forms"
                 / "validate_classifier.py")
    check("validate_classifier.py exists", form_path.is_file())
    src = form_path.read_text()
    try:
        tree = ast.parse(src)
        ok = True
    except SyntaxError:
        ok = False
    check("validate_classifier.py parses cleanly", ok)

    classes = {
        n.name: n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef)
    }
    check(
        "ValidateClassifierForm class defined",
        "ValidateClassifierForm" in classes,
    )

    if "ValidateClassifierForm" in classes:
        cls = classes["ValidateClassifierForm"]
        bases = [
            b.id if isinstance(b, ast.Name) else None
            for b in cls.bases
        ]
        check(
            "ValidateClassifierForm subclasses OperationForm",
            "OperationForm" in bases,
        )
        method_names = {
            n.name for n in cls.body
            if isinstance(n, ast.FunctionDef)
        }
        for required in (
            "build", "collect_args", "target",
            "_build_inputs_group", "_build_style_group",
            "_build_tracking_group", "_build_gantt_group",
            "_on_browse_model", "_on_browse_feature",
            "_toggle_pop_out", "_find_main_window",
        ):
            check(
                f"ValidateClassifierForm.{required} defined",
                required in method_names,
            )

    # ==================================================================
    # 2. Target dispatch — single-core vs multi-core
    # ==================================================================
    check(
        "target() imports ValidateModelOneVideo (single-core path)",
        "from mufasa.plotting.single_run_model_validation_video "
        "import" in src
        and "ValidateModelOneVideo" in src,
    )
    check(
        "target() imports ValidateModelOneVideoMultiprocess "
        "(multi-core path)",
        "from mufasa.plotting.single_run_model_validation_video_mp "
        "import" in src
        and "ValidateModelOneVideoMultiprocess" in src,
    )
    check(
        "target() dispatches on core_cnt (== 1 vs > 1)",
        "core_cnt == 1" in src,
    )

    # ==================================================================
    # 3. Pop-out machinery
    # ==================================================================
    check(
        "Pop-out uses QDockWidget",
        "QDockWidget" in src,
    )
    check(
        "Dock features Movable | Floatable | Closable",
        "DockWidgetMovable" in src
        and "DockWidgetFloatable" in src
        and "DockWidgetClosable" in src,
    )
    check(
        "Dock allows all areas",
        "AllDockWidgetAreas" in src,
    )
    check(
        "_find_main_window walks parent chain to QMainWindow",
        "_find_main_window" in src
        and "QMainWindow" in src,
    )

    # ==================================================================
    # 4. Validation
    # ==================================================================
    check(
        "Validates model path exists on disk",
        "os.path.isfile(model_path)" in src,
    )
    check(
        "Validates feature path exists on disk",
        "os.path.isfile(feature_path)" in src,
    )
    check(
        "Validates threshold in [0.0, 1.0]",
        "0.0 <= threshold <= 1.0" in src,
    )
    check(
        "Validates min_bout is non-negative",
        "min_bout < 0" in src,
    )

    # ==================================================================
    # 5. Major fields present
    # ==================================================================
    for attr in (
        "self.model_path",
        "self.feature_path",
        "self.threshold",
        "self.min_bout_ms",
        "self.font_size",
        "self.text_space",
        "self.circle_size",
        "self.text_opacity",
        "self.text_thickness",
        "self.bp_palette",
        "self.show_pose",
        "self.show_animal_names",
        "self.show_clf_conf",
        "self.show_bbox",
        "self.core_cnt",
        "self.gantt_type",
    ):
        check(
            f"form attribute {attr} defined",
            _attr_defined(src, attr),
        )

    # ==================================================================
    # 6. AUTO handling — _auto_or_int translates AUTO → None
    # ==================================================================
    check(
        "AUTO handling: _auto_or_int returns None for AUTO",
        "_auto_or_int" in src
        and "return None if text == AUTO" in src,
    )

    # ==================================================================
    # 7. Gantt + bbox translations
    # ==================================================================
    check(
        "Gantt 'Frame' → 1 (final frame)",
        '"Gantt chart: final frame only' in src
        and "return 1" in src,
    )
    check(
        "Gantt 'Video' → 2",
        '"Gantt chart: video"' in src
        and "return 2" in src,
    )
    check(
        "Bbox FALSE → None",
        "_bbox_value" in src
        and "BBOX_FALSE" in src,
    )

    # ==================================================================
    # 8. Classifier page wiring
    # ==================================================================
    page_src = (REPO_ROOT / "mufasa" / "ui_qt" / "pages"
                / "classifier_page.py").read_text()
    check(
        "classifier_page imports ValidateClassifierForm",
        "from mufasa.ui_qt.forms.validate_classifier import "
        "ValidateClassifierForm" in page_src,
    )
    check(
        "classifier_page adds 'Validate classifier' section",
        '"Validate classifier"' in page_src
        and "(ValidateClassifierForm, {})" in page_src,
    )
    # Section order: Manage → Run inference → Train → Validate
    check(
        "Validate classifier is section #4 (after Manage, Run "
        "inference, Train)",
        page_src.index('"Manage classifiers"')
        < page_src.index('"Run inference"')
        < page_src.index('"Train classifier"')
        < page_src.index('"Validate classifier"'),
    )

    # ==================================================================
    # 9. 122ar recorded in touched files
    # ==================================================================
    for path in (
        REPO_ROOT / "mufasa" / "ui_qt" / "forms"
        / "validate_classifier.py",
        REPO_ROOT / "mufasa" / "ui_qt" / "pages"
        / "classifier_page.py",
    ):
        check(
            f"{path.name}: records 122ar patch number",
            "122ar" in path.read_text(),
        )

    print(
        f"smoke_122ar_validate_classifier_qt: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
