"""
tests/smoke_122ap_run_inference_qt.py
======================================

Patch 122ap: Qt port of :class:`RunMachineModelsPopUp` — the
per-classifier model-path / threshold / minimum-bout-length
configurator that drives :class:`InferenceBatch`. New form
:class:`RunInferenceForm` is an inline :class:`OperationForm`
with the same in-frame + pop-out-dockable pattern as
122aj's frame labeller and 122al's batch pre-processor. Wired
into the Classifier page as the second section.

AST-only — PySide6 isn't in the sandbox.

Coverage:

1. New form file exists, parses, defines RunInferenceForm
   subclassing OperationForm.
2. Critical methods present (build, collect_args, target,
   _reload, _toggle_pop_out, _write_settings_to_ini).
3. Target drives InferenceBatch with the project config.
4. Pop-out machinery uses QDockWidget with the 122aj feature
   set (Movable | Floatable | Closable, AllDockWidgetAreas).
5. _find_main_window walks parent chain to a QMainWindow
   (mirror 122aj/122al pattern).
6. Per-classifier INI persistence writes the 3 expected
   sections (SML settings / threshold_settings /
   Minimum_bout_lengths) — same shape InferenceBatch reads.
7. v1-TOML-only projects get a clear error message on save
   (not a silent drop).
8. Classifier page imports RunInferenceForm and adds the
   'Run inference' section.
9. classifier.py docstring updated — no longer claims
   RunMachineModelsPopUp is deferred.
10. 122ap recorded in all touched files.
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
    # ==================================================================
    # 1. New form file
    # ==================================================================
    form_path = (REPO_ROOT / "mufasa" / "ui_qt" / "forms"
                 / "run_inference.py")
    check("run_inference.py exists", form_path.is_file())
    src = form_path.read_text()
    try:
        tree = ast.parse(src)
        ok = True
    except SyntaxError:
        ok = False
    check("run_inference.py parses cleanly", ok)

    classes = {
        n.name: n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef)
    }
    check(
        "RunInferenceForm class defined",
        "RunInferenceForm" in classes,
    )

    if "RunInferenceForm" in classes:
        cls = classes["RunInferenceForm"]
        bases = [
            b.id if isinstance(b, ast.Name) else None
            for b in cls.bases
        ]
        check(
            "RunInferenceForm subclasses OperationForm",
            "OperationForm" in bases,
        )
        method_names = {
            n.name for n in cls.body
            if isinstance(n, ast.FunctionDef)
        }
        for required in (
            "build", "collect_args", "target",
            "_reload", "_on_browse",
            "_toggle_pop_out", "_find_main_window",
            "_read_classifier_targets",
            "_read_existing_ini_settings",
            "_write_settings_to_ini",
        ):
            check(
                f"RunInferenceForm.{required} defined",
                required in method_names,
            )

    # ==================================================================
    # 2. Target drives InferenceBatch
    # ==================================================================
    check(
        "target() imports + drives InferenceBatch",
        "from mufasa.model.inference_batch import InferenceBatch"
        in src
        and "InferenceBatch(" in src
        and ".run()" in src,
    )
    check(
        "InferenceBatch invoked with config_path passed-through "
        "(features_dir/save_dir/min_bout defaulted)",
        "config_path=config_path" in src
        and "features_dir=None" in src
        and "save_dir=None" in src
        and "minimum_bout_length=None" in src,
    )

    # ==================================================================
    # 3. Pop-out / dockable machinery
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
        "Dock allows all areas (mirror 122aj/122al)",
        "AllDockWidgetAreas" in src,
    )
    check(
        "_find_main_window walks parent chain to QMainWindow",
        "_find_main_window" in src
        and "QMainWindow" in src,
    )

    # ==================================================================
    # 4. INI persistence — 3 sections, same shape as Tk popup
    # ==================================================================
    check(
        "Writes 'SML settings' section",
        '"SML settings"' in src,
    )
    check(
        "Writes 'threshold_settings' section",
        '"threshold_settings"' in src,
    )
    check(
        "Writes 'Minimum_bout_lengths' section",
        '"Minimum_bout_lengths"' in src,
    )
    check(
        "Writes model_path_{idx} keys per classifier",
        'f"model_path_{idx}"' in src,
    )
    check(
        "Writes threshold_{idx} keys per classifier",
        'f"threshold_{idx}"' in src,
    )
    check(
        "Writes min_bout_{idx} keys per classifier",
        'f"min_bout_{idx}"' in src,
    )
    check(
        "TOML-only project raises a clear error (not silent drop)",
        "project.toml" in src
        and "[classifier_inference]" in src
        and "raise RuntimeError" in src,
    )

    # ==================================================================
    # 5. Validation — per-classifier path/threshold/min-bout
    # ==================================================================
    check(
        "collect_args validates threshold range 0.0–1.0",
        "0.0 <= thr_val <= 1.0" in src,
    )
    check(
        "collect_args validates min_bout is non-negative integer",
        "mb_val < 0" in src,
    )
    check(
        "collect_args validates model path exists on disk",
        "os.path.isfile(path)" in src,
    )
    check(
        "collect_args raises on no classifiers defined",
        "No classifiers defined" in src,
    )

    # ==================================================================
    # 6. Classifier page wiring
    # ==================================================================
    page_src = (REPO_ROOT / "mufasa" / "ui_qt" / "pages"
                / "classifier_page.py").read_text()
    check(
        "classifier_page imports RunInferenceForm",
        "from mufasa.ui_qt.forms.run_inference import "
        "RunInferenceForm" in page_src,
    )
    check(
        "classifier_page adds 'Run inference' section",
        '"Run inference"' in page_src
        and "(RunInferenceForm, {})" in page_src,
    )
    # 'Manage classifiers' should still be section #1
    check(
        "Manage classifiers is still section #1, Run inference "
        "is section #2",
        page_src.index('"Manage classifiers"')
        < page_src.index('"Run inference"'),
    )

    # ==================================================================
    # 7. classifier.py docstring no longer claims popup is deferred
    # ==================================================================
    clf_src = (REPO_ROOT / "mufasa" / "ui_qt" / "forms"
               / "classifier.py").read_text()
    check(
        "classifier.py docstring no longer says "
        "RunMachineModelsPopUp is left as a dedicated dialog",
        "RunMachineModelsPopUp" not in clf_src
        or "deferred" not in clf_src,
    )
    check(
        "classifier.py docstring references the new form",
        "RunInferenceForm" in clf_src,
    )

    # ==================================================================
    # 8. 122ap recorded in all touched files
    # ==================================================================
    for path in (
        REPO_ROOT / "mufasa" / "ui_qt" / "forms" / "run_inference.py",
        REPO_ROOT / "mufasa" / "ui_qt" / "forms" / "classifier.py",
        REPO_ROOT / "mufasa" / "ui_qt" / "pages"
        / "classifier_page.py",
    ):
        check(
            f"{path.name}: records 122ap patch number",
            "122ap" in path.read_text(),
        )

    print(
        f"smoke_122ap_run_inference_qt: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
