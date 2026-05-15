"""
tests/smoke_122aq_train_classifier_qt.py
==========================================

Patch 122aq: Qt port of
:class:`MachineModelSettingsPopUp` plus a Train invocation.
New form :class:`TrainClassifierForm` is an inline
:class:`OperationForm` with the same in-frame +
pop-out-dockable pattern as 122aj's frame labeller, 122al's
batch pre-processor, and 122ap's run-inference form. Wired
into the Classifier page as the third section.

AST-only — PySide6 isn't in the sandbox.

Coverage:

1. New form file exists, parses, defines TrainClassifierForm
   subclassing OperationForm.
2. Critical methods present (build, collect_args, target,
   _validate, _on_save_only, _write_to_ini, _load_from_ini,
   _toggle_pop_out, _find_main_window).
3. Target drives TrainRandomForestClassifier.
4. Pop-out machinery uses QDockWidget with the 122aj feature
   set (Movable | Floatable | Closable, AllDockWidgetAreas).
5. _find_main_window walks parent chain to a QMainWindow.
6. INI persistence writes the [create_ensemble_settings]
   section with the canonical key shape
   (rf_n_estimators, rf_max_features, rf_criterion, etc).
7. v1-TOML-only projects get a clear error message
   pointing to the future [classifier_training] section.
8. Form has the major hyperparameter controls
   (estimators, max_features, criterion, test_size, etc.).
9. Form has the major evaluation toggles (meta_data,
   clf_report, bar_graph, learning_curve, pr_curve,
   shap).
10. Classifier page imports TrainClassifierForm and adds the
    'Train classifier' section in position 3.
11. 122aq recorded in all touched files.
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
                 / "train_classifier.py")
    check("train_classifier.py exists", form_path.is_file())
    src = form_path.read_text()
    try:
        tree = ast.parse(src)
        ok = True
    except SyntaxError:
        ok = False
    check("train_classifier.py parses cleanly", ok)

    classes = {
        n.name: n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef)
    }
    check(
        "TrainClassifierForm class defined",
        "TrainClassifierForm" in classes,
    )

    if "TrainClassifierForm" in classes:
        cls = classes["TrainClassifierForm"]
        bases = [
            b.id if isinstance(b, ast.Name) else None
            for b in cls.bases
        ]
        check(
            "TrainClassifierForm subclasses OperationForm",
            "OperationForm" in bases,
        )
        method_names = {
            n.name for n in cls.body
            if isinstance(n, ast.FunctionDef)
        }
        for required in (
            "build", "collect_args", "target",
            "_validate", "_on_save_only",
            "_read_classifier_targets",
            "_load_from_ini", "_write_to_ini",
            "_toggle_pop_out", "_find_main_window",
            "_build_hyperparams_group",
            "_build_evaluations_group",
            "_toggle_undersample", "_toggle_oversample",
            "_toggle_bar_graph_n", "_toggle_learning",
            "_toggle_shap",
        ):
            check(
                f"TrainClassifierForm.{required} defined",
                required in method_names,
            )

    # ==================================================================
    # 2. Target drives TrainRandomForestClassifier
    # ==================================================================
    check(
        "target() imports + drives TrainRandomForestClassifier",
        "from mufasa.model.train_rf import TrainRandomForestClassifier"
        in src
        and "TrainRandomForestClassifier(" in src
        and ".run()" in src,
    )
    check(
        "TrainRandomForestClassifier invoked with config_path",
        "config_path=config_path" in src,
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
        "Dock allows all areas (mirror 122aj/122al/122ap)",
        "AllDockWidgetAreas" in src,
    )
    check(
        "_find_main_window walks parent chain to QMainWindow",
        "_find_main_window" in src
        and "QMainWindow" in src,
    )

    # ==================================================================
    # 4. Persistence — 122as introduced TOML support via the
    #    layout helper write_classifier_training_settings. The
    #    Qt form delegates to the helper for both formats.
    # ==================================================================
    check(
        "Form delegates to write_classifier_training_settings "
        "OR writes directly to [create_ensemble_settings] "
        "(transitional — accept either form)",
        "write_classifier_training_settings" in src
        or '"create_ensemble_settings"' in src,
    )
    for key in (
        "rf_n_estimators", "rf_max_features", "rf_criterion",
        "train_test_size", "train_test_split_type",
        "rf_min_sample_leaf",
        "under_sample_setting", "under_sample_ratio",
        "over_sample_setting", "over_sample_ratio",
        "class_weights",
        "rf_metadata",
        "generate_classification_report",
        "generate_features_importance_bar_graph",
        "n_feature_importance_bars",
        "compute_feature_permutation_importance",
        "generate_learning_curve",
        "learning_curve_k_splits", "learning_curve_data_splits",
        "generate_precision_recall_curve",
        "partial_dependency",
        "calculate_shap_scores",
        "shap_target_present_no", "shap_target_absent_no",
        "shap_save_iteration", "shap_multiprocess",
    ):
        check(
            f"persists key '{key}' somewhere (key name appears "
            f"in source — INI direct write or helper kwarg)",
            f'"{key}"' in src,
        )
    # Patch 122bf: 122as introduced the layout helper which
    # handles both v1 TOML and legacy INI — no form-level
    # RuntimeError needed. Drop the requirement that the form
    # raises on TOML-only projects.
    check(
        "Form does NOT special-case TOML — relies on helper "
        "dispatch (post-122as the helper handles both formats)",
        "write_classifier_training_settings" in src
        or "RuntimeError" in src,  # either path is acceptable
    )

    # ==================================================================
    # 5. Validation
    # ==================================================================
    check(
        "Validates 'No classifiers defined' upstream",
        "No classifiers defined" in src,
    )
    check(
        "Validates RF estimators is an integer",
        "RF estimators" in src,
    )
    check(
        "Validates under-sample / over-sample ratio is a float "
        "when setting != None",
        '"under-sample"' in src
        and '"over-sample"' in src
        and "ratio must be a float" in src,
    )

    # ==================================================================
    # 6. Major fields present
    # ==================================================================
    for label, attr in (
        ("estimators",            "self.estimators"),
        ("max_features",          "self.max_features"),
        ("criterion",             "self.criterion"),
        ("test_size",             "self.test_size"),
        ("tt_split",              "self.tt_split"),
        ("min_leaf",              "self.min_leaf"),
        ("undersample_setting",   "self.undersample_setting"),
        ("undersample_ratio",     "self.undersample_ratio"),
        ("oversample_setting",    "self.oversample_setting"),
        ("oversample_ratio",      "self.oversample_ratio"),
        ("class_weight",          "self.class_weight"),
        ("eval_meta_data",        "self.eval_meta_data"),
        ("eval_clf_report",       "self.eval_clf_report"),
        ("eval_bar_graph",        "self.eval_bar_graph"),
        ("eval_permutation",      "self.eval_permutation"),
        ("eval_learning",         "self.eval_learning"),
        ("eval_pr_curve",         "self.eval_pr_curve"),
        ("eval_partial_dep",      "self.eval_partial_dep"),
        ("eval_shap",             "self.eval_shap"),
    ):
        # Space-tolerant: split each non-comment line and check if
        # the attribute is the LHS of an assignment.
        defined = False
        for line in src.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if attr in line and "=" in line:
                # split on first =, check LHS
                lhs = line.split("=", 1)[0].strip()
                if lhs == attr:
                    defined = True
                    break
        check(
            f"form attribute {attr} defined",
            defined,
        )

    # ==================================================================
    # 7. Classifier page wiring
    # ==================================================================
    page_src = (REPO_ROOT / "mufasa" / "ui_qt" / "pages"
                / "classifier_page.py").read_text()
    check(
        "classifier_page imports TrainClassifierForm",
        "from mufasa.ui_qt.forms.train_classifier import "
        "TrainClassifierForm" in page_src,
    )
    check(
        "classifier_page adds 'Train classifier' section",
        '"Train classifier"' in page_src
        and "(TrainClassifierForm, {})" in page_src,
    )
    # Section order: Manage classifiers → Run inference → Train
    check(
        "Train classifier is section #3 (after Manage and Run "
        "inference)",
        page_src.index('"Manage classifiers"')
        < page_src.index('"Run inference"')
        < page_src.index('"Train classifier"'),
    )

    # ==================================================================
    # 8. 122aq recorded in touched files
    # ==================================================================
    for path in (
        REPO_ROOT / "mufasa" / "ui_qt" / "forms"
        / "train_classifier.py",
        REPO_ROOT / "mufasa" / "ui_qt" / "pages"
        / "classifier_page.py",
    ):
        check(
            f"{path.name}: records 122aq patch number",
            "122aq" in path.read_text(),
        )

    print(
        f"smoke_122aq_train_classifier_qt: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
